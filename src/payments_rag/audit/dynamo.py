"""The audit log on DynamoDB, for serverless hosting. Same entries and hash chain as audit.log.AuditLog.

Two tables, written together in one transaction per entry:

* entries: pk "chain", sk seq (1, 2, ...); each item holds the canonical JSON, prev_hash and row_hash.
* head: one item (pk "head") with the last seq and its hash.

Why this is tamper-evident:
* Each row hash covers the previous one, so an edited or deleted entry breaks the chain (`verify`).
* The head records how long the chain is, so deleting entries from the END is also caught. (The Postgres
  version could not detect tail truncation.)
* The daily verify writes an anchor (the head's seq and hash) to a third table that only the verify
  function can write. Each run checks that the last anchor is still part of the chain, so history up to the
  previous run cannot be rewritten without the next run noticing.

Limits: IAM cannot force a write to be conditional, so the API's own role (PutItem on entries, UpdateItem on
head) could rewrite entries added since the last anchor and recompute the chain consistently. That window
is at most a day. A principal with full DynamoDB rights can rewrite all three tables; point-in-time
recovery (on in the stack) keeps the history to compare against.
"""

from __future__ import annotations

import os
import random
import time
from typing import Any

from payments_rag.audit.log import GENESIS, _canonical, _row_hash

CHAIN = "chain"
HEAD = "head"
ANCHOR = "anchor"
WRITE_DEADLINE_SECONDS = 3.0


def create_tables(client: Any, entries_table: str, head_table: str) -> None:
    """For tests and local runs; in AWS the CDK stack creates the tables."""
    client.create_table(
        TableName=entries_table,
        KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}, {"AttributeName": "seq", "KeyType": "RANGE"}],
        AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}, {"AttributeName": "seq", "AttributeType": "N"}],
        BillingMode="PAY_PER_REQUEST",
    )
    client.create_table(
        TableName=head_table,
        KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )


def create_anchor_table(client: Any, anchor_table: str) -> None:
    """For tests and local runs; in AWS the CDK stack creates the table."""
    client.create_table(
        TableName=anchor_table,
        KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )


class DynamoAuditLog:
    def __init__(self, entries_table: str, head_table: str, client: Any | None = None, anchor_table: str | None = None):
        self.entries_table = entries_table
        self.head_table = head_table
        self.anchor_table = anchor_table
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            import boto3

            self._client = boto3.client("dynamodb", region_name=os.environ.get("AWS_REGION"))
        return self._client

    def _read_head(self) -> tuple[int, str]:
        item = self.client.get_item(TableName=self.head_table, Key={"pk": {"S": HEAD}}, ConsistentRead=True).get("Item")
        if not item:
            return 0, GENESIS
        return int(item["seq"]["N"]), item["row_hash"]["S"]

    def record(self, entry: dict[str, Any]) -> str:
        """Append one entry and return its row hash. Raises if it cannot be written (callers fail closed)."""
        canonical = _canonical(entry)
        deadline = time.monotonic() + WRITE_DEADLINE_SECONDS
        attempt = 0
        while True:
            seq, prev = self._read_head()
            digest = _row_hash(prev, canonical)
            head_condition = "attribute_not_exists(pk)" if seq == 0 else "seq = :expected"
            head_update: dict[str, Any] = {
                "TableName": self.head_table,
                "Key": {"pk": {"S": HEAD}},
                "UpdateExpression": "SET seq = :next, row_hash = :hash",
                "ConditionExpression": head_condition,
                "ExpressionAttributeValues": {":next": {"N": str(seq + 1)}, ":hash": {"S": digest}},
            }
            if seq:
                head_update["ExpressionAttributeValues"][":expected"] = {"N": str(seq)}
            try:
                self.client.transact_write_items(
                    TransactItems=[
                        {"Put": {
                            "TableName": self.entries_table,
                            "Item": {
                                "pk": {"S": CHAIN}, "seq": {"N": str(seq + 1)},
                                "request_id": {"S": str(entry["request_id"])}, "outcome": {"S": str(entry["outcome"])},
                                "canonical": {"S": canonical}, "prev_hash": {"S": prev}, "row_hash": {"S": digest},
                            },
                            "ConditionExpression": "attribute_not_exists(seq)",
                        }},
                        {"Update": head_update},
                    ]
                )
                return digest
            except self.client.exceptions.TransactionCanceledException:
                # Another writer moved the head: back off (exponential, full jitter), re-read and retry.
                attempt += 1
                pause = random.uniform(0, min(0.5, 0.01 * 2**attempt))
                if time.monotonic() + pause > deadline:
                    raise RuntimeError(f"audit log: could not append within {WRITE_DEADLINE_SECONDS:.0f} s (contention)") from None
                time.sleep(pause)

    def _entries(self, up_to: int) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        kwargs: dict[str, Any] = {
            "TableName": self.entries_table,
            "KeyConditionExpression": "pk = :c AND seq <= :last",
            "ExpressionAttributeValues": {":c": {"S": CHAIN}, ":last": {"N": str(up_to)}},
            "ConsistentRead": True,
        }
        while True:
            page = self.client.query(**kwargs)
            items += page.get("Items", [])
            if "LastEvaluatedKey" not in page:
                return items
            kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]

    def _read_anchor(self) -> tuple[int, str] | None:
        if not self.anchor_table:
            return None
        item = self.client.get_item(TableName=self.anchor_table, Key={"pk": {"S": ANCHOR}}, ConsistentRead=True).get("Item")
        return (int(item["seq"]["N"]), item["row_hash"]["S"]) if item else None

    def verify(self, update_anchor: bool = False) -> dict[str, Any]:
        """Re-walk the chain up to the head. With an anchor table, also check the last anchor is still in the
        chain; with update_anchor (the daily job only), move the anchor to the head when everything is OK."""
        errors: list[str] = []
        # Head first: every entry up to the head's seq exists (they are written in one transaction), so
        # questions answered while this runs cannot make it report a false break.
        head_seq, head_hash = self._read_head()
        anchor = self._read_anchor()
        prev, expected_seq = GENESIS, 1
        hashes: dict[int, str] = {}
        items = self._entries(head_seq)
        for item in items:
            seq = int(item["seq"]["N"])
            if seq != expected_seq:
                errors.append(f"seq {expected_seq}: missing (next present is {seq})")
                expected_seq = seq
            if item["prev_hash"]["S"] != prev:
                errors.append(f"seq {seq}: chain broken (previous row missing or changed)")
            if _row_hash(item["prev_hash"]["S"], item["canonical"]["S"]) != item["row_hash"]["S"]:
                errors.append(f"seq {seq}: hash mismatch (row content was modified)")
            prev = item["row_hash"]["S"]
            hashes[seq] = prev
            expected_seq += 1
        last_seq = int(items[-1]["seq"]["N"]) if items else 0
        if head_seq != last_seq or head_hash != prev:
            errors.append(f"head says seq {head_seq}, log ends at seq {last_seq}: entries removed from the end or head changed")
        if anchor and anchor[0] and hashes.get(anchor[0]) != anchor[1]:
            errors.append(f"anchor at seq {anchor[0]} no longer matches: history before the last verify was rewritten")
        if update_anchor and self.anchor_table and not errors:
            self.client.put_item(TableName=self.anchor_table, Item={
                "pk": {"S": ANCHOR}, "seq": {"N": str(head_seq)}, "row_hash": {"S": head_hash},
            })
        return {"ok": not errors, "rows": len(items), "head_hash": prev, "errors": errors}
