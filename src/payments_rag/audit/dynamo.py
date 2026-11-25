"""The audit log on DynamoDB, for serverless hosting. Same entries and hash chain as audit.log.AuditLog.

Two tables, written together in one transaction per entry:

* entries: pk "chain", sk seq (1, 2, ...); each item holds the canonical JSON, prev_hash and row_hash.
* head: one item (pk "head") with the last seq and its hash.

Why this is tamper-evident:
* Each row hash covers the previous one, so an edited or deleted entry breaks the chain (`verify`).
* The head records how long the chain is, so deleting entries from the END is also caught. (The Postgres
  version could not detect tail truncation.)
* The application's IAM role may only PutItem on entries (conditional on the seq being new) and
  GetItem/UpdateItem on head: it cannot change or delete an entry. The CDK stack grants exactly that.

Limits: an AWS principal with full DynamoDB rights can still rewrite both tables consistently; point-in-time
recovery (on in the stack) keeps the history to compare against.
"""

from __future__ import annotations

import os
import time
from typing import Any

from payments_rag.audit.log import GENESIS, _canonical, _row_hash

CHAIN = "chain"
HEAD = "head"
MAX_ATTEMPTS = 8


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


class DynamoAuditLog:
    def __init__(self, entries_table: str, head_table: str, client: Any | None = None):
        self.entries_table = entries_table
        self.head_table = head_table
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
        for attempt in range(MAX_ATTEMPTS):
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
                time.sleep(0.02 * (attempt + 1))  # another writer moved the head; re-read and retry
        raise RuntimeError(f"audit log: could not append after {MAX_ATTEMPTS} attempts (contention)")

    def _entries(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        kwargs: dict[str, Any] = {
            "TableName": self.entries_table,
            "KeyConditionExpression": "pk = :c",
            "ExpressionAttributeValues": {":c": {"S": CHAIN}},
            "ConsistentRead": True,
        }
        while True:
            page = self.client.query(**kwargs)
            items += page.get("Items", [])
            if "LastEvaluatedKey" not in page:
                return items
            kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]

    def verify(self) -> dict[str, Any]:
        errors: list[str] = []
        prev, expected_seq = GENESIS, 1
        items = self._entries()
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
            expected_seq += 1
        head_seq, head_hash = self._read_head()
        last_seq = int(items[-1]["seq"]["N"]) if items else 0
        if head_seq != last_seq or head_hash != prev:
            errors.append(f"head says seq {head_seq}, log ends at seq {last_seq}: entries removed from the end or head changed")
        return {"ok": not errors, "rows": len(items), "head_hash": prev, "errors": errors}
