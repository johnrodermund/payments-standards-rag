"""Monthly model spend cap on DynamoDB: atomic across every Lambda instance, survives restarts.

One item per UTC month (pk "2026-11") holds:
* committed: settled spend plus reservations still in flight. The cap is enforced on this number.
* spent, calls, input_tokens, output_tokens: settled usage only.

Per model call: `check` reserves the worst-case cost with one conditional update
(`committed <= limit - worst`; DynamoDB conditions cannot add, so the subtraction is done here), `record`
settles it to the actual cost, and `release` gives it back when the call failed. Two concurrent requests
can never both pass the condition with only room for one.

Callers see only "monthly question budget reached": no figures, no table names. Settled spend is printed
as a CloudWatch Embedded Metric Format line (namespace PaymentsRag, metric MonthlySpendUSD) so an alarm can
fire at 80% of the cap without any extra API call.

If a process dies between reserve and settle, that reservation (at most about $0.0006) stays counted until
the month ends. Accepted.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Self

from payments_rag.llm.budget import CHARS_PER_TOKEN, Budget, BudgetExceededError, cost_usd

GENERIC_REFUSAL = "monthly question budget reached; please try again next month"
METRIC_NAMESPACE = "PaymentsRag"


def create_table(client: Any, name: str) -> None:
    """For tests and local runs; in AWS the CDK stack creates the table."""
    client.create_table(
        TableName=name,
        KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )


class Reservation(float):
    """The amount reserved, remembering its month, so a call that runs past midnight at the end of a
    month settles into the month it was counted in."""

    month: str

    def __new__(cls, amount: float, month: str) -> Self:
        obj = super().__new__(cls, amount)
        obj.month = month
        return obj


def _n(value: float) -> dict[str, str]:
    return {"N": f"{value:.10f}"}


class DynamoSpendCap:
    def __init__(self, table: str, limit_usd: float, client: Any | None = None, clock: Callable[[], datetime] | None = None):
        self.table = table
        self.limit_usd = limit_usd
        self._client = client
        self._clock = clock or (lambda: datetime.now(UTC))

    @property
    def client(self) -> Any:
        if self._client is None:
            import boto3

            self._client = boto3.client("dynamodb", region_name=os.environ.get("AWS_REGION"))
        return self._client

    def _month(self, reserved: float = 0.0) -> str:
        return getattr(reserved, "month", None) or self._clock().strftime("%Y-%m")

    def check(self, model: str, request: dict[str, Any], max_tokens: int | None = None) -> float:
        """Reserve the worst-case cost of this call; return the amount reserved. Raises BudgetExceededError."""
        est_input = len(json.dumps(request, default=str)) // CHARS_PER_TOKEN + 1
        worst = cost_usd(model, est_input, int(max_tokens if max_tokens is not None else request.get("max_tokens", 0)))
        if worst > self.limit_usd:
            raise BudgetExceededError(GENERIC_REFUSAL)
        month = self._month()
        try:
            self.client.update_item(
                TableName=self.table,
                Key={"pk": {"S": month}},
                UpdateExpression="ADD committed :w",
                ConditionExpression="attribute_not_exists(committed) OR committed <= :max",
                ExpressionAttributeValues={":w": _n(worst), ":max": _n(self.limit_usd - worst)},
            )
        except self.client.exceptions.ConditionalCheckFailedException as exc:
            raise BudgetExceededError(GENERIC_REFUSAL) from exc
        return Reservation(worst, month)

    def record(self, model: str, input_tokens: int, output_tokens: int, reserved: float = 0.0) -> float:
        """Settle a reservation to the actual cost (reserved=0 records spend that was never reserved)."""
        cost = cost_usd(model, input_tokens, output_tokens)
        response = self.client.update_item(
            TableName=self.table,
            Key={"pk": {"S": self._month(reserved)}},
            UpdateExpression="ADD committed :delta, spent :cost, calls :one, input_tokens :in, output_tokens :out",
            ExpressionAttributeValues={
                ":delta": _n(cost - reserved), ":cost": _n(cost), ":one": {"N": "1"},
                ":in": {"N": str(input_tokens)}, ":out": {"N": str(output_tokens)},
            },
            ReturnValues="UPDATED_NEW",
        )
        spent = float(response.get("Attributes", {}).get("spent", {}).get("N", cost))
        print(json.dumps({  # CloudWatch Embedded Metric Format: becomes a metric when printed by Lambda
            "_aws": {"Timestamp": int(time.time() * 1000), "CloudWatchMetrics": [
                {"Namespace": METRIC_NAMESPACE, "Dimensions": [[]], "Metrics": [{"Name": "MonthlySpendUSD", "Unit": "None"}]}
            ]},
            "MonthlySpendUSD": spent,
        }))
        return cost

    def release(self, reserved: float) -> None:
        if reserved:
            self.client.update_item(
                TableName=self.table,
                Key={"pk": {"S": self._month(reserved)}},
                UpdateExpression="ADD committed :minus",
                ExpressionAttributeValues={":minus": _n(-reserved)},
            )


def budget_from_env() -> Any:
    """DynamoDB cap when SPEND_TABLE is set (serverless), else the local file budget."""
    if table := os.environ.get("SPEND_TABLE"):
        from payments_rag.llm.budget import DEFAULT_LIMIT_USD

        return DynamoSpendCap(table, float(os.environ.get("LLM_BUDGET_USD", DEFAULT_LIMIT_USD)))
    return Budget.from_env()
