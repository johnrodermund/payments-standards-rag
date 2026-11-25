"""Per-client rate limit for POST /ask on DynamoDB, shared by every Lambda instance.

Client identity: behind CloudFront the socket address is AWS's, and `X-Forwarded-For` can be written by the
client, so the key comes from `CloudFront-Viewer-Address` (set by CloudFront, "ip:port"; the port is
dropped). Without that header (local runs) the socket address is used. The key is a SHA-256 of the IP,
so no raw IP address is stored.

Two fixed-window counters per client, per minute and per UTC day, each one atomic `ADD` with a TTL so old
counters delete themselves. Over a limit, the caller gets 429 and `Retry-After` (seconds until the window
resets). If DynamoDB cannot be reached the request is allowed and the error logged: the spend cap still
bounds the cost, so availability wins here.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

log = logging.getLogger("payments_rag.ratelimit")

DEFAULT_PER_MINUTE = 10
DEFAULT_PER_DAY = 100


def create_table(client: Any, name: str) -> None:
    """For tests and local runs; in AWS the CDK stack creates the table (with TTL on expires_at)."""
    client.create_table(
        TableName=name,
        KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )


def client_key(request: Any) -> str:
    viewer = request.headers.get("cloudfront-viewer-address")
    ip = viewer.rsplit(":", 1)[0] if viewer else getattr(request.client, "host", "unknown")
    if ":" in ip:  # IPv6: one client usually holds a whole /64, so count the /64, not the address
        try:
            ip = f"{ipaddress.ip_network(f'{ip}/64', strict=False).network_address}/64"
        except ValueError:
            pass
    salt = os.environ.get("RATE_LIMIT_SALT", "payments-rag")
    return hashlib.sha256(f"{salt}:{ip}".encode()).hexdigest()[:32]


class DynamoRateLimiter:
    def __init__(self, table: str, per_minute: int, per_day: int, client: Any | None = None,
                 clock: Callable[[], datetime] | None = None):
        self.table = table
        self.per_minute = per_minute
        self.per_day = per_day
        self._client = client
        self._clock = clock or (lambda: datetime.now(UTC))

    @property
    def client(self) -> Any:
        if self._client is None:
            import boto3

            self._client = boto3.client("dynamodb", region_name=os.environ.get("AWS_REGION"))
        return self._client

    def _count(self, pk: str, expires: datetime) -> int:
        response = self.client.update_item(
            TableName=self.table,
            Key={"pk": {"S": pk}},
            UpdateExpression="ADD n :one SET expires_at = if_not_exists(expires_at, :exp)",
            ExpressionAttributeValues={":one": {"N": "1"}, ":exp": {"N": str(int(expires.timestamp()))}},
            ReturnValues="UPDATED_NEW",
        )
        return int(response["Attributes"]["n"]["N"])

    def hit(self, key: str) -> int | None:
        """Count one request. None if allowed, else seconds until the caller may retry."""
        now = self._clock()
        minute_end = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        if self._count(f"{key}#m#{now:%Y%m%d%H%M}", minute_end + timedelta(minutes=5)) > self.per_minute:
            return max(1, int((minute_end - now).total_seconds()))
        day_end = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        if self._count(f"{key}#d#{now:%Y%m%d}", day_end + timedelta(hours=1)) > self.per_day:
            return max(1, int((day_end - now).total_seconds()))
        return None


def limiter_from_env() -> DynamoRateLimiter | None:
    table = os.environ.get("RATE_LIMIT_TABLE")
    if not table:
        return None
    return DynamoRateLimiter(
        table,
        int(os.environ.get("ASK_RATE_PER_MINUTE", DEFAULT_PER_MINUTE)),
        int(os.environ.get("ASK_RATE_PER_DAY", DEFAULT_PER_DAY)),
    )


def check(limiter: DynamoRateLimiter | None, request: Any) -> int | None:
    """Retry-After seconds when the caller is over a limit; None otherwise (also when the table is down)."""
    if limiter is None:
        return None
    try:
        return limiter.hit(client_key(request))
    except Exception:
        log.exception("rate limiter unavailable; allowing the request")
        return None
