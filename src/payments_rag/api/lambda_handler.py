"""AWS Lambda entry points (container image).

* `handler`: the FastAPI app behind a Lambda Function URL (payload v2.0), via Mangum. CloudFront is the
  only caller (the URL uses AWS_IAM auth with CloudFront Origin Access Control).
* `verify_handler`: run daily by EventBridge; re-walks the audit hash chain, checks it against yesterday's anchor,
  moves the anchor forward, and raises if anything is broken, so
  the function's Errors metric (and its alarm) fires.

The model API key lives in SSM Parameter Store (SecureString). It is read once per cold start into the
environment variable the SDK reads (API_KEY_ENV), and never logged. The LangSmith key, when tracing is on,
is read the same way (LANGSMITH_KEY_PARAMETER -> LANGSMITH_API_KEY), before the app (and tracing) loads.
Spans are flushed after every invocation, because Lambda freezes the process once the response is sent.
"""

from __future__ import annotations

import os
from typing import Any


def _load_secret(parameter: str | None, target: str) -> None:
    if not parameter or os.environ.get(target):
        return
    import boto3

    ssm = boto3.client("ssm", region_name=os.environ.get("AWS_REGION"))
    os.environ[target] = ssm.get_parameter(Name=parameter, WithDecryption=True)["Parameter"]["Value"].strip()


def load_api_key() -> None:
    _load_secret(os.environ.get("API_KEY_PARAMETER"), os.environ.get("API_KEY_ENV", "ANTHROPIC_API_KEY"))


def load_langsmith_key() -> None:
    try:
        _load_secret(os.environ.get("LANGSMITH_KEY_PARAMETER"), "LANGSMITH_API_KEY")
    except Exception as exc:  # noqa: BLE001 -- tracing must never stop the app from starting
        print(f"LangSmith key not loaded, tracing off: {type(exc).__name__}")


load_api_key()
load_langsmith_key()

from mangum import Mangum

from payments_rag.api.main import app
from payments_rag.observability.tracing import flush

_asgi = Mangum(app, lifespan="off")


def handler(event: Any, context: Any) -> dict[str, Any]:
    try:
        return _asgi(event, context)
    finally:
        flush()


def verify_handler(event: Any, context: Any) -> dict[str, Any]:
    from payments_rag.audit.dynamo import DynamoAuditLog

    log = DynamoAuditLog(os.environ["AUDIT_TABLE"], os.environ["AUDIT_HEAD_TABLE"], anchor_table=os.environ.get("AUDIT_ANCHOR_TABLE"))
    report = log.verify(update_anchor=True)
    print(f"audit verify: {'OK' if report['ok'] else 'TAMPERED'} rows={report['rows']} head={report['head_hash']}")
    if not report["ok"]:
        raise RuntimeError(f"audit chain verification failed: {report['errors'][:5]}")
    return {"ok": True, "rows": report["rows"], "head_hash": report["head_hash"]}
