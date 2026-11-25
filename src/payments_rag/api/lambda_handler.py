"""AWS Lambda entry points (container image).

* `handler`: the FastAPI app behind a Lambda Function URL (payload v2.0), via Mangum. CloudFront is the
  only caller (the URL uses AWS_IAM auth with CloudFront Origin Access Control).
* `verify_handler`: run daily by EventBridge; re-walks the audit hash chain and raises if it is broken, so
  the function's Errors metric (and its alarm) fires.

The model API key lives in SSM Parameter Store (SecureString). It is read once per cold start into the
environment variable the SDK reads (API_KEY_ENV), and never logged.
"""

from __future__ import annotations

import os
from typing import Any


def load_api_key() -> None:
    parameter = os.environ.get("API_KEY_PARAMETER")
    target = os.environ.get("API_KEY_ENV", "ANTHROPIC_API_KEY")
    if not parameter or os.environ.get(target):
        return
    import boto3

    ssm = boto3.client("ssm", region_name=os.environ.get("AWS_REGION"))
    os.environ[target] = ssm.get_parameter(Name=parameter, WithDecryption=True)["Parameter"]["Value"]


load_api_key()

from mangum import Mangum

from payments_rag.api.main import app

handler = Mangum(app, lifespan="off")


def verify_handler(event: Any, context: Any) -> dict[str, Any]:
    from payments_rag.audit.dynamo import DynamoAuditLog

    report = DynamoAuditLog(os.environ["AUDIT_TABLE"], os.environ["AUDIT_HEAD_TABLE"]).verify()
    print(f"audit verify: {'OK' if report['ok'] else 'TAMPERED'} rows={report['rows']} head={report['head_hash']}")
    if not report["ok"]:
        raise RuntimeError(f"audit chain verification failed: {report['errors'][:5]}")
    return {"ok": True, "rows": report["rows"], "head_hash": report["head_hash"]}
