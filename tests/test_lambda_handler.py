import json

import boto3
import pytest
from moto import mock_aws

REGION = "us-east-2"


@pytest.fixture
def aws(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)
    monkeypatch.setenv("AWS_REGION", REGION)
    with mock_aws():
        yield


def url_event(method, path, body=None):
    """A Lambda Function URL (payload v2.0) event, as CloudFront forwards it."""
    return {
        "version": "2.0", "routeKey": "$default", "rawPath": path, "rawQueryString": "",
        "headers": {"content-type": "application/json", "host": "abc.lambda-url.us-east-2.on.aws",
                    "cloudfront-viewer-address": "203.0.113.7:5000"},
        "requestContext": {"http": {"method": method, "path": path, "protocol": "HTTP/1.1", "sourceIp": "1.2.3.4",
                                    "userAgent": "test"}, "domainName": "abc.lambda-url.us-east-2.on.aws",
                           "requestId": "id", "routeKey": "$default", "stage": "$default", "accountId": "123",
                           "apiId": "abc", "time": "25/Nov/2026:00:00:00 +0000", "timeEpoch": 0},
        "body": json.dumps(body) if body is not None else None, "isBase64Encoded": False,
    }


def test_api_key_is_read_from_ssm_into_the_environment(aws, monkeypatch):
    from payments_rag.api.lambda_handler import load_api_key

    boto3.client("ssm", region_name=REGION).put_parameter(
        Name="/payments-rag/anthropic-api-key", Value="sk-test-value", Type="SecureString")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("API_KEY_PARAMETER", "/payments-rag/anthropic-api-key")
    monkeypatch.setenv("API_KEY_ENV", "ANTHROPIC_API_KEY")
    load_api_key()
    import os

    assert os.environ["ANTHROPIC_API_KEY"] == "sk-test-value"


def test_no_parameter_configured_means_nothing_is_read(monkeypatch):
    from payments_rag.api.lambda_handler import load_api_key

    monkeypatch.delenv("API_KEY_PARAMETER", raising=False)
    load_api_key()  # no AWS call, no error


def test_function_url_event_round_trip():
    from payments_rag.api.lambda_handler import handler

    out = handler(url_event("GET", "/health"), None)
    assert out["statusCode"] == 200 and json.loads(out["body"]) == {"status": "ok"}


def test_page_round_trip_through_the_handler():
    from payments_rag.api.lambda_handler import handler

    out = handler(url_event("GET", "/"), None)
    assert out["statusCode"] == 200 and "content-security-policy" in {k.lower() for k in out["headers"]}


def test_verify_handler_raises_when_the_chain_is_broken(aws, monkeypatch):
    from payments_rag.api.lambda_handler import verify_handler
    from payments_rag.audit.dynamo import DynamoAuditLog, create_tables

    client = boto3.client("dynamodb", region_name=REGION)
    create_tables(client, "entries", "head")
    monkeypatch.setenv("AUDIT_TABLE", "entries")
    monkeypatch.setenv("AUDIT_HEAD_TABLE", "head")
    audit = DynamoAuditLog("entries", "head", client=client)
    audit.record({"request_id": "r1", "outcome": "answered", "question": "q"})
    assert verify_handler({}, None)["ok"] is True
    client.delete_item(TableName="entries", Key={"pk": {"S": "chain"}, "seq": {"N": "1"}})
    with pytest.raises(RuntimeError, match="audit chain"):
        verify_handler({}, None)
