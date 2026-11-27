"""Template checks for the serverless stack. They run offline (no AWS calls) and guard the choices that matter."""

import json

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Match, Template

from stack import PaymentsRagStack


def synth(**context) -> Template:
    app = cdk.App(context={"aws:cdk:bundling-stacks": [], **context})
    return Template.from_stack(PaymentsRagStack(app, "T", env=cdk.Environment(account="123456789012", region="us-east-2")))


@pytest.fixture(scope="module")
def default() -> Template:
    return synth()


def functions(template):
    return template.find_resources("AWS::Lambda::Function")


def api_function(template):
    return next(f for f in functions(template).values() if "verify_handler" not in json.dumps(f["Properties"].get("ImageConfig", {})))


def statements(template):
    return [s for p in template.find_resources("AWS::IAM::Policy").values() for s in p["Properties"]["PolicyDocument"]["Statement"]]


def actions_on(template, logical_prefix):
    out = set()
    for s in statements(template):
        resources = json.dumps(s.get("Resource"))
        if logical_prefix in resources:
            acts = s["Action"] if isinstance(s["Action"], list) else [s["Action"]]
            out |= set(acts)
    return out


def test_no_vpc_load_balancer_database_or_nat(default):
    for kind in ("AWS::EC2::VPC", "AWS::ElasticLoadBalancingV2::LoadBalancer", "AWS::RDS::DBInstance", "AWS::EC2::NatGateway", "AWS::ECS::Cluster"):
        default.resource_count_is(kind, 0)


def test_api_runs_as_an_x86_container_image_with_the_memory_index(default):
    fn = api_function(default)["Properties"]
    assert fn["PackageType"] == "Image" and fn["Architectures"] == ["x86_64"]
    assert fn["MemorySize"] >= 2048 and fn["Timeout"] <= 29
    env = fn["Environment"]["Variables"]
    assert env["LLM_PROVIDER"] == "anthropic" and env["CLAUDE_MODEL"] == "claude-haiku-5-5"
    assert env["LLM_BUDGET_USD"] == "5" and env["ASK_RATE_PER_MINUTE"] == "10" and env["ASK_RATE_PER_DAY"] == "100"
    assert env["API_KEY_PARAMETER"] == "/payments-rag/anthropic-api-key" and env["API_KEY_ENV"] == "ANTHROPIC_API_KEY"
    assert {"AUDIT_TABLE", "AUDIT_HEAD_TABLE", "SPEND_TABLE", "RATE_LIMIT_TABLE"} <= set(env)
    assert "sk-" not in json.dumps(default.to_json())


def test_openai_provider_switches_model_and_key_parameter():
    env = api_function(synth(llm_provider="openai"))["Properties"]["Environment"]["Variables"]
    assert env["LLM_PROVIDER"] == "openai" and env["OPENAI_MODEL"] == "gpt-6-luna"
    assert env["API_KEY_PARAMETER"] == "/payments-rag/openai-api-key" and env["API_KEY_ENV"] == "OPENAI_API_KEY"


def test_bad_provider_fails_at_synth_time():
    with pytest.raises(ValueError, match="llm_provider"):
        synth(llm_provider="azure")


def test_function_url_requires_iam_so_only_cloudfront_can_call_it(default):
    default.has_resource_properties("AWS::Lambda::Url", {"AuthType": "AWS_IAM"})
    default.resource_count_is("AWS::CloudFront::OriginAccessControl", 1)
    perms = default.find_resources("AWS::Lambda::Permission")
    granted = {(p["Properties"]["Action"], p["Properties"]["Principal"]) for p in perms.values()}
    assert ("lambda:InvokeFunctionUrl", "cloudfront.amazonaws.com") in granted
    assert ("lambda:InvokeFunction", "cloudfront.amazonaws.com") in granted


def test_cloudfront_is_https_only_uncached_and_forwards_the_viewer_address(default):
    (dist,) = default.find_resources("AWS::CloudFront::Distribution").values()
    behavior = dist["Properties"]["DistributionConfig"]["DefaultCacheBehavior"]
    assert behavior["ViewerProtocolPolicy"] == "redirect-to-https"
    assert "POST" in behavior["AllowedMethods"]
    (policy,) = default.find_resources("AWS::CloudFront::OriginRequestPolicy").values()
    headers = {h.lower() for h in policy["Properties"]["OriginRequestPolicyConfig"]["HeadersConfig"]["Headers"]}
    assert {"cloudfront-viewer-address", "content-type"} <= headers
    assert "host" not in headers  # the Function URL rejects a foreign Host header
    assert "x-amz-content-sha256" not in headers  # CloudFront rejects it in a policy; OAC signs with the viewer's


def test_audit_entries_are_append_only_for_the_api(default):
    acts = actions_on(default, "AuditEntries")
    assert "dynamodb:PutItem" in acts
    assert not acts & {"dynamodb:UpdateItem", "dynamodb:DeleteItem", "dynamodb:BatchWriteItem", "dynamodb:*"}


def test_tables_are_on_demand_and_the_audit_tables_are_protected(default):
    tables = default.find_resources("AWS::DynamoDB::GlobalTable")
    assert len(tables) == 5
    protected = [t for k, t in tables.items() if k.startswith(("AuditEntries", "AuditHead", "AuditAnchor"))]
    assert len(protected) == 3
    for t in protected:
        assert t["DeletionPolicy"] == "Retain"
        replica = t["Properties"]["Replicas"][0]
        assert replica["DeletionProtectionEnabled"] is True
        assert replica["PointInTimeRecoverySpecification"]["PointInTimeRecoveryEnabled"] is True


def test_rate_limit_counters_expire(default):
    default.has_resource_properties("AWS::DynamoDB::GlobalTable", {"TimeToLiveSpecification": {"AttributeName": "expires_at", "Enabled": True}})


def test_key_is_read_from_ssm_not_injected(default):
    ssm = [s for s in statements(default) if "ssm:GetParameter" in json.dumps(s["Action"])]
    assert ssm and "parameter/payments-rag/anthropic-api-key" in json.dumps(ssm[0]["Resource"])
    default.resource_count_is("AWS::SecretsManager::Secret", 0)


def test_daily_audit_verify_runs_the_verify_handler(default):
    verify = [f for f in functions(default).values() if "verify_handler" in json.dumps(f["Properties"].get("ImageConfig", {}))]
    assert len(verify) == 1
    default.has_resource_properties("AWS::Events::Rule", {"ScheduleExpression": "rate(1 day)"})


def test_alarms_go_to_an_sns_topic(default):
    default.resource_count_is("AWS::SNS::Topic", 1)
    names = {a["Properties"].get("MetricName") for a in default.find_resources("AWS::CloudWatch::Alarm").values()}
    assert {"Errors", "Throttles", "MonthlySpendUSD", "ApiTracebacks"} <= names
    default.has_resource_properties("AWS::CloudWatch::Alarm", {"MetricName": "MonthlySpendUSD", "Threshold": 4.0})


def test_alarm_email_is_subscribed_only_when_given(default):
    default.resource_count_is("AWS::SNS::Subscription", 0)
    synth(alarm_email="ops@example.com").has_resource_properties("AWS::SNS::Subscription", {"Protocol": "email", "Endpoint": "ops@example.com"})


def test_logs_are_kept_thirty_days(default):
    default.has_resource_properties("AWS::Logs::LogGroup", {"RetentionInDays": 30})


def test_reserved_concurrency_is_optional(default):
    assert "ReservedConcurrentExecutions" not in api_function(default)["Properties"]
    fn = api_function(synth(reserved_concurrency="5"))["Properties"]
    assert fn["ReservedConcurrentExecutions"] == 5


def test_https_url_is_an_output(default):
    outputs = default.to_json()["Outputs"]
    assert "Url" in outputs and "https://" in json.dumps(outputs["Url"]["Value"])
    default.has_output("ApiKeyParameter", Match.object_like({"Value": "/payments-rag/anthropic-api-key"}))


def test_http_500s_are_caught_from_tracebacks_in_the_api_log(default):
    default.has_resource_properties("AWS::Logs::MetricFilter", {"FilterPattern": '"Traceback"'})


def test_only_the_verify_function_can_write_the_audit_anchor(default):
    assert actions_on(default, "AuditAnchor") == {"dynamodb:GetItem", "dynamodb:PutItem"}
    api_policies = [p for k, p in default.find_resources("AWS::IAM::Policy").items() if k.startswith("ApiServiceRole")]
    assert api_policies and "AuditAnchor" not in json.dumps(api_policies)


def test_model_calls_time_out_before_the_function_does(default):
    env = api_function(default)["Properties"]["Environment"]["Variables"]
    assert int(env["LLM_TIMEOUT_SECONDS"]) * (int(env["LLM_MAX_RETRIES"]) + 1) < 29


def test_langsmith_tracing_is_off_unless_its_key_parameter_is_given(default):
    env = api_function(default)["Properties"]["Environment"]["Variables"]
    assert "LANGSMITH_TRACING" not in env and "LANGSMITH_KEY_PARAMETER" not in env


def test_langsmith_tracing_reads_its_key_from_ssm():
    t = synth(langsmith_key_parameter="/payments-rag/langsmith-api-key")
    env = api_function(t)["Properties"]["Environment"]["Variables"]
    assert env["LANGSMITH_TRACING"] == "true" and env["LANGSMITH_PROJECT"] == "payments-assistant"
    assert env["LANGSMITH_KEY_PARAMETER"] == "/payments-rag/langsmith-api-key"
    assert "LANGSMITH_API_KEY" not in env  # never in the template
    ssm = json.dumps([s["Resource"] for s in statements(t) if "ssm:GetParameter" in json.dumps(s["Action"])])
    assert "parameter/payments-rag/langsmith-api-key" in ssm and "parameter/payments-rag/anthropic-api-key" in ssm
