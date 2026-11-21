"""Template checks for the CDK stack. They run offline (no AWS calls) and guard the choices that matter."""

import json

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Match, Template

from stack import PaymentsRagStack


def synth(**context) -> Template:
    app = cdk.App(context={"aws:cdk:bundling-stacks": [], **context})
    return Template.from_stack(PaymentsRagStack(app, "T", env=cdk.Environment(account="123456789012", region="us-east-2")))


@pytest.fixture(scope="module")
def replay() -> Template:
    return synth()


def container_env(template: Template) -> dict[str, str]:
    (task,) = template.find_resources("AWS::ECS::TaskDefinition").values()
    definition = task["Properties"]["ContainerDefinitions"][0]
    return {e["Name"]: e["Value"] for e in definition["Environment"] if isinstance(e["Value"], str)}


def test_no_nat_gateway_and_database_is_not_public(replay):
    replay.resource_count_is("AWS::EC2::NatGateway", 0)
    replay.has_resource_properties("AWS::RDS::DBInstance", {"PubliclyAccessible": False, "StorageEncrypted": True, "Engine": "postgres"})


def test_default_mode_is_replay_only_with_no_model_credentials(replay):
    env = container_env(replay)
    assert env["LLM_REPLAY_ONLY"] == "1"
    assert "LLM_PROVIDER" not in env and "ANTHROPIC_API_KEY" not in json.dumps(replay.to_json())
    assert not replay.find_resources("AWS::IAM::Policy", {"Properties": {"PolicyDocument": {"Statement": Match.array_with([Match.object_like({"Action": "bedrock:InvokeModel"})])}}})


def test_database_password_is_injected_as_a_secret_never_an_environment_value(replay):
    (task,) = replay.find_resources("AWS::ECS::TaskDefinition").values()
    definition = task["Properties"]["ContainerDefinitions"][0]
    assert {s["Name"] for s in definition["Secrets"]} >= {"PGHOST", "PGUSER", "PGPASSWORD", "INGEST_API_TOKEN"}
    assert "password" not in json.dumps(container_env(replay)).lower()


def test_bedrock_mode_uses_the_task_role_scoped_to_invoke_only():
    template = synth(llm_mode="bedrock", bedrock_model_id="us.anthropic.claude-test", llm_prices='{"us.anthropic.claude-test":[1,5]}')
    env = container_env(template)
    assert env["LLM_PROVIDER"] == "bedrock" and env["BEDROCK_GENERATION_MODEL_ID"] == "us.anthropic.claude-test" and "LLM_REPLAY_ONLY" not in env
    statements = [s for p in template.find_resources("AWS::IAM::Policy").values() for s in p["Properties"]["PolicyDocument"]["Statement"]]
    bedrock = [s for s in statements if s["Action"] == "bedrock:InvokeModel"]
    assert len(bedrock) == 1 and "*" not in {r for r in bedrock[0]["Resource"]}


def test_anthropic_mode_reads_the_key_from_secrets_manager():
    template = synth(llm_mode="anthropic")
    (task,) = template.find_resources("AWS::ECS::TaskDefinition").values()
    assert "ANTHROPIC_API_KEY" in {s["Name"] for s in task["Properties"]["ContainerDefinitions"][0]["Secrets"]}
    assert "ANTHROPIC_API_KEY" not in container_env(template)


def test_bad_configuration_fails_at_synth_time():
    with pytest.raises(ValueError, match="llm_mode"):
        synth(llm_mode="azure")
    with pytest.raises(ValueError, match="bedrock_model_id"):
        synth(llm_mode="bedrock")


def test_allowed_cidr_closes_the_load_balancer_to_everyone_else():
    template = synth(allowed_cidr="203.0.113.7/32")
    rules = [
        rule
        for resource in template.to_json()["Resources"].values()
        if resource["Type"] == "AWS::EC2::SecurityGroup"
        for rule in resource["Properties"].get("SecurityGroupIngress", [])
    ]
    cidrs = {rule.get("CidrIp") for rule in rules}
    assert "0.0.0.0/0" not in cidrs
    assert "203.0.113.7/32" in cidrs
