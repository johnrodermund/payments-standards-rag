"""One serverless stack: CloudFront -> Lambda Function URL (container image) + DynamoDB. No VPC, no servers.

Choices that matter (all overridable with `-c key=value`; see infra/README.md):

* CloudFront is the only way in. The Function URL uses AWS_IAM auth and CloudFront signs requests to it with
  Origin Access Control, so the URL cannot be called directly. For POST, CloudFront needs the body's SHA-256
  in `x-amz-content-sha256`; the web page sends it, and API callers must too.
* The search index ships inside the image (built at `docker build`), so there is no database to run.
* DynamoDB (on demand) holds the audit log (entries the API may only add to, plus a head record), the
  monthly spend cap and the rate-limit counters. The audit tables have deletion protection and
  point-in-time recovery, and are kept if the stack is deleted.
* The model API key is a SecureString in SSM Parameter Store, created by hand in the console (free tier),
  and read by the function at cold start. It is never part of the template.
* A daily EventBridge rule re-verifies the audit hash chain. Alarms go to one SNS topic; pass
  `-c alarm_email=...` at deploy time to subscribe (it is never written to cdk.json).
"""

from __future__ import annotations

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_cloudfront as cloudfront
from aws_cdk import aws_cloudfront_origins as origins
from aws_cdk import aws_cloudwatch as cloudwatch
from aws_cdk import aws_cloudwatch_actions as cw_actions
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sns_subscriptions as subscriptions
from constructs import Construct

PROVIDERS = {
    "anthropic": {"model_env": "CLAUDE_MODEL", "model": "claude-haiku-5-5", "key_env": "ANTHROPIC_API_KEY",
                  "parameter": "/payments-rag/anthropic-api-key"},
    "openai": {"model_env": "OPENAI_MODEL", "model": "gpt-6-luna", "key_env": "OPENAI_API_KEY",
               "parameter": "/payments-rag/openai-api-key"},
}


def settings(scope: Construct) -> dict[str, str]:
    """Deployment settings from CDK context, with defaults. Unknown providers fail at synth time."""
    get = scope.node.try_get_context

    def pick(key: str, default: str) -> str:
        value = get(key)
        return str(value) if value not in (None, "") else default

    provider = pick("llm_provider", "anthropic")
    if provider not in PROVIDERS:
        raise ValueError(f"llm_provider must be one of {sorted(PROVIDERS)}, got {provider!r}")
    p = PROVIDERS[provider]
    return {
        "llm_provider": provider,
        "model": pick("model", p["model"]),
        "api_key_parameter": pick("api_key_parameter", p["parameter"]),
        "llm_budget_usd": pick("llm_budget_usd", "5"),
        "rate_per_minute": pick("rate_per_minute", "10"),
        "rate_per_day": pick("rate_per_day", "100"),
        "memory": pick("memory", "3008"),
        "reserved_concurrency": pick("reserved_concurrency", ""),
        "alarm_email": pick("alarm_email", ""),
    }


class PaymentsRagStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        cfg = settings(self)
        provider = PROVIDERS[cfg["llm_provider"]]

        def table(name: str, sort_key: str | None = None, protect: bool = False, ttl: str | None = None) -> dynamodb.TableV2:
            return dynamodb.TableV2(
                self,
                name,
                partition_key=dynamodb.Attribute(name="pk", type=dynamodb.AttributeType.STRING),
                sort_key=dynamodb.Attribute(name=sort_key, type=dynamodb.AttributeType.NUMBER) if sort_key else None,
                billing=dynamodb.Billing.on_demand(),
                point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(point_in_time_recovery_enabled=protect),
                deletion_protection=protect,
                time_to_live_attribute=ttl,
                removal_policy=RemovalPolicy.RETAIN if protect else RemovalPolicy.DESTROY,
            )

        audit_entries = table("AuditEntries", sort_key="seq", protect=True)
        audit_head = table("AuditHead", protect=True)
        spend = table("Spend", protect=True)
        rate_limits = table("RateLimits", ttl="expires_at")

        api_logs = logs.LogGroup(self, "ApiLogs", retention=logs.RetentionDays.ONE_MONTH, removal_policy=RemovalPolicy.DESTROY)
        verify_logs = logs.LogGroup(self, "VerifyLogs", retention=logs.RetentionDays.ONE_MONTH, removal_policy=RemovalPolicy.DESTROY)

        environment = {
            "AUDIT_TABLE": audit_entries.table_name,
            "AUDIT_HEAD_TABLE": audit_head.table_name,
            "SPEND_TABLE": spend.table_name,
            "RATE_LIMIT_TABLE": rate_limits.table_name,
            "LLM_PROVIDER": cfg["llm_provider"],
            provider["model_env"]: cfg["model"],
            "LLM_BUDGET_USD": cfg["llm_budget_usd"],
            "ASK_RATE_PER_MINUTE": cfg["rate_per_minute"],
            "ASK_RATE_PER_DAY": cfg["rate_per_day"],
            "API_KEY_PARAMETER": cfg["api_key_parameter"],
            "API_KEY_ENV": provider["key_env"],
        }
        image_args = {"directory": "..", "file": "Dockerfile.lambda"}  # the repository root; honours .dockerignore
        reserved = int(cfg["reserved_concurrency"]) if cfg["reserved_concurrency"] else None

        api = lambda_.DockerImageFunction(
            self,
            "Api",
            code=lambda_.DockerImageCode.from_image_asset(**image_args),
            architecture=lambda_.Architecture.X86_64,
            memory_size=int(cfg["memory"]),
            timeout=Duration.seconds(29),  # CloudFront waits 30 s for the origin
            environment=environment,
            log_group=api_logs,
            reserved_concurrent_executions=reserved,
        )
        verify = lambda_.DockerImageFunction(
            self,
            "AuditVerify",
            code=lambda_.DockerImageCode.from_image_asset(**image_args, cmd=["payments_rag.api.lambda_handler.verify_handler"]),
            architecture=lambda_.Architecture.X86_64,
            memory_size=1024,
            timeout=Duration.minutes(5),
            environment={"AUDIT_TABLE": audit_entries.table_name, "AUDIT_HEAD_TABLE": audit_head.table_name},
            log_group=verify_logs,
        )

        # Least privilege on the tables. The API may only add audit entries, never change or delete them.
        api.add_to_role_policy(iam.PolicyStatement(actions=["dynamodb:PutItem"], resources=[audit_entries.table_arn]))
        api.add_to_role_policy(iam.PolicyStatement(actions=["dynamodb:GetItem", "dynamodb:UpdateItem"], resources=[audit_head.table_arn]))
        api.add_to_role_policy(iam.PolicyStatement(actions=["dynamodb:UpdateItem"], resources=[spend.table_arn, rate_limits.table_arn]))
        api.add_to_role_policy(iam.PolicyStatement(
            actions=["ssm:GetParameter"],
            resources=[f"arn:aws:ssm:{self.region}:{self.account}:parameter{cfg['api_key_parameter']}"],
        ))
        api.add_to_role_policy(iam.PolicyStatement(  # SecureString with the AWS-managed aws/ssm key
            actions=["kms:Decrypt"], resources=["*"],
            conditions={"StringEquals": {"kms:ViaService": f"ssm.{self.region}.amazonaws.com"}},
        ))
        verify.add_to_role_policy(iam.PolicyStatement(actions=["dynamodb:Query"], resources=[audit_entries.table_arn]))
        verify.add_to_role_policy(iam.PolicyStatement(actions=["dynamodb:GetItem"], resources=[audit_head.table_arn]))

        url = api.add_function_url(auth_type=lambda_.FunctionUrlAuthType.AWS_IAM)
        forward = cloudfront.OriginRequestPolicy(
            self,
            "Forward",
            comment="Viewer address for the rate limiter, the body hash OAC needs, content type; never Host",
            header_behavior=cloudfront.OriginRequestHeaderBehavior.allow_list(
                "CloudFront-Viewer-Address", "x-amz-content-sha256", "Content-Type", "Accept",
            ),
            query_string_behavior=cloudfront.OriginRequestQueryStringBehavior.all(),
            cookie_behavior=cloudfront.OriginRequestCookieBehavior.none(),
        )
        distribution = cloudfront.Distribution(
            self,
            "Cdn",
            comment="Payments standards assistant",
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.FunctionUrlOrigin.with_origin_access_control(url),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
                cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,
                origin_request_policy=forward,
            ),
            price_class=cloudfront.PriceClass.PRICE_CLASS_100,
            http_version=cloudfront.HttpVersion.HTTP2_AND_3,
        )
        # Function URLs need lambda:InvokeFunction as well as lambda:InvokeFunctionUrl for the caller.
        api.add_permission(
            "CloudFrontInvoke",
            principal=iam.ServicePrincipal("cloudfront.amazonaws.com"),
            action="lambda:InvokeFunction",
            source_arn=f"arn:aws:cloudfront::{self.account}:distribution/{distribution.distribution_id}",
        )

        events.Rule(self, "DailyAuditVerify", schedule=events.Schedule.rate(Duration.days(1)),
                    targets=[targets.LambdaFunction(verify, retry_attempts=1)])

        topic = sns.Topic(self, "Alarms", display_name="Payments assistant alarms")
        if cfg["alarm_email"]:
            topic.add_subscription(subscriptions.EmailSubscription(cfg["alarm_email"]))
        action = cw_actions.SnsAction(topic)
        budget = float(cfg["llm_budget_usd"])
        alarms = [
            api.metric_errors(period=Duration.minutes(5)).create_alarm(
                self, "ApiErrors", threshold=1, evaluation_periods=1, alarm_description="API function crashed (unhandled errors)"),
            api.metric_throttles(period=Duration.minutes(5)).create_alarm(
                self, "ApiThrottles", threshold=1, evaluation_periods=1, alarm_description="API function throttled"),
            verify.metric_errors(period=Duration.days(1)).create_alarm(
                self, "AuditVerifyFailed", threshold=1, evaluation_periods=1, alarm_description="Audit hash chain verification failed"),
            cloudwatch.Metric(namespace="PaymentsRag", metric_name="MonthlySpendUSD", statistic="Maximum",
                              period=Duration.hours(1)).create_alarm(
                self, "SpendNearCap", threshold=round(budget * 0.8, 2), evaluation_periods=1,
                comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
                treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
                alarm_description="Model spend reached 80% of the monthly cap"),
        ]
        for alarm in alarms:
            alarm.add_alarm_action(action)

        CfnOutput(self, "Url", value=f"https://{distribution.distribution_domain_name}")
        CfnOutput(self, "ApiKeyParameter", value=cfg["api_key_parameter"])
        CfnOutput(self, "AuditEntriesTable", value=audit_entries.table_name)
        CfnOutput(self, "AuditHeadTable", value=audit_head.table_name)
        CfnOutput(self, "LlmProvider", value=cfg["llm_provider"])
