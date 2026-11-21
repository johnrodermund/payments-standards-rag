"""One stack: VPC, RDS PostgreSQL (pgvector), ECS Fargate service behind an Application Load Balancer.

Choices that matter (all overridable with `-c key=value`; see infra/README.md):

* No NAT gateway. Tasks sit in public subnets with a public IP (reachable only from the load balancer's
  security group) so they can pull the image and call AWS APIs without a ~$33/month NAT. The database is in
  isolated subnets with no route to the internet.
* Generation mode (`llm_mode`): `replay` (default: recorded answers only, no model call, no key, no cost),
  `anthropic` (key read from an existing Secrets Manager secret) or `bedrock` (task role, no keys).
* This is a demonstration stack, not a production one: HTTP only (no certificate or domain), one task, one
  single-AZ database, deletion protection off, everything destroyed by `cdk destroy`.
"""

from __future__ import annotations

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_ecs_patterns as ecs_patterns
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from aws_cdk import aws_rds as rds
from aws_cdk import aws_secretsmanager as secretsmanager
from constructs import Construct

LLM_MODES = ("replay", "anthropic", "bedrock")


def settings(scope: Construct) -> dict[str, str]:
    """Deployment settings from CDK context, with defaults. Unknown modes fail at synth time."""
    get = scope.node.try_get_context

    def pick(key: str, default: str) -> str:
        value = get(key)
        return str(value) if value not in (None, "") else default

    config = {
        "llm_mode": pick("llm_mode", "replay"),
        "cpu": pick("cpu", "1024"),
        "memory": pick("memory", "2048"),
        "allowed_cidr": pick("allowed_cidr", ""),
        "llm_budget_usd": pick("llm_budget_usd", "0.50"),
        "anthropic_secret_name": pick("anthropic_secret_name", "payments-rag/anthropic-api-key"),
        "bedrock_model_id": pick("bedrock_model_id", ""),
        "llm_prices": pick("llm_prices", ""),
    }
    if config["llm_mode"] not in LLM_MODES:
        raise ValueError(f"llm_mode must be one of {LLM_MODES}, got {config['llm_mode']!r}")
    if config["llm_mode"] == "bedrock" and not config["bedrock_model_id"]:
        raise ValueError("llm_mode=bedrock needs -c bedrock_model_id=<inference profile id>")
    return config


class PaymentsRagStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        cfg = settings(self)

        vpc = ec2.Vpc(
            self,
            "Vpc",
            max_azs=2,
            nat_gateways=0,
            subnet_configuration=[
                ec2.SubnetConfiguration(name="public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24),
                ec2.SubnetConfiguration(name="db", subnet_type=ec2.SubnetType.PRIVATE_ISOLATED, cidr_mask=24),
            ],
        )

        db = rds.DatabaseInstance(
            self,
            "Database",
            engine=rds.DatabaseInstanceEngine.postgres(version=rds.PostgresEngineVersion.VER_16),
            instance_type=ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE4_GRAVITON, ec2.InstanceSize.MICRO),
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_ISOLATED),
            credentials=rds.Credentials.from_generated_secret("payments"),
            database_name="payments",
            allocated_storage=20,
            storage_type=rds.StorageType.GP3,
            storage_encrypted=True,
            multi_az=False,
            publicly_accessible=False,
            backup_retention=Duration.days(1),
            delete_automated_backups=True,
            deletion_protection=False,
            removal_policy=RemovalPolicy.DESTROY,  # demo stack: `cdk destroy` removes the data too
        )

        ingest_token = secretsmanager.Secret(
            self,
            "IngestToken",
            description="Bearer token for POST /ingest",
            generate_secret_string=secretsmanager.SecretStringGenerator(password_length=40, exclude_punctuation=True),
            removal_policy=RemovalPolicy.DESTROY,
        )

        log_group = logs.LogGroup(self, "Logs", retention=logs.RetentionDays.ONE_WEEK, removal_policy=RemovalPolicy.DESTROY)

        environment = {
            # libpq fills the host, user and password from PGHOST/PGUSER/PGPASSWORD (injected from the RDS secret below),
            # so no credential is ever part of the connection string. RDS forces TLS; require it explicitly too.
            "DATABASE_URL": "postgresql:///payments?sslmode=require",
            "AWS_REGION": self.region,
        }
        secrets = {
            "PGHOST": ecs.Secret.from_secrets_manager(db.secret, "host"),
            "PGPORT": ecs.Secret.from_secrets_manager(db.secret, "port"),
            "PGUSER": ecs.Secret.from_secrets_manager(db.secret, "username"),
            "PGPASSWORD": ecs.Secret.from_secrets_manager(db.secret, "password"),
            "INGEST_API_TOKEN": ecs.Secret.from_secrets_manager(ingest_token),
        }
        mode = cfg["llm_mode"]
        if mode == "replay":
            environment["LLM_REPLAY_ONLY"] = "1"
        elif mode == "anthropic":
            key = secretsmanager.Secret.from_secret_name_v2(self, "AnthropicKey", cfg["anthropic_secret_name"])
            secrets["ANTHROPIC_API_KEY"] = ecs.Secret.from_secrets_manager(key)
            environment["LLM_BUDGET_USD"] = cfg["llm_budget_usd"]
        else:
            environment.update({"LLM_PROVIDER": "bedrock", "BEDROCK_GENERATION_MODEL_ID": cfg["bedrock_model_id"], "LLM_BUDGET_USD": cfg["llm_budget_usd"]})
            if cfg["llm_prices"]:
                environment["LLM_PRICES"] = cfg["llm_prices"]

        cluster = ecs.Cluster(self, "Cluster", vpc=vpc)
        service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "Service",
            cluster=cluster,
            cpu=int(cfg["cpu"]),
            memory_limit_mib=int(cfg["memory"]),
            desired_count=1,
            min_healthy_percent=100,  # one task: start the new one before stopping the old
            max_healthy_percent=200,
            public_load_balancer=True,
            assign_public_ip=True,
            task_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            open_listener=not cfg["allowed_cidr"],
            circuit_breaker=ecs.DeploymentCircuitBreaker(rollback=True),
            health_check_grace_period=Duration.seconds(120),
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_asset("..", file="Dockerfile"),  # the repository root; honours .dockerignore
                container_port=8000,
                environment=environment,
                secrets=secrets,
                log_driver=ecs.LogDrivers.aws_logs(stream_prefix="api", log_group=log_group),
            ),
        )
        service.target_group.configure_health_check(path="/health", healthy_http_codes="200", interval=Duration.seconds(30))
        service.target_group.set_attribute("deregistration_delay.timeout_seconds", "10")
        db.connections.allow_default_port_from(service.service, "API tasks to PostgreSQL")
        if cfg["allowed_cidr"]:
            service.load_balancer.connections.allow_from(ec2.Peer.ipv4(cfg["allowed_cidr"]), ec2.Port.tcp(80), "operator access only")

        if mode == "bedrock":
            # Invocation only, and only for Anthropic models/inference profiles (no bedrock:*).
            service.task_definition.task_role.add_to_policy(
                iam.PolicyStatement(
                    actions=["bedrock:InvokeModel"],  # the Converse API is authorised by this action
                    resources=[
                        "arn:aws:bedrock:*::foundation-model/anthropic.*",
                        f"arn:aws:bedrock:*:{self.account}:inference-profile/*anthropic*",
                    ],
                )
            )

        CfnOutput(self, "Url", value=f"http://{service.load_balancer.load_balancer_dns_name}")
        CfnOutput(self, "IngestTokenSecretArn", value=ingest_token.secret_arn)
        CfnOutput(self, "LlmMode", value=mode)
