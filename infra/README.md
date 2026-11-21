# AWS deployment (CDK, Python)

One stack, `PaymentsRag`, in `us-east-2`: a VPC without a NAT gateway, an RDS PostgreSQL 16 instance (pgvector),
and an ECS Fargate service behind an Application Load Balancer. **Status: checked offline (`uv run pytest`, `cdk synth --strict`) and deployed once
on 2026-11-21 in replay mode, then destroyed (measurements in the main README).** It is a demonstration stack: HTTP only (no domain or
certificate), one task, one single-AZ database, deletion protection off, and `cdk destroy` removes the data too.

## What runs where

| Piece | Choice | Why |
|---|---|---|
| Compute | Fargate, 1 vCPU / 2 GB, one task, public subnet with a public IP | no NAT gateway (about $33/month saved); only the load balancer can reach the task |
| Database | RDS PostgreSQL 16, `db.t4g.micro`, 20 GB gp3, encrypted, isolated subnets | the hash-chained audit log needs durable storage; no route to the internet |
| Credentials | RDS secret and ingest token in Secrets Manager, injected as `PG*` variables | the connection string holds no password; the task role holds no model keys |
| Image | built from the repository `Dockerfile` as a CDK asset | embedding model baked in, recorded answers included, non-root user |

## Generation mode (`-c llm_mode=...`)

| Mode | Needs | Behaviour |
|---|---|---|
| `replay` (default) | nothing | recorded answers only; any other question returns 503. No model call, no key, no cost |
| `anthropic` | an existing secret `payments-rag/anthropic-api-key` (`-c anthropic_secret_name=` to rename) | live Claude calls through the spend cap. The ledger is on the container's disk, so it **resets on every deploy or task restart**; set a spend limit in the Anthropic Console as well |
| `bedrock` | `-c bedrock_model_id=<inference profile id>` and working Bedrock quotas; `-c llm_prices='{"<id>":[in,out]}'` | task role limited to `bedrock:InvokeModel` on Anthropic models; no keys |

Other settings: `-c allowed_cidr=1.2.3.4/32` (close the load balancer to everyone else; recommended for any
live-model mode), `-c cpu=`, `-c memory=`, `-c llm_budget_usd=`, `-c region=`.

## Commands

Sign in first (`aws login`, profile `payments-rag`), then from this directory:

```
npx aws-cdk@2 bootstrap            # once per account and region
npx aws-cdk@2 diff
npx aws-cdk@2 deploy -c allowed_cidr=<your-ip>/32
```

The outputs give the URL and the ARN of the ingest-token secret. The index starts empty; fill it once with the
`/ingest` endpoint (same call that was run against the local container):

```
POST <url>/ingest   Authorization: Bearer <token from Secrets Manager>
{"messages": ["pacs.008", "pacs.002"], "source": "extract"}
```

It re-chunks the committed extract and embeds 1,245 chunks on the task's CPU: 4 min 41 s measured on Fargate
(1 vCPU, 2 GB), about 2.5 minutes in a desktop container. Poll `GET /ingest/{id}`.

Tear down when finished: `npx aws-cdk@2 destroy`.

## Cost (estimate, list prices, us-east-2, running 24/7)

Fargate 1 vCPU / 2 GB about $42/month, Application Load Balancer about $17, RDS `db.t4g.micro` plus storage about
$14: roughly **$73/month, $2.40/day**. Nothing is billed until a deploy; destroy the stack when done. These are
estimates, not measured bills. The one real deployment ran about 25 minutes (estimated under $0.10).

## Not covered

HTTPS and a domain, WAF or rate limiting, multi-AZ or autoscaling, Phoenix in the cloud (traces are off), CI
deployment, cdk-nag compliance rules.
