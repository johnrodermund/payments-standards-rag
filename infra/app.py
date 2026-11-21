import os

import aws_cdk as cdk

from stack import PaymentsRagStack

app = cdk.App()
PaymentsRagStack(
    app,
    "PaymentsRag",
    # Account comes from the credentials in use; the region is pinned so a stray AWS_REGION cannot move the stack.
    env=cdk.Environment(account=os.environ.get("CDK_DEFAULT_ACCOUNT"), region=app.node.try_get_context("region") or "us-east-2"),
)
cdk.Tags.of(app).add("project", "payments-standards-rag")
app.synth()
