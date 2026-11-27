"""Test-wide setup.

Tests never send traces anywhere: tracing settings from the developer's environment (a LangSmith key from
other projects, a Phoenix endpoint) are removed before any app module is imported, because the API turns
tracing on at import time. Tests that need tracing install an in-memory exporter themselves.
"""

import os

_TRACING_PREFIXES = ("LANGSMITH_", "LANGCHAIN_")
_TRACING_NAMES = {"PHOENIX_COLLECTOR_ENDPOINT", "PHOENIX_API_KEY", "OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_HEADERS"}

for _name in list(os.environ):
    if _name.startswith(_TRACING_PREFIXES) or _name in _TRACING_NAMES:
        del os.environ[_name]
