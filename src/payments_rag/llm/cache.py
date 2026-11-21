"""Disk cache for LLM responses, keyed by a hash of the full request.

Identical requests cost nothing after the first. The same files double as a
recorded run for demos: copy a cache directory and replay it with no API key.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_DIR = Path(".cache") / "llm"


def default_dir() -> Path:
    return Path(os.environ.get("LLM_CACHE_DIR", DEFAULT_DIR))


class ResponseCache:
    def __init__(self, directory: str | Path | None = None):
        self.directory = Path(directory) if directory else default_dir()

    @staticmethod
    def key(request: dict[str, Any]) -> str:
        canonical = json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None  # a corrupt entry is just a miss

    def put(self, key: str, value: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self._path(key).with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._path(key))
