"""HTTP client for the /api/gpu routes. Standard library only, so the CLI
installs with no dependencies on a bare laptop or cluster login node."""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

DEFAULT_API_URL = "https://api.scigantic.com"
CONFIG_PATH = Path(os.environ.get("SCIGANTIC_GPU_CONFIG", Path.home() / ".config" / "scigantic" / "gpu.json"))


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text())
    except (OSError, ValueError):
        return {}


def save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n")
    try:
        CONFIG_PATH.chmod(0o600)
    except OSError:
        pass


def api_url(override: Optional[str] = None) -> str:
    return (override or os.environ.get("SCIGANTIC_API_URL") or load_config().get("apiUrl") or DEFAULT_API_URL).rstrip("/")


class Client:
    def __init__(self, base: Optional[str] = None, token: Optional[str] = None):
        self.base = api_url(base)
        self.token = token if token is not None else load_config().get("token")

    def request(self, method: str, path: str, body: Any = None, auth: bool = True, timeout: int = 60) -> Any:
        data = None
        headers = {"Accept": "application/json", "User-Agent": "scigantic-gpu"}
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        if auth:
            if not self.token:
                raise ApiError(401, "Not logged in. Run: scigantic-gpu login --email you@example.org")
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                payload = json.loads(e.read() or b"{}")
            except ValueError:
                payload = {}
            msg = payload.get("error") or payload.get("message") or f"HTTP {e.code}"
            if isinstance(msg, dict):
                msg = msg.get("message") or json.dumps(msg)
            raise ApiError(e.code, str(msg)) from None
        except urllib.error.URLError as e:
            raise ApiError(0, f"Could not reach {self.base}: {e.reason}") from None
        if isinstance(payload, dict) and payload.get("success") is False:
            raise ApiError(400, str(payload.get("error") or payload.get("message") or "request failed"))
        return payload.get("data", payload) if isinstance(payload, dict) else payload

    # The upload goes straight to S3 with a presigned PUT, not through the API.
    @staticmethod
    def put_file(url: str, path: Path, content_type: str = "application/gzip") -> None:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            req = urllib.request.Request(url, data=fh, method="PUT", headers={
                "Content-Type": content_type,
                "Content-Length": str(size),
            })
            try:
                with urllib.request.urlopen(req, timeout=600):
                    pass
            except urllib.error.HTTPError as e:
                raise ApiError(e.code, f"Upload failed: HTTP {e.code} {e.read()[:200]!r}") from None


def die(msg: str, code: int = 1) -> "int":
    print(f"error: {msg}", file=sys.stderr)
    return code
