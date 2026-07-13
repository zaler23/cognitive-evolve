from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from cognitive_evolve_runtime.api import guards
from cognitive_evolve_runtime.api.openai_compat import create_app


@pytest.fixture(autouse=True)
def _reset_rate_windows() -> Iterator[None]:
    with guards._RATE_LOCK:
        guards._RATE_WINDOWS.clear()
    yield
    with guards._RATE_LOCK:
        guards._RATE_WINDOWS.clear()


def _scope(token: str, client: str = "203.0.113.7") -> dict[str, object]:
    return {
        "headers": [(b"authorization", f"Bearer {token}".encode())],
        "client": (client, 12345),
    }


def test_invalid_tokens_share_client_bucket_but_valid_keys_use_principals() -> None:
    valid_keys = ("trusted-a", "trusted-b")
    for index in range(10_000):
        key = guards._rate_limit_key(
            _scope(f"invalid-{index}"),
            require_auth=True,
            valid_keys=valid_keys,
        )
        guards._allow_request(key, 120)

    assert set(guards._RATE_WINDOWS) == {"client:203.0.113.7"}
    assert guards._rate_limit_key(_scope("trusted-a"), require_auth=True, valid_keys=valid_keys).startswith("principal:")
    assert guards._rate_limit_key(_scope("trusted-a"), require_auth=True, valid_keys=valid_keys) != guards._rate_limit_key(
        _scope("trusted-b"), require_auth=True, valid_keys=valid_keys
    )


def test_rate_bucket_table_has_hard_cap_and_expires_stale_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [0.0]
    monkeypatch.setattr(guards.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(guards, "_RATE_MAX_BUCKETS", 8)

    for index in range(100):
        guards._allow_request(f"client:{index}", 1)
    assert len(guards._RATE_WINDOWS) == 8

    now[0] = guards._RATE_BUCKET_TTL_SECONDS + 1.0
    guards._allow_request("client:fresh", 1)
    assert set(guards._RATE_WINDOWS) == {"client:fresh"}


def test_valid_auth_principals_keep_independent_rate_limits(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("COGEV_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("COGEV_API_TASK_ROOT", str(tmp_path / "api-runs"))
    monkeypatch.setenv("COGEV_SERVER_REQUIRE_AUTH", "true")
    monkeypatch.setenv("COGEV_SERVER_API_KEY", "trusted-a")
    monkeypatch.setenv("COGEV_SERVER_API_KEYS", "trusted-b")
    monkeypatch.setenv("COGEV_API_RATE_LIMIT_PER_MINUTE", "2")

    with TestClient(create_app()) as client:
        assert client.get("/v1/models", headers={"Authorization": "Bearer invalid-a"}).status_code == 401
        assert client.get("/v1/models", headers={"Authorization": "Bearer invalid-b"}).status_code == 401
        assert client.get("/v1/models", headers={"Authorization": "Bearer invalid-c"}).status_code == 429

        assert client.get("/v1/models", headers={"Authorization": "Bearer trusted-a"}).status_code == 200
        assert client.get("/v1/models", headers={"X-API-Key": "trusted-a"}).status_code == 200
        assert client.get("/v1/models", headers={"Authorization": "Bearer trusted-a"}).status_code == 429
        assert client.get("/v1/models", headers={"X-API-Key": "trusted-b"}).status_code == 200
