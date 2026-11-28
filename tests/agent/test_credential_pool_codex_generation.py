"""Independent Codex rows must refresh the persisted generation, not a stale holder.

Real pool imports, auth-store I/O and HTTP request encoding; only the token endpoint
is replaced. All grants are synthetic and external-login adoption stays disabled.
"""
from __future__ import annotations

import base64
import json
import multiprocessing
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest

import hermes_constants
import agent.credential_pool as cp
from agent.credential_pool import load_pool
from hermes_cli import auth_codex
from hermes_cli.auth import _auth_store_lock


def _jwt(account: str, expires: float) -> str:
    payload = {
        "exp": int(expires), "sub": f"user-{account}",
        "https://api.openai.com/auth": {"chatgpt_account_id": account},
    }
    segment = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    return f"eyJhbGciOiJub25lIn0.{segment}.sig"


@pytest.fixture
def grants(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: root)
    (root / "config.yaml").write_text("auth:\n  adopt_external_logins: false\n")
    now = time.time()
    work = {"access_token": _jwt("work", now + 30000), "refresh_token": "work-0"}
    personal = {
        "id": "personal", "label": "personal", "source": "manual:device_code",
        "auth_type": "oauth", "priority": 1, "access_token": _jwt("personal", now - 60),
        "refresh_token": "personal-0", "last_refresh": "2026-01-01T00:00:00Z",
    }
    (root / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {"openai-codex": {"tokens": work, "last_refresh": "2026-01-01T00:00:00Z"}},
        "credential_pool": {"openai-codex": [
            {"id": "work", "label": "work", "source": "device_code", "auth_type": "oauth",
             "priority": 0, **work, "last_refresh": "2026-01-01T00:00:00Z"}, personal,
        ]},
    }))
    posted = []

    def respond(request):
        assert str(request.url) == auth_codex.CODEX_OAUTH_TOKEN_URL
        token = parse_qs(request.content.decode())["refresh_token"][0]
        posted.append(token)
        if token != f"personal-{len(posted) - 1}":
            return httpx.Response(400, json={"error": {"code": "refresh_token_reused"}})
        return httpx.Response(200, json={
            "access_token": _jwt("personal", now + 30000 + len(posted)),
            "refresh_token": f"personal-{len(posted)}",
        })

    monkeypatch.setattr(auth_codex, "_codex_http_client",
                        lambda **kwargs: httpx.Client(transport=httpx.MockTransport(respond), **kwargs))
    return root, posted, work


def _personal(pool):
    return next(entry for entry in pool.entries() if entry.id == "personal")


def _assert_accounts(root, work):
    store = json.loads((root / "auth.json").read_text())
    assert store["providers"]["openai-codex"]["tokens"] == work
    rows = store["credential_pool"]["openai-codex"]
    assert {row["id"] for row in rows} == {"work", "personal"}
    personal = next(row for row in rows if row["id"] == "personal")
    assert personal.get("last_status") != "dead"
    assert personal["refresh_token"] == "personal-1"


@pytest.mark.parametrize("force", [False, True], ids=["proactive", "401-recovery"])
def test_stale_holder_adopts_peer_generation_without_spending_again(grants, force):
    root, posted, work = grants
    winner = load_pool("openai-codex")
    waiter = load_pool("openai-codex")
    stale = _personal(waiter)

    refreshed = winner._refresh_entry(_personal(winner), force=False)
    assert refreshed is not None and posted == ["personal-0"]
    adopted = waiter._refresh_entry(stale, force=force, failed_key=stale.access_token)

    assert posted == ["personal-0"], "a stale holder POSTed the already consumed refresh token"
    assert adopted is not None and adopted.refresh_token == "personal-1"
    assert _personal(waiter).refresh_token == "personal-1"
    _assert_accounts(root, work)


def test_expired_peer_generation_is_the_only_pair_that_may_be_refreshed(grants):
    root, posted, work = grants
    waiter = load_pool("openai-codex")
    store = json.loads((root / "auth.json").read_text())
    row = next(row for row in store["credential_pool"]["openai-codex"] if row["id"] == "personal")
    row.update(access_token=_jwt("personal", time.time() - 10), refresh_token="personal-1")
    (root / "auth.json").write_text(json.dumps(store))
    posted.append("personal-0")  # generation 0 was consumed by a previous holder

    refreshed = waiter._refresh_entry(_personal(waiter), force=False)

    assert posted == ["personal-0", "personal-1"]
    assert refreshed is not None and refreshed.refresh_token == "personal-2"
    assert json.loads((root / "auth.json").read_text())["providers"]["openai-codex"]["tokens"] == work


@pytest.mark.parametrize("path", ["refresh", "pool-probe", "standalone-probe"])
def test_borrower_holds_the_root_owner_lock_during_refresh(grants, monkeypatch, path):
    root, posted, work = grants
    borrower = root / "profiles" / "borrower"
    borrower.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(borrower))
    pool = load_pool("openai-codex")
    client = auth_codex._codex_http_client
    lock_checks = []

    def owner_is_locked():
        try:
            with _auth_store_lock(timeout_seconds=0, target_path=root / "auth.json"):
                return False
        except TimeoutError:
            return True

    def checked_client(**kwargs):
        with ThreadPoolExecutor(max_workers=1) as executor:
            lock_checks.append(executor.submit(owner_is_locked).result(timeout=5))
        return client(**kwargs)

    monkeypatch.setattr(auth_codex, "_codex_http_client", checked_client)
    monkeypatch.setattr(cp.auth_mod, "_probe_codex_quota_restored", lambda *a, **kw: False)
    monkeypatch.setattr(auth_codex, "_probe_codex_quota_restored", lambda *a, **kw: False)
    entry = replace(_personal(pool), last_status="exhausted", last_error_code=429)
    if path == "refresh":
        refreshed = pool._refresh_entry(entry, force=False)
        assert refreshed is not None
    elif path == "pool-probe":
        assert pool._codex_quota_restored_upstream(entry) is False
    else:
        assert auth_codex._probe_codex_pool_entry_quota_restored(entry.to_dict()) is False

    assert lock_checks == [True], "refresh did not lock the grant's owner"
    assert posted == ["personal-0"]
    _assert_accounts(root, work)
    local = borrower / "auth.json"
    assert not local.exists() or not json.loads(local.read_text()).get("credential_pool", {}).get("openai-codex")


@pytest.mark.parametrize("path", ["pool-probe", "standalone-probe"])
def test_quota_probe_uses_peer_generation_and_preserves_cooldown(grants, monkeypatch, path):
    root, posted, work = grants
    store = json.loads((root / "auth.json").read_text())
    personal = next(row for row in store["credential_pool"]["openai-codex"] if row["id"] == "personal")
    personal.update(last_status="exhausted", last_error_code=429, last_status_at=time.time(),
                    last_error_reason="usage_limit_reached", last_error_reset_at=time.time() + 86400)
    (root / "auth.json").write_text(json.dumps(store))
    first = load_pool("openai-codex")
    second = load_pool("openai-codex")
    monkeypatch.setattr(cp.auth_mod, "_probe_codex_quota_restored", lambda *a, **kw: False)
    monkeypatch.setattr(auth_codex, "_probe_codex_quota_restored", lambda *a, **kw: False)

    assert first._codex_quota_restored_upstream(_personal(first)) is False
    if path == "pool-probe":
        assert second._codex_quota_restored_upstream(_personal(second)) is False
    else:
        assert auth_codex._probe_codex_pool_entry_quota_restored(personal) is False

    assert posted == ["personal-0"]
    _assert_accounts(root, work)
    row = next(row for row in json.loads((root / "auth.json").read_text())["credential_pool"]["openai-codex"]
               if row["id"] == "personal")
    assert row["last_status"] == "exhausted" and row["last_error_code"] == 429
    assert row["last_error_reset_at"] == personal["last_error_reset_at"]


def test_stale_dead_holder_adopts_new_generation_before_pruning(grants):
    root, posted, work = grants
    winner = load_pool("openai-codex")
    waiter = load_pool("openai-codex")
    stale = _personal(waiter)
    assert winner._refresh_entry(_personal(winner), force=False) is not None
    waiter._replace_entry(stale, replace(stale, last_status="dead", last_status_at=time.time() - 172800))

    waiter.has_available()

    assert _personal(waiter).refresh_token == "personal-1"
    assert posted == ["personal-0"]
    _assert_accounts(root, work)


@pytest.mark.parametrize("failure", ["removed", "unreadable", "blank"])
def test_failed_authoritative_read_never_posts_a_stale_generation(grants, monkeypatch, failure):
    root, posted, _work = grants
    pool = load_pool("openai-codex")
    if failure == "unreadable":
        def fail(*args, **kwargs):
            raise OSError("synthetic store read failure")
        monkeypatch.setattr(cp, "_load_auth_store", fail)
    else:
        store = json.loads((root / "auth.json").read_text())
        rows = store["credential_pool"]["openai-codex"]
        if failure == "removed":
            store["credential_pool"]["openai-codex"] = [row for row in rows if row["id"] != "personal"]
        else:
            row = next(row for row in rows if row["id"] == "personal")
            row.pop("access_token")
            row.pop("refresh_token")
        (root / "auth.json").write_text(json.dumps(store))

    with pytest.raises((OSError, RuntimeError)):
        pool._refresh_entry(_personal(pool), force=False)
    assert posted == []


def test_stale_borrower_never_forks_root_grant_after_local_ownership_changes(grants, monkeypatch):
    root, posted, _work = grants
    borrower = root / "profiles" / "borrower"
    borrower.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(borrower))
    pool = load_pool("openai-codex")
    own_row = dict(_personal(pool).to_dict(), id="independent-local", refresh_token="local-0")
    (borrower / "auth.json").write_text(json.dumps({"credential_pool": {"openai-codex": [own_row]}}))

    with pytest.raises(RuntimeError, match="ownership changed"):
        pool._refresh_entry(_personal(pool), force=False)
    assert posted == []
    assert json.loads((borrower / "auth.json").read_text())["credential_pool"]["openai-codex"] == [own_row]


def test_borrowed_commit_failure_is_not_a_success_or_a_retry(grants, monkeypatch):
    root, posted, _work = grants
    borrower = root / "profiles" / "borrower"
    borrower.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(borrower))
    pool = load_pool("openai-codex")
    stale = _personal(pool)

    def fail(*args, **kwargs):
        raise OSError("synthetic commit failure")

    monkeypatch.setattr(cp, "_update_root_pool_rows", fail)
    with pytest.raises(OSError, match="commit failure"):
        pool._refresh_entry(stale, force=False)
    with pytest.raises(RuntimeError, match="not durably persisted"):
        pool._refresh_entry(stale, force=False)
    with pytest.raises(RuntimeError, match="not durably persisted"):
        pool.has_available()
    assert posted == ["personal-0"]
    assert _personal(pool).last_status == "dead"


def test_terminal_personal_failure_never_quarantines_the_work_grant(grants):
    root, posted, work = grants
    posted.append("personal-0")  # The synthetic endpoint rejects this generation as spent.
    pool = load_pool("openai-codex")

    assert pool._refresh_entry(_personal(pool), force=False) is None

    assert {entry.id for entry in pool.entries()} == {"personal", "work"}
    assert _personal(pool).last_status == "dead"
    selected = pool.select()
    assert selected is not None and selected.id == "work"
    assert json.loads((root / "auth.json").read_text())["providers"]["openai-codex"]["tokens"] == work


def _refresh_process(root, home, barrier, spent, posted, results):
    """A separate interpreter holds a stale row while its peer refreshes the same grant."""
    os.environ["HERMES_HOME"] = str(home)
    hermes_constants.get_default_hermes_root = lambda: Path(root)

    def respond(request):
        token = parse_qs(request.content.decode())["refresh_token"][0]
        posted.put(token)
        with spent.get_lock():
            if spent.value:
                return httpx.Response(400, json={"error": {"code": "refresh_token_reused"}})
            spent.value = 1
        return httpx.Response(200, json={
            "access_token": _jwt("personal", time.time() + 30000), "refresh_token": "personal-1",
        })

    auth_codex._codex_http_client = lambda **kwargs: httpx.Client(
        transport=httpx.MockTransport(respond), **kwargs)
    pool = load_pool("openai-codex")
    entry = _personal(pool)
    barrier.wait(timeout=15)
    refreshed = pool._refresh_entry(entry, force=True, failed_key=entry.access_token)
    results.put((str(home), refreshed.refresh_token if refreshed else None, _personal(pool).last_status))


def test_two_processes_share_root_generation_without_reusing_the_grant(grants):
    root, _posted, work = grants
    borrower = root / "profiles" / "borrower"
    borrower.mkdir(parents=True)
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    spent = context.Value("i", 0)
    posted, results = context.Queue(), context.Queue()
    processes = [context.Process(target=_refresh_process, args=(root, home, barrier, spent, posted, results))
                 for home in (root, borrower)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=30)
        assert [process.exitcode for process in processes] == [0, 0]
        outcomes = [results.get(timeout=5) for _ in processes]
        assert all(token == "personal-1" and status != "dead" for _home, token, status in outcomes)
        assert posted.get(timeout=5) == "personal-0"
        assert posted.empty()
        _assert_accounts(root, work)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for queue in (posted, results):
            queue.close()
            queue.join_thread()
