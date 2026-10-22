"""An explicitly selected named host keeps its adapters, credentials and session stores."""
from datetime import UTC, datetime
from pathlib import Path

import pytest


@pytest.fixture
def homes(tmp_path, monkeypatch):
    import hermes_constants
    import hermes_state
    from agent import secret_scope

    root = tmp_path / ".hermes"
    dev = root / "profiles" / "dev"
    doctor = root / "profiles" / "doctor"
    for home, token in ((root, ""), (dev, "dev-token"), (doctor, "doctor-token")):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(
            "gateway:\n  multiplex_profiles: true\n", encoding="utf-8")
        (home / ".env").write_text(f"TELEGRAM_BOT_TOKEN={token}\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(dev))
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    secret_scope.set_multiplex_active(False)
    yield root, dev, doctor
    secret_scope.set_multiplex_active(False)


def test_named_owner_keeps_primary_and_scoped_credentials(homes):
    from aiohttp.test_utils import make_mocked_request
    from agent.secret_scope import get_secret
    from gateway import run, run_bootstrap
    from gateway.config import Platform
    from gateway.platforms.api_server import APIServerAdapter, _api_request_profile
    from gateway.run_shutdown import GatewayShutdownMixin

    root, dev, doctor = homes
    config = run.load_gateway_config_for_runner()
    assert config.platforms[Platform.TELEGRAM].token == "dev-token"
    assert run_bootstrap._launch_home_may_multiplex(config)
    runner = run.GatewayRunner.__new__(run.GatewayRunner)
    runner.config = config
    runner._init_runtime_caches()
    runner.adapters = {Platform.TELEGRAM: object()}
    runner._profile_adapters = {"default": {}, "doctor": {Platform.TELEGRAM: object()}}
    assert runner._primary_profile_name == "dev"
    for home, expected in ((dev, "dev-token"), (doctor, "doctor-token"), (dev, "dev-token")):
        with run._profile_runtime_scope(home):
            assert get_secret("TELEGRAM_BOT_TOKEN") == expected
            assert runner._authorization_adapter(Platform.TELEGRAM, "dev") is runner.adapters[Platform.TELEGRAM]
            assert runner._authorization_adapter(Platform.TELEGRAM, "default") is None
            assert GatewayShutdownMixin._restart_watcher_env()["HERMES_HOME"] == str(dev)
    api = APIServerAdapter.__new__(APIServerAdapter)
    api.gateway_runner = runner
    api._api_key = "dev-api-key-0123456789abcdef"
    assert api._expected_api_key() == api._api_key
    token = _api_request_profile.set("default")
    try:
        with run._profile_runtime_scope(root):
            assert api._expected_api_key() == ""
            assert api._check_auth(make_mocked_request("GET", "/p/default/v1/models")).status == 401
    finally:
        _api_request_profile.reset(token)


def test_named_owner_preserves_current_and_legacy_session_stores(homes):
    from gateway.config import GatewayConfig, Platform
    from gateway.run import _profile_runtime_scope
    from gateway.session import SessionEntry, SessionSource, SessionStore

    root, dev, doctor = homes
    store = SessionStore(dev / "sessions", GatewayConfig(multiplex_profiles=True))
    now = datetime.now(UTC)
    legacy_key = "agent:main:telegram:dm:legacy-chat"
    legacy = SessionEntry(session_key=legacy_key, session_id="legacy-dev", created_at=now,
                          updated_at=now, origin=SessionSource(platform=Platform.TELEGRAM,
                          chat_id="legacy-chat", chat_type="dm"))
    store._routing_db.create_session("legacy-dev", "telegram", profile_name="dev", session_key=legacy_key)
    store._entries[legacy_key] = legacy
    paths = {}
    try:
        for name, home in (("dev", dev), ("default", root), ("doctor", doctor), ("dev", dev)):
            with _profile_runtime_scope(home):
                source = SessionSource(platform=Platform.LOCAL, chat_id="local-" + name,
                                       user_id="owner", chat_type="dm", profile=name)
                key = store._generate_session_key(source)
                db = store._db_for_key(key)
                assert Path(db.db_path) == home / "state.db"
                paths[name] = (key, db)
                if db.get_session("session-" + name) is None:
                    db.create_session("session-" + name, "local", profile_name=name, session_key=key)
                assert store._db_for_key(legacy_key).get_session("legacy-dev")["profile_name"] == "dev"
        for name, (key, db) in paths.items():
            assert store._db_for_key(key).get_session("session-" + name)["profile_name"] == name
            assert all(db.get_session("session-" + other) is None for other in paths if other != name)
    finally:
        for db in store._db_handles.values():
            db.close()
