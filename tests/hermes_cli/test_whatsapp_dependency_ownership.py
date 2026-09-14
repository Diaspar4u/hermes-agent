import subprocess
from pathlib import Path

import pytest

from hermes_cli.main_platform_setup import _whatsapp_install_bridge
from hermes_cli.web_routers.messaging import _ensure_whatsapp_bridge_dependencies


@pytest.fixture(autouse=True)
def _no_real_subprocesses(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: pytest.fail("unexpected Popen"))
    monkeypatch.setattr("pm.ensure", lambda *a, **kw: pytest.fail("unexpected PM acquisition"))


def test_explicit_maintenance_paths_refresh_and_stamp_whatsapp_dependencies(
    tmp_path, monkeypatch
):
    import hermes_cli.main as hm
    import hermes_constants

    checkout = tmp_path / "checkout"
    bridge_dir = checkout / "scripts" / "whatsapp-bridge"
    checkout.mkdir()
    (checkout / "package.json").write_text("{}", encoding="utf-8")
    (bridge_dir / "node_modules").mkdir(parents=True)
    (bridge_dir / "bridge.js").write_text("// bridge")
    (bridge_dir / "package.json").write_text(
        '{"dependencies": {}}', encoding="utf-8"
    )
    (bridge_dir / "package-lock.json").write_text(
        '{"lockfileVersion": 3}', encoding="utf-8"
    )

    monkeypatch.setattr(hm, "PROJECT_ROOT", checkout)
    monkeypatch.setattr(
        hermes_constants,
        "find_node_executable",
        lambda _name: "/usr/bin/npm",
    )
    monkeypatch.setattr(
        hermes_constants,
        "with_hermes_node_path",
        lambda _env=None: {},
    )

    installs = []
    phase = ["cli"]

    def fake_run(command, *, cwd, **kwargs):
        installs.append(phase[0])
        assert command[1] == "ci"
        assert Path(cwd) != bridge_dir
        (Path(cwd) / "node_modules").mkdir()
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    assert _whatsapp_install_bridge(bridge_dir) == bridge_dir
    stamp = bridge_dir / "node_modules" / ".hermes-pkg-hash"
    cli_stamp = stamp.read_text(encoding="utf-8-sig").strip()
    assert cli_stamp

    phase[0] = "dashboard"
    (bridge_dir / "package.json").write_text(
        '{"dependencies": {"a": "1"}}', encoding="utf-8"
    )
    _ensure_whatsapp_bridge_dependencies(bridge_dir)
    dashboard_stamp = stamp.read_text(encoding="utf-8-sig").strip()
    assert dashboard_stamp and dashboard_stamp != cli_stamp

    phase[0] = "update"
    (bridge_dir / "package-lock.json").write_text(
        '{"lockfileVersion": 3, "packages": {"a": {}}}', encoding="utf-8"
    )
    monkeypatch.setattr(
        "hermes_cli.main_install_repair._warn_configured_features_missing_deps",
        lambda: None,
    )
    from hermes_cli import source_build

    monkeypatch.setattr(source_build, "source_build_env", lambda *, explicit: {"PATH": "/resolved/bin"})
    monkeypatch.setattr(source_build.shutil, "which", lambda name, *, path: "/resolved/npm")
    source_build.build_update_products(checkout, desktop=False)
    update_stamp = stamp.read_text(encoding="utf-8-sig").strip()
    assert update_stamp and update_stamp != dashboard_stamp
    assert installs == ["cli", "dashboard", "update"]

@pytest.mark.parametrize("relocated", [False, True])
@pytest.mark.parametrize("npm_available", [False, True])
def test_explicit_callers_share_the_owner_and_translate_success(
    tmp_path, monkeypatch, relocated, npm_available
):
    from types import SimpleNamespace

    import pm
    import hermes_constants
    from gateway.platforms import whatsapp_common
    from hermes_cli import source_build

    bridge = tmp_path / "scripts" / "whatsapp-bridge"
    (bridge / "node_modules").mkdir(parents=True)
    calls = []
    acquisitions = []
    setup_npm = str(tmp_path / "setup-npm")
    setup_env = {"PATH": "/setup/bin"}
    env = {"PATH": "/resolved/bin", "PYTHON": "/selected/python"}

    def acquire(package, *, explicit):
        acquisitions.append((package, explicit))
        return SimpleNamespace(env=setup_env)

    monkeypatch.setattr(pm, "ensure", acquire)
    monkeypatch.setattr(pm, "installed_package", lambda _: SimpleNamespace(binary=Path(setup_npm)))
    monkeypatch.setattr(hermes_constants, "find_node_executable",
                        lambda _: setup_npm if npm_available else None)
    monkeypatch.setattr(hermes_constants, "with_hermes_node_path", lambda: setup_env)
    monkeypatch.setattr(source_build, "source_build_env", lambda *, explicit: env)
    monkeypatch.setattr(source_build.shutil, "which", lambda name, *, path: "/resolved/npm")

    def shared_owner(target, **kwargs):
        calls.append((target, kwargs))
        return tmp_path / "prepared" if relocated else target

    monkeypatch.setattr(whatsapp_common, "prepare_whatsapp_bridge_runtime", shared_owner)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("caller installed independently"))
    expected = tmp_path / "prepared" if relocated else bridge
    assert _whatsapp_install_bridge(bridge) == expected
    assert _ensure_whatsapp_bridge_dependencies(bridge) == expected
    assert source_build.refresh_installed_whatsapp_bridge(tmp_path) is None
    assert calls == [
        (bridge, {"npm": setup_npm, "env": setup_env}),
        (bridge, {"npm": setup_npm, "env": setup_env}),
        (bridge, {"npm": "/resolved/npm", "env": env}),
    ]
    assert acquisitions == ([] if npm_available else [("npm", True), ("npm", True)])


def test_explicit_callers_translate_typed_failure_and_updater_label(tmp_path, monkeypatch, capsys):
    from fastapi import HTTPException
    from gateway.platforms import whatsapp_common
    import hermes_constants
    from hermes_cli import source_build

    bridge = tmp_path / "scripts" / "whatsapp-bridge"
    (bridge / "node_modules").mkdir(parents=True)
    monkeypatch.setattr(hermes_constants, "find_node_executable", lambda _: "/resolved/npm")
    monkeypatch.setattr(hermes_constants, "with_hermes_node_path", lambda: {})
    monkeypatch.setattr(source_build, "source_build_env", lambda *, explicit: {"PATH": "/resolved/bin"})
    monkeypatch.setattr(source_build.shutil, "which", lambda name, *, path: "/resolved/npm")
    monkeypatch.setattr("hermes_cli.main_install_repair._warn_configured_features_missing_deps", lambda: None)
    monkeypatch.setattr(source_build, "source_frontends", lambda _: pytest.fail("update continued after failure"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("independent install"))

    def failure(*args, **kwargs):
        raise whatsapp_common.WhatsAppBridgeDependencyError("transaction failed")

    monkeypatch.setattr(whatsapp_common, "prepare_whatsapp_bridge_runtime", failure)
    assert _whatsapp_install_bridge(bridge) is None
    with pytest.raises(HTTPException) as error:
        _ensure_whatsapp_bridge_dependencies(bridge)
    assert error.value.status_code == 500
    assert error.value.detail == "transaction failed"
    with pytest.raises(RuntimeError, match="WhatsApp bridge dependency refresh failed: transaction failed") as update_error:
        source_build.build_update_products(tmp_path, desktop=False)
    assert isinstance(update_error.value.__cause__, whatsapp_common.WhatsAppBridgeDependencyError)
    assert "transaction failed" in capsys.readouterr().out


def test_updater_preserves_resolved_npm_and_selected_python_in_filtered_environment(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import pm
    from pm import environments, paths
    import hermes_constants
    from hermes_cli import source_build

    bridge = tmp_path / "scripts" / "whatsapp-bridge"
    (bridge / "node_modules").mkdir(parents=True)
    (bridge / "package.json").write_text("{}")
    (bridge / "package-lock.json").write_text('{"lockfileVersion": 3}')
    (bridge / "bridge.js").write_text("// bridge")
    build_env = {"PYTHON": str(tmp_path / "selected-python"), "PATH": "/safe/bin",
                 "HOME": str(tmp_path / "home"), "TMPDIR": str(tmp_path),
                 "NPM_CONFIG_CACHE": str(tmp_path / "cache"),
                 "OPENAI_API_KEY": "provider-secret", "NPM_TOKEN": "registry-secret",
                 "WHATSAPP_ACCESS_TOKEN": "messaging-secret"}
    for key, value in build_env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(environments, "running_from_selected_environment", lambda _: True)
    monkeypatch.setattr(environments, "project_python", lambda _: Path(build_env["PYTHON"]))
    acquisitions = []

    def acquire(package, *, base_env, explicit):
        acquisitions.append((package, explicit))
        return SimpleNamespace(env=base_env)

    monkeypatch.setattr(pm, "ensure", acquire)

    def resolved_npm(name, *, path):
        assert name == "npm" and path == build_env["PATH"]
        return "/resolved/npm"

    monkeypatch.setattr(source_build.shutil, "which", resolved_npm)
    monkeypatch.setattr(hermes_constants, "find_node_executable", lambda _: pytest.fail("resolved npm discarded"))

    def managed_path(env=None):
        result = dict(env or {})
        if not result.get("PATH", "").startswith("/managed/node:"):
            result["PATH"] = "/managed/node:" + result.get("PATH", "")
        return result

    monkeypatch.setattr(hermes_constants, "with_hermes_node_path", managed_path)
    calls = []

    def fake_npm(command, *, cwd, env, **kwargs):
        calls.append(command)
        assert command == ["/resolved/npm", "ci", "--silent"]
        assert Path(cwd) != bridge
        assert env["PYTHON"] == build_env["PYTHON"]
        assert env["PATH"] == "/managed/node:/safe/bin"
        for key in ("HOME", "TMPDIR", "NPM_CONFIG_CACHE"):
            assert env[key] == build_env[key]
        assert not {"OPENAI_API_KEY", "NPM_TOKEN", "WHATSAPP_ACCESS_TOKEN"} & env.keys()
        (Path(cwd) / "node_modules").mkdir()
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_npm)
    assert source_build.refresh_installed_whatsapp_bridge(tmp_path) is None
    assert len(calls) == 1
    assert acquisitions == [("npm", True)]
