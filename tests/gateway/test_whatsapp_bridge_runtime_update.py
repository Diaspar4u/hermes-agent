"""Explicit dependency transactions; runtime resolution and services are out of scope."""

from __future__ import annotations

import errno
import json
import logging
import os
import stat
import subprocess
import sys
from pathlib import Path
from subprocess import CompletedProcess
from typing import cast

import pytest

from gateway.platforms import whatsapp_common


@pytest.fixture(autouse=True)
def _isolated_npm_discovery(tmp_path, monkeypatch):
    for key, name in (("HOME", "native-home"), ("USERPROFILE", "native-home"),
                      ("LOCALAPPDATA", "native-appdata"), ("HERMES_HOME", "isolated-profile")):
        isolated = tmp_path / name
        isolated.mkdir(exist_ok=True)
        monkeypatch.setenv(key, str(isolated))
    monkeypatch.setattr("hermes_constants.find_node_executable", lambda name: "/usr/bin/npm")
    # Any forgotten subprocess fake must fail rather than install anything.
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("unexpected subprocess"))


def _seed_runtime(root: Path, version: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "bridge.js").write_text("// unchanged bridge", encoding="utf-8")
    (root / "package.json").write_text(json.dumps({"version": version}), encoding="utf-8")
    (root / "package-lock.json").write_text(
        json.dumps({"lockfileVersion": 3, "version": version}), encoding="utf-8"
    )


def _staging_leftovers(target: Path) -> list[Path]:
    return sorted(target.glob(".node_modules.*-*"))


def _successful_npm_ci(calls: list[tuple[list[str], Path]]):
    def run(command, *, cwd, **kwargs):
        stage = Path(cwd)
        calls.append((list(command), stage))
        assert not (stage / "node_modules").exists()
        modules = stage / "node_modules"
        modules.mkdir()
        (modules / "installed-version").write_text("new\n", encoding="utf-8")
        return CompletedProcess(command, 0, stdout="", stderr="")

    return run


def _file_snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


@pytest.mark.parametrize("failure", ["write", "flush", "timeout"])
def test_first_lock_byte_contention_uses_acquisition_deadline(tmp_path, monkeypatch, failure):
    target = tmp_path / "bridge"
    target.mkdir()
    original_open = whatsapp_common._secure_open_whatsapp_bridge_lock
    opened = []

    class ContendedFile:
        def __init__(self, inner):
            self.inner = inner
            self.conflicted = False

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def write(self, data):
            if failure == "timeout":
                raise PermissionError(errno.EACCES, "mandatory byte lock is busy")
            result = self.inner.write(data)
            if failure == "write" and not self.conflicted:
                # Model the peer's first byte followed by this writer's mandatory-lock conflict.
                self.inner.flush()
                self.conflicted = True
                raise PermissionError(errno.EACCES, "mandatory byte lock is busy")
            return result

        def flush(self):
            self.inner.flush()
            if failure == "flush" and not self.conflicted:
                self.conflicted = True
                raise PermissionError(errno.EACCES, "mandatory byte lock is busy")

    def open_contended(path):
        wrapper = ContendedFile(original_open(path))
        opened.append(wrapper)
        return wrapper

    monkeypatch.setattr(whatsapp_common, "_secure_open_whatsapp_bridge_lock", open_contended)
    if failure == "timeout":
        with pytest.raises(whatsapp_common.WhatsAppBridgeBusyError):
            with whatsapp_common._exclusive_whatsapp_bridge_transaction(target, timeout=0):
                pytest.fail("busy initialization must not enter the transaction")
    else:
        with whatsapp_common._exclusive_whatsapp_bridge_transaction(target, timeout=1) as locked:
            assert locked == target.resolve()
            assert opened[0].conflicted
    assert len(opened) == 1 and opened[0].closed


def test_lock_identity_canonicalizes_symlink_aliases_and_uses_private_root(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / "home"
    physical = tmp_path / "physical"
    (physical / "bridge").mkdir(parents=True)
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(physical, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)

    physical_lock = whatsapp_common._whatsapp_bridge_transaction_lock_path(
        physical / "bridge"
    )
    alias_lock = whatsapp_common._whatsapp_bridge_transaction_lock_path(
        alias / "bridge"
    )
    assert alias_lock == physical_lock

    with whatsapp_common._exclusive_whatsapp_bridge_transaction(alias / "bridge"):
        assert physical_lock.is_file()
    from hermes_constants import _get_platform_default_hermes_home
    assert physical_lock == (
        _get_platform_default_hermes_home() / ".whatsapp-bridge-locks" / "transaction.lock"
    )
    if os.name != "nt":
        assert physical_lock.parent.stat().st_mode & 0o777 == 0o700
        assert physical_lock.stat().st_mode & 0o777 == 0o600


def test_secure_lock_open_rejects_preplanted_symlink(tmp_path, monkeypatch):
    hermes_home = tmp_path / "home"
    bridge_dir = tmp_path / "bridge"
    bridge_dir.mkdir()
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    lock_path = whatsapp_common._whatsapp_bridge_transaction_lock_path(bridge_dir)
    whatsapp_common._validate_private_lock_root(lock_path.parent)
    victim = tmp_path / "victim"
    victim.write_text("unchanged", encoding="utf-8")
    try:
        lock_path.symlink_to(victim)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(
        whatsapp_common.WhatsAppBridgeDependencyError,
        match="symlink or reparse point",
    ):
        with whatsapp_common._exclusive_whatsapp_bridge_transaction(bridge_dir):
            pass
    assert victim.read_text(encoding="utf-8") == "unchanged"


def test_windows_reparse_metadata_is_rejected_for_lock_files(tmp_path):
    lock_file = tmp_path / "lock"
    lock_file.write_bytes(b"\0")
    real_metadata = lock_file.stat()

    class ReparseMetadata:
        st_mode = real_metadata.st_mode
        st_dev = real_metadata.st_dev
        st_ino = real_metadata.st_ino
        st_uid = real_metadata.st_uid
        st_file_attributes = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

    reparse_metadata = cast(os.stat_result, ReparseMetadata())
    assert whatsapp_common._is_windows_reparse_point(reparse_metadata)
    with pytest.raises(OSError, match="opened regular file"):
        whatsapp_common._validate_lock_file_metadata(
            reparse_metadata, reparse_metadata
        )


def test_shared_dependency_installer_is_deterministic_and_transactional(
    tmp_path, monkeypatch
):
    bridge_dir = tmp_path / "bridge"
    _seed_runtime(bridge_dir, "1.0.0")
    old_modules = bridge_dir / "node_modules"
    old_modules.mkdir()
    (old_modules / "old-working-dependency").write_text("keep\n", encoding="utf-8")
    (old_modules / ".hermes-pkg-hash").write_text("stale", encoding="utf-8")
    calls = []

    def successful_ci(command, *, cwd, **kwargs):
        stage = Path(cwd)
        calls.append((list(command), stage))
        assert stage != bridge_dir
        assert stage.stat().st_dev == bridge_dir.stat().st_dev
        assert (old_modules / "old-working-dependency").read_bytes() == b"keep\n"
        assert (old_modules / ".hermes-pkg-hash").read_bytes() == b"stale"
        (stage / "node_modules").mkdir()
        (stage / "node_modules" / "new-dependency").write_text(
            "installed\n", encoding="utf-8"
        )
        return CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", successful_ci)

    assert whatsapp_common.ensure_whatsapp_bridge_dependencies(bridge_dir) is True

    assert calls[0][0][-2:] == ["ci", "--silent"]
    assert not (old_modules / "old-working-dependency").exists()
    assert (old_modules / "new-dependency").exists()
    assert (old_modules / ".hermes-pkg-hash").read_text(
        encoding="utf-8"
    ) == whatsapp_common.whatsapp_bridge_dependency_fingerprint(bridge_dir)
    assert _staging_leftovers(bridge_dir) == []

    assert whatsapp_common.ensure_whatsapp_bridge_dependencies(bridge_dir) is False
    assert len(calls) == 1


def test_failed_shared_dependency_install_keeps_old_tree_and_stamp(
    tmp_path, monkeypatch
):
    bridge_dir = tmp_path / "bridge"
    _seed_runtime(bridge_dir, "1.0.0")
    modules = bridge_dir / "node_modules"
    modules.mkdir()
    (modules / "old-working-dependency").write_text("keep\n", encoding="utf-8")
    stamp = modules / ".hermes-pkg-hash"
    stamp.write_text("old-stamp", encoding="utf-8")
    before = _file_snapshot(modules)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: CompletedProcess(
            command, 1, stdout="", stderr="offline"
        ),
    )

    with pytest.raises(
        whatsapp_common.WhatsAppBridgeDependencyError, match="offline"
    ):
        whatsapp_common.ensure_whatsapp_bridge_dependencies(bridge_dir)

    assert (modules / "old-working-dependency").exists()
    assert stamp.read_text(encoding="utf-8") == "old-stamp"
    assert _file_snapshot(modules) == before
    assert _staging_leftovers(bridge_dir) == []


def test_npm_unavailable_is_typed_and_never_falls_back_to_bare_path(
    tmp_path, monkeypatch
):
    bridge_dir = tmp_path / "bridge"
    _seed_runtime(bridge_dir, "1.0.0")
    modules = bridge_dir / "node_modules"
    modules.mkdir()
    (modules / "old-working-dependency").write_text("keep\n", encoding="utf-8")
    before = _file_snapshot(modules)
    monkeypatch.setattr("hermes_constants.find_node_executable", lambda name: None)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bare npm must never be attempted")
        ),
    )

    with pytest.raises(
        whatsapp_common.WhatsAppBridgeUnavailableError,
        match="npm is unavailable",
    ):
        whatsapp_common.ensure_whatsapp_bridge_dependencies(bridge_dir)

    assert _file_snapshot(modules) == before


def test_npm_receives_minimal_environment_without_provider_or_messaging_secrets(
    tmp_path, monkeypatch
):
    bridge_dir = tmp_path / "bridge"
    _seed_runtime(bridge_dir, "1.0.0")
    captured = {}
    monkeypatch.setattr("hermes_constants.find_node_executable", lambda name: "/npm")
    monkeypatch.setenv("PATH", "/safe/bin")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    monkeypatch.setenv("NPM_CONFIG_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "messaging-secret")
    monkeypatch.setenv("NPM_TOKEN", "registry-secret")

    def successful_ci(command, *, cwd, env, **kwargs):
        captured.update(env)
        (Path(cwd) / "node_modules").mkdir()
        return CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", successful_ci)
    assert whatsapp_common.ensure_whatsapp_bridge_dependencies(bridge_dir) is True

    assert captured["HOME"] == str(tmp_path / "home")
    assert captured["TMPDIR"] == str(tmp_path / "tmp")
    assert captured["NPM_CONFIG_CACHE"] == str(tmp_path / "cache")
    assert all("secret" not in value for value in captured.values())
    assert "OPENAI_API_KEY" not in captured
    assert "WHATSAPP_ACCESS_TOKEN" not in captured
    assert "NPM_TOKEN" not in captured


@pytest.mark.parametrize("env_source", ["process", "explicit"])
def test_npm_transport_environment_survives_the_public_transaction(
    tmp_path, monkeypatch, env_source
):
    bridge_dir = tmp_path / "bridge"
    _seed_runtime(bridge_dir, "1.0.0")
    profile = tmp_path / "profile"
    managed_node = profile / "node"
    managed_node.mkdir(parents=True)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: profile)
    transport = {
        "HTTP_PROXY": "http://http-upper.example.invalid:8080",
        "HTTPS_PROXY": "http://https-upper.example.invalid:8080",
        "ALL_PROXY": "http://all-upper.example.invalid:8080",
        "NO_PROXY": "localhost,.upper.example.invalid",
        "NPM_CONFIG_PROXY": "http://npm-upper.example.invalid:8080",
        "NPM_CONFIG_HTTPS_PROXY": "http://npm-https-upper.example.invalid:8080",
        "NPM_CONFIG_NOPROXY": ".npm-upper.example.invalid",
        "NPM_CONFIG_REGISTRY": "https://registry-upper.example.invalid/",
        "NPM_CONFIG_CAFILE": str(tmp_path / "upper-ca.pem"),
        "NPM_CONFIG_CA": "fixture-upper-ca",
        "NPM_CONFIG_STRICT_SSL": "true",
        "NPM_CONFIG_CACHE": str(tmp_path / "upper-cache"),
    }
    # Windows process environments cannot hold distinct case variants.
    # Explicit dictionaries still exercise npm's lowercase config precedence.
    if env_source == "explicit" or os.name != "nt":
        transport.update({
            key.lower(): value.replace("upper", "lower")
            for key, value in list(transport.items())
        })
    transport.update({
        "NODE_EXTRA_CA_CERTS": str(tmp_path / "node-ca.pem"),
        "SSL_CERT_FILE": str(tmp_path / "ssl-ca.pem"),
        "SSL_CERT_DIR": str(tmp_path / "ssl-certs"),
    })
    unrelated = {
        "OPENAI_API_KEY": "provider-sentinel",
        "WHATSAPP_ACCESS_TOKEN": "messaging-sentinel",
        "NPM_TOKEN": "registry-sentinel",
        "NPM_CONFIG_UNRELATED_SECRET": "unrelated-sentinel",
        "NODE_OPTIONS": "unrelated-node-option",
    }
    supplied = {
        "PATH": str(tmp_path / "safe-bin"),
        "PYTHON": "/nix/store/python/bin/python3",
        **transport,
        **unrelated,
    }
    before = dict(supplied)
    if env_source == "process":
        for key, value in supplied.items():
            monkeypatch.setenv(key, value)
    else:
        monkeypatch.setenv("HTTP_PROXY", "http://ambient.example.invalid:8080")
        monkeypatch.setenv("NPM_CONFIG_USERCONFIG", str(tmp_path / "ambient.npmrc"))
    captured = []

    def fake_npm(command, *, cwd, env, **kwargs):
        captured.append((command, dict(env)))
        (Path(cwd) / "node_modules").mkdir()
        return CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_npm)
    assert whatsapp_common.ensure_whatsapp_bridge_dependencies(
        bridge_dir, npm="/resolved/npm",
        env=supplied if env_source == "explicit" else None,
    ) is True
    assert supplied == before
    assert len(captured) == 1
    command, child_env = captured[0]
    assert command == ["/resolved/npm", "ci", "--silent"]
    assert not unrelated.keys() & {key.upper() for key in child_env}
    assert child_env["PYTHON"] == supplied["PYTHON"]
    assert child_env["PATH"] == os.pathsep.join((str(managed_node), supplied["PATH"]))
    if env_source == "explicit":
        assert "NPM_CONFIG_USERCONFIG" not in {key.upper() for key in child_env}
    else:
        assert all(os.environ[key] == value for key, value in supplied.items())
    assert not transport.keys() - child_env.keys()
    assert {key: child_env[key] for key in transport} == transport
    assert whatsapp_common.whatsapp_bridge_dependencies_fresh(bridge_dir)
    assert _staging_leftovers(bridge_dir) == []


@pytest.mark.parametrize("env_source,npmrc_kind", [
    ("process", "profile"),
    ("explicit", "profile"),
    ("explicit", "upper"),
    ("explicit", "lower"),
    ("explicit", "both"),
    ("explicit", "empty"),
    ("explicit", "missing"),
])
def test_npm_userconfig_precedence_reaches_the_public_transaction(
    tmp_path, monkeypatch, env_source, npmrc_kind
):
    bridge_dir = tmp_path / "bridge"
    _seed_runtime(bridge_dir, "1.0.0")
    profile = tmp_path / "profile"
    profile.mkdir()
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: profile)
    npmrc = profile / "npmrc"
    if npmrc_kind != "missing":
        npmrc.write_text("registry=https://profile.example.invalid/\n", encoding="utf-8")
    supplied = {}
    if npmrc_kind in {"upper", "both", "empty"}:
        supplied["NPM_CONFIG_USERCONFIG"] = (
            "" if npmrc_kind == "empty" else str(tmp_path / "explicit-upper.npmrc")
        )
    if npmrc_kind in {"lower", "both"}:
        supplied["npm_config_userconfig"] = str(tmp_path / "explicit-lower.npmrc")
    before = dict(supplied)
    if env_source == "process":
        monkeypatch.delenv("NPM_CONFIG_USERCONFIG", raising=False)
        monkeypatch.delenv("npm_config_userconfig", raising=False)
    else:
        monkeypatch.setenv("NPM_CONFIG_USERCONFIG", str(tmp_path / "ambient.npmrc"))
        monkeypatch.setenv("HTTPS_PROXY", "http://ambient.example.invalid:8080")
    captured = []

    def fake_npm(command, *, cwd, env, **kwargs):
        captured.append(dict(env))
        (Path(cwd) / "node_modules").mkdir()
        return CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_npm)
    assert whatsapp_common.ensure_whatsapp_bridge_dependencies(
        bridge_dir, env=supplied if env_source == "explicit" else None,
    ) is True
    assert supplied == before
    assert len(captured) == 1
    child_env = captured[0]
    if env_source == "explicit":
        assert "HTTPS_PROXY" not in {key.upper() for key in child_env}
    expected = supplied or (
        {"NPM_CONFIG_USERCONFIG": str(npmrc)} if npmrc_kind != "missing" else {}
    )
    assert {
        key: value for key, value in child_env.items()
        if key.upper() == "NPM_CONFIG_USERCONFIG"
    } == expected


def test_dependency_error_is_bounded_and_redacts_registry_credentials(
    tmp_path, monkeypatch
):
    bridge_dir = tmp_path / "bridge"
    _seed_runtime(bridge_dir, "1.0.0")
    secret = "npm_abcdefghijklmnopqrstuvwxyz"
    registry_url = f"https://user:{secret}@registry.example.invalid/pkg"
    noisy_error = "\n".join(["diagnostic"] * 30 + [registry_url])
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: CompletedProcess(
            command, 1, stdout="", stderr=noisy_error
        ),
    )

    with pytest.raises(whatsapp_common.WhatsAppBridgeDependencyError) as error:
        whatsapp_common.ensure_whatsapp_bridge_dependencies(bridge_dir)

    detail = str(error.value)
    assert secret not in detail
    assert registry_url not in detail
    assert "<redacted-url>" in detail
    assert len(detail) < 1400


def test_dependency_promotion_failure_rolls_back_exact_old_tree(
    tmp_path, monkeypatch
):
    bridge_dir = tmp_path / "bridge"
    _seed_runtime(bridge_dir, "1.0.0")
    modules = bridge_dir / "node_modules"
    modules.mkdir()
    (modules / "old-working-dependency").write_text("keep\n", encoding="utf-8")
    (modules / ".hermes-pkg-hash").write_text("old-stamp", encoding="utf-8")
    before = _file_snapshot(modules)
    calls = []
    monkeypatch.setattr(
        subprocess, "run", _successful_npm_ci(calls)
    )
    real_replace = os.replace

    def fail_dependency_promotion(source, destination):
        source_path = Path(source)
        if source_path.name == "node_modules" and Path(destination) == modules:
            raise OSError("simulated dependency promotion failure")
        return real_replace(source, destination)

    monkeypatch.setattr(whatsapp_common.os, "replace", fail_dependency_promotion)

    with pytest.raises(whatsapp_common.WhatsAppBridgeDependencyError):
        whatsapp_common.ensure_whatsapp_bridge_dependencies(bridge_dir)

    assert _file_snapshot(modules) == before
    assert not list(bridge_dir.glob(".node_modules.staging-*"))
    assert not list(bridge_dir.glob(".node_modules.backup-*"))


def test_dependency_restore_failure_keeps_recovery_and_both_diagnostics(
    tmp_path, monkeypatch
):
    bridge_dir = tmp_path / "bridge"
    _seed_runtime(bridge_dir, "1.0.0")
    modules = bridge_dir / "node_modules"
    modules.mkdir()
    (modules / "old-working-dependency").write_text("keep\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        subprocess, "run", _successful_npm_ci(calls)
    )
    real_replace = os.replace

    def fail_promotion_and_restore(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path.name == "node_modules" and destination_path == modules:
            raise OSError("simulated dependency promotion failure")
        if source_path.name.startswith(".node_modules.backup-"):
            raise OSError("simulated dependency restore failure")
        return real_replace(source, destination)

    monkeypatch.setattr(
        whatsapp_common.os, "replace", fail_promotion_and_restore
    )

    with pytest.raises(
        whatsapp_common.WhatsAppBridgeDependencyError,
        match="Rollback also failed",
    ) as error:
        whatsapp_common.ensure_whatsapp_bridge_dependencies(bridge_dir)

    detail = str(error.value)
    assert "simulated dependency promotion failure" in detail
    assert "simulated dependency restore failure" in detail
    assert "Recovery data was preserved" in detail
    backups = list(bridge_dir.glob(".node_modules.backup-*"))
    staging = list(bridge_dir.glob(".node_modules.staging-*"))
    assert len(backups) == 1
    assert len(staging) == 1
    assert (backups[0] / "old-working-dependency").exists()


def test_post_promotion_manifest_mismatch_rolls_dependencies_back(
    tmp_path, monkeypatch
):
    bridge_dir = tmp_path / "bridge"
    _seed_runtime(bridge_dir, "1.0.0")
    modules = bridge_dir / "node_modules"
    modules.mkdir()
    (modules / "old-working-dependency").write_text("keep\n", encoding="utf-8")
    (modules / ".hermes-pkg-hash").write_text("old-stamp", encoding="utf-8")
    before = _file_snapshot(modules)
    calls = []
    monkeypatch.setattr(
        subprocess, "run", _successful_npm_ci(calls)
    )
    real_replace = os.replace

    def tamper_after_dependency_promotion(source, destination):
        result = real_replace(source, destination)
        if Path(source).name == "node_modules" and Path(destination) == modules:
            (bridge_dir / "package-lock.json").write_text(
                '{"tampered": true}\n', encoding="utf-8"
            )
        return result

    monkeypatch.setattr(
        whatsapp_common.os, "replace", tamper_after_dependency_promotion
    )

    with pytest.raises(
        whatsapp_common.WhatsAppBridgeDependencyError,
        match="changed after promotion",
    ):
        whatsapp_common.ensure_whatsapp_bridge_dependencies(bridge_dir)

    assert _file_snapshot(modules) == before
    assert not list(bridge_dir.glob(".node_modules.backup-*"))


def test_lock_timeout_fails_closed_before_dependency_freshness_or_install(
    tmp_path, monkeypatch
):
    bridge_dir = tmp_path / "bridge"
    _seed_runtime(bridge_dir, "1.0.0")
    modules = bridge_dir / "node_modules"
    modules.mkdir()
    (modules / "old-working-dependency").write_text("keep\n", encoding="utf-8")
    (modules / ".hermes-pkg-hash").write_text("old-stamp", encoding="utf-8")
    before = _file_snapshot(modules)

    def always_busy(lock_file):
        raise BlockingIOError(errno.EAGAIN, "held by another process")

    monkeypatch.setattr(
        whatsapp_common, "_try_acquire_whatsapp_bridge_file_lock", always_busy
    )
    monkeypatch.setattr(whatsapp_common, "_WHATSAPP_BRIDGE_LOCK_TIMEOUT_SECONDS", 0)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("npm must not run without the transaction lock")
        ),
    )

    monkeypatch.setattr(whatsapp_common, "whatsapp_bridge_dependencies_fresh",
                        lambda _: pytest.fail("freshness checked before lock"))

    with pytest.raises(
        whatsapp_common.WhatsAppBridgeBusyError,
        match="Timed out waiting for another Hermes process",
    ):
        whatsapp_common.ensure_whatsapp_bridge_dependencies(bridge_dir)

    assert _file_snapshot(modules) == before
    lock_path = whatsapp_common._whatsapp_bridge_transaction_lock_path(bridge_dir)
    assert bridge_dir not in lock_path.parents
    assert lock_path.parent != bridge_dir


def test_precommit_staging_cleanup_failure_preserves_primary_error_and_old_stamp(
    tmp_path, monkeypatch, caplog
):
    bridge_dir = tmp_path / "bridge"
    _seed_runtime(bridge_dir, "1.0.0")
    modules = bridge_dir / "node_modules"
    modules.mkdir()
    (modules / "old-working-dependency").write_text("keep\n", encoding="utf-8")
    (modules / ".hermes-pkg-hash").write_text("old-stamp", encoding="utf-8")
    before = _file_snapshot(modules)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: CompletedProcess(
            command, 1, stdout="", stderr="primary npm failure"
        ),
    )
    real_remove = whatsapp_common._remove_path_without_following
    secret = "npm_cleanup_secret"

    def fail_staging_cleanup(path):
        if Path(path).name.startswith(".node_modules.staging-"):
            raise OSError(
                f"cleanup failed at https://user:{secret}@registry.invalid/stage"
            )
        return real_remove(path)

    monkeypatch.setattr(
        whatsapp_common, "_remove_path_without_following", fail_staging_cleanup
    )
    caplog.set_level(logging.WARNING, logger=whatsapp_common.logger.name)

    with pytest.raises(
        whatsapp_common.WhatsAppBridgeDependencyError, match="primary npm failure"
    ):
        whatsapp_common.ensure_whatsapp_bridge_dependencies(bridge_dir)

    assert _file_snapshot(modules) == before
    assert secret not in caplog.text
    assert "<redacted-url>" in caplog.text
    assert "after failure" in caplog.text


def test_postcommit_cleanup_failures_are_redacted_and_do_not_replace_success(
    tmp_path, monkeypatch, caplog
):
    bridge_dir = tmp_path / "bridge"
    _seed_runtime(bridge_dir, "1.0.0")
    modules = bridge_dir / "node_modules"
    modules.mkdir()
    (modules / "old-working-dependency").write_text("keep\n", encoding="utf-8")
    (modules / ".hermes-pkg-hash").write_text("old-stamp", encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        subprocess, "run", _successful_npm_ci(calls)
    )
    real_remove = whatsapp_common._remove_path_without_following
    secret = "npm_cleanup_secret"

    def fail_postcommit_cleanup(path):
        name = Path(path).name
        if name.startswith((".node_modules.backup-", ".node_modules.staging-")):
            raise OSError(
                f"cleanup failed at https://user:{secret}@registry.invalid/postcommit"
            )
        return real_remove(path)

    monkeypatch.setattr(
        whatsapp_common, "_remove_path_without_following", fail_postcommit_cleanup
    )
    caplog.set_level(logging.WARNING, logger=whatsapp_common.logger.name)

    assert whatsapp_common.ensure_whatsapp_bridge_dependencies(bridge_dir) is True

    assert not (modules / "old-working-dependency").exists()
    assert (modules / "installed-version").read_text(encoding="utf-8") == "new\n"
    assert (modules / ".hermes-pkg-hash").read_text(
        encoding="utf-8"
    ) == whatsapp_common.whatsapp_bridge_dependency_fingerprint(bridge_dir)
    assert secret not in caplog.text
    assert "<redacted-url>" in caplog.text
    assert "after activation" in caplog.text


@pytest.mark.parametrize("alias_kind", ["same-path", "symlink", "case", "parent-case"])
def test_cross_profile_alias_installers_contend_and_recheck_freshness(
    tmp_path, monkeypatch, alias_kind
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    bridge_dir = tmp_path / "physical" / "bridge"
    _seed_runtime(bridge_dir, "1.0.0")
    if alias_kind == "symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(bridge_dir, target_is_directory=True)
    elif alias_kind in {"case", "parent-case"}:
        alias = (
            bridge_dir.with_name("BRIDGE") if alias_kind == "case"
            else tmp_path / "PHYSICAL" / "BRIDGE"
        )
        # Never create the alternate spelling: its existence must come from the
        # fixture filesystem's own case-alias semantics, not a second directory.
        try:
            alias.stat()
        except FileNotFoundError:
            pytest.skip("case-sensitive fixture filesystem: case alias does not exist")
    else:
        alias = bridge_dir
    assert os.path.samefile(bridge_dir, alias)
    root_before = bridge_dir.stat()
    alias_before = alias.stat()
    print(f"alias-proof kind={alias_kind} target={bridge_dir} alias={alias} "
          f"samefile={os.path.samefile(bridge_dir, alias)} "
          f"target-id={(root_before.st_dev, root_before.st_ino)} "
          f"alias-id={(alias_before.st_dev, alias_before.st_ino)}")
    modules = bridge_dir / "node_modules"
    modules.mkdir()
    (modules / ".hermes-pkg-hash").write_text("stale", encoding="utf-8")
    modules_before = modules.stat()
    old_bytes = _file_snapshot(modules)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-one"))
    monkeypatch.setattr(whatsapp_common, "_WHATSAPP_BRIDGE_LOCK_TIMEOUT_SECONDS", 0.1)
    entered, release = Event(), Event()
    calls = []

    def held_ci(command, *, cwd, **kwargs):
        calls.append(Path(cwd))
        if len(calls) == 1:
            entered.set()
            assert release.wait(10), "second installer did not finish"
        # A broken lock admits the second fake npm immediately, without making
        # the test depend on a deadlock or executing a real package manager.
        return _successful_npm_ci([])(command, cwd=cwd, **kwargs)

    monkeypatch.setattr(subprocess, "run", held_ci)
    busy = None
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(whatsapp_common.ensure_whatsapp_bridge_dependencies, bridge_dir)
        try:
            assert entered.wait(10), "first installer did not reach npm"
            monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-two"))
            try:
                whatsapp_common.ensure_whatsapp_bridge_dependencies(alias)
            except whatsapp_common.WhatsAppBridgeBusyError as exc:
                busy = exc
            print(f"contention-proof kind={alias_kind} npm-entries={len(calls)} "
                  f"busy={busy is not None} first-blocked={not release.is_set()}")
            assert len(calls) == 1, "second npm entered while first held the transaction"
            assert busy is not None, "alias did not time out on the held file lock"
            assert _file_snapshot(modules) == old_bytes
            held_lock = whatsapp_common._whatsapp_bridge_transaction_lock_path(bridge_dir)
            alias_lock = whatsapp_common._whatsapp_bridge_transaction_lock_path(alias)
            assert os.path.samefile(held_lock, alias_lock)
            held_lock_before = held_lock.stat()
        finally:
            release.set()
        assert first.result(timeout=10) is True
    assert whatsapp_common.ensure_whatsapp_bridge_dependencies(alias) is False
    assert len(calls) == 1
    root_after = bridge_dir.stat()
    assert (root_after.st_dev, root_after.st_ino) == (root_before.st_dev, root_before.st_ino)
    modules_after = modules.stat()
    assert (modules_after.st_dev, modules_after.st_ino) != (
        modules_before.st_dev, modules_before.st_ino
    )
    retry_lock = whatsapp_common._whatsapp_bridge_transaction_lock_path(alias)
    retry_lock_after = retry_lock.stat()
    assert (retry_lock_after.st_dev, retry_lock_after.st_ino) == (
        held_lock_before.st_dev, held_lock_before.st_ino
    )
    assert _staging_leftovers(bridge_dir) == []
    print(f"retry-proof kind={alias_kind} fresh=True npm-entries={len(calls)} "
          "bridge-root-stable=True modules-replaced=True lock-inode-stable=True")


def test_bridge_root_replaced_while_waiting_is_validated_under_stable_lock(tmp_path, monkeypatch):
    bridge_dir = tmp_path / "bridge"
    _seed_runtime(bridge_dir, "1.0.0")
    moved = tmp_path / "old-bridge"
    real_acquire = whatsapp_common._try_acquire_whatsapp_bridge_file_lock
    attempts = []

    def replace_root_during_wait(lock_file):
        attempts.append(lock_file.name)
        if len(attempts) == 1:
            bridge_dir.rename(moved)
            _seed_runtime(bridge_dir, "1.0.0")
            raise BlockingIOError(errno.EAGAIN, "contended before root replacement")
        return real_acquire(lock_file)

    monkeypatch.setattr(whatsapp_common, "_try_acquire_whatsapp_bridge_file_lock",
                        replace_root_during_wait)
    monkeypatch.setattr(subprocess, "run", _successful_npm_ci([]))
    assert whatsapp_common.ensure_whatsapp_bridge_dependencies(bridge_dir) is True
    assert len(attempts) == 2
    assert _staging_leftovers(bridge_dir) == []
    assert whatsapp_common.whatsapp_bridge_dependencies_fresh(bridge_dir)


def test_unusable_bridge_root_identity_fails_closed(tmp_path, monkeypatch):
    from types import SimpleNamespace

    bridge_dir = tmp_path / "bridge"
    _seed_runtime(bridge_dir, "1.0.0")
    real_lstat = Path.lstat
    metadata = bridge_dir.lstat()

    def unavailable_identity(path, *args, **kwargs):
        if path == bridge_dir:
            return SimpleNamespace(st_mode=metadata.st_mode, st_dev=metadata.st_dev,
                                   st_ino=0, st_file_attributes=0)
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", unavailable_identity)
    monkeypatch.setattr(whatsapp_common, "whatsapp_bridge_dependencies_fresh",
                        lambda _: pytest.fail("freshness checked without physical root identity"))
    with pytest.raises(whatsapp_common.WhatsAppBridgeDependencyError, match="physical directory identity"):
        whatsapp_common.ensure_whatsapp_bridge_dependencies(bridge_dir)
    assert not (tmp_path / ".whatsapp-bridge-locks").exists()
    assert _staging_leftovers(bridge_dir) == []


@pytest.mark.parametrize("failure", ["timeout", "missing-modules", "staged-manifest", "live-manifest", "stamp"])
def test_pre_activation_failures_preserve_old_bytes(tmp_path, monkeypatch, failure):
    bridge_dir = tmp_path / "bridge"
    _seed_runtime(bridge_dir, "1.0.0")
    modules = bridge_dir / "node_modules"
    modules.mkdir()
    (modules / "old").write_bytes(b"original")
    (modules / ".hermes-pkg-hash").write_bytes(b"old-stamp\n")
    before = _file_snapshot(modules)
    monkeypatch.setenv("WHATSAPP_NPM_INSTALL_TIMEOUT", "17")

    def failed_ci(command, *, cwd, timeout, **kwargs):
        assert timeout == 17
        assert kwargs["encoding"] == "utf-8"
        assert kwargs["errors"] == "replace"
        stage = Path(cwd)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, timeout)
        if failure != "missing-modules":
            (stage / "node_modules").mkdir()
        if failure == "staged-manifest":
            (stage / "package-lock.json").write_text("tampered")
        if failure == "live-manifest":
            (bridge_dir / "package.json").write_text("tampered")
        if failure == "stamp":
            (stage / "node_modules" / ".hermes-pkg-hash").mkdir()
        return CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", failed_ci)
    with pytest.raises(whatsapp_common.WhatsAppBridgeDependencyError):
        whatsapp_common.ensure_whatsapp_bridge_dependencies(bridge_dir)
    assert _file_snapshot(modules) == before
    assert _staging_leftovers(bridge_dir) == []


def test_post_promotion_verification_exception_restores_old_bytes(tmp_path, monkeypatch):
    bridge_dir = tmp_path / "bridge"
    _seed_runtime(bridge_dir, "1.0.0")
    modules = bridge_dir / "node_modules"
    modules.mkdir()
    (modules / ".hermes-pkg-hash").write_bytes(b"stale\n")
    before = _file_snapshot(modules)
    monkeypatch.setattr(subprocess, "run", _successful_npm_ci([]))
    real_replace = os.replace

    def unreadable_stamp_after_promotion(source, destination):
        if Path(source).name == "node_modules" and Path(destination) == modules:
            assert (Path(source) / ".hermes-pkg-hash").read_text() == (
                whatsapp_common.whatsapp_bridge_dependency_fingerprint(bridge_dir)
            )
            result = real_replace(source, destination)
            (modules / ".hermes-pkg-hash").write_bytes(b"\xff")
            return result
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", unreadable_stamp_after_promotion)
    with pytest.raises(whatsapp_common.WhatsAppBridgeDependencyError):
        whatsapp_common.ensure_whatsapp_bridge_dependencies(bridge_dir)
    assert _file_snapshot(modules) == before
    assert _staging_leftovers(bridge_dir) == []


def test_lock_release_failure_does_not_replace_install_result(tmp_path, monkeypatch, caplog):
    bridge_dir = tmp_path / "bridge"
    _seed_runtime(bridge_dir, "1.0.0")
    monkeypatch.setattr(subprocess, "run", _successful_npm_ci([]))

    def fail_release(_):
        raise OSError("release https://user:secret@registry.invalid")

    monkeypatch.setattr(whatsapp_common, "_release_whatsapp_bridge_file_lock", fail_release)
    assert whatsapp_common.ensure_whatsapp_bridge_dependencies(bridge_dir) is True
    assert "secret" not in caplog.text
    assert "<redacted-url>" in caplog.text


@pytest.mark.parametrize("control_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("phase,prior", [
    (phase, prior)
    for phase in ("backup", "promotion", "verify-stage", "verify-live", "verify-fresh")
    for prior in ("stamped", "unstamped", "absent")
    if phase != "backup" or prior != "absent"
])
def test_activation_control_exception_restores_exact_prior_state(
    tmp_path, monkeypatch, control_type, phase, prior
):
    bridge = tmp_path / "bridge"
    _seed_runtime(bridge, "1.0.0")
    modules = bridge / "node_modules"
    if prior != "absent":
        modules.mkdir()
        (modules / "old").write_bytes(b"old dependency\x00\xff\n")
        if prior == "stamped":
            (modules / ".hermes-pkg-hash").write_bytes(b"old-stamp\r\n")
    before = _file_snapshot(modules)
    monkeypatch.setattr(subprocess, "run", _successful_npm_ci([]))
    real_replace = os.replace
    real_fingerprint = whatsapp_common.whatsapp_bridge_dependency_fingerprint
    real_fresh = whatsapp_common.whatsapp_bridge_dependencies_fresh
    control = control_type("activation interrupted")
    promoted = False
    interrupted = False

    def interrupt(at):
        nonlocal interrupted
        if phase == at and not interrupted:
            interrupted = True
            raise control

    def replace(source, destination):
        nonlocal promoted
        result = real_replace(source, destination)
        if Path(destination).name.startswith(".node_modules.backup-"):
            interrupt("backup")
        elif Path(source).name == "node_modules" and Path(destination) == modules:
            promoted = True
            interrupt("promotion")
        return result

    def fingerprint(path):
        result = real_fingerprint(path)
        if promoted:
            interrupt("verify-live" if Path(path) == bridge else "verify-stage")
        return result

    def fresh(path):
        result = real_fresh(path)
        if promoted:
            assert result is True
            interrupt("verify-fresh")
        return result

    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(whatsapp_common, "whatsapp_bridge_dependency_fingerprint", fingerprint)
    monkeypatch.setattr(whatsapp_common, "whatsapp_bridge_dependencies_fresh", fresh)
    with pytest.raises(control_type) as caught:
        whatsapp_common.ensure_whatsapp_bridge_dependencies(bridge)
    assert caught.value is control
    assert interrupted
    assert modules.exists() == (prior != "absent")
    assert _file_snapshot(modules) == before
    assert _staging_leftovers(bridge) == []


@pytest.mark.parametrize("control_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("phase", ["backup", "promotion"])
@pytest.mark.parametrize("restore_error_type", [OSError, KeyboardInterrupt])
def test_control_exception_rollback_failure_retains_recovery_and_redacted_note(
    tmp_path, monkeypatch, control_type, phase, restore_error_type
):
    bridge = tmp_path / "bridge"
    _seed_runtime(bridge, "1.0.0")
    modules = bridge / "node_modules"
    modules.mkdir()
    (modules / "old").write_bytes(b"original\x00\xff")
    (modules / ".hermes-pkg-hash").write_bytes(b"old-stamp\r\n")
    before = _file_snapshot(modules)
    monkeypatch.setattr(subprocess, "run", _successful_npm_ci([]))
    real_replace = os.replace
    control = control_type("activation interrupted")
    restore_error = restore_error_type(
        "restore denied https://user:rollback-secret@registry.invalid/recovery"
    )

    def replace(source, destination):
        if Path(source).name.startswith(".node_modules.backup-"):
            raise restore_error
        result = real_replace(source, destination)
        if (
            phase == "backup" and Path(destination).name.startswith(".node_modules.backup-")
        ) or (
            phase == "promotion" and Path(source).name == "node_modules"
            and Path(destination) == modules
        ):
            raise control
        return result

    monkeypatch.setattr(os, "replace", replace)
    with pytest.raises(control_type) as caught:
        whatsapp_common.ensure_whatsapp_bridge_dependencies(bridge)
    assert caught.value is control
    notes = "\n".join(getattr(control, "__notes__", []))
    assert "activation interrupted" in notes
    assert "Rollback also failed" in notes
    assert "restore denied" in notes
    assert "Recovery data was preserved" in notes
    assert "rollback-secret" not in notes
    assert "<redacted-url>" in notes
    backups = list(bridge.glob(".node_modules.backup-*"))
    stages = list(bridge.glob(".node_modules.staging-*"))
    assert len(backups) == len(stages) == 1
    assert _file_snapshot(backups[0]) == before
    retained_modules = stages[0] / (
        "node_modules" if phase == "backup" else ".rejected-node_modules"
    )
    assert (retained_modules / "installed-version").read_bytes() == b"new\n"
    assert not modules.exists()


def test_stable_lock_covers_absent_targets_and_context_overrides(tmp_path, monkeypatch):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    bridge = tmp_path / "initially-absent-mirror"
    other = tmp_path / "other-target"
    _seed_runtime(other, "1.0.0")
    monkeypatch.setattr(whatsapp_common, "_WHATSAPP_BRIDGE_LOCK_TIMEOUT_SECONDS", 0)
    with whatsapp_common._exclusive_whatsapp_bridge_transaction(bridge):
        token = set_hermes_home_override(tmp_path / "context-profile")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "different-root"))
        try:
            for target in (bridge, other):
                with pytest.raises(whatsapp_common.WhatsAppBridgeBusyError):
                    whatsapp_common.ensure_whatsapp_bridge_dependencies(target)
        finally:
            reset_hermes_home_override(token)
    assert not bridge.exists()
    assert _staging_leftovers(other) == []


def test_native_lock_unavailable_fails_before_target_mutation(tmp_path, monkeypatch):
    from hermes_constants import _get_platform_default_hermes_home

    bridge = tmp_path / "bridge"
    _seed_runtime(bridge, "1.0.0")
    before = _file_snapshot(bridge)
    native_root = _get_platform_default_hermes_home()
    native_root.write_bytes(b"unavailable home")
    monkeypatch.setattr(subprocess, "run", _successful_npm_ci([]))
    with pytest.raises(whatsapp_common.WhatsAppBridgeDependencyError, match="lock directory"):
        whatsapp_common.ensure_whatsapp_bridge_dependencies(bridge)
    assert _file_snapshot(bridge) == before
    assert _staging_leftovers(bridge) == []
    assert not (bridge.parent / ".whatsapp-bridge-locks").exists()


def _fixture_npm_executable(tmp_path):
    executable = tmp_path / "fixture-npm"
    receipt = tmp_path / "fixture-npm.jsonl"
    executable.write_text(
        f"#!{sys.executable} -B\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        f"with Path({str(receipt)!r}).open('a') as log:\n"
        "    log.write(json.dumps({'argv': sys.argv, 'cwd': os.getcwd(), "
        "'env': dict(os.environ)}) + '\\n')\n"
        "assert sys.argv[1:] == ['ci', '--silent']\n"
        "modules = Path('node_modules')\n"
        "modules.mkdir()\n"
        "(modules / 'installed-version').write_bytes(b'fixture-subprocess\\n')\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    return executable, receipt


def _maintenance_child_environment(tmp_path, profile):
    env = {
        key: os.environ[key] for key in
        ("HOME", "USERPROFILE", "LOCALAPPDATA", "PATH", "TMPDIR")
        if key in os.environ
    }
    env.update({
        "HERMES_HOME": str(tmp_path / profile),
        "PYTHONPATH": str(Path(whatsapp_common.__file__).resolve().parents[2]),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "NPM_TOKEN": "must-not-reach-fixture",
    })
    return env


def _public_installer_child(bridge, executable, env):
    script = """
import json, sys
from pathlib import Path
from gateway.platforms import whatsapp_common as common
from hermes_constants import set_hermes_home_override
set_hermes_home_override(Path(sys.argv[3]) / 'context')
common._WHATSAPP_BRIDGE_LOCK_TIMEOUT_SECONDS = 0.15
try:
    result = common.ensure_whatsapp_bridge_dependencies(Path(sys.argv[1]), npm=sys.argv[2])
    print(json.dumps({'result': result}))
except common.WhatsAppBridgeDependencyError as error:
    print(json.dumps({'error': type(error).__name__, 'detail': str(error)}))
"""
    with subprocess.Popen(
        [sys.executable, "-B", "-c", script, str(bridge), str(executable), env["HERMES_HOME"]],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ) as child:
        output, errors = child.communicate(timeout=20)
        assert child.returncode == 0, errors
        return json.loads(output)


@pytest.mark.macos_only
@pytest.mark.parametrize("alias_kind", ["same-path", "symlink", "case", "parent-case"])
def test_subprocess_public_installer_excluded_during_root_replacement(
    tmp_path, alias_kind
):
    bridge = tmp_path / "physical" / "bridge"
    _seed_runtime(bridge, "1.0.0")
    replacement, moved = tmp_path / "replacement", tmp_path / "moved"
    _seed_runtime(replacement, "2.0.0")
    alias = bridge
    if alias_kind == "symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(bridge, target_is_directory=True)
    elif alias_kind in {"case", "parent-case"}:
        alias = bridge.with_name("BRIDGE") if alias_kind == "case" else (
            tmp_path / "PHYSICAL" / "BRIDGE"
        )
        if not alias.exists():
            pytest.skip("case-sensitive fixture filesystem")
    assert os.path.samefile(bridge, alias)
    old_identity = bridge.stat().st_ino
    executable, receipt = _fixture_npm_executable(tmp_path)
    holder_script = """
import os, sys
from pathlib import Path
from gateway.platforms import whatsapp_common as common
from hermes_constants import set_hermes_home_override
bridge, moved, replacement = map(Path, sys.argv[1:4])
set_hermes_home_override(Path(sys.argv[4]) / 'context')
with common._exclusive_whatsapp_bridge_transaction(bridge, timeout=1):
    os.replace(bridge, moved)
    print('gap', flush=True)
    assert sys.stdin.readline().strip() == 'promote'
    os.replace(replacement, bridge)
    print('replaced', flush=True)
    assert sys.stdin.readline().strip() == 'release'
"""
    holder_env = _maintenance_child_environment(tmp_path, "profile-one")
    contender_env = _maintenance_child_environment(tmp_path, "profile-two")
    with subprocess.Popen(
        [sys.executable, "-B", "-c", holder_script, str(bridge), str(moved),
         str(replacement), holder_env["HERMES_HOME"]],
        env=holder_env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    ) as holder:
        assert holder.stdin is not None and holder.stdout is not None
        try:
            assert holder.stdout.readline().strip() == "gap"
            gap_result = _public_installer_child(alias, executable, contender_env)
            holder.stdin.write("promote\n")
            holder.stdin.flush()
            assert holder.stdout.readline().strip() == "replaced"
            assert bridge.stat().st_ino != old_identity
            replaced_result = _public_installer_child(alias, executable, contender_env)
            no_npm_while_held = not receipt.exists()
        finally:
            _, errors = holder.communicate(input="release\n", timeout=20)
        assert holder.returncode == 0, errors
    assert gap_result.get("error") == "WhatsAppBridgeBusyError", gap_result
    assert replaced_result.get("error") == "WhatsAppBridgeBusyError", replaced_result
    assert no_npm_while_held
    assert _public_installer_child(alias, executable, contender_env) == {"result": True}
    assert _public_installer_child(alias, executable, contender_env) == {"result": False}
    entries = [json.loads(line) for line in receipt.read_text().splitlines()]
    assert len(entries) == 1
    assert entries[0]["argv"] == [str(executable), "ci", "--silent"]
    stage = Path(entries[0]["cwd"])
    assert stage.parent.samefile(bridge)
    assert stage.name.startswith(".node_modules.staging-")
    assert entries[0]["env"]["HOME"] == contender_env["HOME"]
    assert entries[0]["env"]["LOCALAPPDATA"] == contender_env["LOCALAPPDATA"]
    assert "NPM_TOKEN" not in entries[0]["env"]
    assert "HERMES_HOME" not in entries[0]["env"]
    assert (bridge / "node_modules" / "installed-version").read_bytes() == b"fixture-subprocess\n"
    assert whatsapp_common.whatsapp_bridge_dependencies_fresh(bridge)
    assert _staging_leftovers(bridge) == []


@pytest.mark.macos_only
def test_public_subprocess_installer_needs_only_writable_bridge_not_parent(tmp_path):
    if os.getuid() == 0:
        pytest.skip("root bypasses directory permission checks")
    parent = tmp_path / "read-only-parent"
    bridge = parent / "bridge"
    _seed_runtime(bridge, "1.0.0")
    bridge.chmod(0o700)
    executable, receipt = _fixture_npm_executable(tmp_path)
    parent.chmod(0o500)
    try:
        with pytest.raises(PermissionError):
            (parent / "denied-sibling").mkdir()
        result = _public_installer_child(
            bridge, executable, _maintenance_child_environment(tmp_path, "profile")
        )
        assert result == {"result": True}
        assert receipt.is_file()
        assert whatsapp_common.whatsapp_bridge_dependencies_fresh(bridge)
        assert _staging_leftovers(bridge) == []
        assert not (parent / ".whatsapp-bridge-locks").exists()
    finally:
        parent.chmod(0o700)
