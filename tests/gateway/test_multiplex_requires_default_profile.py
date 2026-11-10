"""Explicit named hosts are admitted; cloned profiles do not inherit that ownership."""


def _root(tmp_path):
    root = tmp_path / '.hermes'
    root.mkdir()
    (root / 'config.yaml').write_text('model: {default: m}\n')
    (root / 'profiles' / 'foo').mkdir(parents=True)
    (root / 'profiles' / 'foo' / 'config.yaml').write_text('gateway:\n  multiplex_profiles: true\n')
    return root


def test_explicit_named_profile_launch_may_multiplex(tmp_path, monkeypatch):
    from gateway import run_bootstrap

    root = _root(tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(root / 'profiles' / 'foo'))
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None
    written = []
    monkeypatch.setattr('gateway.run._write_runtime_status_quiet', lambda **kw: written.append(kw))
    assert run_bootstrap._launch_home_may_multiplex() is True
    assert written == []


def test_default_profile_launch_may_multiplex(tmp_path, monkeypatch):
    from gateway import run_bootstrap

    root = _root(tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(root))
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None
    assert run_bootstrap._launch_home_may_multiplex() is True


def test_cloned_profile_does_not_inherit_the_multiplex_flag(tmp_path, monkeypatch):
    import hermes_yaml as yaml
    from hermes_cli.profiles import _strip_multiplex_flag

    nested = tmp_path / 'nested.yaml'
    nested.write_text('model: {default: m}\ngateway:\n  multiplex_profiles: true\n  service_install_choice: skip\n')
    _strip_multiplex_flag(nested)
    assert yaml.safe_load(nested.read_text()) == {'model': {'default': 'm'}, 'gateway': {'service_install_choice': 'skip'}}
    top = tmp_path / 'top.yaml'
    top.write_text('multiplex_profiles: true\nmodel: {default: m}\n')
    _strip_multiplex_flag(top)
    assert yaml.safe_load(top.read_text()) == {'model': {'default': 'm'}}
    untouched = tmp_path / 'plain.yaml'
    untouched.write_text('# keep my comment\nmodel: {default: m}\n')
    _strip_multiplex_flag(untouched)
    assert untouched.read_text() == '# keep my comment\nmodel: {default: m}\n'
