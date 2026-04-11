from __future__ import annotations

import click

from headroom.cli.wrap import _ensure_proxy, _find_persistent_manifest, _recover_persistent_proxy


class _Manifest:
    profile = "default"
    preset = "persistent-service"
    supervisor_kind = "service"
    health_url = "http://127.0.0.1:8787/readyz"


def test_ensure_proxy_recovers_matching_persistent_deployment(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr("headroom.cli.wrap._check_proxy", lambda port: False)
    monkeypatch.setattr("headroom.cli.wrap._find_persistent_manifest", lambda port: _Manifest())
    monkeypatch.setattr("headroom.install.health.probe_ready", lambda url: False)
    monkeypatch.setattr(
        "headroom.install.supervisors.start_supervisor",
        lambda manifest: calls.append(f"start:{manifest.profile}"),
    )
    monkeypatch.setattr(
        "headroom.install.runtime.wait_ready", lambda manifest, timeout_seconds=45: True
    )
    monkeypatch.setattr(
        "headroom.cli.wrap._start_proxy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("ephemeral proxy should not start")
        ),
    )

    result = _ensure_proxy(8787, False)

    assert result is None
    assert calls == ["start:default"]


def test_ensure_proxy_recovers_persistent_deployment_when_socket_is_bound(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr("headroom.cli.wrap._check_proxy", lambda port: True)
    monkeypatch.setattr("headroom.cli.wrap._find_persistent_manifest", lambda port: _Manifest())
    monkeypatch.setattr("headroom.install.health.probe_ready", lambda url: False)
    monkeypatch.setattr(
        "headroom.install.supervisors.start_supervisor",
        lambda manifest: calls.append(f"start:{manifest.profile}"),
    )
    monkeypatch.setattr(
        "headroom.install.runtime.wait_ready", lambda manifest, timeout_seconds=45: True
    )

    result = _ensure_proxy(8787, False)

    assert result is None
    assert calls == ["start:default"]


def test_ensure_proxy_rejects_unhealthy_persistent_deployment(monkeypatch) -> None:
    monkeypatch.setattr("headroom.cli.wrap._check_proxy", lambda port: True)
    monkeypatch.setattr("headroom.cli.wrap._find_persistent_manifest", lambda port: _Manifest())
    monkeypatch.setattr("headroom.install.health.probe_ready", lambda url: False)
    monkeypatch.setattr("headroom.cli.wrap._recover_persistent_proxy", lambda port: False)

    try:
        _ensure_proxy(8787, False)
    except click.ClickException as exc:
        assert "is not healthy" in str(exc)
    else:
        raise AssertionError("expected unhealthy persistent deployment to raise")


def test_find_persistent_manifest_prefers_default_profile(monkeypatch) -> None:
    class DefaultManifest:
        profile = "default"
        port = 8787

    class OtherManifest:
        profile = "custom"
        port = 8787

    monkeypatch.setattr(
        "headroom.install.state.list_manifests",
        lambda: [OtherManifest(), DefaultManifest()],
    )

    manifest = _find_persistent_manifest(8787)

    assert manifest.profile == "default"


def test_recover_persistent_proxy_reuses_healthy_deployment(monkeypatch) -> None:
    monkeypatch.setattr("headroom.cli.wrap._find_persistent_manifest", lambda port: _Manifest())
    monkeypatch.setattr("headroom.install.health.probe_ready", lambda url: True)

    assert _recover_persistent_proxy(8787) is True


def test_recover_persistent_proxy_warns_for_task_deployment(monkeypatch) -> None:
    class TaskManifest(_Manifest):
        supervisor_kind = "task"

    monkeypatch.setattr("headroom.cli.wrap._find_persistent_manifest", lambda port: TaskManifest())
    monkeypatch.setattr("headroom.install.health.probe_ready", lambda url: False)

    assert _recover_persistent_proxy(8787) is False
