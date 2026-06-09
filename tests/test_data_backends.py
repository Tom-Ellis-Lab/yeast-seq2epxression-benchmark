"""GCS backend billing-project resolution (requester-pays bucket).

Pure-logic tests — they only inspect the argv the backend would run and the
resolution precedence, never shelling out to gcloud.
"""

from __future__ import annotations

import pytest

from yeastbench.data.backends import _BILLING_ENV, BackendError, GcsBackend


def _gcloud_backend(billing_project: str | None = None) -> GcsBackend:
    b = GcsBackend(billing_project)
    b._cli = "/usr/bin/gcloud"  # pin the CLI so the test doesn't depend on PATH
    return b


def _gsutil_backend(billing_project: str | None = None) -> GcsBackend:
    b = GcsBackend(billing_project)
    b._cli = "/usr/bin/gsutil"
    return b


def test_gcloud_cmd_carries_billing_flag():
    cmd = _gcloud_backend("clex-415415")._build_cmd("gs://b/x", "/tmp/x")
    assert "--billing-project=clex-415415" in cmd
    assert cmd[:3] == ["/usr/bin/gcloud", "storage", "cp"]
    assert cmd[-2:] == ["gs://b/x", "/tmp/x"]


def test_gsutil_puts_user_project_before_subcommand():
    cmd = _gsutil_backend("clex-415415")._build_cmd("gs://b/x", "/tmp/x")
    # gsutil's -u is a *global* flag: it must precede `cp`, not follow it.
    assert cmd == ["/usr/bin/gsutil", "-u", "clex-415415", "cp", "gs://b/x", "/tmp/x"]


def test_flag_wins_over_env(monkeypatch):
    monkeypatch.setenv(_BILLING_ENV, "from-env")
    cmd = _gcloud_backend("from-flag")._build_cmd("gs://b/x", "/tmp/x")
    assert "--billing-project=from-flag" in cmd


def test_env_used_when_no_flag(monkeypatch):
    monkeypatch.setenv(_BILLING_ENV, "from-env")
    cmd = _gcloud_backend(None)._build_cmd("gs://b/x", "/tmp/x")
    assert "--billing-project=from-env" in cmd


def test_no_billing_project_raises(monkeypatch):
    monkeypatch.delenv(_BILLING_ENV, raising=False)
    with pytest.raises(BackendError, match="requester pays"):
        _gcloud_backend(None)._build_cmd("gs://b/x", "/tmp/x")


def test_no_cli_raises(monkeypatch):
    monkeypatch.delenv(_BILLING_ENV, raising=False)
    b = GcsBackend("clex-415415")
    b._cli = None
    with pytest.raises(BackendError, match="gcloud.*gsutil"):
        b._build_cmd("gs://b/x", "/tmp/x")
