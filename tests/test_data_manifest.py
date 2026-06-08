"""Tests for the data manifest / lock / fetch machinery.

Pure-logic tests run anywhere (no network, no big local data). One guarded
network test fetches a small public file to exercise the real download+verify
path; it skips when offline.
"""

from __future__ import annotations

import urllib.error

import pytest

from yeastbench.data import manifest
from yeastbench.data.backends import HttpBackend
from yeastbench.data.fetch import (
    _file_state,
    artifacts_for,
    build_plans,
)
from yeastbench.data.lock import (
    LOCK_PATH,
    build_lock,
    local_files,
    locked_files,
    read_lock,
    sha256_file,
)
from yeastbench.data.manifest import (
    Artifact,
    BackendKind,
    Kind,
    Mirror,
    artifact_by_id,
)


# ──────────────────────────────────────────────────────────────
# Registry coverage — every task/model has declared data
# ──────────────────────────────────────────────────────────────


def test_every_registered_task_and_model_has_an_artifact():
    problems = manifest.validate_against_registry()
    assert problems == [], "\n".join(problems)


def test_artifact_ids_unique():
    ids = [a.id for a in manifest.ARTIFACTS]
    assert len(ids) == len(set(ids))


# ──────────────────────────────────────────────────────────────
# Selection — closes over `requires`, rejects typos, tolerates baselines
# ──────────────────────────────────────────────────────────────


def test_selection_pulls_required_shared_refs():
    arts = artifacts_for(tasks=["cuperus_utr"])
    ids = {a.id for a in arts}
    assert "cuperus_mpra_5utr" in ids
    assert "refs" in ids  # pulled in via `requires`


def test_selection_groups_chen_libraries_into_one_artifact():
    arts = artifacts_for(tasks=["chen_gfp_r1", "chen_gfp_r2", "chen_tdh3"])
    assert sum(a.id == "chen_synonymous" for a in arts) == 1


def test_selection_unknown_name_raises():
    with pytest.raises(KeyError):
        artifacts_for(tasks=["does_not_exist"])


def test_selection_tolerates_weightless_baseline():
    # cai has no downloadable artifact; selecting it alongside a real task
    # must not raise.
    arts = artifacts_for(tasks=["chen_gfp_r1"], models=["cai"])
    assert any(a.id == "chen_synonymous" for a in arts)


def test_selection_all_when_unfiltered():
    assert artifacts_for() == list(manifest.ARTIFACTS)


# ──────────────────────────────────────────────────────────────
# Lock — build / verify on a synthetic tree (no real data needed)
# ──────────────────────────────────────────────────────────────


def _fake_artifact(dest: str) -> Artifact:
    return Artifact(
        id="fake",
        kind=Kind.TASK_PROCESSED,
        dest=dest,
        mirrors=(Mirror(BackendKind.HTTP, base="https://example.invalid/"),),
        include=("**/*",),
    )


def test_lock_build_and_excludes_cache(tmp_path):
    root = tmp_path
    base = root / "blob"
    (base / "sub").mkdir(parents=True)
    (base / "a.tsv").write_text("hello")
    (base / "sub" / "b.tsv").write_text("world")
    (base / "_cache").mkdir()
    (base / "_cache" / "raw.bin").write_text("ignore me")

    art = _fake_artifact("blob")
    rels = local_files(art, root)
    assert rels == ["a.tsv", "sub/b.tsv"]  # _cache excluded, sorted

    lock = build_lock([art], root)
    assert set(lock["fake"]["files"]) == {"a.tsv", "sub/b.tsv"}
    assert lock["fake"]["files"]["a.tsv"]["sha256"] == sha256_file(base / "a.tsv")
    assert lock["fake"]["files"]["a.tsv"]["size"] == 5


def test_file_state_ok_stale_missing(tmp_path):
    f = tmp_path / "x"
    f.write_text("data")
    good = sha256_file(f)
    assert _file_state(f, good) == "ok"
    assert _file_state(f, "0" * 64) == "stale"
    assert _file_state(tmp_path / "nope", good) == "missing"


# ──────────────────────────────────────────────────────────────
# Planning — mirror choice + per-file state, no network
# ──────────────────────────────────────────────────────────────


def test_build_plans_reports_present_files_as_ok(tmp_path):
    root = tmp_path
    base = root / "blob"
    base.mkdir()
    (base / "a.tsv").write_text("hello")
    art = _fake_artifact("blob")
    lock = {"version": 1, "artifacts": build_lock([art], root)}

    plans = build_plans([art], lock, from_kind=BackendKind.HTTP, data_root=root)
    assert len(plans) == 1
    p = plans[0]
    assert p.mirror is not None and p.mirror.backend == BackendKind.HTTP
    assert [f.state for f in p.files] == ["ok"]
    assert p.to_fetch == []


def test_build_plans_marks_missing_files_for_fetch(tmp_path):
    root = tmp_path
    (root / "blob").mkdir()
    art = _fake_artifact("blob")
    # lock references a file that isn't on disk
    lock = {"version": 1, "artifacts": {"fake": {"files": {
        "ghost.tsv": {"sha256": "0" * 64, "size": 1}}}}}
    plans = build_plans([art], lock, from_kind=None, data_root=root)
    assert [f.state for f in plans[0].files] == ["missing"]
    assert len(plans[0].to_fetch) == 1


# ──────────────────────────────────────────────────────────────
# Committed lock — internal consistency (no local data dependency)
# ──────────────────────────────────────────────────────────────


def test_committed_lock_covers_shorkie_and_refs():
    lock = read_lock(LOCK_PATH)
    shorkie = {lf.relpath for lf in locked_files(lock, "shorkie")}
    assert "params.json" in shorkie
    assert "targets.txt" in shorkie
    assert sum(r.startswith("checkpoints/f") for r in shorkie) == 8

    refs = {lf.relpath for lf in locked_files(lock, "refs")}
    assert {"R64-1-1.fa", "R64-5-1.fa"} <= refs


# ──────────────────────────────────────────────────────────────
# Real network fetch (small, public) — proves download + checksum gate
# ──────────────────────────────────────────────────────────────


@pytest.mark.network
def test_http_backend_fetches_shorkie_params_matching_lock(tmp_path):
    """Fetch the (2.7 KB) Shorkie params.json from its public source and check
    it hashes to what the committed lock expects."""
    art = artifact_by_id("shorkie")
    mirror = art.mirrors[0]
    lock = read_lock(LOCK_PATH)
    want = {lf.relpath: lf.sha256 for lf in locked_files(lock, "shorkie")}

    out = tmp_path / "params.json"
    try:
        HttpBackend().fetch(mirror, mirror.remote_for("params.json"), out)
    except urllib.error.URLError as e:  # offline / CI without egress
        pytest.skip(f"network unavailable: {e}")
    assert sha256_file(out) == want["params.json"]
