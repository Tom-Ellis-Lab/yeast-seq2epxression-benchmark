"""ExoShorkie adapters for the scalar / variant tasks (Wu, Hong, Chen,
MPRA-marginalized, Shalem, eQTL).

ExoShorkie is benchmarked on every task Shorkie runs, with the **identical
readout** — only the model differs. Each Shorkie adapter reduces its task
through ``self.model.forward_track_mean_perbase(x, track_idx)``; the ExoShorkie
wrapper aliases that method (returning its count-space coverage and ignoring the
track argument, since it emits a single coverage vector). So each adapter below
is a thin subclass of the corresponding Shorkie adapter that swaps in an
``ExoShorkie`` model via ``from_students`` and inherits the readout unchanged —
guaranteeing the only experimental variable between the Shorkie and ExoShorkie
rows is the model.

``track_subset`` is passed empty: ExoShorkie has no track axis, and the wrapper
alias ignores the argument. The Brooks coverage adapter lives separately in
``exoshorkie_brooks.py`` (it builds the wrapper directly rather than reusing a
Shorkie class).
"""
from __future__ import annotations

from pathlib import Path

from yeastbench.adapters.shorkie_chen_marginalized import ShorkieChenPredictor
from yeastbench.adapters.shorkie_eqtl import ShorkieVariantScorer
from yeastbench.adapters.shorkie_hong import ShorkieHongPredictor
from yeastbench.adapters.shorkie_mpra_marginalized import (
    ShorkieMPRAMarginalizedPredictor,
)
from yeastbench.adapters.shorkie_shalem import ShorkieShalemPredictor
from yeastbench.adapters.shorkie_wu import ShorkieWuPredictor
from yeastbench.models.exoshorkie import ExoShorkie

# ExoShorkie emits a single coverage vector — no track axis. The wrapper's
# forward_track_mean_perbase alias ignores its track argument, so the Shorkie
# readouts run unchanged with an empty track subset.
_NO_TRACKS: list[int] = []


def _build_model(students_dir, device: str, use_rc: bool) -> ExoShorkie:
    kw = {} if students_dir is None else {"students_dir": students_dir}
    return ExoShorkie.from_students(device=device, use_rc=use_rc, **kw)


class ExoShorkieWuPredictor(ShorkieWuPredictor):
    @classmethod
    def from_students(
        cls,
        fasta_path: str | Path,
        gtf_path: str | Path,
        cassette_fasta: str | Path | None = None,
        students_dir: str | Path | None = None,
        device: str = "cuda",
        use_rc: bool = True,
        batch_size: int = 16,
    ) -> "ExoShorkieWuPredictor":
        return cls(
            model=_build_model(students_dir, device, use_rc),
            fasta_path=fasta_path,
            gtf_path=gtf_path,
            cassette_fasta=cassette_fasta,
            track_subset=_NO_TRACKS,
            batch_size=batch_size,
        )


class ExoShorkieHongPredictor(ShorkieHongPredictor):
    @classmethod
    def from_students(
        cls,
        fasta_path: str | Path,
        cassette_fasta: str | Path | None = None,
        students_dir: str | Path | None = None,
        device: str = "cuda",
        use_rc: bool = True,
        batch_size: int = 16,
        targets_path: str | Path | None = None,
    ) -> "ExoShorkieHongPredictor":
        return cls(
            model=_build_model(students_dir, device, use_rc),
            fasta_path=fasta_path,
            cassette_fasta=cassette_fasta,
            track_subset=_NO_TRACKS,
            batch_size=batch_size,
            targets_path=targets_path,
        )


class ExoShorkieChenPredictor(ShorkieChenPredictor):
    @classmethod
    def from_students(
        cls,
        fasta_path: str | Path,
        library: str,
        hosts_path: str | Path,
        data_dir: str | Path,
        students_dir: str | Path | None = None,
        device: str = "cuda",
        use_rc: bool = True,
        batch_size: int = 4,
    ) -> "ExoShorkieChenPredictor":
        return cls(
            model=_build_model(students_dir, device, use_rc),
            fasta_path=fasta_path,
            library=library,
            hosts_path=hosts_path,
            data_dir=data_dir,
            track_subset=_NO_TRACKS,
            batch_size=batch_size,
        )


class ExoShorkieMPRAMarginalizedPredictor(ShorkieMPRAMarginalizedPredictor):
    @classmethod
    def from_students(
        cls,
        fasta_path: str | Path,
        gtf_path: str | Path,
        students_dir: str | Path | None = None,
        device: str = "cuda",
        use_rc: bool = True,
        batch_size: int = 32,
        n_sample: int | None = None,
        seed: int = 42,
    ) -> "ExoShorkieMPRAMarginalizedPredictor":
        return cls(
            model=_build_model(students_dir, device, use_rc),
            fasta_path=fasta_path,
            gtf_path=gtf_path,
            track_subset=_NO_TRACKS,
            batch_size=batch_size,
            n_sample=n_sample,
            seed=seed,
        )


class ExoShorkieShalemPredictor(ShorkieShalemPredictor):
    @classmethod
    def from_students(
        cls,
        fasta_path: str | Path,
        gtf_path: str | Path,
        host_genes_json: str | Path | None = None,
        students_dir: str | Path | None = None,
        device: str = "cuda",
        use_rc: bool = True,
        batch_size: int = 32,
        n_sample: int | None = None,
        seed: int = 42,
    ) -> "ExoShorkieShalemPredictor":
        return cls(
            model=_build_model(students_dir, device, use_rc),
            fasta_path=fasta_path,
            gtf_path=gtf_path,
            host_genes_json=host_genes_json,
            track_subset=_NO_TRACKS,
            batch_size=batch_size,
            n_sample=n_sample,
            seed=seed,
        )


class ExoShorkieVariantScorer(ShorkieVariantScorer):
    @classmethod
    def from_students(
        cls,
        fasta_path: str | Path,
        gtf_path: str | Path,
        students_dir: str | Path | None = None,
        device: str = "cuda",
        use_rc: bool = True,
        batch_size: int = 8,
    ) -> "ExoShorkieVariantScorer":
        return cls(
            model=_build_model(students_dir, device, use_rc),
            fasta_path=fasta_path,
            gtf_path=gtf_path,
            track_subset=_NO_TRACKS,
            batch_size=batch_size,
        )
