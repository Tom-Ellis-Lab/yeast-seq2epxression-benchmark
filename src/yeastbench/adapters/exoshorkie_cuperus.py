"""ExoShorkie adapter for the Cuperus 5'-UTR reporter task.

Same readout as ``ShorkieCuperusPredictor`` — assemble the CYC1pr + UTR + HIS3 +
CYC1term construct in the HIS3 background, forward, and sum the predicted
coverage over the HIS3-ORF base positions — only the model differs. The Shorkie
adapter reduces through ``self.model.forward_track_mean_perbase(x, track_idx)``,
which the ExoShorkie wrapper aliases (count-space coverage, track arg ignored),
so this is a thin subclass that injects an ExoShorkie model via ``from_students``
and inherits the readout unchanged (track_subset passed empty — ExoShorkie has a
single coverage output).
"""
from __future__ import annotations

from pathlib import Path

from yeastbench.adapters.shorkie_cuperus import ShorkieCuperusPredictor
from yeastbench.models.exoshorkie import ExoShorkie

_NO_TRACKS: list[int] = []


class ExoShorkieCuperusPredictor(ShorkieCuperusPredictor):
    @classmethod
    def from_students(
        cls,
        fasta_path: str | Path,
        construct_json: str | Path | None = None,
        students_dir: str | Path | None = None,
        device: str = "cuda",
        use_rc: bool = True,
        batch_size: int = 16,
    ) -> "ExoShorkieCuperusPredictor":
        kw = {} if students_dir is None else {"students_dir": students_dir}
        model = ExoShorkie.from_students(device=device, use_rc=use_rc, **kw)
        return cls(
            model=model,
            fasta_path=fasta_path,
            construct_json=construct_json,
            track_subset=_NO_TRACKS,
            batch_size=batch_size,
        )
