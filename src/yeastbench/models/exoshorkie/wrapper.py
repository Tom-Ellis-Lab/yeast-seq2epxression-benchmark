"""Benchmark-side wrapper around the 6-student ExoShorkie ensemble.

Each student is a ``ShorkieModule`` built from the ExoShorkie config (Shorkie
trunk + linear ``Dense(1)`` per-bin head), distilled from one published
per-genome 40-model ensemble. The students predict in the authors' log-z space;
this wrapper inverts to count space and combines the 6 students into one
coverage prediction.

Order of operations (mirrors the Yorzoi wrapper, whose nonlinear Borzoi inverse
must precede every average):

1. Per student *g*, denorm to count space ``c_g = clamp(expm1(z*sigma_g + mu_g), 0)``.
   The inverse is nonlinear and each student has its own ``(mu_g, sigma_g)``, so
   denorm happens **before** any mean — averaging in z-space would be ill-defined
   (no single ``(mu, sigma)`` inverts a cross-genome z-mean).
2. RC averaging, also in count space: a second forward on the reverse complement,
   denorm, flip the bin axis back to forward orientation, average with the forward
   pass (the authors' ``predict.py`` reverses RC bins with ``[::-1]``; we promote
   the average to count space because the inverse is nonlinear). NB ExoShorkie's
   single output is strand-matched-to-input, so the RC pass estimates the opposite
   strand — RC averaging yields a strand-symmetric (total-ish) coverage. Set
   ``use_rc=False`` for forward-strand-only.
3. Mean across the 6 students.

Per-base unbinning divides each 16 bp bin total by 16 and repeats — matching the
Shorkie/Yorzoi ``_unbin_per_base`` so ExoShorkie's per-base scale, CDS sums, and
``log2(sum+1)`` readouts are directly comparable to the other models. (This
diverges from the authors' bare ``np.repeat``, which they use only for ISM ratios
where the 16× cancels; the bin-level denorm is identical either way.)

Adapters take an ``ExoShorkie`` instance and call ``forward_perbase(x)`` /
``forward_count_bins(x)``; ``forward_track_mean_perbase(x, track_subset=None)`` is
a drop-in alias so the Shorkie coverage scaffolds (Brooks, the marginalized
engine) work unchanged — ExoShorkie emits a single coverage vector, so the
``track_subset`` argument is accepted and ignored.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Sequence

from yeastbench.adapters._exoshorkie_constants import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    DEFAULT_STUDENTS_DIR,
    EXOSHORKIE_GENOMES,
    EXOSHORKIE_LOGZ_STATS,
    OUTPUT_BINS,
    SEQ_LEN,
)

if TYPE_CHECKING:
    import torch

    from yeastbench.models.shorkie.nn import ShorkieModule


def _unbin_per_base(binned: "torch.Tensor", bin_width: int) -> "torch.Tensor":
    """Spread each bin total over its ``bin_width`` bases (÷ width, repeat).
    Summing a base interval recovers the bin total — same convention as the
    Shorkie/Yorzoi wrappers, so per-base scales are comparable across models."""
    return binned.repeat_interleave(bin_width, dim=-1) / float(bin_width)


class ExoShorkie:
    """6-student ExoShorkie ensemble wrapper (count-space coverage out)."""

    SEQ_LEN: ClassVar[int] = SEQ_LEN
    OUTPUT_BINS: ClassVar[int] = OUTPUT_BINS
    BIN_WIDTH: ClassVar[int] = BIN_WIDTH
    CROP_BP_EACH_SIDE: ClassVar[int] = CROP_BP_EACH_SIDE

    def __init__(
        self,
        students: list["ShorkieModule"],
        stats: list[tuple[float, float]],
        device: "str | torch.device" = "cuda",
        use_rc: bool = True,
    ) -> None:
        import torch as _torch

        if not students:
            raise ValueError("Must provide at least one ExoShorkie student")
        if len(students) != len(stats):
            raise ValueError(
                f"students ({len(students)}) and stats ({len(stats)}) length mismatch"
            )
        self.students = students
        self.device = _torch.device(device)
        self.use_rc = use_rc
        # (mu, sigma) per student as device tensors, pre-broadcast for the
        # elementwise denorm.
        self._mu = [
            _torch.tensor(float(mu), device=self.device, dtype=_torch.float32)
            for mu, _ in stats
        ]
        self._sigma = [
            _torch.tensor(float(sigma), device=self.device, dtype=_torch.float32)
            for _, sigma in stats
        ]
        for m in self.students:
            m.to(self.device).eval()

    @classmethod
    def from_students(
        cls,
        students_dir: str | Path = DEFAULT_STUDENTS_DIR,
        genomes: Sequence[str] = tuple(EXOSHORKIE_GENOMES),
        stats: dict[str, tuple[float, float]] | None = None,
        device: "str | torch.device" = "cuda",
        use_rc: bool = True,
    ) -> "ExoShorkie":
        """Load ``<students_dir>/<genome>.pt`` for each genome and pair it with
        its ``(mu, sigma)``. Each ``.pt`` carries its own ``exo_config``; the
        student is rebuilt and ``load_state_dict``-ed (the weights are PyTorch,
        not a TF ``.h5``, so ``from_tf_checkpoint`` is not used)."""
        import torch as _torch

        from yeastbench.models.shorkie.nn import ShorkieModule

        stats = stats or EXOSHORKIE_LOGZ_STATS
        students_dir = Path(students_dir)
        loaded: list[ShorkieModule] = []
        paired: list[tuple[float, float]] = []
        for genome in genomes:
            if genome not in stats:
                raise KeyError(f"no (mu, sigma) for genome {genome!r}")
            ckpt_path = students_dir / f"{genome}.pt"
            ckpt = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
            student = ShorkieModule(ckpt["exo_config"])
            student.load_state_dict(ckpt["state_dict"])
            if student._species_channel != 119 or not student._encode_n_channel:
                raise ValueError(
                    f"{ckpt_path} is not an ExoShorkie student "
                    f"(species_channel={student._species_channel}, "
                    f"encode_n={student._encode_n_channel})"
                )
            loaded.append(student)
            paired.append(tuple(stats[genome]))
        return cls(loaded, paired, device=device, use_rc=use_rc)

    def _denorm(self, z: "torch.Tensor", i: int) -> "torch.Tensor":
        """Count-space coverage for student ``i``: ``clamp(expm1(z*sigma+mu), 0)``."""
        import torch as _torch

        return _torch.clamp(_torch.expm1(z * self._sigma[i] + self._mu[i]), min=0.0)

    def forward_count_bins(self, x: "torch.Tensor") -> "torch.Tensor":
        """Ensemble count-space coverage, binned: ``(B, OUTPUT_BINS)``.

        Denorm-then-mean across students, RC averaging in count space."""
        import torch as _torch

        B = x.shape[0]
        acc = _torch.zeros(B, OUTPUT_BINS, device=self.device, dtype=_torch.float32)
        x_rc = x.flip(dims=[1, 2]) if self.use_rc else None
        for i, student in enumerate(self.students):
            c = self._denorm(student(x).float(), i)
            if self.use_rc:
                c_rc = self._denorm(student(x_rc).float(), i).flip(dims=[-1])
                c = 0.5 * (c + c_rc)
            acc.add_(c)
        acc.div_(len(self.students))
        return acc

    def forward_perbase(self, x: "torch.Tensor") -> "torch.Tensor":
        """Per-base count-space coverage: ``(B, OUTPUT_BINS * BIN_WIDTH)`` =
        ``(B, 14336)``. ``forward_count_bins`` followed by the ÷16 unbin."""
        return _unbin_per_base(self.forward_count_bins(x), BIN_WIDTH)

    def forward_track_mean_perbase(
        self, x: "torch.Tensor", track_subset: "torch.Tensor | None" = None
    ) -> "torch.Tensor":
        """Drop-in alias for the Shorkie coverage scaffolds. ExoShorkie emits a
        single coverage vector, so ``track_subset`` is accepted and ignored."""
        return self.forward_perbase(x)
