"""Shorkie model package.

`ShorkieModule` is the pytorch `nn.Module` (lives in `nn.py`). The
benchmark-side wrapper `Shorkie` (in `wrapper.py`, added in a later
step of this refactor) owns device handling, RC averaging, the
8-fold ensemble loop, and the batched forward path."""
from yeastbench.models.shorkie.nn import ShorkieModule
from yeastbench.models.shorkie.wrapper import Shorkie

__all__ = ["Shorkie", "ShorkieModule"]
