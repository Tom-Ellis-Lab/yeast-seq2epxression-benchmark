"""ExoShorkie model package.

ExoShorkie reuses the Shorkie ``nn.Module`` (``ShorkieModule`` with the linear
``Dense(1)`` head; see ``yeastbench.models.shorkie.nn``). The benchmark-side
wrapper ``ExoShorkie`` (in ``wrapper.py``) owns the 6-student ensemble, the
count-space log-z inverse, RC averaging, and the batched forward path."""
from yeastbench.models.exoshorkie.wrapper import ExoShorkie

__all__ = ["ExoShorkie"]
