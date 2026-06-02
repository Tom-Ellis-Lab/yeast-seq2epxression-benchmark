"""Generate the fixed 16,384 bp test window used by the ExoShorkie
characterization tests. Run once; the resulting .npy is committed so the
PyTorch test and the (separate-env) TF-reference generator load the
identical sequence.

    uv run python tests/data/exoshorkie/_make_window.py
"""
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
SEQ_LEN = 16384

# Deterministic pseudo-random ACGT sequence. The model mapping is what's
# under test, not biology, so a fixed random window is fine.
rng = np.random.RandomState(0)
seq = rng.randint(0, 4, size=SEQ_LEN).astype(np.int8)  # 0=A,1=C,2=G,3=T
np.save(HERE / "test_window_16384.npy", seq)
print(f"wrote {HERE / 'test_window_16384.npy'}  ({seq.shape}, dtype={seq.dtype})")
