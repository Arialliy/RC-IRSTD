from __future__ import annotations

"""Test-suite resource policy.

Small convolutional smoke tests can become dramatically slower when a CI host
exposes a very large CPU thread pool.  Pinning PyTorch to one thread makes the
software validation deterministic and does not change training defaults.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch


torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # PyTorch only permits setting this before parallel work starts.  A plugin
    # may already have initialized it, in which case the intra-op limit above
    # is still sufficient for the tests.
    pass
