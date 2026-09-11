from __future__ import annotations

import random

import numpy as np
import torch

from fedllm_heterogeneity.training import seed_everything


def test_seed_everything_repeats_python_numpy_and_torch_rngs():
    seed_everything(123)
    first = (random.random(), np.random.random(), torch.rand(3))
    seed_everything(123)
    second = (random.random(), np.random.random(), torch.rand(3))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
