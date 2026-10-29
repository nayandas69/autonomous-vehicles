"""Scoped RNG and backend settings for isolated CPU research experiments."""

import random
from contextlib import contextmanager

import numpy as np
import torch


@contextmanager
def reproducible(seed: int = 42, threads: int = 1):
    """Restore global RNG and thread settings after a CPU experiment.

    This context is process-global and must not be entered concurrently from
    multiple threads. It deliberately does not touch CUDA state.
    """
    if seed < 0 or threads < 1:
        raise ValueError("Seed must be nonnegative and threads positive")
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    original_threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        torch.set_num_threads(original_threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def seed_for(base: int, namespace: str, index: int = 0) -> int:
    """Stable independent seeds without Python's randomized process hash."""
    import hashlib

    if base < 0 or index < 0 or not namespace:
        raise ValueError("Need nonnegative seeds/indices and nonempty namespace")
    digest = hashlib.sha256(f"{base}:{namespace}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)
