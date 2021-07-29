"""Utilities for tests."""
import numpy as np


class AbstractTestCase:
    """Parent test case for all tests."""

    random_seed: int = 42
    float_dtype: type = np.float64
    finfo: np.finfo

    @classmethod
    def setup_class(cls):
        """Setup state for tests."""
        np.random.seed(seed=cls.random_seed)
        cls.finfo = np.finfo(cls.float_dtype)
