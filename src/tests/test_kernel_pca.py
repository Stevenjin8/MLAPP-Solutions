"""Tests for kernel PCA."""
import numpy as np
import pytest
from .utils import AbstractTestCase
from mlapp_models.utils import pairwise_rbf_kernel
from mlapp_models.kernel_pca import KernelPCA


class TestKernelPCA(AbstractTestCase):
    """Tests for kernel PCA."""

    @classmethod
    def setup_class(cls):
        super().setup_class()
        # Probably not that clean, but this data works really well.
        cls.n = 200
        q = 4 * np.random.rand(cls.n // 2, 1) * 2 * np.pi
        X_1 = np.concatenate((np.sin(q), np.cos(q)), axis=1) * (
            np.random.normal(scale=0.7, size=(cls.n // 2, 1))
        )

        q = 4 * np.random.rand(cls.n // 2, 1) * 2 * np.pi
        X_2 = np.concatenate((np.sin(q), np.cos(q)), axis=1) * (
            6 + np.random.normal(scale=0.7, size=(cls.n // 2, 1))
        )
        cls.X = np.append(X_1, X_2, axis=0)

    def test_fit(self):
        """Test that we can discrimate the classes with one dimension."""
        model = KernelPCA(self.X, pairwise_rbf_kernel)
        z = model.predict(self.X, n_components=1)
        assert z.shape == (self.n, 1)
        median = np.median(z)
        results = np.logical_not(
            np.logical_xor(z[: self.n // 2] < median, z[self.n // 2 :] > median)
        )
        assert results.all()
