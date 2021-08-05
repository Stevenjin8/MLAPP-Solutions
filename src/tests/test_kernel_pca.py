"""Tests for kernel PCA."""
import numpy as np
import pytest
from mlapp_models.kernel_pca import KernelPCA
from mlapp_models.utils import pairwise_rbf_kernel
from sklearn import datasets

from .utils import AbstractTestCase


class TestKernelPCA(AbstractTestCase):
    """Tests for kernel PCA."""

    @classmethod
    def setup_class(cls):
        super().setup_class()
        # Probably not that clean, but this data works really well.
        cls.n = 800
        cls.X, cls.y = datasets.make_circles(
            n_samples=cls.n, factor=0, noise=0.1, shuffle=False
        )

    def test_fit(self):
        """Test that we can discrimate the classes with one dimension."""
        model = KernelPCA(self.X, pairwise_rbf_kernel)
        z = model.predict(self.X, n_components=1)
        assert z.shape == (self.n, 1)
        median = np.median(z)
        results = np.logical_not(
            np.logical_xor(z[self.y == 0] < median, z[self.y == 1] > median)
        )
        assert results.all()
