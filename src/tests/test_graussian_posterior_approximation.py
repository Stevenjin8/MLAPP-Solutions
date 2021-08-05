from mlapp_models.gaussian_posterior_approximation import (
    sufficient_statistics,
    true_log_posterior,
    laplace_approx,
)
import pytest
import numpy as np
from .utils import AbstractTestCase


class TestGaussianPosteriorApproximation(AbstractTestCase):
    """Tests functions for posterior approximation."""

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.data = np.random.normal(0, 1, 100)

    def test_sufficient_statistics(self):
        """Test that we can get sufficient statistics."""
        assert (
            len(self.data),
            self.data.var(),
            self.data.mean(),
        ) == sufficient_statistics(self.data)

    def test_approximation(self):
        """Test the mode and that extreme values are close to -inf"""
        ell = np.log(self.data.var() ** 0.5)
        mu = self.data.mean()
        eps = 0.00001
        assert laplace_approx(mu, ell, self.data) > laplace_approx(
            mu + eps, ell + eps, self.data
        )
        assert laplace_approx(mu, ell, self.data) > laplace_approx(
            mu - eps, ell - eps, self.data
        )
        assert true_log_posterior(mu, ell, self.data) > true_log_posterior(
            mu + eps, ell + eps, self.data
        )
        assert true_log_posterior(mu, ell, self.data) > true_log_posterior(
            mu - eps, ell - eps, self.data
        )

    def test_extreme_values(self):
        """Test that extreme values behave as expected."""
        with pytest.warns(RuntimeWarning) as warninfo:
            assert np.exp(laplace_approx(999999, 999999, self.data)) == 0
        with pytest.warns(RuntimeWarning) as warninfo:
            assert np.exp(laplace_approx(999999, 999999, self.data)) == 0
