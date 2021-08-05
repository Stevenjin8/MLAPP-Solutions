"""Test Bivariate Gibbs sampler."""
import numpy as np
import pytest
from mlapp_models.normal_bivariate_sampler import NormalBivariateGibbs
from numpy import testing

from .utils import AbstractTestCase


class TestNormalBivariateGibbs(AbstractTestCase):
    """Tests for the Gibbs sampler."""

    def test_sample(self):
        """The empirical distribution should be similar to the real one."""
        cov = np.array([[2, 1], [1, 2]])
        mean = np.array([-1, 1])
        model = NormalBivariateGibbs(mean=mean, cov=cov)
        num_samples = 100000
        samples = model.sample(num_samples=num_samples)

        sample_mean = samples.mean(axis=0)
        sample_cov = samples - sample_mean
        sample_cov = sample_cov.T @ sample_cov / num_samples
        testing.assert_allclose(sample_mean, mean, atol=0.01)
        testing.assert_allclose(sample_cov, cov, atol=0.01)

    def test_invalid_cov(self):
        """Test that model behaves as expected when inputting a non positive definite covariance."""
        mean = np.array([0, 0])
        cov = -np.eye(2)
        model = NormalBivariateGibbs(mean, cov)
        with pytest.warns(RuntimeWarning) as warninfo:
            samples = model.sample(10)
        warning_message = "invalid value encountered in double_scalars"
        assert all(w.message.args == (warning_message,) for w in warninfo.list)
        for x in np.nditer(samples):
            assert np.isnan(x)

        cov = np.ones((2, 2))
        model = NormalBivariateGibbs(mean, cov)
        samples = model.sample(10)
        # Since the correlation is one, and we start sampling at the mode/mean, it
        # should get stuck there.
        assert abs(samples).sum() == 0
