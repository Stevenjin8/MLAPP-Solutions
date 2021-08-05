import numpy as np
import pytest
from mlapp_models.dirichlet_gaussian_mixture import DirichletGaussianMixtureSampler
from numpy import testing
from scipy import stats

from .utils import AbstractTestCase


class TestDirichletGaussianMixtureSample(AbstractTestCase):
    """Tests for sampling the posterior of a Gaussian mixture model."""

    def test_easy_data(self):
        """Test that sampling the posterior for easy data works as expected."""
        n = 50
        X = stats.multivariate_normal.rvs(mean=np.array([-5, -5]), size=n // 2)
        X = np.append(
            X, stats.multivariate_normal.rvs(mean=np.array([5, 5]), size=n // 2), axis=0
        )
        model = DirichletGaussianMixtureSampler(
            alpha=1,
            prior_cov=np.eye(2),
            prior_mean=np.zeros(2),
            obs_cov=np.eye(2),
            data=X,
        )

        model.burn_in()

        samples = [model.sample() for _ in range(100)]
        num_correct = sum(
            (s[0] == s[: n // 2]).all() and (s[-1] == s[n // 2 :]).all()
            for s in samples
        )
        assert num_correct / len(samples) > 0.95

    def test_num_clusters(self):
        """The number of different clusters generated should depend on obs_cov and alpha."""
        n = 100
        n_samples = 5
        d = 5
        X = stats.multivariate_normal.rvs(mean=np.zeros(d), size=n)
        model1 = DirichletGaussianMixtureSampler(
            alpha=3,
            prior_cov=np.eye(d),
            prior_mean=np.zeros(d),
            obs_cov=np.eye(d) * 0.5,
            data=X,
        )
        samples1 = [model1.sample() for _ in range(n_samples)]
        model2 = DirichletGaussianMixtureSampler(
            alpha=10,
            prior_cov=np.eye(d),
            prior_mean=np.zeros(d),
            obs_cov=np.eye(d) * 0.5,
            data=X,
        )
        samples2 = [model2.sample() for _ in range(n_samples)]

        # not all samples should be the same
        assert not all(np.array_equal(samples2[0], s) for s in samples2)
        assert sum(s.shape[1] for s in samples2) > sum(s.shape[1] for s in samples1)

        model1 = DirichletGaussianMixtureSampler(
            alpha=3,
            prior_cov=np.eye(d),
            prior_mean=np.zeros(d),
            obs_cov=np.eye(d) * 0.5,
            data=X,
        )
        samples1 = [model1.sample() for _ in range(n_samples)]
        model2 = DirichletGaussianMixtureSampler(
            alpha=3,
            prior_cov=np.eye(d),
            prior_mean=np.zeros(d),
            obs_cov=np.eye(d) * 0.1,
            data=X,
        )

        # not all samples should be the same
        samples2 = [model2.sample() for _ in range(n_samples)]
        assert not all(np.array_equal(samples2[0], s) for s in samples2)
        assert sum(s.shape[1] for s in samples2) > sum(s.shape[1] for s in samples1)

    def test_setters_and_getters(self):
        n = 5
        X = np.ones(n)
        model = DirichletGaussianMixtureSampler(
            alpha=1,
            prior_cov=np.eye(2) * 2,
            prior_mean=np.zeros(2),
            obs_cov=np.eye(2) * 2,
            data=X,
        )

        assert model.n == n
        testing.assert_array_equal(model.prior_cov, np.eye(2) * 2)
        testing.assert_array_equal(model.prior_cov_inv, np.eye(2) * 0.5)
        testing.assert_array_equal(model.obs_cov, np.eye(2) * 2)
        testing.assert_array_equal(model.obs_cov_inv, np.eye(2) * 0.5)
