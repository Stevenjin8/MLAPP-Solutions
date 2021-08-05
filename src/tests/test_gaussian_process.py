"""Tests for Gaussian process models."""
from math import isclose

import numpy as np
import pytest
from mlapp_models.gaussian_process import (
    GaussianProcessClassifier,
    GaussianProcessRegressor,
)
from numpy import testing

from .utils import AbstractTestCase


class TestGaussianProcessRegressor(AbstractTestCase):
    """Test gaussian process regressor."""

    def test_no_noise(self):
        """Test that the model behaves as expected when there is no noise."""
        X = np.array([[0], [1], [2]])
        y = np.array([1, 2, 3])
        model = GaussianProcessRegressor(var_eps=0, var0=1, ell=0.5)
        model.fit(X, y)
        preds_mean, preds_covar = model.predict(X)
        preds_var = np.diag(preds_covar)
        testing.assert_allclose(preds_mean, y)
        testing.assert_allclose(preds_var, np.zeros_like(preds_var), atol=1e-8)

        preds_mean, preds_var = model.predict(np.array([[0.5]]))
        assert 1 < preds_mean[0] < 2
        assert preds_var[0, 0] > 1e-2

        # fitting different y with same x should give error
        X = np.array([[0, 1], [0, 1], [2, 0]])
        y = np.array([1, 2, 3])
        with pytest.raises(np.linalg.LinAlgError) as excinfo:
            model.fit(X, y)
        assert "singular" in excinfo.value.args[0].lower()

    def test_some_noise(self):
        """Test that adding some noise to the model makes it behave as expected."""
        X = np.array([[0], [1], [2]])
        y = np.array([1, 2, 3])
        var_eps = 0.001
        var0 = 1
        model = GaussianProcessRegressor(var_eps=var_eps, var0=var0, ell=0.5)
        model.fit(X, y)
        preds_mean, preds_covar = model.predict(X)
        preds_var = np.diag(preds_covar)
        testing.assert_allclose(preds_mean, y, atol=1e-2)
        testing.assert_allclose(preds_var, np.ones_like(preds_var) * 0.002, atol=1e-2)

        # Stack points
        X = np.array([[0], [0]])
        y = np.array([1, 2])
        model.fit(X, y)
        preds_mean, preds_var = model.predict(np.array([[0]]))
        assert isclose(preds_mean[0], 1.5, abs_tol=1e-2)
        assert 0 < preds_var[0, 0] < 0.002

        # predict some far off point
        preds_mean, preds_var = model.predict(np.array([[9999999]]))
        assert isclose(preds_mean[0], 0)
        assert isclose(preds_var[0, 0], var0, abs_tol=1e-2)


class TestGaussianProcessClassifier(AbstractTestCase):
    """Test for GP classifier."""

    def test_fit(self):
        """Test that the model can fit to some data."""
        model = GaussianProcessClassifier(var0=0.001, ell=4)
        self.assert_classifier_fit_blobs(
            model,
            blob_kwargs={"centers": np.array([[5, 0], [0, -5]])},
            zero_one_labels=False,
            min_accuracy=1,
        )
        model = GaussianProcessClassifier(var0=0.001, ell=0.1)
        self.assert_classifier_fit_moons(model, zero_one_labels=False, min_accuracy=1)
        model = GaussianProcessClassifier(var0=0.001, ell=0.1)
        self.assert_classifier_fit_circles(model, zero_one_labels=False, min_accuracy=1)

    def test_easy_data(self):
        """The model should perform as expected on easy data."""
        model = GaussianProcessClassifier(var0=0.01, ell=0.5)

        n = 10
        X = np.array([[1, 1], [-1, -1]])
        y = np.array([1, -1])
        model.fit(X, y)

        X = np.ones((n, 2)) * np.linspace(-1, 1, n).reshape(-1, 1)
        means = model.predict_mean(X)
        assert list(means) == sorted(means)
        assert (means[: n // 2] < 0).all()
        assert (means[n // 2 :] > 0).all()
        variances = np.diag(model.predict_var(X))
        assert list(variances[: n // 2]) == sorted(variances[: n // 2])
        assert list(variances[n // 2 :]) == sorted(variances[n // 2 :], reverse=True)
