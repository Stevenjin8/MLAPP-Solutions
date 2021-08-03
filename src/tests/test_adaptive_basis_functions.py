"""Tests for the adaptive_basis_function module."""
import numpy as np
import pytest
from mlapp_models.adaptive_basis_functions import AdaBoost, MLPClassifier, Stump
from numpy import testing
from scipy import stats
from sklearn.datasets import make_moons
from .utils import AbstractTestCase


class TestStump(AbstractTestCase):
    """Test that the decision stump is working correctly."""

    def test_fit(self):
        """Test that we can fit easy data."""
        model = Stump()
        X = np.array([[1, 2], [2, 3], [3, 1], [-1, 2], [-2, 1], [-10, 100]])
        y = np.array([1, 1, 1, -1, -1, -1])
        weights = np.ones_like(y)
        model.fit(X, y, weights)
        testing.assert_array_equal(y, model.predict(X))
        assert model.feature == 0
        assert model.threshold == 1

    def test_fit_homogenous_data(self):
        """Test that we can fit homogenous labels with perfect accuracy."""
        model = Stump()
        n = 6
        d = 3
        X = np.random.rand(n, d)
        y = np.ones(n)
        weights = np.ones_like(y)
        model.fit(X, y, weights)
        testing.assert_array_equal(y, model.predict(X))

        # See todo in corresponding function.
        # y *= -1
        # model.fit(X, y, weights)
        # testing.assert_array_equal(y, model.predict(X))

    def test_invalid_inputs(self):
        """Test that the model raises the correct errors."""
        model = Stump()
        with pytest.raises(ValueError) as excinfo:
            model.fit(np.ones((13, 2)), np.ones(10), np.ones(10))
        assert excinfo.value.args == ("First dimension of arguments must be equal.",)
        with pytest.raises(ValueError) as excinfo:
            model.fit(np.ones((1, 2)), np.ones(1), np.zeros(1))
        assert excinfo.value.args == ("Weights must not be all 0.",)

    def test_weights(self):
        """Test that weights affect output."""
        n = 200
        d = 2
        X = np.zeros((n, d))
        X[: n // 2, 0] = X[n // 2 :, 0] = np.linspace(0, 1, n // 2)
        X[:, 1] = np.zeros(n)
        y = np.ones(n)
        y[n // 2 :] = -1

        weights = np.ones_like(y)
        weights[n // 2 :] = 2
        model = Stump()
        model.fit(X, y, weights)
        assert model.threshold == 1

        weights = weights[::-1]
        model = Stump()
        model.fit(X, y, weights)
        assert model.threshold == 0


class TestAdaBoost(AbstractTestCase):
    """Test that we can overfit."""

    def test_fit(self):
        """Create some data and overfit."""
        n = 40
        d = 5
        X, y = self.gaussian_data(n=n, d=d)
        y[y == 0] = -1

        model = AdaBoost(1000)
        model.fit(X, y)
        y_pred = model.predict(X)
        assert (y_pred == y).all()
        assert (
            len(model.learners)
            == len(model.learner_weights)
            == model.num_learners
            == 1000
        )


class TestMlpClassifier(AbstractTestCase):
    """Tests for MLP Classifier."""

    def test_overfit(self):
        """Negative regularization + linearly separable data should mean overflow."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 1, 1])
        model = MLPClassifier(input_size=2, num_hidden=1, reg=-1)
        with pytest.warns(RuntimeWarning) as warninfo:
            model.fit(X, y, lr=9999, epochs=999, progress_bar=False)
        self.assert_overflow_warning(warninfo)
        assert np.isnan(model.v.sum())
        assert np.isnan(model.w.sum())

        model = MLPClassifier(input_size=2, num_hidden=1, reg=0.001)
        model.fit(X, y, lr=0.1, epochs=999, progress_bar=False)
        assert not np.isnan(model.v.sum())
        assert not np.isnan(model.w.sum())

    def test_fit(self):
        """Should be able to fit to simple data."""
        model = MLPClassifier(input_size=2, num_hidden=10)
        self.assert_classifier_fit_circles(
            model, min_accuracy=0.95, fit_kwargs={"epochs": 2000, "lr": 0.5}
        )
        model = MLPClassifier(input_size=2, num_hidden=10)
        self.assert_classifier_fit_circles(
            model, min_accuracy=0.95, fit_kwargs={"epochs": 2000, "lr": 0.5}
        )
        model = MLPClassifier(input_size=2, num_hidden=2)
        self.assert_classifier_fit_blobs(
            model,
            min_accuracy=1,
            fit_kwargs={"epochs": 20, "lr": 1},
            blob_kwargs={"centers": np.array([[5, 0], [0, -5]])},
        )
