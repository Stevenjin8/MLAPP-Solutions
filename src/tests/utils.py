"""Utilities for tests."""
from typing import Dict, Optional, Tuple

import numpy as np
from _pytest.recwarn import WarningsChecker
from mlapp_models.abstract import AbstractModel
from scipy import stats
from sklearn.datasets import make_blobs, make_circles, make_moons


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

    @staticmethod
    def gaussian_data(n: int, d: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create Gaussian data with binary labels."""
        X = stats.multivariate_normal.rvs(mean=-np.ones(d), cov=np.eye(d), size=n // 2)
        X = np.append(
            X,
            stats.multivariate_normal.rvs(mean=np.ones(d), cov=np.eye(d), size=n // 2),
            axis=0,
        )
        y = np.ones(n)
        y[: n // 2] = 0
        return X, y

    @staticmethod
    def assert_overflow_warning(warninfo: WarningsChecker):
        """Assert that a warning is an overflow warning."""
        assert any("overflow" in warning.message.args[0] for warning in warninfo.list)

    @staticmethod
    def assert_classifier_fit(
        model: AbstractModel,
        X: np.ndarray,
        y: np.ndarray,
        fit_kwargs: Dict[str, any],
        min_accuracy: float,
    ):
        """Assert that a model can fit to some data."""

        model.fit(X, y, **fit_kwargs)
        preds = model.predict(X)
        accuracy = (preds == y).sum() / len(X)
        assert accuracy >= min_accuracy

    @classmethod
    def assert_classifier_fit_moons(
        cls,
        model: AbstractModel,
        moon_kwargs: Optional[Dict[str, any]] = None,
        fit_kwargs: Optional[Dict[str, any]] = None,
        min_accuracy: float = 0.95,
    ):
        moon_kwargs = moon_kwargs or {"noise": 0.1}
        fit_kwargs = fit_kwargs or {}
        X, y = make_moons(**moon_kwargs)
        cls.assert_classifier_fit(
            model=model, X=X, y=y, fit_kwargs=fit_kwargs, min_accuracy=min_accuracy
        )

    @classmethod
    def assert_classifier_fit_circles(
        cls,
        model: AbstractModel,
        circle_kwargs: Optional[Dict[str, any]] = None,
        fit_kwargs: Optional[Dict[str, any]] = None,
        min_accuracy: float = 0.95,
    ):
        moon_kwargs = circle_kwargs or {"noise": 0.1, "factor": 0.5}
        fit_kwargs = fit_kwargs or {}
        X, y = make_circles(**moon_kwargs)
        cls.assert_classifier_fit(
            model=model, X=X, y=y, fit_kwargs=fit_kwargs, min_accuracy=min_accuracy
        )

    @classmethod
    def assert_classifier_fit_blobs(
        cls, model, blob_kwargs=None, fit_kwargs=None, min_accuracy: float = 0.95
    ):
        blob_kwargs = blob_kwargs or {}
        fit_kwargs = fit_kwargs or {}
        X, y = make_blobs(**blob_kwargs)
        cls.assert_classifier_fit(
            model=model, X=X, y=y, fit_kwargs=fit_kwargs, min_accuracy=min_accuracy
        )
