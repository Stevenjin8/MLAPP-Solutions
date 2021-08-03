"""Abstract interfaces for models."""
import numpy as np


class AbstractModel:
    """Interface of machine learning model."""

    def fit(X: np.ndarray, y: np.ndarray, **kwargs) -> any:
        """Fit model to data."""
        raise NotImplementedError("You must implement this method.")

    def predict(X: np.ndarray) -> np.ndarray:
        """Make predictions of data."""
        raise NotImplementedError("You must implement this method.")
