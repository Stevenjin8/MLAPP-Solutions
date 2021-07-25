"""Module with adaptive basis functions"""
from functools import cached_property
from math import prod
from typing import Optional, List, Union, Tuple

import numpy as np
import numba
from tqdm import tqdm


@numba.experimental.jitclass()
class Stump:
    """Tree with a depth of 1.

    Attributes
    ----------
    feature: int
        The index of the feature to split on.
    threshold: float
        The threshold to split on
    """

    feature: int
    threshold: float

    def __init__(self):
        """Define init so we can jit the code."""

    def fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> None:
        """Fit model to data.

        Parameters
        ----------
        X : np.ndarray
            A two dimensional array with the inputs of the data.
        y : np.ndarray
            A one dimensional array with the lables of the data. Should be in {-1, 1}.
        weights : np.ndarray
            The weight for each data point. Doesn't have to sum to one, but values
            should be positive.
        """
        best_error = np.inf
        best_indices: Tuple[int, int] = (0, 0)
        for i in range(len(X)):
            for j in range(X.shape[1]):
                left_indices = X[:, j] < X[i, j]
                right_indices = np.logical_not(left_indices)
                left_weights = weights[left_indices]
                right_weights = weights[right_indices]
                left_y = y[left_indices]
                right_y = y[right_indices]

                error = (
                    left_weights[left_y != -1].sum()
                    + right_weights[right_y != -1].sum()
                )
                error = error / weights.sum()
                if error < best_error:
                    best_error = error
                    best_indices = (i, j)

        self.threshold = X[best_indices]
        self.feature = best_indices[1]

    def predict(self, X: np.ndarray):
        """Make predictions using discrimitaiton function."""
        preds = np.ones(X.shape[0])
        preds[X[:, self.feature] < self.threshold] = -1
        return preds


class AdaBoost:
    """Adaptive boost binary classification.

    Attributes
    ----------
    num_learners : int
        Number of weak learners.
    learners : List[Stump]
        The weak learners.
    learner_weights : np.ndarray
        The weight for each learner.
    """

    num_learners: int
    learners: List[Stump]
    learner_weights: np.ndarray

    def __init__(self, num_learners: int):
        """Initialize an instance of the class.

        Parameters
        ----------
        See class docstring.
        """
        self.num_learners = num_learners
        self.learners = []
        self.learner_weights = np.ones(num_learners)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Decision function for the classification."""
        # Matrix where predictions[i, j] is the prediction (1 or -1) for data point i
        # by learner j.
        predictions = np.zeros((len(X), self.num_learners))
        for i, learner in enumerate(self.learners):
            predictions[:, i] = learner.predict(X)
        return predictions @ self.learner_weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class of the rows of X."""
        preds = np.ones(len(X))
        scores = self.score(X)
        preds[scores < 0] = -1
        return preds

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit model to data.

        Parameters
        ----------
        X : np.ndarray
            2d array with the training inputs.
        y : np.ndarray
            The true class labels.
        """
        weights = np.ones_like(y) / len(y)
        for i in tqdm(range(self.num_learners)):
            learner = Stump()
            # Its important to shuffle the data to avoid "local minimums".
            idx = np.random.permutation(len(X))
            learner.fit(X[idx], y[idx], weights[idx])

            preds = learner.predict(X)
            err = weights[preds != y].sum() / weights.sum()
            learner_weight = np.log((1 - err) / err)
            weights = weights * np.exp(learner_weight * (preds != y))

            weights = weights / abs(weights).sum()  # normalize weights

            self.learners.append(learner)
            self.learner_weights[i] = learner_weight

        # normalize weights
        self.learner_weights = self.learner_weights / abs(self.learner_weights).sum()


# @numba.experimental.jitclass([("v", numba.float64[:, :]), ("w", numba.float64[:, :])])
class MLPClassifier:
    """Multilayer perceptron binary classifier with one hidden layer."""

    num_hidden: int
    input_size: int
    reg: float
    v: np.ndarray
    w: np.ndarray

    def __init__(self, input_size: int, num_hidden: int, reg: float = 0.00001):
        """Initialize an instance of the class."""
        self.num_hidden = num_hidden
        self.input_size = input_size
        self.v = np.random.rand(num_hidden, input_size) - 0.5
        self.w = np.random.rand(num_hidden, 1) - 0.5
        self.reg = reg

    def activation_function(self, a: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid activation function."""
        positive_idx = a >= 0
        negative_idx = np.logical_not(positive_idx)
        output = np.zeros_like(a)
        output[positive_idx] = 1 / (1 + np.exp(-a[positive_idx]))
        c = np.exp(a[negative_idx])
        output[negative_idx] = c / (1 + c)
        return output

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find the values of the basis functions and probability that y=1 for each
        row in `X`.

        Parameters
        ----------
        X : np.ndarray
            Two dimensional array with rows as data points.

        Returns
        -------
        z : np.ndarray
            The values of the basis functions.
        np.ndarray
            p(y=1 | X)
        """
        z = self.activation_function(X @ self.v.T)
        return z, self.activation_function(z @ self.w)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict classes for each row of `X`. Predict 1 if p(y=1 | x_i) > threshold.

        Parameters
        ----------
        X : np.ndarray
        threshold : float

        Returns
        -------
        preds : np.ndarray
            The predictions.
        """
        preds = np.ones(len(X))
        _, probs = self.forward(X)
        probs = probs.reshape(-1)
        preds[probs <= threshold] = 0
        return preds

    @cached_property
    def num_parameters(self) -> int:
        """Number of parameters (excluding biases)."""
        return len(self.w) + prod(self.v.shape) - len(self.v)

    def loss(self, X: np.ndarray, y: np.ndarray, y_hat: np.ndarray) -> float:
        """Negative log likelyhood functions.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        y : np.ndarray
            The true class labels.
        y_hat : np.ndarray
            The probability that the class of each row in x equals 1.

        Returns
        -------
        float
            The mean loss.
        """
        losses = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return losses.mean() + self.reg / self.num_parameters * (
            (self.v[:, -1] ** 2).sum() + (self.w ** 2).sum()
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.1,
        epochs: int = 20000,
        progress_bar: bool = True,
    ) -> np.ndarray:
        """Fit model parameters to data.

        Parameters
        ----------
        X : np.ndarray
            The training inputs.
        y : np.ndarray
            The training class labels.
        lr : float, default=0.1
            The step size for gradient optimization.
        epochs : int, default = 20,000
            The number of updates to perform over the data.

        Returns
        -------
        losses : np.ndarray
            The loss at each epoch.
        """
        y = y.reshape(-1, 1)
        lr = lr / len(X)
        losses = np.zeros(epochs)
        for i in tqdm(range(epochs), disable=not progress_bar):
            a = X @ self.v.T
            z = self.activation_function(a)
            y_hat = self.activation_function(z @ self.w)

            # Backprop
            grad_b = y_hat - y
            grad_w = z.T @ grad_b + 2 * self.reg / self.num_parameters * self.w
            grad_z = self.w.T * grad_b
            grad_a = grad_z * z * (1 - z)
            reg_grad_v = 2 * self.reg / self.num_parameters * self.v
            reg_grad_v[:, -1] = 0  # Don't regularize weights.
            grad_v = grad_a.T @ X + reg_grad_v
            # Update weights
            self.w -= lr * grad_w
            self.v -= lr * grad_v

            loss = self.loss(X, y, y_hat)
            losses[i] = loss
        return losses
