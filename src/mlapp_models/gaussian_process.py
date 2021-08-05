"""Gaussian processes."""
from typing import Optional, Tuple

import numpy as np

from .utils import pairwise_rbf_kernel, sigmoid


class GaussianProcessRegressor:
    r"""Perform regression using gaussian process with zero mean.

    Attributes
    ----------
    covar : np.ndarray
        The covariance matrix given by k(x_train, x_train) + I*eps.
    covar_inv : np.ndarray
        The inverse of `covar`.
    var_eps : float
        The variance noise gaussian noise $y_i = f_i + epsilon_i$, where
        epsilon_i ~ \mathcal{N}(0, var_eps)
    kernel : callable
        Vectorized kernel function to act as a proxy for covariances. Should give 0
        if inputs are extreme different and should rise as inputs become similar.
    x_train : np.ndarray
        A D x N matrix with the training inputs.
    y_train: an N dimensional array with the training outputs.
    var0 : float
        The prior variance
    ell : scale parameter for kernel
    See:
    https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote15.html
    """

    covar: np.ndarray
    covar_inv: np.ndarray
    var_eps: float

    def __init__(self, var_eps: float, var0: float, ell: float) -> None:
        """Set parameters for the class. See class docstring for arguments."""
        self.var_eps = var_eps
        self.var0 = var0
        self.ell = ell

    def kernel(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.var0 * pairwise_rbf_kernel(x, y, ell=self.ell)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Make as many calculations as possible before inference time.
        Which is really calculating and setting `covar` and  `covar_inv`.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.covar = self.kernel(x_train, x_train) + np.eye(len(x_train)) * self.var_eps
        self.covar_inv = np.linalg.inv(self.covar)

    def predict(self, x_star: np.array) -> Tuple[np.array, np.array]:
        """Find $p(f_star | x_train, y_train)$

        Parameters
        ----------
        x_start : np.array
            The inputs

        Returns
        -------
        mu_star : np.array
            The mean of f_star.
        covar_star : np.ndarray
            The covariance of f_star. Use the diag of this for variance of individual
            outputs.

        """
        # If the joint $p(f*, f)$ is gaussian, so is the conditional $p(f* | f)$.
        k_star = self.kernel(self.x_train, x_star)
        k_2star = self.kernel(x_star, x_star) + np.eye(len(x_star)) * self.var_eps
        mu_star = k_star.T @ self.covar_inv @ self.y_train
        covar_star = k_2star - k_star.T @ self.covar_inv @ k_star
        return mu_star, covar_star


class GaussianProcessClassifier:
    """Binary classification using Gaussian processes.

    Attributes
    ----------
    kernel: Kernel
        Vectorized kernel function.
    kernel_kwargs: dict
        Extra keyword arguments for the kernel function
    x_train: np.ndarray
        The trainign inputs.
    f_hat: np.ndarray
        The expected value of f at x_train
    w: np.ndarray
        The Hessian at the last iteration.
    """

    kernel_kwargs: dict
    x_train: np.ndarray
    f_hat: Optional[np.ndarray]
    w: Optional[np.ndarray]
    var_eps: float

    def __init__(
        self, var_eps: float = 1.0, var0: float = 1.0, ell: float = 1.0
    ) -> None:
        """Initialize the class."""
        self.var_eps = var_eps
        self.var0 = var0
        self.ell = ell

    def kernel(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """RBF kernel function."""
        return self.var0 * pairwise_rbf_kernel(x, y, ell=self.ell)

    @classmethod
    def grad_loss(cls, y: np.ndarray, f: np.ndarray, k_inv: np.ndarray) -> np.ndarray:
        """Compute the gradient for the loss function.

        Parameters
        ----------
        y : np.ndarray
            The noisy observations for f.
        f : np.ndarray
            The true function values.
        k_inv : np.ndarray
            The inverse of the Grahm matrix.
        """
        return (sigmoid(f * y) - 1) * y + k_inv @ f

    @classmethod
    def hess_loss(cls, y: np.ndarray, f: np.ndarray, k_inv: np.ndarray) -> np.ndarray:
        """Compute  the hession of the loss.

        See `grad_loss` for parameters.
        """
        return -np.diag(sigmoid(y * f) * (1 - sigmoid(y * f))) + k_inv

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        n_iter: int = 101,
    ) -> None:
        """Fit model to data with IRLS.

        Parameters
        ----------
        x_train : np.ndarray
            The input matrix.
        y_train : np.ndarray
            Binary class labels in {-1, 1}.
        n_iter : int, optional
            The number of iterations. Defaults to 15.
        """
        self.x_train = x_train

        self.k = self.kernel(x_train, x_train) + np.eye(len(x_train)) * self.var_eps
        self.k_inv = np.linalg.inv(self.k)  # You're really not supposed to invert K

        f = np.random.rand(*y_train.shape)
        for _ in range(n_iter):
            grad = self.grad_loss(y_train, f, self.k_inv)
            hess = self.hess_loss(y_train, f, self.k_inv)
            f = f - np.linalg.inv(hess) @ grad  # Neither should you invert W
        self.f_hat = f
        self.w = hess

    def predict_mean(self, x_star: np.ndarray) -> np.ndarray:
        """Predict the mean value for new points.

        Parameters
        ----------
        x_star : np.ndarray
            The points in question.

        Returns
        -------
        The mean f for the new points.
        """
        k_star = self.kernel(x_star, self.x_train)
        return k_star @ self.k_inv @ self.f_hat

    def predict_var(self, x_star: np.ndarray) -> np.ndarray:
        """Calculate the predictive posterior variance.

        Parameters
        ----------
        x_star: np.ndarray
            The data for which to calculate the variance.

        Returns:
        np.ndarray
            The covariance matrix. Usually, you will only use the diagonal.


        """
        k_2star = self.kernel(x_star, x_star) + np.eye(len(x_star)) * self.var_eps
        k_star = self.kernel(x_star, self.x_train)

        return (
            k_2star
            - k_star @ self.k_inv @ k_star.T
            + k_star @ self.k_inv @ np.linalg.inv(self.k_inv + self.w) @ k_star.T
        )

    def predict(self, x_star) -> np.ndarray:
        """Predict labels."""
        preds = np.ones(len(x_star))
        means = self.predict_mean(x_star)
        preds[means < 0] = -1
        return preds
