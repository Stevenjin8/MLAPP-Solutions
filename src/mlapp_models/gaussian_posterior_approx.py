"""Utility functions for Laplacian approximation of the posterior of a Gaussian.

Note: THIS IS NOT ABOUT APPROXIMATING THE POSTERIOR OF A GAUSSIAN USING A LAPLACE
DISTRIBUTION.
"""
import numpy as np
from scipy.stats import multivariate_normal


def sufficient_statistics(data: np.ndarray) -> tuple:
    """Utility function to unpack arguments and get statistics of the inputs.

    Arguments
    ---------
    data : np.ndarray
        a 1-D array with observations from a univariate distribution.

    Returns
    -------
        n : int
            The length of `data`.
        s2 : float
            The sample variance.
        x_bar : float
            The sample mean.
    """
    n = len(data)
    s2 = data.var()
    x_bar = data.mean()
    return n, s2, x_bar


def true_log_posterior(mu: np.ndarray, ell: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Find the log of the unormalized posterior of a univariate gaussian with
    uninformative prior (equation 21.203).

    Parameters
    ----------
    mu : np.ndarray
        1-D array of mu values to evaluate the posterior.
    ell : np.ndarray
        1-D array of ell (log sigma) values to evaluate the posterior.
    data : np.ndarray
        1-D array with the data to condition the posterior on.

    Returns
    -------
    np.ndarray
        1-D array with the log posterior where the nth element is the posterior
        of the ith elements of `mu` and `ell`.
    """
    n, s2, x_bar = sufficient_statistics(data)
    sigma = np.exp(ell)
    return -n * ell - (n * s2 + n * (x_bar - mu) ** 2) / (2 * sigma ** 2)


def laplace_approx(mu: np.ndarray, ell: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Laplacian approximation of the posterior of a univariate gaussian with
    uninformative prior.

    Parameters
    ----------
    mu : np.ndarray
        1-D array of mu values to evaluate the posterior.
    ell : np.ndarray
        1-D array of ell (log sigma) values to evaluate the posterior.
    data : np.ndarray
        1-D array with the data to condition the posterior on.

    Returns
    -------
    np.ndarray
        1-D array with the approximate log posterior where the nth element is the
        approximate posterior of the ith elements of `mu` and `ell`.
    """
    n, s2, x_bar = sufficient_statistics(data)
    sigma = np.exp(ell)
    H = np.array(
        [
            [-n / s2, 0],
            [0, -2 * n],
        ]
    )  # equation 21.207
    post_mu = np.array([x_bar, 0.5 * np.log(s2)])
    inputs = np.array([mu, ell]).T
    return np.log(
        multivariate_normal.pdf(
            inputs,
            post_mu,
            -np.linalg.inv(H),
        )
    )