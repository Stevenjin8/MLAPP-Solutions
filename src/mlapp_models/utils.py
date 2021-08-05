"""Reusable utility functions."""
from typing import Callable, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function"""
    positives = z > 0
    negatives = np.logical_not(positives)
    results = np.zeros_like(z)
    results[negatives] = np.exp(z[negatives])
    results[negatives] = results[negatives] / (1 + results[negatives])
    results[positives] = 1 / (1 + np.exp(-z[positives]))
    return results


def pairwise_rbf_kernel(x: np.ndarray, y: np.ndarray, ell: float = 1.0) -> np.ndarray:
    r"""Vectorized pairwaise rbf kernel.

    Arguments
    ---------
    x : np.ndarray
        1-D array with the first arguments for the kernel.
    y : np.ndarray
        1-D array with the second arguments for the kernel. Does not need to have the
        same size as `x`.
    ell : float
        Scale parameter. Equivalent to $\frac1{2\sigma}$.

    Returns
    -------
    np.ndarray
        A `size(x), size(y)` array with the pairwise kernel values.
    """
    return np.exp(-(((np.expand_dims(x, axis=1) - y) / ell) ** 2).sum(axis=2))


def contour_plot(
    funcs: Iterable[Callable[[np.ndarray], np.ndarray]],
    contour_func: Callable[..., plt.Artist] = plt.contour,
    x_center: int = 0,
    x_range: int = 1,
    x_samples: int = 100,
    y_center: int = 0,
    y_range: int = 1,
    y_samples: int = 100,
    func_kwargs: Optional[dict] = None,
    contour_kwargs: Optional[dict] = None,
) -> List[plt.Artist]:
    func_kwargs = func_kwargs or {}
    contour_kwargs = contour_kwargs or {}
    xlist = np.linspace(x_center - x_range, x_center + x_range, x_samples)
    ylist = np.linspace(y_center - y_range, y_center + y_range, y_samples)
    X, Y = np.meshgrid(xlist, ylist)
    inputs = np.array([X.reshape(-1), Y.reshape(-1)]).T
    Zs = [func(inputs, **func_kwargs).reshape(x_samples, y_samples) for func in funcs]
    return [contour_func(X, Y, Z, **contour_kwargs) for Z in Zs]
