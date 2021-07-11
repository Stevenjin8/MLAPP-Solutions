"""Reusable utility functions."""
from typing import Callable, Iterable, List, Optional

import numpy as np
import matplotlib.pyplot as plt


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
