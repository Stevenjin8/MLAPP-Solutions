"""Module with sampling for normal bivariate distributions."""
import numpy as np


class NormalBivariateGibbs:
    """Gibbs sampler for a bivariate normal distribution."""

    def __init__(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        burn_in: int = 10,
    ):
        """Initialize an instance of the class.

        Paramerters
        -----------
        mean : np.ndarray
            Array of length 2 with the mean of the distribution
        cov : np.ndarray
            2x2 symmetric positive-definite array with the covariance of the distribution.
        burn_in : int
            The first n samples to toss.
        step : int
            The step between samples to return.
        """
        self.mean = mean
        self.cov = cov
        self.burn_in = burn_in

    def conditional_sample(self, x_j: float, i: int) -> float:
        """Sample from p(x_i | x_j)."""
        j = int(not bool(i))
        m = self.mean[i] + self.cov[i, j] / self.cov[j, j] * (x_j - self.mean[j])
        s2 = self.cov[i, i] - self.cov[i, j] ** 2 / self.cov[j, j]
        return np.random.normal(loc=m, scale=s2 ** 0.5)

    def sample(self, num_samples: int) -> np.ndarray:
        num_samples = self.burn_in + num_samples
        samples = np.zeros((num_samples, 2))
        for i in range(1, num_samples):
            samples[i, 0] = self.conditional_sample(samples[i - 1, 1], 0)
            samples[i, 1] = self.conditional_sample(samples[i, 0], 1)
        return samples[self.burn_in :]
