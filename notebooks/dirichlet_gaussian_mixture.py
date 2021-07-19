"""Sample the posterior of a Dirichlet Gaussian mixture model."""
import warnings

import numpy as np
import numpy.linalg as la
import tqdm
from scipy import stats


class DirichletGaussianMixtureSampler:
    """Sample the posterior of a Dirichlet Gaussian mixture model with know priors and
    likelihood covariances with collapsed Gibbs sampler.
    See sections 4.6.1 and 25.2.4 and
    https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf

    Attributes
    ----------
    alpha : float
        Parameter controlling how many different clusters there will be. High alpha means
        more clusters.
    prior_cov : np.ndarray
        DxD matrix with the covariance of the means of the clusters.
    prior_cov_inv : np.ndarray
        DxD matrix with the inverse of `prior_cov`
    obs_cov : np.ndarray
        DxD matrix with the covariance of the observations.
    obs_cov_inv : np.ndarray
        DxD matrix with the inverse of `obs_cov`.
    prior_mean : np.ndarray
        D-dimensional array with the mean of cluster
    clusters : np.ndarray
        NxK matrix with cluster assignments one-hot encoded. Note that K can vary.
    data : np.ndarray
        DxN matrix with the data from which to sample the posterior
    burn_in_iter : int
        The number of samples to burn in.
    """

    alpha: float
    _prior_cov: np.ndarray
    _obs_cov: np.ndarray
    _prior_cov_inv: np.ndarray
    _obs_cov_inv: np.ndarray
    prior_mean: np.ndarray
    clusters: np.ndarray
    data: np.ndarray
    burn_in_iter: int

    def __init__(
        self,
        alpha: float,
        prior_cov: np.ndarray,
        prior_mean: np.ndarray,
        obs_cov: np.ndarray,
        data: np.ndarray,
        burn_in_iter: int = 10,
    ):
        """Initialize an instance of the class

        Arguments
        ---------
        See class docstring.
        """
        self.alpha = alpha
        self.prior_cov = prior_cov
        self.prior_mean = prior_mean
        self.obs_cov = obs_cov
        self.data = data
        # Initialize cluster to a column of ones.
        self.clusters = np.ones((self.n, 1))
        self.burn_in_iter = burn_in_iter
        self._samples_generated = 0

    @property
    def prior_cov(self) -> np.ndarray:
        """Get the covariance of the prior."""
        return self._prior_cov

    @property
    def prior_cov_inv(self) -> np.ndarray:
        """Get the inverse of the covariance of the prior."""
        return self._prior_cov_inv

    @prior_cov.setter
    def prior_cov(self, prior_cov: np.ndarray) -> None:
        """Set these together so we don't have to compute the inverse every time."""
        self._prior_cov = prior_cov
        self._prior_cov_inv = la.inv(prior_cov)

    @property
    def obs_cov(self) -> np.ndarray:
        """Get the observation covariance."""
        return self._obs_cov

    @property
    def obs_cov_inv(self):
        """Get the inverse of the the observation covariance."""
        return self._obs_cov_inv

    @obs_cov.setter
    def obs_cov(self, obs_cov: np.ndarray):
        """Set these together so we don't have to compute the inverse every time."""
        self._obs_cov = obs_cov
        self._obs_cov_inv = la.inv(obs_cov)

    @property
    def _burned_in(self):
        """Check if we have burned in."""
        return self._samples_generated >= self.burn_in_iter

    @property
    def n(self):
        """Number of data points in data."""
        return len(self.data)

    def post_pred(self, new_x: np.ndarray, data: np.ndarray) -> float:
        """Get the posterior predictive density for a new datapoint given some data.

        Arguments
        ---------
        new_x : np.ndarray
            The new point which we want to evaluate.
        data : np.ndarray
            The data to condition on.

        Returns
        -------
        float : The pdf of the posterior predictive distribution at `new_x`.
        """
        n = len(data)
        sample_mean = data.mean(axis=0) if n else None

        # Equation 4.173
        post_cov = la.inv(self.prior_cov_inv + n * self.obs_cov_inv)
        # Equation 4.174
        post_mean = post_cov @ (
            la.inv(post_cov) @ self.prior_mean
            + (n * self.obs_cov_inv @ sample_mean if n else 0)
        )
        # See section 2.4 of
        # https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
        post_pred_mean = post_mean
        post_pred_cov = post_cov + self.obs_cov

        return stats.multivariate_normal.pdf(new_x, post_pred_mean, post_pred_cov)

    @property
    def cluster_priors(self) -> np.ndarray:
        """Find the prior probability of each cluster and a new cluster.

        Returns
        -------
        np.ndarray
            A K+1 length array where each element represents the prior probability of
            the corresponding cluster and the last element represents the probability
            of a new cluster.
        """
        cluster_priors = np.zeros(self.clusters.shape[1] + 1)
        cluster_priors[:-1] = self.clusters.sum(axis=0)
        cluster_priors[-1] = self.alpha
        cluster_priors /= cluster_priors.sum()
        return cluster_priors

    def get_cluster_likelihoods(self, x: np.ndarray) -> np.ndarray:
        """Get the likelyhoods of `x` of each cluster (including a new cluster) using
        the posterior predictive distributions of each cluster.

        Parameters
        ----------
        x : np.ndarray
            D length array in question

        Returns
        -------
        np.ndarray
            K+1 dimensional array with the posterior predictive likelihoods for each
            cluster.
        """

        cluster_likelihoods = np.zeros(self.clusters.shape[1] + 1)
        for j in range(len(cluster_likelihoods) - 1):
            cluster_likelihoods[j] = self.post_pred(
                x, self.data[self.clusters[:, j] == 1]
            )
        cluster_likelihoods[-1] = self.post_pred(x, np.array([]))  # No data
        return cluster_likelihoods

    def get_cluster_posterior(self, x: np.ndarray) -> np.ndarray:
        """Get the posteriors distribution of clusters for `x`.

        Parameters
        ----------
        x : np.ndarray
            D length array with the data point in question

        Returns
        -------
        np.ndarray
            The posterior distribution over clusters, including a new cluster.
        """
        cluster_post = self.cluster_priors * self.get_cluster_likelihoods(x)
        cluster_post /= cluster_post.sum()
        eps = np.finfo(cluster_post.dtype).eps
        # This is to make sure probabilities sum to 1. Otherwise we get an error from
        # The multinomial sampling later on.
        if (eps := cluster_post.sum() - 1) > 0:
            cluster_post[np.argmax(cluster_post)] -= eps
        return cluster_post

    def update_clusters(self, i: int, sample: np.ndarray) -> None:
        """Update the clusters. This might create a new column. This will also delete
        any clusters with no points.

        Parameters
        ----------
        i : int
            The index to update
        sample : np.ndarray
            A K+1 length one-hot encoding with the chosen cluster. If the last elemnent
            is 1, then a new column will be created in `self.clusters`.
        """
        if sample[-1]:
            self.clusters = np.append(self.clusters, np.zeros((self.n, 1)), 1)
            self.clusters[i] = sample
        else:
            self.clusters[i] = sample[:-1]

        self.clusters = np.delete(self.clusters, self.clusters.sum(axis=0) == 0, 1)

    def burn_in(self) -> None:
        """Perform a burn-in."""
        for _ in tqdm.trange(self._samples_generated, self.burn_in_iter):
            self._sample()
            self._samples_generated += 1

    def _sample(self) -> np.ndarray:
        """Generate a sample from the posterior cluster assignments using collapsed
        Gibbs sampling. The trick is that we don't have to sample the cluster means,
        because we can integrate it out using the posterior predictive distribution.

        Returns
        -------
        np.ndarray
            DxK matrix with one-hot encodings of cluster assingments.
        """
        indices = np.arange(self.n)  # Shuffle by indices
        np.random.shuffle(indices)
        for i in indices:
            x = self.data[i]
            self.clusters[i, :] = 0
            cluster_post = self.get_cluster_posterior(x)
            sample = stats.multinomial.rvs(1, cluster_post)
            self.update_clusters(i, sample)
        # Only add if we haven't burned in.
        self._samples_generated += not self._burned_in
        return self.clusters

    def sample(self):
        """Wrapper around `_sample` to check that the model has burned in.

        Returns
        -------
        np.ndarray
            DxK matrix with a sample from the posterior.
        """
        if not self._burned_in:
            warnings.warn(
                "Samples may not be representative as the model has not performed a burn-in.",
                RuntimeWarning,
            )
        return self._sample()
