"""Module with Hidden Markov Models (Chapter 17)."""
from typing import List, Optional

import numpy as np
from tqdm import tqdm


class HiddenMarkovModel:
    r"""Hidden Markov Model with 1-D discrete visible output.

    Attributes
    ----------
    K : int
        Cardinality of latent dimension.
    L : int
        Cardinality of the visible dimension.
    Psi : np.ndarray
        :math:`K \times K` transition (matrix) array with
        :math:`\psi_{i,j} = p(z_t=j | p(z_{t-1}=i)`.
    B : np.ndarray
        :math:`L \times K` array with :math:`b_{i,j} = p(x=i | z=j)`.
    pi : np.array
        Size :math:`K` array with :math:`pi_k = p(z_1=k)`.
    _latent_choices : np.ndarray
        The choices of the latent variable. Should be equal to
        :code:`np.array([0, 1, ..., self.K - 1])`
    _visible_choices : np.ndarray
        The choices of the visible variable. Should be equal to
        :code:`np.array([0, 1, ..., self.L - 1])`
    """

    K: int
    L: int
    B: np.ndarray
    Psi: np.ndarray
    pi: np.ndarray
    _latent_choices: np.ndarray
    _visible_choices: np.ndarray

    def __init__(
        self,
        K: int,
        L: int,
        Psi: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        pi: Optional[np.ndarray] = None,
    ):
        """Initialize an instance of the class.

        Parameters
        ----------
        See class docstring.
        """
        self.K = K
        self.L = L
        self._latent_choices = np.arange(K, dtype=np.int8)
        self._visible_choices = np.arange(L, dtype=np.int8)
        # Initialize parameters with random values.
        self.Psi = np.random.rand(K, K) if Psi is None else Psi
        self.Psi = self.Psi / self.Psi.sum(axis=1, keepdims=True)
        self.B = np.random.rand(L, K) if B is None else B
        self.B = self.B / self.B.sum(axis=0)
        self.pi = np.random.rand(K) if pi is None else pi
        self.pi = self.pi / self.pi.sum()

    def _alphas(self, x: np.ndarray) -> np.ndarray:
        r"""Compute the distribution of z given data up to time t. It is
        faster to compute them all at the same time, as the formula is recursive.

        Parameters
        ----------
        x : np.ndarray
            A 1-D array of observations of length :math:`T`. Items in :code:`x` should
            be nonegative integers less than :code:`self.K`.

        Returns
        -------
        alphas : np.ndarray
            A 2-D array with the distribution of the hidden variables given data up to
            time `t`. In other words, :math:`\alpha_{z,t} = p(z|\mathbf{x}_{0:t})`.
        """
        alphas = np.zeros((self.K, len(x)))
        alphas[:, 0] = self.pi * self.B[x[0]]
        alphas[:, 0] = alphas[:, 0] / alphas[:, 0].sum()
        for i in range(1, len(x)):
            # equation 17.48
            alphas[:, i] = self.B[x[i]] * (self.Psi.T @ alphas[:, i - 1])
            alphas[:, i] = alphas[:, i] / alphas[:, i].sum()
        return alphas

    def _betas(self, x: np.ndarray) -> np.ndarray:
        r"""Compute the likelyhood of the data w.r.t. the latent state.

        Parameters
        ----------
        x : np.ndarray
            A 1-D array of observations of length :math:`N`. Items in :code:`x` should
            be nonegative integers less than :code:`self.K`.

        Returns
        -------
        betas : np.ndarray
            A :math:`K \times N` array with the distribution of the hidden variables
            given data up to time :math:`t`. In other words,
            :math:`\beta_{i,t} = p(x_{t+1:T}|z_t=i)` (eq. 17.53).
        """
        betas = np.ones((self.K, len(x)))
        for i in range(len(x) - 2, -1, -1):  # iterate backwards.
            # eq. 17.59
            betas[:, i] = self.Psi @ (self.B[x[i + 1]] * betas[:, i + 1])
        return betas

    def _xis(self, x: np.ndarray) -> np.ndarray:
        r"""Find the "two-slice smoothed marginals" for the data.

        Parameters
        ----------
        x : np.ndarray
            A 1-D array of observations of length :math:`T`. Items in :code:`x` should
            be nonegative integers less than :code:`self.K`.

        Returns
        -------
        xis : np.ndarray
            A :math:`T \times K \times K` array where
            :math:`\xi_{t,i,j} = p(z_t=i, z_{t+1}=j|x)` (eq. 17.62).

        Notes
        -----
        Note that although eq. 17.67 is correct, the jump between 17.62 to 17.63 is
        not.
        """
        alphas = self._alphas(x)
        betas = self._betas(x)
        phis = self.B[x].T  # This gives :math:`\phi_t` for :math:`t=0...T-1`.
        # Vectorized version of eq. 17.67 with normalization.
        xis = self.Psi * np.einsum(
            "ij,kj->jik",
            alphas[:, :-1],
            phis[:, 1:] * betas[:, 1:],
        )
        return xis / xis.sum(axis=(1, 2), keepdims=True)

    def _gammas(self, x: np.ndarray) -> np.ndarray:
        r"""Find the smoothed posterior marginal.

        Parameters
        ----------
        x : np.ndarray
            A 1-D array of observations of length :math:`T`. Items in :code:`x` should
            be nonegative integers less than :code:`self.K`.

        Returns
        -------
        A :math:`K \times T` dimensional array where
        :math:`\gamma_{j,t} = p(z_t=j|\mathbf{x}_{0:T})` (eq. 17.52).
        """
        gammas = self._alphas(x) * self._betas(x)  # eq. 17.53
        return gammas / gammas.sum(axis=0)

    def fit(self, samples: List[np.ndarray], num_iter: int) -> None:
        """Use Baum-Welch to fit model to the data.

        Parameters:
        samples : list of np.ndarray
            A list of 1-D arrays. Items in should be nonegative integers less than
            :code:`self.K`.
        num_iter: int
            The number of iterations to perform.
        """
        for _ in tqdm(range(num_iter)):
            # Compute the sufficient statistics
            all_gammas = [self._gammas(x) for x in samples]
            # eq. 17.98
            init_state_counts = sum(gammas[:, 0] for gammas in all_gammas)
            # eq. 17.99
            transition_counts = sum(self._xis(x).sum(axis=0) for x in samples)
            # eq. 17.100
            state_counts = sum(gammas.sum(axis=1) for gammas in all_gammas)
            # eq. 17.104
            condition_observations = np.zeros((self.L, self.K))
            for i, x in enumerate(samples):
                for t, x_j in enumerate(x):
                    condition_observations[x_j, :] += all_gammas[i][:, t]

            # Update model parameters.
            # eq. 17.103
            self.pi = init_state_counts / init_state_counts.sum()
            # eq. 17.103
            self.Psi = transition_counts / transition_counts.sum(axis=1, keepdims=True)
            # eq. 17.105
            self.B = condition_observations / state_counts

    def _sample_latent(self, length: int) -> np.ndarray:
        """Sample the latent values.

        Parameters
        ----------
        length : int
            The length of the sample.

        Returns
        -------
        z : np.ndarray
            A 1-D array of length :code:`length`.
        """
        z = np.zeros((length,), dtype=np.int8)  # need the `dtype` argument to index.
        z[0] = np.random.choice(self._latent_choices, p=self.pi)
        for i in range(1, length):
            z[i] = np.random.choice(self._latent_choices, p=self.Psi[int(z[i - 1])])
        return z

    def _sample_visible(self, z: np.array) -> np.ndarray:
        """Sample visible values given the latent ones.

        Parameters
        ----------
        z : np.ndarray
            The latent values.

        Returns
        -------
        x : np.ndarray
            The visible ones.
        """
        x = np.zeros_like(z)
        for i in self._latent_choices:
            idx = z == i
            x[idx] = np.random.choice(
                self._visible_choices, size=idx.sum(), p=self.B[:, i]
            )
        return x

    def sample(self, lengths: List[int]) -> List[np.ndarray]:
        """Create samples using the model's parameters.

        Parameters
        ----------
        lengths: list of int
            The length of each sample. Must be a postive integer.

        Returns
        -------
        samples : list of np.ndarray
        """
        samples = []
        for length in lengths:
            samples.append(self._sample_visible(self._sample_latent(length)))
        return samples

    def viberti(self, x: np.ndarray) -> np.ndarray:
        """Find the mode of the posterior.

        Parameters
        ----------
        x : np.ndarray
            The observations.

        Returns
        -------
        np.ndarray
            The mode of the posterior.
        """
        trellis = np.zeros((self.K, len(x)))
        trellis[:, 0] = self.pi * self.B[x[0]]
        trellis[:, 0] = trellis[:, 0] / trellis[:, 0].sum()
        # Store the most probable paths.
        paths = np.ones_like(trellis, dtype=np.int8) * self._latent_choices.reshape(
            -1, 1
        )
        for t in range(1, len(x)):
            transition = trellis[:, t - 1 : t] * self.Psi * self.B[x[t]]
            idx = transition.argmax(axis=0)
            paths[:, :t] = paths[idx, :t]
            deltas = transition[idx, np.arange(self.K)]
            trellis[:, t] = deltas / deltas.sum()

        return paths[trellis[:, -1].argmax()]
