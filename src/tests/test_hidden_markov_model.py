import numpy as np
import pytest
from mlapp_models.hidden_markov_model import HiddenMarkovModel
from numpy import testing

from .utils import AbstractTestCase


class TestHiddenMarkovModel(AbstractTestCase):
    """Tests for hidden markov models."""

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.K = 2
        cls.L = 3
        cls.Psi = np.array([[0.9, 0.1], [0.2, 0.8]])
        cls.B = np.array([[0.9, 0], [0.1, 0.1], [0, 0.9]])
        cls.pi = np.array([0.6, 0.4])
        cls.true_model = HiddenMarkovModel(
            K=cls.K, L=cls.L, Psi=cls.Psi, B=cls.B, pi=cls.pi
        )

    def test_invalid_inputs(self):
        """Test that we get numerical errors where we expect them."""
        # really extreme parameters.
        Psi = np.eye(2)
        B = np.eye(2)
        pi = np.array([1, 0])
        model = HiddenMarkovModel(K=2, L=2, Psi=Psi, B=B, pi=pi)

        with pytest.warns(RuntimeWarning):
            alphas = model._alphas(np.array([1, 1], dtype=np.int8))
        for y in np.nditer(alphas):
            assert np.isnan(y)

        with pytest.warns(RuntimeWarning):
            alphas = model._alphas(np.array([0, 1], dtype=np.int8))
        testing.assert_array_equal(alphas, np.array([[1, np.nan], [0, np.nan]]))

    def test_fit(self):
        """Test that we can model to some data."""
        # Warm-start the model so we don't get stuck at local optima (and train faster).
        Psi = np.array([[0.8, 0.2], [0.3, 0.7]])
        B = np.array([[0.7, 0], [0.3, 0.05], [0, 0.95]])
        pi = np.array([0.8, 0.2])
        model = HiddenMarkovModel(K=self.K, L=self.L, Psi=Psi, B=B, pi=pi)
        samples = self.true_model.sample([50] * 70)
        model.fit(samples=samples, num_iter=20)

        # Check that parameters are similar up to permutation
        idx = [0, 1] if model.pi[0] > model.pi[1] else [1, 0]
        idx = np.array(idx, dtype=np.int8)
        model.Psi = model.Psi[idx]
        model.B = model.B[:, idx]
        model.pi = model.pi[idx]

        testing.assert_allclose(model.Psi, self.true_model.Psi, atol=0.03)
        testing.assert_allclose(model.B, self.true_model.B, atol=0.03)
        testing.assert_allclose(model.pi, self.true_model.pi, atol=0.1)

    def test_viberti(self):
        """Test that we can find the mode of the posterior given the parameters."""
        pi = np.array([0.6, 0.4])
        x = np.array([0, 1, 0, 1, 0, 0, 2, 1, 1, 2, 0, 0, 0], dtype=np.int8)
        expected = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0], dtype=np.int8)
        z_star = self.true_model.viberti(x)
        testing.assert_array_equal(z_star, expected)
