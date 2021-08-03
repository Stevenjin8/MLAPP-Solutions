"""Tests for the utils module."""
import numpy as np
from mlapp_models.utils import pairwise_rbf_kernel, sigmoid
from numpy import testing
import pytest

from .utils import AbstractTestCase


class TestSigmoid(AbstractTestCase):
    """Test the Sigmoid function."""

    def test_values(self):
        """Test easy inputs."""
        testing.assert_array_equal(np.array([0.5]), sigmoid(np.array([0.0])))

        inputs = np.array([-10, 20, 0.001, -1])
        expected_output = 1 / (1 + np.exp(-inputs))
        testing.assert_array_equal(expected_output, sigmoid(inputs))

    def test_extreme_values(self):
        """Test that we can input extreme values without numerical errors."""
        inputs = np.array([self.finfo.max, self.finfo.min])
        testing.assert_array_equal(
            np.array([1, 0], dtype=self.float_dtype), sigmoid(inputs)
        )


class TestRbf(AbstractTestCase):
    """Tests for RBF kernel."""

    def test_shape(self):
        """Test that the shapes are correct."""
        x1 = np.random.random(size=(4, 2))
        x2 = np.random.random(size=(10, 2))
        assert pairwise_rbf_kernel(x1, x2).shape == (4, 10)
        assert pairwise_rbf_kernel(x2, x1).shape == (10, 4)

        x1 = np.random.random(size=(5, 1))
        x2 = np.random.random(size=(11, 1))
        assert pairwise_rbf_kernel(x1, x2).shape == (5, 11)
        assert pairwise_rbf_kernel(x2, x1).shape == (11, 5)

        with pytest.raises(ValueError):
            x1 = np.random.random(size=(1, 2))
            x2 = np.random.random(size=(1, 4))
            # pylint: disable=expression-not-assigned
            pairwise_rbf_kernel(x1, x2).shape

    def test_values(self):
        """Test that values are correct."""
        testing.assert_allclose(
            np.array([[1, np.exp(-0.04)]]),
            pairwise_rbf_kernel(np.array([[0.44]]), np.array([[0.44], [0.84]]), ell=2),
        )

        x1 = np.random.random(size=(4, 2))
        x2 = np.random.random(size=(10, 2))
        assert (
            pairwise_rbf_kernel(x1, x2, ell=2) > pairwise_rbf_kernel(x1, x2, ell=0.5)
        ).all()

    def test_extreme_values(self):
        """Test that we can use extreme values."""

        x1 = np.full((2, 2), self.finfo.max)
        x2 = np.full((2, 2), self.finfo.min)
        with pytest.warns(RuntimeWarning) as warninfo:
            testing.assert_array_equal(
                np.zeros((2, 2), dtype=self.float_dtype), pairwise_rbf_kernel(x1, x2)
            )
        assert "overflow" in warninfo.list[0].message.args[0]
