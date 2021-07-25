"""Tests for the utils module."""
import numpy as np
from mlapp_models.utils import sigmoid
from numpy import testing


class TestSigmoid:
    """Test the Sigmoid function."""

    def test_values(self):
        """Test easy inputs."""
        testing.assert_array_equal(np.array([0.5]), sigmoid(np.array([0.0])))

        inputs = np.array([-10, 20, 0.001, -1])
        expected_output = 1 / (1 + np.exp(-inputs))
        testing.assert_array_equal(expected_output, sigmoid(inputs))

    def test_extreme_values(self):
        """Test that we can input extreme values without numerical errors."""
        dtype = np.float64
        max_value = np.finfo(dtype).max
        inputs = np.array([max_value, -max_value])
        testing.assert_array_equal(np.array([1, 0], dtype=dtype), sigmoid(inputs))
