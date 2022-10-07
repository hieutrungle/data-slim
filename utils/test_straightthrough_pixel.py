# import tensorflow as tf
import unittest
import numpy as np
from utils import straight_through_pixels


def main():
    pass


class test_compare_replace(unittest.TestCase):

    def test_floats(self):
        x = np.array([0.63696169, -0.26978671, 0.04097352, 0.01652764, 0.81327024, 0.91275558,
                      0.60663578, 0.72949656, 0.54362499, 0.93507242])
        x_hat = np.array([0.81585355, -0.0027385,  -0.85740428, 0.03358558, 0.72965545, 0.17565562,
                          0.86317892, 0.54146122, 0.29971189, 0.42268722])
        x_hat = straight_through_pixels.compare_replace(
            x, x_hat, tolerance=0.5)
        self.assertEqual(straight_through_pixels.get_unsatisfied_values_indices(
            x, x_hat, tolerance=0.5)[1], [[]], "Should be empty")

    def test_random(self):
        rng = np.random.default_rng(1)
        len_i = rng.integers(0, 10)
        len_j = rng.integers(0, 1000)
        len_k = rng.integers(0, 1000)
        len_t = rng.integers(0, 5)
        x = rng.random((len_i, len_j, len_k, len_t))
        x_hat = rng.random((len_i, len_j, len_k, len_t))
        x_hat = straight_through_pixels.compare_replace(
            x, x_hat, tolerance=0.5)
        self.assertEqual(straight_through_pixels.get_unsatisfied_values_indices(
            x, x_hat, tolerance=0.5)[1], [[]], "Should be empty")


class test_get_unsatisfied_indices(unittest.TestCase):

    def test_integers(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        x_hat = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(straight_through_pixels.get_unsatisfied_values_indices(
            x, x_hat)[1], [[]], "Should be empty")

    def test_floats(self):

        x = [0.63696169, -0.26978671, 0.04097352, 0.01652764, 0.81327024, 0.91275558,
             0.60663578, 0.72949656, 0.54362499, 0.93507242]
        x_hat = [0.81585355, -0.0027385,  -0.85740428, 0.03358558, 0.72965545, 0.17565562,
                 0.86317892, 0.54146122, 0.29971189, 0.42268722]
        self.assertEqual(straight_through_pixels.get_unsatisfied_values_indices(
            x, x_hat, tolerance=0.5)[1], [[2, 5, 9]], "Should be [2, 5, 9]")

    def test_random(self):
        rng = np.random.default_rng(12345)
        x = rng.random(1000)
        x_hat = rng.random(1000)
        indices = []
        tolerance = 0.5
        for i in range(len(x)):
            if abs(x[i] - x_hat[i]) > tolerance:
                indices.append(i)
        self.assertEqual(straight_through_pixels.get_unsatisfied_values_indices(
            x, x_hat, tolerance=tolerance)[1], [indices])

    def test_2d(self):
        rng = np.random.default_rng(12345)
        x = rng.random((10, 2))
        x_hat = rng.random((10, 2))
        indices = []
        tolerance = 0.5
        for i in range(len(x)):
            for j in range(len(x[0])):
                if abs(x[i][j] - x_hat[i][j]) > tolerance:
                    indices.append(i*len(x[0]) + j)
        self.assertEqual(straight_through_pixels.get_unsatisfied_values_indices(
            x, x_hat, tolerance=tolerance)[1], [indices])


class test_reverse_flatten_indices(unittest.TestCase):

    def test_random(self):
        rng = np.random.default_rng(1)
        len_i = 100
        len_j = 100
        len_k = 32
        len_t = 123
        x = rng.random((len_i, len_j, len_k, len_t))
        x_flatten = x.flatten()
        flatten_indices = 321
        full_index = straight_through_pixels.reverse_flatten_indices(
            x.shape, flatten_indices)
        self.assertEqual(x_flatten[flatten_indices], x[full_index])


if __name__ == '__main__':
    unittest.main()
