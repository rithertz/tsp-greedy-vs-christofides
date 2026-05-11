import unittest

import numpy as np

from christofides import christofides_tsp
from exact import held_karp_tsp
from generate import distance_matrix
from greedy import greedy_tsp
from metrics import mst_lower_bound


class TspAlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )
        self.dist_matrix = distance_matrix(self.points)

    def assert_valid_tour(self, tour, expected_nodes):
        self.assertEqual(tour[0], 0)
        self.assertEqual(tour[-1], 0)
        self.assertEqual(set(tour[:-1]), expected_nodes)
        self.assertEqual(len(tour), len(expected_nodes) + 1)

    def test_held_karp_reconstructs_hamiltonian_cycle(self):
        tour, length = held_karp_tsp(self.dist_matrix)

        self.assert_valid_tour(tour, {0, 1, 2, 3})
        self.assertAlmostEqual(length, 4.0)

    def test_held_karp_handles_single_node(self):
        tour, length = held_karp_tsp(np.array([[0.0]]))

        self.assertEqual(tour, [0, 0])
        self.assertEqual(length, 0.0)

    def test_christofides_handles_single_node(self):
        tour, length = christofides_tsp(np.array([[0.0]]))

        self.assertEqual(tour, [0, 0])
        self.assertEqual(length, 0.0)

    def test_greedy_handles_empty_instance(self):
        tour, length = greedy_tsp(np.zeros((0, 0)))

        self.assertEqual(tour, [])
        self.assertEqual(length, 0.0)

    def test_greedy_rejects_invalid_start_node(self):
        with self.assertRaises(ValueError):
            greedy_tsp(self.dist_matrix, start=10)

    def test_greedy_returns_valid_tour(self):
        tour, length = greedy_tsp(self.dist_matrix)

        self.assert_valid_tour(tour, {0, 1, 2, 3})
        self.assertGreaterEqual(length, 4.0)

    def test_mst_is_lower_bound_for_exact_tour(self):
        _, optimal_length = held_karp_tsp(self.dist_matrix)

        self.assertLessEqual(mst_lower_bound(self.dist_matrix), optimal_length)


if __name__ == "__main__":
    unittest.main()
