import unittest
import numpy as np
from zlepy.gen import (
    argsort,
    perm_inverse,
    po_groups,
    edges_to_adjacency_list,
    add_missing_nodes,
    topological_sort,
    longest_path_dag,
    transitive_reduction,
    level_sort,
    reverse_graph,
)

class TestGen(unittest.TestCase):

    def test_argsort(self):
        seq = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        result = argsort(seq)
        self.assertEqual(result, [1, 3, 6, 0, 9, 2, 4, 8, 10, 7, 5])

    def test_perm_inverse(self):
        perm = np.array([2, 0, 1])
        result = perm_inverse(perm)
        np.testing.assert_array_equal(result, np.array([1, 2, 0]))

    def test_po_groups(self):
        po = np.array([0, 1, 1, 2, 2, 3, 3, 4])
        result = po_groups(po)
        expected = np.array([[0, 1, 1, 2, 2, 3, 3, 4]])
        self.assertTrue(np.array_equal(result, expected))

    def test_edges_to_adjacency_list(self):
        edges = [0, 1, 1, 2, 2, 3]
        result = edges_to_adjacency_list(edges)
        expected = {0: [1], 1: [2], 2: [3], 3: []}
        self.assertEqual(result, expected)

    def test_add_missing_nodes(self):
        edge_list = {0: [1, 2], 1: [3]}
        result = add_missing_nodes(edge_list)
        expected = {0: [1, 2], 1: [3], 2: [], 3: []}
        self.assertEqual(result, expected)

    def test_topological_sort(self):
        graph = {0: [1, 2], 1: [3], 2: [3], 3: []}
        result = topological_sort(graph)
        self.assertEqual(result, [0, 1, 2, 3])

    def test_longest_path_dag(self):
        graph = {0: [1, 2], 1: [3], 2: [3], 3: []}
        result = longest_path_dag(graph)
        expected = {0: 0, 1: 1, 2: 1, 3: 2}
        self.assertEqual(result, expected)

    def test_transitive_reduction(self):
        graph = {0: [1, 2, 3], 1: [3], 2: [3], 3: []}
        result = transitive_reduction(graph)
        expected = {0: [1, 2], 1: [3], 2: [3], 3: []}
        self.assertEqual(result, expected)

    def test_level_sort(self):
        graph = {0: [1, 2], 1: [3], 2: [3], 3: []}
        result = level_sort(graph)
        expected = {0: [0], 1: [1, 2], 2: [3]}
        self.assertEqual(result, expected)

    def test_reverse_graph(self):
        graph = {0: [1, 2], 1: [3], 2: [3], 3: []}
        result = reverse_graph(graph)
        expected = {1: [0], 2: [0], 3: [1, 2]}
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()