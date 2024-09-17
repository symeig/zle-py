import unittest
import numpy as np
from zlepy.gen import (
    argsort,
    perm_inverse,
    po_groups,
    edges_to_adjacency_list,
    merge_components,
    add_missing_nodes,
    topological_sort,
    longest_path_dag,
    transitive_reduction,
    level_sort,
    invert_dict_of_lists,
    reverse_graph,
    compute_degrees,
    find_max_chain,
    in_out_dict,
    contract_graph
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

    def test_merge_components(self):
        comps = [{0: [1, 2]}, {3: [4, 5]}]
        result = merge_components(comps)
        expected = {0: [1, 2], 3: [4, 5]}
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

    def test_invert_dict_of_lists(self):
        d = {0: [1, 2], 1: [3, 4]}
        result = invert_dict_of_lists(d)
        expected = {1: 0, 2: 0, 3: 1, 4: 1}
        self.assertEqual(result, expected)

    def test_reverse_graph(self):
        graph = {0: [1, 2], 1: [3], 2: [3], 3: []}
        result = reverse_graph(graph)
        expected = {1: [0], 2: [0], 3: [1, 2]}
        self.assertEqual(result, expected)

    def test_compute_degrees(self):
        graph = {0: [1, 2], 1: [3], 2: [3], 3: []}
        result = compute_degrees(graph, 1)
        self.assertEqual(result, [1, 1])

    def test_find_max_chain(self):
        graph = {0: [1], 1: [2], 2: [3], 3: []}
        in_out = {0: [0, 1], 1: [1, 1], 2: [1, 1], 3: [1, 0]}
        result = find_max_chain(0, graph, in_out)
        self.assertEqual(result, [0, 1, 2, 3])

    def test_in_out_dict(self):
        graph = {0: [1, 2], 1: [3], 2: [3], 3: []}
        result = in_out_dict(graph)
        expected = {0: [0, 2], 1: [1, 1], 2: [1, 1], 3: [2, 0]}
        self.assertEqual(result, expected)

    def test_contract_graph(self):
        graph = {0: [1, 2], 1: [3], 2: [3], 3: []}
        result = contract_graph(graph)
        self.assertEqual(len(result), 3)  # Check if it returns 3 items (in_out, chains, block_graph)
        self.assertIsInstance(result[0], dict)
        self.assertIsInstance(result[1], list)
        self.assertIsInstance(result[2], dict)

if __name__ == '__main__':
    unittest.main()