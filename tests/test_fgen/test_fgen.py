import unittest
from zlepy import get_po, rebuildlists, argsort, perm_inverse

class TestFgenGen(unittest.TestCase):
    def test_get_po(self):
        result = get_po([5])
        self.assertIsInstance(result, list)

    def test_rebuildlists(self):
        result = rebuildlists(3)
        self.assertIsInstance(result, list)

    def test_argsort(self):
        result = argsort([3, 1, 2])
        self.assertEqual(result, [1, 2, 0])

    def test_perm_inverse(self):
        import numpy as np
        perm = np.array([2, 0, 1])
        result = perm_inverse(perm)
        np.testing.assert_array_equal(result, np.array([1, 2, 0]))


    # TODO add test for genmat


if __name__ == '__main__':
    unittest.main()