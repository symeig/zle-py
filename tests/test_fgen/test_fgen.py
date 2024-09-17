import unittest
from zlepy.fgen import get_po, rebuildlists

class TestFgenGen(unittest.TestCase):
    def test_get_po(self):
        result = get_po([5])
        self.assertIsInstance(result, list)

    def test_rebuildlists(self):
        result = rebuildlists(3)
        self.assertIsInstance(result, list)

    # TODO add test for genmat


if __name__ == '__main__':
    unittest.main()