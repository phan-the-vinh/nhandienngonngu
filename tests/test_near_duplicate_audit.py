import sys
import unittest
from pathlib import Path

import numpy as np


AI_DIR = Path(__file__).resolve().parents[1] / "ai"
sys.path.insert(0, str(AI_DIR))

from audit_near_duplicates import UnionFind, _stable_group_id  # noqa: E402


class UnionFindTests(unittest.TestCase):
    def test_transitive_groups_are_split_safe(self):
        groups = UnionFind(5)
        groups.union(0, 1)
        groups.union(1, 2)
        self.assertEqual(groups.find(0), groups.find(2))
        self.assertNotEqual(groups.find(0), groups.find(3))

    def test_group_id_is_stable_under_file_order(self):
        first = _stable_group_id("A", ["b.jpg", "a.jpg"], False)
        second = _stable_group_id("A", ["a.jpg", "b.jpg"], False)
        self.assertEqual(first, second)
        self.assertNotEqual(first, _stable_group_id("B", ["a.jpg", "b.jpg"], False))

    def test_parent_array_starts_as_singletons(self):
        groups = UnionFind(3)
        np.testing.assert_array_equal(groups.parent, np.arange(3))


if __name__ == "__main__":
    unittest.main()
