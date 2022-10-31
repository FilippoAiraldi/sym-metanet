import unittest
from functools import cached_property
from sym_metanet.util import cached_property_clearer


class DummyWithCachedProperty:
    def __init__(self) -> None:
        self.counter = 0

    @cached_property
    def a_cached_property(self) -> int:
        self.counter += 1
        return self.counter

    @cached_property_clearer(a_cached_property)
    def clear_cache(self) -> None:
        return


class TestUtil(unittest.TestCase):
    def test_cached_property_clearer(self):
        dummy = DummyWithCachedProperty()
        dummy.a_cached_property
        dummy.a_cached_property
        self.assertEqual(dummy.counter, 1)
        dummy.clear_cache()
        dummy.a_cached_property
        self.assertEqual(dummy.counter, 2)
        dummy.clear_cache()
        dummy.a_cached_property
        dummy.a_cached_property
        self.assertEqual(dummy.counter, 3)


if __name__ == '__main__':
    unittest.main()
