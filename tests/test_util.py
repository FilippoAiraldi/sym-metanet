import unittest
import numpy as np
from functools import cached_property
from pymetanet.util import (
    NamedObject,
    as1darray,
    cached_property_clearer
)


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
    def test_NamedObject(self):
        name = 'This is a random name'
        obj = NamedObject(name=name)
        self.assertEqual(name, obj.name)

    def test_as1darray(self):
        inputs = [
            (1, 42),
            (3, 42),
            (3, [42, 42, 42]),
            (3, (42, 42, 42)),
            (3, (42.0, 42.0, 42.0)),
            (3, np.array((42, 42, 42))),
        ]
        for input in inputs:
            n, x = input
            y = as1darray(x, n)
            np.testing.assert_equal(x, y)
            self.assertEqual(y.ndim, 1)
            self.assertEqual(y.size, n)

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
