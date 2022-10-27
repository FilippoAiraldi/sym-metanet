import unittest
import numpy as np
from pymetanet.util import NamedObject, as1darray


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


if __name__ == '__main__':
    unittest.main()
