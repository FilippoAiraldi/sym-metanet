import unittest
import warnings
import sys
sys.path.insert(1, 'src')


class TestImport(unittest.TestCase):
    def test_import__warns__when_no_import_succeeds(self):
        import sys
        for module in ('casadi', 'numpy'):
            sys.modules[module] = None
        with warnings.catch_warnings(record=True) as w:
            import sym_metanet as metanet

        from sym_metanet.errors import EngineNotFoundWarning
        self.assertFalse(hasattr(metanet, 'engine'))
        self.assertEqual(len(w), 1)
        self.assertIs(w[0].category, EngineNotFoundWarning)


if __name__ == '__main__':
    unittest.main()
