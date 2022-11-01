import unittest
from importlib.abc import MetaPathFinder
import warnings


class ForbiddenModules(MetaPathFinder):
    '''Prevents some modules from being loaded.'''

    def __init__(self, modules):
        super().__init__()
        self.modules = modules

    def find_spec(self, fullname, path, target=None):
        if fullname in self.modules:
            raise ImportError(fullname)


class TestImport(unittest.TestCase):
    def test_import__warns__when_no_import_succeeds(self):
        forbidden_modules = {'casadi'}
        import sys
        sys.meta_path.insert(0, ForbiddenModules(forbidden_modules))
        with warnings.catch_warnings(record=True) as w:
            import sym_metanet as metanet

        # can only perform this import after the warning-raising import
        from sym_metanet.errors import EngineNotFoundWarning
        self.assertFalse(hasattr(metanet, 'engine'))
        self.assertEqual(len(w), 1)
        self.assertIs(w[0].category, EngineNotFoundWarning)


if __name__ == '__main__':
    unittest.main()
