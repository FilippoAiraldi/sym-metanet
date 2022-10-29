import unittest
from importlib.abc import MetaPathFinder


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
        with self.assertWarns(Warning):
            import sym_metanet as metanet

if __name__ == '__main__':
    unittest.main()
