import unittest
import sym_metanet as metanet


class TestEngines(unittest.TestCase):
    def test_get_current_engine(self):
        self.assertIs(metanet.engines.get_current_engine(), metanet.engine)

    def test_get_available_engines(self):
        engines = metanet.engines.get_available_engines()
        self.assertIsInstance(engines, dict)
        self.assertIsInstance(next(iter(engines.keys())), str)

    def test_use(self):
        old_engine = metanet.engine
        new_engine = metanet.engines.use('casadi')
        self.assertIsNot(metanet.engine, old_engine)
        self.assertIs(metanet.engine, new_engine)


if __name__ == '__main__':
    unittest.main()
