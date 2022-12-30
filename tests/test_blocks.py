import unittest

import numpy as np

from sym_metanet import (
    CongestedDestination,
    Destination,
    Link,
    MeteredOnRamp,
    Origin,
    SimpleMeteredOnRamp,
    engines,
)

engine = engines.use("numpy", var_type="randn")


class TestLinks(unittest.TestCase):
    def test_init_vars__without_inital_condition__creates_vars(self):
        nb_seg = 4
        L = Link[np.ndarray](nb_seg, 3, 1, 180, 30, 100, 1.8)
        L.init_vars()
        self.assertIsNot(L.states, None)
        self.assertIs(L.actions, None)
        self.assertIs(L.disturbances, None)
        for n in ["rho", "v"]:
            self.assertIn(n, L.states)
            self.assertEqual(L.states[n].shape, (nb_seg,))

    def test_init_vars__with_inital_condition__copies_vars(self):
        nb_seg = 4
        L = Link[np.ndarray](nb_seg, 3, 1, 180, 30, 100, 1.8)
        init_conds = {"rho": np.random.rand(nb_seg), "v": np.random.rand(nb_seg)}
        L.init_vars(init_conds)
        self.assertIsNot(L.states, None)
        self.assertIs(L.actions, None)
        self.assertIs(L.disturbances, None)
        for n in ["rho", "v"]:
            self.assertIn(n, L.states)
            self.assertEqual(L.states[n].shape, (nb_seg,))
            np.testing.assert_equal(init_conds[n], L.states[n])


class TestDestinations(unittest.TestCase):
    def test_init_vars__no_value_is_initialized(self):
        D = Destination()
        D.init_vars()
        self.assertIs(D.states, None)
        self.assertIs(D.actions, None)
        self.assertIs(D.disturbances, None)


class TestCongestedDestinations(unittest.TestCase):
    def test_init_vars__without_inital_condition__creates_vars(self):
        # sourcery skip: class-extract-method
        D = CongestedDestination[np.ndarray]()
        D.init_vars()
        self.assertIs(D.states, None)
        self.assertIs(D.actions, None)
        self.assertIsNot(D.disturbances, None)
        self.assertIn("d", D.disturbances)
        self.assertIn(D.disturbances["d"].shape, {(1,), ()})

    def test_init_vars__with_inital_condition__copies_vars(self):
        D = CongestedDestination[np.ndarray]()
        init_conds = {"d": np.random.rand(1)}
        D.init_vars(init_conds)
        self.assertIs(D.states, None)
        self.assertIs(D.actions, None)
        self.assertIsNot(D.disturbances, None)
        self.assertIn("d", D.disturbances)
        self.assertIn(D.disturbances["d"].shape, {(1,), ()})
        np.testing.assert_equal(init_conds["d"], D.disturbances["d"])


class TestOrigins(unittest.TestCase):
    def test_init_vars__no_value_is_initialized(self):
        origin = Origin()
        origin.init_vars()
        self.assertIs(origin.states, None)
        self.assertIs(origin.actions, None)
        self.assertIs(origin.disturbances, None)


class TestMeteredOnRamp(unittest.TestCase):
    def test_init_vars__without_inital_condition__creates_vars(self):
        R = MeteredOnRamp[np.ndarray](1e5)
        R.init_vars()
        self.assertIsNot(R.states, None)
        self.assertIsNot(R.actions, None)
        self.assertIsNot(R.disturbances, None)
        for n, d in [("w", R.states), ("r", R.actions), ("d", R.disturbances)]:
            self.assertIn(n, d)
            self.assertIn(d[n].shape, {(1,), ()})

    def test_init_vars__with_inital_condition__copies_vars(self):
        R = MeteredOnRamp[np.ndarray](1e5)
        init_conds = {
            "w": np.random.rand(1),
            "r": np.random.rand(1),
            "d": np.random.rand(1),
        }
        R.init_vars(init_conds)
        self.assertIsNot(R.states, None)
        self.assertIsNot(R.actions, None)
        self.assertIsNot(R.disturbances, None)
        for n, d in [("w", R.states), ("r", R.actions), ("d", R.disturbances)]:
            self.assertIn(n, d)
            self.assertIn(d[n].shape, {(1,), ()})
            np.testing.assert_equal(init_conds[n], d[n])


class TestSimpleMeteredOnRamp(unittest.TestCase):
    def test_init_vars__without_inital_condition__creates_vars(self):
        R = SimpleMeteredOnRamp[np.ndarray](1e5)
        R.init_vars()
        self.assertIsNot(R.states, None)
        self.assertIsNot(R.actions, None)
        self.assertIsNot(R.disturbances, None)
        for n, d in [("w", R.states), ("q", R.actions), ("d", R.disturbances)]:
            self.assertIn(n, d)
            self.assertIn(d[n].shape, {(1,), ()})

    def test_init_vars__with_inital_condition__copies_vars(self):
        R = SimpleMeteredOnRamp[np.ndarray](1e5)
        init_conds = {
            "w": np.random.rand(1),
            "q": np.random.rand(1),
            "d": np.random.rand(1),
        }
        R.init_vars(init_conds)
        self.assertIsNot(R.states, None)
        self.assertIsNot(R.actions, None)
        self.assertIsNot(R.disturbances, None)
        for n, d in [("w", R.states), ("q", R.actions), ("d", R.disturbances)]:
            self.assertIn(n, d)
            self.assertIn(d[n].shape, {(1,), ()})
            np.testing.assert_equal(init_conds[n], d[n])


if __name__ == "__main__":
    unittest.main()
