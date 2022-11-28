import re
import unittest
from typing import Iterable
from unittest.mock import MagicMock

import numpy as np

from sym_metanet import (
    CongestedDestination,
    Destination,
    Link,
    MeteredOnRamp,
    Network,
    Node,
    Origin,
    SimpleMeteredOnRamp,
    engines,
)
from sym_metanet.blocks.base import NO_VARS

engine = engines.use("numpy", var_type="randn")


class TestLinks(unittest.TestCase):
    def test_init_vars__without_inital_condition__creates_vars(self):
        nb_seg = 4
        L = Link[np.ndarray](nb_seg, 3, 1, 180, 30, 100, 1.8)
        L.init_vars()
        self.assertIsNot(L.states, NO_VARS)
        self.assertIs(L.actions, NO_VARS)
        self.assertIs(L.disturbances, NO_VARS)
        for n in ["rho", "v"]:
            self.assertIn(n, L.states)
            self.assertEqual(L.states[n].shape, (nb_seg,))

    def test_init_vars__with_inital_condition__copies_vars(self):
        nb_seg = 4
        L = Link[np.ndarray](nb_seg, 3, 1, 180, 30, 100, 1.8)
        init_conds = {"rho": np.random.rand(nb_seg), "v": np.random.rand(nb_seg)}
        L.init_vars(init_conds)
        self.assertIsNot(L.states, NO_VARS)
        self.assertIs(L.actions, NO_VARS)
        self.assertIs(L.disturbances, NO_VARS)
        for n in ["rho", "v"]:
            self.assertIn(n, L.states)
            self.assertEqual(L.states[n].shape, (nb_seg,))
            np.testing.assert_equal(init_conds[n], L.states[n])


class TestDestinations(unittest.TestCase):
    def test_init_vars__no_value_is_initialized(self):
        D = Destination()
        D.init_vars()
        self.assertIs(D.states, NO_VARS)
        self.assertIs(D.states, NO_VARS)
        self.assertIs(D.disturbances, NO_VARS)


class TestCongestedDestinations(unittest.TestCase):
    def test_init_vars__without_inital_condition__creates_vars(self):
        D = CongestedDestination[np.ndarray]()
        D.init_vars()
        self.assertIs(D.states, NO_VARS)
        self.assertIs(D.states, NO_VARS)
        self.assertIsNot(D.disturbances, NO_VARS)
        self.assertIn("d", D.disturbances)
        self.assertIn(D.disturbances["d"].shape, {(1,), ()})

    def test_init_vars__with_inital_condition__copies_vars(self):
        D = CongestedDestination[np.ndarray]()
        init_conds = {"d": np.random.rand(1)}
        D.init_vars(init_conds)
        self.assertIs(D.states, NO_VARS)
        self.assertIs(D.states, NO_VARS)
        self.assertIsNot(D.disturbances, NO_VARS)
        self.assertIn("d", D.disturbances)
        self.assertIn(D.disturbances["d"].shape, {(1,), ()})
        np.testing.assert_equal(init_conds["d"], D.disturbances["d"])


class TestOrigins(unittest.TestCase):
    def test_init_vars__no_value_is_initialized(self):
        O = Origin()
        O.init_vars()
        self.assertIs(O.states, NO_VARS)
        self.assertIs(O.states, NO_VARS)
        self.assertIs(O.disturbances, NO_VARS)


class TestMeteredOnRamp(unittest.TestCase):
    def test_init_vars__without_inital_condition__creates_vars(self):
        R = MeteredOnRamp[np.ndarray](1e5)
        R.init_vars()
        self.assertIsNot(R.states, NO_VARS)
        self.assertIsNot(R.actions, NO_VARS)
        self.assertIsNot(R.disturbances, NO_VARS)
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
        self.assertIsNot(R.states, NO_VARS)
        self.assertIsNot(R.actions, NO_VARS)
        self.assertIsNot(R.disturbances, NO_VARS)
        for n, d in [("w", R.states), ("r", R.actions), ("d", R.disturbances)]:
            self.assertIn(n, d)
            self.assertIn(d[n].shape, {(1,), ()})
            np.testing.assert_equal(init_conds[n], d[n])


class TestSimpleMeteredOnRamp(unittest.TestCase):
    def test_init_vars__without_inital_condition__creates_vars(self):
        R = SimpleMeteredOnRamp[np.ndarray](1e5)
        R.init_vars()
        self.assertIsNot(R.states, NO_VARS)
        self.assertIsNot(R.actions, NO_VARS)
        self.assertIsNot(R.disturbances, NO_VARS)
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
        self.assertIsNot(R.states, NO_VARS)
        self.assertIsNot(R.actions, NO_VARS)
        self.assertIsNot(R.disturbances, NO_VARS)
        for n, d in [("w", R.states), ("q", R.actions), ("d", R.disturbances)]:
            self.assertIn(n, d)
            self.assertIn(d[n].shape, {(1,), ()})
            np.testing.assert_equal(init_conds[n], d[n])


class TestNetwork(unittest.TestCase):
    def assertAtLeastOneRegexMatch(self, pattern: str, strings: Iterable[str]):
        for string in strings:
            if re.match(pattern, string):
                return
        raise self.failureException(f"no string matches pattern: {pattern}")

    def test_add_node__adds_node_correctly(self):
        node = Node(name="This is a random name")
        net = Network(name="Another random name")
        net.add_node(node)
        self.assertIn(node, net.nodes)
        self.assertIn(node.name, net.nodes_by_name)

    def test_add_nodes__adds_nodes_correctly(self):
        node1 = Node(name="This is a random name1")
        node2 = Node(name="This is a random name2")
        net = Network(name="Another random name")
        net.add_nodes((node1, node2))
        self.assertIn(node1, net.nodes)
        self.assertIn(node2, net.nodes)
        self.assertIn(node1.name, net.nodes_by_name)
        self.assertIn(node1.name, net.nodes_by_name)

    def test_add_link__adds_link_correctly(self):
        upnode = Node(name="N1")
        downnode = Node(name="N2")
        link = Link(4, 3, 1, 180, 30, 100, 1.8, name="L1")
        net = Network(name="Net")
        net.add_nodes((upnode, downnode))
        net.add_link(upnode, link, downnode)
        self.assertIs(link, net.links[upnode, downnode])
        self.assertEqual(len(net.links), 1)
        self.assertIn(link.name, net.links_by_name)
        self.assertIn(link, net.nodes_by_link)
        self.assertEqual(net.nodes_by_link[link], (upnode, downnode))

    def test_add_links__adds_links_correctly(self):
        upnode1 = Node(name="N11")
        downnode1 = Node(name="N12")
        link1 = Link(4, 3, 1, 180, 30, 100, 1.8, name="L1")
        upnode2 = Node(name="N21")
        downnode2 = Node(name="N21")
        link2 = Link(4, 3, 1, 180, 30, 100, 1.8, name="L2")
        net = Network(name="Net")
        net.add_nodes((upnode1, downnode1, upnode2, downnode2))
        net.add_links(((upnode1, link1, downnode1), (upnode2, link2, downnode2)))
        self.assertEqual(len(net.links), 2)
        for data in ((link1, upnode1, downnode1), (link2, upnode2, downnode2)):
            link, nodeup, nodedown = data
            self.assertEqual(net.nodes_by_link[link], (nodeup, nodedown))
            self.assertIs(link, net.links[nodeup, nodedown])
            self.assertIn(link.name, net.links_by_name)
            self.assertIn(link, net.nodes_by_link)

    def test_add_path__adds_nodes_and_links_correctly(self):
        N1 = Node(name="N1")
        N2 = Node(name="N2")
        N3 = Node(name="N3")
        L1 = Link(4, 3, 1, 180, 30, 100, 1.8, name="L1")
        L2 = Link(4, 3, 1, 180, 30, 100, 1.8, name="L2")
        L3 = Link(4, 3, 1, 180, 30, 100, 1.8, name="L3")
        O1 = Origin(name="23423")
        D1 = Destination(name="23421")
        net = Network(name="Net")
        net.add_path(origin=O1, path=(N1, L1, N2), destination=D1)
        net.add_path(origin=O1, path=(N1, L2, N3, L3, N2), destination=D1)
        for n in (N1, N2, N3):
            self.assertIn(n, net.nodes)
        self.assertEqual(len(net.links), 3)
        self.assertIs(L1, net.links[N1, N2])
        self.assertIs(L2, net.links[N1, N3])
        self.assertIs(L3, net.links[N3, N2])
        self.assertIn(O1, net.origins)
        self.assertIn(D1, net.destinations)

    def test_add_path__raises__with_single_node(self):
        net = Network(name="Net")
        with self.assertRaises(ValueError):
            net.add_path(path=(Node(name="N1"),))

    def test_add_origin__adds_origin_correctly(self):
        node = Node(name="This is a random name")
        origin = Origin(name="23423")
        net = Network(name="Another random name")
        net.add_node(node)
        net.add_origin(origin, node)
        self.assertIn(origin, net.origins)
        self.assertIs(net.origins[origin], node)
        self.assertIn(origin.name, net.origins_by_name)
        self.assertIs(net.origins_by_name[origin.name], origin)
        self.assertIn(node, net.origins_by_node)
        self.assertIs(net.origins_by_node[node], origin)

    def test_add_destination__adds_destination_correctly(self):
        node = Node(name="This is a random name")
        destination = Destination(name="23423")
        net = Network(name="Another random name")
        net.add_node(node)
        net.add_destination(destination, node)
        self.assertIn(destination, net.destinations)
        self.assertIs(net.destinations[destination], node)
        self.assertIn(destination.name, net.destinations_by_name)
        self.assertIs(net.destinations_by_name[destination.name], destination)
        self.assertIn(node, net.destinations_by_node)
        self.assertIs(net.destinations_by_node[node], destination)

    def test_out_links__gets_correct_outward_links(self):
        N1 = Node(name="N1")
        N2 = Node(name="N2")
        N3 = Node(name="N3")
        L1 = Link(4, 3, 1, 180, 30, 100, 1.8, name="L1")
        L2 = Link(4, 3, 1, 180, 30, 100, 1.8, name="L2")
        net = Network(name=".Net")
        net.add_links(((N1, L1, N2), (N1, L2, N3)))
        self.assertEqual({(N1, N3, L2), (N1, N2, L1)}, set(net.out_links(N1)))

    def test_in_links__gets_correct_inward_links(self):
        N1 = Node(name="N1")
        N2 = Node(name="N2")
        N3 = Node(name="N3")
        L1 = Link(4, 3, 1, 180, 30, 100, 1.8, name="L1")
        L2 = Link(4, 3, 1, 180, 30, 100, 1.8, name="L2")
        net = Network(name=".Net")
        net.add_links(((N1, L1, N2), (N3, L2, N2)))
        self.assertEqual({(N1, N2, L1), (N3, N2, L2)}, set(net.in_links(N2)))

    def test_is_valid__raises__condition_1a(self):
        # duplicate link
        N1 = Node(name="N1")
        N2 = Node(name="N2")
        N3 = Node(name="N3")
        L = Link(4, 3, 1, 180, 30, 100, 1.8, name="L1")
        net = Network(name=".Net")
        net.add_nodes((N1, N2, N3))
        net.add_link(N1, L, N2)
        net.add_link(N1, L, N3)
        ok, msgs = net.is_valid(raises=False)
        self.assertFalse(ok)
        self.assertAtLeastOneRegexMatch(
            "Element L1 is duplicated in the network.", msgs
        )

    def test_is_valid__raises__condition_1b(self):
        # duplicate origin
        N1 = Node(name="N1")
        N2 = Node(name="N2")
        net = Network(name=".Net")
        O = MeteredOnRamp(100, name="O")
        net.add_origin(O, N1)
        net.add_origin(O, N2)
        ok, msgs = net.is_valid(raises=False)
        self.assertFalse(ok)
        self.assertAtLeastOneRegexMatch("Element O is duplicated in the network.", msgs)

    def test_is_valid__raises__condition_1c(self):
        # duplicate destination
        N1 = Node(name="N1")
        N2 = Node(name="N2")
        net = Network(name=".Net")
        D = Destination(name="D")
        net.add_destination(D, N1)
        net.add_destination(D, N2)
        ok, msgs = net.is_valid(raises=False)
        self.assertFalse(ok)
        self.assertAtLeastOneRegexMatch("Element D is duplicated in the network.", msgs)

    def test_is_valid__raises__condition2(self):
        N = Node(name="N")
        O = MeteredOnRamp(100, name="23423")
        D = Destination(name="23423")
        net = Network(name=".Net")
        net.add_node(N)
        net.add_origin(O, N)
        net.add_destination(D, N)
        ok, msgs = net.is_valid(raises=False)
        self.assertFalse(ok)
        self.assertAtLeastOneRegexMatch(
            "Node N must either have an origin or a destination, but not " "both.", msgs
        )

    def test_is_valid__raises__condition_3(self):
        net = Network(name=".Net")
        net.add_node(Node(name="N"))
        ok, msgs = net.is_valid(raises=False)
        self.assertFalse(ok)
        self.assertAtLeastOneRegexMatch("Node N is connected to no link.", msgs)

    def test_is_valid__raises__condition_4_and_5(self):
        N1 = Node(name="N1")
        N2 = Node(name="N2")
        L = Link(4, 3, 1, 180, 30, 100, 1.8, name="L1")
        net = Network(name=".Net that does not raise")
        net.add_link(N1, L, N2)
        ok, msgs = net.is_valid(raises=False)
        self.assertFalse(ok)
        self.assertAtLeastOneRegexMatch(
            "Node N1 has neither any entering links nor an origin.", msgs
        )
        self.assertAtLeastOneRegexMatch(
            "Node N2 has neither any exiting links nor a destination.", msgs
        )

    def test_is_valid__raises__condition_6(self):
        N1 = Node(name="N1")
        N2 = Node(name="N2")
        N3 = Node(name="N3")
        L1 = Link(4, 3, 1, 180, 30, 100, 1.8, name="L1")
        L2 = Link(4, 3, 1, 180, 30, 100, 1.8, name="L3")
        O = Origin(name="O")
        net = Network(name=".Net")
        net.add_nodes((N1, N2, N3))
        net.add_link(N1, L1, N2)
        net.add_link(N2, L2, N3)
        net.add_origin(O, N2)
        ok, msgs = net.is_valid(raises=False)
        self.assertFalse(ok)
        self.assertAtLeastOneRegexMatch(
            r"Expected node N2 to have no entering links, as it is connected "
            r"to origin O \(only ramps support entering links\).",
            msgs,
        )

    def test_is_valid__raises__condition_7(self):
        N1 = Node(name="N1")
        N2 = Node(name="N2")
        N3 = Node(name="N3")
        L1 = Link(4, 3, 1, 180, 30, 100, 1.8, name="L1")
        L2 = Link(4, 3, 1, 180, 30, 100, 1.8, name="L3")
        O = Origin(name="O")
        net = Network(name=".Net")
        net.add_nodes((N1, N2, N3))
        net.add_link(N1, L1, N2)
        net.add_link(N1, L2, N3)
        net.add_origin(O, N1)
        ok, msgs = net.is_valid(raises=False)
        self.assertFalse(ok)
        self.assertAtLeastOneRegexMatch(
            "Expected node N1 to have at most one exiting link, as it is "
            "connected to origin O.",
            msgs,
        )

    def test_is_valid__raises__condition_8(self):
        N1 = Node(name="N1")
        N2 = Node(name="N2")
        N3 = Node(name="N3")
        L1 = Link(4, 3, 1, 180, 30, 100, 1.8, name="L1")
        L2 = Link(4, 3, 1, 180, 30, 100, 1.8, name="L3")
        D = Destination(name="D")
        net = Network(name=".Net")
        net.add_nodes((N1, N2, N3))
        net.add_link(N1, L1, N3)
        net.add_link(N2, L2, N3)
        net.add_destination(D, N3)
        ok, msgs = net.is_valid(raises=False)
        self.assertFalse(ok)
        self.assertAtLeastOneRegexMatch(
            "Expected node N3 to have at most one entering link, as it is "
            "connected to destination D.",
            msgs,
        )

    def test_is_valid__raises__condition_9(self):
        N1 = Node(name="N1")
        N2 = Node(name="N2")
        L = Link(4, 3, 1, 180, 30, 100, 1.8, name="L1")
        net = Network(name=".Net that does not raise")
        D = Destination(name="D")
        net.add_link(N1, L, N2)
        net.add_destination(D, N1)
        ok, msgs = net.is_valid(raises=False)
        self.assertFalse(ok)
        self.assertAtLeastOneRegexMatch(
            "Expected node N1 to have no exiting links, as it is connected "
            "to destination D.",
            msgs,
        )

    def test_step__calls_init_vars_and_step_in_elements(self):
        L = 1
        lanes = 2
        C = (3500, 2000)
        tau = 18 / 3600
        kappa = 40
        eta = 60
        rho_max = 180
        delta = 0.0122
        T = 10 / 3600
        a = 1.867
        v_free = 102
        rho_crit = 33.5
        N1 = Node(name="N1")
        N2 = Node(name="N2")
        N3 = Node(name="N3")
        L1 = Link(2, lanes, L, rho_max, rho_crit, v_free, a, name="L1")
        L2 = Link(1, lanes, L, rho_max, rho_crit, v_free, a, name="L2")
        O1 = MeteredOnRamp(C[0], name="O1")
        O2 = SimpleMeteredOnRamp(C[1], name="O2")
        D1 = CongestedDestination(name="D1")
        net = (
            Network(name="A1")
            .add_path(origin=O1, path=(N1, L1, N2, L2, N3), destination=D1)
            .add_origin(O2, N2)
        )

        self.assertTrue(net.is_valid(raises=False)[0])

        for el in net.elements:
            el.init_vars = MagicMock(return_value=None)
            el.step = MagicMock(return_value=None)

        net.step(T=T, tau=tau, eta=eta, kappa=kappa, delta=delta)

        for _, _, link in net.links:
            link.init_vars.assert_called_once()
            link.step.assert_called_once()
        for origin in net.origins:
            origin.init_vars.assert_called_once()
            origin.step.assert_called_once()
        for destination in net.destinations:
            destination.init_vars.assert_called_once()
            destination.step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
