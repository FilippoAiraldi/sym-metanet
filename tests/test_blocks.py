import unittest
from unittest.mock import MagicMock
from typing import List
import numpy as np
from sym_metanet.blocks.base import NO_VARS, ElementBase
from sym_metanet import (
    Node,
    Link,
    Origin,
    MeteredOnRamp,
    SimpleMeteredOnRamp,
    Network,
    Destination,
    CongestedDestination,
    InvalidNetworkError,
)
from sym_metanet import engines


engine = engines.use('numpy', var_type='randn')


class TestNodes(unittest.TestCase):
    def test_init_vars__raises(self):
        N = Node()
        with self.assertRaises(RuntimeError):
            N.init_vars()

    def test_step__raises(self):
        N = Node()
        with self.assertRaises(RuntimeError):
            N.step()


class TestLinks(unittest.TestCase):
    def test_init_vars__without_inital_condition__creates_vars(self):
        nb_seg = 4
        L = Link[np.ndarray](nb_seg, 3, 1, 100, 30, 1.8)
        L.init_vars()
        self.assertIsNot(L.vars, NO_VARS)
        for n in ['rho', 'v']:
            self.assertIn(n, L.vars)
            self.assertEqual(L.vars[n].shape, (nb_seg, 1))

    def test_init_vars__with_inital_condition__copies_vars(self):
        nb_seg = 4
        L = Link[np.ndarray](nb_seg, 3, 1, 100, 30, 1.8)
        init_conds = {
            'rho': np.random.rand(nb_seg, 1),
            'v': np.random.rand(nb_seg, 1)
        }
        L.init_vars(init_conds)
        self.assertIsNot(L, NO_VARS)
        for n in ['rho', 'v']:
            self.assertIn(n, L.vars)
            self.assertEqual(L.vars[n].shape, (nb_seg, 1))
            np.testing.assert_equal(init_conds[n], L.vars[n])


class TestDestinations(unittest.TestCase):
    def test_init_vars__no_value_is_initialized(self):
        D = Destination()
        D.init_vars()
        self.assertEqual(D.vars, NO_VARS)


class TestCongestedDestinations(unittest.TestCase):
    def test_init_vars__without_inital_condition__creates_vars(self):
        D = CongestedDestination[np.ndarray]()
        D.init_vars()
        self.assertIsNot(D.vars, NO_VARS)
        self.assertIn('d', D.vars)
        self.assertIn(D.vars['d'].shape, {(1, 1), (1,), ()})

    def test_init_vars__with_inital_condition__copies_vars(self):
        D = CongestedDestination[np.ndarray]()
        init_conds = {'d': np.random.rand(1, 1)}
        D.init_vars(init_conds)
        self.assertIsNot(D, NO_VARS)
        self.assertIn('d', D.vars)
        self.assertIn(D.vars['d'].shape, {(1, 1), (1,), ()})
        np.testing.assert_equal(init_conds['d'], D.vars['d'])


class TestOrigins(unittest.TestCase):
    def test_init_vars__no_value_is_initialized(self):
        O = Origin()
        O.init_vars()
        self.assertEqual(O.vars, NO_VARS)


class TestMeteredOnRamp(unittest.TestCase):
    def test_init_vars__without_inital_condition__creates_vars(self):
        R = MeteredOnRamp[np.ndarray](1e5)
        R.init_vars()
        self.assertIsNot(R.vars, NO_VARS)
        for n in ['w', 'r', 'd']:
            self.assertIn(n, R.vars)
            self.assertIn(R.vars[n].shape, {(1, 1), (1,), ()})

    def test_init_vars__with_inital_condition__copies_vars(self):
        R = MeteredOnRamp[np.ndarray](1e5)
        init_conds = {
            'w': np.random.rand(1, 1),
            'r': np.random.rand(1, 1),
            'd': np.random.rand(1, 1)
        }
        R.init_vars(init_conds)
        self.assertIsNot(R, NO_VARS)
        for n in ['w', 'r', 'd']:
            self.assertIn(n, R.vars)
            self.assertIn(R.vars[n].shape, {(1, 1), (1,), ()})
            np.testing.assert_equal(init_conds[n], R.vars[n])


class TestSimpleMeteredOnRamp(unittest.TestCase):
    def test_init_vars__without_inital_condition__creates_vars(self):
        R = SimpleMeteredOnRamp[np.ndarray](1e5)
        R.init_vars()
        self.assertIsNot(R.vars, NO_VARS)
        for n in ['w', 'q', 'd']:
            self.assertIn(n, R.vars)
            self.assertIn(R.vars[n].shape, {(1, 1), (1,), ()})

    def test_init_vars__with_inital_condition__copies_vars(self):
        R = SimpleMeteredOnRamp[np.ndarray](1e5)
        init_conds = {
            'w': np.random.rand(1, 1),
            'q': np.random.rand(1, 1),
            'd': np.random.rand(1, 1)
        }
        R.init_vars(init_conds)
        self.assertIsNot(R, NO_VARS)
        for n in ['w', 'q', 'd']:
            self.assertIn(n, R.vars)
            self.assertIn(R.vars[n].shape, {(1, 1), (1,), ()})
            np.testing.assert_equal(init_conds[n], R.vars[n])


class TestNetwork(unittest.TestCase):
    def test_add_node__adds_node_correctly(self):
        node = Node(name='This is a random name')
        net = Network(name='Another random name')
        net.add_node(node)
        self.assertIn(node, net.nodes)
        self.assertIn(node.name, net.nodes_by_name)

    def test_add_nodes__adds_nodes_correctly(self):
        node1 = Node(name='This is a random name1')
        node2 = Node(name='This is a random name2')
        net = Network(name='Another random name')
        net.add_nodes((node1, node2))
        self.assertIn(node1, net.nodes)
        self.assertIn(node2, net.nodes)
        self.assertIn(node1.name, net.nodes_by_name)
        self.assertIn(node1.name, net.nodes_by_name)

    def test_add_link__adds_link_correctly(self):
        upnode = Node(name='N1')
        downnode = Node(name='N2')
        link = Link(4, 3, 1, 100, 30, 1.8, name='L1')
        net = Network(name='Net')
        net.add_nodes((upnode, downnode))
        net.add_link(upnode, link, downnode)
        self.assertIs(link, net.links[upnode, downnode])
        self.assertEqual(len(net.links), 1)
        self.assertIn(link.name, net.links_by_name)
        self.assertIn(link, net.nodes_by_link)
        self.assertEqual(net.nodes_by_link[link], (upnode, downnode))

    def test_add_links__adds_links_correctly(self):
        upnode1 = Node(name='N11')
        downnode1 = Node(name='N12')
        link1 = Link(4, 3, 1, 100, 30, 1.8, name='L1')
        upnode2 = Node(name='N21')
        downnode2 = Node(name='N21')
        link2 = Link(4, 3, 1, 100, 30, 1.8, name='L2')
        net = Network(name='Net')
        net.add_nodes((upnode1, downnode1, upnode2, downnode2))
        net.add_links((
            (upnode1, link1, downnode1), (upnode2, link2, downnode2)))
        self.assertEqual(len(net.links), 2)
        for data in ((link1, upnode1, downnode1), (link2, upnode2, downnode2)):
            link, nodeup, nodedown = data
            self.assertEqual(net.nodes_by_link[link], (nodeup, nodedown))
            self.assertIs(link, net.links[nodeup, nodedown])
            self.assertIn(link.name, net.links_by_name)
            self.assertIn(link, net.nodes_by_link)

    def test_add_path__adds_nodes_and_links_correctly(self):
        N1 = Node(name='N1')
        N2 = Node(name='N2')
        N3 = Node(name='N3')
        L1 = Link(4, 3, 1, 100, 30, 1.8, name='L1')
        L2 = Link(4, 3, 1, 100, 30, 1.8, name='L2')
        L3 = Link(4, 3, 1, 100, 30, 1.8, name='L3')
        O1 = Origin(name='23423')
        D1 = Destination(name='23421')
        net = Network(name='Net')
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
        net = Network(name='Net')
        with self.assertRaises(ValueError):
            net.add_path(path=(Node(name='N1'),))

    def test_add_origin__adds_origin_correctly(self):
        node = Node(name='This is a random name')
        origin = Origin(name='23423')
        net = Network(name='Another random name')
        net.add_node(node)
        net.add_origin(origin, node)
        self.assertIn(origin, net.origins)
        self.assertIs(net.origins[origin], node)
        self.assertIn(origin.name, net.origins_by_name)

    def test_add_destination__adds_destination_correctly(self):
        node = Node(name='This is a random name')
        destination = Destination(name='23423')
        net = Network(name='Another random name')
        net.add_node(node)
        net.add_destination(destination, node)
        self.assertIn(destination, net.destinations)
        self.assertIs(net.destinations[destination], node)
        self.assertIn(destination.name, net.destinations_by_name)

    def test_out_links__gets_correct_outward_links(self):
        N1 = Node(name='N1')
        N2 = Node(name='N2')
        N3 = Node(name='N3')
        L1 = Link(4, 3, 1, 100, 30, 1.8, name='L1')
        L2 = Link(4, 3, 1, 100, 30, 1.8, name='L2')
        net = Network(name='.Net')
        net.add_links(((N1, L1, N2), (N1, L2, N3)))
        self.assertEqual({(N1, N3, L2), (N1, N2, L1)}, set(net.out_links(N1)))

    def test_in_links__gets_correct_inward_links(self):
        N1 = Node(name='N1')
        N2 = Node(name='N2')
        N3 = Node(name='N3')
        L1 = Link(4, 3, 1, 100, 30, 1.8, name='L1')
        L2 = Link(4, 3, 1, 100, 30, 1.8, name='L2')
        net = Network(name='.Net')
        net.add_links(((N1, L1, N2), (N3, L2, N2)))
        self.assertEqual({(N1, N2, L1), (N3, N2, L2)}, set(net.in_links(N2)))

    def test_is_valid__raises__with_node_with_origin_and_destination(self):
        N = Node(name='N')
        O = MeteredOnRamp(100, name='23423')
        D = Destination(name='23423')
        net = Network(name='.Net')
        net.add_node(N)
        net.add_origin(O, N)
        net.add_destination(D, N)
        with self.assertRaises(InvalidNetworkError):
            net.is_valid(raises=True)

    def test_is_valid__raises__with_duplicate_link(self):
        N1 = Node(name='N1')
        N2 = Node(name='N2')
        N3 = Node(name='N3')
        L = Link(4, 3, 1, 100, 30, 1.8, name='L1')
        net = Network(name='.Net')
        net.add_nodes((N1, N2, N3))
        net.add_link(N1, L, N2)
        net.add_link(N1, L, N3)
        with self.assertRaises(InvalidNetworkError):
            net.is_valid(raises=True)

    def test_is_valid__raises__with_duplicate_origin(self):
        N1 = Node(name='N1')
        N2 = Node(name='N2')
        net = Network(name='.Net')
        O = MeteredOnRamp(100, name='23423')
        net.add_origin(O, N1)
        net.add_origin(O, N2)
        with self.assertRaises(InvalidNetworkError):
            net.is_valid(raises=True)

    def test_is_valid__raises__with_duplicate_destination(self):
        N1 = Node(name='N1')
        N2 = Node(name='N2')
        net = Network(name='.Net')
        D = Destination(name='23423')
        net.add_destination(D, N1)
        net.add_destination(D, N2)
        with self.assertRaises(InvalidNetworkError):
            net.is_valid(raises=True)

    def test_is_valid__raises__destination_on_node_with_out_link(self):
        N1 = Node(name='N1')
        N2 = Node(name='N2')
        L = Link(4, 3, 1, 100, 30, 1.8, name='L1')
        net = Network(name='.Net that does not raise')
        D = Destination(name='23423')
        net.add_link(N1, L, N2)
        net.add_destination(D, N1)
        with self.assertRaises(InvalidNetworkError):
            net.is_valid(raises=True)

    def test_is_valid__raises__origin_on_node_with_in_link(self):
        N1 = Node(name='N1')
        N2 = Node(name='N2')
        L = Link(4, 3, 1, 100, 30, 1.8, name='L1')
        net = Network(name='.Net that does not raise')
        R = MeteredOnRamp(100, name='ramp does not raise')
        net.add_link(N1, L, N2)
        net.add_origin(R, N2)
        net.is_valid(raises=True)
        O = Origin(name='origin raises')
        net.add_origin(O, N2)
        with self.assertRaises(InvalidNetworkError):
            net.is_valid(raises=True)

    def test_init_vars__calls_init_vars_in_elements(self):
        L = 1
        lanes = 2
        C = (3500, 2000)
        a_sym = 1.8
        v_free_sym = 110
        rho_crit_sym = 30
        N1 = Node(name='N1')
        N2 = Node(name='N2')
        N3 = Node(name='N3')
        L1 = Link(4, lanes, L, v_free_sym, rho_crit_sym, a_sym, name='L1')
        L2 = Link(2, lanes, L, v_free_sym, rho_crit_sym, a_sym, name='L2')
        O1 = MeteredOnRamp(C[0], name='O1')
        O2 = SimpleMeteredOnRamp(C[1], name='O2')
        D1 = Destination(name='D1')
        net = (Network(name='A1')
               .add_path(origin=O1, path=(N1, L1, N2, L2, N3), destination=D1)
               .add_origin(O2, N2))

        elements: List[ElementBase] = [N1, N2, N3, L1, L2, O1, O2, D1]
        for el in elements:
            el.init_vars = MagicMock(return_value=None)

        engine = object()
        init_conds = {el: object() for el in elements}
        net.init_vars(init_conds, engine)

        for el in elements:
            if isinstance(el, Node):
                el.init_vars.assert_not_called()
            else:
                el.init_vars.assert_called_once()
                el.init_vars.assert_called_with(
                    init_conditions=init_conds[el],
                    engine=engine)


if __name__ == '__main__':
    unittest.main()
