import unittest
from sym_metanet import (
    Node,
    Link,
    Origin,
    Network,
    Destination,
    InvalidNetworkError,
)
from sym_metanet.blocks.origins import MeteredOnRamp


class TestNetwork(unittest.TestCase):
    def test_add_node(self):
        node = Node(name='This is a random name')
        net = Network(name='Another random name')
        net.add_node(node)
        self.assertIn(node, net.nodes)
        self.assertIn(node.name, net.nodes_by_name)

    def test_add_nodes(self):
        node1 = Node(name='This is a random name1')
        node2 = Node(name='This is a random name2')
        net = Network(name='Another random name')
        net.add_nodes((node1, node2))
        self.assertIn(node1, net.nodes)
        self.assertIn(node2, net.nodes)
        self.assertIn(node1.name, net.nodes_by_name)
        self.assertIn(node1.name, net.nodes_by_name)

    def test_add_link(self):
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

    def test_add_links(self):
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

    def test_add_path(self):
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

    def test_add_origin(self):
        node = Node(name='This is a random name')
        origin = Origin(name='23423')
        net = Network(name='Another random name')
        net.add_node(node)
        net.add_origin(origin, node)
        self.assertIn(origin, net.origins)
        self.assertIs(net.origins[origin], node)
        self.assertIn(origin.name, net.origins_by_name)

    def test_add_destination(self):
        node = Node(name='This is a random name')
        destination = Destination(name='23423')
        net = Network(name='Another random name')
        net.add_node(node)
        net.add_destination(destination, node)
        self.assertIn(destination, net.destinations)
        self.assertIs(net.destinations[destination], node)
        self.assertIn(destination.name, net.destinations_by_name)

    def test_out_links(self):
        N1 = Node(name='N1')
        N2 = Node(name='N2')
        N3 = Node(name='N3')
        L1 = Link(4, 3, 1, 100, 30, 1.8, name='L1')
        L2 = Link(4, 3, 1, 100, 30, 1.8, name='L2')
        net = Network(name='.Net')
        net.add_links(((N1, L1, N2), (N1, L2, N3)))
        self.assertEqual({(N1, N3, L2), (N1, N2, L1)}, set(net.out_links(N1)))

    def test_in_links(self):
        N1 = Node(name='N1')
        N2 = Node(name='N2')
        N3 = Node(name='N3')
        L1 = Link(4, 3, 1, 100, 30, 1.8, name='L1')
        L2 = Link(4, 3, 1, 100, 30, 1.8, name='L2')
        net = Network(name='.Net')
        net.add_links(((N1, L1, N2), (N3, L2, N2)))
        self.assertEqual({(N1, N2, L1), (N3, N2, L2)}, set(net.in_links(N2)))

    def test_validate__raises__with_node_with_origin_and_destination(self):
        N = Node(name='N')
        O = MeteredOnRamp(100, name='23423')
        D = Destination(name='23423')
        net = Network(name='.Net')
        net.add_node(N)
        net.add_origin(O, N)
        net.add_destination(D, N)
        with self.assertRaises(InvalidNetworkError):
            net.validate()

    def test_validate__raises__with_duplicate_link(self):
        N1 = Node(name='N1')
        N2 = Node(name='N2')
        N3 = Node(name='N3')
        L = Link(4, 3, 1, 100, 30, 1.8, name='L1')
        net = Network(name='.Net')
        net.add_nodes((N1, N2, N3))
        net.add_link(N1, L, N2)
        net.add_link(N1, L, N3)
        with self.assertRaises(InvalidNetworkError):
            net.validate()

    def test_validate__raises__with_duplicate_origin(self):
        N1 = Node(name='N1')
        N2 = Node(name='N2')
        net = Network(name='.Net')
        O = MeteredOnRamp(100, name='23423')
        net.add_origin(O, N1)
        net.add_origin(O, N2)
        with self.assertRaises(InvalidNetworkError):
            net.validate()

    def test_validate__raises__with_duplicate_destination(self):
        N1 = Node(name='N1')
        N2 = Node(name='N2')
        net = Network(name='.Net')
        D = Destination(name='23423')
        net.add_destination(D, N1)
        net.add_destination(D, N2)
        with self.assertRaises(InvalidNetworkError):
            net.validate()

    def test_validate__raises__destination_on_node_with_out_link(self):
        N1 = Node(name='N1')
        N2 = Node(name='N2')
        L = Link(4, 3, 1, 100, 30, 1.8, name='L1')
        net = Network(name='.Net that does not raise')
        D = Destination(name='23423')
        net.add_link(N1, L, N2)
        net.add_destination(D, N1)
        with self.assertRaises(InvalidNetworkError):
            net.validate()

    def test_validate__raises__origin_on_node_with_in_link(self):
        N1 = Node(name='N1')
        N2 = Node(name='N2')
        L = Link(4, 3, 1, 100, 30, 1.8, name='L1')
        net = Network(name='.Net that does not raise')
        R = MeteredOnRamp(100, name='ramp does not raise')
        net.add_link(N1, L, N2)
        net.add_origin(R, N2)
        net.validate()
        O = Origin(name='origin raises')
        net.add_origin(O, N2)
        with self.assertRaises(InvalidNetworkError):
            net.validate()


if __name__ == '__main__':
    unittest.main()
