import unittest
from pymetanet import (
    Node,
    Link,
    Network,
    MainstreamOrigin,
    Destination
)


class TestNetwork(unittest.TestCase):
    def test_add_node(self):
        node = Node(name='This is a random name')
        net = Network(name='Another random name')
        net.add_node(node)
        self.assertIn(node, net.nodes)

    def test_add_nodes(self):
        node1 = Node(name='This is a random name1')
        node2 = Node(name='This is a random name2')
        net = Network(name='Another random name')
        net.add_nodes(node1, node2)
        self.assertIn(node1, net.nodes)
        self.assertIn(node2, net.nodes)

    def test_add_link(self):
        upnode = Node(name='N1')
        downnode = Node(name='N2')
        link = Link(4, 3, 1, 100, 30, 1.8, name='L1')
        net = Network(name='Net')
        net.add_nodes(upnode, downnode)
        net.add_link(upnode, link, downnode)
        self.assertIs(link, net.links[upnode, downnode])
        self.assertEqual((link,), list(zip(*net.links))[1])

    def test_add_links(self):
        upnode1 = Node(name='N11')
        downnode1 = Node(name='N12')
        link1 = Link(4, 3, 1, 100, 30, 1.8, name='L1')
        upnode2 = Node(name='N21')
        downnode2 = Node(name='N21')
        link2 = Link(4, 3, 1, 100, 30, 1.8, name='L2')
        net = Network(name='Net')
        net.add_nodes(upnode1, downnode1, upnode2, downnode2)
        net.add_links((upnode1, link1, downnode1), (upnode2, link2, downnode2))
        self.assertIs(link1, net.links[upnode1, downnode1])
        self.assertIs(link2, net.links[upnode2, downnode2])
        self.assertEqual({link1, link2}, set(list(zip(*net.links))[1]))

    def test_add_path(self):
        N1 = Node(name='N1')
        N2 = Node(name='N2')
        N3 = Node(name='N3')
        L1 = Link(4, 3, 1, 100, 30, 1.8, name='L1')
        L2 = Link(4, 3, 1, 100, 30, 1.8, name='L2')
        L3 = Link(4, 3, 1, 100, 30, 1.8, name='L3')
        O1 = MainstreamOrigin(name='23423')
        D1 = Destination(name='23421')
        net = Network(name='Net')
        net.add_path(O1, (N1, L1, N2), D1)
        net.add_path(O1, (N1, L2, N3, L3, N2), D1)
        for n in (N1, N2, N3):
            self.assertIn(n, net.nodes)
        self.assertEqual({L1, L2, L3}, set(list(zip(*net.links))[1]))
        self.assertIs(L1, net.links[N1, N2])
        self.assertIs(L2, net.links[N1, N3])
        self.assertIs(L3, net.links[N3, N2])
        self.assertIn(O1, net.origins)
        self.assertIn(D1, net.destinations)

    def test_add_path__with_single_node__raises(self):
        net = Network(name='Net')
        with self.assertRaises(ValueError):
            net.add_path(None, (Node(name='N1'),), None)

    def test_add_origin(self):
        node = Node(name='This is a random name')
        origin = MainstreamOrigin(name='23423')
        net = Network(name='Another random name')
        net.add_node(node)
        net.add_origin(origin, node)
        self.assertIn(origin, net.origins)
        self.assertIs(net.origins[origin], node)

    def test_add_destination(self):
        node = Node(name='This is a random name')
        destination = Destination(name='23423')
        net = Network(name='Another random name')
        net.add_node(node)
        net.add_destination(destination, node)
        self.assertIn(destination, net.destinations)
        self.assertIs(net.destinations[destination], node)


if __name__ == '__main__':
    unittest.main()
