# python3
from enum import Enum
import collections
import sys
import threading

sys.setrecursionlimit(10 ** 6)  # max depth of recursion
threading.stack_size(2 ** 26)  # new thread will get stack of such size


class Ordered_Sets(collections.MutableSet):

    def __init__( self, iterable=None ):
        self.end = end = []
        end += [None, end, end]  # sentinel node for doubly linked list
        self.map = {}  # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__( self ):
        return len(self.map)

    def __contains__( self, key ):
        return key in self.map

    def add( self, key ):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard( self, key ):
        if key in self.map:
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev
    def __iter__( self ):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]
    def __reversed__( self ):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]
    def pop( self, last=True ):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key
    def __repr__( self ):
        if not self:
            return '%s()' % (self.__class__.__name__,)


def post_orders_( adjacents ):
    """
    Order the nodes of the graph according to their
    post order. Uses (possibly repeated) Depth First Search on
    the graph.
    """

    def dfs( node, order, traversed ):
        traversed.add(node)
        for adj in adjacents[node]:
            if adj in traversed:
                continue
            dfs(adj, order, traversed)
        if node in vertices:
            vertices.remove(node)
        order.add(node)

    post_order = Ordered_Sets([])
    traversed = set([])
    vertices = set([node for node in range(len(adjacents))])

    while True:
        dfs(vertices.pop(), post_order, traversed)
        if len(post_order) == len(adjacents):
            break

    assert len(post_order) == len(adjacents)
    return list(post_order)


def connected_component_( adjacents, node, found ):


    connected = set([])

    def dfs( node, connected ):
        connected.add(node)
        found.add(node)
        for adj in adjacents[node]:
            if adj in found or adj in connected:
                continue
            dfs(adj, connected)

    dfs(node, connected)
    return connected


def analyse_connected_components_( n, adjacents, reverse ):
    # Ensure topological ordering.
    order = post_orders_(reverse)
    # print('orders: {0}'.format(orders))

    order_pointer = len(order) - 1
    found = set([])
    ccs = []
    while order_pointer >= 0:
        if order[order_pointer] in found:
            order_pointer -= 1
            continue

        ccs.append(connected_component_(adjacents, order[order_pointer], found))

    assert len(found) == len(adjacents), 'found {0} nodes, but {1} were specified'.format(len(found), n)
    return ccs


class ImplicationGraph(object):
    var_dict = {}
    node_dict = {}
    adjacents = None
    reversed_adjs = None

    def __init__( self, n, clauses ):
        node_num = 0
        self.adjacents = [[] for _ in range(2 * n)]
        self.reversed_adjs = [[] for _ in range(2 * n)]

        for clause in clauses:
            left = clause[0]
            right = clause[1]
            for term in [left, right]:
                if not term in self.node_dict:
                    self.var_dict[node_num] = term
                    self.node_dict[term] = node_num
                    node_num += 1
                if not -term in self.node_dict:
                    self.var_dict[node_num] = -term
                    self.node_dict[-term] = node_num
                    node_num += 1

            self.adjacents[self.node_dict[-left]].append(self.node_dict[right])
            self.reversed_adjs[self.node_dict[right]].append(self.node_dict[-left])
            # edges.append((node_dict[-left], node_dict[right]))

            self.adjacents[self.node_dict[-right]].append(self.node_dict[left])
            self.reversed_adjs[self.node_dict[left]].append(self.node_dict[-right])
            # edges.append((node_dict[-right], node_dict[left]))

        self.adjacents = self.adjacents[:node_num]
        self.reversed_adjs = self.reversed_adjs[:node_num]


class Colour(Enum):
    R = 0
    G = 1
    B = 2


def get_node_colour( var ):
    node = (var - 1) // 3
    c = var % 3
    if c == 0:
        return node, Colour(2)
    if c == 2:
        return node, Colour(1)
    if c == 1:
        return node, Colour(0)


def generate_2sat_clauses( n, edges, colours ):
    """
    If C is the set of colours (R, G, B), the colour c of each node must change to one of the
    colours in the set: C difference (c).
    It must also be the case that the colour c of any two adjacent nodes is not the same.
    """

    red = Colour(0)
    green = Colour(1)
    blue = Colour(2)
    rgb = set([red, green, blue])

    clauses = []

    for node_ in range(1, n + 1):
        node = node_ * 3 - 2
        c1 = Colour[colours[node_ - 1]]
        others = rgb.difference(set([c1]))
        c2 = others.pop()
        c3 = others.pop()
        c1_var = node + c1.value
        c2_var = node + c2.value
        c3_var = node + c3.value
        clauses += [[c2_var, c3_var], [-c2_var, -c3_var], [-c1_var, -c1_var]]

    for edge in edges:
        # Add adjacency conditions.
        left = edge[0] * 3 - 2
        right = edge[1] * 3 - 2
        clauses += [[-left, -right], [-(left + 1), -(right + 1)], [-(left + 2), -(right + 2)]]

    return clauses


def assign_new_colors( n, edges, colours ):
    """
    Arguments: #   * `n` - the number of vertices.
      * `edges` - list of edges, each edge is a tuple (u, v), 1 <= u, v <= n.
      * `colors` - list consisting of `n` characters, each belonging to the set {'R', 'G', 'B'}.
    Return value:
      * If there exists a proper recoloring, return value is a list containing new colors, similar to the `colors` argument.
      * Otherwise, return value is None.
    """
    num_vars = n * 3
    clauses = generate_2sat_clauses(n, edges, colours[0])
    graph = ImplicationGraph(num_vars, clauses)

    ccs = analyse_connected_components_(num_vars, graph.adjacents, graph.reversed_adjs)

    result = collections.defaultdict(lambda: None)

    for cc in ccs:
        cc_vars = set([])
        for node in cc:

            # Check valid solution.
            litteral = graph.var_dict[node]
            if abs(litteral) in cc_vars:
                return None
            else:
                cc_vars.add(abs(litteral))

            if result[abs(litteral)] is None:
                if litteral < 0:
                    result[abs(litteral)] = 0
                else:
                    result[abs(litteral)] = 1

    result_colours = []
    for key in sorted(result.keys()):
        if result[key] == 1:
            node, colour = get_node_colour(key)
            result_colours.append(colour.name)
    return result_colours


def main():
    n, m = map(int, input().split())
    colors = input().split()
    edges = []
    for i in range(m):
        u, v = map(int, input().split())
        edges.append((u, v))
    new_colors = assign_new_colors(n, edges, colors)
    if new_colors is None:
        print("Impossible")
    else:
        print(''.join(new_colors))


main()