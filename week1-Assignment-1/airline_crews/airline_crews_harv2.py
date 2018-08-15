# python3
import copy
import queue

class MaxMatching:
    def read_data(self):
        n, m = map(int, input().split())
        adj_matrix = [list(map(int, input().split())) for i in range(n)]
        return adj_matrix
    def write_response(self, matching):
        line = [str(-1 if x == -1 else x + 1) for x in matching]
        print(' '.join(line))
    def find_matching(self, adj_matrix):
        # Source: 1
        # Flights: 2
        # Crews: 3
        # Sink: 4
        n = len(adj_matrix)
        m = len(adj_matrix[0])
        matching = [-1] * n
        busy_right = [False] * m
        def bfs():
            visited_Node = set()
            q = queue.Queue()
            q.put((1, None))
            visited_Node.add((1, None))
            path = []
            parent = dict()
            while not q.empty():
                current_node = q.get()
                if 1 == current_node[0]: # Source
                    for i in range(n):
                        if -1 == matching[i]:
                            visited_Node.add((2, i))
                            parent[(2, i)] = (1, None)
                            q.put((2, i))
                elif 2 == current_node[0]: # Flights
                    i = current_node[1]
                    for j in range(m):
                        if 1 == adj_matrix[i][j] and j != matching[i] and not (3, j) in visited_Node:
                            visited_Node.add((3, j))
                            parent[(3, j)] = current_node
                            q.put((3, j))
                elif 3 == current_node[0]:
                    j = current_node[1]
                    if not busy_right[j]:
                        prevNode = current_node
                        current_node = (4, j)
                        while True:
                            path.insert(0, (prevNode, current_node))
                            if 1 == prevNode[0]:
                                break
                            current_node = prevNode
                            prevNode = parent[current_node]
                        for e in path:
                            if 2 == e[0][0]:
                                matching[e[0][1]] = e[1][1]
                            elif 3 == e[0][0] and 4 == e[1][0]:
                                busy_right[e[1][1]] = True
                        #print(path)
                        return True # There is a path
                    else:
                        for i in range(n):
                            if j == matching[i] and not (2, i) in visited_Node:
                                visited_Node.add((2, i))
                                parent[(2, i)] = current_node
                                q.put((2, i))
            return False # There isn't a path

        while bfs():
            continue
        return matching
    def solve(self):
        adj_matrix = self.read_data()
        matching = self.find_matching(adj_matrix)
        self.write_response(matching)

if __name__ == '__main__':
    max_matching = MaxMatching()
    max_matching.solve()