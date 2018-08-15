//
//  ex_bessie.cpp
//  
//
//  Created by 陈贝茜 on 2018/6/26.
//

#include "ex_bessie.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm> // std::min
#include <limits>

using std::vector;
using std::string;
using std::cout;
using std::endl;

/* This class implements a bit unusual scheme for storing edges of the graph,
 * in order to retrieve the backward edge for a given edge quickly. */
class FlowGraph {
    
private:
    vector<vector<int>> C, T;
    int nodes;
    
    int BFS(const vector<vector<int>> &C, const vector<vector<int>> &T, int s, int t, const vector<vector<int>> &F, vector<int> &P) {
        vector<int> M(nodes);
        P[s] = -2;
        M[s] = std::numeric_limits<int>::max();
        
        vector<int> Q;
        Q.push_back(s);
        while (Q.size() > 0) {
            int u = Q.front();
            Q.erase(Q.begin());
            for (int v = 0; v < nodes; ++v) {
                if (v != u) {
                    if (T[u][v] > 0 && P[v] == -1) {
                        P[v] = u;
                        M[v] = std::min(M[u], C[u][v] - F[u][v]);
                        if (v != t) {
                            Q.push_back(v);
                        } else {
                            return M[t];
                        }
                    }
                }
            }
        }
        return 0;
    }
    
public:
    void read_data(const string &filename) {
        std::ifstream fs(filename);
        fs >> nodes;
        C.resize(nodes, vector<int>(nodes, 0));
        T.resize(nodes, vector<int>(nodes, 0));
        
        int edges; fs >> edges;
        for (int i = 0; i < edges; ++i) {
            int from, to, capacity;
            fs >> from >> to >> capacity;
            C[from - 1][to - 1] += capacity;
            T[from - 1][to - 1] += capacity;
        }
    }
    
    int size() const {
        return nodes;
    }
    
    int max_flow(int from, int to) {
        int flow = 0;
        vector<vector<int>> F(nodes, vector<int>(nodes, 0));
        
        while (true) {
            vector<int> P(nodes, -1);
            int m = BFS(C, T, from, to, F, P);
            
            if (m == 0) {
                break;
            } else {
                flow += m;
                int v = to;
                while (v != from) {
                    int u = P[v];
                    F[u][v] += m;
                    F[v][u] -= m;
                    T[u][v] -= m;
                    T[v][u] += m;
                    v = u;
                }
            }
        }
        return flow;
    }
};


int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Please specify a file name for test." << endl;
    } else {
        std::ios_base::sync_with_stdio(false);
        string filename(argv[1]);
        FlowGraph graph;
        graph.read_data(filename);
        
        std::cout << graph.max_flow(0, graph.size() - 1) << "\n";
    }
    
    return 0;
}
