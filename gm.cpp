#include <iostream>
#include <vector>

#include "graph.h"

std::vector<int> gm(const Graph &graph) {
    int n = graph.getNumVertices();

    // color: Vertex indexed array, storing color of each vertex
    std::vector<int> color(n);

    // colorMask: Color indexed array, marking colors that a particular vertex v cannot have
    std::vector<int> colorMask(n);

    // worklist: Set of vertices that still need to be considered
    std::vector<int> W;
    for(int i = 0; i < n; i++) {
        W.push_back(i);
    }

    while(!W.empty()) {
        // Determine which colors are permissible for each vertex
        // TODO parallelize this loop
        for(int v : W) {
            for(int w : graph.getNeighbors(v)) {
                colorMask.at(color.at(w)) = v;
            }
            for(int i = 0; i < n; i++) {
                if(colorMask.at(i) != v) {
                    color.at(v) = i;
                    break;
                }
            }
        }

        // Initialize the remaining worklist
        // TODO parallelize this loop
        std::vector<int> R;
        for(int v : W) {
            for(int w : graph.getNeighbors(v)) {
                if((color.at(v) == color.at(w)) && (v < w)) {
                    R.push_back(v);
                }
            }
        }

        // Update current worklist
        std::swap(W, R);
    }

    return color;
}

int main() {
    Graph graph = Graph("graphs/ex0");
    graph.print();
    std::vector<int> color = gm(graph);
    printColoring(color);
    std::cout << "Is coloring valid: " << checkColoring(graph, color) << "\n";
    return 0;
}
