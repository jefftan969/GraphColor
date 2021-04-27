#include "graph.h"

#include <iostream>
#include <string>
#include <vector>

std::vector<int> sequentialColoring(Graph& graph) {
    int numVertices = graph.getNumVertices();

    // Create the vector in which the final colors will be stored
    // Element at index i will be the color of vertex i
    std::vector<int> coloring(numVertices);

    // Create vector which will keep track of neighbors' colors
    std::vector<int> usedColors(numVertices);

    // Loop through the vertices to determine color
    for (int vertex = 0; vertex < numVertices; vertex++) {

        // Loop through neighbors to keep track of their colors
        const std::vector<int>& neighbors = graph.getNeighbors(vertex);

        for (int j = 0; j < (int) neighbors.size(); j++) {
            int neighbor = neighbors.at(j);
            usedColors.at(coloring.at(neighbor)) = vertex;
        }

        // Assign the minimum color which is not shared by neighbors
        for (int i = 0; i < numVertices; i++) {
            if (usedColors.at(i) != vertex) {
                coloring.at(vertex) = i;
                break;
            }
        }
    }

    return coloring;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " [in_filename]\n";
        exit(0);
    }

    const std::string in_filename = argv[1];
    Graph graph = Graph(in_filename);
    
    auto t1 = getTime();
    std::vector<int> coloring = sequentialColoring(graph);
    auto t2 = getTime();
    std::cout << "Time: " << getMillis(t1, t2) << "ms\n";

    bool validColoring = checkColoring(graph, coloring);
    if (!validColoring) std::cout << "Invalid coloring\n";

    return 0;
}
