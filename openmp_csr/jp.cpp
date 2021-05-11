#include "graph.h"

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <stdlib.h>
#include <pthread.h>
#include <omp.h>

std::vector<int> jpColoring(Graph& graph) {
    int numVertices = graph.getNumVertices();
    const int *vertices = graph.getCSRVertices();
    const int *neighbors = graph.getCSRNeighbors();
    int currentColor = 0;

    // Create the vector in which the final colors will be stored
    // Element at index i will be the color of vertex i
    std::vector<int> coloring(numVertices, -1);

    // Keep a worklist, the set of vertices that need to be considered
    std::vector<int> W(numVertices);
    for(int i = 0; i < numVertices; i++) {
        W[i] = i;
    }

    // Used to initialize RNG
    std::random_device rd;

    while(!W.empty()) {
        // Assign random weights to each vertex
        std::vector<int> weights(numVertices);
        #pragma omp parallel
        {
            std::default_random_engine rng(rd());
            #pragma omp for
            for (int i = 0; i < (int) W.size(); i++) {
                int v = W[i];
                weights[v] = rng();
            }
        }

        // Initialize the independent set of vertices
        std::vector<int> S;

        // Add vertices to the independent set
        #pragma omp parallel
        {
            std::vector<int> localS;

            #pragma omp for schedule(static)
            for (int i = 0; i < (int) W.size(); i++) {
                int vertex = W[i];
                bool flag = true;
                for (int j = vertices[vertex]; j < vertices[vertex+1]; j++) {
                    int neighbor = neighbors[j];
                    if (weights[vertex] < weights[neighbor]) {
                        flag = false;
                        break;
                    }
                }

                if (flag) {
                    localS.push_back(vertex);
                }
            }

            #pragma omp critical(updateS)
            {
                S.insert(S.end(), localS.begin(), localS.end());
            }
        }

        // Color the independent set
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < (int) S.size(); j++) {
            int vertex = S[j];
            coloring[vertex] = currentColor;
        }

        // Update the working set and color
        if (!S.empty()) {
            std::vector<int> newW;
            // Set difference requires both arrays to be sorted, W remains
            // sorted throughout but we need to sort S here since threads may
            // insert out-of-order into S
            std::sort(S.begin(), S.end());
            std::set_difference(W.begin(), W.end(), S.begin(), S.end(), 
                                std::inserter(newW, newW.end()));
            std::swap(W, newW);
            currentColor++;
        }
    }
    return coloring;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " -p [processor count] [in_filename]\n";
        exit(0);
    }
    std::string argument = argv[1];
    if (argument.compare("-p") != 0) {
        std::cout << "Usage: " << argv[0] << " -p [processor count] [in_filename]\n";
        exit(0);
    }
    int NCORES = std::stoi(argv[2]);
    omp_set_num_threads(NCORES);
    const std::string in_filename = argv[3];
    Graph graph = Graph(in_filename);
    
    auto t1 = getTime();
    std::vector<int> coloring = jpColoring(graph);
    auto t2 = getTime();
    std::cout << "Time: " << getMillis(t1, t2) << "ms\n";

    int numColors = numColorsUsed(coloring);
    std::cout << "Colors used: " << numColors << "\n";

    bool validColoring = checkColoring(graph, coloring);
    if (!validColoring) std::cout << "Invalid coloring\n";

    return 0;
}
