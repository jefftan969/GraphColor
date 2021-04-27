#include "graph.h"

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <stdlib.h>
#include <pthread.h>
#include <omp.h>

// Include timing library
#include <chrono>

std::vector<int> jpColoring(Graph& graph) {
    int numVertices = graph.getNumVertices();
    int currentColor = 0;

    // Create the vector in which the final colors will be stored
    // Element at index i will be the color of vertex i
    std::vector<int> coloring(numVertices, -1);

    // Keep a worklist, the set of vertices that need to be considered
    std::vector<int> W;
    for(int i = 0; i < numVertices; i++) {
        W.push_back(i);
    }

    // Used to initialize RNG
    std::random_device rd;

    while(!W.empty()) {
        // Assign random weights to each vertex
        // TODO: This parallel loop takes very long
        std::vector<int> weights(numVertices);
        #pragma omp parallel
        {
            std::default_random_engine rng(rd());
            #pragma omp for
            for (int i = 0; i < numVertices; i++) {
                weights.at(i) = rng();
            }
        }

        // Initialize the independent set of vertices
        std::vector<int> S;

        // Add vertices to the independent set
        #pragma omp parallel
        {
            std::vector<int> S_local;

            #pragma omp for
            for (int i = 0; i < (int) W.size(); i++) {
                int vertex = W.at(i);
                bool flag = true;
                const std::vector<int>& neighbors = graph.getNeighbors(vertex);
                for (int j = 0; j < (int) neighbors.size(); j++) {
                    int neighbor = neighbors.at(j);
                    if (weights.at(vertex) < weights.at(neighbor)) {
                        flag = false;
                        break;
                    }
                }

                if (flag) {
                    S_local.push_back(vertex);
                }
            }

            #pragma omp critical(updateS)
            {
                S.insert(S.end(), S_local.begin(), S_local.end());
            }
        }

        // Color the independent set
        // TODO: This parallel loop takes very long
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < (int) S.size(); j++) {
            int vertex = S.at(j);
            coloring.at(vertex) = currentColor;
        }

        // Update the working set and color
        if (!S.empty()) {
            std::vector<int> newW;
            std::set_difference(W.begin(), W.end(), S.begin(), S.end(), 
                                std::inserter(newW, newW.end()));
            W = newW;
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

    bool validColoring = checkColoring(graph, coloring);
    if (!validColoring) std::cout << "Invalid coloring\n";

    return 0;
}
