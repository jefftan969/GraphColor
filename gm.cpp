#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <pthread.h>
#include <omp.h>

#include "graph.h"

std::vector<int> gmColoring(const Graph &graph) {
    int n = graph.getNumVertices();

    // Vertex indexed array, storing color of each vertex
    std::vector<int> coloring(n);


    // Keep a worklist, the set of vertices that still need to be considered
    std::vector<int> W;
    for(int i = 0; i < n; i++) {
        W.push_back(i);
    }

    while(!W.empty()) {
        // Determine which colors are permissible for each vertex
        #pragma omp parallel
        {
            // Color indexed array, marking colors that a particular vertex v cannot have
            std::vector<int> colorMask(n);

            #pragma omp for
            for(int i = 0; i < (int)W.size(); i++) {
                int v = W.at(i);
                const std::vector<int> &neighbors = graph.getNeighbors(v);
                for(int j = 0; j < (int)neighbors.size(); j++) {
                    int w = neighbors.at(j);
                    colorMask.at(coloring.at(w)) = v;
                }
                for(int i = 0; i < n; i++) {
                    if(colorMask.at(i) != v) {
                        coloring.at(v) = i;
                        break;
                    }
                }
            }
        }

        // Initialize the remaining worklist
        std::vector<int> R;
        #pragma omp parallel
        {
            // Local copy of remaining worklist, merged with R after loop
            std::vector<int> R_local;

            #pragma omp for
            for(int i = 0; i < (int)W.size(); i++) {
                int v = W.at(i);
                const std::vector<int> &neighbors = graph.getNeighbors(v);
                for(int j = 0; j < (int)neighbors.size(); j++) {
                    int w = neighbors.at(j);
                    if((coloring.at(v) == coloring.at(w)) && (v < w)) {
                        R_local.push_back(v);
                    }
                }
            }

            #pragma omp critical (R_join)
            {
                R.insert(R.end(), R_local.begin(), R_local.end());
            }
        }

        // Update current worklist
        std::swap(W, R);
    }

    return coloring;
}

int main(int argc, char *argv[]) {
    if(argc < 4) {
        std::cout << "Usage: " << argv[0] << " -p [processor_count] [in_filename]\n";
        exit(-1);
    }

    const std::string flag = argv[1];
    if(flag.compare("-p") != 0) {
        std::cout << "Usage: " << argv[0] << " -p [processor_count] [in_filename]\n";
        exit(-1);
    }

    int NCORES = std::stoi(argv[2]);
    omp_set_num_threads(NCORES);
    const std::string in_filename = argv[3];
    Graph graph = Graph(in_filename);

    auto t1 = getTime();
    std::vector<int> coloring = gmColoring(graph);
    auto t2 = getTime();
    std::cout << "Time: " << getMillis(t1, t2) << "ms\n";

    bool validColoring = checkColoring(graph, coloring);
    if(!validColoring) {
        std::cout << "Invalid coloring\n";
    }
    return 0;
}
