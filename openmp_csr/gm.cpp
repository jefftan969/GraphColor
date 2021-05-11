#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <pthread.h>
#include <omp.h>

#include "graph.h"

std::vector<int> gmColoring(const Graph &graph) {
    int n = graph.getNumVertices();
    const int *vertices = graph.getCSRVertices();
    const int *neighbors = graph.getCSRNeighbors();
    int numThreads = omp_get_max_threads();

    // Vertex indexed array, storing color of each vertex
    std::vector<int> coloring(n);

    // Keep a worklist, the set of vertices that still need to be considered
    std::vector<int> W(n);
    for(int i = 0; i < n; i++) {
        W[i] = i;
    }

    // Color indexed array for each thread, marking colors that a particular vertex v cannot have
    std::vector<int> *colorMask = new std::vector<int>[numThreads];
    for(int tid = 0; tid < numThreads; tid++) {
        colorMask[tid].resize(n);
    }

    while(!W.empty()) {
        // Determine which colors are permissible for each vertex
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();

            #pragma omp for
            for(int i = 0; i < (int)W.size(); i++) {
                int v = W[i];
                for(int j = vertices[v]; j < vertices[v+1]; j++) {
                    int w = neighbors[j];
                    colorMask[tid][coloring[w]] = v;
                }
                for(int i = 0; i < n; i++) {
                    if(colorMask[tid][i] != v) {
                        coloring[v] = i;
                        break;
                    }
                }
            }
        }

        // Initialize the remaining worklist
        std::vector<int> R;
        #pragma omp parallel
        {
            std::vector<int> localR;

            #pragma omp for
            for(int i = 0; i < (int)W.size(); i++) {
                int v = W[i];
                for(int j = vertices[v]; j < vertices[v+1]; j++) {
                    int w = neighbors[j];
                    if((v < w) && (coloring[v] == coloring[w])) {
                        localR.push_back(v);
                    }
                }
            }

            #pragma omp critical (joinR)
            {
                R.insert(R.end(), localR.begin(), localR.end());
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
    const std::string argument = argv[1];
    if(argument.compare("-p") != 0) {
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

    int numColors = numColorsUsed(coloring);
    std::cout << "Colors used: " << numColors << "\n";

    bool validColoring = checkColoring(graph, coloring);
    if(!validColoring) {
        std::cout << "Invalid coloring\n";
    }

    return 0;
}
