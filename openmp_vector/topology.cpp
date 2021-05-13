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

std::vector<int> topologyDrivenColoring(Graph& graph) {
  bool changed = false;
  int numVertices = graph.getNumVertices();

  // Create the vector in which the final colors will be stored
  // Element at index i will be the color of vertex i
  std::vector<int> coloring(numVertices, -1);

  std::vector<bool> isColored(numVertices, false);

  do {
    changed = false;
    #pragma omp parallel for schedule(static)
    for (int v = 0; v < numVertices; v++) {
      if (!isColored.at(v)) {
        const std::vector<int>& neighbors = graph.getNeighbors(v);
        std::vector<int> colorMask(numVertices);

        for (int j = 0; j < (int) neighbors.size(); j++) {
          int neighbor = neighbors.at(j);
          int neighborColor = coloring.at(neighbor);
          if (neighborColor >= 0) colorMask.at(neighborColor) = v;
        }

        // Assign minimum color
        for (int j = 0; j < numVertices; j++) {
            if (colorMask.at(j) != v) {
                coloring.at(v) = j;
                isColored.at(v) = true;
                changed = true;
                break;
            }
        }
      }
    }

    #pragma omp parallel for schedule(static)
    for (int v = 0; v < numVertices; v++) {
      const std::vector<int>& neighbors = graph.getNeighbors(v);
      std::vector<int> colorMask(numVertices);

      for (int j = 0; j < (int) neighbors.size(); j++) {
        int neighbor = neighbors.at(j);
        int neighborColor = coloring.at(neighbor);
        int vColor = coloring.at(v);
        if (vColor == neighborColor && v < neighbor) {
          isColored.at(v) = false;
        }
      }
    }

  } while (changed);
    
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
    std::vector<int> coloring = topologyDrivenColoring(graph);
    auto t2 = getTime();
    std::cout << "Time: " << getMillis(t1, t2) << "ms\n";

    int numColors = numColorsUsed(coloring);
    std::cout << "Colors used: " << numColors << "\n";

    bool validColoring = checkColoring(graph, coloring);
    if (!validColoring) std::cout << "Invalid coloring\n";

    return 0;
}
