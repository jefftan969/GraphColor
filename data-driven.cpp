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

std::vector<int> dataDrivenColoring(Graph& graph) {
  int numVertices = graph.getNumVertices();

  // Create the vector in which the final colors will be stored
  // Element at index i will be the color of vertex i
  std::vector<int> coloring(numVertices, -1);

  // Initialize the worklist
  std::vector<int> inW;
  for (int i = 0; i < numVertices; i++) {
    inW.push_back(i);
  }


  while(!inW.empty()) {
    #pragma omp parallel for schedule(static)
    for (int v = 0; v < numVertices; v++) {
      const std::vector<int>& neighbors = graph.getNeighbors(v);
      std::vector<int> colorMask(numVertices);

      for (int j = 0; j < (int) neighbors.size(); j++) {
        int neighbor = neighbors.at(j);
        if(coloring.at(neighbor) >= 0) {
          colorMask.at(coloring.at(neighbor)) = v;
        }
      }
      int minColor = 0;
      for (int j = 0; j < (int) coloring.size(); j++) {
        if (colorMask.at(j) != v) {
          minColor = j;
          break;
        }
      }
      coloring.at(v) = minColor;
    }

    std::vector<int> outW;

    #pragma omp parallel
    {
      std::vector<int> localOutW;

      #pragma omp for schedule(static)
      for (int v = 0; v < numVertices; v++) {
        const std::vector<int>& neighbors = graph.getNeighbors(v);

        for (int j = 0; j < (int) neighbors.size(); j++) {
          int neighbor = neighbors.at(j);
          int neighborColor = coloring.at(neighbor);
          int vColor = coloring.at(v);
          if (vColor == neighborColor && v < neighbor) {
            localOutW.push_back(v);
          }
        }
      }
      #pragma omp critical (outW_join)
      {
        outW.insert(outW.end(), localOutW.begin(), localOutW.end());
      }
    }

    std::swap(inW, outW);
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
    std::vector<int> coloring = dataDrivenColoring(graph);
    auto t2 = getTime();
    std::cout << "Time: " << getMillis(t1, t2) << "ms\n";

    int numColors = numColorsUsed(coloring);
    std::cout << "Colors used: " << numColors << "\n";

    bool validColoring = checkColoring(graph, coloring);
    if (!validColoring) std::cout << "Invalid coloring\n";

    return 0;
}
