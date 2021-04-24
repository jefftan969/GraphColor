#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "graph.h"

Graph::Graph(const std::string &filename) {
    std::ifstream file(filename.c_str());
    std::string line;

    std::getline(file, line);
    numVertices_ = std::stoi(line);

    // Construct graph data structure
    graph_.resize(numVertices_);

    // Read input from file
    while(std::getline(file, line)) {
        int v1, v2;
        std::stringstream ss(line);
        ss >> v1 >> v2;

        graph_.at(v1).push_back(v2);
        graph_.at(v2).push_back(v1);
    }
}

size_t Graph::getNumVertices(void) {
    return numVertices_;
}

const std::vector<int>& Graph::getNeighbors(int vertex) {
    return graph_.at(vertex);
}

void Graph::print(void) {
    for(size_t i = 0; i < numVertices_; i++) {
        std::cout << i << ": ";
        for(size_t j = 0; j < graph_.at(i).size(); j++) {
            std::cout << graph_.at(i).at(j) << " ";
        }
        std::cout << "\n";
    }
}
