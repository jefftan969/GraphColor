#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "graph.h"

Graph::Graph(const std::string &filename) :
    filename_(filename)
{
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

int Graph::getNumVertices(void) const {
    return numVertices_;
}

const std::vector<int>& Graph::getNeighbors(int vertex) const {
    return graph_.at(vertex);
}

void Graph::print(void) const {
    std::cout << "Graph " << filename_ << ": \n";
    for(int v = 0; v < numVertices_; v++) {
        std::cout << v << ": ";
        for(int w : getNeighbors(v)) {
            std::cout << w << " ";
        }
        std::cout << "\n";
    }
}

void printColoring(const std::vector<int> &coloring) {
    std::cout << "Coloring: \n";
    for(size_t v = 0; v < coloring.size(); v++) {
        std::cout << v << ": " << coloring.at(v) << "\n";
    }
}

bool checkColoring(const Graph &graph, const std::vector<int> &coloring) {
    bool isValid = true;
    for(int v = 0; v < graph.getNumVertices(); v++) {
        for(int w : graph.getNeighbors(v)) {
            if((v < w) && (coloring.at(v) == coloring.at(w))) {
                isValid = false;
                std::cout << "Edge (" << v << ", " << w << ") invalid: "
                          << "color[" << v << "] = color[" << w << "] = " << coloring.at(v) << "\n";
            }
        }
    }
    return isValid;
}
