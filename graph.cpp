#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>

#include "graph.h"

/**
 * @brief Constructs a graph from the given input file.
 *        The first line of the file is the number of vertices in the graph,
 *        and the remaining lines "<v1> <v2>" denote edges in the graph,
 *        where v1 and v2 are the vertices in the edge separated by a space.
 *        Graphs are represented as adjacency lists using a vector of vectors.
 * @param[in] filename Input file containing the graph
 */
Graph::Graph(const std::string &in_filename) :
    filename_(filename)
{
    // Try to open file
    std::ifstream file(filename.c_str());
    if(!file.is_open()) {
        std::cerr << "Error: File '" << filename << "' does not exist\n";
        exit(-1);
    }

    // Read number of vertices
    std::string line;
    std::getline(file, line);
    std::stringstream ss(line);
    ss >> numVertices_;

    // Construct graph data structure
    graph_.resize(numVertices_);

    // Read edges from file
    while(std::getline(file, line)) {
        int v1, v2;
        std::stringstream ss(line);
        ss >> v1 >> v2;

        graph_.at(v1).push_back(v2);
        graph_.at(v2).push_back(v1);
    }
}

/**
 * @brief Returns the number of vertices in the graph
 */
int Graph::getNumVertices(void) const {
    return numVertices_;
}

/**
 * @brief Returns a vector of neighbors for the given vertex
 */
const std::vector<int>& Graph::getNeighbors(int vertex) const {
    return graph_.at(vertex);
}

/**
 * @brief Prints the adjacency list of the current graph
 */
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

/**
 * @brief Prints a given vertex coloring
 * @param[in] coloring A vertex indexed vector, storing the color of each vertex
 */
void printColoring(const std::vector<int> &coloring) {
    std::cout << "Coloring: \n";
    for(size_t v = 0; v < coloring.size(); v++) {
        std::cout << v << ": " << coloring.at(v) << "\n";
    }
}

/**
 * @brief Given a graph, checks whether the given vertex coloring is correct,
 *        printing out any invalid edges where both vertices have the same color
 * @param[in] graph The graph to check against
 * @param[in] coloring A vertex indexed vector, storing the color of each vertex
 */
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
