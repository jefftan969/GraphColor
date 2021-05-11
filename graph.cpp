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
Graph::Graph(const std::string &filename) :
    filename_(filename),
    numVertices_(0),
    numEdges_(0),
    vertices_(nullptr),
    neighbors_(nullptr)
{
    // Read edgelist from stdin if filename is "-"
    if(filename == "-") {
        parseEdgeList(std::cin);
        return;
    } 
    
    // Otherwise, try to open file
    std::ifstream file(filename.c_str());
    if(!file.is_open()) {
        std::cerr << "Error: File '" << filename << "' does not exist\n";
        exit(-1);
    }

    if(hasExtension(filename, ".col")) {
        // Interpret graph as DIMACS standard format
        parseDimacs(file);
    } else if(hasExtension(filename, ".col.b")) {
        // Interpret graph as DIMACS binary format
        parseDimacsBinary(file);
    } else if(hasExtension(filename, ".mtx")) {
        // Interpret graph as Matrix Market format
        parseMatrixMarket(file);
    } else if(hasExtension(filename, ".txt")) {
        // Interpret graph as edge list format
        parseEdgeList(file);
    } else {
        std::cerr << "Error: File '" << filename << "' has an unknown format\n";
        exit(-1);
    }

    // Create CSR graph representation for CUDA implementations
    getCSR();
}

/**
 * @brief Destructor for graph object
 */
Graph::~Graph(void) {
    delete vertices_;
    delete neighbors_;
}

/**
 * @brief Parse graph representation as DIMACS edge list format
 * @param[in] input Input stream
 */
void Graph::parseDimacs(std::istream &input) {
    std::string line;
    while(std::getline(input, line)) {
        char linetype;
        std::stringstream ss(line);
        ss >> linetype;

        if(linetype == 'e') {
            // Read edge, note that DIMACS edge list format is 1-indexed
            int v1, v2;
            ss >> v1 >> v2;
            graph_[v1-1].push_back(v2-1);
            graph_[v2-1].push_back(v1-1);
            numEdges_++;

        } else if(linetype == 'p') {
            // Read number of vertices
            std::string filetype;
            ss >> filetype;
            if(filetype != "edge") {
                std::cerr << "Error: File is not in DIMACS edge list format\n";
                exit(-1);
            }

            ss >> numVertices_;
            graph_.resize(numVertices_);
        }
    }
}

/**
 * @brief Parse graph representation as DIMACS binary format
 * @param[in] input Input stream
 */
void Graph::parseDimacsBinary(std::istream &input) {
    std::string line;

    // Read preamble
    std::getline(input, line);
    while(std::getline(input, line)) {
        char linetype;
        std::stringstream ss(line);
        ss >> linetype;

        if(linetype == 'p') {
            // Read number of vertices
            std::string filetype;
            ss >> filetype;
            if(filetype != "edge") {
                std::cerr << "Error: File is not in DIMACS binary format\n";
                exit(-1);
            }

            ss >> numVertices_;
            graph_.resize(numVertices_);
            break;
        }
    }

    // Read rows of adjacency matrix, and add edges to graph data structure
    char *adjacencyRow = new char[numVertices_/8 + 1];
    for(int i = 0; i < numVertices_; i++) {
        input.read(adjacencyRow, i/8 + 1);

        for(int j = 0; j <= i; j++) {
            int bit = 0x07 - (j & 0x07);
            int byte = j >> 3;
            char mask = 0x01 << bit;

            if((adjacencyRow[byte] & mask) == mask) {
                graph_[i].push_back(j);
                graph_[j].push_back(i);
                numEdges_++;
            }
        }
    }
}

/**
 * @brief Parse graph representation as Matrix Market adjacency matrix format
 * @param[in] input Input stream
 */
void Graph::parseMatrixMarket(std::istream &input) {
    std::string line;
    
    // Only parsing files of coordinate format, not array
    std::getline(input, line);
    if (line.find("coordinate") == std::string::npos) {
        std::cerr << "Error: File is not of coordinate format.\n";
        exit(-1);
    }

    // Continue to read file until past comment lines
    while(line.at(0) == '%') {
        std::getline(input, line);
    }

    // Obtain the number of vertices from the first line of file
    std::stringstream ss(line);
    ss >> numVertices_;
    graph_.resize(numVertices_);

    while(std::getline(input, line)) {
        std::stringstream ss(line);

        // Note: ignoring the last number of line, which gives weights
        int v1, v2;
        ss >> v1 >> v2;

        // Matrix Market format is indexed at 1, while we index at 0
        graph_[v1-1].push_back(v2-1);
        graph_[v2-1].push_back(v1-1);
        numEdges_++;
    }
}

/**
 * @brief Parse graph representation as custom edge list format
 * @param[in] input Input stream
 */
void Graph::parseEdgeList(std::istream &input) {
    // First line contains number of vertices, subsequent lines each denote an edge
    std::string line;
    std::getline(input, line);
    std::stringstream ss(line);

    ss >> numVertices_;
    graph_.resize(numVertices_);

    while(std::getline(input, line)) {
        std::stringstream ss(line);
        
        int v1, v2;
        ss >> v1 >> v2;
        graph_[v1].push_back(v2);
        graph_[v2].push_back(v1);
        numEdges_++;
    }
}

/**
 * @brief Represent graph in compressed sparse row format for CUDA programs
 *        For each vertex v, the neighbors of v are stored between indices
 *        vertices_[v] (inclusive) and vertices_[v+1] (exclusive) in edges_
 */
void Graph::getCSR(void) {
    vertices_ = new int[numVertices_ + 1];
    neighbors_ = new int[2*numEdges_];

    int offset = 0;
    for(int v = 0; v < numVertices_; v++) {
        vertices_[v] = offset;
        const std::vector<int> &neighbors = getNeighbors(v);
        for(int j = 0; j < (int)neighbors.size(); j++) {
            neighbors_[offset] = neighbors.at(j);
            offset++;
        }
    }
    vertices_[numVertices_] = offset;
}

/**
 * @brief Returns the number of vertices in the graph
 */
int Graph::getNumVertices(void) const {
    return numVertices_;
}

/**
 * @brief Returns the number of edges in the graph
 */
int Graph::getNumEdges(void) const {
    return numEdges_;
}

/**
 * @brief Returns vertices array for CSR representation of graph
 */
const int *Graph::getCSRVertices(void) const {
    return vertices_;
}

/**
 * @brief Returns neighbors array for CSR representation of graph
 */
const int *Graph::getCSRNeighbors(void) const {
    return neighbors_;
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
    const int *vertices = getCSRVertices();
    const int *neighbors = getCSRNeighbors();
    for(int v = 0; v < numVertices_; v++) {
        std::cout << v << ": ";
        for(int j = vertices[v]; j < vertices[v+1]; j++) {
            int w = neighbors[j];
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
    for(int v = 0; v < (int)coloring.size(); v++) {
        std::cout << v << ": " << coloring[v] << "\n";
    }
}

/**
 * @brief Given a graph, checks whether the given vertex coloring is correct,
 *        printing out any invalid edges where both vertices have the same color
 * @param[in] graph The graph to check against
 * @param[in] coloring A vertex indexed vector, storing the color of each vertex
 */
bool checkColoring(const Graph &graph, const std::vector<int> &coloring) {
    const int *vertices = graph.getCSRVertices();
    const int *neighbors = graph.getCSRNeighbors();
    bool isValid = true;
    for(int v = 0; v < graph.getNumVertices(); v++) {
        for(int j = vertices[v]; j < vertices[v+1]; j++) {
            int w = neighbors[j];
            if((v < w) && (coloring[v] == coloring[w])) {
                isValid = false;
                std::cout << "Edge (" << v << ", " << w << ") invalid: "
                          << "color[" << v << "] = color[" << w << "] = " << coloring[v] << "\n";
            }
        }
    }
    return isValid;
}

/**
 * @brief Returns the number of colors used in a given vertex coloring
 * @param[in] coloring A vertex indexed vector, storing the color of each vertex
 */
int numColorsUsed(const std::vector<int> &coloring) {
    int maxColor = -1;
    for(int v = 0; v < (int)coloring.size(); v++) {
        if(coloring[v] > maxColor) {
            maxColor = coloring[v];
        }
    }
    return maxColor + 1; // Colors are 0-indexed
}
