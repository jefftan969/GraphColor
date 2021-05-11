#ifndef _GRAPH_H
#define _GRAPH_H

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

class Graph {
private:
    std::string filename_;
    int numVertices_;
    int numEdges_;
    int* vertices_;
    int* edges_;
    std::vector< std::vector<int> > graph_;

    void parseDimacs(std::istream &file);
    void parseDimacsBinary(std::istream &file);
    void parseMatrixMarket(std::istream &file);
    void parseEdgeList(std::istream &file);
    void getCSR(void);

public:
    Graph(const std::string &filename);
    ~Graph(void);
    int getNumVertices(void) const;
    int getNumEdges(void) const;
    int* getCSRVertices(void) const;
    int* getCSREdges(void) const;
    const std::vector<int>& getNeighbors(int vertex) const;
    void print(void) const;
};

void printColoring(const std::vector<int> &coloring);
bool checkColoring(const Graph &graph, const std::vector<int> &coloring);
int numColorsUsed(const std::vector<int> &coloring);

/**
 * @brief Helper function to check whether a filename has a given extension
 * @param[in] filename Filename to check
 * @param[in] extension Desired file extension
 */
inline bool hasExtension(const std::string &filename, const std::string &extension) {
    return (filename.length() >= extension.length()) &&
           (filename.substr(filename.length() - extension.length()) == extension);
}

/**
 * @brief Returns the current time
 */
inline std::chrono::time_point<std::chrono::high_resolution_clock> getTime() {
    return std::chrono::high_resolution_clock::now();
}

/**
 * @brief Calculates the duration in milliseconds between the two given times
 * @param[in] t1 Start time
 * @param[in] t2 End time
 */
inline double getMillis(const std::chrono::time_point<std::chrono::high_resolution_clock> t1,
                        const std::chrono::time_point<std::chrono::high_resolution_clock> t2) {
    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    return ms_double.count();
}

#endif // _GRAPH_H
