#ifndef _GRAPH_H
#define _GRAPH_H

#include <string>
#include <vector>

class Graph {
private:
    std::string filename_;
    int numVertices_;
    std::vector<std::vector<int> > graph_;

public:
    Graph(const std::string &filename);
    int getNumVertices(void) const;
    const std::vector<int>& getNeighbors(int vertex) const;
    void print(void) const;
};

void printColoring(const std::vector<int> &coloring);
bool checkColoring(const Graph &graph, const std::vector<int> &coloring);

#endif
