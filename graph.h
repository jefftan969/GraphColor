#ifndef _GRAPH_H
#define _GRAPH_H

#include <string>
#include <vector>

class Graph {
private:
    size_t numVertices_;
    std::vector<std::vector<int> > graph_;

public:
    Graph(const std::string &filename);
    size_t getNumVertices(void);
    const std::vector<int>& getNeighbors(int vertex);
    void print(void);
};

#endif
