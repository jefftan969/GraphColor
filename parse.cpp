#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

void parse(const std::string &filename) {
    std::ifstream file(filename.c_str());
    std::string line;

    std::getline(file, line);
    size_t numVertices = std::stoi(line);

    // Construct graph data structure
    std::vector<std::vector<int> > graph(numVertices);

    // Read input from file
    while(std::getline(file, line)) {
        int v1;
        int v2;
        std::stringstream ss(line);
        ss >> v1;
        ss >> v2;

        graph.at(v1).push_back(v2);
        graph.at(v2).push_back(v1);
    }

    for(size_t i = 0; i < numVertices; i++) {
        std::cout << i << ": ";
        for(size_t j = 0; j < graph.at(i).size(); j++) {
            std::cout << graph.at(i).at(j) << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    parse("graphs/ex2");
}
