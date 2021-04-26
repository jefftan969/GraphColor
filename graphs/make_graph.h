#ifndef _MAKE_GRAPH_H
#define _MAKE_GRAPH_H

#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <math.h>

/**
 * @brief Generates a graph with n vertices using the Erdos-Renyi method
 *        Each edge in the graph is included independently with probability p
 *        In the generated graph file, the first line is "n p" and the remaining
 *        lines each represent an edge in the outputted graph
 * @param[in] out_filename Filename of output graph file
 * @param[in] n Number of vertices in graph
 * @param[in] p Probability that each edge is included in graph. If p is not
 *              between 0 and 1, it is randomly chosen between min_p and max_p
 */
void make_graph(const std::string &out_filename, int n, double p) {
    std::random_device rd;
    std::default_random_engine rng(rd());

    if(p < 0 || p > 1) {
        std::cerr << "Error: p = " << p << " is invalid\n";
        return;
    }

    // An edge between two vertices is included independently with probability p
    std::ofstream file(out_filename.c_str());
    if(!file.good()) {
        std::cerr << "Error: Cannot create file " << out_filename << "\n";
        return;
    }
    file << n << ' ' << p << '\n';

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for(int u = 0; u < n; u++) {
        for(int v = u; v < n; v++) {
            if(uniform(rng) < p) {
                file << u << ' ' << v << '\n';
            }
        }
    }
}

#endif
