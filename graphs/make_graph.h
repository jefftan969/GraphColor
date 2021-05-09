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
 *        In the generated graph file, the first line is "n p seed" and the
 *        remaining lines each represent an edge in the outputted graph
 * @param[in] out_filename Filename of output graph file
 * @param[in] n Number of vertices in graph
 * @param[in] p Probability that each edge is included in graph. If p is not
 *              between 0 and 1, it is randomly chosen between min_p and max_p
 * @param[in] seed Seed for random number generator
 */
void make_graph(const std::string &out_filename, int n, double p, int seed) {
    std::default_random_engine rng(seed);

    if(p < 0 || p > 1) {
        std::cerr << "Error: p = " << p << " is invalid\n";
        return;
    }

    // Try to open file, write to stdout if filename is "-"
    std::ostream *output = nullptr;
    if(out_filename == "-") {
        output = &std::cout;
    } else {
        output = new std::ofstream((out_filename + ".txt").c_str());
    }
    if(!output->good()) {
        std::cerr << "Error: Cannot create file " << out_filename << "\n";
        return;
    }
    *output << n << ' ' << p << ' ' << seed << '\n';

    // An edge between two vertices is included independently with probability p
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for(int u = 0; u < n; u++) {
        for(int v = u; v < n; v++) {
            if(uniform(rng) < p) {
                *output << u << ' ' << v << '\n';
            }
        }
    }

    if(out_filename != "-") {
        delete output;
    }
}

#endif // _MAKE_GRAPH_H
