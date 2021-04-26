#include <iostream>
#include <random>
#include <string>
#include <math.h>

#include "make_graph.h"

int main(int argc, char *argv[]) {
    std::random_device rd;
    std::default_random_engine rng(rd());

    if(argc < 3) {
        std::cout << "Usage: " << argv[0] << " [out_filename] [num_vertices] [p (optional)]\n";
        return -1;
    }

    const std::string out_filename = argv[1];
    int n = std::stoi(argv[2]);
    double p;
    if(argc == 4) {
        p = std::stod(argv[3]);
    } else {
        double min_p = 1.0 / (2 * n);
        double max_p = 2.0 * log(n) / n;
        std::uniform_real_distribution<double> p_dist(min_p, max_p);
        p = p_dist(rng);
    }

    make_graph(out_filename, n, p);
    return 0;
}
