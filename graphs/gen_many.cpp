#include <iostream>
#include <random>
#include <string>
#include <math.h>

#include "make_graph.h"

int main(int argc, char *argv[]) {
    std::random_device rd;
    std::default_random_engine rng(rd());

    if(argc < 5) {
        std::cout << "Usage: " << argv[0] << " [name_prefix] [min_size] [max_size] [num_graphs] [p (optional)]\n";
        return -1;
    }
 
    const std::string filename_prefix = argv[1];
    int min_size = std::stoi(argv[2]);
    int max_size = std::stoi(argv[3]);
    int num_graphs = std::stoi(argv[4]);

    std::uniform_int_distribution<int> n_dist(min_size, max_size);
    for(int i = 0; i < num_graphs; i++) {
        const std::string filename = filename_prefix + std::to_string(i);
        int n = n_dist(rng);
        
        double p;
        if(argc == 6) {
            p = std::stod(argv[5]);
        } else {
            double min_p = 1.0 / (2 * n);
            double max_p = 2.0 * log(n) / n;
            std::uniform_real_distribution<double> p_dist(min_p, max_p);
            p = p_dist(rng);
        }

        make_graph(filename, n, p);
    }
    return 0;
}
