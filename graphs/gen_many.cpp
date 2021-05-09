#include <iostream>
#include <random>
#include <string>
#include <getopt.h>
#include <math.h>

#include "make_graph.h"

void usage(char *program_name) {
    std::cout << "Usage: " << program_name <<
        " -o <out_prefix> -n <min_size> -N <max_size> -c <num_graphs> [-p <edge_prob>] [-s <rng_seed>]\n";
}

int main(int argc, char *argv[]) {
    std::random_device rd;

    std::string filename_prefix;
    int min_size = -1;
    int max_size = -1;
    int num_graphs = -1;
    double p = -1;
    int seed = rd();

    // Parameter parsing
    int opt;
    while((opt = getopt(argc, argv, "o:n:N:c:p:s:")) != -1) {
        switch(opt) {
        case 'o':
            filename_prefix = optarg;
            break;

        case 'n':
            min_size = std::stoi(optarg);
            break;

        case 'N':
            max_size = std::stoi(optarg);
            break;

        case 'c':
            num_graphs = std::stoi(optarg);
            break;

        case 'p':
            p = std::stod(optarg);
            break;

        case 's':
            seed = std::stoi(optarg);
            break;

        default:
            usage(argv[0]);
            return -1;
        }
    }
    if(filename_prefix == "" || min_size == -1 || max_size == -1 || num_graphs == -1) {
        usage(argv[0]);
        return -1;
    }

    std::default_random_engine rng(seed);
    std::uniform_int_distribution<int> n_dist(min_size, max_size);
    for(int i = 0; i < num_graphs; i++) {
        const std::string out_filename = filename_prefix + std::to_string(i);
        int n = n_dist(rng);
        
        if(p == -1) {
            // Randomly generate p if not provided
            double pMin = 1.0 / (2 * n);
            double pMax = 2.0 * log(n) / n;
            std::uniform_real_distribution<double> pDist(pMin, pMax);
            p = pDist(rng);
        }

        make_graph(out_filename + ".txt", n, p, rng());
    }
    return 0;
}
