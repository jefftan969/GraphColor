#include <iostream>
#include <random>
#include <string>
#include <getopt.h>
#include <math.h>

#include "make_graph.h"

void usage(char *program_name) {
    std::cout << "Usage: " << program_name <<
        " -o <out_filename> -n <num_vertices> [-p <edge_prob>] [-s <rng_seed>]\n";
}

int main(int argc, char *argv[]) {
    std::random_device rd;

    std::string out_filename;
    int n = -1;
    double p = -1;
    int seed = rd();

    // Parameter parsing
    int opt;
    while((opt = getopt(argc, argv, "o:n:p:s:")) != -1) {
        switch(opt) {
        case 'o':
            out_filename = optarg;
            break;

        case 'n':
            n = std::stoi(optarg);
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
    if(out_filename == "" || n == -1) {
        usage(argv[0]);
        return -1;
    }

    if(p == -1) {
        // Randomly generate p if not provided
        double pMin = 1.0 / (2 * n);
        double pMax = 2.0 * log(n) / n;
        std::default_random_engine rng(seed);
        std::uniform_real_distribution<double> pDist(pMin, pMax);
        p = pDist(rng);
    }

    make_graph(out_filename + ".txt", n, p, seed);
    return 0;
}
