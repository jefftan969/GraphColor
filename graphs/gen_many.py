#!/usr/bin/python3

import sys
from gen_one import *

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: %s [min_size] [max_size] [num_graphs] [filename_prefix]" % sys.argv[0])

    min_size = int(sys.argv[1])
    max_size = int(sys.argv[2])
    num_graphs = int(sys.argv[3])
    filename_prefix = sys.argv[4]

    for i in range(num_graphs):
        make_graph(random.randint(min_size, max_size), "%s%d" % (filename_prefix, i))
