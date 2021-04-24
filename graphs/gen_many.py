#!/usr/bin/python3

import sys
from gen_one import *

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: %s [filename_prefix] [min_size] [max_size] [num_graphs] [p (optional)]" % sys.argv[0])
        exit()

    filename_prefix = sys.argv[1]
    min_size = int(sys.argv[2])
    max_size = int(sys.argv[3])
    num_graphs = int(sys.argv[4])
    p = float(sys.argv[5]) if len(sys.argv) == 6 else None

    for i in range(num_graphs):
        filename = "%s%d" % (filename_prefix, i)
        num_vertices = random.randint(min_size, max_size)
        make_graph(filename, num_vertices, p)
