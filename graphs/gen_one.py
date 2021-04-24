#!/usr/bin/python3

import math
import random
import sys

def make_graph(n, out_filename):
    graph = [set() for _ in range(n)]

    # An edge between two vertices is included with probability p
    min_p = 1/(2*n)
    max_p = 2*math.log(n)/n
    p = random.uniform(min_p, max_p)

    with open(out_filename, "w") as f:
        f.write("%d\n" % n)
        for u in range(n):
            for v in range(n):
                if u > v and random.random() < p:
                    f.write("%d %d\n" % (v,u))
                    graph[u].add(v)
                    graph[v].add(u)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: %s [num_vertices] [out_filename]" % sys.argv[0])
        exit()

    num_vertices = int(sys.argv[1])
    out_filename = sys.argv[2]
    make_graph(num_vertices, out_filename)
