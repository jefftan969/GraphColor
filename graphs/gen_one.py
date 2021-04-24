#!/usr/bin/python3

import math
import random
import sys

def make_graph(out_filename, n, p=None):
    graph = [set() for _ in range(n)]

    if p is None:
        min_p = 1 / (2 * n)
        max_p = 2 * math.log(n) / n
        p = random.uniform(min_p, max_p)

    # An edge between two vertices is included with probability p
    with open(out_filename, "w") as f:
        f.write("%d\n" % n)
        for u in range(n):
            for v in range(n):
                if u > v and random.random() < p:
                    f.write("%d %d\n" % (u,v))
                    graph[u].add(v)
                    graph[v].add(u)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: %s [out_filename] [num_vertices] [p (optional)]" % sys.argv[0])
        exit()

    out_filename = sys.argv[1]
    num_vertices = int(sys.argv[2])
    p = float(sys.argv[3]) if len(sys.argv) == 4 else None

    make_graph(out_filename, num_vertices, p)
