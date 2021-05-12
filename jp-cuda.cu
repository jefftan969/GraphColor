#include "graph.h"

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>

#include <curand.h>
#include <curand_kernel.h>

// Include timing library
#include <chrono>

// Define CUDA related things
#define THREADS_PER_BLOCK = 512;

__constant__ int NUM_VERTICES;
__constant__ int **GRAPH;
__constant__ int *NEIGHBOR_SIZES;

__global__ void setWeights(int Wsize, int *vertexWeights, int random_seed) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    printf("%d ", i);
    if (0 <= i && i < Wsize) {
        curandState_t state;
        curand_init(random_seed, blockIdx.x, 0, &state);
        vertexWeights[i] = curand(&state);
    }
}

__global__ void colorGraph(int *W, int Wsize, int *vertexWeights, int *coloring, int color) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    printf("%d ", i);
    if (0 <= i && i < Wsize) {
        int vertex = W[i];
        bool flag = true;

        int *neighbors = GRAPH[vertex];
        for (int j = 0; j < NEIGHBOR_SIZES[vertex]; j++) {
            int neighbor = neighbors[j];
            if (vertexWeights[vertex] <= vertexWeights[neighbor]) {
                flag = false;
                break;
            }
        }

        if (flag) {
            coloring[vertex] = color;
            W[i] = -1;
            printf("flag was true\n");
        } 
    }
}

void colorGraphRoutine(Graph graph, int *coloring) {
    const int threadsPerBlock = 32;

    // Set up initial worklist to contain all vertices
    int *W_host = (int*) malloc(sizeof(int) * graph.getNumVertices());
    int *W;
    cudaMalloc((void**)&W, sizeof(int) * graph.getNumVertices());

    for (int i = 0; i < graph.getNumVertices(); i++) {
        W_host[i] = i;
    }

    int Wsize = graph.getNumVertices();

    // Set color
    int color = 0;

    while (Wsize > 0) {
        cudaMemcpy(W, W_host, sizeof(int) * graph.getNumVertices(), cudaMemcpyHostToDevice);

        const int blocksPerGrid = (Wsize + threadsPerBlock - 1) / threadsPerBlock; 
        
        // Initialize vertex weights array
        int *vertexWeights; 
        cudaMalloc((void**)&vertexWeights, sizeof(int) * Wsize);
        cudaMemset(vertexWeights, 0, sizeof(int) * Wsize);

        // Generate random weights for graph
        setWeights<<<blocksPerGrid, threadsPerBlock>>>(Wsize, vertexWeights, time(NULL));
        cudaDeviceSynchronize();

        int *weights = (int*) malloc(sizeof(int) * Wsize);
        cudaMemcpy(weights, vertexWeights, sizeof(int) * Wsize, cudaMemcpyDeviceToHost);
        
        // Printing the random weights
        for (int i = 0; i < Wsize; i++) printf("%d ", weights[i]);
        printf("\n");

        // Color the graph using JP algorithm
        colorGraph<<<blocksPerGrid, threadsPerBlock>>>(W, Wsize, vertexWeights, coloring, color);
        cudaDeviceSynchronize();

        // Update the worklist to not include colored vertices
        printf("Removing colored vertices\n");
        cudaMemcpy(W_host, W, sizeof(int) * graph.getNumVertices(), cudaMemcpyDeviceToHost);
        int index = 0;
        for (int i = 0; i < Wsize; i++) {
            if (W_host[i] != -1) {
                W_host[index] = W_host[i];
                index++;
            }
        }
        Wsize = index;
        printf("Wsize: %d\n", Wsize);

        printf("Freeing vertex weights\n");
        cudaFree(&vertexWeights);
        // Update the color
        color++;
    }

}

void setup(Graph graph) {
    // Create graph from file
    int numVertices = graph.getNumVertices();

    // Convert into 2D array so we can store for CUDA
    int **adjacencyList = (int**) malloc(sizeof(int*) * numVertices);
    int *neighborSizes = (int*) malloc(sizeof(int) * numVertices);

    // Go through vertices and set their neighbor list
    for (int v = 0; v < numVertices; v++) {
      std::vector<int> neighbors = graph.getNeighbors(v);
      int numNeighbors = neighbors.size();
      neighborSizes[v] = numNeighbors;

      int *vNeighbors = (int*) malloc(sizeof(int) * numNeighbors);
      for (int j = 0; j < numNeighbors; j++) {
        vNeighbors[j] = neighbors.at(j);
      }
      
      adjacencyList[v] = vNeighbors;
    }

    // Add graph to special global memory for CUDA
    cudaMalloc((void**)&NUM_VERTICES, sizeof(int));
    cudaMalloc((void**)&GRAPH, sizeof(int*) * numVertices);
    cudaMalloc((void**)&NEIGHBOR_SIZES, sizeof(int) * numVertices);
    cudaMemcpyToSymbol(NUM_VERTICES, &numVertices, sizeof(int));
    cudaMemcpyToSymbol(GRAPH, adjacencyList, sizeof(int*) * numVertices);
    cudaMemcpyToSymbol(NEIGHBOR_SIZES, neighborSizes, sizeof(int) * numVertices);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << "[in_filename]\n";
        exit(0);
    }
    const std::string in_filename = argv[1];
    Graph graph = Graph(in_filename);

    // CONVERTING GRAPH CLASS TO 2D ARRAY
    setup(graph);
    printf("Done setting up graph\n");
    fflush(stdout);
    int *coloring;
    cudaMalloc((void**)&coloring, sizeof(int) * graph.getNumVertices());
    cudaMemset(coloring, 0, sizeof(int) * graph.getNumVertices());

    auto t1 = getTime();
    printf("About to color graph\n");
    fflush(stdout);
    colorGraphRoutine(graph, coloring);
    auto t2 = getTime();
    std::cout << "Time: " << getMillis(t1, t2) << "ms\n";

    std::vector<int> p_coloring;
    for (int i = 0; i < graph.getNumVertices(); i++) {
        p_coloring.push_back(coloring[i]);
    }
    bool validColoring = checkColoring(graph, p_coloring);
    if (!validColoring) std::cout << "Invalid coloring\n";

    return 0;
}
