#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <omp.h>

#include "graph.h"

#include <curand.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 512

#define DEBUG
#ifdef DEBUG
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if(code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
        if(abort) {
            exit(code);
        }
    }
}
#else
#define cudaCheckError(ans) ans
#endif

// Stores all data structures allocated on CUDA device
struct cudaContext {
    int numVertices;
    int numEdges;
    unsigned int *weights;    // Length numVertices   - List of weights for vertices
    int *vertices;            // Length numVertices+1 - List of graph vertices, in CSR representation
    int *neighbors;           // Length 2*numEdges    - List of vertex neighbors, in CSR representation
    int *worklist;            // Length numVertices   - Boolean array indicating whether each vertex is in worklist
    int *coloring;            // Length numVertices   - Integer array indicating color of each vertex
    curandState_t *states;    // Length numVertices   - Random state for each vertex
    int *worklistEmptyFlag;   // Boolean flag indicating whether the worklist is empty
};

/**
 * @brief Set all of the random states to be used for curand
 * @param[in] context All data structures allocated on CUDA device
 * @param[in] seed The seed to be used for creating the curand states
 */
__global__ void kernelRandInit(struct cudaContext context, unsigned int seed) {
    int v = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if(v < context.numVertices) {
        curand_init(seed, v, 0, &context.states[v]);
    }
}

/**
 * @brief Set all vertex weights to be random numbers, which will allow us to find independent sets
 * @param[in] context All data structures allocated on CUDA device
 */
__global__ void kernelSetWeights(struct cudaContext context) {
    int v = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    if (v < context.numVertices && context.worklist[v]) {
        context.weights[v] = curand(&context.states[v]);
    }
}

/**
 * @brief Creates an independent set of vertices and colors them, and removes colored vertices from worklist
 * @param[in] context All data structures allocated on CUDA device
 * @param[in] color The color to fill in the independent set with
 */
__global__ void kernelColorJP(struct cudaContext context, int color) {
    int v = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    int numVertices = context.numVertices;
    const int *vertices = context.vertices;
    const int *neighbors = context.neighbors;
    unsigned int *weights = context.weights;
    int *worklist = context.worklist;
    int *coloring = context.coloring;
    int *worklistEmptyFlag = context.worklistEmptyFlag;

    if(v < numVertices && worklist[v]) {
        // Check neighbors and determine the remaining worklist
        for(int i = vertices[v]; i < vertices[v+1]; i++) {
            int w = neighbors[i];
            if(weights[v] < weights[w]) {
                // Indicate that worklist is not empty and elements still remain
                *worklistEmptyFlag = 0;
                return;
            }
        }
        coloring[v] = color;
        worklist[v] = 0;
    }
}

/**
 * @brief Copy graph representation to CUDA device and allocate all necessary data structures
 * @param[in] graph The graph to color
 * @return context A cudaContext struct containing all device data structures
 */
struct cudaContext setup(const Graph &graph) {
    int numVertices = graph.getNumVertices();
    int numEdges = graph.getNumEdges();
    const int *vertices = graph.getCSRVertices();
    const int *neighbors = graph.getCSRNeighbors();

    // Initialize data structures
    struct cudaContext context;
    context.numVertices = numVertices;
    context.numEdges = numEdges;
    cudaMalloc(&context.weights, sizeof(unsigned int) * numVertices);
    cudaMalloc(&context.vertices, sizeof(int) * (numVertices + 1));
    cudaMalloc(&context.neighbors, sizeof(int) * 2 * numEdges);
    cudaMalloc(&context.worklist, sizeof(int) * numVertices);
    cudaMalloc(&context.coloring, sizeof(int) * numVertices);
    cudaMalloc(&context.states, sizeof(curandState_t) * numVertices);
    cudaMalloc(&context.worklistEmptyFlag, sizeof(int));
   
    // Note that cudaMemset sets values per byte, so 0x01 => 0x01010101 = 16843009
    cudaMemcpy(context.vertices, vertices, sizeof(int) * (numVertices + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(context.neighbors, neighbors, sizeof(int) * 2 * numEdges, cudaMemcpyHostToDevice);
    cudaMemset(context.worklist, 0x01, sizeof(int) * numVertices);
    cudaMemset(context.weights, 0x00, sizeof(int) * numVertices);
    cudaMemset(context.coloring, 0x00, sizeof(int) * numVertices);
    cudaMemset(context.states, 0x00, sizeof(curandState_t) * numVertices);
    cudaMemset(context.worklistEmptyFlag, 0x00, sizeof(int));

    return context;
}

/**
 * @brief Frees all data structures in the given CUDA context
 * @param[in] context The CUDA context to free
 */
void freeCudaContext(struct cudaContext context) {
    cudaFree(context.weights);
    cudaFree(context.vertices);
    cudaFree(context.neighbors);
    cudaFree(context.worklist);
    cudaFree(context.coloring);
    cudaFree(context.states);
    cudaFree(context.worklistEmptyFlag);
}

/**
 * @brief Performs graph coloring using the JP algorithm
 * @param[in] context A cudaContext sturct containing all device data structures
 * @return coloring The outputted coloring
 */
const int *jpColoring(struct cudaContext context) {
    int numVertices = context.numVertices;

    // Define kernel size and device flags
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((numVertices + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int worklistEmptyFlag = 0;

    // Initialize random state per vertex
    kernelRandInit<<<gridDim, blockDim>>>(context, time(NULL));

    // Loop until worklist is empty
    for(int color = 0; !worklistEmptyFlag; color++) {
        std::cout << "Loop color " << color << "\n";

        // Set random vertex weights for each vertex in worklist
        // All other vertex weights are set beforehand to 0
        cudaMemset(context.weights, 0x00, sizeof(unsigned int) * numVertices);
        kernelSetWeights<<<gridDim, blockDim>>>(context);
        cudaDeviceSynchronize();
 
        // Resolve conflicts and determine the remaining worklist
        cudaMemset(context.worklistEmptyFlag, 0x01, sizeof(int));
        kernelColorJP<<<gridDim, blockDim>>>(context, color);
        cudaDeviceSynchronize();
        cudaMemcpy(&worklistEmptyFlag, context.worklistEmptyFlag, sizeof(int), cudaMemcpyDeviceToHost);
    }
    
    // Retrieve coloring from device
    int *coloring = new int[numVertices];
    cudaMemcpy(coloring, context.coloring, sizeof(int) * numVertices, cudaMemcpyDeviceToHost);
    return coloring;
}

int main(int argc, char *argv[]) {
    if(argc < 2) {
        std::cout << "Usage: " << argv[0] << " [in_filename]\n";
        exit(-1);
    }
    const std::string in_filename = argv[1];
    Graph graph = Graph(in_filename);
    struct cudaContext context = setup(graph);

    auto t1 = getTime();
    const int *coloring = jpColoring(context);
    auto t2 = getTime();
    std::cout << "Time: " << getMillis(t1, t2) << "ms\n";

    int numColors = numColorsUsed(coloring, graph.getNumVertices());
    std::cout << "Colors used: " << numColors << "\n";

    bool validColoring = checkColoring(graph, coloring);
    if(!validColoring) {
        std::cout << "Invalid coloring\n";
    }

    freeCudaContext(context);
    delete coloring;
    return 0;
}
