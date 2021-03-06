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

// NOTE: This code currently infinite-loops if the number of colors is greater than COLOR_MASK_SIZE.
// These values are chosen due to the limited number of registers available in CUDA
#define COLOR_MASK_SIZE 500
#define BLOCK_SIZE 128

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
    int workSize;   // Number of vertices still in work list
    unsigned int *weights;   // Length numVertices     - List of weights for vertices
    int *vertices;  // Length numVertices+1   - List of graph vertices, in CSR representation
    int *neighbors; // Length 2*numEdges      - List of vertex neighbors, in CSR representation
    int *workVerts; // Length numVertices     - Integer array which has the list of vertices we're working on
    int *coloring;  // Length numVertices     - Integer array indicating color of each vertex
    int *worklistChangedFlag; // Boolean flag indicating whether the worklist was changed
};

/**
 * @brief Set all of the random states to be used for curand
 * @param[in] context All data structures allocated on CUDA device
 * @param[in] seed The seed to be used for creating the curand states
 * @param[out] states The random states created by the kernel function
 */
__global__ void kernelRandInit(struct cudaContext context, unsigned int seed, curandState_t *states) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (i < context.workSize) {
        int v = context.workVerts[i];
        curand_init(seed, v, 0, &states[v]);
    }
}

/**
 * @brief Set all vertex weights to be random numbers, which will allow us to find independent sets
 * @param[in] context All data structures allocated on CUDA device
 * @param[in] states The curand states to initialize random numbers
 */
__global__ void kernelSetWeights(struct cudaContext context, curandState_t *states) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    int numVertices = context.numVertices;
    if (i < context.workSize) {
        int v = context.workVerts[i];
        context.weights[v] = curand(&states[v]);
    }
}

/**
 * @brief Creates an independent set of vertices and colors them, and removes colored vertices from worklist
 * @param[in] context All data structures allocated on CUDA device
 * @param[in] color The color to fill in the independent set with
 */
__global__ void kernelColorJP(struct cudaContext context, int color) {
    int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    int workSize = context.workSize;
    const int *vertices = context.vertices;
    const int *neighbors = context.neighbors;
    unsigned int *weights = context.weights;
    int *workVerts = context.workVerts;
    int *coloring = context.coloring;
    int *worklistChangedFlag = context.worklistChangedFlag;

    if(index < workSize) {
        int v = workVerts[index];
        // Check neighbors and determine the remaining worklist
        for(int i = vertices[v]; i < vertices[v+1]; i++) {
            int w = neighbors[i];
            if(weights[v] <= weights[w] && v != w) {
                // Indicate that worklist is not empty and elements still remain
                return;
            }
        }
        *worklistChangedFlag = 1;
        coloring[v] = color;
        workVerts[index] = -1;
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
    context.workSize = numVertices;
    cudaMalloc(&context.weights, sizeof(int) * numVertices);
    cudaMalloc(&context.vertices, sizeof(int) * (numVertices + 1));
    cudaMalloc(&context.neighbors, sizeof(int) * 2 * numEdges);
    cudaMalloc(&context.workVerts, sizeof(int) * numVertices);
    cudaMalloc(&context.coloring, sizeof(int) * numVertices);
    cudaMalloc(&context.worklistChangedFlag, sizeof(int));

    // Create initial worklist
    int *worklist = new int[numVertices];
    for (int i = 0; i < numVertices; i++) worklist[i] = i;
   
    // Note that cudaMemset sets values per byte, so 0x01 => 0x01010101 = 16843009
    cudaMemcpy(context.vertices, vertices, sizeof(int) * (numVertices + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(context.neighbors, neighbors, sizeof(int) * 2 * numEdges, cudaMemcpyHostToDevice);
    cudaMemcpy(context.workVerts, worklist, sizeof(int) * numVertices, cudaMemcpyHostToDevice);
    cudaMemset(context.weights, 0x00, sizeof(int) * numVertices);
    cudaMemset(context.coloring, 0x00, sizeof(int) * numVertices);
    cudaMemset(context.worklistChangedFlag, 0x00, sizeof(int));

    delete worklist;
    return context;
}

/**
 * @brief Performs graph coloring using the JP algorithm
 * @param[in] context A cudaContext sturct containing all device data structures
 * @return coloring The outputted coloring
 */
const int *jpColoring(struct cudaContext context) {
    int numVertices = context.numVertices;

    // Define kernel size and device flags
    int worklistChangedFlag = 0;
    int color = 0;
    int *workingVerts = new int[numVertices];
    
    dim3 blockDimRand(BLOCK_SIZE);
    dim3 gridDimRand((numVertices + BLOCK_SIZE - 1) / BLOCK_SIZE);
    curandState_t *states;
    cudaMalloc((void**)&states, sizeof(curandState_t) * numVertices);
    kernelRandInit<<<gridDimRand, blockDimRand>>>(context, time(NULL), states);
    cudaDeviceSynchronize();

    
    // Loop until worklist is empty
    while(context.workSize != 0) {        
        dim3 blockDim(BLOCK_SIZE);
        dim3 gridDim((context.workSize + BLOCK_SIZE - 1) / BLOCK_SIZE);
   
        // Set random vertex weights for each vertex
        cudaMemset(context.weights, 0x00, sizeof(int) * numVertices);
        kernelSetWeights<<<gridDim, blockDim>>>(context, states);
        cudaDeviceSynchronize();
        //printf("\n");
 
        // Color the graph and determine which vertices are yet to be colored
        cudaMemset(context.worklistChangedFlag, 0x00, sizeof(int));
        kernelColorJP<<<gridDim, blockDim>>>(context, color);
        cudaDeviceSynchronize();
        cudaMemcpy(&worklistChangedFlag, context.worklistChangedFlag, sizeof(int), cudaMemcpyDeviceToHost);

        // Update list of worklist vertices
        cudaMemcpy(workingVerts, context.workVerts, sizeof(int) * context.workSize, cudaMemcpyDeviceToHost);
        int index = 0;
        for (int i = 0; i < context.workSize; i++) {
            if (workingVerts[i] != -1) {
                workingVerts[index] = workingVerts[i];
                index++;
            }
        }
        cudaMemcpy(context.workVerts, workingVerts, sizeof(int) * context.workSize, cudaMemcpyHostToDevice);
        context.workSize = index;

        if (worklistChangedFlag) color++;
    }
    // Cleanup used memory
    delete workingVerts;
    cudaFree(states);

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

    delete coloring;
    return 0;
}
