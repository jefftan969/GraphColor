#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <omp.h>

#include "graph.h"

// NOTE: This code currently infinite-loops if the number of colors is greater than COLOR_MASK_SIZE.
// These values are assigned due to constraints on the number of registers available in CUDA
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

__global__ void kernelSpeculate(int numVertices, int numEdges, int blockSize,
        const int *vertices, const int *neighbors, int *worklist, int *coloring) {
    int v = blockIdx.x * blockSize + threadIdx.x;
    int colorMask[COLOR_MASK_SIZE];

    if(v < numVertices && worklist[v]) {
//        printf("%d in if\n", v);
        // Determine which colors are permissible for each vertex
        for(int j = vertices[v]; j < vertices[v+1]; j++) {
            int w = neighbors[j];
//            printf("%d in for, neighbor %d, coloring[w]=%d\n", v, w, coloring[w]);
            colorMask[coloring[w]] = v;
        }
//        printf("%d colorMask[0]=%d\n", v, colorMask[0]);
        for(int i = 0; i < COLOR_MASK_SIZE && i < numVertices; i++) {
//            printf("%d in second for\n", v);
            if(colorMask[i] != v) {
//                printf("%d in setting coloring with i=%d\n", v, i);
                coloring[v] = i;
//                printf("%d Coloring: %d %d %d %d %d\n", v, coloring[0], coloring[1], coloring[2], coloring[3], coloring[4]);
                break;
            }
        }
    }
}

__global__ void kernelResolveConflicts(int numVertices, int numEdges, int blockSize,
        const int *vertices, const int *neighbors, int *worklist, int *coloring, int *worklistEmpty) {
    int v = blockIdx.x * blockSize + threadIdx.x;
    if(v < numVertices && worklist[v]) {
//        printf("%d worklist empty %d\n", v, *worklistEmpty);
        // Resolve conflicts and determine the remaining worklist
        for(int j = vertices[v]; j < vertices[v+1]; j++) {
            int w = neighbors[j];
            if((v < w) && (coloring[v] == coloring[w])) {
//                printf("%d Conflict detected: worklist empty %d\n", v, *worklistEmpty);
                // Indicate that elements still remain in the worklist
                *worklistEmpty = 0;
                return;
            }
        }
        worklist[v] = 0;
    }
}

const int *gmColoring(const Graph &graph) {
    int numVertices = graph.getNumVertices();
    int numEdges = graph.getNumEdges();
    const int *vertices = graph.getCSRVertices();
    const int *neighbors = graph.getCSRNeighbors();

    // Initialize memory and copy CSR representation to CUDA device
    int *cudaVertices = NULL;
    int *cudaNeighbors = NULL;
    int *cudaWorklist = NULL;
    int *cudaColoring = NULL;
    int *cudaWorklistEmpty = NULL;

    cudaMalloc(&cudaVertices, sizeof(int) * (numVertices + 1));
    cudaMalloc(&cudaNeighbors, sizeof(int) * 2 * numEdges);
    cudaMalloc(&cudaWorklist, sizeof(int) * numVertices);
    cudaMalloc(&cudaColoring, sizeof(int) * numVertices);
    cudaMalloc(&cudaWorklistEmpty, sizeof(int));
   
    // Note that cudaMemset sets values per byte, so 0x01 => 0x01010101 = 16843009
    cudaMemcpy(cudaVertices, vertices, sizeof(int) * (numVertices + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaNeighbors, neighbors, sizeof(int) * 2 * numEdges, cudaMemcpyHostToDevice);
    cudaMemset(cudaWorklist, 0x01, sizeof(int) * numVertices);
    cudaMemset(cudaColoring, 0x00, sizeof(int) * numVertices);
    cudaMemset(cudaWorklistEmpty, 0x00, sizeof(int));

    int localWorklistEmpty = 0;

    // Kernel size, must ensure that not all computations are running at once in single block
    int blockSize = 0;
    int gridSize = 0;
    if(numVertices < 2 * BLOCK_SIZE) {
        blockSize = (numVertices + 1) / 2;
        gridSize = 2;
    } else {
        blockSize = BLOCK_SIZE;
        gridSize = (numVertices + blockSize - 1) / blockSize;
    }
    dim3 blockDim(blockSize);
    dim3 gridDim(gridSize);

    int *coloring = new int[numVertices];

    // Loop until worklist is empty
    while(true) {
        // Determine which colors are permissible for each vertex
        kernelSpeculate<<<gridDim, blockDim>>>(numVertices, numEdges, blockSize,
                cudaVertices, cudaNeighbors, cudaWorklist, cudaColoring);
        cudaDeviceSynchronize();

        // Resolve conflicts and determine the remaining worklist
        cudaMemset(cudaWorklistEmpty, 0x01, sizeof(int));
        kernelResolveConflicts<<<gridDim, blockDim>>>(numVertices, numEdges, blockSize,
                cudaVertices, cudaNeighbors, cudaWorklist, cudaColoring, cudaWorklistEmpty);
        cudaMemcpy(&localWorklistEmpty, cudaWorklistEmpty, sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        cudaMemcpy(coloring, cudaColoring, sizeof(int) * numVertices, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
//        printColoring(coloring, numVertices);

        // Abort if worklist is empty
        if(localWorklistEmpty) {
            break;
        }
    }
    
    // Retrieve coloring from device
//    int *coloring = new int[numVertices];
//    cudaMemcpy(coloring, cudaColoring, sizeof(int) * numVertices, cudaMemcpyDeviceToHost);
    return coloring;
}

/*
std::vector<int> gmColoring(const Graph &graph) {
    int n = graph.getNumVertices();
    const int *vertices = graph.getCSRVertices();
    const int *neighbors = graph.getCSRNeighbors();

    // Vertex indexed array, storing color of each vertex
    std::vector<int> coloring(n);

    // Keep a worklist, the set of vertices that still need to be considered
    std::vector<int> W(n);
    for(int i = 0; i < n; i++) {
        W[i] = i;
    }

    // Color indexed array for each thread, marking colors that a particular vertex v cannot have
    std::vector<int> *colorMask = new std::vector<int>[numThreads];
    for(int tid = 0; tid < numThreads; tid++) {
        colorMask[tid].resize(n);
    }

    while(!W.empty()) {
        // Determine which colors are permissible for each vertex
        #pragma omp parallel
        {
            int tid = 0;

            #pragma omp for
            for(int i = 0; i < (int)W.size(); i++) {
                int v = W[i];
                for(int j = vertices[v]; j < vertices[v+1]; j++) {
                    int w = neighbors[j];
                    colorMask[tid][coloring[w]] = v;
                }
                for(int i = 0; i < n; i++) {
                    if(colorMask[tid][i] != v) {
                        coloring[v] = i;
                        break;
                    }
                }
            }
        }

        // Initialize the remaining worklist
        std::vector<int> R;
        #pragma omp parallel
        {
            std::vector<int> localR;

            #pragma omp for
            for(int i = 0; i < (int)W.size(); i++) {
                int v = W[i];
                for(int j = vertices[v]; j < vertices[v+1]; j++) {
                    int w = neighbors[j];
                    if((v < w) && (coloring[v] == coloring[w])) {
                        localR.push_back(v);
                    }
                }
            }

            #pragma omp critical (joinR)
            {
                R.insert(R.end(), localR.begin(), localR.end());
            }
        }

        // Update current worklist
        std::swap(W, R);
    }

    return coloring;
}
*/

int main(int argc, char *argv[]) {
    if(argc < 2) {
        std::cout << "Usage: " << argv[0] << " [in_filename]\n";
        exit(-1);
    }
    const std::string in_filename = argv[1];
    Graph graph = Graph(in_filename);

    auto t1 = getTime();
    const int *coloring = gmColoring(graph);
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
