CXXFLAGS += -Wall -Wextra -pthread -fopenmp -g -O3 -DNDEBUG -std=c++11
NVCCFLAGS += -g -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc -std=c++11

all: sequential jp gm topology data-driven jp-cuda

%: graph.cpp %.cpp
	g++ $^ -o $@ $(CXXFLAGS)

jp-cuda: graph.cpp jp-cuda.cu
	nvcc graph.cpp jp-cuda.cu -o jp-cuda $(NVCCFLAGS)

clean:
	rm -f ./sequential
	rm -f ./jp
	rm -f ./gm
	rm -f ./topology
	rm -f ./data-driven
	rm -f ./jp-cuda
