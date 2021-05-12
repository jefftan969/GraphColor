NVCCFLAGS += -g -O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc -std=c++11

# all: sequential jp gm topology data-driven
all: gm jp

%: ../graph.cpp %.cu
	nvcc $^ -I ../ -o $@ $(NVCCFLAGS)

clean:
	rm -f ./jp
	rm -f ./gm
