CXXFLAGS += -g -O3 -DNDEBUG -std=c++11

# all: sequential jp gm topology data-driven
all: gm

%: ../graph.cpp %.cu
	nvcc $^ -I ../ -o $@ $(CXXFLAGS)

clean:
	rm -f ./sequential
	rm -f ./jp
	rm -f ./gm
	rm -f ./topology
	rm -f ./data-driven
