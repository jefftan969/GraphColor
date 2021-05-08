CXXFLAGS += -Wall -Wextra -pthread -fopenmp -g -O3 -DNDEBUG -std=c++11

all: sequential jp gm topology data-driven

sequential: graph.cpp sequential.cpp
	g++ graph.cpp sequential.cpp -o sequential $(CXXFLAGS)

jp: graph.cpp jp.cpp
	g++ graph.cpp jp.cpp -o jp $(CXXFLAGS)

gm: graph.cpp gm.cpp
	g++ graph.cpp gm.cpp -o gm $(CXXFLAGS)

topology: graph.cpp topology.cpp
	g++ graph.cpp topology.cpp -o topology $(CXXFLAGS)

data-driven: graph.cpp data-driven.cpp
	g++ graph.cpp data-driven.cpp -o data-driven $(CXXFLAGS)

clean:
	rm -f ./sequential
	rm -f ./jp
	rm -f ./gm
	rm -f ./topology
	rm -f ./data-driven
