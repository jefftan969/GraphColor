CXXFLAGS += -Wall -Wextra -pthread -fopenmp -g -O3 -DNDEBUG -std=c++11

all: sequential jp gm

sequential: graph.cpp sequential.cpp
	g++ graph.cpp sequential.cpp -o sequential $(CXXFLAGS)

jp: graph.cpp jp.cpp
	g++ graph.cpp jp.cpp -o jp $(CXXFLAGS)

gm: graph.cpp gm.cpp
	g++ graph.cpp gm.cpp -o gm $(CXXFLAGS)

clean:
	rm -f ./sequential
	rm -f ./jp
	rm -f ./gm
