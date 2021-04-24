CXXFLAGS += -Wall -Wextra -pthread -fopenmp -g -O3 -DNDEBUG -std=c++11

all: release

release: parse.cpp
	g++ parse.cpp -o parse $(CXXFLAGS)

clean:
	rm -f ./parse
