CXXFLAGS += -Wall -Wextra -g -O3 -DNDEBUG -std=c++11

all: gen_one gen_many

gen_one: make_graph.h gen_one.cpp
	g++ gen_one.cpp -o gen_one $(CXXFLAGS)

gen_many: make_graph.h gen_many.cpp
	g++ gen_many.cpp -o gen_many $(CXXFLAGS)

clean:
	rm -f ./gen_one
	rm -f ./gen_many
