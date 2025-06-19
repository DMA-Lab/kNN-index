//
// Created by sunnysab on 25-2-22.
//

#include <iostream>
#include "graph.h"
#include "file.h"


int main() {
    auto [graph, graph_success] = load_graph<Graph2>();
    if (!graph_success) {
        std::cerr << "Error: failed to open graph file" << std::endl;
        return 0;
    }

    auto total_degree = 0u;
    for (auto &[v, arr]: graph.edges) {
        auto degree = arr.size();
        total_degree += degree;
    }

    std::cout << "total vertex: " << graph.num_vertex << std::endl;
    std::cout << "average degree: " << 1.0 * total_degree / graph.num_vertex << std::endl;
    return 0;
}