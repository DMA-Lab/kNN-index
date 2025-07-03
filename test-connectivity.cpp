#include <iostream>
#include <queue>

#include "file.h"
#include "graph.h"

// POI反向索引，预计算时需要
using POIInvertedIndex = boost::unordered_flat_map<Vertex, std::vector<const OnEdgePOI*>>;


void check_connectivity(const Graph& graph, Vertex start_node, const POIInvertedIndex& poi_index) {
    if (graph.vertices.find(start_node) == graph.vertices.end()) {
        std::cout << "Start node " << start_node << " does not exist." << std::endl;
        return;
    }

    std::queue<Vertex> q;
    q.push(start_node);

    boost::unordered_flat_set<Vertex> visited_vertices;
    visited_vertices.insert(start_node);

    boost::unordered_flat_set<unsigned int> found_pois;

    while (!q.empty()) {
        Vertex u = q.front();
        q.pop();

        // 检查这个顶点相关的POI
        if (poi_index.count(u)) {
            for (const auto* poi : poi_index.at(u)) {
                found_pois.insert(poi->poi_id);
            }
        }

        // 扩展到邻居
        for (const auto& [v, weight] : graph.get_adjacent_vertices(u)) {
            if (visited_vertices.find(v) == visited_vertices.end()) {
                visited_vertices.insert(v);
                q.push(v);
            }
        }
    }

    std::cout << "\n--- Connectivity Check from Vertex " << start_node << " ---" << std::endl;
    std::cout << "Total reachable vertices: " << visited_vertices.size() << std::endl;
    std::cout << "Total unique POIs found in this component: " << found_pois.size() << std::endl;
    std::cout << "------------------------------------------" << std::endl;
}


POIInvertedIndex build_poi_inverted_index(const std::vector<OnEdgePOI>& pois) {
    POIInvertedIndex index;
    for (const auto& poi : pois) {
        index[poi.u].push_back(&poi);
        index[poi.v].push_back(&poi);
    }
    return index;
}

int main(int argc, char* argv[]) {
    // --- 1. 参数解析 ---
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <start_vertex_id>" << std::endl;
        std::cerr << "  <start_vertex_id>: The vertex to start the connectivity check from." << std::endl;
        return 1;
    }

    Vertex start_node;
    try {
        start_node = std::stoul(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "Error: Invalid vertex ID provided. " << e.what() << std::endl;
        return 1;
    }

    // --- 2. 加载数据 ---
    std::cout << "Loading graph..." << std::endl;
    auto [g, flag1] = load_graph<Graph>();
    if (!flag1) {
        std::cerr << "Failed to load graph." << std::endl;
        return 1;
    }
    std::cout << "Graph loaded. Vertices: " << g.num_vertex << std::endl;

    std::cout << "Loading POIs..." << std::endl;
    auto [pois, flag2] = load_objects();
    if (!flag2) {
        std::cerr << "Failed to load POIs." << std::endl;
        return 1;
    }
    std::cout << "POIs loaded. Count: " << pois.size() << std::endl;

    // --- 3. 准备并调用检查函数 ---
    std::cout << "\nBuilding POI inverted index for the check..." << std::endl;
    POIInvertedIndex poi_index = build_poi_inverted_index(pois);

    check_connectivity(g, start_node, poi_index);

    return 0;
}