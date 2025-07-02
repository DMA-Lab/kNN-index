#include <iostream>
#include <vector>
#include <queue>
#include <chrono>
#include <string>
#include <iomanip>

#include "file.h"
#include "graph.h"
#include "precomputation.h" // 引入我们的新模块

// --- 在线查询所需的数据结构 ---
// 注意：在线查询时，我们需要将缓存中的 POI ID 转换回 POI 指针
// 这需要一个从 POI ID 到 POI 指针的映射
using POIMap = boost::unordered_flat_map<unsigned int, const OnEdgePOI*>;

/**
 * @brief (在线) 查询函数 - 现在使用全量预计算缓存
 */
std::vector<QueryResult> find_k_nearest_neighbors_with_cache(
    const Graph& graph,
    const PrecomputationCacheFull& precomputation_cache,
    const POIMap& poi_map,
    Vertex query_q,
    int k2)
{
    auto total_start_time = std::chrono::high_resolution_clock::now();

    std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<PQEntry>> pq;
    std::priority_queue<QueryResult> knn_results;
    boost::unordered_flat_map<Vertex, Weight> dist_map;
    boost::unordered_flat_set<unsigned int> visited_pois;
    Weight theta = std::numeric_limits<Weight>::max();

    if (graph.vertices.find(query_q) == graph.vertices.end()) {
        std::cerr << "Error: Query vertex " << query_q << " does not exist in the graph." << std::endl;
        return {};
    }

    dist_map[query_q] = 0;
    pq.push({0, query_q});

    auto update_results = [&](Weight dist, const OnEdgePOI* poi) {
        if (visited_pois.count(poi->poi_id)) return;
        if (knn_results.size() < k2) {
            knn_results.push({dist, poi});
            visited_pois.insert(poi->poi_id);
            if (knn_results.size() == k2) theta = knn_results.top().distance;
        } else if (dist < theta) {
            knn_results.pop();
            knn_results.push({dist, poi});
            visited_pois.insert(poi->poi_id);
            theta = knn_results.top().distance;
        }
    };

    while (!pq.empty()) {
        auto [current_dist, u] = pq.top();
        pq.pop();

        if (current_dist > theta) break;
        if (dist_map.count(u) && current_dist > dist_map.at(u)) continue;

        // *** 核心修改：直接使用全量缓存 ***
        if (precomputation_cache.count(u)) {
            for (const auto& [dist_from_u, poi_id] : precomputation_cache.at(u)) {
                if (poi_map.count(poi_id)) {
                    const OnEdgePOI* poi = poi_map.at(poi_id);
                    update_results(current_dist + dist_from_u, poi);
                }
            }
        }

        for (const auto& [v, weight] : graph.get_adjacent_vertices(u)) {
            Weight new_dist = current_dist + weight;
            if (!dist_map.count(v) || new_dist < dist_map.at(v)) {
                dist_map[v] = new_dist;
                pq.push({new_dist, v});
            }
        }
    }

    auto total_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> online_query_time = total_end_time - total_start_time;

    std::cout << "------------------------------------------" << std::endl;
    std::cout << "ONLINE QUERY TIME: " << std::fixed << std::setprecision(4) << online_query_time.count() << " ms" << std::endl;
    std::cout << "------------------------------------------" << std::endl;

    std::vector<QueryResult> final_results;
    while (!knn_results.empty()) {
        final_results.push_back(knn_results.top());
        knn_results.pop();
    }
    std::reverse(final_results.begin(), final_results.end());
    return final_results;
}


int main(int argc, char* argv[]) {
    Vertex query_q = 10000;
    constexpr int k1 = 5;
    constexpr int k2 = 20; // k' > k

    // --- 1. 加载基础数据 ---
    std::cout << "Loading graph from data source..." << std::endl;
    auto [g, flag1] = load_graph<Graph>();
    if (!flag1) return 1;
    std::cout << "Graph loaded successfully. Vertices: " << g.num_vertex << std::endl;

    std::cout << "Loading POIs from data source..." << std::endl;
    auto [pois, flag2] = load_objects();
    if (!flag2) return 1;
    std::cout << "POIs loaded successfully. Count: " << pois.size() << std::endl;

    // --- 2. 处理预计算缓存 ---
    std::string cache_filename = "knn_cache_v" + std::to_string(g.num_vertex) + "_k" + std::to_string(k1) + ".cache";
    PrecomputationCacheFull cache = load_cache_from_file(cache_filename, k1);

    if (cache.empty()) {
        std::cout << "Cache not found or invalid. Starting full precomputation..." << std::endl;
        cache = compute_all_vertices_knn(g, pois, k1);
        save_cache_to_file(cache_filename, cache, k1);
    }

    // --- 3. 准备在线查询所需的数据结构 ---
    // 创建一个 POI ID -> POI* 的映射，用于在线查询时快速转换
    POIMap poi_map;
    for (const auto& poi : pois) {
        poi_map[poi.poi_id] = &poi;
    }

    // --- 4. 执行在线查询 ---
    std::cout << "\nStarting online query with precomputed cache..." << std::endl;
    std::cout << "Query Vertex: " << query_q << ", k1=" << k1 << ", k2=" << k2 << std::endl;

    std::vector<QueryResult> results = find_k_nearest_neighbors_with_cache(g, cache, poi_map, query_q, k2);

    // --- 5. 打印结果 ---
    if (results.empty()) {
        std::cout << "\nNo neighbors found." << std::endl;
    } else {
        std::cout << "\nFound " << results.size() << " nearest neighbors:" << std::endl;
        int rank = 1;
        for (const auto& result : results) {
            std::cout << "  " << std::setw(3) << rank++ << ". "
                      << "POI ID: " << std::setw(6) << result.poi->poi_id
                      << " on Edge(" << result.poi->u << "," << result.poi->v << ")"
                      << ", Distance: " << result.distance
                      << std::endl;
        }
    }

    return 0;
}