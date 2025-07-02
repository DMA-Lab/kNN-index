#ifndef PRECOMPUTATION_H
#define PRECOMPUTATION_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <atomic>
#include <omp.h>
#include "graph.h"

// --- 类型定义 ---
// 用于在缓存中存储的kNN结果，存储POI ID而不是指针
using CachedNNResult = std::pair<Weight, unsigned int>; // {distance, poi_id}
// 缓存的数据结构： 顶点 -> k1-NN列表
using PrecomputationCacheFull = boost::unordered_flat_map<Vertex, std::vector<CachedNNResult>>;

// POI反向索引，预计算时需要
using POIInvertedIndex = boost::unordered_flat_map<Vertex, std::vector<const OnEdgePOI*>>;

// 用于计算过程的临时数据结构
struct QueryResult {
    Weight distance;
    const OnEdgePOI* poi;
    bool operator<(const QueryResult& other) const { return distance < other.distance; }
};
using PQEntry = std::pair<Weight, Vertex>;


/**
 * @brief 为单个顶点计算 k1-NN (这是暴力计算的核心)
 */
std::vector<CachedNNResult> compute_k1_nn_for_vertex_offline(
    const Graph& graph,
    const POIInvertedIndex& poi_index,
    Vertex start_node,
    int k1)
{
    if (k1 <= 0) return {};

    std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<PQEntry>> pq;
    pq.push({0, start_node});

    boost::unordered_flat_map<Vertex, Weight> dist_map;
    dist_map[start_node] = 0;

    std::priority_queue<QueryResult> nearest_pois;
    Weight theta_k1 = std::numeric_limits<Weight>::max();
    boost::unordered_flat_set<unsigned int> found_poi_ids;

    auto update_k1_results = [&](Weight dist, const OnEdgePOI* poi) {
        if (found_poi_ids.count(poi->poi_id)) return;
        if (nearest_pois.size() < k1) {
            nearest_pois.push({dist, poi});
            found_poi_ids.insert(poi->poi_id);
            if (nearest_pois.size() == k1) theta_k1 = nearest_pois.top().distance;
        } else if (dist < theta_k1) {
            const auto& worst_poi = nearest_pois.top();
            found_poi_ids.erase(worst_poi.poi->poi_id);
            nearest_pois.pop();
            nearest_pois.push({dist, poi});
            found_poi_ids.insert(poi->poi_id);
            theta_k1 = nearest_pois.top().distance;
        }
    };

    while (!pq.empty()) {
        auto [current_dist, u] = pq.top();
        pq.pop();
        if (current_dist > theta_k1) break;
        if (dist_map.count(u) && current_dist > dist_map.at(u)) continue;

        if (poi_index.count(u)) {
            for (const auto* poi : poi_index.at(u)) {
                Weight edge_weight = graph.get_weight(poi->u, poi->v);
                if (edge_weight == std::numeric_limits<Weight>::max()) continue;
                Weight offset = (poi->u == u) ? poi->offset_from_u : (edge_weight - poi->offset_from_u);
                update_k1_results(current_dist + offset, poi);
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

    std::vector<CachedNNResult> results;
    results.reserve(nearest_pois.size());
    while(!nearest_pois.empty()){
        auto const& res = nearest_pois.top();
        results.push_back({res.distance, res.poi->poi_id});
        nearest_pois.pop();
    }
    std::reverse(results.begin(), results.end());
    return results;
}

using CachedNNResult = std::pair<Weight, unsigned int>;
using PrecomputationCacheFull = boost::unordered_flat_map<Vertex, std::vector<CachedNNResult>>;
using POIInvertedIndex = boost::unordered_flat_map<Vertex, std::vector<const OnEdgePOI*>>;
using PQEntry = std::pair<Weight, Vertex>;

std::vector<CachedNNResult> compute_k1_nn_for_vertex_offline(const Graph&, const POIInvertedIndex&, Vertex, int); // 声明


/**
 * @brief 暴力计算图中所有顶点的 k1-NN (多核并行版本)
 */
PrecomputationCacheFull compute_all_vertices_knn(
    const Graph& graph,
    const std::vector<OnEdgePOI>& pois,
    int k1)
{
    std::cout << "Starting full precomputation for all " << graph.num_vertex
              << " vertices using " << omp_get_max_threads() << " threads..." << std::endl; // 显示线程数
    auto start_time = std::chrono::high_resolution_clock::now();

    POIInvertedIndex poi_index;
    for (const auto& poi : pois) {
        poi_index[poi.u].push_back(&poi);
        poi_index[poi.v].push_back(&poi);
    }

    PrecomputationCacheFull cache;
    // <<< 3. 使用线程安全的原子计数器
    std::atomic<size_t> processed_count = 0;

    // 将图的顶点集转换为 vector，以便 OpenMP 能更好地进行并行化
    std::vector<Vertex> vertices_vec(graph.vertices.begin(), graph.vertices.end());

    // <<< 4. 使用 OpenMP 并行化 for 循环
    #pragma omp parallel for schedule(dynamic)
    for (int v : vertices_vec) {
        // 每个线程独立计算一个顶点的kNN
        auto result_list = compute_k1_nn_for_vertex_offline(graph, poi_index, v, k1);

        // <<< 5. 使用 critical section 保护对共享 cache 的写入，防止数据竞争
        #pragma omp critical
        {
            cache[v] = std::move(result_list);
        }

        // 更新进度计数器
        size_t current_count = ++processed_count;
        if (current_count % 1000 == 0) {
            // 使用 \r 会导致多线程输出混乱，改为每1000次打印一行新状态
            // 为了避免输出过于频繁，只让一个线程来打印
            #pragma omp critical
            {
                // 再次检查，防止多个线程同时满足条件
                if (current_count % 1000 == 0) {
                     std::cout << "Processed " << current_count << " / " << graph.num_vertex << " vertices...\n";
                }
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
    std::cout << "\nFull precomputation finished in " << elapsed.count() / 1000.0 << " seconds." << std::endl;

    return cache;
}

/**
 * @brief 将预计算缓存写入文件
 */
bool save_cache_to_file(const std::string& filename, const PrecomputationCacheFull& cache, int k1) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        std::cerr << "Error: Cannot open file for writing: " << filename << std::endl;
        return false;
    }

    // 写入头部信息
    const uint32_t MAGIC_NUMBER = 0x4B4E4E43; // "KNNC"
    ofs.write(reinterpret_cast<const char*>(&MAGIC_NUMBER), sizeof(MAGIC_NUMBER));
    uint32_t k1_val = k1;
    ofs.write(reinterpret_cast<const char*>(&k1_val), sizeof(k1_val));
    uint64_t num_vertices = cache.size();
    ofs.write(reinterpret_cast<const char*>(&num_vertices), sizeof(num_vertices));

    // 写入每个顶点的数据
    for (const auto& [vertex, nn_list] : cache) {
        uint32_t v_id = vertex;
        ofs.write(reinterpret_cast<const char*>(&v_id), sizeof(v_id));
        uint32_t list_size = nn_list.size();
        ofs.write(reinterpret_cast<const char*>(&list_size), sizeof(list_size));
        ofs.write(reinterpret_cast<const char*>(nn_list.data()), list_size * sizeof(CachedNNResult));
    }

    std::cout << "Cache successfully saved to " << filename << std::endl;
    return true;
}

/**
 * @brief 从文件加载预计算缓存
 */
PrecomputationCacheFull load_cache_from_file(const std::string& filename, int expected_k1) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        // 文件不存在是正常情况，静默返回
        return {};
    }

    // 校验头部信息
    uint32_t magic_number;
    ifs.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    if (magic_number != 0x4B4E4E43) {
        std::cerr << "Error: Invalid cache file format." << std::endl;
        return {};
    }

    uint32_t k1_val;
    ifs.read(reinterpret_cast<char*>(&k1_val), sizeof(k1_val));
    if (k1_val != expected_k1) {
        std::cerr << "Warning: Cache file k1 (" << k1_val << ") does not match expected k1 (" << expected_k1 << "). Ignoring cache." << std::endl;
        return {};
    }

    uint64_t num_vertices;
    ifs.read(reinterpret_cast<char*>(&num_vertices), sizeof(num_vertices));

    PrecomputationCacheFull cache;
    cache.reserve(num_vertices);

    for (uint64_t i = 0; i < num_vertices; ++i) {
        uint32_t v_id;
        ifs.read(reinterpret_cast<char*>(&v_id), sizeof(v_id));
        uint32_t list_size;
        ifs.read(reinterpret_cast<char*>(&list_size), sizeof(list_size));
        std::vector<CachedNNResult> nn_list(list_size);
        ifs.read(reinterpret_cast<char*>(nn_list.data()), list_size * sizeof(CachedNNResult));
        cache[v_id] = std::move(nn_list);
    }
    
    std::cout << "Cache successfully loaded from " << filename << std::endl;
    return cache;
}

#endif // PRECOMPUTATION_H