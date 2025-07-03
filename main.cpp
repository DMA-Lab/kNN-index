#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <algorithm>
#include <utility>
#include <set>
#include <chrono>
#include <iomanip>
#include <functional>
#include <ranges>
#include "boost/unordered/unordered_flat_map.hpp"
#include "boost/unordered/unordered_flat_set.hpp"
#include "graph.h" // 假设包含 Graph2 类定义
#include "file.h"  // 假设包含 load_graph, load_objects

// --- 辅助工具：计时器 ---
class Timer {
public:
    Timer() : start_time(std::chrono::high_resolution_clock::now()) {}
    void reset() { start_time = std::chrono::high_resolution_clock::now(); }
    double elapsed() const {
        return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    }
    void print_elapsed(const std::string& prefix = "") const {
        std::cout << prefix << std::fixed << std::setprecision(4) << elapsed() << " 秒" << std::endl;
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
};

// --- 核心数据结构 ---

// kNN查询结果中的一个邻居
struct Neighbor {
    unsigned int id; // 对象的唯一ID (poi_id)
    Weight dist;
    bool operator>(const Neighbor& other) const { return dist > other.dist; }
    bool operator<(const Neighbor& other) const { return dist < other.dist; }
};
using KnnResult = std::vector<Neighbor>;

// 用于从文件加载的原始对象结构
struct LoadedObject {
    unsigned int poi_id;
    Vertex u, v;
    double ratio;
};

// --- 1. 静态的、可持久化的 V2V (Vertex-to-Vertex) 索引 ---
template <typename Graph>
class V2VIndex {
public:
    V2VIndex() = default; // for loading
    V2VIndex(const Graph& g, int k);

    void build();
    bool save(const std::string& path) const;
    bool load(const std::string& path, const Graph& g); // 需要原始图来恢复BN-Graph

    int get_index_k() const { return k_index_; }
    const std::vector<std::pair<Vertex, Weight>>& get_vertex_knn(Vertex u) const;
    const Graph& get_graph() const { return *graph_ptr_; }

private:
    void generate_vertex_order();
    void build_bn_graph();
    void build_vertex_knn_cache();

    const Graph* graph_ptr_ = nullptr; // 指向原始图
    int k_index_ = 0;

    Graph bn_graph_;
    std::vector<Vertex> vertex_order_pi_;
    boost::unordered_flat_map<Vertex, std::vector<std::pair<Vertex, Weight>>> vertex_knn_cache_;

    // 用于load时返回空vector的静态成员
    inline static const std::vector<std::pair<Vertex, Weight>> empty_v_knn_{};
};

template <typename Graph>
V2VIndex<Graph>::V2VIndex(const Graph& g, int k) : graph_ptr_(&g), k_index_(k) {}

template <typename Graph>
const std::vector<std::pair<Vertex, Weight>>& V2VIndex<Graph>::get_vertex_knn(Vertex u) const {
    auto it = vertex_knn_cache_.find(u);
    if (it != vertex_knn_cache_.end()) {
        return it->second;
    }
    return empty_v_knn_;
}

template <typename Graph>
void V2VIndex<Graph>::build() {
    if (!graph_ptr_ || k_index_ <= 0) {
        std::cerr << "[错误] V2VIndex 未正确初始化！" << std::endl;
        return;
    }
    std::cout << "开始构建 V2V 索引 (k=" << k_index_ << ")..." << std::endl;
    Timer total_timer;

    std::cout << "  (1/3) 生成顶点排序..." << std::endl;
    generate_vertex_order();

    std::cout << "  (2/3) 构建 BN-Graph..." << std::endl;
    build_bn_graph();

    std::cout << "  (3/3) 构建 V-V kNN 缓存..." << std::endl;
    build_vertex_knn_cache();

    total_timer.print_elapsed("V2V 索引构建总耗时: ");
}

template <typename Graph>
void V2VIndex<Graph>::generate_vertex_order() {
    if (!graph_ptr_) return;
    const Graph& g = *graph_ptr_;

    vertex_order_pi_.reserve(g.num_vertex);

    // 用来追踪每个顶点的当前度数
    boost::unordered_flat_map<Vertex, int> current_degrees;
    // 使用 std::set 模拟一个优先队列，它会根据度数和顶点ID自动排序
    // pair: {degree, vertex_id}
    std::set<std::pair<int, Vertex>> degree_pq;

    // 初始化度数和优先队列
    for (const auto& v : g.vertices) {
        if (v == 0) continue; // 假设顶点0无效或不存在
        int degree = g.out_degree(v);
        current_degrees[v] = degree;
        degree_pq.insert({degree, v});
    }

    while (!degree_pq.empty()) {
        // 直接从优先队列的开头获取度数最小的顶点 (O(log N))
        auto it = degree_pq.begin();
        Vertex best_v = it->second;

        // 从优先队列中移除
        degree_pq.erase(it);

        vertex_order_pi_.push_back(best_v);

        // 更新其邻居的度数
        for (const auto& [neighbor_v, _] : g.get_adjacent_vertices(best_v)) {
            // 如果邻居在我们的追踪列表里 (即它是一个有效的、未处理的顶点)
            if (current_degrees.count(neighbor_v)) {
                // 在优先队列中更新一个元素的最高效方法是：删除旧的，插入新的
                if (degree_pq.erase({current_degrees[neighbor_v], neighbor_v})) {
                    --current_degrees[neighbor_v];
                    degree_pq.insert({current_degrees[neighbor_v], neighbor_v});
                }
            }
        }
        // 从追踪map中移除已处理的顶点，防止在后续的邻居更新中被再次操作
        current_degrees.erase(best_v);
    }
}

template <typename Graph>
void V2VIndex<Graph>::build_bn_graph() {
    if (!graph_ptr_) return;
    const Graph& g = *graph_ptr_;

    // 初始化 BN-Graph 为原始图的副本
    bn_graph_ = g;

    // 创建一个从顶点ID到其在排序中位置（rank）的映射，方便快速查找
    boost::unordered_flat_map<Vertex, int> rank;
    for (size_t i = 0; i < vertex_order_pi_.size(); ++i) {
        rank[vertex_order_pi_[i]] = i;
    }

    // --- 步骤 1: 边插入 (Edge Insertion) ---
    for (const auto& w : vertex_order_pi_) {
        std::vector<Vertex> higher_rank_neighbors;
        for (const auto& [neighbor, _] : bn_graph_.get_adjacent_vertices(w)) {
            if (rank.count(neighbor) && rank.at(neighbor) > rank.at(w)) {
                higher_rank_neighbors.push_back(neighbor);
            }
        }

        for (size_t i = 0; i < higher_rank_neighbors.size(); ++i) {
            for (size_t j = i + 1; j < higher_rank_neighbors.size(); ++j) {
                Vertex u = higher_rank_neighbors[i];
                Vertex v = higher_rank_neighbors[j];
                Weight new_dist = bn_graph_.get_weight(u, w) + bn_graph_.get_weight(w, v);
                if (!bn_graph_.has_edge(u, v) || new_dist < bn_graph_.get_weight(u, v)) {
                    bn_graph_.connect(u, v, new_dist);
                }
            }
        }
    }

    // --- 步骤 2: 边删除/剪枝 (Edge Deletion) ---
    boost::unordered_flat_set<std::pair<Vertex, Vertex>> edges_to_remove;
    for (int w : std::ranges::reverse_view(vertex_order_pi_)) {
        std::vector<Vertex> higher_rank_neighbors;
        for (const auto& [neighbor, _] : bn_graph_.get_adjacent_vertices(w)) {
            if (rank.count(neighbor) && rank.at(neighbor) > rank.at(w)) {
                higher_rank_neighbors.push_back(neighbor);
            }
        }

        for (size_t i = 0; i < higher_rank_neighbors.size(); ++i) {
            for (size_t j = i + 1; j < higher_rank_neighbors.size(); ++j) {
                Vertex u = higher_rank_neighbors[i];
                Vertex v = higher_rank_neighbors[j];

                Weight dist_wu = bn_graph_.get_weight(w, u);
                Weight dist_wv = bn_graph_.get_weight(w, v);
                Weight dist_uv = bn_graph_.get_weight(u, v);

                if (dist_wv + dist_uv < dist_wu) {
                    edges_to_remove.insert({std::min(w, u), std::max(w, u)});
                }
                if (dist_wu + dist_uv < dist_wv) {
                    edges_to_remove.insert({std::min(w, v), std::max(w, v)});
                }
            }
        }
    }

    if (!edges_to_remove.empty()) {
        std::cout << "    - 剪枝 " << edges_to_remove.size() << " 条冗余边从 BN-Graph." << std::endl;
        for (const auto& edge_pair : edges_to_remove) {
            bn_graph_.disconnect(edge_pair.first, edge_pair.second);
        }
    }
}

template <typename Graph>
void V2VIndex<Graph>::build_vertex_knn_cache() {
    boost::unordered_flat_map<Vertex, int> rank;
    for (size_t i = 0; i < vertex_order_pi_.size(); ++i) {
        rank[vertex_order_pi_[i]] = i;
    }

    using VertexNeighbor = std::pair<Vertex, Weight>;

    // --- 阶段 1: 自底向上计算 partial V-V kNN ---
    // 使用最大堆（按距离比较）来维护k个最近的顶点邻居
    auto max_heap_comp = [](const VertexNeighbor& a, const VertexNeighbor& b) { return a.second < b.second; };
    using MaxHeap = std::priority_queue<VertexNeighbor, std::vector<VertexNeighbor>, decltype(max_heap_comp)>;

    boost::unordered_flat_map<Vertex, std::vector<VertexNeighbor>> partial_v_knn;

    for (const auto& u : vertex_order_pi_) {
        MaxHeap max_heap(max_heap_comp);
        max_heap.push({u, 0}); // 到自身的距离为0

        // 从低阶邻居 v 传播 kNN 结果
        for (const auto& [v, w_uv] : bn_graph_.get_adjacent_vertices(u)) {
            if (!rank.count(v) || rank.at(v) > rank.at(u)) continue;
            if (!partial_v_knn.count(v)) continue;

            for (const auto& [neighbor_of_v, dist_vn] : partial_v_knn.at(v)) {
                Weight new_dist = w_uv + dist_vn;
                if (max_heap.size() < static_cast<size_t>(k_index_)) {
                    max_heap.push({neighbor_of_v, new_dist});
                } else if (new_dist < max_heap.top().second) {
                    max_heap.pop();
                    max_heap.push({neighbor_of_v, new_dist});
                }
            }
        }

        std::vector<VertexNeighbor> result;
        result.reserve(max_heap.size());
        while (!max_heap.empty()) {
            result.push_back(max_heap.top());
            max_heap.pop();
        }
        std::sort(result.begin(), result.end(), [](const auto& a, const auto& b){ return a.second < b.second; });
        partial_v_knn[u] = std::move(result);
    }

    // --- 阶段 2: 自顶向下计算最终 V-V kNN ---
    // 使用最小堆来合并多个有序列表
    auto min_heap_comp = [](const VertexNeighbor& a, const VertexNeighbor& b) { return a.second > b.second; };
    using MinHeap = std::priority_queue<VertexNeighbor, std::vector<VertexNeighbor>, decltype(min_heap_comp)>;

    for (int u : std::ranges::reverse_view(vertex_order_pi_)) {
        MinHeap min_heap(min_heap_comp);
        boost::unordered_flat_set<Vertex> visited; // 用于合并时去重

        // 1. 加入来自 partial_knn 的结果
        if (partial_v_knn.count(u)) {
            for (const auto& neighbor : partial_v_knn.at(u)) {
                min_heap.push(neighbor);
                visited.insert(neighbor.first);
            }
        }

        // 2. 加入来自高阶邻居 w 的最终 kNN 结果
        for (const auto& [w, w_uw] : bn_graph_.get_adjacent_vertices(u)) {
            if (!rank.count(w) || rank.at(w) < rank.at(u)) continue;
            if (!vertex_knn_cache_.count(w)) continue;

            for (const auto& [neighbor_of_w, dist_wn] : vertex_knn_cache_.at(w)) {
                if (!visited.count(neighbor_of_w)) {
                    min_heap.push({neighbor_of_w, w_uw + dist_wn});
                    visited.insert(neighbor_of_w);
                }
            }
        }

        // 从合并后的结果中选出最终的 Top-K
        std::vector<VertexNeighbor> final_knn;
        final_knn.reserve(k_index_);
        boost::unordered_flat_set<Vertex> final_visited_ids;
        while (!min_heap.empty() && final_knn.size() < static_cast<size_t>(k_index_)) {
            VertexNeighbor top = min_heap.top();
            min_heap.pop();
            if(!final_visited_ids.count(top.first)){
               final_knn.push_back(top);
               final_visited_ids.insert(top.first);
            }
        }
        vertex_knn_cache_[u] = std::move(final_knn);
    }
}

template <typename Graph>
bool V2VIndex<Graph>::save(const std::string& path) const {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        std::cerr << "[错误] 无法打开文件进行写入: " << path << std::endl;
        return false;
    }

    // 写入元数据
    ofs.write(reinterpret_cast<const char*>(&k_index_), sizeof(k_index_));

    // 写入 vertex_order_pi_
    size_t pi_size = vertex_order_pi_.size();
    ofs.write(reinterpret_cast<const char*>(&pi_size), sizeof(pi_size));
    ofs.write(reinterpret_cast<const char*>(vertex_order_pi_.data()), pi_size * sizeof(Vertex));

    // 写入 vertex_knn_cache_
    size_t cache_size = vertex_knn_cache_.size();
    ofs.write(reinterpret_cast<const char*>(&cache_size), sizeof(cache_size));
    for (const auto& [u, neighbors] : vertex_knn_cache_) {
        ofs.write(reinterpret_cast<const char*>(&u), sizeof(u));
        size_t neighbors_size = neighbors.size();
        ofs.write(reinterpret_cast<const char*>(&neighbors_size), sizeof(neighbors_size));
        ofs.write(reinterpret_cast<const char*>(neighbors.data()), neighbors_size * sizeof(std::pair<Vertex, Weight>));
    }

    std::cout << "V2V 索引已成功保存到: " << path << std::endl;
    return true;
}

template <typename Graph>
bool V2VIndex<Graph>::load(const std::string& path, const Graph& g) {
    std::ifstream ifs;
    ifs.exceptions(std::ifstream::failbit | std::ifstream::badbit); // 设置异常处理
    ifs.open(path, std::ios::binary);

    graph_ptr_ = &g;

    // 读取元数据并检查
    if (!ifs.read(reinterpret_cast<char*>(&k_index_), sizeof(k_index_))) return false;

    // 读取 vertex_order_pi_
    size_t pi_size;
    ifs.read(reinterpret_cast<char*>(&pi_size), sizeof(pi_size));
    vertex_order_pi_.resize(pi_size);
    ifs.read(reinterpret_cast<char*>(vertex_order_pi_.data()), pi_size * sizeof(Vertex));

    // 重建 BN-Graph (必须，因为查询处理器可能需要)
    std::cout << "从加载的数据重建 BN-Graph..." << std::endl;
    build_bn_graph();

    // 读取 vertex_knn_cache_
    size_t cache_size;
    ifs.read(reinterpret_cast<char*>(&cache_size), sizeof(cache_size));
    vertex_knn_cache_.reserve(cache_size);
    for (size_t i = 0; i < cache_size; ++i) {
        Vertex u;
        ifs.read(reinterpret_cast<char*>(&u), sizeof(u));
        size_t neighbors_size;
        ifs.read(reinterpret_cast<char*>(&neighbors_size), sizeof(neighbors_size));
        std::vector<std::pair<Vertex, Weight>> neighbors(neighbors_size);
        ifs.read(reinterpret_cast<char*>(neighbors.data()), neighbors_size * sizeof(std::pair<Vertex, Weight>));
        vertex_knn_cache_[u] = std::move(neighbors);
    }

    std::cout << "V2V 索引已成功从 " << path << " 加载 (k=" << k_index_ << ")" << std::endl;
    return true;
}


// --- 2. 动态的 kNN 查询处理器 ---
template <typename Graph>
class KnnQueryProcessor {
public:
    KnnQueryProcessor(const V2VIndex<Graph>& index);
    KnnResult query(Vertex query_point, const std::vector<OnEdgePOI>& pois, int k_query) const;

private:
    const V2VIndex<Graph>& index_;
    boost::unordered_flat_map<Vertex, std::vector<size_t>> build_poi_map(const std::vector<OnEdgePOI>& pois) const;
};

template <typename Graph>
KnnQueryProcessor<Graph>::KnnQueryProcessor(const V2VIndex<Graph>& index) : index_(index) {}

template<typename Graph>
boost::unordered_flat_map<Vertex, std::vector<size_t>> KnnQueryProcessor<Graph>::build_poi_map(const std::vector<OnEdgePOI>& pois) const {
    boost::unordered_flat_map<Vertex, std::vector<size_t>> poi_map;
    for (size_t i = 0; i < pois.size(); ++i) {
        poi_map[pois[i].u].push_back(i);
        poi_map[pois[i].v].push_back(i);
    }
    return poi_map;
}

template <typename Graph>
KnnResult KnnQueryProcessor<Graph>::query(Vertex query_point, const std::vector<OnEdgePOI>& pois, int k_query) const {
    Timer query_timer;
    std::cout << "执行查询: u=" << query_point << ", k=" << k_query << ", POI数量=" << pois.size() << std::endl;

    // 动态构建本次查询的 POI 反向索引
    auto poi_map = build_poi_map(pois);

    std::priority_queue<Neighbor> max_heap;

    // --- 阶段1: 利用 V2V 索引进行初步查询 ---
    std::cout << "  - 阶段1: 使用 V2V 索引 (k=" << index_.get_index_k() << ") 查找种子结果..." << std::endl;
    const auto& nearest_vertices = index_.get_vertex_knn(query_point);
    for (const auto& [v, dist_uv] : nearest_vertices) {
        if (!poi_map.count(v)) continue;

        for (size_t poi_idx : poi_map.at(v)) {
            const auto& poi = pois[poi_idx];
            Weight total_dist = dist_uv;
            if (poi.u == v) {
                total_dist += poi.offset_from_u;
            } else {
                total_dist += (index_.get_graph().get_weight(poi.u, poi.v) - poi.offset_from_u);
            }
            if (max_heap.size() < static_cast<size_t>(k_query)) {
                max_heap.push({poi.poi_id, total_dist});
            } else if (total_dist < max_heap.top().dist) {
                max_heap.pop();
                max_heap.push({poi.poi_id, total_dist});
            }
        }
    }

    // --- 阶段2: 如果需要，使用 Dijkstra 进行回退查询 ---
    if (k_query > index_.get_index_k()) {
        std::cout << "  - 阶段2: 查询k > 索引k，启动 Dijkstra 回退..." << std::endl;
        Weight search_bound = std::numeric_limits<Weight>::max();
        if (max_heap.size() == static_cast<size_t>(k_query)) {
            search_bound = max_heap.top().dist;
        }

        using DijkstraState = std::pair<Weight, Vertex>;
        std::priority_queue<DijkstraState, std::vector<DijkstraState>, std::greater<DijkstraState>> pq;
        boost::unordered_flat_map<Vertex, Weight> dists;

        pq.push({0, query_point});
        dists[query_point] = 0;

        while (!pq.empty()) {
            auto [dist, u] = pq.top();
            pq.pop();

            if (dist > dists[u]) continue;
            if (dist > search_bound) break; // 剪枝

            // 检查附着在u上的POI
            if (poi_map.count(u)) {
                for (size_t poi_idx : poi_map.at(u)) {
                    const auto& poi = pois[poi_idx];
                    Weight total_dist = dist;
                     if (poi.u == u) {
                        total_dist += poi.offset_from_u;
                    } else {
                        total_dist += (index_.get_graph().get_weight(poi.u, poi.v) - poi.offset_from_u);
                    }

                    if (max_heap.size() < static_cast<size_t>(k_query)) {
                        max_heap.push({poi.poi_id, total_dist});
                         if(max_heap.size() == static_cast<size_t>(k_query)) search_bound = max_heap.top().dist;
                    } else if (total_dist < max_heap.top().dist) {
                        max_heap.pop();
                        max_heap.push({poi.poi_id, total_dist});
                        search_bound = max_heap.top().dist;
                    }
                }
            }

            // 扩展邻居
            for (const auto& [v, weight] : index_.get_graph().get_adjacent_vertices(u)) {
                if (!dists.count(v) || dists.at(v) > dist + weight) {
                    dists[v] = dist + weight;
                    pq.push({dists[v], v});
                }
            }
        }
    }

    // --- 整理并返回结果 ---
    KnnResult final_result;
    while (!max_heap.empty()) {
        final_result.push_back(max_heap.top());
        max_heap.pop();
    }
    std::sort(final_result.begin(), final_result.end());

    query_timer.print_elapsed("总查询耗时: ");
    return final_result;
}

// --- main 函数：演示工作流 ---
int main() {
    const std::string INDEX_FILE = "v2v_index.bin";
    const int K_INDEX = 50; // 构建索引时使用的k，应该比通常查询的k大

    // 1. 加载图
    auto [g, success] = load_graph<Graph2>();
    if (!success) return 1;

    // 2. 构建或加载 V2V 索引
    V2VIndex<Graph2> index;
    bool mov_success = false;
    // try {
    //     index.load(INDEX_FILE, g);
    //     if (index.get_index_k() != K_INDEX) {
    //         std::cout << "警告: 加载的索引k值(" << index.get_index_k()
    //                   << ")与期望值(" << K_INDEX << ")不符。将继续使用加载的索引。" << std::endl;
    //     }
    //     else {
    //         mov_success = true;
    //     }
    // } catch (const std::ifstream::failure& e) {}

    if (!mov_success) {
        std::cout << "未发现索引文件，开始构建新索引..." << std::endl;
        index = V2VIndex<Graph2>(g, K_INDEX);
        index.build();
        if (!index.save(INDEX_FILE)) {
            std::cerr << "[错误] 无法保存索引到文件: " << INDEX_FILE << std::endl;
        }
    }

    // 3. 准备 POI 数据 (这里模拟动态接收)
    auto [pois, success2] = load_objects();
    if (!success2) return 1;
    std::cout << "加载了 " << pois.size() << " 个 POI 对象。" << std::endl;

    // 4. 创建查询处理器并执行查询
    KnnQueryProcessor<Graph2> qp(index);
    Vertex query_point = 600;

    // 查询1: k < 索引k (纯索引模式)
    std::cout << "\n--- 查询1: k=10 (小于索引k=" << index.get_index_k() << ") ---" << std::endl;
    KnnResult result1 = qp.query(query_point, pois, 10);
    for (const auto& n : result1) std::cout << "  - POI: " << n.id << ", Dist: " << n.dist << std::endl;

    // 查询2: k > 索引k (混合模式)
    std::cout << "\n--- 查询2: k=100 (大于索引k=" << index.get_index_k() << ") ---" << std::endl;
    KnnResult result2 = qp.query(query_point, pois, 100);
    for (const auto& n : result2) std::cout << "  - POI: " << n.id << ", Dist: " << n.dist << std::endl;

    return 0;
}