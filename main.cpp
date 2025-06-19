#include <iostream>
#include <ranges>
#include <vector>
#include <queue>
#include <algorithm>
#include <utility>
#include <set> // 使用 std::set 作为优先队列
#include <chrono> // 添加计时功能
#include <iomanip> // 用于格式化输出
#include "boost/unordered/unordered_flat_map.hpp"
#include "boost/unordered/unordered_flat_set.hpp"
#include "graph.h"
#include "file.h"

// 计时器类
class Timer {
public:
    Timer() : start_time(std::chrono::high_resolution_clock::now()) {}
    
    // 重置计时器
    void reset() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    // 返回从创建或上次重置以来经过的时间（秒）
    double elapsed() const {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = now - start_time;
        return diff.count();
    }
    
    // 打印经过的时间并返回秒数
    double print_elapsed(const std::string& prefix = "") const {
        double seconds = elapsed();
        std::cout << prefix << std::fixed << std::setprecision(3) 
                  << seconds << " 秒" << std::endl;
        return seconds;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
};

// --- kNN 相关数据结构 ---
struct Neighbor {
    Vertex id;
    Weight dist;
    bool operator>(const Neighbor& other) const { return dist > other.dist; }
    bool operator<(const Neighbor& other) const { return dist < other.dist; }
};
using KnnResult = std::vector<Neighbor>;


/**
 * @brief 创建增强图，将边上对象转换为新顶点
 * @param original_graph 原始路网图
 * @param pois 边上对象列表
 * @param poi_vertex_start_id 新的 POI 顶点 ID 的起始编号，必须大于原始图中最大的顶点ID
 * @return 一个包含增强图和新候选对象集的 pair
 */
template <typename Graph>
std::pair<Graph, boost::unordered_flat_set<Vertex>>
create_augmented_graph(
    const Graph& original_graph,
    const std::vector<OnEdgePOI>& pois,
    Vertex poi_vertex_start_id
) {
    std::cout << "创建增强图中..." << std::endl;
    Timer timer;
    
    Graph augmented_graph = original_graph;
    boost::unordered_flat_set<Vertex> new_candidate_objects;

    Vertex current_new_vertex_id = poi_vertex_start_id;
    for (const auto& poi : pois) {
        Vertex new_poi_vertex = current_new_vertex_id++;

        augmented_graph.insert(new_poi_vertex);
        new_candidate_objects.insert(new_poi_vertex);

        Vertex u = poi.u;
        Vertex v = poi.v;
        Weight original_weight = original_graph.get_weight(u, v);

        // 断开原始边，连接到新顶点
        augmented_graph.disconnect(u, v);
        augmented_graph.connect(u, new_poi_vertex, poi.offset_from_u);
        augmented_graph.connect(new_poi_vertex, v, original_weight - poi.offset_from_u);
        std::cout << "  - POI " << poi.poi_id << " 已作为新顶点 " << new_poi_vertex << " 嵌入图中。" << std::endl;
    }

    std::cout << "增强图创建完毕。新图顶点数: " << augmented_graph.num_vertex << std::endl;
    timer.print_elapsed("增强图创建耗时: ");
    return {augmented_graph, new_candidate_objects};
}


// --- KnnIndex 类定义 ---
// 这个类现在将直接在增强图上工作，其内部逻辑无需改变。
template <typename Graph>
class KnnIndex {
public:
    KnnIndex(const Graph& g, int k, const boost::unordered_flat_set<Vertex>& objects);
    void build_index();
    [[nodiscard]] KnnResult get_knn(Vertex u) const;

private:
    const Graph& graph_;
    const int k_;
    boost::unordered_flat_set<Vertex> candidate_objects_;

    Graph bn_graph_;
    std::vector<Vertex> vertex_order_pi_;
    boost::unordered_flat_map<Vertex, KnnResult> knn_index_;

    void generate_vertex_order();
    void build_bn_graph();
    void build_index_internal();
    boost::unordered_flat_map<Vertex, KnnResult> build_partial_knn();
};

// KnnIndex 类的实现...
// (这部分代码与上一版基本相同，因为它现在操作的是一个所有对象都已是顶点的图)
template <typename Graph>
KnnIndex<Graph>::KnnIndex(const Graph& g, int k, const boost::unordered_flat_set<Vertex>& objects)
    : graph_(g), k_(k), candidate_objects_(objects) {
    if (candidate_objects_.empty()) {
        for (const auto& v : graph_.vertices) {
            candidate_objects_.insert(v);
        }
    }
}

template <typename Graph>
void KnnIndex<Graph>::build_index() {
    std::cout << "开始在图上构建 kNN 索引..." << std::endl;
    std::cout << "  - 图顶点数: " << graph_.num_vertex << std::endl;
    std::cout << "  - 候选对象数: " << candidate_objects_.size() << std::endl;
    Timer total_timer;

    std::cout << "  (1/3) 生成顶点排序..." << std::endl;
    Timer step_timer;
    generate_vertex_order();
    step_timer.print_elapsed("    顶点排序耗时: ");

    std::cout << "  (2/3) 构建桥接邻居保留图 (BN-Graph)..." << std::endl;
    step_timer.reset();
    build_bn_graph();
    step_timer.print_elapsed("    BN-Graph构建总耗时: ");

    std::cout << "  (3/3) 运行双向算法填充索引..." << std::endl;
    step_timer.reset();
    build_index_internal();
    step_timer.print_elapsed("    索引填充耗时: ");

    std::cout << "kNN 索引构建完成！" << std::endl;
    total_timer.print_elapsed("索引构建总耗时: ");
}

template <typename Graph>
KnnResult KnnIndex<Graph>::get_knn(Vertex u) const {
    Timer timer;
    auto it = knn_index_.find(u);
    KnnResult result;
    if (it != knn_index_.end()) {
        result = it->second;
    }
    timer.print_elapsed("kNN查询耗时: ");
    return result;
}

template <typename Graph>
void KnnIndex<Graph>::generate_vertex_order() {
    // MODIFIED: Replaced O(N^2) linear scan with an O(N log N) priority queue approach.
    vertex_order_pi_.reserve(graph_.num_vertex);

    // 用来追踪每个顶点的当前度数
    boost::unordered_flat_map<Vertex, int> current_degrees;
    // 使用 std::set 模拟一个优先队列，它会根据度数自动排序
    // pair: {degree, vertex_id}
    std::set<std::pair<int, Vertex>> degree_pq;

    // 初始化度数和优先队列
    for (const auto& v : graph_.vertices) {
        if (v == 0) continue; // 遵守规则：顶点0是无效的
        int degree = graph_.out_degree(v);
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
        for (const auto& neighbor_pair : graph_.get_adjacent_vertices(best_v)) {
            Vertex neighbor_v = neighbor_pair.first;

            // 如果邻居在我们的追踪列表里 (即它是一个有效的、未处理的顶点)
            if (current_degrees.count(neighbor_v)) {
                // 在优先队列中更新一个元素的最高效方法是：删除旧的，插入新的

                // 1. 删除旧的条目
                if (degree_pq.erase({current_degrees[neighbor_v], neighbor_v})) {
                    // 2. 更新度数
                    current_degrees[neighbor_v]--;
                    // 3. 插入新条目
                    degree_pq.insert({current_degrees[neighbor_v], neighbor_v});
                }
            }
        }
        // 从追踪map中移除已处理的顶点，防止在后续的邻居更新中被再次操作
        current_degrees.erase(best_v);
    }
}

template <typename Graph>
void KnnIndex<Graph>::build_bn_graph() {
    // 初始化 BN-Graph 为原始图的副本
    bn_graph_ = graph_;

    // 创建一个从顶点ID到其在排序中位置（rank）的映射，方便快速查找
    boost::unordered_flat_map<Vertex, int> rank;
    for (size_t i = 0; i < vertex_order_pi_.size(); ++i) {
        rank[vertex_order_pi_[i]] = i;
    }

    // --- 步骤 1: 边插入 (Edge Insertion) ---
    // 按顶点排序的升序遍历 (w 的 rank 从低到高)
    std::cout << "    - BN-Graph: Running Edge Insertion..." << std::endl;
    Timer edge_insertion_timer;
    for (const auto& w : vertex_order_pi_) {
        std::vector<Vertex> higher_rank_neighbors;
        // 找到所有 rank 比 w 高的邻居
        for (const auto& neighbor_pair : bn_graph_.get_adjacent_vertices(w)) {
            Vertex neighbor = neighbor_pair.first;
            if (rank.count(neighbor) && rank.at(neighbor) > rank.at(w)) {
                higher_rank_neighbors.push_back(neighbor);
            }
        }

        // 为这些高阶邻居两两之间添加快捷方式边
        for (size_t i = 0; i < higher_rank_neighbors.size(); ++i) {
            for (size_t j = i + 1; j < higher_rank_neighbors.size(); ++j) {
                Vertex u = higher_rank_neighbors[i];
                Vertex v = higher_rank_neighbors[j];
                // 路径 u -> w -> v 的长度
                Weight new_dist = bn_graph_.get_weight(u, w) + bn_graph_.get_weight(w, v);
                if (!bn_graph_.has_edge(u, v) || new_dist < bn_graph_.get_weight(u, v)) {
                    bn_graph_.connect(u, v, new_dist);
                }
            }
        }
    }
    edge_insertion_timer.print_elapsed("      边插入耗时: ");

    // --- 步骤 2: 边删除/剪枝 (Edge Deletion) ---
    // 这是新增的部分，对应论文算法1的第10-16行
    std::cout << "    - BN-Graph: Running Edge Deletion/Pruning..." << std::endl;
    Timer edge_deletion_timer;
    
    // 使用一个集合来存储待删除的边，避免在遍历时直接修改图结构
    boost::unordered_flat_set<std::pair<Vertex, Vertex>> edges_to_remove;

    // 按顶点排序的降序遍历 (w 的 rank 从高到低)
    for (int w : std::ranges::reverse_view(vertex_order_pi_)) {
        std::vector<Vertex> higher_rank_neighbors;
        // 同样，找到所有 rank 比 w 高的邻居
        for (const auto& neighbor_pair : bn_graph_.get_adjacent_vertices(w)) {
            Vertex neighbor = neighbor_pair.first;
            if (rank.count(neighbor) && rank.at(neighbor) > rank.at(w)) {
                higher_rank_neighbors.push_back(neighbor);
            }
        }

        // 检查这些高阶邻居，利用三角不等式进行剪枝
        for (size_t i = 0; i < higher_rank_neighbors.size(); ++i) {
            for (size_t j = i + 1; j < higher_rank_neighbors.size(); ++j) {
                Vertex u = higher_rank_neighbors[i];
                Vertex v = higher_rank_neighbors[j];

                // 检查路径 w -> u -> v 是否比 w -> v 更优
                // 注意：由于图是无向的，我们需要检查两个方向
                // 论文伪代码中 `phi((w, v), G') + phi((v, u), G') < phi((w, u), G')`
                // 对应到我们的实现就是 dist(w,v) + dist(v,u) < dist(w,u)

                Weight dist_wu = bn_graph_.get_weight(w, u);
                Weight dist_wv = bn_graph_.get_weight(w, v);
                Weight dist_uv = bn_graph_.get_weight(u, v);

                // 检查路径 w-v-u 是否可以优化 w-u
                if (dist_wv + dist_uv < dist_wu) {
                    // 如果是，说明边(w, u)是冗余的，可以被移除
                    // 为了保证对称性，我们对顶点ID排序来规范化pair
                    edges_to_remove.insert({std::min(w, u), std::max(w, u)});
                }

                // 检查路径 w-u-v 是否可以优化 w-v
                if (dist_wu + dist_uv < dist_wv) {
                    // 如果是，说明边(w, v)是冗余的，可以被移除
                    edges_to_remove.insert({std::min(w, v), std::max(w, v)});
                }
            }
        }
    }

    // 最后，从图中实际删除所有被标记的边
    if (!edges_to_remove.empty()) {
        std::cout << "    - Pruning " << edges_to_remove.size() << " redundant edges from BN-Graph." << std::endl;
        for (const auto& edge_pair : edges_to_remove) {
            bn_graph_.disconnect(edge_pair.first, edge_pair.second);
        }
    }
    edge_deletion_timer.print_elapsed("      边删除/剪枝耗时: ");
}

template <typename Graph>
boost::unordered_flat_map<Vertex, KnnResult> KnnIndex<Graph>::build_partial_knn() {
    Timer timer;
    boost::unordered_flat_map<Vertex, KnnResult> partial_knn_map;
    boost::unordered_flat_map<Vertex, int> rank;
    for (size_t i = 0; i < vertex_order_pi_.size(); ++i) {
        rank[vertex_order_pi_[i]] = i;
    }

    for (const auto& u : vertex_order_pi_) {
        std::priority_queue<Neighbor> max_heap;
        if (candidate_objects_.count(u)) {
            max_heap.push({u, 0});
        }

        for (const auto& v_pair : bn_graph_.get_adjacent_vertices(u)) {
            Vertex v = v_pair.first;
            if (!rank.count(v) || rank.at(v) > rank.at(u)) continue;

            if (partial_knn_map.count(v)) {
                for (const auto& neighbor_of_v : partial_knn_map.at(v)) {
                    Weight new_dist = bn_graph_.get_weight(u, v) + neighbor_of_v.dist;
                    if (max_heap.size() < static_cast<size_t>(k_)) {
                        max_heap.push({neighbor_of_v.id, new_dist});
                    } else if (new_dist < max_heap.top().dist) {
                        max_heap.pop();
                        max_heap.push({neighbor_of_v.id, new_dist});
                    }
                }
            }
        }

        KnnResult partial_knn;
        while (!max_heap.empty()) {
            partial_knn.push_back(max_heap.top());
            max_heap.pop();
        }
        std::sort(partial_knn.begin(), partial_knn.end());
        partial_knn_map[u] = partial_knn;
    }
    timer.print_elapsed("    部分kNN构建耗时: ");
    return partial_knn_map;
}

template <typename Graph>
void KnnIndex<Graph>::build_index_internal() {
    auto partial_knn_map = build_partial_knn();
    Timer timer;
    
    boost::unordered_flat_map<Vertex, int> rank;
    for (size_t i = 0; i < vertex_order_pi_.size(); ++i) {
        rank[vertex_order_pi_[i]] = i;
    }

    for (auto it = vertex_order_pi_.rbegin(); it != vertex_order_pi_.rend(); ++it) {
        Vertex u = *it;
        std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<Neighbor>> min_heap;
        boost::unordered_flat_set<Vertex> visited_neighbors;

        if (partial_knn_map.count(u)) {
            for (const auto& neighbor : partial_knn_map.at(u)) {
                min_heap.push(neighbor);
                visited_neighbors.insert(neighbor.id);
            }
        }

        for (const auto& w_pair : bn_graph_.get_adjacent_vertices(u)) {
            Vertex w = w_pair.first;
            if (!rank.count(w) || rank.at(w) < rank.at(u)) continue;

            if (knn_index_.count(w)) {
                for (const auto& neighbor_of_w : knn_index_.at(w)) {
                    if (visited_neighbors.find(neighbor_of_w.id) == visited_neighbors.end()) {
                        Weight new_dist = bn_graph_.get_weight(u, w) + neighbor_of_w.dist;
                        min_heap.push({neighbor_of_w.id, new_dist});
                        visited_neighbors.insert(neighbor_of_w.id);
                    }
                }
            }
        }

        KnnResult final_knn;
        boost::unordered_flat_set<Vertex> final_visited;
        while (!min_heap.empty() && final_knn.size() < static_cast<size_t>(k_)) {
            Neighbor top = min_heap.top();
            min_heap.pop();
            if(final_visited.find(top.id) == final_visited.end()){
               final_knn.push_back(top);
               final_visited.insert(top.id);
            }
        }
        knn_index_[u] = final_knn;
    }
    timer.print_elapsed("    最终索引构建耗时: ");
}

int main() {
    Timer total_timer;
    
    std::cout << "开始加载图数据..." << std::endl;
    Timer load_timer;
    auto [g, success] = load_graph<Graph2>();
    if (!success) {
        std::cerr << "加载图失败！" << std::endl;
        return 1;
    }
    load_timer.print_elapsed("图加载耗时: ");

    std::cout << "开始加载对象数据..." << std::endl;
    load_timer.reset();
    auto [objects, success2] = load_objects();
    if (!success2) {
        std::cerr << "加载对象失败！" << std::endl;
        return 1;
    }
    load_timer.print_elapsed("对象加载耗时: ");

    // 3. 创建增强图
    //    新的 POI 顶点 ID 从 5 开始
    auto [augmented_graph, candidate_objects] = create_augmented_graph(g, objects, g.vertices.size() + 10000);

    // 4. 在增强图上构建索引
    int k = 20;
    KnnIndex index(augmented_graph, k, candidate_objects);
    index.build_index();

    // 5. 进行查询
    Vertex query_point = 600;
    std::cout << "\n--- 查询从顶点 " << query_point << " 出发的 " << k << " 个最近邻居 ---" << std::endl;
    KnnResult result = index.get_knn(query_point);

    // 6. 打印结果
    // 注意：结果中的 ID 是 POI 在增强图中的新顶点ID
    if (result.empty()) {
        std::cout << "未找到邻居。" << std::endl;
    } else {
        std::cout << "查询结果：" << std::endl;
        for (const auto& neighbor : result) {
            std::cout << "  - 对象(顶点ID): " << neighbor.id
                      << ", 距离: " << neighbor.dist << std::endl;
        }
    }

    // 示例：查询另一个点 2
    query_point = 2;
    std::cout << "\n--- 查询从顶点 " << query_point << " 出发的 " << k << " 个最近邻居 ---" << std::endl;
    result = index.get_knn(query_point);
    if (result.empty()) {
        std::cout << "未找到邻居。" << std::endl;
    } else {
        std::cout << "查询结果：" << std::endl;
        for (const auto& neighbor : result) {
            std::cout << "  - 对象(顶点ID): " << neighbor.id 
                      << ", 距离: " << neighbor.dist << std::endl;
        }
    }

    total_timer.print_elapsed("\n程序总运行时间: ");
    return 0;
}
