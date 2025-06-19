#include <vector>
#include "graph.h"


bool GraphByAdjacencyList::insert(const Vertex v) {
    //插入顶点集后返回的是插入的位置和一个是否插入成功的判断
    auto [position, result] = vertices.insert(v);
    if (result) {
        num_vertex++;
    }
    return result;
}

// 添加有向边
void GraphByAdjacencyList::add_directional_edge(const Vertex v1, const Vertex v2, Weight w) {
    const auto tier1_end = edges.end();
    const auto tier1_it = edges.find(v1);

    // TODO: 作为补救措施，如果 v1, v2 不存在，就插入 v1, v2. 但这样做会影响性能.
    // for (auto v: {v1, v2}) {
    //     if (!vertices.contains(v)) {
    //         insert(v);
    //     }
    // }

    if (tier1_it == tier1_end) {
        edges[v1] = {{v2, w}};
    }
    else {
        auto tier2_end = tier1_it->second.end();
        auto tier2_it = tier1_it->second.find(v2);
        if (tier2_it == tier2_end) {
            tier1_it->second[v2] = w;
        }
        else {
            tier2_it->second = w;
        }
    }
}

// 删除从 v1 到 v2 的有向边
Weight GraphByAdjacencyList::remove_directional_edge(Vertex v1, Vertex v2) {
    auto tier1_it = edges.find(v1);
    if (tier1_it == edges.end()) {
        return InfWeight;
    }
    auto tier2_it = tier1_it->second.find(v2);
    if (tier2_it == tier1_it->second.end()) {
        return InfWeight;
    }
    Weight weight = tier2_it->second;
    tier1_it->second.erase(tier2_it);
    return weight;
}

// 删除两个顶点之间的边
Weight GraphByAdjacencyList::disconnect(Vertex v1, Vertex v2) {
    Weight weight1 = remove_directional_edge(v1, v2);
    Weight weight2 = remove_directional_edge(v2, v1);
    if (weight1 == InfWeight || weight2 == InfWeight) {
        return InfWeight;
    }
    return weight1;
}

// 连接两顶点，其实是在添加一条无向边，也就是执行两遍添加有向边操作
void GraphByAdjacencyList::connect(const Vertex v1, const Vertex v2, Weight w) {
    add_directional_edge(v1, v2, w);
    add_directional_edge(v2, v1, w);
}

// 判断边 src -> dst 是否存在
bool GraphByAdjacencyList::has_edge(const Vertex v1, const Vertex v2) const {
    auto tier1_end = edges.end();
    auto tier1_it = edges.find(v1);
    if (tier1_it == tier1_end) {
        return false;
    }
    const auto tier2_end = tier1_it->second.end();
    const auto tier2_it = tier1_it->second.find(v2);
    return tier2_it != tier2_end;
}


// 获得两个顶点之间的权值
Weight GraphByAdjacencyList::get_weight(const Vertex v1, const Vertex v2) const {
    // 无环路，顶点等于自身的时候权值为0
    if (v1 == v2) {
        return 0;
    }

    const auto tier1_end = edges.end();
    const auto tier1_it = edges.find(v1);
    if (tier1_it == tier1_end) {
        return InfWeight;
    }

    const auto tier2_end = tier1_it->second.end();
    const auto tier2_it = tier1_it->second.find(v2);
    if (tier2_it == tier2_end) {
        return InfWeight;
    }

    return tier2_it->second;
}

// 获得某顶点的相邻顶点及其权值，返回一个迭代器
EdgeSet GraphByAdjacencyList::get_adjacent_vertices(const Vertex src) const {
    const auto tier1_end = edges.end();
    const auto tier1_it = edges.find(src);

    if (tier1_it == tier1_end) {
        return EdgeSet::empty();
    }

    const auto tier2_map = &tier1_it->second;
    return EdgeSet{tier2_map};
}

size_t GraphByAdjacencyList::out_degree(Vertex v) const {
    const auto tier1_end = edges.end();
    const auto tier1_it = edges.find(v);

    if (tier1_it == tier1_end) {
        return 0;
    }

    const auto map = &tier1_it->second;
    return map->size();
}


bool GraphByAdjacencyArray::insert(const Vertex v) {
    //插入顶点集后返回的是插入的位置和一个是否插入成功的判断
    auto [position, result] = vertices.insert(v);
    if (result) {
        num_vertex++;
    }
    return result;
}

// 添加有向边
void GraphByAdjacencyArray::add_directional_edge(const Vertex v1, const Vertex v2, Weight w) {
    const auto tier1_end = edges.end();
    const auto tier1_it = edges.find(v1);

    if (tier1_it == tier1_end) {
        edges[v1] = std::vector<std::pair<Vertex, Weight>>();
        edges[v1].reserve(4);
        edges[v1].emplace_back(v2, w);
    }
    else {
        auto tier2_end = tier1_it->second.end();
        auto tier2_it = tier1_it->second.begin();
        for (; tier2_it != tier2_end; ++tier2_it) {
            if (tier2_it->first == v2) {
                tier2_it->second = w;
                return;
            }
        }
        tier1_it->second.emplace_back(v2, w);
    }
}

// 删除从 v1 到 v2 的有向边
Weight GraphByAdjacencyArray::remove_directional_edge(Vertex v1, Vertex v2) {
    auto tier1_it = edges.find(v1);
    if (tier1_it == edges.end()) {
        return InfWeight;
    }
    auto tier2_it = tier1_it->second.begin();
    auto tier2_end = tier1_it->second.end();
    for (; tier2_it != tier2_end; ++tier2_it) {
        if (tier2_it->first == v2) {
            auto w = tier2_it->second;
            tier1_it->second.erase(tier2_it);
            return w;
        }
    }
    return InfWeight;
}

// 删除两个顶点之间的边
Weight GraphByAdjacencyArray::disconnect(Vertex v1, Vertex v2) {
    Weight weight1 = remove_directional_edge(v1, v2);
    Weight weight2 = remove_directional_edge(v2, v1);
    if (weight1 == InfWeight || weight2 == InfWeight) {
        return InfWeight;
    }
    return weight1;
}

// 连接两顶点，其实是在添加一条无向边，也就是执行两遍添加有向边操作
void GraphByAdjacencyArray::connect(const Vertex v1, const Vertex v2, Weight w) {
    add_directional_edge(v1, v2, w);
    add_directional_edge(v2, v1, w);
}

// 判断边 src -> dst 是否存在
bool GraphByAdjacencyArray::has_edge(const Vertex v1, const Vertex v2) const {
    auto tier1_end = edges.end();
    auto tier1_it = edges.find(v1);
    if (tier1_it == tier1_end) {
        return false;
    }

    return std::ranges::find(tier1_it->second, v2, &std::pair<Vertex, Weight>::first) != tier1_it->second.end();
}


// 获得两个顶点之间的权值
Weight GraphByAdjacencyArray::get_weight(const Vertex v1, const Vertex v2) const {
    // 无环路，顶点等于自身的时候权值为0
    if (v1 == v2) {
        return 0;
    }

    const auto tier1_end = edges.end();
    const auto tier1_it = edges.find(v1);
    if (tier1_it == tier1_end) {
        return InfWeight;
    }

    auto it = std::ranges::find(tier1_it->second, v2, &std::pair<Vertex, Weight>::first);
    if (it != tier1_it->second.end()) {
        return it->second;
    }
    return InfWeight;
}

// 获得某顶点的相邻顶点及其权值，返回一个迭代器
const std::vector<std::pair<Vertex, Weight>>& GraphByAdjacencyArray::get_adjacent_vertices(const Vertex src) const {
    const auto tier1_end = edges.end();
    const auto tier1_it = edges.find(src);

    if (tier1_it == tier1_end) {
        throw std::out_of_range("vertex not found");
    }
    return tier1_it->second;
}

size_t GraphByAdjacencyArray::out_degree(Vertex v) const {
    const auto tier1_end = edges.end();
    const auto tier1_it = edges.find(v);

    if (tier1_it == tier1_end) {
        throw std::out_of_range("vertex not found");
    }
    return tier1_it->second.size();
}
