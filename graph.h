//
// Created by XuYat on 2023/10/4.
//

#pragma once


#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>
#include <vector>
#include <optional>



using Vertex = int;
using Weight = unsigned int;

constexpr Weight InfWeight = ~0;


class EdgeIterator: public std::iterator_traits<std::pair<Vertex, Weight>> {
    std::optional<boost::unordered_flat_map<Vertex, Weight>::const_iterator> it;

public:
    explicit EdgeIterator() = default;
    explicit EdgeIterator(boost::unordered_flat_map<Vertex, Weight>::const_iterator it): it(it) {}

    static EdgeIterator invalid() {
        return EdgeIterator();
    }

    bool operator==(const EdgeIterator &other) const {
        return *it == *other.it;
    }

    bool operator!=(const EdgeIterator &other) const {
        return *it != *other.it;
    }

    EdgeIterator &operator++() {
        ++(*it);
        return *this;
    }

    std::pair<Vertex, Weight> operator*() const {
        return **it;
    }
};


class EdgeSet {
    std::optional<const boost::unordered_flat_map<Vertex, Weight>*> map;

public:
    EdgeSet(const boost::unordered_flat_map<Vertex, Weight>  *map) {
        this->map = map;
    }

    static EdgeSet empty() {
        return {nullptr};
    }

    EdgeIterator begin() {
        if (nullptr == this->map) {
            return EdgeIterator::invalid();
        }

        return EdgeIterator{(*map)->begin()};
    }

    EdgeIterator end() {
        if (nullptr == this->map) {
            return EdgeIterator::invalid();
        }

        return EdgeIterator{(*map)->end()};
    }

    [[nodiscard]] std::vector<std::pair<Vertex, Weight>> to_vector() const {
        std::vector<std::pair<Vertex, Weight>> result;
        if (nullptr == this->map) {
            return result;
        }

        for (const auto& [v, w]: **map) {
            result.emplace_back(v, w);
        }
        return result;
    }
};

/// 基于邻接表实现的有向图
struct GraphByAdjacencyList {
    unsigned int  num_vertex = 0;


    // 顶点集
    boost::unordered_flat_set<Vertex>  vertices;
    // 将两个顶点打包为一个pair，建立一对顶点到边的映射
    boost::unordered_flat_map<Vertex, boost::unordered_flat_map<Vertex, Weight>> edges;

    GraphByAdjacencyList() = default;
    GraphByAdjacencyList(const GraphByAdjacencyList &other) = default;

    // 判断两个顶点是否相邻
    [[nodiscard]] bool has_edge(Vertex v1, Vertex v2) const;

    // 插入顶点. 如果顶点已经存在，返回 false
    bool insert(Vertex v);

    // 添加有向边
    void add_directional_edge(Vertex v1, Vertex v2, Weight w);

    // 移除有向边
    Weight remove_directional_edge(Vertex v1, Vertex v2);

    // 连接两个顶点
    void connect(Vertex v1, Vertex v2, Weight w);

    // 删除两个顶点之间的边
    Weight disconnect(Vertex v1, Vertex v2);

    // 获得两点之间的边权
    [[nodiscard]] Weight get_weight(Vertex v1, Vertex v2) const;

    // 获得某顶点的相邻顶点及其权值
    [[nodiscard]] EdgeSet get_adjacent_vertices(Vertex src) const;

    // 获取某一个顶点的出边个数
    size_t out_degree(Vertex v) const;
};

using Graph = GraphByAdjacencyList;


/// 基于邻接表实现的有向图
struct GraphByAdjacencyArray {
    unsigned int  num_vertex = 0;


    // 顶点集
    boost::unordered_flat_set<Vertex>  vertices;
    // 将两个顶点打包为一个pair，建立一对顶点到边的映射
    // 考虑到
    boost::unordered_flat_map<Vertex, std::vector<std::pair<Vertex, Weight>>> edges;

    GraphByAdjacencyArray() = default;
    GraphByAdjacencyArray(const GraphByAdjacencyArray &other) = default;

    // 判断两个顶点是否相邻
    [[nodiscard]] bool has_edge(Vertex v1, Vertex v2) const;

    // 插入顶点. 如果顶点已经存在，返回 false
    bool insert(Vertex v);

    // 添加有向边
    void add_directional_edge(Vertex v1, Vertex v2, Weight w);

    // 移除有向边
    Weight remove_directional_edge(Vertex v1, Vertex v2);

    // 连接两个顶点
    void connect(Vertex v1, Vertex v2, Weight w);

    // 删除两个顶点之间的边
    Weight disconnect(Vertex v1, Vertex v2);

    // 获得两点之间的边权
    [[nodiscard]] Weight get_weight(Vertex v1, Vertex v2) const;

    // 获得某顶点的相邻顶点及其权值
    [[nodiscard]] const std::vector<std::pair<Vertex, Weight>>& get_adjacent_vertices(Vertex src) const;

    // 获取某一个顶点的出边个数
    size_t out_degree(Vertex v) const;
};

using Graph2 = GraphByAdjacencyArray;

// 边上对象 (POI) 的数据结构
struct OnEdgePOI {
    unsigned int poi_id;
    Vertex u, v; // POI 所在边的端点
    Weight offset_from_u;
};