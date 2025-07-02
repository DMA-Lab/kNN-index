

#include <iostream>
#include <fstream>
#include <format>
#include "graph.h"


template<typename Graph>
std::pair<Graph, bool> load_graph() {
    //USA-road-t.NY大图
    //my-graph小图
    //my-subgraph可划分子图

    auto path = "/home/sunnysab/Code/2-YTU/bap/dataset/USA-road-d.NY.gr";
    // auto path = "dataset/ouyang-paper.gr";
    // auto path = "dataset/ppt.gr";

    Graph g;
    std::ifstream file;

    file.open(path, std::ios::binary);
    if (!file) {
        std::cerr << std::format("can't open file {}: {}\n", path, std::strerror(errno));
        return {g, false};
    }

    for (std::string line; getline(file, line);) {
        std::stringstream ss(line); //ss的用法与cin/cout一样

        char operation; //用来保存数据集的第一个操作符
        ss >> operation;

        unsigned int num_vertex, num_edge;
        switch (operation) {
        case'a': {
            Vertex start, end;
            Weight edgeweight;
            ss >> start >> end >> edgeweight; //读入这一行数据
            g.add_directional_edge(start, end, edgeweight);
            break;
        }
        case'p': {
            std::string sp;
            ss >> sp >> num_vertex >> num_edge;
            std::cout << "该图中顶点数为：" << num_vertex << "," << "边数为：" << num_edge << std::endl;
            for (int v = 1; v <= num_vertex; ++v) {
                g.insert(v);
            }
            break;
        }
        case'c': {
            //auto声明变量是自动类型，注意：使用auto必须要进行初始化！
            auto comment = line.substr(0);
            std::cout << comment << std::endl;
        }
        }
    }

    file.close();
    return {g, true};
}

std::pair<std::vector<OnEdgePOI>, bool> load_objects();

using ObjSet = std::vector<OnEdgePOI>;