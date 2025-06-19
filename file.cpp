#include <cstring>
#include <format>
#include <fstream>
#include <sstream>
#include <iostream>
#include "graph.h"

using namespace std;


std::pair<std::vector<OnEdgePOI>, bool> load_objects() {
    auto path = "/home/sunnysab/Code/2-YTU/bap/dataset/USA-road-d.FLA.mos.gr";
    // auto path = "dataset/ouyang-paper.co";
    // auto path = "dataset/ppt.co";

    std::vector<OnEdgePOI> obj_set;
    ifstream file;

    file.open(path, std::ios::binary);
    if (!file) {
        cerr << std::format("can't open file {}: {}\n", path, std::strerror(errno));
        return {obj_set, false};
    }

    int expected_vertex_count = 0;
    for (string line; getline(file, line);) {
        stringstream ss(line); //ss的用法与cin/cout一样

        char operation; //用来保存数据集的第一个操作符
        ss >> operation;

        unsigned int num_vertex, num_edge;
        switch (operation) {
        case 'a': {
            Vertex v1, v2;
            size_t offset;
            ss >> v1 >> v2 >> offset;
            break;
        }
        }
    }

    file.close();
    return {obj_set, true};
}

