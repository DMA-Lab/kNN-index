cmake_minimum_required(VERSION 4.0)
project(kNN_Index)

set(CMAKE_CXX_STANDARD 20)

add_executable(kNN_Index main.cpp
    file.cpp graph.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")