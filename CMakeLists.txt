cmake_minimum_required(VERSION 4.0)
project(kNN_Index)

set(CMAKE_CXX_STANDARD 20)

add_executable(kNN_Index main.cpp
    file.cpp graph.cpp)

add_executable(simple-solution simple.cpp file.cpp graph.cpp)
add_executable(test-connectivity test-connectivity.cpp file.cpp graph.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    target_link_libraries(simple-solution PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP support enabled.")
else()
    message(WARNING "OpenMP not found. Building in single-threaded mode.")
endif()