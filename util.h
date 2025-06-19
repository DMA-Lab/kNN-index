//
// Created by sunnysab on 25-1-10.
//

#pragma once

#include <chrono>
#include <string>
#include <iostream>


template <typename F>
void calc_time(F const& f) {
    calc_time("", f);
}

template <typename F>
void calc_time(const std::string &func, F const& f) {
    auto start = std::chrono::system_clock::now();
    f();
    auto end = std::chrono::system_clock::now();
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::string duration_string;
    auto duration = microseconds.count();

    if (duration >= 1000) {
        duration_string = std::to_string(duration / 1000) + "ms";
    } else {
        duration_string = std::to_string(duration) + "us";
    }

    if (func.empty()) {
        std::cout << "run in " << duration_string << std::endl;
    } else {
        std::cout << func << " run in " << duration_string << std::endl;
    }
}