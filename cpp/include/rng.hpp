#pragma once
#include <random>

class RNG {
public:
    explicit RNG(unsigned int seed) : gen_(seed) {}
    double uniform01() { return dist_(gen_); }

private:
    std::mt19937 gen_;
    std::uniform_real_distribution<double> dist_{0.0, 1.0};
};
