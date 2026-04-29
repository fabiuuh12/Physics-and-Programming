#include "cislunar_nav.hpp"

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>

void test_rhs_finite() {
    cnav::State state{0.84, 0.0, 0.0, 0.24};
    cnav::State rhs = cnav::cr3bp_rhs(state, 0.0121505856);
    for (double value : rhs) {
        assert(std::isfinite(value));
    }
}

void test_rk4_short_propagation() {
    cnav::State state{0.84, 0.0, 0.0, 0.24};
    for (int i = 0; i < 20; ++i) {
        state = cnav::rk4_step(state, 0.01, 0.0121505856);
    }
    for (double value : state) {
        assert(std::isfinite(value));
    }
    assert(std::sqrt(state[0] * state[0] + state[1] * state[1]) < 5.0);
}

void test_risk_monotonicity() {
    cnav::RiskWeights weights;
    const double base = cnav::navigation_risk(0.001, 0.0001, 0.7, 0.0, 0.0, weights);
    assert(cnav::navigation_risk(0.002, 0.0001, 0.7, 0.0, 0.0, weights) > base);
    assert(cnav::navigation_risk(0.001, 0.0001, 0.1, 0.0, 0.0, weights) > base);
    assert(cnav::navigation_risk(0.001, 0.0001, 0.7, 1.0, 0.0, weights) > base);
    assert(cnav::navigation_risk(0.001, 0.0001, 0.7, 0.0, 1.0, weights) > base);
}

void test_csv_output() {
    const char *path = "/private/tmp/cislunar_demo_test.csv";
    cnav::write_demo_csv(path, {0.84, 0.0, 0.0, 0.24}, 4, 0.01, 0.0121505856);
    std::ifstream input(path);
    int lines = 0;
    std::string line;
    while (std::getline(input, line)) {
        ++lines;
    }
    assert(lines == 6);
}

int main() {
    test_rhs_finite();
    test_rk4_short_propagation();
    test_risk_monotonicity();
    test_csv_output();
    std::cout << "Cislunar C++ numerical tests passed\n";
    return 0;
}
