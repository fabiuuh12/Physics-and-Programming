#include "asf_sim.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

namespace {

void test_uniform_structure_zero_acceleration() {
    asf::Config config;
    config.nx = 16;
    config.ny = 16;
    config.lambda_A = 0.3;
    asf::Grid phi(config.nx, config.ny, 0.0);
    asf::Grid structure(config.nx, config.ny, 2.5);
    asf::Grid eff = asf::effective_potential(phi, structure, config);
    auto accel = asf::acceleration(eff, config.dx, config.dy);
    for (double value : accel.first.values) {
        assert(std::abs(value) < 1.0e-12);
    }
    for (double value : accel.second.values) {
        assert(std::abs(value) < 1.0e-12);
    }
}

void test_lambda_zero_recovers_newtonian() {
    asf::Config config;
    config.nx = 8;
    config.ny = 8;
    config.lambda_A = 0.0;
    asf::Grid phi(config.nx, config.ny);
    asf::Grid structure(config.nx, config.ny, 4.0);
    for (int y = 0; y < config.ny; ++y) {
        for (int x = 0; x < config.nx; ++x) {
            phi(x, y) = static_cast<double>(x + y);
        }
    }
    asf::Grid eff = asf::effective_potential(phi, structure, config);
    for (int y = 0; y < config.ny; ++y) {
        for (int x = 0; x < config.nx; ++x) {
            assert(std::abs(eff(x, y) - phi(x, y)) < 1.0e-12);
        }
    }
}

void test_laplacian_quadratic() {
    const int n = 24;
    const double dx = 0.1;
    const double dy = 0.1;
    asf::Grid field(n, n);
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            const double xx = static_cast<double>(x) * dx;
            const double yy = static_cast<double>(y) * dy;
            field(x, y) = xx * xx + yy * yy;
        }
    }
    asf::Grid lap = asf::laplacian(field, dx, dy);
    for (int y = 2; y < n - 2; ++y) {
        for (int x = 2; x < n - 2; ++x) {
            assert(std::abs(lap(x, y) - 4.0) < 1.0e-10);
        }
    }
}

void test_poisson_residual_decreases() {
    asf::Config config;
    config.nx = 32;
    config.ny = 32;
    config.iterations = 120;
    asf::Grid source(config.nx, config.ny, 0.0);
    source(config.nx / 2, config.ny / 2) = 1.0;
    asf::Grid zero(config.nx, config.ny, 0.0);
    const double initial = asf::poisson_residual(zero, source, config.dx, config.dy);
    auto solved = asf::poisson_jacobi(source, config);
    assert(!solved.second.empty());
    assert(std::isfinite(solved.second.back()));
    assert(solved.second.back() < initial);
}

}  // namespace

int main() {
    test_uniform_structure_zero_acceleration();
    test_lambda_zero_recovers_newtonian();
    test_laplacian_quadratic();
    test_poisson_residual_decreases();
    std::cout << "ASF C++ numerical tests passed\n";
    return 0;
}
