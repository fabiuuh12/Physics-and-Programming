#include "asf_sim.hpp"

#include <cmath>
#include <iostream>

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: asf_static <config.json> <output.csv>\n";
        return 2;
    }

    try {
        asf::Config config = asf::load_config(argv[1]);
        asf::FieldState state = asf::build_controlled_fields(config, true);
        asf::Grid source(config.nx, config.ny);
        for (int y = 0; y < config.ny; ++y) {
            for (int x = 0; x < config.nx; ++x) {
                source(x, y) = 4.0 * M_PI * config.G * state.rho(x, y);
            }
        }
        auto solved = asf::poisson_jacobi(source, config);
        asf::Grid phi_eff = asf::effective_potential(solved.first, state.structure, config);
        asf::write_csv(argv[2], state, solved.first, phi_eff);
        std::cout << "Wrote " << argv[2] << " with final residual "
                  << (solved.second.empty() ? 0.0 : solved.second.back()) << "\n";
    } catch (const std::exception &error) {
        std::cerr << "ASF static run failed: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
