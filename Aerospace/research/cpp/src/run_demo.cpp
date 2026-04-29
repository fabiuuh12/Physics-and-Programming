#include "cislunar_nav.hpp"

#include <iostream>

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: cislunar_demo <output.csv>\n";
        return 2;
    }
    try {
        cnav::write_demo_csv(argv[1], {0.84, 0.0, 0.0, 0.24}, 360, 0.01, 0.0121505856);
        std::cout << "Wrote " << argv[1] << "\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
    return 0;
}
