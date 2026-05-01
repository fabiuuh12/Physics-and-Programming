#include "orbit.hpp"

#include <filesystem>
#include <iostream>
#include <map>

int main() {
    const ai_space::SimConfig cfg;
    const ai_space::RunResult result = ai_space::run_rendezvous(cfg);

    const std::filesystem::path output_dir = "simulations/cpp_rendezvous";
    std::filesystem::create_directories(output_dir);
    ai_space::write_csv(result, (output_dir / "trajectory.csv").string());

    const ai_space::Sample& first = result.samples.front();
    const ai_space::Sample& last = result.samples.back();

    double min_distance = first.distance_km;
    std::map<std::string, int> action_counts;
    for (const ai_space::Sample& sample : result.samples) {
        min_distance = std::min(min_distance, sample.distance_km);
        action_counts[sample.action] += 1;
    }

    std::cout << "C++ autonomous orbital rendezvous simulation\n";
    std::cout << "success: " << std::boolalpha << result.success << '\n';
    std::cout << "simulated time: " << last.time_min << " min\n";
    std::cout << "initial distance: " << first.distance_km << " km\n";
    std::cout << "final distance: " << last.distance_km << " km\n";
    std::cout << "minimum distance: " << min_distance << " km\n";
    std::cout << "final relative speed: " << last.relative_speed_km_s << " km/s\n";
    std::cout << "fuel proxy, delta-v used: " << last.fuel_delta_v_km_s * 1000.0 << " m/s\n";
    std::cout << "action counts:";
    for (const auto& [action, count] : action_counts) {
        std::cout << ' ' << action << '=' << count;
    }
    std::cout << "\nCSV written to: " << (output_dir / "trajectory.csv").string() << '\n';

    return 0;
}
