#pragma once

#include <algorithm>
#include <cmath>

namespace blackhole {

enum class State { Dormant, Feeding, ActiveNucleus, RelativisticJet };

inline double saturate(double value) { return std::clamp(value, 0.0, 1.0); }

inline double horizonRadius(double mass) {
    return 0.72 + 0.58 * std::sqrt(saturate(mass));
}

inline double diskFraction(double mass) {
    return saturate((mass - 0.04) / 0.72);
}

inline double temperature(double mass) {
    const double disk = diskFraction(mass);
    return disk * disk * (0.35 + 0.65 * mass);
}

inline double jetPower(double mass) {
    const double onset = saturate((mass - 0.62) / 0.30);
    return onset * onset * (3.0 - 2.0 * onset);
}

inline State state(double mass) {
    if (mass < 0.12) return State::Dormant;
    if (mass < 0.42) return State::Feeding;
    if (mass < 0.68) return State::ActiveNucleus;
    return State::RelativisticJet;
}

inline const char* stateName(double mass) {
    switch (state(mass)) {
        case State::Dormant: return "DORMANT BLACK HOLE";
        case State::Feeding: return "ACCRETION DISK FORMING";
        case State::ActiveNucleus: return "ACTIVE GALACTIC NUCLEUS";
        case State::RelativisticJet: return "RELATIVISTIC JETS ACTIVE";
    }
    return "DORMANT BLACK HOLE";
}

inline double feed(double mass, double rate, double dt) {
    return saturate(mass + std::max(0.0, rate) * std::max(0.0, dt));
}

}  // namespace blackhole
