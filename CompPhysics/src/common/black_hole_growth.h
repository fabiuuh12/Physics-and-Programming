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

inline double luminosity(double mass) {
    const double hot = saturate((mass - 0.18) / 0.76);
    return hot * hot * (3.0 - 2.0 * hot);
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
        case State::RelativisticJet: return mass > 0.84 ? "QUASAR - RELATIVISTIC JETS" : "RELATIVISTIC JETS ACTIVE";
    }
    return "DORMANT BLACK HOLE";
}

inline double feed(double mass, double rate, double dt) {
    return saturate(mass + std::max(0.0, rate) * std::max(0.0, dt));
}

inline double rewind(double mass, double rate, double dt) {
    return saturate(mass - std::max(0.0, rate) * std::max(0.0, dt));
}

// Thin-lens inverse mapping used by the real-time post-process, in screen units.
inline double lensedSourceRadius(double imageRadius, double einsteinRadius) {
    if (std::abs(imageRadius) < 1e-9) return 0.0;
    return imageRadius - einsteinRadius * einsteinRadius / imageRadius;
}

inline double relativisticBeaming(double dopplerFactor) {
    return std::pow(std::max(0.0, dopplerFactor), 3.0);
}

inline double iscoRadius(double spin) {
    const double a = std::clamp(spin, -0.998, 0.998);
    const double z1 = 1.0 + std::cbrt(1.0 - a * a) *
        (std::cbrt(1.0 + a) + std::cbrt(1.0 - a));
    const double z2 = std::sqrt(3.0 * a * a + z1 * z1);
    const double branch = a >= 0.0 ? -1.0 : 1.0;
    return 3.0 + z2 + branch * std::sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2));
}

inline double radiativeEfficiency(double spin) {
    const double r = iscoRadius(spin);
    return 1.0 - std::sqrt(std::max(0.0, 1.0 - 2.0 / (3.0 * r)));
}

inline double spinJetMultiplier(double spin) {
    const double a = std::abs(std::clamp(spin, -0.998, 0.998));
    return 0.30 + 0.70 * a * a;
}

inline double properTimeRate(double horizonRadii) {
    if (horizonRadii <= 1.0) return 0.0;
    return std::sqrt(std::max(0.0, 1.0 - 1.0 / horizonRadii));
}

inline const char* observerRegime(double horizonRadii) {
    if (horizonRadii > 9.0) return "DISTANT FIELD";
    if (horizonRadii > 4.5) return "STRONG LENSING";
    if (horizonRadii > 2.4) return "PHOTON REGION";
    return "NEAR-HORIZON FLIGHT";
}

inline double interiorSkyRate(double horizonRadii) {
    if (horizonRadii >= 1.0) return 1.0;
    const double depth = saturate(1.0 - horizonRadii);
    return 1.0 + 4.0 * depth * depth;
}

}  // namespace blackhole
