#pragma once

#include <algorithm>
#include <cmath>

namespace tidal {

constexpr double pi = 3.14159265358979323846;

inline double smoothstep(double edge0, double edge1, double x) {
    if (edge0 == edge1) return x < edge0 ? 0.0 : 1.0;
    x = std::clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return x * x * (3.0 - 2.0 * x);
}

// Difference in black-hole acceleration across a star of radius R at distance r.
inline double tidalAcceleration(double gm, double distance, double stellarRadius) {
    const double nearSide = std::max(0.05, distance - stellarRadius);
    const double farSide = distance + stellarRadius;
    return gm / (nearSide * nearSide) - gm / (farSide * farSide);
}

inline double orbitalAngularSpeed(double gm, double radius) {
    return std::sqrt(gm / (radius * radius * radius));
}

inline double horizonFade(double radius, double horizonRadius) {
    return smoothstep(horizonRadius * 0.88, horizonRadius * 1.18, radius);
}

enum class Phase { Approach, TidalStretch, Disruption, Accretion };

inline Phase phase(double time) {
    if (time < 2.6) return Phase::Approach;
    if (time < 4.4) return Phase::TidalStretch;
    if (time < 8.0) return Phase::Disruption;
    return Phase::Accretion;
}

inline const char* phaseName(double time) {
    switch (phase(time)) {
        case Phase::Approach: return "APPROACH";
        case Phase::TidalStretch: return "TIDAL STRETCHING";
        case Phase::Disruption: return "STELLAR DISRUPTION";
        case Phase::Accretion: return "DEBRIS ACCRETION";
    }
    return "APPROACH";
}

}  // namespace tidal
