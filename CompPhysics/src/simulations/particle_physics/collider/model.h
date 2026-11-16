#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace collider {

constexpr float pi = 3.14159265358979323846f;
struct Vec3 { float x{}, y{}, z{}; };
struct Rgba { unsigned char r{}, g{}, b{}, a{255}; };

enum class Species { Electron, Muon, Photon, Hadron };

struct Particle {
    Species species{};
    float charge{};
    float momentum{};
    float transverse{};
    float azimuth{};
    float longitudinal{};
    float phase{};
    Rgba color{};
};

struct Event {
    std::vector<Particle> particles;
    std::uint32_t number = 1;
    float missingEt = 0.0f;
    float totalEnergy = 0.0f;
};

inline std::uint32_t hash(std::uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

inline float random01(std::uint32_t& state) {
    return (hash(state) & 0x00ffffffu) / 16777215.0f;
}

inline Rgba speciesColor(Species species, float charge) {
    switch (species) {
        case Species::Electron: return charge < 0 ? Rgba{65, 221, 255, 255} : Rgba{255, 91, 191, 255};
        case Species::Muon: return charge < 0 ? Rgba{124, 148, 255, 255} : Rgba{194, 129, 255, 255};
        case Species::Photon: return Rgba{255, 224, 103, 255};
        case Species::Hadron: return charge == 0 ? Rgba{255, 154, 78, 255} : Rgba{105, 239, 173, 255};
    }
    return {255, 255, 255, 255};
}

inline Event generateEvent(std::uint32_t eventNumber) {
    std::uint32_t rng = 0x9e3779b9u ^ (eventNumber * 0x85ebca6bu);
    Event event;
    event.number = eventNumber;
    int count = 38 + static_cast<int>(random01(rng) * 18.0f);
    event.particles.reserve(static_cast<size_t>(count));

    for (int i = 0; i < count; ++i) {
        float pick = random01(rng);
        Species species = pick < .12f ? Species::Photon : pick < .24f ? Species::Muon : pick < .40f ? Species::Electron : Species::Hadron;
        float charge = species == Species::Photon ? 0.0f : (random01(rng) < .48f ? -1.0f : 1.0f);
        if (species == Species::Hadron && random01(rng) < .16f) charge = 0.0f;
        float momentum = 4.0f + 92.0f * std::pow(random01(rng), 2.1f);
        float thetaBias = random01(rng) * 2.0f - 1.0f;
        float transverse = std::sqrt(std::max(0.05f, 1.0f - thetaBias * thetaBias));
        float longitudinal = thetaBias * (.42f + .58f * random01(rng));
        float phi = random01(rng) * 2.0f * pi;
        event.particles.push_back({species, charge, momentum, transverse, phi, longitudinal,
                                   random01(rng) * 2.0f * pi, speciesColor(species, charge)});
        event.totalEnergy += momentum;
    }

    // A balanced high-energy pair makes each event immediately legible.
    float pairPhi = random01(rng) * 2.0f * pi;
    event.particles.push_back({Species::Muon, -1, 78, .95f, pairPhi, .15f, 0, speciesColor(Species::Muon, -1)});
    event.particles.push_back({Species::Muon, 1, 75, .93f, pairPhi + pi, -.12f, 0, speciesColor(Species::Muon, 1)});
    event.totalEnergy += 153;
    event.missingEt = 8.0f + random01(rng) * 42.0f;
    return event;
}

inline float maxRadius(Species species) {
    switch (species) {
        case Species::Electron: return 4.25f;
        case Species::Photon: return 4.15f;
        case Species::Hadron: return 5.55f;
        case Species::Muon: return 8.25f;
    }
    return 5.0f;
}

// The beam axis is X and the solenoid field is parallel to it. Charged tracks
// therefore curve in the transverse YZ plane. Display units are deliberately
// compressed, but curvature still scales with qB/pT.
inline Vec3 trackPoint(const Particle& particle, float distance, float fieldTesla) {
    float transverseDistance = distance * particle.transverse;
    float x = distance * particle.longitudinal;
    if (std::abs(particle.charge) < .01f) {
        return {x, transverseDistance * std::cos(particle.azimuth), transverseDistance * std::sin(particle.azimuth)};
    }
    float radius = std::max(.42f, particle.momentum * particle.transverse / (9.5f * fieldTesla));
    float turn = particle.charge * transverseDistance / radius;
    float y = radius * (std::sin(particle.azimuth + turn) - std::sin(particle.azimuth)) / particle.charge;
    float z = -radius * (std::cos(particle.azimuth + turn) - std::cos(particle.azimuth)) / particle.charge;
    return {x, y, z};
}

inline float radialDistance(Vec3 point) {
    return std::sqrt(point.y * point.y + point.z * point.z);
}

inline Vec3 endpoint(const Particle& particle, float fieldTesla) {
    float limit = maxRadius(particle.species);
    Vec3 point{};
    for (float d = .04f; d < 18.0f; d += .04f) {
        point = trackPoint(particle, d, fieldTesla);
        if (radialDistance(point) >= limit || std::abs(point.x) >= 9.2f) break;
    }
    return point;
}

struct Timeline {
    float time = 0.0f;
    float speed = 1.0f;
    bool paused = false;
    bool autoRepeat = true;
    static constexpr float duration = 11.5f;

    void update(float dt) {
        if (paused) return;
        time += std::min(dt, .05f) * speed;
        if (time >= duration) time = autoRepeat ? std::fmod(time, duration) : duration;
    }
    float beam() const { return std::clamp(time / 2.55f, 0.0f, 1.0f); }
    float collision() const { return std::clamp((time - 2.45f) / .28f, 0.0f, 1.0f); }
    float tracks() const { return std::clamp((time - 2.62f) / 2.4f, 0.0f, 1.0f); }
    float reveal() const { return std::clamp((time - 4.25f) / 1.6f, 0.0f, 1.0f); }
    float holdFade() const { return 1.0f - std::clamp((time - 10.65f) / .8f, 0.0f, 1.0f); }
    void restart() { time = 0.0f; paused = false; }
};

}  // namespace collider
