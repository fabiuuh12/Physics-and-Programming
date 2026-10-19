#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <string_view>
#include <vector>

namespace {

constexpr int kScreenWidth = 1600;
constexpr int kScreenHeight = 960;
constexpr int kParticlesPerPlanet = 520;
constexpr int kStarCount = 420;
constexpr int kDustCount = 140;
constexpr Vector3 kSunPosition = {-23.5f, 0.0f, 0.0f};

struct OrbitCameraState {
    float yaw = 0.82f;
    float pitch = 0.34f;
    float distance = 34.0f;
};

struct BackdropStar {
    Vector3 pos{};
    float size = 0.0f;
    float alpha = 0.0f;
};

struct DustMote {
    Vector3 pos{};
    float radius = 0.0f;
    Color color{};
};

struct PlanetPreset {
    const char* name = "";
    const char* blurb = "";
    Color bodyColor{};
    Color fieldColor{};
    Color auroraColor{};
    Color accentColor{};
    float displayRadius = 0.0f;
    float magneticMoment = 0.0f;
    float atmosphereShield = 0.0f;
    float tailBias = 0.0f;
    float dipoleTiltDeg = 0.0f;
    float rotationSpeed = 0.0f;
    bool globalDipole = false;
    Vector3 center{};
};

struct PlanetState {
    PlanetPreset preset{};
    float magnetopauseRadius = 0.0f;
    float bowShockRadius = 0.0f;
    float tailLength = 0.0f;
    float auroraStrength = 0.0f;
    float fieldCompression = 0.0f;
};

struct WindParticle {
    int planetIndex = 0;
    Vector3 localPos{};
    Vector3 prevLocalPos{};
    float speedJitter = 0.0f;
    float lane = 0.0f;
    float swirl = 0.0f;
    float size = 0.0f;
};

float RandRange(std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    return dist(rng);
}

float Clamp01(float v) {
    return std::clamp(v, 0.0f, 1.0f);
}

float SmoothStep(float a, float b, float x) {
    if (a == b) return 0.0f;
    const float t = Clamp01((x - a) / (b - a));
    return t * t * (3.0f - 2.0f * t);
}

Color LerpColor(Color a, Color b, float t) {
    const float u = Clamp01(t);
    return Color{
        static_cast<unsigned char>(a.r + (b.r - a.r) * u),
        static_cast<unsigned char>(a.g + (b.g - a.g) * u),
        static_cast<unsigned char>(a.b + (b.b - a.b) * u),
        static_cast<unsigned char>(a.a + (b.a - a.a) * u),
    };
}

Color Brighten(Color c, float amount) {
    return LerpColor(c, WHITE, amount);
}

Color WithAlpha(Color c, float alpha01) {
    c.a = static_cast<unsigned char>(255.0f * Clamp01(alpha01));
    return c;
}

Vector3 SafeNormalize(Vector3 v, Vector3 fallback) {
    const float len = Vector3Length(v);
    if (len < 1.0e-5f) return fallback;
    return Vector3Scale(v, 1.0f / len);
}

float HashNoise(float a, float b, float c) {
    return 0.5f + 0.5f * std::sin(a * 12.9898f + b * 78.233f + c * 37.719f);
}

Vector3 RotatePlanetLocal(Vector3 p, float spin, float tiltDeg) {
    p = Vector3RotateByAxisAngle(p, {0.0f, 1.0f, 0.0f}, spin);
    if (std::fabs(tiltDeg) > 0.001f) {
        p = Vector3RotateByAxisAngle(p, {0.0f, 0.0f, 1.0f}, tiltDeg * DEG2RAD);
    }
    return p;
}

Vector3 SurfacePoint(const PlanetPreset& preset, float radius, float latitude, float longitude, float time) {
    Vector3 p = {
        radius * std::cos(latitude) * std::cos(longitude),
        radius * std::sin(latitude),
        radius * std::cos(latitude) * std::sin(longitude),
    };
    p = RotatePlanetLocal(p, time * preset.rotationSpeed, preset.dipoleTiltDeg * 0.22f);
    return Vector3Add(preset.center, p);
}

void UpdateOrbitCameraDragOnly(Camera3D* camera, OrbitCameraState* orbit) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        const Vector2 delta = GetMouseDelta();
        orbit->yaw -= delta.x * 0.0038f;
        orbit->pitch += delta.y * 0.0038f;
        orbit->pitch = std::clamp(orbit->pitch, -1.18f, 1.18f);
    }

    orbit->distance -= GetMouseWheelMove() * 0.95f;
    orbit->distance = std::clamp(orbit->distance, 18.0f, 56.0f);

    const float cp = std::cos(orbit->pitch);
    camera->target = {1.8f, 0.0f, 0.0f};
    camera->position = Vector3Add(camera->target, {
        orbit->distance * cp * std::cos(orbit->yaw),
        orbit->distance * std::sin(orbit->pitch),
        orbit->distance * cp * std::sin(orbit->yaw),
    });
}

std::array<PlanetState, 4> MakePlanets() {
    return {{
        PlanetState{
            PlanetPreset{
                "Mercury",
                "Weak dipole. The solar wind compresses the dayside to a tight cavity close to the surface.",
                Color{180, 164, 148, 255},
                Color{110, 182, 255, 255},
                Color{170, 220, 255, 255},
                Color{216, 202, 184, 255},
                0.76f,
                0.12f,
                0.06f,
                0.22f,
                8.0f,
                0.22f,
                true,
                {0.0f, 4.8f, -8.0f},
            },
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
        },
        PlanetState{
            PlanetPreset{
                "Earth",
                "A strong global field forms a broad dayside shield, bright cusps, and a long structured magnetotail.",
                Color{66, 122, 214, 255},
                Color{106, 214, 255, 255},
                Color{96, 255, 188, 255},
                Color{102, 170, 255, 255},
                0.92f,
                1.0f,
                0.18f,
                0.78f,
                11.0f,
                0.36f,
                true,
                {0.0f, 4.8f, 8.0f},
            },
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
        },
        PlanetState{
            PlanetPreset{
                "Mars",
                "Mars lacks a powerful global dipole. Its upper atmosphere and patchy crustal fields still stand off the flow.",
                Color{198, 92, 70, 255},
                Color{255, 164, 112, 255},
                Color{255, 208, 150, 255},
                Color{224, 130, 104, 255},
                0.84f,
                0.025f,
                0.28f,
                0.36f,
                0.0f,
                0.18f,
                false,
                {0.0f, -4.8f, -8.0f},
            },
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
        },
        PlanetState{
            PlanetPreset{
                "Jupiter",
                "Its magnetic moment is enormous. The flow bends far upstream and the tail dominates the entire panel.",
                Color{214, 168, 124, 255},
                Color{150, 118, 255, 255},
                Color{255, 198, 124, 255},
                Color{250, 214, 164, 255},
                1.22f,
                16.0f,
                0.44f,
                1.55f,
                10.0f,
                0.12f,
                true,
                {0.0f, -4.8f, 8.0f},
            },
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
        },
    }};
}

std::vector<BackdropStar> MakeBackdropStars() {
    std::mt19937 rng(8142);
    std::vector<BackdropStar> stars;
    stars.reserve(kStarCount);

    for (int i = 0; i < kStarCount; ++i) {
        const float theta = RandRange(rng, 0.0f, 2.0f * PI);
        const float phi = RandRange(rng, -0.46f * PI, 0.46f * PI);
        const float radius = RandRange(rng, 54.0f, 78.0f);
        stars.push_back({
            {
                radius * std::cos(phi) * std::cos(theta),
                radius * std::sin(phi),
                radius * std::cos(phi) * std::sin(theta),
            },
            RandRange(rng, 0.025f, 0.115f),
            RandRange(rng, 0.18f, 0.95f),
        });
    }

    return stars;
}

std::vector<DustMote> MakeDustCloud() {
    std::mt19937 rng(5221);
    std::vector<DustMote> motes;
    motes.reserve(kDustCount);

    for (int i = 0; i < kDustCount; ++i) {
        const float x = RandRange(rng, -36.0f, 34.0f);
        const float y = RandRange(rng, -18.0f, 18.0f);
        const float z = RandRange(rng, -26.0f, 26.0f);
        const float warm = RandRange(rng, 0.0f, 1.0f);
        motes.push_back({
            {x, y, z},
            RandRange(rng, 0.12f, 0.48f),
            LerpColor(Color{90, 126, 210, 255}, Color{214, 128, 86, 255}, warm),
        });
    }

    return motes;
}

void UpdatePlanetDerivedState(std::array<PlanetState, 4>* planets, float windSpeed, float windDensity, float imfTiltDeg, float time) {
    const float windPressure = std::max(0.16f, windDensity * windSpeed * windSpeed);
    const float imfFactor = 1.0f + 0.10f * std::sin(imfTiltDeg * DEG2RAD);
    const float pulse = 1.0f + 0.06f * std::sin(time * 0.82f);

    for (PlanetState& planet : *planets) {
        const float momentTerm = std::pow((planet.preset.magneticMoment * planet.preset.magneticMoment) / windPressure, 1.0f / 6.0f);
        const float standoff =
            planet.preset.displayRadius *
            (1.02f + planet.preset.atmosphereShield + (planet.preset.globalDipole ? 2.15f * momentTerm : 1.38f * momentTerm));

        planet.magnetopauseRadius = std::clamp(standoff * imfFactor * pulse, planet.preset.displayRadius * 1.28f, 9.6f);
        planet.tailLength = planet.magnetopauseRadius * (2.2f + planet.preset.tailBias * 1.95f + windSpeed * 0.34f);
        planet.bowShockRadius = planet.magnetopauseRadius * (1.18f + 0.08f * windDensity);
        planet.auroraStrength = Clamp01(0.18f + 0.34f * windDensity + 0.12f * windSpeed + (planet.preset.globalDipole ? 0.26f : 0.10f));
        planet.fieldCompression = Clamp01((windPressure - 0.3f) / 3.1f);
    }
}

float MagnetopauseBoundaryYZ(const PlanetState& planet, float localX) {
    const float mp = planet.magnetopauseRadius;

    if (localX < -mp * 1.14f || localX > planet.tailLength) return -1.0f;

    if (localX <= 0.0f) {
        const float xNorm = (localX + mp * 0.08f) / (mp * (0.94f - 0.12f * planet.fieldCompression));
        const float inside = std::max(0.0f, 1.0f - xNorm * xNorm);
        return mp * (0.20f + 0.96f * std::sqrt(inside));
    }

    const float decay = 1.0f / (1.0f + localX / (planet.tailLength * 0.52f));
    return mp * (0.18f + (planet.preset.globalDipole ? 0.98f : 0.58f) * decay);
}

float BowShockBoundaryYZ(const PlanetState& planet, float localX) {
    const float nose = -planet.bowShockRadius * 1.08f;
    const float tail = planet.tailLength * 1.08f;
    if (localX < nose || localX > tail) return -1.0f;

    if (localX <= 0.0f) {
        const float xNorm = (localX + planet.bowShockRadius * 0.08f) / planet.bowShockRadius;
        const float inside = std::max(0.0f, 1.0f - xNorm * xNorm);
        return planet.bowShockRadius * (0.30f + 1.05f * std::sqrt(inside));
    }

    const float decay = 1.0f / (1.0f + localX / (tail * 0.44f));
    return planet.bowShockRadius * (0.28f + 1.08f * decay);
}

Vector3 SunLocalCenter(const PlanetState& planet) {
    return Vector3Subtract(kSunPosition, planet.preset.center);
}

Vector3 IncomingCenterlineLocal(const PlanetState& planet, float localX) {
    const Vector3 source = SunLocalCenter(planet);
    if (localX <= 0.0f) {
        const float t = Clamp01((localX - source.x) / std::max(0.001f, -source.x));
        return {
            localX,
            Lerp(source.y, 0.0f, SmoothStep(0.0f, 1.0f, t)),
            Lerp(source.z, 0.0f, SmoothStep(0.0f, 1.0f, t)),
        };
    }
    return {localX, 0.0f, 0.0f};
}

void SeedParticleAlongFlow(WindParticle* particle, const PlanetState& planet, std::mt19937* rng, float x) {
    const Vector3 centerline = IncomingCenterlineLocal(planet, x);
    const float radialScale = x < 0.0f ? 0.42f : 0.95f;
    const float dy = particle->lane * 0.80f + RandRange(*rng, -0.42f, 0.42f) * radialScale;
    const float dz = std::sin(particle->swirl) * 0.65f + RandRange(*rng, -0.38f, 0.38f) * radialScale;
    particle->localPos = {x, centerline.y + dy, centerline.z + dz};
    particle->prevLocalPos = particle->localPos;
}

Vector3 DeflectLocalFlow(const PlanetState& planet, Vector3 local, float time, float imfTiltDeg) {
    const float imfTilt = imfTiltDeg * DEG2RAD;
    const Vector3 centerline = IncomingCenterlineLocal(planet, local.x);
    Vector3 centered = {local.x, local.y - centerline.y, local.z - centerline.z};
    const float yz = std::sqrt(centered.y * centered.y + centered.z * centered.z);
    const float shock = BowShockBoundaryYZ(planet, centered.x);
    const float magnetopause = MagnetopauseBoundaryYZ(planet, centered.x);
    const Vector3 yzDir =
        yz > 1.0e-4f ? Vector3Scale({0.0f, centered.y, centered.z}, 1.0f / yz) : Vector3{0.0f, 1.0f, 0.0f};

    if (shock > 0.0f && yz < shock) {
        const float proximity = 1.0f - yz / shock;
        centered = Vector3Add(centered, Vector3Scale(yzDir, proximity * (1.35f + 0.55f * planet.fieldCompression)));
        centered.x -= proximity * (0.10f + 0.04f * planet.fieldCompression);
        centered.y += proximity * std::sin(imfTilt) * 0.72f;
        centered.z += proximity * std::cos(imfTilt) * 0.20f;
    }

    if (magnetopause > 0.0f && yz < magnetopause) {
        const float proximity = 1.0f - yz / magnetopause;
        const Vector3 swirlDir =
            yz > 1.0e-4f ? Vector3Normalize({0.0f, -centered.z, centered.y}) : Vector3{0.0f, 0.0f, 1.0f};
        const float swirlStrength = planet.preset.globalDipole ? (1.80f + 1.05f * planet.preset.tailBias) : 0.68f;
        const float shieldPush = planet.preset.globalDipole ? 2.10f : 1.05f;

        centered = Vector3Add(centered, Vector3Scale(yzDir, proximity * (shieldPush + 0.80f * planet.fieldCompression)));
        centered = Vector3Add(centered, Vector3Scale(swirlDir, proximity * swirlStrength));
        centered.x -= proximity * (0.18f + 0.10f * planet.preset.tailBias);
        centered.y += proximity * std::sin(imfTilt) * (planet.preset.globalDipole ? 1.05f : 0.42f);
        centered.z += 0.28f * proximity * std::sin(time * 0.8f + centered.x * 0.12f);
    }

    return {centered.x, centered.y + centerline.y, centered.z + centerline.z};
}

void RespawnParticle(WindParticle* particle, const PlanetState& planet, std::mt19937* rng) {
    particle->speedJitter = RandRange(*rng, 0.82f, 1.24f);
    particle->lane = RandRange(*rng, -1.0f, 1.0f);
    particle->swirl = RandRange(*rng, 0.0f, 2.0f * PI);
    particle->size = RandRange(*rng, 0.028f, 0.074f);
    const float sourceX = SunLocalCenter(planet).x + RandRange(*rng, 2.4f, 5.6f);
    SeedParticleAlongFlow(particle, planet, rng, sourceX);
}

std::vector<WindParticle> MakeWindParticles(const std::array<PlanetState, 4>& planets) {
    std::mt19937 rng(2571);
    std::vector<WindParticle> particles;
    particles.reserve(static_cast<std::size_t>(planets.size()) * kParticlesPerPlanet);

    for (int planetIndex = 0; planetIndex < static_cast<int>(planets.size()); ++planetIndex) {
        for (int i = 0; i < kParticlesPerPlanet; ++i) {
            WindParticle particle{};
            particle.planetIndex = planetIndex;
            RespawnParticle(&particle, planets[planetIndex], &rng);
            const float sourceX = SunLocalCenter(planets[planetIndex]).x + 2.2f;
            const float x = RandRange(rng, sourceX, planets[planetIndex].tailLength + 2.6f);
            SeedParticleAlongFlow(&particle, planets[planetIndex], &rng, x);
            particles.push_back(particle);
        }
    }

    return particles;
}

void UpdateWindParticles(std::vector<WindParticle>* particles,
                         const std::array<PlanetState, 4>& planets,
                         float dt,
                         float time,
                         float windSpeed,
                         float imfTiltDeg) {
    static std::mt19937 rng(6621);

    for (WindParticle& particle : *particles) {
        const PlanetState& planet = planets[particle.planetIndex];
        const float speed = (4.8f + 3.2f * windSpeed) * particle.speedJitter;

        particle.prevLocalPos = particle.localPos;
        particle.localPos.x += speed * dt;
        const Vector3 centerline = IncomingCenterlineLocal(planet, particle.localPos.x);
        particle.localPos.y += (centerline.y - particle.localPos.y) * std::min(1.0f, 1.9f * dt);
        particle.localPos.z += (centerline.z - particle.localPos.z) * std::min(1.0f, 1.9f * dt);
        particle.localPos.y += 0.24f * std::sin(time * 1.5f + particle.swirl + particle.localPos.x * 0.22f) * dt;
        particle.localPos.z += 0.24f * std::cos(time * 1.24f + particle.lane * 4.2f + particle.localPos.x * 0.17f) * dt;
        particle.localPos = DeflectLocalFlow(planet, particle.localPos, time, imfTiltDeg);

        const float yz = std::sqrt(particle.localPos.y * particle.localPos.y + particle.localPos.z * particle.localPos.z);
        const float bodyRadius = planet.preset.displayRadius;

        if (yz < bodyRadius * (planet.preset.globalDipole ? 0.92f : 1.08f) &&
            std::fabs(particle.localPos.x) < bodyRadius * 1.08f) {
            RespawnParticle(&particle, planet, &rng);
            continue;
        }

        if (particle.localPos.x > planet.tailLength + 6.4f ||
            std::fabs(particle.localPos.y) > 10.0f ||
            std::fabs(particle.localPos.z) > 10.0f) {
            RespawnParticle(&particle, planet, &rng);
        }
    }
}

template <std::size_t N>
void DrawRibbonStrip(const std::array<Vector3, N>& left, const std::array<Vector3, N>& right, Color color, float alpha) {
    for (std::size_t i = 1; i < N; ++i) {
        const float fade0 = alpha * (1.0f - static_cast<float>(i - 1) / static_cast<float>(N - 1));
        const float fade1 = alpha * (1.0f - static_cast<float>(i) / static_cast<float>(N - 1));
        const Color c0 = WithAlpha(color, fade0);
        const Color c1 = WithAlpha(color, fade1);
        DrawTriangle3D(left[i - 1], right[i - 1], right[i], c0);
        DrawTriangle3D(left[i - 1], right[i], left[i], c1);
    }
}

void DrawSun(float time) {
    DrawSphereEx(kSunPosition, 2.8f, 30, 48, Color{255, 180, 84, 255});
    DrawSphereEx(kSunPosition, 3.6f, 24, 40, Fade(Color{255, 188, 96, 255}, 0.18f));
    DrawSphereEx(kSunPosition, 4.5f, 18, 28, Fade(Color{255, 142, 76, 255}, 0.10f));

    for (int ring = 0; ring < 6; ++ring) {
        const float radius = 3.4f + ring * 0.42f + 0.10f * std::sin(time * 1.3f + ring);
        DrawCircle3D(kSunPosition, radius, {1.0f, 0.0f, 0.0f}, 90.0f, Fade(Color{255, 214, 130, 255}, 0.14f));
        DrawCircle3D(kSunPosition, radius * 0.82f, {0.0f, 1.0f, 0.0f}, 0.0f, Fade(Color{255, 166, 92, 255}, 0.10f));
    }

    for (int i = 0; i < 26; ++i) {
        const float a = (2.0f * PI * i) / 26.0f;
        const float wobble = 0.32f * std::sin(time * 1.6f + i * 0.7f);
        const Vector3 p0 = Vector3Add(kSunPosition, {0.0f, std::cos(a) * 2.9f, std::sin(a) * 2.9f});
        const Vector3 p1 = Vector3Add(kSunPosition, {0.0f, std::cos(a) * (4.2f + wobble), std::sin(a) * (4.2f + wobble)});
        DrawLine3D(p0, p1, Fade(Color{255, 220, 156, 255}, 0.20f));
    }
}

void DrawBackground(const std::vector<BackdropStar>& stars, const std::vector<DustMote>& dust, float time) {
    for (const BackdropStar& star : stars) {
        DrawSphere(star.pos, star.size, Fade(WHITE, star.alpha));
    }

    for (std::size_t i = 0; i < dust.size(); ++i) {
        const DustMote& mote = dust[i];
        const float pulse = 0.5f + 0.5f * std::sin(time * 0.35f + static_cast<float>(i) * 0.13f);
        DrawSphere(mote.pos, mote.radius * (0.7f + 0.4f * pulse), Fade(mote.color, 0.05f + 0.04f * pulse));
    }
}

void DrawWindGuide(const PlanetState& planet, bool selected) {
    const Color guide = Fade(selected ? Brighten(planet.preset.fieldColor, 0.32f) : Color{140, 182, 226, 255}, selected ? 0.32f : 0.15f);
    const Vector3 source = SunLocalCenter(planet);

    for (int lane = -2; lane <= 2; ++lane) {
        const float laneOffset = lane * 0.56f;
        const Vector3 p0 = Vector3Add(planet.preset.center, {source.x + 3.0f, source.y + laneOffset * 0.55f, source.z + laneOffset});
        const Vector3 p1 = Vector3Add(planet.preset.center, {-9.8f, laneOffset * 0.25f, laneOffset * 0.18f});
        const Vector3 tipA = Vector3Add(p1, {-0.70f, 0.18f, 0.0f});
        const Vector3 tipB = Vector3Add(p1, {-0.70f, -0.18f, 0.0f});
        DrawLine3D(p0, p1, guide);
        DrawLine3D(tipA, p1, guide);
        DrawLine3D(tipB, p1, guide);
    }
}

void DrawMagnetopauseShell(const PlanetState& planet, bool selected) {
    const std::array<float, 3> layerScales = {1.00f, 1.07f, 1.15f};
    const std::array<float, 3> alphas = {selected ? 0.22f : 0.13f, selected ? 0.11f : 0.07f, selected ? 0.06f : 0.04f};

    for (std::size_t layer = 0; layer < layerScales.size(); ++layer) {
        const float layerScale = layerScales[layer];
        const Color shellColor = WithAlpha(planet.preset.fieldColor, alphas[layer]);

        for (int plane = 0; plane < 12; ++plane) {
            const float angle = plane * PI / 12.0f;
            Vector3 prev{};
            bool hasPrev = false;

            for (int i = 0; i <= 120; ++i) {
                const float t = static_cast<float>(i) / 120.0f;
                const float x = Lerp(-planet.magnetopauseRadius * 1.04f * layerScale, planet.tailLength * layerScale, t);
                const float r = MagnetopauseBoundaryYZ(planet, x / layerScale) * layerScale;
                if (r <= 0.02f) {
                    hasPrev = false;
                    continue;
                }

                const Vector3 p = Vector3Add(planet.preset.center, {x, r * std::cos(angle), r * std::sin(angle)});
                if (hasPrev) DrawLine3D(prev, p, shellColor);
                prev = p;
                hasPrev = true;
            }
        }
    }

    for (int ring = 0; ring < 8; ++ring) {
        const float x = Lerp(-planet.magnetopauseRadius * 0.92f, planet.tailLength * 0.62f, static_cast<float>(ring) / 7.0f);
        const float r = MagnetopauseBoundaryYZ(planet, x);
        if (r > 0.05f) {
            DrawCircle3D(Vector3Add(planet.preset.center, {x, 0.0f, 0.0f}), r, {1.0f, 0.0f, 0.0f}, 90.0f,
                         WithAlpha(planet.preset.fieldColor, selected ? 0.12f : 0.06f));
        }
    }
}

void DrawBowShock(const PlanetState& planet, float time, bool selected) {
    const Color shockColor = WithAlpha(Color{255, 204, 128, 255}, selected ? 0.22f : 0.12f);

    for (int plane = 0; plane < 8; ++plane) {
        const float angle = plane * PI / 8.0f;
        Vector3 prev{};
        bool hasPrev = false;

        for (int i = 0; i <= 90; ++i) {
            const float t = static_cast<float>(i) / 90.0f;
            const float x = Lerp(-planet.bowShockRadius * 1.06f, planet.tailLength * 1.08f, t);
            const float r = BowShockBoundaryYZ(planet, x);
            if (r <= 0.02f) {
                hasPrev = false;
                continue;
            }

            const float ripple = 1.0f + 0.045f * std::sin(time * 1.3f + angle * 4.0f + t * 10.0f);
            const Vector3 p = Vector3Add(planet.preset.center, {x, r * ripple * std::cos(angle), r * ripple * std::sin(angle)});
            if (hasPrev) DrawLine3D(prev, p, shockColor);
            prev = p;
            hasPrev = true;
        }
    }
}

void DrawTailRibbon(const PlanetState& planet, float time, float phase, float thickness, Color color, float alpha) {
    constexpr std::size_t kSegments = 34;
    std::array<Vector3, kSegments> left{};
    std::array<Vector3, kSegments> right{};

    for (std::size_t i = 0; i < kSegments; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(kSegments - 1);
        const float x = Lerp(planet.preset.displayRadius * 0.6f, planet.tailLength * 0.98f, t);
        const float waveA = std::sin(time * 1.2f + phase + t * 8.0f);
        const float waveB = std::cos(time * 0.8f + phase * 1.7f + t * 6.5f);
        const Vector3 center = Vector3Add(planet.preset.center, {
            x,
            0.55f * thickness * waveA,
            1.2f * thickness * waveB + thickness * 0.2f * std::sin(phase),
        });

        Vector3 widthDir = Vector3Normalize({0.0f, 0.65f + 0.25f * std::sin(phase + t * 5.0f), 1.0f});
        widthDir = Vector3Scale(widthDir, thickness * (1.0f - 0.55f * t));
        left[i] = Vector3Subtract(center, widthDir);
        right[i] = Vector3Add(center, widthDir);
    }

    DrawRibbonStrip(left, right, color, alpha);
}

void DrawSolarWindStreamlines(const PlanetState& planet, float time, float imfTiltDeg, bool selected) {
    const int lanesY = 5;
    const int lanesZ = 9;
    const Color baseColor = selected ? Brighten(planet.preset.fieldColor, 0.20f) : Color{150, 206, 255, 255};
    const Vector3 source = SunLocalCenter(planet);

    for (int iy = -lanesY; iy <= lanesY; ++iy) {
        for (int iz = -lanesZ; iz <= lanesZ; ++iz) {
            const float baseY = iy * 0.60f;
            const float baseZ = iz * 0.58f;
            const float sourceY = source.y + iy * 0.28f;
            const float sourceZ = source.z + iz * 0.28f;
            const float laneRadius = std::sqrt((sourceY - source.y) * (sourceY - source.y) + (sourceZ - source.z) * (sourceZ - source.z));
            if (laneRadius < planet.preset.displayRadius * 0.60f) continue;

            Vector3 prev{};
            bool hasPrev = false;

            for (int step = 0; step <= 70; ++step) {
                const float t = static_cast<float>(step) / 70.0f;
                Vector3 local = {
                    Lerp(source.x + 2.8f, planet.tailLength + 5.0f, t),
                    Lerp(sourceY, baseY, SmoothStep(0.0f, 1.0f, t)) + 0.16f * std::sin(time * 1.0f + iz * 0.5f + t * 9.0f),
                    Lerp(sourceZ, baseZ, SmoothStep(0.0f, 1.0f, t)) + 0.16f * std::cos(time * 1.2f + iy * 0.4f + t * 7.0f),
                };
                local = DeflectLocalFlow(planet, local, time, imfTiltDeg);
                const Vector3 world = Vector3Add(planet.preset.center, local);

                if (hasPrev) {
                    const float fade = selected ? 0.22f : 0.10f;
                    DrawLine3D(prev, world, WithAlpha(baseColor, fade * (1.0f - 0.34f * t)));
                }
                prev = world;
                hasPrev = true;
            }
        }
    }
}

Vector3 DipoleFieldPoint(float shellRadius, float latitude, float longitude, float tailStretch, float tiltRad) {
    const float c = std::cos(latitude);
    const float s = std::sin(latitude);
    const float radial = shellRadius * c * c;
    Vector3 p = {radial * c, radial * s, 0.0f};

    if (p.x > 0.0f) {
        p.x *= 1.0f + tailStretch * (p.x / std::max(0.001f, shellRadius));
    } else {
        p.x *= 0.76f;
    }

    p = Vector3RotateByAxisAngle(p, {1.0f, 0.0f, 0.0f}, longitude);
    p = Vector3RotateByAxisAngle(p, {0.0f, 0.0f, 1.0f}, tiltRad);
    return p;
}

void DrawDipoleFieldLines(const PlanetState& planet, float time, float imfTiltDeg, bool selected) {
    if (!planet.preset.globalDipole) {
        const Color drapeColor = WithAlpha(planet.preset.fieldColor, selected ? 0.24f : 0.12f);
        for (int i = 0; i < 11; ++i) {
            const float z = -3.0f + i * 0.60f;
            Vector3 prev{};
            bool hasPrev = false;
            for (int j = 0; j <= 56; ++j) {
                const float t = static_cast<float>(j) / 56.0f;
                const float x = Lerp(-planet.magnetopauseRadius, planet.magnetopauseRadius * 0.9f, t);
                const float arch = planet.magnetopauseRadius * (0.86f - std::pow(2.0f * t - 1.0f, 2.0f) * 0.64f);
                const Vector3 p = Vector3Add(planet.preset.center, {
                    x,
                    arch + 0.20f * std::sin(time * 1.7f + i * 0.4f + t * 6.0f),
                    z,
                });
                if (hasPrev) DrawLine3D(prev, p, drapeColor);
                prev = p;
                hasPrev = true;
            }
        }
        return;
    }

    const float tilt = (planet.preset.dipoleTiltDeg + 0.34f * imfTiltDeg) * DEG2RAD;
    const Color fieldColor = WithAlpha(planet.preset.fieldColor, selected ? 0.34f : 0.16f);

    for (int shell = 0; shell < 8; ++shell) {
        const float shellRadius = planet.preset.displayRadius * (1.45f + shell * 0.48f);
        for (int az = 0; az < 10; ++az) {
            const float longitude = (2.0f * PI * az) / 10.0f;
            Vector3 prev{};
            bool hasPrev = false;

            for (int i = 0; i <= 132; ++i) {
                const float lat = -1.20f + 2.40f * static_cast<float>(i) / 132.0f;
                Vector3 p = DipoleFieldPoint(shellRadius, lat, longitude, 0.48f + 0.24f * planet.preset.tailBias, tilt);
                if (Vector3Length(p) < planet.preset.displayRadius * 1.03f) {
                    hasPrev = false;
                    continue;
                }

                p.y += 0.05f * std::sin(time * 1.45f + shell * 0.8f + lat * 5.0f);
                p = Vector3Add(p, planet.preset.center);
                if (hasPrev) DrawLine3D(prev, p, fieldColor);
                prev = p;
                hasPrev = true;
            }
        }
    }
}

void DrawAuroraCurtains(const PlanetState& planet, float time) {
    if (!planet.preset.globalDipole) {
        const Color c = WithAlpha(planet.preset.auroraColor, 0.16f + 0.10f * planet.auroraStrength);
        for (int i = 0; i < 6; ++i) {
            const float z = -0.64f + i * 0.26f;
            const Vector3 p0 = Vector3Add(planet.preset.center, {-planet.preset.displayRadius * 0.98f, 0.34f, z});
            const Vector3 p1 = Vector3Add(planet.preset.center, {-planet.preset.displayRadius * 1.18f, 0.72f + 0.10f * std::sin(time * 2.0f + i), z});
            DrawLine3D(p0, p1, c);
        }
        return;
    }

    for (int hemisphere = -1; hemisphere <= 1; hemisphere += 2) {
        constexpr std::size_t kSegments = 28;
        std::array<Vector3, kSegments> inner{};
        std::array<Vector3, kSegments> outer{};

        for (std::size_t i = 0; i < kSegments; ++i) {
            const float t = static_cast<float>(i) / static_cast<float>(kSegments - 1);
            const float lon = t * 2.0f * PI;
            const float lat = hemisphere * (1.04f - 0.08f * std::sin(lon * 2.0f + time * 0.4f));
            const float flutter = 1.0f + 0.18f * std::sin(time * 2.4f + lon * 4.0f + hemisphere);
            inner[i] = SurfacePoint(planet.preset, planet.preset.displayRadius * 1.01f, lat, lon, time);
            outer[i] = SurfacePoint(planet.preset, planet.preset.displayRadius * (1.18f + 0.12f * flutter * planet.auroraStrength), lat, lon, time);
        }

        DrawRibbonStrip(inner, outer, planet.preset.auroraColor, 0.18f + 0.14f * planet.auroraStrength);
    }
}

void DrawPlanetSurfaceScatter(const PlanetPreset& preset, float time) {
    if (preset.name == std::string_view("Earth")) {
        for (int latBand = -5; latBand <= 5; ++latBand) {
            const float lat = latBand * 0.22f;
            for (int i = 0; i < 28; ++i) {
                const float lon = i * 2.0f * PI / 28.0f;
                const float noise = HashNoise(lat * 2.4f, lon * 3.6f, 0.7f);
                if (noise < 0.54f) continue;
                const Color land = LerpColor(Color{74, 152, 92, 255}, Color{168, 138, 84, 255}, HashNoise(lat, lon, 2.1f));
                DrawSphere(SurfacePoint(preset, preset.displayRadius * 1.012f, lat, lon, time), 0.040f + 0.028f * noise, land);
            }
        }

        for (int band = -4; band <= 4; ++band) {
            const float lat = band * 0.20f + 0.04f * std::sin(time * 0.4f + band);
            for (int i = 0; i < 24; ++i) {
                const float lon = i * 2.0f * PI / 24.0f + time * 0.12f;
                const float cloud = HashNoise(lat * 4.0f, lon * 2.8f, 3.4f);
                if (cloud < 0.60f) continue;
                DrawSphere(SurfacePoint(preset, preset.displayRadius * 1.06f, lat, lon, time),
                           0.030f + 0.032f * cloud,
                           Fade(Color{236, 244, 255, 255}, 0.48f));
            }
        }
        return;
    }

    if (preset.name == std::string_view("Jupiter")) {
        for (int band = -11; band <= 11; ++band) {
            const float lat = band * 0.10f;
            const Color bandColor =
                LerpColor(Color{196, 146, 100, 255}, Color{238, 202, 154, 255}, 0.5f + 0.5f * std::sin(band * 0.8f));
            for (int i = 0; i < 34; ++i) {
                const float lon = i * 2.0f * PI / 34.0f;
                const float storm = HashNoise(lat * 2.0f, lon * 4.2f, 4.8f);
                DrawSphere(SurfacePoint(preset, preset.displayRadius * 1.01f, lat + 0.022f * std::sin(lon * 4.0f + band), lon, time),
                           0.050f + 0.020f * storm,
                           LerpColor(bandColor, preset.accentColor, 0.22f * storm));
            }
        }
        DrawSphere(SurfacePoint(preset, preset.displayRadius * 1.04f, -0.18f, time * 0.05f + 1.2f, time),
                   0.15f,
                   Fade(Color{228, 172, 126, 255}, 0.92f));
        return;
    }

    if (preset.name == std::string_view("Mercury")) {
        for (int i = 0; i < 120; ++i) {
            const float lat = -1.15f + 2.30f * static_cast<float>(i % 15) / 14.0f;
            const float lon = (2.0f * PI * static_cast<float>(i)) / 37.0f;
            const float crater = HashNoise(lat * 3.4f, lon * 4.8f, 7.2f);
            const Color c = LerpColor(Color{128, 118, 110, 255}, Color{214, 198, 180, 255}, crater);
            DrawSphere(SurfacePoint(preset, preset.displayRadius * 1.01f, lat, lon, time), 0.020f + 0.020f * crater, c);
        }
        return;
    }

    for (int i = 0; i < 110; ++i) {
        const float lat = -1.10f + 2.20f * static_cast<float>(i % 11) / 10.0f;
        const float lon = (2.0f * PI * static_cast<float>(i)) / 29.0f;
        const float dust = HashNoise(lat * 4.2f, lon * 2.2f, 5.0f);
        const Color c = LerpColor(Color{144, 80, 62, 255}, Color{220, 148, 112, 255}, dust);
        DrawSphere(SurfacePoint(preset, preset.displayRadius * 1.01f, lat, lon, time), 0.024f + 0.020f * dust, c);
    }

    for (int cap = -1; cap <= 1; cap += 2) {
        for (int i = 0; i < 18; ++i) {
            const float lon = i * 2.0f * PI / 18.0f;
            DrawSphere(SurfacePoint(preset, preset.displayRadius * 1.03f, cap * 1.22f, lon, time), 0.030f, Fade(WHITE, 0.76f));
        }
    }
}

void DrawPlanetBody(const PlanetState& planet, float time, bool selected) {
    const float r = planet.preset.displayRadius;
    const Color body = planet.preset.bodyColor;

    DrawSphereEx(planet.preset.center, r * 1.52f, 26, 36, WithAlpha(planet.preset.fieldColor, selected ? 0.16f : 0.09f));
    DrawSphereEx(planet.preset.center, r, 28, 44, body);
    DrawSphereEx(Vector3Add(planet.preset.center, {r * 0.14f, 0.0f, 0.0f}), r * 0.98f, 24, 36, Fade(BLACK, 0.22f));
    DrawSphereWires(planet.preset.center, r * 1.01f, 18, 18, WithAlpha(Brighten(body, 0.44f), 0.18f));

    if (planet.preset.globalDipole || planet.preset.atmosphereShield > 0.10f) {
        DrawSphereEx(Vector3Add(planet.preset.center, {-r * 0.05f, 0.0f, 0.0f}),
                     r * (1.08f + 0.05f * planet.auroraStrength),
                     20,
                     28,
                     WithAlpha(planet.preset.accentColor, planet.preset.globalDipole ? 0.10f : 0.08f));
    }

    DrawPlanetSurfaceScatter(planet.preset, time);
    DrawAuroraCurtains(planet, time);

    if (planet.preset.globalDipole) {
        const float tilt = planet.preset.dipoleTiltDeg * DEG2RAD;
        Vector3 axis = Vector3RotateByAxisAngle({0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, tilt);
        axis = Vector3Normalize(axis);
        const Vector3 north = Vector3Add(planet.preset.center, Vector3Scale(axis, r * 1.06f));
        const Vector3 south = Vector3Add(planet.preset.center, Vector3Scale(axis, -r * 1.06f));
        DrawLine3D(south, north, WithAlpha(planet.preset.auroraColor, 0.32f));
    }
}

void DrawParticles(const std::vector<WindParticle>& particles, const std::array<PlanetState, 4>& planets, int selectedPlanet) {
    for (const WindParticle& particle : particles) {
        const PlanetState& planet = planets[particle.planetIndex];
        const Vector3 p = Vector3Add(planet.preset.center, particle.localPos);
        const Vector3 q = Vector3Add(planet.preset.center, particle.prevLocalPos);

        const float yz = std::sqrt(particle.localPos.y * particle.localPos.y + particle.localPos.z * particle.localPos.z);
        const float shell = MagnetopauseBoundaryYZ(planet, particle.localPos.x);
        const bool insideField = shell > 0.0f && yz < shell;
        Color color = insideField ? planet.preset.auroraColor : Color{178, 228, 255, 255};

        if (particle.planetIndex != selectedPlanet) {
            color = WithAlpha(color, insideField ? 0.58f : 0.34f);
        }

        DrawLine3D(q, p, WithAlpha(color, insideField ? 0.30f : 0.16f));
        DrawSphere(p, particle.size, color);
    }
}

void DrawPanelLabels(const std::array<PlanetState, 4>& planets, const Camera3D& camera, int selectedPlanet) {
    for (int i = 0; i < static_cast<int>(planets.size()); ++i) {
        const PlanetState& planet = planets[i];
        const Vector3 anchor = Vector3Add(planet.preset.center, {0.0f, planet.bowShockRadius + 1.2f, 0.0f});
        const Vector2 screen = GetWorldToScreen(anchor, camera);
        if (screen.x < -220.0f || screen.x > static_cast<float>(GetScreenWidth()) + 220.0f ||
            screen.y < -80.0f || screen.y > static_cast<float>(GetScreenHeight()) + 80.0f) {
            continue;
        }

        const bool selected = i == selectedPlanet;
        const int boxW = 210;
        const int boxH = 62;
        const int x = static_cast<int>(screen.x - boxW * 0.5f);
        const int y = static_cast<int>(screen.y - boxH * 0.5f);
        const Color accent = selected ? Brighten(planet.preset.fieldColor, 0.18f) : planet.preset.fieldColor;

        DrawRectangleRounded(Rectangle{static_cast<float>(x), static_cast<float>(y), static_cast<float>(boxW), static_cast<float>(boxH)},
                             0.20f,
                             8,
                             Fade(Color{8, 13, 24, 255}, selected ? 0.88f : 0.70f));
        DrawRectangleLinesEx(Rectangle{static_cast<float>(x), static_cast<float>(y), static_cast<float>(boxW), static_cast<float>(boxH)},
                             1.2f,
                             Fade(accent, selected ? 0.76f : 0.34f));
        DrawText(TextFormat("%d  %s", i + 1, planet.preset.name), x + 12, y + 10, 24, Color{236, 240, 248, 255});
        DrawText(TextFormat("shield %.1f Rp", planet.magnetopauseRadius / planet.preset.displayRadius), x + 12, y + 36, 18, accent);
    }
}

void DrawComparisonPanel(const std::array<PlanetState, 4>& planets,
                         int selectedPlanet,
                         float windSpeed,
                         float windDensity,
                         float imfTiltDeg,
                         bool paused) {
    const Rectangle panel = {1000.0f, 22.0f, 568.0f, 310.0f};
    DrawRectangleRounded(panel, 0.04f, 10, Fade(Color{6, 10, 20, 255}, 0.84f));
    DrawRectangleLinesEx(panel, 1.1f, Fade(Color{94, 146, 220, 255}, 0.22f));

    DrawText("Planet Magnetosphere Compare", 1024, 38, 32, Color{232, 239, 248, 255});
    DrawText("Dense 3D toy visualization of how solar wind pressure meets planetary shielding.", 1024, 72, 18, Color{164, 186, 222, 255});
    DrawText("Mouse orbit | wheel zoom | 1..4 select | [ ] wind speed | - / + density | I/K IMF tilt | P pause | R reset",
             1024,
             98,
             16,
             Color{138, 214, 255, 255});

    DrawText(TextFormat("wind %.2fx   density %.2fx   IMF %+0.0f deg%s",
                        windSpeed,
                        windDensity,
                        imfTiltDeg,
                        paused ? "   [PAUSED]" : ""),
             1024,
             126,
             22,
             Color{255, 216, 142, 255});

    int y = 170;
    for (int i = 0; i < static_cast<int>(planets.size()); ++i) {
        const PlanetState& planet = planets[i];
        const bool selected = i == selectedPlanet;
        const Color accent = selected ? Brighten(planet.preset.fieldColor, 0.18f) : planet.preset.fieldColor;
        const float ratio = planet.magnetopauseRadius / planet.preset.displayRadius;
        const int barW = static_cast<int>(std::clamp(ratio / 9.0f, 0.0f, 1.0f) * 244.0f);

        DrawText(planet.preset.name, 1024, y, 20, selected ? Color{245, 248, 252, 255} : Color{208, 220, 236, 255});
        DrawRectangle(1148, y + 5, 252, 13, Fade(Color{44, 58, 84, 255}, 0.88f));
        DrawRectangle(1148, y + 5, barW, 13, accent);
        DrawText(TextFormat("%.1f Rp", ratio), 1414, y - 1, 18, accent);
        y += 30;
    }

    const PlanetState& active = planets[selectedPlanet];
    DrawRectangleRounded(Rectangle{1024.0f, 292.0f, 514.0f, 148.0f}, 0.06f, 8, Fade(Color{14, 22, 38, 255}, 0.92f));
    DrawText(active.preset.name, 1042, 308, 28, Brighten(active.preset.fieldColor, 0.14f));
    DrawText(active.preset.blurb, 1042, 342, 18, Color{196, 208, 226, 255});
    DrawText(TextFormat("bow shock %.1f Rp   tail %.1f Rp   aurora %.0f%%",
                        active.bowShockRadius / active.preset.displayRadius,
                        active.tailLength / active.preset.displayRadius,
                        active.auroraStrength * 100.0f),
             1042,
             388,
             19,
             Color{255, 218, 148, 255});
}

void DrawScreenEffects() {
    DrawRectangleGradientV(0, 0, GetScreenWidth(), GetScreenHeight(), Color{5, 7, 16, 0}, Color{0, 0, 0, 120});

    BeginBlendMode(BLEND_ADDITIVE);
    DrawCircleGradient(124, 120, 260.0f, Fade(Color{255, 168, 88, 255}, 0.18f), Fade(Color{255, 128, 64, 255}, 0.0f));
    DrawCircleGradient(0, GetScreenHeight() / 2, 420.0f, Fade(Color{255, 136, 78, 255}, 0.05f), Fade(Color{255, 136, 78, 255}, 0.0f));
    DrawCircleGradient(GetScreenWidth() - 170, GetScreenHeight() - 100, 340.0f, Fade(Color{86, 126, 255, 255}, 0.05f), Fade(Color{86, 126, 255, 255}, 0.0f));
    EndBlendMode();
}

}  // namespace

int main() {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(kScreenWidth, kScreenHeight, "Planet Magnetosphere Compare - C++ (raylib)");
    SetWindowMinSize(1180, 740);
    SetTargetFPS(120);

    Camera3D camera{};
    camera.position = {29.0f, 13.0f, 24.0f};
    camera.target = {1.8f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 40.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    OrbitCameraState orbit{};
    float windSpeed = 1.0f;
    float windDensity = 1.0f;
    float imfTiltDeg = 8.0f;
    bool paused = false;
    int selectedPlanet = 1;

    std::array<PlanetState, 4> planets = MakePlanets();
    UpdatePlanetDerivedState(&planets, windSpeed, windDensity, imfTiltDeg, 0.0f);
    std::vector<BackdropStar> stars = MakeBackdropStars();
    std::vector<DustMote> dust = MakeDustCloud();
    std::vector<WindParticle> particles = MakeWindParticles(planets);

    while (!WindowShouldClose()) {
        const float dt = std::max(1.0e-4f, GetFrameTime());
        const float time = static_cast<float>(GetTime());

        if (IsKeyPressed(KEY_ONE)) selectedPlanet = 0;
        if (IsKeyPressed(KEY_TWO)) selectedPlanet = 1;
        if (IsKeyPressed(KEY_THREE)) selectedPlanet = 2;
        if (IsKeyPressed(KEY_FOUR)) selectedPlanet = 3;
        if (IsKeyPressed(KEY_P)) paused = !paused;

        if (IsKeyPressed(KEY_R)) {
            windSpeed = 1.0f;
            windDensity = 1.0f;
            imfTiltDeg = 8.0f;
            paused = false;
            UpdatePlanetDerivedState(&planets, windSpeed, windDensity, imfTiltDeg, time);
            particles = MakeWindParticles(planets);
        }

        if (IsKeyDown(KEY_RIGHT_BRACKET)) windSpeed = std::min(2.4f, windSpeed + 0.55f * dt);
        if (IsKeyDown(KEY_LEFT_BRACKET)) windSpeed = std::max(0.35f, windSpeed - 0.55f * dt);
        if (IsKeyDown(KEY_EQUAL) || IsKeyDown(KEY_KP_ADD)) windDensity = std::min(2.5f, windDensity + 0.60f * dt);
        if (IsKeyDown(KEY_MINUS) || IsKeyDown(KEY_KP_SUBTRACT)) windDensity = std::max(0.25f, windDensity - 0.60f * dt);
        if (IsKeyDown(KEY_I)) imfTiltDeg = std::min(45.0f, imfTiltDeg + 28.0f * dt);
        if (IsKeyDown(KEY_K)) imfTiltDeg = std::max(-45.0f, imfTiltDeg - 28.0f * dt);

        UpdateOrbitCameraDragOnly(&camera, &orbit);
        UpdatePlanetDerivedState(&planets, windSpeed, windDensity, imfTiltDeg, time);
        if (!paused) {
            UpdateWindParticles(&particles, planets, dt, time, windSpeed, imfTiltDeg);
        }

        BeginDrawing();
        ClearBackground(Color{3, 5, 12, 255});
        DrawScreenEffects();

        BeginMode3D(camera);
        DrawBackground(stars, dust, time);
        DrawSun(time);

        for (int i = 0; i < static_cast<int>(planets.size()); ++i) {
            const bool selected = i == selectedPlanet;
            DrawWindGuide(planets[i], selected);
            DrawSolarWindStreamlines(planets[i], time, imfTiltDeg, selected);
            DrawBowShock(planets[i], time, selected);
            DrawMagnetopauseShell(planets[i], selected);
            DrawDipoleFieldLines(planets[i], time, imfTiltDeg, selected);
            DrawTailRibbon(planets[i], time, 0.7f, planets[i].preset.displayRadius * 1.6f, planets[i].preset.fieldColor, selected ? 0.10f : 0.05f);
            DrawTailRibbon(planets[i], time, 2.2f, planets[i].preset.displayRadius * 1.1f, planets[i].preset.auroraColor, selected ? 0.11f : 0.06f);
            DrawPlanetBody(planets[i], time, selected);
        }

        DrawParticles(particles, planets, selectedPlanet);
        EndMode3D();

        DrawPanelLabels(planets, camera, selectedPlanet);
        DrawComparisonPanel(planets, selectedPlanet, windSpeed, windDensity, imfTiltDeg, paused);
        DrawText("Planetary magnetic shielding under a shared stellar wind", 28, 28, 34, Color{236, 242, 250, 255});
        DrawText("More cinematic than literal: the scene exaggerates structure so the magnetospheres read clearly in motion.",
                 28,
                 64,
                 18,
                 Color{170, 190, 222, 255});
        DrawFPS(28, GetScreenHeight() - 36);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
