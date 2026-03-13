#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kScreenWidth = 1440;
constexpr int kScreenHeight = 920;
constexpr float kMinRadiusRs = 0.12f;
constexpr float kMaxRadiusRs = 18.0f;
constexpr float kEventHorizonRs = 1.0f;
constexpr int kStarCount = 720;
constexpr int kDiskParticleCount = 1800;
constexpr int kTunnelParticleCount = 540;
constexpr int kExitStarCount = 980;
constexpr int kGasCloudCount = 42;
constexpr float kDiskInnerRadius = 2.2f;
constexpr float kDiskOuterRadius = 12.0f;

struct BackgroundStar {
    float theta = 0.0f;
    float phi = 0.0f;
    float radius = 0.0f;
    float twinkle = 0.0f;
    float temperature = 0.0f;
    float size = 0.0f;
};

struct DiskParticle {
    float radius = 0.0f;
    float angle = 0.0f;
    float angularSpeed = 0.0f;
    float height = 0.0f;
    float phase = 0.0f;
    float heat = 0.0f;
    float size = 0.0f;
    float inward = 0.0f;
};

struct TunnelParticle {
    float lane = 0.0f;
    float depth = 0.0f;
    float speed = 0.0f;
    float swirl = 0.0f;
    float heat = 0.0f;
};

struct DestinationPlanet {
    Vector3 center{};
    float radius = 1.0f;
    float orbitPhase = 0.0f;
    float orbitAmp = 0.0f;
    Color inner{};
    Color outer{};
    Color glow{};
};

struct GasCloud {
    Vector3 center{};
    float radius = 1.0f;
    float drift = 0.0f;
    float pulse = 0.0f;
    Color color{};
};

struct LookState {
    float yaw = 0.0f;
    float pitch = 0.0f;
};

struct PhysicsState {
    float radiusRs = 10.0f;
    float zoom01 = 0.0f;
    float rawZoom = 0.0f;
    float clockRate = 1.0f;
    float tidal = 0.0f;
    float lensing = 0.0f;
    float frontShift = 1.0f;
    float photonRingStrength = 0.0f;
    float wormholeBlend = 0.0f;
    float tunnelTravel = 0.0f;
    float exitProgress = 0.0f;
    float phase01 = 0.0f;
    bool insideHorizon = false;
};

float Clamp01(float v) {
    return std::clamp(v, 0.0f, 1.0f);
}

float ClampRange(float v, float lo, float hi) {
    return std::clamp(v, lo, hi);
}

float Mix(float a, float b, float t) {
    return a + (b - a) * Clamp01(t);
}

Vector3 MixVec(Vector3 a, Vector3 b, float t) {
    const float u = Clamp01(t);
    return {
        a.x + (b.x - a.x) * u,
        a.y + (b.y - a.y) * u,
        a.z + (b.z - a.z) * u
    };
}

float SmoothStep(float a, float b, float x) {
    if (std::fabs(b - a) < 1e-6f) return x >= b ? 1.0f : 0.0f;
    float t = Clamp01((x - a) / (b - a));
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

Color ScaleColor(Color c, float scale, unsigned char alpha) {
    const float s = std::max(0.0f, scale);
    return Color{
        static_cast<unsigned char>(std::clamp(c.r * s, 0.0f, 255.0f)),
        static_cast<unsigned char>(std::clamp(c.g * s, 0.0f, 255.0f)),
        static_cast<unsigned char>(std::clamp(c.b * s, 0.0f, 255.0f)),
        alpha,
    };
}

std::string FormatFloat(float value, int decimals = 2) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(decimals) << value;
    return out.str();
}

Color TemperatureColor(float t) {
    return LerpColor(Color{150, 196, 255, 255}, Color{255, 230, 174, 255}, t);
}

Color ShiftSpectrum(Color base, float shift) {
    const float t = Clamp01((shift - 0.8f) / 4.0f);
    if (shift >= 1.0f) return LerpColor(base, Color{110, 220, 255, 255}, t);
    return LerpColor(base, Color{255, 100, 66, 255}, Clamp01((1.0f - shift) / 0.8f));
}

std::vector<BackgroundStar> BuildStars() {
    std::mt19937 rng(424242);
    std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * PI);
    std::uniform_real_distribution<float> phiDist(-0.48f * PI, 0.48f * PI);
    std::uniform_real_distribution<float> unitDist(0.0f, 1.0f);

    std::vector<BackgroundStar> stars;
    stars.reserve(kStarCount);
    for (int i = 0; i < kStarCount; ++i) {
        stars.push_back({
            angleDist(rng),
            phiDist(rng),
            Mix(80.0f, 125.0f, unitDist(rng)),
            angleDist(rng),
            unitDist(rng),
            Mix(0.04f, 0.18f, unitDist(rng)),
        });
    }
    return stars;
}

std::vector<DiskParticle> BuildDiskParticles() {
    std::mt19937 rng(31337);
    std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * PI);
    std::uniform_real_distribution<float> unitDist(0.0f, 1.0f);

    std::vector<DiskParticle> particles;
    particles.reserve(kDiskParticleCount);
    for (int i = 0; i < kDiskParticleCount; ++i) {
        const float u = unitDist(rng);
        const float radius = Mix(kDiskInnerRadius, kDiskOuterRadius, std::pow(u, 0.72f));
        const float heat = 1.0f - Clamp01((radius - kDiskInnerRadius) / (kDiskOuterRadius - kDiskInnerRadius));
        particles.push_back({
            radius,
            angleDist(rng),
            0.30f / std::pow(radius, 1.08f) + Mix(-0.008f, 0.008f, unitDist(rng)),
            Mix(-0.25f, 0.25f, unitDist(rng)) * (0.2f + 0.3f * radius / kDiskOuterRadius),
            angleDist(rng),
            heat,
            Mix(0.025f, 0.085f, unitDist(rng)),
            Mix(0.02f, 0.13f, unitDist(rng)),
        });
    }
    return particles;
}

std::vector<TunnelParticle> BuildTunnelParticles() {
    std::mt19937 rng(7171);
    std::uniform_real_distribution<float> unitDist(0.0f, 1.0f);

    std::vector<TunnelParticle> particles;
    particles.reserve(kTunnelParticleCount);
    for (int i = 0; i < kTunnelParticleCount; ++i) {
        particles.push_back({
            unitDist(rng),
            Mix(0.0f, 16.0f, unitDist(rng)),
            Mix(0.9f, 3.2f, unitDist(rng)),
            Mix(0.5f, 2.2f, unitDist(rng)),
            unitDist(rng),
        });
    }
    return particles;
}

std::vector<BackgroundStar> BuildExitStars() {
    std::mt19937 rng(919191);
    std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * PI);
    std::uniform_real_distribution<float> phiDist(-0.42f * PI, 0.42f * PI);
    std::uniform_real_distribution<float> unitDist(0.0f, 1.0f);

    std::vector<BackgroundStar> stars;
    stars.reserve(kExitStarCount);
    for (int i = 0; i < kExitStarCount; ++i) {
        stars.push_back({
            angleDist(rng),
            phiDist(rng),
            Mix(120.0f, 190.0f, unitDist(rng)),
            angleDist(rng),
            Mix(0.15f, 1.0f, unitDist(rng)),
            Mix(0.03f, 0.16f, unitDist(rng)),
        });
    }
    return stars;
}

std::vector<DestinationPlanet> BuildDestinationPlanets() {
    return {
        {{-13.0f, -2.8f, -36.0f}, 2.8f, 0.7f, 0.45f, Color{202, 170, 116, 255}, Color{88, 62, 40, 255}, Color{255, 196, 120, 90}},
        {{16.5f, 6.0f, -52.0f}, 4.2f, 1.8f, 0.30f, Color{120, 174, 210, 255}, Color{28, 40, 76, 255}, Color{146, 216, 255, 80}},
        {{4.5f, -7.8f, -27.0f}, 1.9f, 2.5f, 0.65f, Color{210, 120, 160, 255}, Color{78, 28, 56, 255}, Color{255, 150, 200, 70}},
    };
}

std::vector<GasCloud> BuildGasClouds() {
    std::mt19937 rng(606060);
    std::uniform_real_distribution<float> xDist(-32.0f, 28.0f);
    std::uniform_real_distribution<float> yDist(-16.0f, 16.0f);
    std::uniform_real_distribution<float> zDist(-88.0f, -24.0f);
    std::uniform_real_distribution<float> unitDist(0.0f, 1.0f);

    std::vector<GasCloud> clouds;
    clouds.reserve(kGasCloudCount);
    for (int i = 0; i < kGasCloudCount; ++i) {
        const float hue = unitDist(rng);
        clouds.push_back({
            {xDist(rng), yDist(rng), zDist(rng)},
            Mix(1.8f, 7.5f, unitDist(rng)),
            Mix(0.10f, 0.45f, unitDist(rng)),
            Mix(0.0f, 2.0f * PI, unitDist(rng)),
            LerpColor(Color{60, 130, 190, 255}, Color{110, 255, 240, 255}, hue),
        });
    }
    return clouds;
}

float RadiusToZoom01(float radiusRs) {
    const float logMin = std::log(kMinRadiusRs);
    const float logMax = std::log(kMaxRadiusRs);
    return 1.0f - (std::log(std::clamp(radiusRs, kMinRadiusRs, kMaxRadiusRs)) - logMin) / (logMax - logMin);
}

float Zoom01ToRadius(float zoom01) {
    const float logMin = std::log(kMinRadiusRs);
    const float logMax = std::log(kMaxRadiusRs);
    return std::exp(Mix(logMax, logMin, Clamp01(zoom01)));
}

PhysicsState EvaluatePhysics(float radiusRs, float zoom01) {
    PhysicsState state{};
    state.rawZoom = zoom01;
    state.zoom01 = ClampRange(zoom01, 0.0f, 1.45f);
    const float physicalZoom = Clamp01(state.zoom01 / 0.72f);
    state.radiusRs = std::clamp(Zoom01ToRadius(physicalZoom), kMinRadiusRs, kMaxRadiusRs);
    state.phase01 = state.zoom01;
    state.insideHorizon = state.phase01 >= 0.42f;

    const float outsideRadius = std::max(state.radiusRs, 1.0008f);
    state.clockRate = state.insideHorizon ? 0.0f : std::sqrt(1.0f - 1.0f / outsideRadius);
    state.tidal = 2.0f / (state.radiusRs * state.radiusRs * state.radiusRs);
    state.lensing = std::max(SmoothStep(0.05f, 1.3f, 1.0f / state.radiusRs), SmoothStep(0.18f, 0.48f, state.phase01));
    state.photonRingStrength = std::max(SmoothStep(0.06f, 1.0f, 1.0f / state.radiusRs), SmoothStep(0.24f, 0.54f, state.phase01));

    float beta = state.insideHorizon
        ? Mix(0.92f, 0.997f, Clamp01((kEventHorizonRs - state.radiusRs) / (kEventHorizonRs - kMinRadiusRs)))
        : std::clamp(std::sqrt(1.0f / outsideRadius), 0.0f, 0.994f);
    const float srForward = std::sqrt((1.0f + beta) / std::max(0.001f, 1.0f - beta));
    const float gravFactor = state.insideHorizon ? 4.8f : 1.0f / std::max(0.12f, state.clockRate);
    state.frontShift = std::clamp(srForward * gravFactor, 1.0f, 12.0f);

    state.wormholeBlend = SmoothStep(0.48f, 0.70f, state.phase01);
    state.tunnelTravel = SmoothStep(0.68f, 1.08f, state.phase01);
    state.exitProgress = SmoothStep(1.02f, 1.42f, state.phase01);
    return state;
}

float RadiusProfile(float u, float throatRadius, float flare) {
    return throatRadius + flare * (u * u);
}

Vector3 WormholePoint(float u, float theta, float throatRadius, float flare) {
    const float r = RadiusProfile(u, throatRadius, flare);
    return {r * std::cos(theta), r * std::sin(theta), u};
}

Vector3 DiskParticlePosition(const DiskParticle& p, float timeSeconds, const PhysicsState& state, bool trailPoint) {
    const float trailTime = trailPoint ? 0.06f : 0.0f;
    const float swirl = std::sin(timeSeconds * 0.9f + p.phase) * 0.06f;
    const float localRadius = p.radius * (1.0f - p.inward * state.lensing * 0.12f);
    const float angle = p.angle + (timeSeconds - trailTime) * p.angularSpeed * (1.0f + 0.16f * state.photonRingStrength);
    Vector3 pos = {
        std::cos(angle) * localRadius,
        p.height + std::sin((timeSeconds - trailTime) * 2.0f + p.phase) * 0.09f + swirl,
        std::sin(angle) * localRadius
    };
    pos = Vector3RotateByAxisAngle(pos, {1.0f, 0.0f, 0.0f}, 0.95f);
    pos = Vector3RotateByAxisAngle(pos, {0.0f, 0.0f, 1.0f}, -0.28f);
    return pos;
}

Vector3 InfallPathPosition(float radiusRs, float timeSeconds) {
    const float angle = -0.30f + 0.04f * std::sin(timeSeconds * 0.25f);
    Vector3 p = {
        std::cos(angle) * radiusRs * 0.30f,
        0.12f * radiusRs + 0.10f * std::sin(timeSeconds * 0.55f),
        radiusRs * 1.28f
    };
    p = Vector3RotateByAxisAngle(p, {0.0f, 1.0f, 0.0f}, 0.12f);
    return p;
}

Camera3D BuildFallingCamera(const PhysicsState& state, LookState* look, float timeSeconds, float dt) {
    Camera3D camera{};
    camera.projection = CAMERA_PERSPECTIVE;

    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        const Vector2 delta = GetMouseDelta();
        look->yaw -= delta.x * 0.0036f;
        look->pitch += delta.y * 0.0032f;
    } else {
        const float recenter = 0.85f + 2.2f * state.wormholeBlend;
        look->yaw = Mix(look->yaw, 0.0f, dt * recenter);
        look->pitch = Mix(look->pitch, 0.0f, dt * (recenter + 0.2f));
    }
    const float freeLookScale = 1.0f - 0.88f * state.wormholeBlend;
    look->yaw = std::clamp(look->yaw, -1.25f * freeLookScale, 1.25f * freeLookScale);
    look->pitch = std::clamp(look->pitch, -1.05f * freeLookScale, 1.05f * freeLookScale);

    const float throatRadius = Mix(1.25f, 2.60f, state.wormholeBlend);
    const float flare = Mix(0.18f, 0.055f, state.wormholeBlend);

    const Vector3 outsidePos = InfallPathPosition(state.radiusRs, timeSeconds);
    const Vector3 outsideTarget = {
        0.0f,
        -0.08f * state.phase01,
        Mix(0.0f, -1.2f, state.phase01)
    };

    const float tunnelU = Mix(8.0f, -18.0f, state.tunnelTravel);
    const float orbit = Mix(0.12f, 0.02f, state.tunnelTravel);
    const Vector3 wormholePos = {
        orbit * std::cos(timeSeconds * 0.55f),
        orbit * std::sin(timeSeconds * 0.45f),
        tunnelU
    };
    const Vector3 wormholeTarget = {
        0.04f * std::sin(timeSeconds * 0.35f),
        0.03f * std::cos(timeSeconds * 0.30f),
        tunnelU - 7.0f
    };
    const Vector3 exitTarget = {
        Mix(0.0f, 18.0f, state.exitProgress),
        Mix(0.0f, 6.0f, state.exitProgress),
        Mix(tunnelU - 7.0f, -78.0f, state.exitProgress)
    };
    const Vector3 exitPos = {
        Mix(wormholePos.x, 6.0f, state.exitProgress),
        Mix(wormholePos.y, 1.5f, state.exitProgress),
        Mix(wormholePos.z, -44.0f, state.exitProgress)
    };

    camera.position = MixVec(outsidePos, MixVec(wormholePos, exitPos, state.exitProgress), state.wormholeBlend);
    camera.target = MixVec(outsideTarget, MixVec(wormholeTarget, exitTarget, state.exitProgress), state.wormholeBlend);

    Vector3 lookDir = Vector3Normalize(Vector3Subtract(camera.target, camera.position));
    Vector3 baseUp = {0.0f, 1.0f, 0.0f};
    Vector3 right = Vector3Normalize(Vector3CrossProduct(lookDir, baseUp));
    if (Vector3Length(right) < 0.001f) right = {1.0f, 0.0f, 0.0f};
    baseUp = Vector3Normalize(Vector3CrossProduct(right, lookDir));

    if (std::fabs(look->yaw) > 0.0001f) {
        lookDir = Vector3RotateByAxisAngle(lookDir, baseUp, look->yaw);
        right = Vector3Normalize(Vector3CrossProduct(lookDir, baseUp));
    }
    if (std::fabs(look->pitch) > 0.0001f) {
        lookDir = Vector3RotateByAxisAngle(lookDir, right, look->pitch);
        baseUp = Vector3Normalize(Vector3CrossProduct(right, lookDir));
    }

    const float roll = 0.02f * std::sin(timeSeconds * 1.4f) + 0.08f * state.wormholeBlend * std::sin(timeSeconds * 0.7f);
    camera.up = Vector3RotateByAxisAngle(baseUp, lookDir, roll);
    camera.target = Vector3Add(camera.position, Vector3Scale(lookDir, 5.0f));
    camera.fovy = Mix(56.0f, 86.0f, state.wormholeBlend);

    (void)throatRadius;
    (void)flare;
    return camera;
}

void DrawScreenSpaceBlackHoleAnchor(const Camera3D& camera, const PhysicsState& state) {
    if (state.wormholeBlend > 0.45f) return;

    const Vector3 centerWorld = {0.0f, 0.0f, 0.0f};
    const Vector2 center = GetWorldToScreen(centerWorld, camera);
    if (center.x < -300.0f || center.x > static_cast<float>(GetScreenWidth()) + 300.0f ||
        center.y < -300.0f || center.y > static_cast<float>(GetScreenHeight()) + 300.0f) {
        return;
    }

    const Vector3 forward = Vector3Normalize(Vector3Subtract(camera.target, camera.position));
    Vector3 right = Vector3CrossProduct(forward, camera.up);
    if (Vector3Length(right) < 0.001f) right = {1.0f, 0.0f, 0.0f};
    right = Vector3Normalize(right);

    const Vector2 edge = GetWorldToScreen(Vector3Scale(right, 1.05f), camera);
    float pixelRadius = Vector2Distance(center, edge);
    pixelRadius = std::clamp(pixelRadius, 18.0f, 220.0f);

    const float lensedRadius = pixelRadius * (1.10f + 0.48f * state.lensing);
    const Color halo = Fade(ShiftSpectrum(Color{255, 196, 120, 255}, state.frontShift), 0.08f + 0.10f * state.photonRingStrength);
    DrawRing(center, lensedRadius * 1.03f, lensedRadius * 1.22f, 0.0f, 360.0f, 90, halo);
    DrawCircleV(center, lensedRadius, Fade(BLACK, 0.90f));
}

void DrawBackgroundGradient(const PhysicsState& state) {
    const Color top = LerpColor(
        LerpColor(Color{5, 8, 18, 255}, Color{14, 10, 12, 255}, state.phase01 * 0.7f),
        Color{14, 34, 48, 255},
        state.wormholeBlend * 0.72f
    );
    const Color bottom = LerpColor(
        LerpColor(Color{3, 4, 8, 255}, Color{18, 6, 8, 255}, state.phase01 * 0.85f),
        Color{4, 20, 30, 255},
        state.wormholeBlend * 0.82f
    );
    DrawRectangleGradientV(0, 0, GetScreenWidth(), GetScreenHeight(), top, bottom);
}

void DrawBackgroundStars3D(const std::vector<BackgroundStar>& stars, const PhysicsState& state, float timeSeconds) {
    const float fade = 1.0f - 0.72f * state.wormholeBlend;
    for (const BackgroundStar& star : stars) {
        const float theta = star.theta + timeSeconds * 0.0016f;
        const float phi = star.phi + 0.015f * std::sin(timeSeconds * 0.12f + star.twinkle);
        const Vector3 p = {
            star.radius * std::cos(phi) * std::cos(theta),
            star.radius * std::sin(phi),
            star.radius * std::cos(phi) * std::sin(theta)
        };
        const float twinkle = 0.55f + 0.45f * std::sin(timeSeconds * 1.2f + star.twinkle);
        Color c = ShiftSpectrum(TemperatureColor(star.temperature), Mix(1.0f, state.frontShift, 0.15f));
        c = ScaleColor(c, fade * twinkle, static_cast<unsigned char>((90 + 110 * twinkle) * fade));
        DrawSphere(p, star.size, c);
    }
}

void DrawExitStars3D(const std::vector<BackgroundStar>& stars, const PhysicsState& state, float timeSeconds) {
    const float fade = state.exitProgress * state.wormholeBlend;
    if (fade <= 0.01f) return;

    for (const BackgroundStar& star : stars) {
        const float theta = star.theta + timeSeconds * 0.0008f;
        const float phi = star.phi + 0.008f * std::sin(timeSeconds * 0.18f + star.twinkle);
        Vector3 p = {
            star.radius * std::cos(phi) * std::cos(theta),
            star.radius * std::sin(phi),
            star.radius * std::cos(phi) * std::sin(theta)
        };
        p.z -= 78.0f;
        p.x += 10.0f;
        p.y += 4.0f;

        const float twinkle = 0.68f + 0.32f * std::sin(timeSeconds * 1.6f + star.twinkle);
        Color c = LerpColor(Color{160, 205, 255, 255}, Color{190, 255, 248, 255}, star.temperature);
        c = ScaleColor(c, fade * twinkle, static_cast<unsigned char>((80 + 130 * twinkle) * fade));
        DrawSphere(p, star.size, c);
    }
}

void DrawAccretionDisk3D(const std::vector<DiskParticle>& particles, const Camera3D& camera,
                         const PhysicsState& state, float timeSeconds, float fade) {
    const Vector3 camDir = Vector3Normalize(Vector3Subtract(camera.position, camera.target));
    for (const DiskParticle& p : particles) {
        const Vector3 pos = DiskParticlePosition(p, timeSeconds, state, false);
        const Vector3 prev = DiskParticlePosition(p, timeSeconds, state, true);
        const Vector3 tangent = Vector3Normalize(Vector3Subtract(pos, prev));
        const float towardCamera = Clamp01(0.5f + 0.5f * Vector3DotProduct(tangent, camDir));
        const float shift = Mix(std::max(0.2f, 1.0f / state.frontShift), state.frontShift, towardCamera);
        const float heatBand = 0.4f + 0.6f * p.heat;
        Color base = LerpColor(Color{255, 116, 56, 255}, Color{255, 234, 174, 255}, heatBand);
        Color c = ShiftSpectrum(base, shift);
        c = ScaleColor(c, fade * (0.55f + 0.85f * towardCamera + 0.25f * state.photonRingStrength),
                       static_cast<unsigned char>((88 + 120 * heatBand) * fade));
        DrawLine3D(prev, pos, Fade(c, 0.20f * fade));
        DrawSphere(pos, p.size * (0.9f + 0.6f * towardCamera), c);
    }
}

void DrawBlackHoleCore(const PhysicsState& state, float timeSeconds, float fade) {
    const float ringRadius = Mix(1.5f, 2.0f, state.lensing);
    const int segments = 160;
    for (int i = 0; i < segments; ++i) {
        const float a0 = 2.0f * PI * static_cast<float>(i) / static_cast<float>(segments) + timeSeconds * 0.28f;
        const float a1 = 2.0f * PI * static_cast<float>(i + 1) / static_cast<float>(segments) + timeSeconds * 0.28f;
        const Vector3 p0 = {std::cos(a0) * ringRadius, 0.0f, std::sin(a0) * ringRadius};
        const Vector3 p1 = {std::cos(a1) * ringRadius, 0.0f, std::sin(a1) * ringRadius};
        const float glow = 0.45f + 0.55f * std::sin(a0 * 8.0f - timeSeconds * 2.0f);
        Color c = ShiftSpectrum(Color{255, 196, 116, 255}, Mix(1.0f, state.frontShift, 0.35f));
        c = ScaleColor(c, fade * (0.65f + 0.5f * glow), static_cast<unsigned char>((44 + 120 * state.photonRingStrength) * fade));
        DrawLine3D(p0, p1, c);
    }

    for (int strand = 0; strand < 16; ++strand) {
        const float base = 2.0f * PI * static_cast<float>(strand) / 16.0f;
        Vector3 prev{};
        for (int s = 0; s < 32; ++s) {
            const float t = static_cast<float>(s) / 31.0f;
            const float r = Mix(2.8f, 0.4f, t);
            const float z = Mix(4.2f, -0.5f, t);
            Vector3 p = {
                std::cos(base + timeSeconds * 0.5f + t * 2.4f) * r,
                std::sin(base * 1.3f + t * 4.0f + timeSeconds * 0.8f) * r * 0.18f,
                z
            };
            if (s > 0) DrawLine3D(prev, p, Fade(Color{255, 120, 84, 255}, (0.06f + 0.12f * (1.0f - t)) * fade));
            prev = p;
        }
    }

    DrawSphere({0.0f, 0.0f, 0.0f}, 0.92f, Fade(BLACK, fade));
    DrawSphereWires({0.0f, 0.0f, 0.0f}, 1.06f, 20, 20, Fade(Color{180, 205, 230, 255}, 0.14f * fade));
}

void DrawWormholeSurface(const PhysicsState& state, float timeSeconds, float cameraZ) {
    const float blend = state.wormholeBlend;
    if (blend <= 0.01f) return;

    const float throatRadius = Mix(1.25f, 2.60f, blend);
    const float flare = Mix(0.18f, 0.055f, blend);
    const int rings = 96;
    const int segs = 64;
    const float uMin = cameraZ - 22.0f;
    const float uMax = cameraZ + 16.0f;

    for (int i = 0; i < rings - 1; ++i) {
        const float u0 = uMin + (uMax - uMin) * static_cast<float>(i) / static_cast<float>(rings - 1);
        const float u1 = uMin + (uMax - uMin) * static_cast<float>(i + 1) / static_cast<float>(rings - 1);
        const float uMid = 0.5f * (u0 + u1);
        if (std::fabs(uMid - cameraZ) < 1.8f) continue;
        for (int j = 0; j < segs; ++j) {
            const float t0 = 2.0f * PI * static_cast<float>(j) / static_cast<float>(segs);
            const float t1 = 2.0f * PI * static_cast<float>(j + 1) / static_cast<float>(segs);
            Vector3 p00 = WormholePoint(u0, t0, throatRadius, flare);
            Vector3 p01 = WormholePoint(u0, t1, throatRadius, flare);
            Vector3 p10 = WormholePoint(u1, t0, throatRadius, flare);

            const float glow = 0.24f + 0.76f * (1.0f - std::min(1.0f, std::fabs(uMid - cameraZ) / 20.0f));
            const float pulse = 0.60f + 0.40f * std::sin(timeSeconds * 1.3f + uMid * 1.2f + t0 * 4.0f);
            Color c = LerpColor(Color{38, 84, 126, 255}, Color{98, 236, 255, 255}, glow * pulse);
            c = ScaleColor(c, blend * (1.0f - 0.55f * state.exitProgress), static_cast<unsigned char>(18 + 74 * blend * glow));
            DrawTriangle3D(p00, p10, p01, c);
        }
    }

    for (int ring = 0; ring < 16; ++ring) {
        const float u = uMin + (uMax - uMin) * static_cast<float>(ring) / 15.0f;
        if (std::fabs(u - cameraZ) < 1.4f) continue;
        for (int j = 0; j < segs; ++j) {
            const float t0 = 2.0f * PI * static_cast<float>(j) / static_cast<float>(segs);
            const float t1 = 2.0f * PI * static_cast<float>(j + 1) / static_cast<float>(segs);
            Vector3 p0 = WormholePoint(u, t0, throatRadius, flare);
            Vector3 p1 = WormholePoint(u, t1, throatRadius, flare);
            DrawLine3D(p0, p1, Fade(Color{90, 190, 255, 255}, (0.04f + 0.12f * blend) * (1.0f - 0.60f * state.exitProgress)));
        }
    }
}

void DrawDestinationGasClouds(const std::vector<GasCloud>& clouds, const PhysicsState& state, float timeSeconds) {
    const float fade = state.exitProgress * state.wormholeBlend;
    if (fade <= 0.01f) return;

    for (const GasCloud& cloud : clouds) {
        const float pulse = 0.55f + 0.45f * std::sin(timeSeconds * cloud.drift + cloud.pulse);
        const Vector3 center = {
            cloud.center.x + std::sin(timeSeconds * cloud.drift + cloud.pulse) * 1.4f,
            cloud.center.y + std::cos(timeSeconds * cloud.drift * 0.8f + cloud.pulse) * 0.9f,
            cloud.center.z
        };
        Color c = ScaleColor(cloud.color, fade * (0.45f + 0.55f * pulse), static_cast<unsigned char>(10 + 42 * fade * pulse));
        DrawSphere(center, cloud.radius * (0.86f + 0.18f * pulse), c);
        DrawSphere(center, cloud.radius * 0.58f, Fade(c, 0.16f));
    }
}

void DrawDestinationPlanets(const std::vector<DestinationPlanet>& planets, const PhysicsState& state, float timeSeconds) {
    const float fade = state.exitProgress * state.wormholeBlend;
    if (fade <= 0.01f) return;

    for (const DestinationPlanet& planet : planets) {
        Vector3 center = planet.center;
        center.x += std::sin(timeSeconds * 0.18f + planet.orbitPhase) * planet.orbitAmp;
        center.y += std::cos(timeSeconds * 0.14f + planet.orbitPhase) * planet.orbitAmp * 0.45f;

        DrawSphere(center, planet.radius, ScaleColor(planet.outer, fade, 255));
        DrawSphere(center, planet.radius * 0.76f, ScaleColor(planet.inner, fade, 215));
        DrawSphereWires(center, planet.radius * 1.08f, 14, 14, Fade(planet.glow, 0.12f * fade));
        DrawSphere(center, planet.radius * 1.18f, Fade(planet.glow, 0.035f * fade));
    }
}

void DrawExitQuasar(const PhysicsState& state, float timeSeconds) {
    const float fade = state.exitProgress * state.wormholeBlend;
    if (fade <= 0.01f) return;

    const Vector3 core = {18.0f, 6.0f, -68.0f};
    DrawSphere(core, 2.1f, Fade(BLACK, 0.95f));

    const int segments = 180;
    for (int i = 0; i < segments; ++i) {
        const float a0 = 2.0f * PI * static_cast<float>(i) / static_cast<float>(segments) + timeSeconds * 0.22f;
        const float a1 = 2.0f * PI * static_cast<float>(i + 1) / static_cast<float>(segments) + timeSeconds * 0.22f;
        const float ringR0 = Mix(4.5f, 11.0f, 0.5f + 0.5f * std::sin(a0 * 3.0f - timeSeconds));
        const float ringR1 = Mix(4.5f, 11.0f, 0.5f + 0.5f * std::sin(a1 * 3.0f - timeSeconds));
        Vector3 p0 = {core.x + std::cos(a0) * ringR0, core.y + std::sin(a0) * 0.8f, core.z + std::sin(a0) * ringR0 * 0.18f};
        Vector3 p1 = {core.x + std::cos(a1) * ringR1, core.y + std::sin(a1) * 0.8f, core.z + std::sin(a1) * ringR1 * 0.18f};
        Color c = ShiftSpectrum(Color{255, 206, 126, 255}, Mix(1.4f, 3.4f, fade));
        c = ScaleColor(c, 0.35f + 0.95f * fade, static_cast<unsigned char>(30 + 140 * fade));
        DrawLine3D(p0, p1, c);
    }

    for (int i = 0; i < 42; ++i) {
        const float t = static_cast<float>(i) / 41.0f;
        const float jetLen = Mix(2.0f, 26.0f, t);
        const float jetPulse = 0.72f + 0.28f * std::sin(timeSeconds * 2.0f - t * 7.0f);
        Color jet = ScaleColor(Color{120, 235, 255, 255}, fade * jetPulse, static_cast<unsigned char>(20 + 90 * fade * (1.0f - t)));
        DrawSphere({core.x, core.y + jetLen, core.z}, Mix(0.22f, 1.2f, 1.0f - t), jet);
        DrawSphere({core.x, core.y - jetLen, core.z}, Mix(0.22f, 1.2f, 1.0f - t), jet);
    }
}

void DrawQuasarLensingArcs(const PhysicsState& state, float timeSeconds) {
    const float fade = state.exitProgress * state.wormholeBlend;
    if (fade <= 0.01f) return;

    const Vector3 center = {18.0f, 6.0f, -68.0f};
    for (int arc = 0; arc < 3; ++arc) {
        const float radius = 8.0f + 3.5f * arc;
        const float tilt = 0.32f + 0.08f * arc;
        for (int i = 0; i < 72; ++i) {
            const float a0 = -0.9f + 1.6f * static_cast<float>(i) / 72.0f + timeSeconds * 0.05f * (arc + 1);
            const float a1 = -0.9f + 1.6f * static_cast<float>(i + 1) / 72.0f + timeSeconds * 0.05f * (arc + 1);
            Vector3 p0 = {center.x + std::cos(a0) * radius, center.y + std::sin(a0 * 1.8f) * tilt, center.z + std::sin(a0) * radius * 0.18f};
            Vector3 p1 = {center.x + std::cos(a1) * radius, center.y + std::sin(a1 * 1.8f) * tilt, center.z + std::sin(a1) * radius * 0.18f};
            DrawLine3D(p0, p1, Fade(Color{180, 235, 255, 255}, 0.05f + 0.10f * fade));
        }
    }
}

void DrawTunnelParticles(const std::vector<TunnelParticle>& particles, const PhysicsState& state, float timeSeconds) {
    const float blend = state.wormholeBlend;
    if (blend <= 0.01f) return;

    const float throatRadius = Mix(1.25f, 2.60f, blend);
    const float flare = Mix(0.18f, 0.055f, blend);

    for (const TunnelParticle& p : particles) {
        const float u = 18.0f - std::fmod(p.depth + timeSeconds * p.speed * Mix(1.0f, 2.4f, blend), 36.0f);
        const float theta = p.lane * 2.0f * PI + timeSeconds * p.swirl + 0.14f * std::sin(timeSeconds + u * 0.7f);
        const float radialBias = Mix(0.42f, 0.90f, 0.5f + 0.5f * std::sin(p.depth + timeSeconds * 0.9f));
        Vector3 pos = WormholePoint(u, theta, throatRadius * radialBias, flare * 0.65f);
        Vector3 prev = WormholePoint(u + 0.22f * p.speed, theta - 0.05f, throatRadius * radialBias, flare * 0.65f);

        Color c = ShiftSpectrum(LerpColor(Color{90, 220, 255, 255}, Color{255, 232, 178, 255}, p.heat), Mix(1.0f, state.frontShift, 0.2f));
        c = ScaleColor(c, (0.38f + 0.82f * blend) * (1.0f - 0.48f * state.exitProgress), static_cast<unsigned char>(44 + 126 * blend));
        DrawLine3D(prev, pos, Fade(c, 0.22f));
        DrawSphere(pos, Mix(0.025f, 0.06f, p.heat), c);
    }
}

void DrawExitBloom(const PhysicsState& state) {
    const float glow = 0.18f + 0.72f * state.wormholeBlend;
    DrawCircleGradient(GetScreenWidth() / 2, GetScreenHeight() / 2, 180.0f + 320.0f * state.tunnelTravel,
                       Fade(Color{210, 246, 255, 255}, 0.04f * glow + 0.08f * state.exitProgress),
                       Color{0, 0, 0, 0});
}

void DrawScene(const Camera3D& camera, const std::vector<BackgroundStar>& stars, const std::vector<BackgroundStar>& exitStars,
               const std::vector<DiskParticle>& disk, const std::vector<TunnelParticle>& tunnel,
               const std::vector<DestinationPlanet>& planets, const std::vector<GasCloud>& clouds,
               const PhysicsState& state, float timeSeconds) {
    BeginMode3D(camera);
    DrawBackgroundStars3D(stars, state, timeSeconds);
    DrawExitStars3D(exitStars, state, timeSeconds);
    DrawDestinationGasClouds(clouds, state, timeSeconds);
    DrawWormholeSurface(state, timeSeconds, camera.position.z);
    DrawTunnelParticles(tunnel, state, timeSeconds);
    DrawExitQuasar(state, timeSeconds);
    DrawQuasarLensingArcs(state, timeSeconds);
    DrawDestinationPlanets(planets, state, timeSeconds);

    const float diskFade = 1.0f - state.wormholeBlend * 0.92f;
    if (diskFade > 0.01f) {
        DrawAccretionDisk3D(disk, camera, state, timeSeconds, diskFade);
        DrawBlackHoleCore(state, timeSeconds, diskFade);
    }
    EndMode3D();
}

std::string PhaseLabel(const PhysicsState& state) {
    if (!state.insideHorizon) return "approach";
    if (state.wormholeBlend < 0.35f) return "horizon crossing";
    if (state.tunnelTravel < 0.45f) return "speculative throat transition";
    if (state.exitProgress < 0.45f) return "wormhole interior";
    if (state.exitProgress < 0.85f) return "emerging from wormhole";
    return "exit galaxy";
}

void DrawCompactHud(const PhysicsState& state, float currentRadiusRs, bool autoDive) {
    Rectangle card = {24.0f, static_cast<float>(GetScreenHeight()) - 116.0f, 420.0f, 86.0f};
    DrawRectangleRounded(card, 0.18f, 12, Color{7, 10, 15, 168});
    DrawRectangleRoundedLinesEx(card, 0.18f, 12, 1.0f, Color{62, 78, 102, 190});

    DrawText("Black Hole Infall -> Wormhole", static_cast<int>(card.x) + 14, static_cast<int>(card.y) + 10, 22, Color{236, 240, 246, 255});
    std::string line1 = "phase: " + PhaseLabel(state) + "   r = " + FormatFloat(currentRadiusRs) + " Rs";
    std::string line2 = "clock " + FormatFloat(state.clockRate, 3) + "   tidal " + FormatFloat(state.tidal, 2) + "   blend " + FormatFloat(state.wormholeBlend, 2);
    DrawText(line1.c_str(), static_cast<int>(card.x) + 14, static_cast<int>(card.y) + 40, 17, Color{162, 186, 214, 255});
    DrawText(line2.c_str(), static_cast<int>(card.x) + 14, static_cast<int>(card.y) + 60, 17, Color{110, 220, 255, 255});
    if (autoDive) DrawText("AUTO DIVE", static_cast<int>(card.x) + 314, static_cast<int>(card.y) + 10, 15, Color{255, 144, 96, 255});
}

void DrawHelpOverlay() {
    DrawText("drag to look   wheel / W,S change depth   space auto-dive   R reset   H hide help",
             24, 24, 18, Color{176, 184, 196, 185});

    Rectangle note = {static_cast<float>(GetScreenWidth()) - 360.0f, 24.0f, 328.0f, 88.0f};
    DrawRectangleRounded(note, 0.16f, 12, Color{8, 12, 18, 158});
    DrawRectangleRoundedLinesEx(note, 0.16f, 12, 1.0f, Color{64, 78, 100, 190});
    DrawText("Speculative visualization", static_cast<int>(note.x) + 14, static_cast<int>(note.y) + 12, 20, Color{234, 238, 244, 255});
    DrawText("GR-inspired infall, then a speculative", static_cast<int>(note.x) + 14, static_cast<int>(note.y) + 42, 16, Color{170, 182, 198, 240});
    DrawText("wormhole exit into another galaxy.", static_cast<int>(note.x) + 14, static_cast<int>(note.y) + 62, 16, Color{170, 182, 198, 240});
}

}  // namespace

int main() {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(kScreenWidth, kScreenHeight, "Black Hole Infall to Wormhole - C++ (raylib)");
    SetWindowMinSize(1180, 720);
    SetTargetFPS(60);

    const std::vector<BackgroundStar> stars = BuildStars();
    const std::vector<BackgroundStar> exitStars = BuildExitStars();
    const std::vector<DiskParticle> disk = BuildDiskParticles();
    const std::vector<TunnelParticle> tunnel = BuildTunnelParticles();
    const std::vector<DestinationPlanet> planets = BuildDestinationPlanets();
    const std::vector<GasCloud> clouds = BuildGasClouds();

    LookState look;
    float targetZoom01 = RadiusToZoom01(9.5f);
    float currentZoom01 = targetZoom01;
    float currentRadiusRs = Zoom01ToRadius(currentZoom01);
    float timeSeconds = 0.0f;
    bool autoDive = false;
    bool showHelp = true;

    while (!WindowShouldClose()) {
        const float dt = GetFrameTime();
        timeSeconds += dt;

        if (IsKeyPressed(KEY_SPACE)) autoDive = !autoDive;
        if (IsKeyPressed(KEY_H)) showHelp = !showHelp;
        if (IsKeyPressed(KEY_R)) {
            targetZoom01 = RadiusToZoom01(9.5f);
            currentZoom01 = targetZoom01;
            currentRadiusRs = Zoom01ToRadius(currentZoom01);
            autoDive = false;
            look = {};
        }

        targetZoom01 += GetMouseWheelMove() * 0.05f;
        if (IsKeyDown(KEY_W) || IsKeyDown(KEY_UP)) targetZoom01 += dt * 0.46f;
        if (IsKeyDown(KEY_S) || IsKeyDown(KEY_DOWN)) targetZoom01 -= dt * 0.46f;
        if (autoDive) targetZoom01 += dt * Mix(0.10f, 0.26f, Clamp01(currentZoom01 / 1.45f));
        targetZoom01 = ClampRange(targetZoom01, 0.0f, 1.45f);

        currentZoom01 = Mix(currentZoom01, targetZoom01, 1.0f - std::exp(-dt * 5.4f));
        currentRadiusRs = Zoom01ToRadius(currentZoom01);

        const PhysicsState state = EvaluatePhysics(currentRadiusRs, currentZoom01);
        const Camera3D camera = BuildFallingCamera(state, &look, timeSeconds, dt);

        BeginDrawing();
        DrawBackgroundGradient(state);
        DrawExitBloom(state);
        DrawScene(camera, stars, exitStars, disk, tunnel, planets, clouds, state, timeSeconds);
        DrawScreenSpaceBlackHoleAnchor(camera, state);
        DrawCompactHud(state, currentRadiusRs, autoDive);
        if (showHelp) DrawHelpOverlay();
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
