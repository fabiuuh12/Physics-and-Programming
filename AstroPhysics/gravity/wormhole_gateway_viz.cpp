#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int kScreenWidth = 1440;
constexpr int kScreenHeight = 900;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kHalfLength = 8.0f;

struct FlowParticle {
    float u;
    float theta;
    float radialFraction;
    float speed;
    float glow;
    float phase;
};

struct RimMote {
    int mouthSign;
    float theta;
    float ringFraction;
    float speed;
    float lift;
    float phase;
    float size;
};

struct EnergyStreak {
    float z;
    float theta;
    float radius;
    float speed;
    float age;
    float ttl;
    float width;
    int direction;
};

struct DistantStar {
    Vector3 position;
    float size;
    float tint;
    int side;
};

struct NebulaBlob {
    Vector3 position;
    float radius;
    float alpha;
    float tint;
    int side;
};

struct EnvironmentPalette {
    const char* name;
    Color mouth;
    Color inner;
    Color fog;
    Color star;
    Color nebula;
    Color accent;
};

struct CameraPreset {
    float yaw;
    float pitch;
    float distance;
    Vector3 target;
};

struct TransitState {
    bool active = false;
    int direction = 1;
    float z = -kHalfLength - 6.0f;
};

float RandomFloat(float minValue, float maxValue) {
    return minValue + (maxValue - minValue) * (static_cast<float>(GetRandomValue(0, 10000)) / 10000.0f);
}

Color LerpColor(Color a, Color b, float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return Color{
        static_cast<unsigned char>(a.r + (b.r - a.r) * t),
        static_cast<unsigned char>(a.g + (b.g - a.g) * t),
        static_cast<unsigned char>(a.b + (b.b - a.b) * t),
        static_cast<unsigned char>(a.a + (b.a - a.a) * t),
    };
}

void UpdateOrbitCamera(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 delta = GetMouseDelta();
        *yaw -= delta.x * 0.0035f;
        *pitch += delta.y * 0.0033f;
        *pitch = std::clamp(*pitch, -1.46f, 1.46f);
    }

    *distance -= GetMouseWheelMove() * 0.9f;
    *distance = std::clamp(*distance, 3.0f, 70.0f);

    float cp = std::cos(*pitch);
    camera->position = Vector3Add(camera->target, {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    });
}

void ApplyPreset(const CameraPreset& preset, Camera3D* camera, float* yaw, float* pitch, float* distance) {
    *yaw = preset.yaw;
    *pitch = preset.pitch;
    *distance = preset.distance;
    camera->target = preset.target;
    float cp = std::cos(*pitch);
    camera->position = Vector3Add(camera->target, {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    });
}

float WormholeRadius(float z, float throatRadius, float flare, float pulse) {
    float x = std::abs(z) / kHalfLength;
    float flareShape = flare * (0.28f + 2.3f * x * x + 0.85f * x * x * x);
    float breathing = pulse * (0.10f + 0.15f * x);
    return throatRadius + flareShape + breathing;
}

Vector3 TubePoint(float z, float theta, float throatRadius, float flare, float pulse) {
    float radius = WormholeRadius(z, throatRadius, flare, pulse);
    return {radius * std::cos(theta), radius * std::sin(theta), z};
}

Color TubeColor(float z, const EnvironmentPalette& nearPalette, const EnvironmentPalette& farPalette, float pulse) {
    float t = (z + kHalfLength) / (2.0f * kHalfLength);
    Color base = LerpColor(nearPalette.inner, farPalette.inner, t);
    return LerpColor(base, Color{255, 255, 255, 255}, pulse * 0.10f);
}

Vector3 DistortAroundMouth(Vector3 p, float mouthZ, float distortion) {
    float planeDistance = std::abs(p.z - mouthZ);
    float radial = std::sqrt(p.x * p.x + p.y * p.y);
    constexpr float influence = 11.0f;
    if (planeDistance > 7.5f || radial > influence) return p;

    float falloff = (1.0f - radial / influence) * std::exp(-planeDistance * 0.55f);
    falloff = std::max(falloff, 0.0f);
    float twist = distortion * 0.46f * falloff;
    float c = std::cos(twist);
    float s = std::sin(twist);
    Vector2 xy{
        p.x * c - p.y * s,
        p.x * s + p.y * c,
    };
    float pull = 1.0f - distortion * 0.16f * falloff;
    return {xy.x * pull, xy.y * pull, p.z - (p.z - mouthZ) * distortion * 0.06f * falloff};
}

void InitializeFlowParticles(std::vector<FlowParticle>* particles) {
    particles->clear();
    particles->reserve(340);
    for (int i = 0; i < 340; ++i) {
        particles->push_back(FlowParticle{
            RandomFloat(0.0f, 1.0f),
            RandomFloat(0.0f, 2.0f * kPi),
            RandomFloat(0.15f, 0.92f),
            RandomFloat(0.04f, 0.13f),
            RandomFloat(0.35f, 1.0f),
            RandomFloat(0.0f, 2.0f * kPi),
        });
    }
}

void InitializeRimMotes(std::vector<RimMote>* motes) {
    motes->clear();
    motes->reserve(180);
    for (int mouthSign : {-1, 1}) {
        for (int i = 0; i < 90; ++i) {
            motes->push_back(RimMote{
                mouthSign,
                RandomFloat(0.0f, 2.0f * kPi),
                RandomFloat(0.15f, 1.0f),
                RandomFloat(0.6f, 1.9f),
                RandomFloat(-0.3f, 0.3f),
                RandomFloat(0.0f, 2.0f * kPi),
                RandomFloat(0.03f, 0.09f),
            });
        }
    }
}

void InitializeStars(std::vector<DistantStar>* stars) {
    stars->clear();
    stars->reserve(170);
    for (int side : {-1, 1}) {
        for (int i = 0; i < 85; ++i) {
            float theta = RandomFloat(0.0f, 2.0f * kPi);
            float radial = RandomFloat(4.0f, 28.0f);
            float z = side * RandomFloat(kHalfLength + 2.5f, 34.0f);
            stars->push_back(DistantStar{
                {radial * std::cos(theta), radial * std::sin(theta) * RandomFloat(0.45f, 1.1f), z},
                RandomFloat(0.05f, 0.17f),
                RandomFloat(0.0f, 1.0f),
                side,
            });
        }
    }
}

void InitializeNebula(std::vector<NebulaBlob>* blobs) {
    blobs->clear();
    blobs->reserve(24);
    for (int side : {-1, 1}) {
        for (int i = 0; i < 12; ++i) {
            float theta = RandomFloat(0.0f, 2.0f * kPi);
            float radial = RandomFloat(3.0f, 16.0f);
            float z = side * RandomFloat(kHalfLength + 4.0f, 24.0f);
            blobs->push_back(NebulaBlob{
                {radial * std::cos(theta), radial * std::sin(theta), z},
                RandomFloat(1.8f, 4.6f),
                RandomFloat(0.03f, 0.10f),
                RandomFloat(0.0f, 1.0f),
                side,
            });
        }
    }
}

void SpawnEnergyStreak(std::vector<EnergyStreak>* streaks, int direction, float speedBoost) {
    streaks->push_back(EnergyStreak{
        direction > 0 ? -kHalfLength - 0.6f : kHalfLength + 0.6f,
        RandomFloat(0.0f, 2.0f * kPi),
        RandomFloat(0.02f, 0.20f),
        RandomFloat(7.0f, 12.0f) + speedBoost * 3.5f,
        0.0f,
        RandomFloat(1.0f, 2.2f),
        RandomFloat(0.04f, 0.09f),
        direction,
    });
}

void DrawWarpedRing(float z, float baseRadius, Color color, float wobble, float time, float phase) {
    const int segments = 120;
    for (int i = 0; i < segments; ++i) {
        float a0 = (2.0f * kPi * i) / static_cast<float>(segments);
        float a1 = (2.0f * kPi * (i + 1)) / static_cast<float>(segments);
        float r0 = baseRadius + wobble * std::sin(a0 * 6.0f + time * 2.2f + phase);
        float r1 = baseRadius + wobble * std::sin(a1 * 6.0f + time * 2.2f + phase);
        DrawLine3D({r0 * std::cos(a0), r0 * std::sin(a0), z},
                   {r1 * std::cos(a1), r1 * std::sin(a1), z}, color);
    }
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Wormhole Gateway 3D - C++ (raylib)");
    SetTargetFPS(60);

    const EnvironmentPalette nearPalette{
        "Cygnus Drift",
        Color{110, 224, 255, 255},
        Color{72, 132, 255, 255},
        Color{44, 98, 170, 255},
        Color{180, 220, 255, 255},
        Color{66, 120, 214, 255},
        Color{150, 244, 255, 255},
    };
    const std::array<EnvironmentPalette, 3> farPalettes = {{
        {"Amber Reach", Color{255, 180, 118, 255}, Color{255, 110, 78, 255}, Color{184, 74, 54, 255},
         Color{255, 224, 196, 255}, Color{214, 86, 72, 255}, Color{255, 212, 140, 255}},
        {"Crimson Veil", Color{255, 110, 138, 255}, Color{255, 74, 120, 255}, Color{144, 34, 86, 255},
         Color{255, 192, 220, 255}, Color{180, 28, 96, 255}, Color{255, 154, 188, 255}},
        {"Solar Ash", Color{255, 208, 110, 255}, Color{255, 152, 84, 255}, Color{160, 100, 52, 255},
         Color{255, 238, 204, 255}, Color{190, 126, 62, 255}, Color{255, 230, 150, 255}},
    }};

    Camera3D camera{};
    camera.position = {11.5f, 5.0f, 15.0f};
    camera.target = {0.0f, 0.0f, -kHalfLength};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 42.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    const std::array<CameraPreset, 3> leftPresets = {{
        {0.82f, 0.26f, 14.0f, {0.0f, 0.0f, -kHalfLength}},
        {1.58f, 0.02f, 12.0f, {0.0f, 0.0f, -2.0f}},
        {0.06f, 0.42f, 27.0f, {0.0f, 0.0f, 0.0f}},
    }};
    const std::array<CameraPreset, 3> rightPresets = {{
        {-2.36f, 0.26f, 14.0f, {0.0f, 0.0f, kHalfLength}},
        {-1.58f, 0.02f, 12.0f, {0.0f, 0.0f, 2.0f}},
        {3.08f, 0.42f, 27.0f, {0.0f, 0.0f, 0.0f}},
    }};

    float camYaw = leftPresets[0].yaw;
    float camPitch = leftPresets[0].pitch;
    float camDistance = leftPresets[0].distance;

    float throatRadius = 1.12f;
    float flare = 1.15f;
    float swirlIntensity = 1.35f;
    float transitSpeed = 1.0f;
    float distortion = 0.85f;
    bool paused = false;
    int farPaletteIndex = 0;
    int currentSide = -1;
    float time = 0.0f;
    float entrancePulse = 0.0f;
    float exitPulse = 0.0f;
    float delayedPulseTimer = -1.0f;
    float streakAccumulator = 0.0f;

    TransitState transit;
    std::vector<FlowParticle> flowParticles;
    std::vector<RimMote> rimMotes;
    std::vector<EnergyStreak> energyStreaks;
    std::vector<DistantStar> stars;
    std::vector<NebulaBlob> nebulaBlobs;
    InitializeFlowParticles(&flowParticles);
    InitializeRimMotes(&rimMotes);
    InitializeStars(&stars);
    InitializeNebula(&nebulaBlobs);
    energyStreaks.reserve(200);

    ApplyPreset(leftPresets[0], &camera, &camYaw, &camPitch, &camDistance);

    while (!WindowShouldClose()) {
        const EnvironmentPalette& farPalette = farPalettes[farPaletteIndex];

        auto applyContextPreset = [&](int index) {
            if (index < 0 || index > 2) return;
            if (currentSide < 0) ApplyPreset(leftPresets[index], &camera, &camYaw, &camPitch, &camDistance);
            else ApplyPreset(rightPresets[index], &camera, &camYaw, &camPitch, &camDistance);
        };

        auto beginTransit = [&]() {
            transit.active = true;
            transit.direction = currentSide < 0 ? 1 : -1;
            transit.z = currentSide < 0 ? -kHalfLength - 6.2f : kHalfLength + 6.2f;
            entrancePulse = 1.20f;
            exitPulse = 0.0f;
            delayedPulseTimer = 0.55f;
        };

        if (IsKeyPressed(KEY_ONE) && !transit.active) applyContextPreset(0);
        if (IsKeyPressed(KEY_TWO) && !transit.active) applyContextPreset(1);
        if (IsKeyPressed(KEY_THREE) && !transit.active) applyContextPreset(2);
        if (IsKeyPressed(KEY_FOUR) && !transit.active) beginTransit();

        if (IsKeyPressed(KEY_T) && !transit.active) beginTransit();
        if (IsKeyPressed(KEY_M)) farPaletteIndex = (farPaletteIndex + 1) % static_cast<int>(farPalettes.size());
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            throatRadius = 1.12f;
            flare = 1.15f;
            swirlIntensity = 1.35f;
            transitSpeed = 1.0f;
            distortion = 0.85f;
            paused = false;
            farPaletteIndex = 0;
            currentSide = -1;
            time = 0.0f;
            entrancePulse = 0.0f;
            exitPulse = 0.0f;
            delayedPulseTimer = -1.0f;
            streakAccumulator = 0.0f;
            transit = TransitState{};
            InitializeFlowParticles(&flowParticles);
            InitializeRimMotes(&rimMotes);
            InitializeStars(&stars);
            InitializeNebula(&nebulaBlobs);
            energyStreaks.clear();
            ApplyPreset(leftPresets[0], &camera, &camYaw, &camPitch, &camDistance);
        }

        if (IsKeyDown(KEY_UP)) throatRadius = std::min(2.2f, throatRadius + 0.55f * GetFrameTime());
        if (IsKeyDown(KEY_DOWN)) throatRadius = std::max(0.55f, throatRadius - 0.55f * GetFrameTime());
        if (IsKeyDown(KEY_RIGHT_BRACKET)) flare = std::min(2.2f, flare + 0.65f * GetFrameTime());
        if (IsKeyDown(KEY_LEFT_BRACKET)) flare = std::max(0.20f, flare - 0.65f * GetFrameTime());
        if (IsKeyDown(KEY_RIGHT)) swirlIntensity = std::min(3.2f, swirlIntensity + 1.0f * GetFrameTime());
        if (IsKeyDown(KEY_LEFT)) swirlIntensity = std::max(0.10f, swirlIntensity - 1.0f * GetFrameTime());
        if (IsKeyDown(KEY_EQUAL)) transitSpeed = std::min(2.6f, transitSpeed + 1.1f * GetFrameTime());
        if (IsKeyDown(KEY_MINUS)) transitSpeed = std::max(0.25f, transitSpeed - 1.1f * GetFrameTime());
        if (IsKeyDown(KEY_D)) distortion = std::min(2.3f, distortion + 1.1f * GetFrameTime());
        if (IsKeyDown(KEY_S)) distortion = std::max(0.1f, distortion - 1.1f * GetFrameTime());

        if (!transit.active) UpdateOrbitCamera(&camera, &camYaw, &camPitch, &camDistance);

        float dt = GetFrameTime();
        if (!paused) {
            time += dt;
            entrancePulse = std::max(0.0f, entrancePulse - dt);
            exitPulse = std::max(0.0f, exitPulse - dt);
            if (delayedPulseTimer > 0.0f) {
                delayedPulseTimer -= dt;
                if (delayedPulseTimer <= 0.0f) exitPulse = 1.1f;
            }

            float pulseWave = 0.5f + 0.5f * std::sin(time * 1.4f);
            for (FlowParticle& particle : flowParticles) {
                particle.u += particle.speed * dt * (0.6f + transitSpeed * 0.5f) * (0.7f + 0.7f * pulseWave);
                if (particle.u > 1.0f) particle.u -= 1.0f;
                particle.theta += dt * swirlIntensity * (0.9f + particle.radialFraction * 0.8f);
            }
            for (RimMote& mote : rimMotes) {
                mote.theta += mote.speed * dt * (0.9f + 0.3f * pulseWave);
            }

            if (transit.active) {
                transit.z += transit.direction * dt * transitSpeed * 9.5f;
                float sway = 0.22f + 0.08f * distortion;
                camera.position = {
                    sway * std::sin(time * 1.9f),
                    sway * std::cos(time * 1.6f),
                    transit.z,
                };
                camera.target = {
                    0.18f * std::sin(time * 2.2f),
                    0.14f * std::cos(time * 2.0f),
                    transit.z + transit.direction * 3.6f,
                };
                camera.fovy = 42.0f + 6.0f * std::sin(std::clamp((std::abs(transit.z) / (kHalfLength + 6.2f)), 0.0f, 1.0f) * kPi);

                if ((transit.direction > 0 && transit.z > kHalfLength + 6.4f) ||
                    (transit.direction < 0 && transit.z < -kHalfLength - 6.4f)) {
                    transit.active = false;
                    currentSide *= -1;
                    camera.fovy = 42.0f;
                    applyContextPreset(0);
                }
            } else {
                camera.fovy = 42.0f;
            }

            streakAccumulator += dt * (5.0f + 4.0f * swirlIntensity + (transit.active ? 10.0f : 0.0f));
            while (streakAccumulator >= 1.0f) {
                streakAccumulator -= 1.0f;
                SpawnEnergyStreak(&energyStreaks, 1, transitSpeed + swirlIntensity * 0.3f);
                if (GetRandomValue(0, 100) > 40) SpawnEnergyStreak(&energyStreaks, -1, transitSpeed + swirlIntensity * 0.3f);
            }

            for (EnergyStreak& streak : energyStreaks) {
                streak.age += dt;
                streak.z += streak.direction * streak.speed * dt;
                streak.theta += dt * swirlIntensity * 1.4f;
            }
            energyStreaks.erase(
                std::remove_if(energyStreaks.begin(), energyStreaks.end(), [](const EnergyStreak& streak) {
                    return streak.age > streak.ttl || std::abs(streak.z) > kHalfLength + 10.0f;
                }),
                energyStreaks.end());
        }

        float throatPulse = 0.5f + 0.5f * std::sin(time * 1.35f);
        float pulseValue = throatPulse * 0.75f + std::max(entrancePulse, exitPulse) * 0.25f;
        float entranceMouthZ = -kHalfLength;
        float exitMouthZ = kHalfLength;

        BeginDrawing();
        ClearBackground(Color{3, 5, 11, 255});

        BeginMode3D(camera);

        for (const NebulaBlob& blob : nebulaBlobs) {
            const EnvironmentPalette& palette = blob.side < 0 ? nearPalette : farPalette;
            Color c = LerpColor(palette.nebula, palette.fog, blob.tint);
            DrawSphere(blob.position, blob.radius, Fade(c, blob.alpha));
        }

        for (const DistantStar& star : stars) {
            const EnvironmentPalette& palette = star.side < 0 ? nearPalette : farPalette;
            float mouthZ = star.side < 0 ? entranceMouthZ : exitMouthZ;
            Vector3 warped = DistortAroundMouth(star.position, mouthZ, distortion);
            Color c = LerpColor(palette.star, Color{255, 255, 255, 255}, star.tint);
            DrawSphere(warped, star.size, Fade(c, 0.92f));
        }

        const int rings = 74;
        const int segments = 56;
        for (int i = 0; i < rings - 1; ++i) {
            float z0 = -kHalfLength + (2.0f * kHalfLength * i) / static_cast<float>(rings - 1);
            float z1 = -kHalfLength + (2.0f * kHalfLength * (i + 1)) / static_cast<float>(rings - 1);
            float pulse0 = pulseValue * std::sin(time * 1.1f + z0 * 0.35f);
            float pulse1 = pulseValue * std::sin(time * 1.1f + z1 * 0.35f);
            for (int j = 0; j < segments; ++j) {
                float a0 = (2.0f * kPi * j) / static_cast<float>(segments);
                float a1 = (2.0f * kPi * (j + 1)) / static_cast<float>(segments);
                Vector3 p00 = TubePoint(z0, a0, throatRadius, flare, pulse0);
                Vector3 p01 = TubePoint(z0, a1, throatRadius, flare, pulse0);
                Vector3 p10 = TubePoint(z1, a0, throatRadius, flare, pulse1);
                Vector3 p11 = TubePoint(z1, a1, throatRadius, flare, pulse1);

                float centerGlow = 1.0f - std::abs(z0) / kHalfLength;
                Color c = TubeColor(z0, nearPalette, farPalette, pulseValue);
                c.a = static_cast<unsigned char>(28 + 58 * centerGlow + 25 * pulseValue);
                DrawTriangle3D(p00, p10, p01, c);
                DrawTriangle3D(p01, p10, p11, c);
            }
        }

        for (int i = 0; i < 28; ++i) {
            float z = -kHalfLength + std::fmod(time * (3.0f + transitSpeed * 2.0f) + i * 0.68f, 2.0f * kHalfLength);
            float pulse = std::sin(time * 1.2f + z * 0.32f);
            float radius = WormholeRadius(z, throatRadius, flare, pulseValue * pulse);
            Color contour = Fade(TubeColor(z, nearPalette, farPalette, pulseValue), 0.10f + 0.08f * pulseValue);
            DrawWarpedRing(z, radius, contour, 0.08f + swirlIntensity * 0.04f, time * swirlIntensity, i * 0.5f);
        }

        DrawWarpedRing(entranceMouthZ, WormholeRadius(entranceMouthZ, throatRadius, flare, pulseValue) + 0.25f,
                       Fade(nearPalette.mouth, 0.62f + entrancePulse * 0.15f), 0.20f + entrancePulse * 0.35f, time, 0.0f);
        DrawWarpedRing(entranceMouthZ, WormholeRadius(entranceMouthZ, throatRadius, flare, pulseValue) + 0.65f,
                       Fade(nearPalette.accent, 0.32f), 0.26f + entrancePulse * 0.28f, time, 1.2f);
        DrawWarpedRing(exitMouthZ, WormholeRadius(exitMouthZ, throatRadius, flare, pulseValue) + 0.25f,
                       Fade(farPalette.mouth, 0.62f + exitPulse * 0.15f), 0.20f + exitPulse * 0.35f, time, 0.8f);
        DrawWarpedRing(exitMouthZ, WormholeRadius(exitMouthZ, throatRadius, flare, pulseValue) + 0.65f,
                       Fade(farPalette.accent, 0.32f), 0.26f + exitPulse * 0.28f, time, 2.1f);

        for (const RimMote& mote : rimMotes) {
            float mouthZ = mote.mouthSign < 0 ? entranceMouthZ : exitMouthZ;
            const EnvironmentPalette& palette = mote.mouthSign < 0 ? nearPalette : farPalette;
            float mouthRadius = WormholeRadius(mouthZ, throatRadius, flare, pulseValue);
            float ringRadius = mouthRadius + 0.2f + mote.ringFraction * 1.6f;
            Vector3 position{
                ringRadius * std::cos(mote.theta),
                ringRadius * std::sin(mote.theta),
                mouthZ + mote.lift + 0.18f * std::sin(time * 2.0f + mote.phase),
            };
            DrawSphere(position, mote.size, Fade(palette.accent, 0.12f + 0.20f * mote.ringFraction));
        }

        for (const FlowParticle& particle : flowParticles) {
            float z = -kHalfLength + particle.u * (2.0f * kHalfLength);
            float centerFactor = 1.0f - std::abs(z) / kHalfLength;
            float radius = WormholeRadius(z, throatRadius, flare, pulseValue);
            float lane = radius * (0.22f + particle.radialFraction * 0.58f +
                                   0.06f * std::sin(time * 2.6f + particle.phase));
            float theta = particle.theta + swirlIntensity * z * 0.18f;
            Vector3 position{lane * std::cos(theta), lane * std::sin(theta), z};
            Color c = TubeColor(z, nearPalette, farPalette, pulseValue);
            c = LerpColor(c, Color{255, 255, 255, 255}, particle.glow * 0.25f + centerFactor * 0.22f);
            DrawSphere(position, 0.03f + 0.045f * particle.glow + 0.02f * centerFactor, Fade(c, 0.16f + 0.18f * particle.glow));
        }

        for (const EnergyStreak& streak : energyStreaks) {
            Color c = TubeColor(streak.z, nearPalette, farPalette, pulseValue);
            c = LerpColor(c, Color{255, 255, 255, 255}, 0.45f);
            float fade = 1.0f - std::clamp(streak.age / streak.ttl, 0.0f, 1.0f);
            Vector3 position{
                streak.radius * std::cos(streak.theta),
                streak.radius * std::sin(streak.theta),
                streak.z,
            };
            Vector3 tail = {
                position.x * 0.85f,
                position.y * 0.85f,
                position.z - streak.direction * 0.85f,
            };
            DrawLine3D(tail, position, Fade(c, 0.35f * fade));
            DrawSphere(position, streak.width, Fade(c, 0.32f * fade));
        }

        EndMode3D();

        Vector2 leftMouthScreen = GetWorldToScreen({0.0f, 0.0f, entranceMouthZ}, camera);
        Vector2 rightMouthScreen = GetWorldToScreen({0.0f, 0.0f, exitMouthZ}, camera);
        DrawCircleGradient(static_cast<int>(leftMouthScreen.x), static_cast<int>(leftMouthScreen.y),
                           140.0f + entrancePulse * 55.0f,
                           Fade(nearPalette.mouth, 0.10f + entrancePulse * 0.05f),
                           Fade(Color{0, 0, 0, 0}, 0.0f));
        DrawCircleGradient(static_cast<int>(rightMouthScreen.x), static_cast<int>(rightMouthScreen.y),
                           140.0f + exitPulse * 55.0f,
                           Fade(farPalette.mouth, 0.10f + exitPulse * 0.05f),
                           Fade(Color{0, 0, 0, 0}, 0.0f));

        DrawRectangle(16, 16, 760, 118, Fade(Color{8, 14, 28, 255}, 0.78f));
        DrawText("Wormhole Gateway", 30, 26, 34, Color{236, 240, 248, 255});
        DrawText("Mouse orbit | wheel zoom | 1-3 presets | 4/T transit | Up/Down throat | [ ] flare | Left/Right swirl",
                 30, 62, 18, Color{166, 186, 210, 255});
        DrawText("-/= transit speed | D increase distortion | S decrease distortion | M switch destination | P pause | R reset",
                 30, 84, 18, Color{166, 186, 210, 255});

        DrawRectangle(GetScreenWidth() - 350, 20, 318, 190, Fade(Color{8, 14, 28, 255}, 0.80f));
        DrawText("Gateway Status", GetScreenWidth() - 328, 30, 26, Color{236, 240, 248, 255});
        char status[420];
        std::snprintf(status, sizeof(status),
                      "Near field: %s\nFar field: %s\nThroat radius: %.2f\nFlare: %.2f\nSwirl: %.2f\nTransit speed: %.2f\nDistortion: %.2f\nMode: %s\nObserver side: %s%s",
                      nearPalette.name, farPalette.name, throatRadius, flare, swirlIntensity, transitSpeed, distortion,
                      transit.active ? "Transit" : "Orbit",
                      currentSide < 0 ? "Near field" : "Far field",
                      paused ? "\n[PAUSED]" : "");
        DrawText(status, GetScreenWidth() - 328, 68, 20, Color{124, 228, 255, 255});

        DrawRectangle(GetScreenWidth() - 350, 226, 318, 96, Fade(Color{8, 14, 28, 255}, 0.74f));
        DrawText("Pulse Echo", GetScreenWidth() - 328, 236, 22, Color{236, 240, 248, 255});
        DrawRectangle(GetScreenWidth() - 324, 272, 280, 10, Fade(Color{40, 56, 84, 255}, 0.95f));
        DrawRectangle(GetScreenWidth() - 324, 272, static_cast<int>(280.0f * std::clamp(entrancePulse / 1.2f, 0.0f, 1.0f)), 10, nearPalette.accent);
        DrawRectangle(GetScreenWidth() - 324, 290, static_cast<int>(280.0f * std::clamp(exitPulse / 1.1f, 0.0f, 1.0f)), 10, farPalette.accent);
        DrawText("entrance pulse", GetScreenWidth() - 324, 256, 16, Color{188, 204, 224, 255});
        DrawText("exit echo", GetScreenWidth() - 324, 304, 16, Color{188, 204, 224, 255});

        DrawCircleGradient(GetScreenWidth() / 2, GetScreenHeight() / 2, 300.0f,
                           Fade(LerpColor(nearPalette.fog, farPalette.fog, 0.5f), 0.035f),
                           Fade(Color{0, 0, 0, 0}, 0.0f));
        DrawFPS(30, GetScreenHeight() - 42);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
