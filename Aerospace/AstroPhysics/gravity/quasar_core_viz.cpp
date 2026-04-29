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

struct DiskParticle {
    float radiusBase;
    float radialNoise;
    float theta;
    float omega;
    float yBase;
    float heat;
    float alpha;
    float size;
    float streak;
    float phase;
    int band;
};

struct CoronaParticle {
    float radius;
    float theta;
    float omega;
    float height;
    float pulse;
    float size;
};

struct JetPacket {
    float axial;
    float radial;
    float theta;
    float speed;
    float age;
    float ttl;
    float width;
    float brightness;
    int direction;
};

struct Star {
    Vector3 position;
    float size;
    Color color;
};

struct CameraPreset {
    float yaw;
    float pitch;
    float distance;
    Vector3 target;
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

Color DiskHeatColor(float heat) {
    if (heat > 0.82f) {
        return LerpColor(Color{160, 214, 255, 255}, Color{255, 249, 228, 255}, (heat - 0.82f) / 0.18f);
    }
    if (heat > 0.52f) {
        return LerpColor(Color{255, 188, 92, 255}, Color{255, 244, 212, 255}, (heat - 0.52f) / 0.30f);
    }
    return LerpColor(Color{214, 72, 30, 255}, Color{255, 190, 92, 255}, heat / 0.52f);
}

Vector3 RotateX(Vector3 p, float angle) {
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    return {p.x, p.y * c - p.z * s, p.y * s + p.z * c};
}

Vector3 RotateZ(Vector3 p, float angle) {
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    return {p.x * c - p.y * s, p.x * s + p.y * c, p.z};
}

void UpdateOrbitCamera(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 delta = GetMouseDelta();
        *yaw -= delta.x * 0.0034f;
        *pitch += delta.y * 0.0032f;
        *pitch = std::clamp(*pitch, -1.42f, 1.42f);
    }

    *distance -= GetMouseWheelMove() * 0.95f;
    *distance = std::clamp(*distance, 5.0f, 75.0f);

    const float cp = std::cos(*pitch);
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
    const float cp = std::cos(*pitch);
    camera->position = Vector3Add(camera->target, {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    });
}

Vector3 TorusPoint(float u, float v, float majorRadius, float minorRadius) {
    Vector3 p{
        (majorRadius + minorRadius * std::cos(v)) * std::cos(u),
        minorRadius * std::sin(v),
        (majorRadius + minorRadius * std::cos(v)) * std::sin(u),
    };
    p = RotateX(p, 0.92f);
    p = RotateZ(p, 0.16f);
    return p;
}

void InitializeDisk(std::vector<DiskParticle>* particles) {
    particles->clear();
    const std::array<int, 5> counts = {170, 220, 250, 230, 170};
    const std::array<std::array<float, 2>, 5> radii = {{
        {2.05f, 2.75f},
        {2.75f, 3.85f},
        {3.85f, 5.05f},
        {5.05f, 6.55f},
        {6.55f, 8.10f},
    }};

    for (int band = 0; band < static_cast<int>(counts.size()); ++band) {
        for (int i = 0; i < counts[band]; ++i) {
            const float radius = RandomFloat(radii[band][0], radii[band][1]);
            const float heat = std::clamp(1.0f - band * 0.18f + RandomFloat(-0.05f, 0.05f), 0.12f, 1.0f);
            const float omega = (1.8f - band * 0.18f) / std::pow(radius, 0.94f);
            particles->push_back(DiskParticle{
                radius,
                RandomFloat(0.06f, 0.60f),
                RandomFloat(0.0f, 2.0f * kPi),
                omega,
                RandomFloat(-0.7f, 0.7f),
                heat,
                RandomFloat(0.38f, 0.96f),
                RandomFloat(0.045f, 0.095f),
                RandomFloat(0.10f, 0.38f),
                RandomFloat(0.0f, 2.0f * kPi),
                band,
            });
        }
    }
}

void InitializeCorona(std::vector<CoronaParticle>* particles) {
    particles->clear();
    particles->reserve(160);
    for (int i = 0; i < 160; ++i) {
        particles->push_back(CoronaParticle{
            RandomFloat(1.6f, 3.4f),
            RandomFloat(0.0f, 2.0f * kPi),
            RandomFloat(0.7f, 2.6f),
            RandomFloat(-1.9f, 1.9f),
            RandomFloat(0.0f, 2.0f * kPi),
            RandomFloat(0.035f, 0.090f),
        });
    }
}

void InitializeStars(std::vector<Star>* stars) {
    stars->clear();
    stars->reserve(220);
    for (int i = 0; i < 220; ++i) {
        float theta = RandomFloat(0.0f, 2.0f * kPi);
        float phi = RandomFloat(0.25f, kPi - 0.25f);
        float radius = RandomFloat(30.0f, 80.0f);
        Vector3 pos{
            radius * std::sin(phi) * std::cos(theta),
            radius * std::cos(phi) * RandomFloat(0.4f, 1.1f),
            radius * std::sin(phi) * std::sin(theta),
        };
        float tint = RandomFloat(0.0f, 1.0f);
        stars->push_back(Star{
            pos,
            RandomFloat(0.05f, 0.18f),
            LerpColor(Color{150, 176, 255, 255}, Color{255, 236, 210, 255}, tint),
        });
    }
}

void SpawnJetPacket(std::vector<JetPacket>* packets, int direction, float jetPower, float flareStrength) {
    packets->push_back(JetPacket{
        RandomFloat(1.35f, 1.85f),
        RandomFloat(0.04f, 0.26f + 0.12f * jetPower),
        RandomFloat(0.0f, 2.0f * kPi),
        RandomFloat(4.0f, 8.2f) + jetPower * 1.6f + flareStrength * 2.0f,
        0.0f,
        RandomFloat(2.0f, 3.6f),
        RandomFloat(0.07f, 0.17f),
        RandomFloat(0.55f, 1.0f),
        direction,
    });
}

void DrawRing(float radius, float y, int segments, Color color, float wobble, float time) {
    for (int i = 0; i < segments; ++i) {
        float a0 = (2.0f * kPi * i) / static_cast<float>(segments);
        float a1 = (2.0f * kPi * (i + 1)) / static_cast<float>(segments);
        float r0 = radius + wobble * std::sin(time * 2.8f + a0 * 5.0f);
        float r1 = radius + wobble * std::sin(time * 2.8f + a1 * 5.0f);
        Vector3 p0{r0 * std::cos(a0), y, r0 * std::sin(a0)};
        Vector3 p1{r1 * std::cos(a1), y, r1 * std::sin(a1)};
        DrawLine3D(p0, p1, color);
    }
}

void DrawTorus(float majorRadius, float minorRadius, float opacity, float time) {
    const int majorSegments = 72;
    const int minorSegments = 24;
    for (int i = 0; i < majorSegments; ++i) {
        float u0 = (2.0f * kPi * i) / static_cast<float>(majorSegments);
        float u1 = (2.0f * kPi * (i + 1)) / static_cast<float>(majorSegments);
        for (int j = 0; j < minorSegments; ++j) {
            float v0 = (2.0f * kPi * j) / static_cast<float>(minorSegments);
            float v1 = (2.0f * kPi * (j + 1)) / static_cast<float>(minorSegments);
            Vector3 p00 = TorusPoint(u0, v0, majorRadius, minorRadius);
            Vector3 p10 = TorusPoint(u1, v0, majorRadius, minorRadius);
            Vector3 p01 = TorusPoint(u0, v1, majorRadius, minorRadius);
            float rim = 0.5f + 0.5f * std::sin(v0 + time * 0.8f);
            Color c = Color{
                static_cast<unsigned char>(80 + 120 * rim),
                static_cast<unsigned char>(42 + 65 * rim),
                static_cast<unsigned char>(20 + 25 * rim),
                static_cast<unsigned char>(20 + opacity * (45 + 55 * rim)),
            };
            DrawLine3D(p00, p10, c);
            DrawLine3D(p00, p01, Fade(c, 0.85f));
        }
    }
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Quasar Core 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {15.0f, 7.2f, 15.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 42.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    std::array<CameraPreset, 4> presets = {{
        {0.82f, 0.33f, 20.0f, {0.0f, 0.0f, 0.0f}},
        {1.58f, 0.04f, 17.5f, {0.0f, 0.1f, 0.0f}},
        {0.24f, 0.64f, 24.0f, {0.0f, 0.5f, 0.0f}},
        {1.55f, 1.02f, 23.0f, {0.0f, 6.0f, 0.0f}},
    }};

    float camYaw = presets[0].yaw;
    float camPitch = presets[0].pitch;
    float camDistance = presets[0].distance;

    float blackHoleMass = 280.0f;
    float jetPower = 1.25f;
    float diskThickness = 0.72f;
    float torusOpacity = 0.65f;
    float beamingScale = 1.45f;
    bool paused = false;

    float time = 0.0f;
    float flareTimer = 0.0f;
    float flareStrength = 0.0f;
    float jetSpawnAccumulator = 0.0f;

    std::vector<DiskParticle> diskParticles;
    std::vector<CoronaParticle> coronaParticles;
    std::vector<JetPacket> jetPackets;
    std::vector<Star> stars;
    InitializeDisk(&diskParticles);
    InitializeCorona(&coronaParticles);
    InitializeStars(&stars);
    jetPackets.reserve(540);

    ApplyPreset(presets[0], &camera, &camYaw, &camPitch, &camDistance);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_ONE)) ApplyPreset(presets[0], &camera, &camYaw, &camPitch, &camDistance);
        if (IsKeyPressed(KEY_TWO)) ApplyPreset(presets[1], &camera, &camYaw, &camPitch, &camDistance);
        if (IsKeyPressed(KEY_THREE)) ApplyPreset(presets[2], &camera, &camYaw, &camPitch, &camDistance);
        if (IsKeyPressed(KEY_FOUR)) ApplyPreset(presets[3], &camera, &camYaw, &camPitch, &camDistance);

        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_B)) beamingScale = (beamingScale > 1.46f) ? 1.0f : 2.25f;
        if (IsKeyPressed(KEY_F)) flareTimer = 1.65f;
        if (IsKeyPressed(KEY_R)) {
            blackHoleMass = 280.0f;
            jetPower = 1.25f;
            diskThickness = 0.72f;
            torusOpacity = 0.65f;
            beamingScale = 1.45f;
            paused = false;
            time = 0.0f;
            flareTimer = 0.0f;
            flareStrength = 0.0f;
            jetSpawnAccumulator = 0.0f;
            InitializeDisk(&diskParticles);
            InitializeCorona(&coronaParticles);
            InitializeStars(&stars);
            jetPackets.clear();
            ApplyPreset(presets[0], &camera, &camYaw, &camPitch, &camDistance);
        }

        if (IsKeyDown(KEY_UP)) blackHoleMass = std::min(520.0f, blackHoleMass + 110.0f * GetFrameTime());
        if (IsKeyDown(KEY_DOWN)) blackHoleMass = std::max(120.0f, blackHoleMass - 110.0f * GetFrameTime());
        if (IsKeyDown(KEY_RIGHT)) jetPower = std::min(3.2f, jetPower + 1.0f * GetFrameTime());
        if (IsKeyDown(KEY_LEFT)) jetPower = std::max(0.2f, jetPower - 1.0f * GetFrameTime());
        if (IsKeyDown(KEY_RIGHT_BRACKET)) diskThickness = std::min(1.45f, diskThickness + 0.55f * GetFrameTime());
        if (IsKeyDown(KEY_LEFT_BRACKET)) diskThickness = std::max(0.18f, diskThickness - 0.55f * GetFrameTime());
        if (IsKeyDown(KEY_EQUAL)) torusOpacity = std::min(1.0f, torusOpacity + 0.7f * GetFrameTime());
        if (IsKeyDown(KEY_MINUS)) torusOpacity = std::max(0.05f, torusOpacity - 0.7f * GetFrameTime());

        UpdateOrbitCamera(&camera, &camYaw, &camPitch, &camDistance);

        const float dt = GetFrameTime();
        if (!paused) {
            time += dt;
            flareTimer = std::max(0.0f, flareTimer - dt);
            flareStrength = std::sin((flareTimer / 1.65f) * kPi);
            flareStrength = std::max(flareStrength, 0.0f);

            const float massSpinScale = 0.72f + blackHoleMass / 260.0f;
            for (DiskParticle& particle : diskParticles) {
                float innerBoost = 1.0f + (1.0f - particle.radiusBase / 8.2f) * (0.35f + flareStrength * 0.65f);
                particle.theta += particle.omega * massSpinScale * innerBoost * dt;
            }
            for (CoronaParticle& particle : coronaParticles) {
                particle.theta += particle.omega * (1.0f + flareStrength * 0.5f) * dt;
            }

            jetSpawnAccumulator += dt * (18.0f + jetPower * 22.0f + flareStrength * 18.0f);
            while (jetSpawnAccumulator >= 1.0f) {
                jetSpawnAccumulator -= 1.0f;
                SpawnJetPacket(&jetPackets, 1, jetPower, flareStrength);
                SpawnJetPacket(&jetPackets, -1, jetPower, flareStrength);
            }

            for (JetPacket& packet : jetPackets) {
                packet.age += dt;
                packet.axial += packet.speed * dt * (1.0f + 0.25f * jetPower);
                packet.theta += dt * (1.0f + 0.8f * jetPower);
                packet.radial *= 0.992f;
            }
            jetPackets.erase(
                std::remove_if(jetPackets.begin(), jetPackets.end(), [](const JetPacket& packet) {
                    return packet.age > packet.ttl || packet.axial > 24.0f;
                }),
                jetPackets.end());
        }

        const float horizonRadius = 1.0f + (blackHoleMass - 120.0f) / 500.0f;
        const float photonRingRadius = horizonRadius * 1.85f;
        const float diskGlowRadius = 7.8f + flareStrength * 0.8f;
        const Vector3 observerDirection = Vector3Normalize(Vector3Subtract(camera.position, camera.target));

        BeginDrawing();
        ClearBackground(Color{4, 6, 12, 255});

        BeginMode3D(camera);
        DrawGrid(24, 1.4f);

        for (const Star& star : stars) {
            DrawSphere(star.position, star.size, Fade(star.color, 0.90f));
        }

        for (int i = 0; i < 8; ++i) {
            float radius = 10.0f + i * 0.95f;
            float alpha = 0.07f - i * 0.006f;
            DrawRing(radius, -0.04f * i, 84, Fade(Color{90, 130, 185, 255}, alpha), 0.04f, time * 0.2f + i);
        }

        DrawTorus(7.9f, 1.45f + diskThickness * 0.25f, torusOpacity, time);

        for (const DiskParticle& particle : diskParticles) {
            float bandFactor = 1.0f - particle.band / 5.0f;
            float radius = particle.radiusBase +
                           std::sin(time * (1.2f + 0.20f * particle.band) + particle.phase) * particle.radialNoise * 0.12f;
            float y = particle.yBase * 0.22f * diskThickness +
                      std::sin(time * 2.8f + particle.phase * 1.7f) * 0.05f * diskThickness * (0.2f + bandFactor);
            Vector3 position{radius * std::cos(particle.theta), y, radius * std::sin(particle.theta)};

            Vector3 tangent{-std::sin(particle.theta), 0.12f * y, std::cos(particle.theta)};
            tangent = Vector3Normalize(tangent);
            float viewBoost = std::clamp((Vector3DotProduct(tangent, observerDirection) + 1.0f) * 0.5f, 0.0f, 1.0f);
            float beaming = 0.65f + std::pow(viewBoost, 1.0f + beamingScale) * (0.8f + jetPower * 0.25f);

            Color baseColor = DiskHeatColor(std::clamp(particle.heat + flareStrength * bandFactor * 0.20f, 0.0f, 1.0f));
            Color drawColor = Fade(baseColor, std::clamp(particle.alpha * beaming, 0.0f, 1.0f));

            float streakLength = particle.streak * (1.0f + bandFactor * 0.7f + flareStrength * bandFactor);
            Vector3 tail = Vector3Subtract(position, Vector3Scale(tangent, streakLength));
            DrawLine3D(tail, position, Fade(drawColor, 0.55f));
            DrawSphere(position, particle.size * (0.9f + 0.8f * bandFactor + flareStrength * 0.4f), drawColor);
        }

        for (const CoronaParticle& particle : coronaParticles) {
            float lift = particle.height + std::sin(time * 1.8f + particle.pulse) * 0.25f;
            Vector3 position{
                particle.radius * std::cos(particle.theta),
                lift,
                particle.radius * std::sin(particle.theta),
            };
            float pulse = 0.55f + 0.45f * std::sin(time * 4.0f + particle.pulse);
            Color coronaColor = Fade(Color{185, 226, 255, 255}, 0.10f + pulse * 0.24f + flareStrength * 0.12f);
            DrawSphere(position, particle.size * (1.0f + flareStrength * 0.8f), coronaColor);
        }

        for (int side = -1; side <= 1; side += 2) {
            float spineLength = 18.0f + jetPower * 4.0f;
            Color spineColor = (side > 0) ? Color{120, 220, 255, 120} : Color{92, 178, 250, 96};
            Color plumeColor = (side > 0) ? Color{80, 186, 255, 40} : Color{70, 160, 230, 32};
            DrawCylinderEx({0.0f, horizonRadius * side, 0.0f}, {0.0f, spineLength * side, 0.0f},
                           0.24f + flareStrength * 0.08f, 0.08f, 16, spineColor);
            DrawCylinderEx({0.0f, horizonRadius * side, 0.0f}, {0.0f, (10.0f + jetPower * 4.0f) * side, 0.0f},
                           0.95f + jetPower * 0.25f, 0.22f, 20, plumeColor);
        }

        for (const JetPacket& packet : jetPackets) {
            float travel = packet.axial;
            Vector3 position{
                packet.radial * std::cos(packet.theta),
                packet.direction * travel,
                packet.radial * std::sin(packet.theta),
            };
            float ageT = std::clamp(packet.age / packet.ttl, 0.0f, 1.0f);
            float beaming = (packet.direction > 0)
                                ? (0.7f + 0.6f * std::pow(std::max(0.0f, Vector3DotProduct(observerDirection, {0.0f, 1.0f, 0.0f})), 2.0f))
                                : 0.85f;
            Color packetColor = LerpColor(Color{110, 200, 255, 255}, Color{255, 255, 255, 255}, packet.brightness);
            packetColor = Fade(packetColor, (1.0f - ageT) * (0.25f + packet.brightness * 0.65f) * beaming);
            Vector3 tail = {position.x * 0.80f, position.y - packet.direction * 0.55f, position.z * 0.80f};
            DrawLine3D(tail, position, Fade(packetColor, 0.55f));
            DrawSphere(position, packet.width * (1.0f + 0.5f * packet.brightness), packetColor);
        }

        DrawSphere({0.0f, 0.0f, 0.0f}, horizonRadius, Color{6, 6, 8, 255});
        DrawSphereWires({0.0f, 0.0f, 0.0f}, horizonRadius * 1.08f, 18, 18, Fade(Color{90, 116, 180, 255}, 0.16f));
        DrawRing(photonRingRadius, 0.0f, 120, Fade(Color{255, 230, 176, 255}, 0.55f + flareStrength * 0.25f), 0.04f, time);
        DrawRing(photonRingRadius * 1.12f, 0.03f, 96, Fade(Color{255, 144, 76, 255}, 0.35f), 0.07f, time * 1.2f);
        DrawRing(photonRingRadius * 0.92f, -0.02f, 96, Fade(Color{196, 220, 255, 255}, 0.22f + flareStrength * 0.15f), 0.03f, time * 1.5f);

        EndMode3D();

        Vector2 coreScreen = GetWorldToScreen({0.0f, 0.0f, 0.0f}, camera);
        if (coreScreen.x > -200.0f && coreScreen.x < static_cast<float>(GetScreenWidth()) + 200.0f &&
            coreScreen.y > -200.0f && coreScreen.y < static_cast<float>(GetScreenHeight()) + 200.0f) {
            DrawCircleGradient(static_cast<int>(coreScreen.x), static_cast<int>(coreScreen.y),
                               220.0f + flareStrength * 65.0f,
                               Fade(Color{255, 180, 74, 255}, 0.12f + flareStrength * 0.08f),
                               Fade(Color{0, 0, 0, 0}, 0.0f));
            DrawCircleGradient(static_cast<int>(coreScreen.x), static_cast<int>(coreScreen.y),
                               130.0f + flareStrength * 38.0f,
                               Fade(Color{255, 248, 220, 255}, 0.08f + flareStrength * 0.08f),
                               Fade(Color{0, 0, 0, 0}, 0.0f));
        }

        DrawRectangle(16, 16, 600, 114, Fade(Color{10, 16, 28, 255}, 0.78f));
        DrawText("Quasar Core Observatory", 30, 26, 34, Color{236, 240, 248, 255});
        DrawText("Mouse orbit | wheel zoom | 1-4 presets | Up/Down mass | Left/Right jet power | [ ] disk thickness",
                 30, 62, 18, Color{162, 182, 208, 255});
        DrawText("-/= torus opacity | F flare burst | B beaming boost | P pause | R reset",
                 30, 84, 18, Color{162, 182, 208, 255});

        DrawRectangle(GetScreenWidth() - 332, 20, 300, 156, Fade(Color{10, 16, 28, 255}, 0.80f));
        DrawText("Active Nucleus", GetScreenWidth() - 312, 30, 26, Color{236, 240, 248, 255});

        char status[320];
        std::snprintf(status, sizeof(status),
                      "BH mass: %.0f\nJet power: %.2f\nDisk thickness: %.2f\nTorus opacity: %.2f\nBeaming scale: %.2f\nFlare: %.2f%s",
                      blackHoleMass, jetPower, diskThickness, torusOpacity, beamingScale, flareStrength,
                      paused ? "\n[PAUSED]" : "");
        DrawText(status, GetScreenWidth() - 312, 66, 20, Color{124, 228, 255, 255});

        DrawRectangle(GetScreenWidth() - 332, 194, 300, 110, Fade(Color{10, 16, 28, 255}, 0.72f));
        DrawText("Disk Temperature Gradient", GetScreenWidth() - 312, 204, 22, Color{236, 240, 248, 255});
        for (int i = 0; i < 220; ++i) {
            float t = i / 219.0f;
            Color c = DiskHeatColor(1.0f - t);
            DrawRectangle(GetScreenWidth() - 302 + i, 242, 1, 18, c);
        }
        DrawText("hot inner disk", GetScreenWidth() - 302, 266, 18, Color{200, 218, 236, 255});
        DrawText("cool dusty edge", GetScreenWidth() - 178, 266, 18, Color{200, 218, 236, 255});

        DrawRectangle(0, GetScreenHeight() - 110, GetScreenWidth(), 110, Fade(Color{4, 4, 8, 255}, 0.16f));
        DrawCircleGradient(GetScreenWidth() / 2, GetScreenHeight() / 2,
                           diskGlowRadius * 18.0f,
                           Fade(Color{255, 168, 92, 255}, 0.03f + flareStrength * 0.03f),
                           Fade(Color{0, 0, 0, 0}, 0.0f));
        DrawFPS(30, GetScreenHeight() - 42);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
