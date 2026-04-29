#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kScreenWidth = 1320;
constexpr int kScreenHeight = 860;

constexpr int kGridX = 70;
constexpr int kGridZ = 70;
constexpr float kXMin = -8.0f;
constexpr float kXMax = 8.0f;
constexpr float kZMin = -8.0f;
constexpr float kZMax = 8.0f;
constexpr float kFieldHeightScale = 0.42f;

struct HelicalPacket {
    Vector2 pos;
    Vector2 vel;
    float amplitude;
    float sigma;
    float helixRadius;
    float turnsPerUnit;
    float omega;
    float phase;
    bool rightHanded;
    Color color;
    float age;
};

struct MergeFlash {
    Vector3 pos;
    Color color;
    float radius;
    float life;
};

void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        const Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.15f, 1.25f);
    }

    *distance -= GetMouseWheelMove() * 0.75f;
    *distance = std::clamp(*distance, 6.0f, 28.0f);

    const float cp = std::cos(*pitch);
    const Vector3 offset = {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}

float Saturate(float x) {
    return std::clamp(x, 0.0f, 1.0f);
}

Color WithAlpha(Color c, unsigned char alpha) {
    c.a = alpha;
    return c;
}

Color MixColor(Color a, Color b, float t) {
    t = Saturate(t);
    return Color{
        static_cast<unsigned char>(a.r + (b.r - a.r) * t),
        static_cast<unsigned char>(a.g + (b.g - a.g) * t),
        static_cast<unsigned char>(a.b + (b.b - a.b) * t),
        static_cast<unsigned char>(a.a + (b.a - a.a) * t),
    };
}

Vector2 PropagationDir2(const HelicalPacket& packet) {
    if (Vector2Length(packet.vel) < 1e-4f) return {1.0f, 0.0f};
    return Vector2Normalize(packet.vel);
}

Vector3 PropagationDir3(const HelicalPacket& packet) {
    const Vector2 d = PropagationDir2(packet);
    return {d.x, 0.0f, d.y};
}

Vector3 HelixBasisA(const HelicalPacket& packet) {
    Vector3 up = {0.0f, 1.0f, 0.0f};
    Vector3 dir = PropagationDir3(packet);
    Vector3 side = Vector3CrossProduct(up, dir);
    if (Vector3Length(side) < 1e-4f) side = {1.0f, 0.0f, 0.0f};
    return Vector3Normalize(side);
}

Vector3 HelixBasisB(const HelicalPacket& packet) {
    Vector3 dir = PropagationDir3(packet);
    Vector3 a = HelixBasisA(packet);
    return Vector3Normalize(Vector3CrossProduct(dir, a));
}

float PacketEnvelope(const HelicalPacket& packet, float x, float z) {
    const float dx = x - packet.pos.x;
    const float dz = z - packet.pos.y;
    const float r2 = dx * dx + dz * dz;
    return std::exp(-r2 / (2.0f * packet.sigma * packet.sigma));
}

float PacketContribution(const HelicalPacket& packet, float x, float z, float time) {
    const Vector2 rel = {x - packet.pos.x, z - packet.pos.y};
    const float proj = Vector2DotProduct(rel, PropagationDir2(packet));
    const float envelope = PacketEnvelope(packet, x, z);
    const float handed = packet.rightHanded ? 1.0f : -1.0f;
    const float phase = 2.0f * PI * packet.turnsPerUnit * proj - handed * packet.omega * time + packet.phase;
    return packet.amplitude * envelope * std::cos(phase);
}

float SampleField(const std::vector<HelicalPacket>& packets, float x, float z, float time) {
    float value = 0.0f;
    for (const HelicalPacket& packet : packets) value += PacketContribution(packet, x, z, time);
    return value;
}

Vector3 GridPoint(int ix, int iz, float value) {
    const float x = kXMin + (kXMax - kXMin) * static_cast<float>(ix) / static_cast<float>(kGridX - 1);
    const float z = kZMin + (kZMax - kZMin) * static_cast<float>(iz) / static_cast<float>(kGridZ - 1);
    return {x, kFieldHeightScale * value, z};
}

HelicalPacket MakePacket(Vector2 pos,
                         Vector2 vel,
                         float amplitude,
                         float sigma,
                         float helixRadius,
                         float turnsPerUnit,
                         float omega,
                         float phase,
                         bool rightHanded,
                         Color color) {
    return {pos, vel, amplitude, sigma, helixRadius, turnsPerUnit, omega, phase, rightHanded, color, 0.0f};
}

Vector3 HelixPoint(const HelicalPacket& packet, float localS, float time) {
    const Vector3 center = {packet.pos.x, 0.32f, packet.pos.y};
    const Vector3 dir = PropagationDir3(packet);
    const Vector3 a = HelixBasisA(packet);
    const Vector3 b = HelixBasisB(packet);
    const float envelope = std::exp(-(localS * localS) / (2.0f * packet.sigma * packet.sigma * 0.09f));
    const float radius = packet.helixRadius * envelope;
    const float handed = packet.rightHanded ? 1.0f : -1.0f;
    const float angle = handed * (2.0f * PI * packet.turnsPerUnit * localS - packet.omega * time) + packet.phase;

    Vector3 point = Vector3Add(center, Vector3Scale(dir, localS));
    point = Vector3Add(point, Vector3Scale(a, radius * std::cos(angle)));
    point = Vector3Add(point, Vector3Scale(b, radius * std::sin(angle)));
    return point;
}

void DrawHelicalPacket(const HelicalPacket& packet, float time) {
    const float span = 3.2f * packet.sigma;
    const int segments = 72;
    const Vector3 center = {packet.pos.x, 0.32f, packet.pos.y};
    const Vector3 dir = PropagationDir3(packet);

    Vector3 prevAxis = Vector3Add(center, Vector3Scale(dir, -span));
    Vector3 prev = HelixPoint(packet, -span, time);
    for (int i = 1; i <= segments; ++i) {
        const float s = -span + 2.0f * span * static_cast<float>(i) / static_cast<float>(segments);
        const Vector3 axis = Vector3Add(center, Vector3Scale(dir, s));
        const Vector3 point = HelixPoint(packet, s, time);
        DrawLine3D(prevAxis, axis, WithAlpha(packet.color, 34));
        DrawLine3D(prev, point, packet.color);
        prevAxis = axis;
        prev = point;
    }

    DrawSphere(center, 0.12f, packet.color);
    DrawSphere(center, 0.22f, WithAlpha(packet.color, 30));
}

std::string HudText(int waveCount,
                    int mergeCount,
                    bool paused,
                    bool autoSpawn,
                    float fieldGain,
                    bool rightHanded) {
    std::ostringstream os;
    os.setf(std::ios::fixed);
    os.precision(2);
    os << "waves=" << waveCount
       << "  merges=" << mergeCount
       << "  gain=" << fieldGain
       << "  hand=" << (rightHanded ? "right" : "left")
       << "  auto=" << (autoSpawn ? "on" : "off");
    if (paused) os << "  [PAUSED]";
    return os.str();
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Helical Probability Wave Field 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {11.4f, 7.4f, 11.2f};
    camera.target = {0.0f, 0.5f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 42.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.82f;
    float camPitch = 0.34f;
    float camDistance = 16.6f;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> sideOffset(-2.8f, 2.8f);
    std::uniform_real_distribution<float> sigmaDist(0.85f, 1.35f);
    std::uniform_real_distribution<float> phaseDist(0.0f, 2.0f * PI);
    std::uniform_real_distribution<float> ampDist(0.75f, 1.15f);
    std::uniform_real_distribution<float> speedDist(1.05f, 1.95f);
    std::uniform_real_distribution<float> radiusDist(0.24f, 0.42f);

    std::vector<HelicalPacket> packets;
    std::vector<MergeFlash> flashes;
    int mergeCount = 0;
    bool paused = false;
    bool autoSpawn = false;
    bool nextRightHanded = true;
    float autoSpawnTimer = 0.0f;
    float time = 0.0f;
    float fieldGain = 1.0f;

    auto spawnEdgePacket = [&](int side) {
        const float sigma = sigmaDist(rng);
        const float amplitude = ampDist(rng);
        const float speed = speedDist(rng);
        const float radius = radiusDist(rng);
        const float phase = phaseDist(rng);
        const float turns = 0.75f + 0.22f * speed;
        const float omega = 4.4f + 1.0f * speed;

        if (side == 0) {
            packets.push_back(MakePacket({kXMin + 0.4f, sideOffset(rng)}, {speed, 0.0f}, amplitude, sigma, radius, turns, omega, phase, nextRightHanded, Color{90, 210, 255, 255}));
        } else if (side == 1) {
            packets.push_back(MakePacket({kXMax - 0.4f, sideOffset(rng)}, {-speed, 0.0f}, amplitude, sigma, radius, turns, omega, phase, nextRightHanded, Color{255, 150, 118, 255}));
        } else if (side == 2) {
            packets.push_back(MakePacket({sideOffset(rng), kZMin + 0.4f}, {0.0f, speed}, amplitude, sigma, radius, turns, omega, phase, nextRightHanded, Color{120, 255, 175, 255}));
        } else {
            packets.push_back(MakePacket({sideOffset(rng), kZMax - 0.4f}, {0.0f, -speed}, amplitude, sigma, radius, turns, omega, phase, nextRightHanded, Color{220, 170, 255, 255}));
        }
    };

    auto spawnCollisionPair = [&]() {
        packets.push_back(MakePacket({kXMin + 0.55f, -0.8f}, {1.8f, 0.0f}, 1.0f, 0.95f, 0.36f, 1.18f, 6.0f, 0.2f, true, Color{100, 220, 255, 255}));
        packets.push_back(MakePacket({kXMax - 0.55f, 0.8f}, {-1.7f, 0.0f}, 0.94f, 1.05f, 0.31f, 1.02f, 5.5f, 1.5f, false, Color{255, 164, 120, 255}));
    };

    packets.push_back(MakePacket({-4.5f, -1.3f}, {1.28f, 0.22f}, 1.0f, 1.0f, 0.34f, 1.08f, 5.8f, 0.3f, true, Color{90, 210, 255, 255}));
    packets.push_back(MakePacket({4.6f, 1.4f}, {-1.33f, -0.18f}, 0.92f, 1.1f, 0.30f, 0.96f, 5.4f, 1.5f, false, Color{255, 150, 118, 255}));

    std::vector<float> field(kGridX * kGridZ, 0.0f);
    std::vector<float> density(kGridX * kGridZ, 0.0f);
    std::vector<Color> tint(kGridX * kGridZ, Color{100, 155, 230, 180});

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_A)) autoSpawn = !autoSpawn;
        if (IsKeyPressed(KEY_H)) nextRightHanded = !nextRightHanded;
        if (IsKeyPressed(KEY_R)) {
            packets.clear();
            flashes.clear();
            mergeCount = 0;
            paused = false;
            autoSpawn = false;
            autoSpawnTimer = 0.0f;
            time = 0.0f;
            fieldGain = 1.0f;
            nextRightHanded = true;
            packets.push_back(MakePacket({-4.5f, -1.3f}, {1.28f, 0.22f}, 1.0f, 1.0f, 0.34f, 1.08f, 5.8f, 0.3f, true, Color{90, 210, 255, 255}));
            packets.push_back(MakePacket({4.6f, 1.4f}, {-1.33f, -0.18f}, 0.92f, 1.1f, 0.30f, 0.96f, 5.4f, 1.5f, false, Color{255, 150, 118, 255}));
        }

        if (IsKeyPressed(KEY_ONE)) spawnEdgePacket(0);
        if (IsKeyPressed(KEY_TWO)) spawnEdgePacket(1);
        if (IsKeyPressed(KEY_THREE)) spawnEdgePacket(2);
        if (IsKeyPressed(KEY_FOUR)) spawnEdgePacket(3);
        if (IsKeyPressed(KEY_SPACE)) spawnCollisionPair();

        if (IsKeyPressed(KEY_LEFT_BRACKET)) fieldGain = std::max(0.45f, fieldGain - 0.08f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) fieldGain = std::min(1.8f, fieldGain + 0.08f);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        if (!paused) {
            const float dt = GetFrameTime();
            time += dt;

            if (autoSpawn) {
                autoSpawnTimer += dt;
                if (autoSpawnTimer > 1.35f) {
                    autoSpawnTimer = 0.0f;
                    spawnEdgePacket(GetRandomValue(0, 3));
                }
            }

            for (HelicalPacket& packet : packets) {
                packet.pos = Vector2Add(packet.pos, Vector2Scale(packet.vel, dt));
                packet.age += dt;
            }

            for (MergeFlash& flash : flashes) {
                flash.life -= dt;
                flash.radius += dt * 0.7f;
            }
            flashes.erase(std::remove_if(flashes.begin(), flashes.end(),
                                         [](const MergeFlash& flash) { return flash.life <= 0.0f; }),
                          flashes.end());

            std::vector<HelicalPacket> next;
            std::vector<bool> consumed(packets.size(), false);

            for (std::size_t i = 0; i < packets.size(); ++i) {
                if (consumed[i]) continue;
                HelicalPacket merged = packets[i];

                for (std::size_t j = i + 1; j < packets.size(); ++j) {
                    if (consumed[j]) continue;

                    const float dist = Vector2Distance(merged.pos, packets[j].pos);
                    const float mergeRadius = 0.62f * (merged.sigma + packets[j].sigma);
                    if (dist > mergeRadius) continue;

                    const float w1 = std::max(0.05f, std::fabs(merged.amplitude));
                    const float w2 = std::max(0.05f, std::fabs(packets[j].amplitude));
                    const float sum = w1 + w2;

                    merged.pos = Vector2Scale(Vector2Add(Vector2Scale(merged.pos, w1), Vector2Scale(packets[j].pos, w2)), 1.0f / sum);
                    merged.vel = Vector2Scale(Vector2Add(Vector2Scale(merged.vel, w1), Vector2Scale(packets[j].vel, w2)), 0.92f / sum);
                    merged.sigma = std::clamp((merged.sigma * w1 + packets[j].sigma * w2) / sum * 1.06f, 0.7f, 2.2f);
                    merged.helixRadius = std::clamp((merged.helixRadius * w1 + packets[j].helixRadius * w2) / sum * 1.05f, 0.18f, 0.52f);
                    merged.turnsPerUnit = (merged.turnsPerUnit * w1 + packets[j].turnsPerUnit * w2) / sum;
                    merged.omega = (merged.omega * w1 + packets[j].omega * w2) / sum;
                    merged.phase = 0.5f * (merged.phase + packets[j].phase);
                    merged.amplitude = std::clamp(std::sqrt(merged.amplitude * merged.amplitude + packets[j].amplitude * packets[j].amplitude) * 0.82f, 0.22f, 2.2f);
                    merged.rightHanded = (w1 >= w2) ? merged.rightHanded : packets[j].rightHanded;
                    merged.color = MixColor(merged.color, packets[j].color, w2 / sum);
                    merged.age = 0.0f;

                    flashes.push_back({{merged.pos.x, 0.35f, merged.pos.y}, merged.color, 0.25f, 0.7f});
                    consumed[j] = true;
                    ++mergeCount;
                }

                if (std::fabs(merged.pos.x) <= 9.4f && std::fabs(merged.pos.y) <= 9.4f && merged.age < 12.0f) {
                    next.push_back(merged);
                }
            }

            packets = std::move(next);
        }

        for (int ix = 0; ix < kGridX; ++ix) {
            for (int iz = 0; iz < kGridZ; ++iz) {
                const float x = kXMin + (kXMax - kXMin) * static_cast<float>(ix) / static_cast<float>(kGridX - 1);
                const float z = kZMin + (kZMax - kZMin) * static_cast<float>(iz) / static_cast<float>(kGridZ - 1);
                float f = 0.0f;
                float d = 0.0f;
                Color mix = Color{100, 155, 230, 180};
                float blendWeight = 0.0f;

                for (const HelicalPacket& packet : packets) {
                    const float env = PacketEnvelope(packet, x, z);
                    f += PacketContribution(packet, x, z, time);
                    d += std::fabs(packet.amplitude) * env;
                    const float w = env * 0.8f;
                    mix = MixColor(mix, packet.color, Saturate(w / std::max(0.001f, blendWeight + w)));
                    blendWeight += w;
                }

                const int idx = ix * kGridZ + iz;
                field[idx] = fieldGain * f;
                density[idx] = d;
                tint[idx] = WithAlpha(mix, static_cast<unsigned char>(60.0f + 120.0f * Saturate(0.22f + d)));
            }
        }

        BeginDrawing();
        ClearBackground(Color{5, 8, 16, 255});

        DrawRectangleGradientV(0, 0, kScreenWidth, 140, Color{10, 18, 30, 220}, Color{10, 18, 30, 20});
        DrawRectangleGradientV(0, kScreenHeight - 120, kScreenWidth, 120, Color{8, 12, 20, 20}, Color{8, 12, 20, 210});

        BeginMode3D(camera);

        DrawPlane({0.0f, -0.03f, 0.0f}, {18.5f, 18.5f}, Color{14, 20, 30, 255});

        for (int i = -8; i <= 8; ++i) {
            const Color gridColor = (i == 0) ? Color{80, 130, 185, 110} : Color{40, 62, 88, 70};
            DrawLine3D({static_cast<float>(i), 0.0f, kZMin}, {static_cast<float>(i), 0.0f, kZMax}, gridColor);
            DrawLine3D({kXMin, 0.0f, static_cast<float>(i)}, {kXMax, 0.0f, static_cast<float>(i)}, gridColor);
        }

        for (int ix = 0; ix < kGridX - 1; ++ix) {
            for (int iz = 0; iz < kGridZ - 1; ++iz) {
                const int i00 = ix * kGridZ + iz;
                const int i10 = (ix + 1) * kGridZ + iz;
                const int i01 = ix * kGridZ + (iz + 1);
                const int i11 = (ix + 1) * kGridZ + (iz + 1);

                const Vector3 p00 = GridPoint(ix, iz, field[i00]);
                const Vector3 p10 = GridPoint(ix + 1, iz, field[i10]);
                const Vector3 p01 = GridPoint(ix, iz + 1, field[i01]);
                const Vector3 p11 = GridPoint(ix + 1, iz + 1, field[i11]);

                DrawTriangle3D(p00, p10, p01, tint[i00]);
                DrawTriangle3D(p10, p11, p01, tint[i11]);
                DrawLine3D(p00, p10, WithAlpha(tint[i00], 40));
                DrawLine3D(p00, p01, WithAlpha(tint[i00], 40));
            }
        }

        for (const HelicalPacket& packet : packets) {
            DrawHelicalPacket(packet, time);
        }

        for (const MergeFlash& flash : flashes) {
            DrawSphere(flash.pos, flash.radius, WithAlpha(flash.color, static_cast<unsigned char>(170.0f * flash.life)));
            DrawSphereWires(flash.pos, flash.radius * 1.45f, 8, 8, WithAlpha(flash.color, static_cast<unsigned char>(220.0f * flash.life)));
        }

        EndMode3D();

        DrawText("Helical Probability Wave Field", 20, 18, 30, Color{234, 240, 250, 255});
        DrawText("Each excitation is rendered as an actual traveling helix riding over the field surface. When two packet cores overlap, they fuse into one helix.", 20, 54, 18, Color{168, 184, 208, 255});
        DrawText("1/2/3/4 spawn from edges | SPACE collision pair | H toggle next handedness", 20, 80, 17, Color{168, 184, 208, 255});
        DrawText("Mouse drag orbit | wheel zoom | A auto | [ ] field gain | P pause | R reset", 20, 104, 17, Color{168, 184, 208, 255});

        const std::string hud = HudText(static_cast<int>(packets.size()), mergeCount, paused, autoSpawn, fieldGain, nextRightHanded);
        DrawText(hud.c_str(), 20, 134, 20, Color{126, 226, 255, 255});

        DrawRectangleRounded({986.0f, 20.0f, 310.0f, 114.0f}, 0.08f, 14, Color{10, 18, 31, 205});
        DrawRectangleRoundedLinesEx({986.0f, 20.0f, 310.0f, 114.0f}, 0.08f, 14, 2.0f, Color{49, 79, 113, 255});
        DrawText("field surface", 1006, 38, 17, Color{224, 232, 243, 255});
        DrawText("probability-style scalar envelope", 1006, 60, 16, Color{172, 190, 214, 255});
        DrawText("coils above surface", 1006, 84, 17, Color{255, 226, 146, 255});
        DrawText("actual helical wave packets", 1006, 106, 16, Color{172, 190, 214, 255});

        DrawFPS(20, 164);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
