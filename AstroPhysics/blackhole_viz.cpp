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

constexpr int kScreenWidth = 1280;
constexpr int kScreenHeight = 820;

constexpr float kEventHorizonRadius = 0.65f;
constexpr float kPhotonRingRadius = 1.15f;
constexpr float kDiskInnerRadius = 1.35f;
constexpr float kDiskOuterRadius = 4.5f;
constexpr float kGravityMu = 12.0f;

struct DustParticle {
    Vector3 pos;
    Vector3 vel;
    float size;
    float heat;
};

float RandRange(std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    return d(rng);
}

DustParticle SpawnDust(std::mt19937& rng) {
    const float r = RandRange(rng, kDiskInnerRadius, kDiskOuterRadius);
    const float phi = RandRange(rng, 0.0f, 2.0f * PI);
    const float thickness = RandRange(rng, -0.15f, 0.15f);

    Vector3 pos = {r * std::cos(phi), thickness, r * std::sin(phi)};

    const float orbitalSpeed = std::sqrt(kGravityMu / r);
    Vector3 tangent = {-std::sin(phi), 0.0f, std::cos(phi)};
    Vector3 vel = Vector3Scale(tangent, orbitalSpeed * RandRange(rng, 0.92f, 1.06f));

    Vector3 inward = Vector3Normalize(Vector3Negate(pos));
    vel = Vector3Add(vel, Vector3Scale(inward, RandRange(rng, 0.03f, 0.1f)));
    vel.y += RandRange(rng, -0.05f, 0.05f);

    const float size = RandRange(rng, 0.03f, 0.07f);
    const float heat = RandRange(rng, 0.45f, 1.0f);
    return {pos, vel, size, heat};
}

void ResetDisk(std::vector<DustParticle>* disk, std::mt19937& rng, int count) {
    disk->clear();
    disk->reserve(count);
    for (int i = 0; i < count; ++i) {
        disk->push_back(SpawnDust(rng));
    }
}

Color DiskColor(float heat) {
    const float h = std::clamp(heat, 0.0f, 1.0f);
    const unsigned char r = static_cast<unsigned char>(180 + 70 * h);
    const unsigned char g = static_cast<unsigned char>(100 + 90 * h);
    const unsigned char b = static_cast<unsigned char>(45 + 35 * (1.0f - h));
    return Color{r, g, b, 220};
}

std::string HudText(float t, float speed, int particles, int swallowed, bool paused) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(2)
       << "t=" << t
       << "  speed=" << speed << "x"
       << "  particles=" << particles
       << "  swallowed=" << swallowed;
    if (paused) {
        os << "  [PAUSED]";
    }
    return os.str();
}

void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 delta = GetMouseDelta();
        *yaw -= delta.x * 0.0035f;
        *pitch += delta.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }

    *distance -= GetMouseWheelMove() * 0.6f;
    *distance = std::clamp(*distance, 3.0f, 26.0f);

    const float cp = std::cos(*pitch);
    const Vector3 offset = {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}

void DrawCircle3DXZ(float radius, int segments, Color color) {
    for (int i = 0; i < segments; ++i) {
        const float a0 = 2.0f * PI * static_cast<float>(i) / static_cast<float>(segments);
        const float a1 = 2.0f * PI * static_cast<float>(i + 1) / static_cast<float>(segments);
        DrawLine3D(
            {radius * std::cos(a0), 0.0f, radius * std::sin(a0)},
            {radius * std::cos(a1), 0.0f, radius * std::sin(a1)},
            color
        );
    }
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Black Hole 3D Visualization - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {7.5f, 4.0f, 7.5f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    std::random_device rd;
    std::mt19937 rng(rd());

    int desiredParticles = 520;
    int swallowed = 0;
    float simTime = 0.0f;
    float speed = 1.0f;
    bool paused = false;
    float camYaw = 0.78f;
    float camPitch = 0.38f;
    float camDistance = 11.0f;

    std::vector<DustParticle> disk;
    ResetDisk(&disk, rng, desiredParticles);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) {
            paused = !paused;
        }
        if (IsKeyPressed(KEY_R)) {
            simTime = 0.0f;
            swallowed = 0;
            ResetDisk(&disk, rng, desiredParticles);
        }
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) {
            speed = std::min(4.0f, speed + 0.25f);
        }
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) {
            speed = std::max(0.25f, speed - 0.25f);
        }
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) {
            desiredParticles = std::min(1400, desiredParticles + 100);
            ResetDisk(&disk, rng, desiredParticles);
        }
        if (IsKeyPressed(KEY_LEFT_BRACKET)) {
            desiredParticles = std::max(120, desiredParticles - 100);
            ResetDisk(&disk, rng, desiredParticles);
        }

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        if (!paused) {
            const float frameDt = GetFrameTime() * speed;
            simTime += frameDt;

            for (DustParticle& d : disk) {
                const Vector3 r = d.pos;
                const float r2 = Vector3DotProduct(r, r) + 0.04f;
                const float invR = 1.0f / std::sqrt(r2);
                const float invR3 = invR * invR * invR;

                Vector3 gravity = Vector3Scale(r, -kGravityMu * invR3);
                Vector3 drag = Vector3Scale(d.vel, -0.025f);
                d.vel = Vector3Add(d.vel, Vector3Scale(Vector3Add(gravity, drag), frameDt));
                d.pos = Vector3Add(d.pos, Vector3Scale(d.vel, frameDt));

                const float radius = std::sqrt(Vector3DotProduct(d.pos, d.pos));
                d.heat = std::clamp(1.25f - (radius - kDiskInnerRadius) / (kDiskOuterRadius - kDiskInnerRadius), 0.2f, 1.0f);
            }

            const int before = static_cast<int>(disk.size());
            disk.erase(
                std::remove_if(disk.begin(), disk.end(), [](const DustParticle& d) {
                    const float r = std::sqrt(Vector3DotProduct(d.pos, d.pos));
                    return r < kEventHorizonRadius * 1.02f || r > 11.0f;
                }),
                disk.end()
            );
            swallowed += before - static_cast<int>(disk.size());

            while (static_cast<int>(disk.size()) < desiredParticles) {
                disk.push_back(SpawnDust(rng));
            }
        }

        BeginDrawing();
        ClearBackground(Color{4, 6, 14, 255});

        BeginMode3D(camera);

        DrawGrid(28, 0.5f);
        DrawCircle3DXZ(kDiskInnerRadius, 96, Color{255, 140, 70, 70});
        DrawCircle3DXZ(kPhotonRingRadius, 120, Color{150, 210, 255, 100});
        DrawCircle3DXZ(kDiskOuterRadius, 120, Color{90, 130, 190, 45});

        DrawSphere({0.0f, 0.0f, 0.0f}, kPhotonRingRadius, Color{80, 130, 200, 18});
        DrawSphere({0.0f, 0.0f, 0.0f}, kEventHorizonRadius, BLACK);

        for (const DustParticle& d : disk) {
            DrawSphere(d.pos, d.size, DiskColor(d.heat));
        }

        EndMode3D();

        DrawText("Black Hole + Accretion Disk (3D)", 20, 18, 30, Color{232, 238, 248, 255});
        DrawText("Hold left mouse: orbit | wheel: zoom | P pause | +/- speed | [ ] density | R reset", 20, 54, 20, Color{164, 183, 210, 255});
        const std::string hud = HudText(simTime, speed, static_cast<int>(disk.size()), swallowed, paused);
        DrawText(hud.c_str(), 20, 84, 21, Color{126, 224, 255, 255});
        DrawFPS(20, 116);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
