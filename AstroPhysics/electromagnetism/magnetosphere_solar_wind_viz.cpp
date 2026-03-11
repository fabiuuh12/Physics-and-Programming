#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace {

constexpr int kScreenWidth = 1440;
constexpr int kScreenHeight = 900;
constexpr int kWindParticleCount = 720;
constexpr float kPlanetRadius = 1.55f;
constexpr float kMagnetopauseRadius = 4.8f;

struct OrbitCameraState {
    float yaw = 0.62f;
    float pitch = 0.28f;
    float distance = 18.0f;
};

struct WindParticle {
    Vector3 pos{};
    float speed = 0.0f;
    float lane = 0.0f;
    float size = 0.0f;
};

float RandRange(std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    return dist(rng);
}

void UpdateOrbitCameraDragOnly(Camera3D* camera, OrbitCameraState* orbit) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        const Vector2 delta = GetMouseDelta();
        orbit->yaw -= delta.x * 0.0039f;
        orbit->pitch += delta.y * 0.0039f;
        orbit->pitch = std::clamp(orbit->pitch, -1.25f, 1.25f);
    }

    orbit->distance -= GetMouseWheelMove() * 0.8f;
    orbit->distance = std::clamp(orbit->distance, 7.0f, 34.0f);

    const float cp = std::cos(orbit->pitch);
    camera->target = {0.0f, 0.0f, 0.0f};
    camera->position = Vector3Add(camera->target, {
        orbit->distance * cp * std::cos(orbit->yaw),
        orbit->distance * std::sin(orbit->pitch),
        orbit->distance * cp * std::sin(orbit->yaw),
    });
}

std::vector<WindParticle> MakeWindParticles() {
    std::mt19937 rng(8080);
    std::vector<WindParticle> particles;
    particles.reserve(kWindParticleCount);

    for (int i = 0; i < kWindParticleCount; ++i) {
        particles.push_back({
            {RandRange(rng, -20.0f, 10.0f), RandRange(rng, -7.0f, 7.0f), RandRange(rng, -7.0f, 7.0f)},
            RandRange(rng, 5.0f, 10.0f),
            RandRange(rng, -1.0f, 1.0f),
            RandRange(rng, 0.03f, 0.075f),
        });
    }

    return particles;
}

Vector3 DipoleCurvePoint(float shellRadius, float latitude, float longitude) {
    const float c = std::cos(latitude);
    const float s = std::sin(latitude);
    const float radial = shellRadius * c * c;
    Vector3 p = {radial * c, radial * s, 0.0f};
    p = Vector3RotateByAxisAngle(p, {0.0f, 1.0f, 0.0f}, longitude);
    return p;
}

void DrawDipoleFieldLines() {
    for (int shell = 0; shell < 5; ++shell) {
        const float shellRadius = 3.1f + shell * 0.9f;
        for (int az = 0; az < 8; ++az) {
            const float longitude = (2.0f * PI * az) / 8.0f;
            Vector3 prev{};
            bool hasPrev = false;
            for (int i = 0; i <= 120; ++i) {
                const float t = -1.22f + 2.44f * static_cast<float>(i) / 120.0f;
                const Vector3 p = DipoleCurvePoint(shellRadius, t, longitude);
                if (Vector3Length(p) < kPlanetRadius * 1.02f) {
                    hasPrev = false;
                    continue;
                }
                if (hasPrev) DrawLine3D(prev, p, Fade(Color{110, 192, 255, 255}, 0.32f));
                prev = p;
                hasPrev = true;
            }
        }
    }
}

void DrawBowShock(float time) {
    for (int ring = 0; ring < 6; ++ring) {
        const float radiusY = 4.6f + ring * 0.42f;
        const float radiusZ = 4.0f + ring * 0.38f;
        const float offsetX = 2.6f + ring * 0.34f;
        for (int i = 0; i < 72; ++i) {
            const float a0 = (2.0f * PI * i) / 72.0f;
            const float a1 = (2.0f * PI * (i + 1)) / 72.0f;
            const Vector3 p0 = {offsetX + 0.2f * std::sin(time * 1.1f + a0 * 3.0f), radiusY * std::cos(a0), radiusZ * std::sin(a0)};
            const Vector3 p1 = {offsetX + 0.2f * std::sin(time * 1.1f + a1 * 3.0f), radiusY * std::cos(a1), radiusZ * std::sin(a1)};
            DrawLine3D(p0, p1, Fade(Color{255, 192, 110, 255}, 0.18f));
        }
    }
}

void UpdateWindParticles(std::vector<WindParticle>* particles, float dt, float time) {
    std::mt19937 rng(5150);

    for (WindParticle& particle : *particles) {
        particle.pos.x += particle.speed * dt;

        const float yz = std::sqrt(particle.pos.y * particle.pos.y + particle.pos.z * particle.pos.z);
        const float noseDx = particle.pos.x - 1.2f;
        const float boundary = kMagnetopauseRadius + 1.2f / (0.6f + std::max(-2.0f, particle.pos.x + 4.0f));

        if (noseDx > -1.4f && noseDx < 5.2f && yz < boundary) {
            const float theta = std::atan2(particle.pos.z, particle.pos.y);
            const float push = (boundary - yz) * 1.8f;
            particle.pos.y += std::cos(theta) * push * dt * 4.0f;
            particle.pos.z += std::sin(theta) * push * dt * 4.0f;
            particle.pos.x += (0.5f + yz * 0.08f) * dt;
        }

        particle.pos.y += 0.15f * std::sin(time * 1.8f + particle.lane * 4.0f + particle.pos.x * 0.2f) * dt;
        particle.pos.z += 0.14f * std::cos(time * 1.4f + particle.lane * 5.0f + particle.pos.x * 0.17f) * dt;

        if (particle.pos.x > 12.0f || std::fabs(particle.pos.y) > 10.0f || std::fabs(particle.pos.z) > 10.0f) {
            particle.pos.x = RandRange(rng, -22.0f, -14.0f);
            particle.pos.y = RandRange(rng, -7.0f, 7.0f);
            particle.pos.z = RandRange(rng, -7.0f, 7.0f);
            particle.speed = RandRange(rng, 5.0f, 10.0f);
            particle.lane = RandRange(rng, -1.0f, 1.0f);
        }
    }
}

}  // namespace

int main() {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(kScreenWidth, kScreenHeight, "Magnetosphere Solar Wind 3D - C++ (raylib)");
    SetWindowMinSize(980, 640);
    SetTargetFPS(120);

    Camera3D camera{};
    camera.position = {14.0f, 5.2f, 12.5f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 46.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    OrbitCameraState orbit{};
    std::vector<WindParticle> wind = MakeWindParticles();

    while (!WindowShouldClose()) {
        const float dt = std::max(1.0e-4f, GetFrameTime());
        const float time = static_cast<float>(GetTime());
        UpdateOrbitCameraDragOnly(&camera, &orbit);
        UpdateWindParticles(&wind, dt, time);

        BeginDrawing();
        ClearBackground(Color{4, 7, 14, 255});
        BeginMode3D(camera);

        DrawGrid(30, 1.0f);
        DrawBowShock(time);
        DrawDipoleFieldLines();

        DrawSphere({0.0f, 0.0f, 0.0f}, kPlanetRadius * 1.34f, Fade(Color{72, 168, 255, 255}, 0.08f));
        DrawSphere({0.0f, 0.0f, 0.0f}, kPlanetRadius, Color{52, 102, 188, 255});
        DrawSphereWires({0.0f, 0.0f, 0.0f}, kPlanetRadius * 1.02f, 16, 16, Fade(Color{180, 228, 255, 255}, 0.18f));

        DrawCylinder({0.0f, kPlanetRadius * 0.96f, 0.0f}, 0.42f, 0.10f, 0.44f, 14, Fade(Color{72, 255, 188, 255}, 0.18f));
        DrawCylinder({0.0f, -kPlanetRadius * 0.96f, 0.0f}, 0.42f, 0.10f, 0.44f, 14, Fade(Color{72, 255, 188, 255}, 0.18f));

        for (const WindParticle& p : wind) {
            const Vector3 tail = {p.pos.x - 0.28f, p.pos.y, p.pos.z};
            DrawLine3D(tail, p.pos, Fade(Color{170, 222, 255, 255}, 0.32f));
            DrawSphere(p.pos, p.size, Fade(Color{190, 232, 255, 255}, 0.85f));
        }

        EndMode3D();

        DrawRectangle(14, 14, 470, 92, Fade(BLACK, 0.28f));
        DrawText("Magnetosphere vs Solar Wind", 26, 24, 30, Color{234, 241, 252, 255});
        DrawText("Solar wind compresses the dayside field and stretches the nightside magnetotail.", 26, 58, 19, Color{170, 192, 223, 255});
        DrawText("Mouse orbit | wheel zoom", 26, 82, 18, Color{132, 220, 255, 255});
        DrawFPS(GetScreenWidth() - 96, 18);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
