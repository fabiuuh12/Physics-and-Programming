#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <vector>

namespace {
constexpr int kScreenWidth = 1280;
constexpr int kScreenHeight = 820;
constexpr float kPi = 3.14159265358979323846f;

struct ShellPoint {
    Vector3 dir;
    float phase;
};

void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0034f;
        *pitch += d.y * 0.0034f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }
    *distance -= GetMouseWheelMove() * 0.8f;
    *distance = std::clamp(*distance, 8.0f, 95.0f);
    float cp = std::cos(*pitch);
    camera->position = Vector3Add(camera->target, {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw)
    });
}

std::vector<ShellPoint> BuildShellPoints(int n) {
    std::vector<ShellPoint> pts;
    pts.reserve(n);
    for (int i = 0; i < n; ++i) {
        float t = (i + 0.5f) / static_cast<float>(n);
        float y = 1.0f - 2.0f * t;
        float r = std::sqrt(std::max(0.0f, 1.0f - y * y));
        float phi = (kPi * (3.0f - std::sqrt(5.0f))) * static_cast<float>(i);
        Vector3 d = {r * std::cos(phi), y, r * std::sin(phi)};
        pts.push_back({d, 0.13f * static_cast<float>(i)});
    }
    return pts;
}
}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Supernova Remnant Expansion 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {14.0f, 9.0f, 14.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float camYaw = 0.8f;
    float camPitch = 0.35f;
    float camDistance = 26.0f;

    float energy = 1.0f;
    float density = 1.0f;
    float gradient = 0.35f;
    float timeScale = 1.0f;
    bool paused = false;
    float age = 0.02f;
    std::deque<float> shockHistory(360, 0.0f);
    std::vector<ShellPoint> shell = BuildShellPoints(1300);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            energy = 1.0f;
            density = 1.0f;
            gradient = 0.35f;
            timeScale = 1.0f;
            paused = false;
            age = 0.02f;
            shockHistory.assign(360, 0.0f);
        }
        if (IsKeyDown(KEY_UP)) energy = std::min(5.0f, energy + 1.0f * GetFrameTime());
        if (IsKeyDown(KEY_DOWN)) energy = std::max(0.2f, energy - 1.0f * GetFrameTime());
        if (IsKeyDown(KEY_RIGHT)) density = std::min(4.0f, density + 1.0f * GetFrameTime());
        if (IsKeyDown(KEY_LEFT)) density = std::max(0.2f, density - 1.0f * GetFrameTime());
        if (IsKeyDown(KEY_RIGHT_BRACKET)) gradient = std::min(1.0f, gradient + 0.8f * GetFrameTime());
        if (IsKeyDown(KEY_LEFT_BRACKET)) gradient = std::max(0.0f, gradient - 0.8f * GetFrameTime());
        if (IsKeyDown(KEY_EQUAL)) timeScale = std::min(4.0f, timeScale + 1.2f * GetFrameTime());
        if (IsKeyDown(KEY_MINUS)) timeScale = std::max(0.2f, timeScale - 1.2f * GetFrameTime());

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        if (!paused) {
            age += GetFrameTime() * timeScale;
            if (age > 28.0f) {
                age = 0.02f;
                shockHistory.assign(360, 0.0f);
            }
        }

        float sedov = std::pow(std::max(0.02f, age), 0.4f) * std::pow(energy / std::max(0.18f, density), 0.2f);
        float shellRadius = 2.2f * sedov;
        float shockSpeed = 0.4f * shellRadius / std::max(0.04f, age);
        shockHistory.push_back(shockSpeed);
        if (shockHistory.size() > 360) shockHistory.pop_front();

        BeginDrawing();
        ClearBackground(Color{6, 9, 16, 255});
        BeginMode3D(camera);

        DrawGrid(30, 1.2f);
        DrawSphere({0.0f, 0.0f, 0.0f}, 0.24f, Color{190, 215, 255, 255});  // Compact remnant.

        for (const ShellPoint& p : shell) {
            float localDensity = 1.0f + gradient * p.dir.x;  // Denser medium on +X slows expansion.
            localDensity = std::max(0.25f, localDensity);
            float localR = shellRadius / std::pow(localDensity, 0.23f);
            float turbulence = 1.0f + 0.08f * std::sin(4.0f * age + p.phase);
            Vector3 pos = Vector3Scale(p.dir, localR * turbulence);

            float heat = std::exp(-age / 18.0f);
            unsigned char rr = static_cast<unsigned char>(155 + 90 * heat);
            unsigned char gg = static_cast<unsigned char>(95 + 70 * heat);
            unsigned char bb = static_cast<unsigned char>(130 + 110 * (1.0f - heat));
            DrawPoint3D(pos, Color{rr, gg, bb, 215});
        }

        DrawSphereWires({0.0f, 0.0f, 0.0f}, shellRadius, 18, 18, Fade(Color{255, 175, 120, 255}, 0.25f));
        DrawLine3D({-10.0f, 0.0f, 0.0f}, {10.0f, 0.0f, 0.0f}, Fade(SKYBLUE, 0.3f));  // ISM gradient axis.
        EndMode3D();

        DrawRectangle(870, 516, 388, 236, Fade(Color{18, 26, 42, 255}, 0.92f));
        DrawText("Shock Speed Proxy", 892, 536, 22, Color{220, 230, 244, 255});
        for (int i = 1; i < static_cast<int>(shockHistory.size()); ++i) {
            float s0 = std::min(16.0f, shockHistory[i - 1]);
            float s1 = std::min(16.0f, shockHistory[i]);
            int x0 = 900 + i - 1;
            int x1 = 900 + i;
            int y0 = 732 - static_cast<int>((s0 / 16.0f) * 166.0f);
            int y1 = 732 - static_cast<int>((s1 / 16.0f) * 166.0f);
            DrawLine(x0, y0, x1, y1, Color{130, 240, 188, 255});
        }

        DrawText("Supernova Remnant Expansion (Sedov-like)", 20, 18, 30, Color{232, 238, 248, 255});
        DrawText("Mouse orbit | wheel zoom | Up/Down energy | Left/Right density | [ ] gradient | +/- time scale | P pause | R reset",
                 20, 54, 18, Color{164, 183, 210, 255});
        char status[230];
        std::snprintf(status, sizeof(status), "E=%.2f  rho=%.2f  grad=%.2f  age=%.2f  R=%.2f%s",
                      energy, density, gradient, age, shellRadius, paused ? " [PAUSED]" : "");
        DrawText(status, 20, 84, 20, Color{126, 224, 255, 255});
        DrawFPS(20, 112);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
