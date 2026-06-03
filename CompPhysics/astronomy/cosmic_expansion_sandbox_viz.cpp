#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
constexpr int kW = 1280;
constexpr int kH = 820;

void UpdateOrbitCameraDragOnly(Camera3D* c, float* yaw, float* pitch, float* dist) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.3f, 1.3f);
    }
    *dist -= GetMouseWheelMove() * 1.0f;
    *dist = std::clamp(*dist, 6.0f, 70.0f);
    float cp = std::cos(*pitch);
    c->position = Vector3Add(c->target, {*dist * cp * std::cos(*yaw), *dist * std::sin(*pitch), *dist * cp * std::sin(*yaw)});
}
}

int main() {
    InitWindow(kW, kH, "Cosmic Expansion Sandbox 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D cam{};
    cam.position = {14, 10, 14};
    cam.target = {0, 0, 0};
    cam.up = {0, 1, 0};
    cam.fovy = 45;
    cam.projection = CAMERA_PERSPECTIVE;
    float yaw = 0.8f, pitch = 0.36f, dist = 25.0f;

    float omegaM = 0.30f;
    float omegaL = 0.70f;
    float a = 1.0f;
    float adot = 0.40f;
    bool paused = false;

    std::vector<Vector3> comoving;
    for (int x = -4; x <= 4; ++x) {
        for (int y = -4; y <= 4; ++y) {
            for (int z = -4; z <= 4; ++z) {
                if ((x + y + z) % 3 == 0) comoving.push_back({x * 0.8f, y * 0.8f, z * 0.8f});
            }
        }
    }

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) { omegaM = 0.30f; omegaL = 0.70f; a = 1.0f; adot = 0.40f; paused = false; }
        if (IsKeyDown(KEY_UP)) omegaL = std::min(1.4f, omegaL + 0.6f * GetFrameTime());
        if (IsKeyDown(KEY_DOWN)) omegaL = std::max(0.0f, omegaL - 0.6f * GetFrameTime());
        if (IsKeyDown(KEY_RIGHT)) omegaM = std::min(1.6f, omegaM + 0.6f * GetFrameTime());
        if (IsKeyDown(KEY_LEFT)) omegaM = std::max(0.0f, omegaM - 0.6f * GetFrameTime());
        UpdateOrbitCameraDragOnly(&cam, &yaw, &pitch, &dist);

        if (!paused) {
            float dt = GetFrameTime();
            // Simple Friedmann-like toy dynamics: a'' = -0.5*Omega_m/a^2 + Omega_L*a
            float addot = -0.5f * omegaM / std::max(0.1f, a * a) + omegaL * a;
            adot += addot * dt * 0.35f;
            a += adot * dt * 0.35f;
            a = std::clamp(a, 0.25f, 5.0f);
            if (a <= 0.26f || a >= 4.95f) adot *= -0.3f;
        }

        BeginDrawing();
        ClearBackground(Color{6, 10, 18, 255});
        BeginMode3D(cam);
        DrawGrid(20, 1.0f);
        for (const auto& c : comoving) {
            Vector3 p = Vector3Scale(c, a);
            float speedProxy = Vector3Length(c) * std::fabs(adot);
            unsigned char r = (unsigned char)std::clamp(80.0f + 35.0f * speedProxy, 80.0f, 255.0f);
            unsigned char b = (unsigned char)std::clamp(255.0f - 30.0f * speedProxy, 70.0f, 255.0f);
            DrawSphere(p, 0.11f, Color{r, 170, b, 235});
            DrawLine3D({0, 0, 0}, p, Fade(Color{120, 180, 255, 255}, 0.10f));
        }
        EndMode3D();

        DrawText("Cosmic Expansion Sandbox (matter vs dark energy)", 20, 18, 30, Color{232, 238, 248, 255});
        DrawText("Mouse drag orbit | wheel zoom | Left/Right Omega_m | Up/Down Omega_Lambda | P pause | R reset", 20, 54, 18, Color{160, 182, 210, 255});
        char s[260];
        std::snprintf(s, sizeof(s), "Omega_m=%.2f  Omega_Lambda=%.2f  a=%.2f  adot=%.2f%s", omegaM, omegaL, a, adot, paused ? "  [PAUSED]" : "");
        DrawText(s, 20, 82, 20, Color{126, 224, 255, 255});
        DrawFPS(20, 110);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
