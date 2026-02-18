#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>

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
    *dist -= GetMouseWheelMove() * 0.7f;
    *dist = std::clamp(*dist, 5.0f, 45.0f);
    float cp = std::cos(*pitch);
    c->position = Vector3Add(c->target, {*dist * cp * std::cos(*yaw), *dist * std::sin(*pitch), *dist * cp * std::sin(*yaw)});
}
}

int main() {
    InitWindow(kW, kH, "Pulsar Timing + Gravitational Waves 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D cam{};
    cam.position = {9, 6, 9};
    cam.target = {0, 0, 0};
    cam.up = {0, 1, 0};
    cam.fovy = 45;
    cam.projection = CAMERA_PERSPECTIVE;
    float yaw = 0.82f, pitch = 0.34f, dist = 14.0f;

    float omega = 1.5f;
    float gwAmp = 0.015f;
    float t = 0.0f;
    bool paused = false;
    std::deque<float> residuals(420, 0.0f);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) { omega = 1.5f; gwAmp = 0.015f; t = 0.0f; residuals.assign(420, 0.0f); paused = false; }
        if (IsKeyDown(KEY_UP)) gwAmp = std::min(0.08f, gwAmp + 0.03f * GetFrameTime());
        if (IsKeyDown(KEY_DOWN)) gwAmp = std::max(0.0f, gwAmp - 0.03f * GetFrameTime());
        if (IsKeyDown(KEY_RIGHT)) omega = std::min(5.0f, omega + 1.2f * GetFrameTime());
        if (IsKeyDown(KEY_LEFT)) omega = std::max(0.4f, omega - 1.2f * GetFrameTime());
        UpdateOrbitCameraDragOnly(&cam, &yaw, &pitch, &dist);

        if (!paused) {
            t += GetFrameTime();
            float res = gwAmp * std::sin(2.0f * PI * 0.6f * t) + 0.005f * std::sin(2.0f * PI * 0.11f * t);
            residuals.push_back(res);
            if (residuals.size() > 420) residuals.pop_front();
        }

        float phase = omega * t;
        Vector3 pulsarPos{2.0f * std::cos(phase), 0.2f * std::sin(phase * 0.7f), 2.0f * std::sin(phase)};
        Vector3 compPos = Vector3Negate(pulsarPos);
        Vector3 beamDir = Vector3Normalize({std::cos(phase * 7.0f), 0.2f, std::sin(phase * 7.0f)});

        BeginDrawing();
        ClearBackground(Color{6, 9, 17, 255});
        BeginMode3D(cam);
        DrawGrid(16, 0.8f);
        DrawSphere(pulsarPos, 0.34f, Color{150, 220, 255, 255});
        DrawSphere(compPos, 0.24f, Color{255, 180, 130, 255});
        DrawLine3D(pulsarPos, compPos, Fade(SKYBLUE, 0.35f));
        DrawLine3D(pulsarPos, Vector3Add(pulsarPos, Vector3Scale(beamDir, 6.0f)), Fade(Color{120, 255, 180, 255}, 0.7f));
        DrawLine3D(pulsarPos, Vector3Add(pulsarPos, Vector3Scale(beamDir, -6.0f)), Fade(Color{120, 255, 180, 255}, 0.7f));
        EndMode3D();

        DrawRectangle(820, 520, 430, 230, Fade(Color{20, 28, 44, 255}, 0.9f));
        DrawText("Timing Residuals", 840, 536, 22, Color{220, 230, 244, 255});
        for (int i = 1; i < (int)residuals.size(); ++i) {
            int x0 = 840 + i - 1;
            int x1 = 840 + i;
            int y0 = 650 - (int)(residuals[i - 1] * 1700.0f);
            int y1 = 650 - (int)(residuals[i] * 1700.0f);
            DrawLine(x0, y0, x1, y1, Color{120, 240, 180, 255});
        }

        DrawText("Pulsar Timing + Gravitational Wave Perturbation", 20, 18, 30, Color{232, 238, 248, 255});
        DrawText("Mouse drag orbit | wheel zoom | Up/Down GW amplitude | Left/Right orbit rate | P pause | R reset", 20, 54, 18, Color{160, 182, 210, 255});
        char s[220];
        std::snprintf(s, sizeof(s), "gw_amp=%.4f  omega=%.2f%s", gwAmp, omega, paused ? "  [PAUSED]" : "");
        DrawText(s, 20, 82, 20, Color{126, 224, 255, 255});
        DrawFPS(20, 110);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
