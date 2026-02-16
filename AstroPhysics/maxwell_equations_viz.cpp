#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

namespace {
constexpr int kScreenWidth = 1280;
constexpr int kScreenHeight = 820;

void UpdateOrbitCameraDragOnly(Camera3D* c, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }
    *distance -= GetMouseWheelMove() * 0.6f;
    *distance = std::clamp(*distance, 4.0f, 34.0f);
    float cp = std::cos(*pitch);
    c->position = Vector3Add(c->target, {*distance * cp * std::cos(*yaw), *distance * std::sin(*pitch), *distance * cp * std::sin(*yaw)});
}

void DrawArrow(Vector3 a, Vector3 b, Color c) {
    DrawLine3D(a, b, c);
    Vector3 d = Vector3Normalize(Vector3Subtract(b, a));
    Vector3 s = Vector3Normalize(Vector3CrossProduct(d, {0.0f, 1.0f, 0.0f}));
    if (Vector3Length(s) < 1e-4f) s = {1.0f, 0.0f, 0.0f};
    DrawLine3D(b, Vector3Add(b, Vector3Add(Vector3Scale(d, -0.14f), Vector3Scale(s, 0.08f))), c);
    DrawLine3D(b, Vector3Add(b, Vector3Add(Vector3Scale(d, -0.14f), Vector3Scale(s, -0.08f))), c);
}
} // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Maxwell Equations Field Intuition 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.4f, 5.2f, 8.8f};
    camera.target = {0.0f, 0.4f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.84f, camPitch = 0.34f, camDistance = 13.2f;

    int mode = 0; // 0 divE,1 divB,2 curlE,3 curlB
    bool paused = false;
    float t = 0.0f;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) { mode = 0; paused = false; t = 0.0f; }
        if (IsKeyPressed(KEY_ONE)) mode = 0;
        if (IsKeyPressed(KEY_TWO)) mode = 1;
        if (IsKeyPressed(KEY_THREE)) mode = 2;
        if (IsKeyPressed(KEY_FOUR)) mode = 3;

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);
        if (!paused) t += GetFrameTime();

        BeginDrawing();
        ClearBackground(Color{6, 9, 16, 255});
        BeginMode3D(camera);

        DrawGrid(24, 0.5f);

        for (int ix = -4; ix <= 4; ++ix) {
            for (int iz = -4; iz <= 4; ++iz) {
                Vector3 p = {0.7f * ix, 0.4f, 0.7f * iz};
                Vector3 v = {0.0f, 0.0f, 0.0f};
                Color c = Color{160, 210, 255, 240};

                if (mode == 0) { // div E = rho/eps0
                    v = Vector3Scale(Vector3Normalize(p), 0.35f + 0.15f * std::sin(t));
                    c = Color{130, 220, 255, 240};
                } else if (mode == 1) { // div B = 0
                    v = {-(p.z), 0.0f, p.x};
                    v = Vector3Scale(Vector3Normalize(v), 0.42f);
                    c = Color{255, 190, 120, 240};
                } else if (mode == 2) { // curl E = -dB/dt
                    v = {-(p.z), 0.0f, p.x};
                    v = Vector3Scale(Vector3Normalize(v), 0.25f + 0.15f * std::sin(2.0f * t));
                    c = Color{140, 210, 255, 240};
                } else { // curl B = mu0J + mu0eps0 dE/dt
                    v = {-(p.z), 0.0f, p.x};
                    v = Vector3Scale(Vector3Normalize(v), 0.25f + 0.15f * std::cos(2.0f * t));
                    c = Color{255, 180, 120, 240};
                }

                DrawArrow(p, Vector3Add(p, v), c);
            }
        }

        if (mode == 0) DrawSphere({0.0f, 0.4f, 0.0f}, 0.12f, Color{255, 140, 120, 255});
        if (mode == 3) DrawCylinder({0.0f, 0.4f, 0.0f}, 0.08f, 0.08f, 1.2f, 16, Color{255, 170, 120, 180});

        EndMode3D();

        const char* labels[4] = {
            "1: Gauss(E)  div E = rho/eps0",
            "2: Gauss(B)  div B = 0",
            "3: Faraday   curl E = -dB/dt",
            "4: Ampere-Maxwell  curl B = mu0J + mu0eps0 dE/dt"
        };

        DrawText("Maxwell Equations: Field Intuition", 20, 18, 29, Color{232, 238, 248, 255});
        DrawText("Hold left mouse: orbit | wheel: zoom | 1..4 equation mode | P pause | R reset", 20, 54, 18, Color{164, 183, 210, 255});
        DrawText(labels[mode], 20, 82, 20, Color{190, 220, 255, 255});
        if (paused) DrawText("[PAUSED]", 20, 110, 20, Color{255, 210, 150, 255});
        DrawFPS(20, 138);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
