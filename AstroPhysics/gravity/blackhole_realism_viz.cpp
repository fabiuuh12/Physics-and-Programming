#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>

namespace {
constexpr int kScreenWidth = 1280;
constexpr int kScreenHeight = 820;

void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0034f;
        *pitch += d.y * 0.0034f;
        *pitch = std::clamp(*pitch, -1.3f, 1.3f);
    }
    *distance -= GetMouseWheelMove() * 0.8f;
    *distance = std::clamp(*distance, 6.0f, 60.0f);
    float cp = std::cos(*pitch);
    camera->position = Vector3Add(camera->target, {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw)
    });
}
}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Black Hole Realism (WIP) 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {10.0f, 7.0f, 10.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float camYaw = 0.82f;
    float camPitch = 0.35f;
    float camDistance = 18.0f;

    float t = 0.0f;
    while (!WindowShouldClose()) {
        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);
        t += GetFrameTime();

        BeginDrawing();
        ClearBackground(Color{6, 9, 16, 255});
        BeginMode3D(camera);

        DrawGrid(20, 1.0f);
        DrawSphere({0.0f, 0.0f, 0.0f}, 1.2f, Color{14, 14, 18, 255});
        for (int i = 0; i < 96; ++i) {
            float a0 = (2.0f * PI * i) / 96.0f;
            float a1 = (2.0f * PI * (i + 1)) / 96.0f;
            float r = 2.2f + 0.2f * std::sin(t * 1.8f + i * 0.15f);
            Vector3 p0 = {r * std::cos(a0), 0.1f * std::sin(t * 2.0f + i), r * std::sin(a0)};
            Vector3 p1 = {r * std::cos(a1), 0.1f * std::sin(t * 2.0f + i + 1), r * std::sin(a1)};
            DrawLine3D(p0, p1, Color{255, 176, 112, 220});
        }
        EndMode3D();

        DrawText("Black Hole Realism (WIP placeholder)", 20, 18, 30, Color{232, 238, 248, 255});
        DrawText("Mouse orbit | wheel zoom", 20, 54, 20, Color{164, 183, 210, 255});
        DrawFPS(20, 84);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
