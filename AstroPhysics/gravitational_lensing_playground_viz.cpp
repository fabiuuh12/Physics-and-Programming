#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

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
    *dist -= GetMouseWheelMove() * 0.8f;
    *dist = std::clamp(*dist, 6.0f, 55.0f);
    float cp = std::cos(*pitch);
    c->position = Vector3Add(c->target, {*dist * cp * std::cos(*yaw), *dist * std::sin(*pitch), *dist * cp * std::sin(*yaw)});
}

Vector3 DeflectedRay(Vector3 src, Vector3 lensPos, float lensMass, float t) {
    Vector3 p = Vector3Lerp(src, {7.0f, src.y, src.z}, t);
    Vector3 dl = Vector3Subtract(p, lensPos);
    float b2 = dl.y * dl.y + dl.z * dl.z + 0.12f;
    float k = lensMass / b2;
    p.y -= k * dl.y * 0.02f;
    p.z -= k * dl.z * 0.02f;
    return p;
}
}

int main() {
    InitWindow(kW, kH, "Gravitational Lensing Playground 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D cam{};
    cam.position = {13, 8, 14};
    cam.target = {0, 0, 0};
    cam.up = {0, 1, 0};
    cam.fovy = 45;
    cam.projection = CAMERA_PERSPECTIVE;
    float yaw = 0.78f, pitch = 0.34f, dist = 21.0f;

    float lensMass = 120.0f;
    Vector3 lensPos{0.0f, 0.0f, 0.0f};
    bool paused = false;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) { lensMass = 120.0f; lensPos = {0, 0, 0}; paused = false; }
        if (IsKeyDown(KEY_UP)) lensMass = std::min(300.0f, lensMass + 70.0f * GetFrameTime());
        if (IsKeyDown(KEY_DOWN)) lensMass = std::max(20.0f, lensMass - 70.0f * GetFrameTime());
        if (IsKeyDown(KEY_W)) lensPos.y += 2.0f * GetFrameTime();
        if (IsKeyDown(KEY_S)) lensPos.y -= 2.0f * GetFrameTime();
        if (IsKeyDown(KEY_A)) lensPos.z -= 2.0f * GetFrameTime();
        if (IsKeyDown(KEY_D)) lensPos.z += 2.0f * GetFrameTime();
        UpdateOrbitCameraDragOnly(&cam, &yaw, &pitch, &dist);

        BeginDrawing();
        ClearBackground(Color{6, 10, 18, 255});
        BeginMode3D(cam);
        DrawSphere(lensPos, 0.6f, Color{255, 210, 120, 255});
        DrawSphereWires(lensPos, 2.2f, 24, 24, Fade(SKYBLUE, 0.22f));
        DrawGrid(20, 0.9f);

        for (int iy = -7; iy <= 7; ++iy) {
            for (int iz = -7; iz <= 7; ++iz) {
                Vector3 src{-7.0f, iy * 0.35f, iz * 0.35f};
                Vector3 prev = src;
                for (int i = 1; i <= 60; ++i) {
                    float t = (float)i / 60.0f;
                    Vector3 now = DeflectedRay(src, lensPos, lensMass, t);
                    DrawLine3D(prev, now, Fade(Color{130, 220, 255, 255}, 0.5f));
                    prev = now;
                }
            }
        }
        EndMode3D();

        DrawText("Gravitational Lensing Playground (3D ray deflection)", 20, 18, 30, Color{232, 238, 248, 255});
        DrawText("Mouse drag orbit | wheel zoom | Up/Down lens mass | WASD move lens in sky plane | P pause | R reset", 20, 54, 18, Color{160, 182, 210, 255});
        char s[220];
        std::snprintf(s, sizeof(s), "lens_mass=%.1f  lens_y=%.2f lens_z=%.2f%s", lensMass, lensPos.y, lensPos.z, paused ? "  [PAUSED]" : "");
        DrawText(s, 20, 82, 20, Color{126, 224, 255, 255});
        DrawFPS(20, 110);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
