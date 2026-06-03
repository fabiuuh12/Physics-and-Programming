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
    *dist = std::clamp(*dist, 6.0f, 50.0f);
    float cp = std::cos(*pitch);
    c->position = Vector3Add(c->target, {*dist * cp * std::cos(*yaw), *dist * std::sin(*pitch), *dist * cp * std::sin(*yaw)});
}
}

int main() {
    InitWindow(kW, kH, "H-R Diagram Evolution 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D cam{};
    cam.position = {11.5f, 6.5f, 11.0f};
    cam.target = {0, 3, 0};
    cam.up = {0, 1, 0};
    cam.fovy = 45.0f;
    cam.projection = CAMERA_PERSPECTIVE;
    float yaw = 0.82f, pitch = 0.34f, dist = 18.0f;

    float mass = 2.2f;     // solar masses
    float age = 0.0f;      // normalized
    float radius = 1.8f;   // normalized
    bool paused = false;
    std::deque<Vector3> trail;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            mass = 2.2f; age = 0.0f; radius = 1.8f; paused = false; trail.clear();
        }
        if (IsKeyDown(KEY_UP)) mass = std::min(20.0f, mass + 5.0f * GetFrameTime());
        if (IsKeyDown(KEY_DOWN)) mass = std::max(0.4f, mass - 5.0f * GetFrameTime());
        UpdateOrbitCameraDragOnly(&cam, &yaw, &pitch, &dist);

        if (!paused) {
            float dt = GetFrameTime();
            age += 0.08f * dt * std::pow(mass, 0.35f);
            if (age > 1.4f) age = 0.0f;

            // Hydrostatic-like balance toy model: pressure support vs self-gravity.
            float grav = mass * mass / std::max(0.25f, radius * radius);
            float pressure = 2.8f * std::pow(mass, 1.2f) / std::max(0.25f, radius * radius * radius);
            radius += (pressure - grav) * 0.22f * dt;
            radius = std::clamp(radius, 0.3f, 7.0f);

            float temp = 3200.0f + 7200.0f * std::pow(mass, 0.52f) / std::pow(radius, 0.35f);
            float lum = std::pow(mass, 3.4f) * std::pow(std::max(0.2f, radius), 0.45f);
            // RGB transition for giant phase.
            if (age > 0.85f) {
                temp *= (1.0f - 0.45f * (age - 0.85f) / 0.55f);
                lum *= (1.0f + 18.0f * (age - 0.85f) / 0.55f);
            }

            float x = (15000.0f - std::clamp(temp, 2500.0f, 15000.0f)) / 1200.0f - 5.0f;
            float y = std::log10(std::max(0.001f, lum)) * 2.4f + 1.5f;
            float z = age * 12.0f - 6.0f;
            trail.push_back({x, y, z});
            if (trail.size() > 700) trail.pop_front();
        }

        BeginDrawing();
        ClearBackground(Color{7, 10, 18, 255});
        BeginMode3D(cam);
        DrawGrid(22, 0.8f);
        DrawCubeWires({0, 4, 0}, 12, 9, 13, Fade(SKYBLUE, 0.45f));

        for (size_t i = 1; i < trail.size(); ++i) {
            float a = (float)i / (float)trail.size();
            DrawLine3D(trail[i - 1], trail[i], Fade(Color{255, 180, 120, 255}, a));
        }
        if (!trail.empty()) DrawSphere(trail.back(), 0.22f, Color{255, 235, 170, 255});
        EndMode3D();

        DrawText("H-R Diagram Evolution (3D track with gravity-pressure balance)", 20, 18, 28, Color{232, 238, 248, 255});
        DrawText("Mouse drag orbit | wheel zoom | Up/Down stellar mass | P pause | R reset", 20, 54, 18, Color{160, 182, 210, 255});
        char s[220];
        std::snprintf(s, sizeof(s), "mass=%.2f Msun  radius=%.2f  age=%.2f%s", mass, radius, age, paused ? "  [PAUSED]" : "");
        DrawText(s, 20, 82, 20, Color{126, 224, 255, 255});
        DrawFPS(20, 110);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
