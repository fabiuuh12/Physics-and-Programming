#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
constexpr int kW = 1280;
constexpr int kH = 820;

struct Particle { Vector3 p; Vector3 v; };

void UpdateOrbitCameraDragOnly(Camera3D* c, float* yaw, float* pitch, float* dist) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.3f, 1.3f);
    }
    *dist -= GetMouseWheelMove() * 0.8f;
    *dist = std::clamp(*dist, 5.0f, 60.0f);
    float cp = std::cos(*pitch);
    c->position = Vector3Add(c->target, {*dist * cp * std::cos(*yaw), *dist * std::sin(*pitch), *dist * cp * std::sin(*yaw)});
}
}

int main() {
    InitWindow(kW, kH, "Black Hole Accretion + Relativistic Beaming 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D cam{};
    cam.position = {10, 7, 10};
    cam.target = {0, 0, 0};
    cam.up = {0, 1, 0};
    cam.fovy = 45;
    cam.projection = CAMERA_PERSPECTIVE;
    float yaw = 0.8f, pitch = 0.35f, dist = 15.5f;

    float mass = 220.0f;
    bool paused = false;

    std::vector<Particle> disk;
    disk.reserve(700);
    for (int i = 0; i < 700; ++i) {
        float a = (2.0f * PI * i) / 700.0f;
        float r = 2.6f + 4.8f * ((float)GetRandomValue(0, 1000) / 1000.0f);
        Vector3 p{r * std::cos(a), GetRandomValue(-12, 12) / 100.0f, r * std::sin(a)};
        float vTan = std::sqrt(mass / r) * 0.22f;
        disk.push_back({p, {-vTan * std::sin(a), 0.0f, vTan * std::cos(a)}});
    }

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) { mass = 220.0f; paused = false; }
        if (IsKeyDown(KEY_UP)) mass = std::min(420.0f, mass + 120.0f * GetFrameTime());
        if (IsKeyDown(KEY_DOWN)) mass = std::max(80.0f, mass - 120.0f * GetFrameTime());
        UpdateOrbitCameraDragOnly(&cam, &yaw, &pitch, &dist);

        if (!paused) {
            float dt = GetFrameTime();
            for (auto& p : disk) {
                float r = std::max(0.55f, Vector3Length(p.p));
                Vector3 a = Vector3Scale(p.p, -mass / (r * r * r) * 0.22f);
                p.v = Vector3Add(p.v, Vector3Scale(a, dt));
                p.p = Vector3Add(p.p, Vector3Scale(p.v, dt));
                if (r < 1.25f || r > 12.0f) {
                    float ang = GetRandomValue(0, 628) / 100.0f;
                    float rr = 3.0f + GetRandomValue(0, 450) / 100.0f;
                    p.p = {rr * std::cos(ang), GetRandomValue(-10, 10) / 100.0f, rr * std::sin(ang)};
                    float vTan = std::sqrt(mass / rr) * 0.22f;
                    p.v = {-vTan * std::sin(ang), 0.0f, vTan * std::cos(ang)};
                }
            }
        }

        BeginDrawing();
        ClearBackground(Color{5, 8, 16, 255});
        BeginMode3D(cam);
        DrawSphere({0, 0, 0}, 0.9f, BLACK);
        DrawSphereWires({0, 0, 0}, 1.35f, 24, 24, Fade(Color{140, 170, 220, 255}, 0.35f));
        Vector3 obsDir = Vector3Normalize(Vector3Subtract(cam.position, cam.target));
        for (const auto& p : disk) {
            Vector3 velDir = Vector3Normalize(p.v);
            float boost = std::clamp((Vector3DotProduct(velDir, obsDir) + 1.0f) * 0.5f, 0.0f, 1.0f);
            Color c{
                (unsigned char)(160 + 90 * boost),
                (unsigned char)(120 + 90 * boost),
                (unsigned char)(255 - 130 * boost),
                220
            };
            DrawSphere(p.p, 0.07f, c);
        }
        EndMode3D();

        DrawText("Black Hole Accretion Disk + Relativistic Beaming", 20, 18, 30, Color{232, 238, 248, 255});
        DrawText("Mouse drag orbit | wheel zoom | Up/Down BH mass | P pause | R reset", 20, 54, 18, Color{160, 182, 210, 255});
        char s[220];
        std::snprintf(s, sizeof(s), "M_BH=%.1f%s", mass, paused ? "  [PAUSED]" : "");
        DrawText(s, 20, 82, 20, Color{126, 224, 255, 255});
        DrawFPS(20, 110);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
