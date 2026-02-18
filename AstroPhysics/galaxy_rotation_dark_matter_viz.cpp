#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
constexpr int kW = 1280;
constexpr int kH = 820;

struct Star { Vector3 p; Vector3 v; };

void UpdateOrbitCameraDragOnly(Camera3D* c, float* yaw, float* pitch, float* dist) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.3f, 1.3f);
    }
    *dist -= GetMouseWheelMove() * 1.0f;
    *dist = std::clamp(*dist, 12.0f, 90.0f);
    float cp = std::cos(*pitch);
    c->position = Vector3Add(c->target, {*dist * cp * std::cos(*yaw), *dist * std::sin(*pitch), *dist * cp * std::sin(*yaw)});
}

float EnclosedMass(float r, bool withDM, float bulgeM, float haloScale) {
    float vis = bulgeM * (1.0f - std::exp(-r / 2.4f));
    float dm = withDM ? haloScale * r * r / (1.0f + 0.08f * r * r) : 0.0f;
    return vis + dm;
}
}

int main() {
    InitWindow(kW, kH, "Galaxy Rotation + Dark Matter 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D cam{};
    cam.position = {22, 14, 22};
    cam.target = {0, 0, 0};
    cam.up = {0, 1, 0};
    cam.fovy = 45;
    cam.projection = CAMERA_PERSPECTIVE;
    float yaw = 0.8f, pitch = 0.35f, dist = 36.0f;

    float bulgeM = 320.0f;
    float halo = 55.0f;
    bool withDM = true;
    bool paused = false;

    std::vector<Star> stars;
    stars.reserve(900);
    for (int i = 0; i < 900; ++i) {
        float a = (2.0f * PI * i) / 900.0f + GetRandomValue(-80, 80) * 0.001f;
        float r = 1.5f + 16.0f * std::sqrt((float)GetRandomValue(0, 1000) / 1000.0f);
        Vector3 p{r * std::cos(a), GetRandomValue(-25, 25) / 100.0f, r * std::sin(a)};
        float mEn = EnclosedMass(r, withDM, bulgeM, halo);
        float vTan = std::sqrt(mEn / std::max(0.3f, r));
        Vector3 t{-std::sin(a), 0.0f, std::cos(a)};
        stars.push_back({p, Vector3Scale(t, vTan)});
    }

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_D)) withDM = !withDM;
        if (IsKeyPressed(KEY_R)) {
            paused = false;
            withDM = true;
        }
        if (IsKeyDown(KEY_UP)) halo = std::min(140.0f, halo + 45.0f * GetFrameTime());
        if (IsKeyDown(KEY_DOWN)) halo = std::max(0.0f, halo - 45.0f * GetFrameTime());

        UpdateOrbitCameraDragOnly(&cam, &yaw, &pitch, &dist);

        if (!paused) {
            float dt = GetFrameTime();
            for (auto& s : stars) {
                float r = std::max(0.45f, std::sqrt(s.p.x * s.p.x + s.p.z * s.p.z));
                float mEn = EnclosedMass(r, withDM, bulgeM, halo);
                float aMag = mEn / (r * r);
                Vector3 a{-aMag * s.p.x / r, -0.25f * s.p.y, -aMag * s.p.z / r};
                s.v = Vector3Add(s.v, Vector3Scale(a, dt));
                s.p = Vector3Add(s.p, Vector3Scale(s.v, dt));
            }
        }

        BeginDrawing();
        ClearBackground(Color{5, 8, 16, 255});
        BeginMode3D(cam);
        DrawSphere({0, 0, 0}, 0.8f, Color{255, 220, 130, 255});
        DrawSphereWires({0, 0, 0}, 6.0f, 24, 24, Fade(SKYBLUE, withDM ? 0.15f : 0.05f));
        DrawSphereWires({0, 0, 0}, 12.0f, 24, 24, Fade(SKYBLUE, withDM ? 0.20f : 0.05f));
        for (const auto& s : stars) {
            float r = std::sqrt(s.p.x * s.p.x + s.p.z * s.p.z);
            Color c = (r < 7.0f) ? Color{255, 200, 130, 220} : Color{140, 200, 255, 210};
            DrawSphere(s.p, 0.07f, c);
        }
        EndMode3D();

        DrawText("Galaxy Rotation Curves + Dark Matter Halo", 20, 18, 30, Color{232, 238, 248, 255});
        DrawText("Mouse drag orbit | wheel zoom | D toggle dark matter | Up/Down halo strength | P pause | R reset", 20, 54, 18, Color{160, 182, 210, 255});
        char s[240];
        std::snprintf(s, sizeof(s), "dark_matter=%s  halo=%.1f%s", withDM ? "ON" : "OFF", halo, paused ? "  [PAUSED]" : "");
        DrawText(s, 20, 82, 20, Color{126, 224, 255, 255});
        DrawFPS(20, 110);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
