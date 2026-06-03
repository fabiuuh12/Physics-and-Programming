#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>

namespace {
constexpr int kW = 1280;
constexpr int kH = 820;
constexpr float kG = 1.0f;

void UpdateOrbitCameraDragOnly(Camera3D* c, float* yaw, float* pitch, float* dist) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.3f, 1.3f);
    }
    *dist -= GetMouseWheelMove() * 0.6f;
    *dist = std::clamp(*dist, 5.0f, 40.0f);
    float cp = std::cos(*pitch);
    c->position = Vector3Add(c->target, {*dist * cp * std::cos(*yaw), *dist * std::sin(*pitch), *dist * cp * std::sin(*yaw)});
}

float TransitFlux(Vector3 planetPos, float starR, float planetR) {
    // Observer fixed on +X axis looking toward origin.
    if (planetPos.x < 0.0f) return 1.0f;
    float d = std::sqrt(planetPos.y * planetPos.y + planetPos.z * planetPos.z);
    float overlap = std::max(0.0f, starR + planetR - d);
    float frac = std::clamp(overlap / (2.0f * planetR), 0.0f, 1.0f);
    return 1.0f - frac * frac * (planetR * planetR) / (starR * starR);
}
}

int main() {
    InitWindow(kW, kH, "Exoplanet Transit Lab 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D cam{};
    cam.position = {8.8f, 4.8f, 8.8f};
    cam.target = {0, 0, 0};
    cam.up = {0, 1, 0};
    cam.fovy = 45.0f;
    cam.projection = CAMERA_PERSPECTIVE;
    float yaw = 0.8f, pitch = 0.33f, dist = 14.5f;

    float starMass = 130.0f;
    float starR = 1.2f;
    float planetR = 0.26f;
    bool paused = false;

    Vector3 p = {0.0f, 0.0f, 4.5f};
    Vector3 v = {5.1f, 0.0f, 0.0f};
    std::deque<float> fluxHistory(360, 1.0f);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            p = {0.0f, 0.0f, 4.5f};
            v = {5.1f, 0.0f, 0.0f};
            fluxHistory.assign(360, 1.0f);
            starMass = 130.0f;
            planetR = 0.26f;
            paused = false;
        }
        if (IsKeyDown(KEY_UP)) starMass = std::min(260.0f, starMass + 45.0f * GetFrameTime());
        if (IsKeyDown(KEY_DOWN)) starMass = std::max(35.0f, starMass - 45.0f * GetFrameTime());
        if (IsKeyDown(KEY_RIGHT_BRACKET)) planetR = std::min(0.56f, planetR + 0.32f * GetFrameTime());
        if (IsKeyDown(KEY_LEFT_BRACKET)) planetR = std::max(0.08f, planetR - 0.32f * GetFrameTime());

        UpdateOrbitCameraDragOnly(&cam, &yaw, &pitch, &dist);

        if (!paused) {
            float dt = GetFrameTime();
            Vector3 r = Vector3Negate(p);
            float rmag = std::max(0.35f, Vector3Length(r));
            Vector3 a = Vector3Scale(r, kG * starMass / (rmag * rmag * rmag));
            v = Vector3Add(v, Vector3Scale(a, dt));
            p = Vector3Add(p, Vector3Scale(v, dt));
            float flux = TransitFlux(p, starR, planetR);
            fluxHistory.push_back(flux);
            if (fluxHistory.size() > 360) fluxHistory.pop_front();
        }

        BeginDrawing();
        ClearBackground(Color{7, 10, 18, 255});
        BeginMode3D(cam);
        DrawSphere({0, 0, 0}, starR, Color{255, 196, 96, 255});
        DrawSphere(p, planetR, Color{130, 200, 255, 255});
        DrawLine3D({0, 0, 0}, p, Fade(SKYBLUE, 0.3f));
        DrawGrid(20, 0.8f);
        EndMode3D();

        DrawRectangle(880, 520, 360, 220, Fade(Color{20, 28, 44, 255}, 0.9f));
        DrawText("Light Curve", 900, 536, 22, Color{220, 230, 244, 255});
        for (int i = 1; i < (int)fluxHistory.size(); ++i) {
            float f0 = fluxHistory[i - 1];
            float f1 = fluxHistory[i];
            int x0 = 900 + i - 1;
            int x1 = 900 + i;
            int y0 = 720 - (int)((f0 - 0.88f) / 0.14f * 160.0f);
            int y1 = 720 - (int)((f1 - 0.88f) / 0.14f * 160.0f);
            DrawLine(x0, y0, x1, y1, Color{120, 240, 170, 255});
        }

        DrawText("Exoplanet Transit Lab (3D gravity orbit)", 20, 18, 30, Color{232, 238, 248, 255});
        DrawText("Mouse drag orbit | wheel zoom | Up/Down star mass | [ ] planet radius | P pause | R reset", 20, 54, 18, Color{160, 182, 210, 255});
        char s[220];
        std::snprintf(s, sizeof(s), "M*=%.1f  Rp=%.2f  flux=%.4f%s", starMass, planetR, fluxHistory.back(), paused ? "  [PAUSED]" : "");
        DrawText(s, 20, 82, 20, Color{126, 224, 255, 255});
        DrawFPS(20, 110);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
