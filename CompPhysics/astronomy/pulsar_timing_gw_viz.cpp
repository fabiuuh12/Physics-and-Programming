#include "raylib.h"
#include "../common/studio.h"
#include "../common/physics_models.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>

namespace {
constexpr int kW = 1280;
constexpr int kH = 820;

void UpdateOrbitCameraDragOnly(Camera3D* c, float* yaw, float* pitch, float* dist) {
    studio::pan(c, *yaw, *pitch, *dist);
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON) && !studio::panGesture()) {
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
    if (std::getenv("COMPPHYSICS_SMOKE_FRAMES")) SetConfigFlags(FLAG_WINDOW_HIDDEN);
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
    physics::Clock clock;
    int samples=0;
    std::deque<float> residuals(420, 0.0f);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) { omega = 1.5f; gwAmp = 0.015f; t = 0.0f; residuals.assign(420, 0.0f); paused = false; clock.reset(); samples=0; }
        if (IsKeyDown(KEY_UP)) gwAmp = std::min(0.08f, gwAmp + 0.03f * GetFrameTime());
        if (IsKeyDown(KEY_DOWN)) gwAmp = std::max(0.0f, gwAmp - 0.03f * GetFrameTime());
        if (IsKeyDown(KEY_RIGHT)) omega = std::min(5.0f, omega + 1.2f * GetFrameTime());
        if (IsKeyDown(KEY_LEFT)) omega = std::max(0.4f, omega - 1.2f * GetFrameTime());
        UpdateOrbitCameraDragOnly(&cam, &yaw, &pitch, &dist);

        if (!paused) clock.advance(GetFrameTime(),[&](double dt) {
            t += float(dt);
            if (++samples%4!=0) return;
            float res = gwAmp * std::sin(2.0f * PI * 0.6f * t) + 0.005f * std::sin(2.0f * PI * 0.11f * t);
            residuals.push_back(res);
            if (residuals.size() > 420) residuals.pop_front();
        });

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

        studio::plot({float(GetScreenWidth()-438),float(GetScreenHeight()-293),410,225},
                     "Synthetic timing residual / last 7 s",residuals,-0.09f,0.09f,Color{142,226,185,255});

        studio::title("Pulsar Timing + Gravitational Wave Perturbation", studio::Style::Observatory);
        studio::help("Mouse drag orbit | wheel zoom | Up/Down GW amplitude | Left/Right orbit rate | P pause | R reset");
        char s[220];
        std::snprintf(s, sizeof(s), "gw_amp=%.4f  omega=%.2f%s", gwAmp, omega, paused ? "  [PAUSED]" : "");
        studio::readout(s);
        studio::note("Illustrative sinusoidal timing perturbation / not a gravitational-wave inference model");
        studio::fps();
        EndDrawing();
        if (studio::smokeFrame(__FILE__)) break;
    }

    studio::unload();
    CloseWindow();
    return 0;
}
