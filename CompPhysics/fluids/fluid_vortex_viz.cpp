#include "raylib.h"
#include "../common/studio.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <vector>

namespace {
constexpr int kScreenWidth = 1280;
constexpr int kScreenHeight = 820;

struct Marker { Vector3 pos; std::deque<Vector3> trail; };

void UpdateOrbitCameraDragOnly(Camera3D* c, float* yaw, float* pitch, float* distance) {
    studio::pan(c, *yaw, *pitch, *distance);
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON) && !studio::panGesture()) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }
    *distance -= GetMouseWheelMove() * 0.6f;
    *distance = std::clamp(*distance, 4.0f, 36.0f);
    float cp = std::cos(*pitch);
    c->position = Vector3Add(c->target, {*distance * cp * std::cos(*yaw), *distance * std::sin(*pitch), *distance * cp * std::sin(*yaw)});
}



}

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Fluid Vortex 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.5f, 5.4f, 8.8f};
    camera.target = {0.0f, 0.5f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float camYaw=0.84f, camPitch=0.34f, camDistance=13.5f;

    float strength = 2.0f;
    bool paused = false;

    std::vector<Marker> marks;
    for (int i=0;i<70;++i) {
        float a = 2.0f*PI*static_cast<float>(i)/70.0f;
        float r = 0.6f + 2.8f * std::fmod(i * 0.617f, 1.0f);
        Marker m;
        m.pos = {r*std::cos(a), 0.5f, r*std::sin(a)};
        m.trail.push_back(m.pos);
        marks.push_back(m);
    }

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            for (size_t i=0;i<marks.size();++i) {
                float a = 2.0f*PI*static_cast<float>(i)/70.0f;
                float r = 0.6f + 2.8f * std::fmod(i * 0.617f, 1.0f);
                marks[i].pos = {r*std::cos(a), 0.5f, r*std::sin(a)};
                marks[i].trail.clear();
                marks[i].trail.push_back(marks[i].pos);
            }
            paused = false;
            strength = 2.0f;
        }
        if (IsKeyPressed(KEY_LEFT_BRACKET)) strength = std::max(0.2f, strength - 0.2f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) strength = std::min(6.0f, strength + 0.2f);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        if (!paused) {
            float dt = std::min(GetFrameTime(), 0.1f);
            for (auto& m : marks) {
                // The prescribed axisymmetric field has constant radius: integrate
                // its angular velocity exactly to avoid artificial radial drift.
                float r2=m.pos.x*m.pos.x+m.pos.z*m.pos.z;
                float angle=strength*dt/(r2+0.12f);
                float oldX=m.pos.x;
                m.pos.x=oldX*std::cos(angle)-m.pos.z*std::sin(angle);
                m.pos.z=oldX*std::sin(angle)+m.pos.z*std::cos(angle);
                m.trail.push_back(m.pos);
                if (m.trail.size() > 150) m.trail.pop_front();
            }
        }

        BeginDrawing();
        ClearBackground(Color{6,9,16,255});
        BeginMode3D(camera);
        DrawCylinder({0,0.5f,0}, 0.18f, 0.18f, 1.1f, 20, Color{255,170,120,180});

        for (const auto& m : marks) {
            for (size_t i=1;i<m.trail.size();++i) DrawLine3D(m.trail[i-1], m.trail[i], Color{120,200,255,120});
            DrawSphere(m.pos, 0.04f, Color{140,220,255,230});
        }

        EndMode3D();

        studio::title("Fluid Vortex (Swirl Flow Field)", studio::Style::Field);
        studio::help("Hold left mouse: orbit | wheel: zoom | [ ] vortex strength | P pause | R reset");

        char buf[160];
        snprintf(buf, sizeof(buf), "strength=%.2f%s", strength, paused ? "  [PAUSED]" : "");
        studio::readout(buf);
        studio::note("Prescribed swirl / exact circular advection / no radial flow");
        studio::fps();

        EndDrawing();
        if (studio::smokeFrame(__FILE__)) break;
    }

    studio::unload();
    CloseWindow();
    return 0;
}
