#include "raylib.h"
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

}

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Fluid Mechanics Channel Flow 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.4f, 5.0f, 8.8f};
    camera.target = {0.0f, 0.6f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float camYaw=0.84f, camPitch=0.34f, camDistance=13.2f;

    float umax = 3.2f;
    bool paused = false;

    std::vector<Marker> markers;
    for (int iy=0; iy<9; ++iy) {
        for (int iz=0; iz<7; ++iz) {
            float y = -1.2f + 2.4f * iy / 8.0f;
            float z = -1.0f + 2.0f * iz / 6.0f;
            Marker m;
            m.pos = {-5.2f, y + 0.6f, z};
            m.trail.push_back(m.pos);
            markers.push_back(m);
        }
    }

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            int idx = 0;
            for (int iy=0; iy<9; ++iy) {
                for (int iz=0; iz<7; ++iz) {
                    float y = -1.2f + 2.4f * iy / 8.0f;
                    float z = -1.0f + 2.0f * iz / 6.0f;
                    markers[idx].pos = {-5.2f, y + 0.6f, z};
                    markers[idx].trail.clear();
                    markers[idx].trail.push_back(markers[idx].pos);
                    idx++;
                }
            }
            paused = false;
            umax = 3.2f;
        }
        if (IsKeyPressed(KEY_LEFT_BRACKET)) umax = std::max(0.5f, umax - 0.2f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) umax = std::min(8.0f, umax + 0.2f);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        if (!paused) {
            float dt = GetFrameTime();
            for (auto& m : markers) {
                float yRel = (m.pos.y - 0.6f) / 1.2f; // -1..1
                float u = umax * (1.0f - yRel * yRel); // parabolic profile
                m.pos.x += u * dt;
                if (m.pos.x > 5.2f) m.pos.x = -5.2f;
                m.trail.push_back(m.pos);
                if (m.trail.size() > 120) m.trail.pop_front();
            }
        }

        BeginDrawing();
        ClearBackground(Color{6,9,16,255});
        BeginMode3D(camera);

        DrawGrid(24, 0.5f);
        DrawCubeWires({0.0f, 0.6f, 0.0f}, 10.8f, 2.5f, 2.2f, Color{130,180,255,180});

        for (const auto& m : markers) {
            for (size_t i=1;i<m.trail.size();++i) DrawLine3D(m.trail[i-1], m.trail[i], Color{120,200,255,100});
            DrawSphere(m.pos, 0.035f, Color{140,220,255,230});
        }

        EndMode3D();

        DrawText("Channel Flow (Laminar Poiseuille Profile)", 20, 18, 29, Color{232,238,248,255});
        DrawText("Hold left mouse: orbit | wheel: zoom | [ ] max center velocity | P pause | R reset", 20, 54, 18, Color{164,183,210,255});

        char buf[180];
        snprintf(buf, sizeof(buf), "Umax=%.2f (center fastest, wall near zero)%s", umax, paused ? "  [PAUSED]" : "");
        DrawText(buf, 20, 82, 20, Color{126,224,255,255});
        DrawFPS(20, 110);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
