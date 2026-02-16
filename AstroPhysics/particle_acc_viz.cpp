#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <deque>
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
    *distance = std::clamp(*distance, 4.0f, 36.0f);
    float cp = std::cos(*pitch);
    c->position = Vector3Add(c->target, {*distance * cp * std::cos(*yaw), *distance * std::sin(*pitch), *distance * cp * std::sin(*yaw)});
}
}

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Particle Accelerator 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.8f, 5.3f, 9.0f};
    camera.target = {0,0.5f,0};
    camera.up = {0,1,0};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float camYaw=0.84f, camPitch=0.34f, camDistance=14.0f;

    float ringR = 3.2f;
    float B = 1.0f;
    float q = 1.0f;
    float m = 1.0f;
    float speed = 2.0f;
    float theta = 0.0f;
    bool paused=false;

    std::deque<Vector3> trail;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused=!paused;
        if (IsKeyPressed(KEY_R)) { B=1.0f; speed=2.0f; theta=0.0f; trail.clear(); paused=false; }
        if (IsKeyPressed(KEY_LEFT_BRACKET)) B = std::max(0.2f, B-0.1f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) B = std::min(4.0f, B+0.1f);
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) speed = std::max(0.2f, speed-0.1f);
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) speed = std::min(8.0f, speed+0.1f);

        UpdateOrbitCameraDragOnly(&camera,&camYaw,&camPitch,&camDistance);

        if (!paused) {
            float dt = GetFrameTime();
            float omega = q * B / m;
            theta += omega * dt * speed;
        }

        Vector3 p = {ringR*std::cos(theta), 0.5f, ringR*std::sin(theta)};
        trail.push_back(p);
        if (trail.size()>600) trail.pop_front();

        BeginDrawing();
        ClearBackground(Color{6,9,16,255});
        BeginMode3D(camera);

        DrawGrid(26, 0.5f);
        for (int i=0;i<120;++i) {
            float a0 = 2.0f*PI*i/120.0f;
            float a1 = 2.0f*PI*(i+1)/120.0f;
            DrawLine3D({ringR*std::cos(a0),0.5f,ringR*std::sin(a0)}, {ringR*std::cos(a1),0.5f,ringR*std::sin(a1)}, Color{130,180,255,160});
        }

        for (size_t i=1;i<trail.size();++i) DrawLine3D(trail[i-1], trail[i], Color{255,170,120,140});
        DrawSphere(p, 0.12f, Color{255, 210, 130, 255});

        DrawLine3D({0,0.0f,0},{0,2.0f,0}, Color{120,220,255,220});

        EndMode3D();

        DrawText("Particle Accelerator: Circular Motion in Magnetic Field", 20, 18, 29, Color{232,238,248,255});
        DrawText("Hold left mouse: orbit | wheel: zoom | [ ] B-field | +/- speed | P pause | R reset", 20, 54, 18, Color{164,183,210,255});

        std::ostringstream os;
        os << std::fixed << std::setprecision(3)
           << "B=" << B << "  q/m=" << (q/m) << "  omega=qB/m=" << (q*B/m) << "  speed scale=" << speed;
        if (paused) os << "  [PAUSED]";
        DrawText(os.str().c_str(), 20, 82, 20, Color{126,224,255,255});
        DrawFPS(20, 110);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
