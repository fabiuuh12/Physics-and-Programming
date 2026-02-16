#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

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
    InitWindow(kScreenWidth, kScreenHeight, "Hawking Radiation Spectrum 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.0f, 5.2f, 9.2f};
    camera.target = {0,0.6f,0};
    camera.up = {0,1,0};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float camYaw=0.84f, camPitch=0.34f, camDistance=13.8f;

    float massBH = 4.0f;
    bool paused = false;
    float t = 0.0f;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) { massBH=4.0f; paused=false; t=0.0f; }
        if (IsKeyPressed(KEY_LEFT_BRACKET)) massBH = std::max(0.8f, massBH - 0.1f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) massBH = std::min(8.0f, massBH + 0.1f);

        UpdateOrbitCameraDragOnly(&camera,&camYaw,&camPitch,&camDistance);
        if (!paused) t += GetFrameTime();

        float T = 1.0f / std::max(0.1f, massBH);

        BeginDrawing();
        ClearBackground(Color{6,9,16,255});
        BeginMode3D(camera);

        DrawGrid(24,0.5f);
        DrawSphere({-2.8f,0.7f,0}, 0.22f*massBH, BLACK);
        DrawSphere({-2.8f,0.7f,0}, 0.22f*massBH + 0.1f, Color{120,170,230,35});

        for (int i=0;i<40;++i) {
            float E = 0.2f + 0.22f * i;
            float I = 1.0f / (std::exp(E / T) - 1.0f);
            I = std::clamp(I * 0.06f, 0.0f, 1.0f);
            float x = -0.8f + 0.23f*i;
            float h = 0.1f + 3.0f*I;
            Color c = Color{static_cast<unsigned char>(90 + 160*I), static_cast<unsigned char>(130 + 110*I), static_cast<unsigned char>(180 + 70*I), 230};
            DrawCube({x, 0.05f + h*0.5f, -1.1f}, 0.16f, h, 0.5f, c);
        }

        for (int i=0;i<22;++i) {
            float a = 2.0f*PI*i/22.0f + t*(0.4f+0.6f*T);
            float r = 1.3f + 0.3f*std::sin(t + i*0.4f);
            Vector3 p = {-2.8f + r*std::cos(a), 0.7f + 0.2f*std::sin(2*a), r*std::sin(a)};
            DrawSphere(p, 0.03f + 0.02f*T, Color{255, 200, 130, 220});
        }

        EndMode3D();

        DrawText("Hawking Radiation: Higher Temperature for Smaller Mass", 20, 18, 29, Color{232,238,248,255});
        DrawText("Hold left mouse: orbit | wheel: zoom | [ ] black hole mass | P pause | R reset", 20, 54, 18, Color{164,183,210,255});

        std::ostringstream os;
        os << std::fixed << std::setprecision(3) << "M=" << massBH << "  T_H~1/M=" << T << "  (bars: thermal spectrum)";
        if (paused) os << "  [PAUSED]";
        DrawText(os.str().c_str(), 20, 82, 20, Color{126,224,255,255});
        DrawFPS(20,110);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
