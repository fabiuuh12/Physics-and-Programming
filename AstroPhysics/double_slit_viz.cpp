#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
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
    *distance = std::clamp(*distance, 4.0f, 30.0f);
    float cp = std::cos(*pitch);
    c->position = Vector3Add(c->target, {*distance * cp * std::cos(*yaw), *distance * std::sin(*pitch), *distance * cp * std::sin(*yaw)});
}

float Intensity(float y, float sep, float lambda, float D) {
    float r1 = std::sqrt(D*D + (y-0.5f*sep)*(y-0.5f*sep));
    float r2 = std::sqrt(D*D + (y+0.5f*sep)*(y+0.5f*sep));
    float phase = 2.0f * PI * (r1-r2) / lambda;
    float env = std::exp(-0.08f * y * y);
    return env * 0.5f * (1.0f + std::cos(phase));
}
}

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Double Slit Interference 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.2f, 5.0f, 8.8f};
    camera.target = {1.6f, 0.4f, 0.0f};
    camera.up = {0,1,0};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float camYaw=0.84f, camPitch=0.34f, camDistance=13.0f;

    float sep = 1.2f;
    float lambda = 0.8f;
    bool paused = false;
    float t = 0.0f;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) { sep=1.2f; lambda=0.8f; paused=false; t=0.0f; }
        if (IsKeyPressed(KEY_LEFT_BRACKET)) sep = std::max(0.4f, sep-0.05f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) sep = std::min(2.4f, sep+0.05f);
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) lambda = std::max(0.2f, lambda-0.03f);
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) lambda = std::min(1.6f, lambda+0.03f);

        UpdateOrbitCameraDragOnly(&camera,&camYaw,&camPitch,&camDistance);
        if (!paused) t += GetFrameTime();

        const float barrierX = 0.0f;
        const float screenX = 6.5f;

        BeginDrawing();
        ClearBackground(Color{6,9,16,255});
        BeginMode3D(camera);

        DrawGrid(24,0.5f);
        DrawCube({barrierX,0.3f,0}, 0.16f, 4.0f, 2.2f, Color{110,120,150,140});
        DrawCube({barrierX, 0.5f*sep+0.45f, 0}, 0.2f, 1.1f, 2.4f, Color{6,9,16,255});
        DrawCube({barrierX,-0.5f*sep-0.45f, 0}, 0.2f, 1.1f, 2.4f, Color{6,9,16,255});
        DrawCube({screenX,0.3f,0}, 0.12f, 5.2f, 2.4f, Color{120,140,180,180});

        Vector3 s1 = {barrierX, 0.5f*sep, 0.0f};
        Vector3 s2 = {barrierX,-0.5f*sep, 0.0f};
        DrawSphere(s1,0.08f,Color{130,220,255,255});
        DrawSphere(s2,0.08f,Color{130,220,255,255});

        for (int i=0;i<140;++i) {
            float y = -2.4f + 4.8f * i / 139.0f;
            float I = Intensity(y, sep, lambda, screenX-barrierX);
            Color c = Color{static_cast<unsigned char>(90 + 160*I), static_cast<unsigned char>(120 + 110*I), static_cast<unsigned char>(170 + 80*I), 255};
            DrawSphere({screenX+0.12f, y, 0.0f}, 0.02f + 0.04f*I, c);
        }

        for (int i=0;i<12;++i) {
            float r = std::fmod(t*2.0f + 0.4f*i, 10.0f);
            int seg = 60;
            for (int k=0;k<seg;++k) {
                float a0 = 2.0f*PI*k/seg;
                float a1 = 2.0f*PI*(k+1)/seg;
                DrawLine3D({s1.x, s1.y + r*std::cos(a0), s1.z + r*std::sin(a0)}, {s1.x, s1.y + r*std::cos(a1), s1.z + r*std::sin(a1)}, Color{120,200,255,80});
                DrawLine3D({s2.x, s2.y + r*std::cos(a0), s2.z + r*std::sin(a0)}, {s2.x, s2.y + r*std::cos(a1), s2.z + r*std::sin(a1)}, Color{120,200,255,80});
            }
        }

        EndMode3D();

        DrawText("Double Slit: Interference Pattern on Screen", 20, 18, 29, Color{232,238,248,255});
        DrawText("Hold left mouse: orbit | wheel: zoom | [ ] slit separation | +/- wavelength | P pause | R reset", 20, 54, 18, Color{164,183,210,255});
        std::ostringstream os;
        os << std::fixed << std::setprecision(2) << "sep=" << sep << "  lambda=" << lambda;
        if (paused) os << "  [PAUSED]";
        DrawText(os.str().c_str(), 20, 82, 20, Color{126,224,255,255});
        DrawFPS(20,110);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
