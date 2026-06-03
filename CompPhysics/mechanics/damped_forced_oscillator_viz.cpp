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
    *distance = std::clamp(*distance, 4.0f, 32.0f);
    float cp = std::cos(*pitch);
    c->position = Vector3Add(c->target, {*distance * cp * std::cos(*yaw), *distance * std::sin(*pitch), *distance * cp * std::sin(*yaw)});
}

}

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Damped Forced Oscillator 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.0f, 5.0f, 8.8f};
    camera.target = {0.0f, 0.5f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float camYaw=0.84f, camPitch=0.34f, camDistance=13.0f;

    float m=1.0f, k=10.0f, c=1.2f;
    float F0=4.0f, w=2.4f;
    float x=1.0f, v=0.0f;
    float t=0.0f;
    bool paused=false;
    std::deque<float> hist;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused=!paused;
        if (IsKeyPressed(KEY_R)) { x=1.0f; v=0.0f; t=0.0f; paused=false; F0=4.0f; w=2.4f; c=1.2f; hist.clear(); }
        if (IsKeyPressed(KEY_LEFT_BRACKET)) w = std::max(0.2f, w-0.1f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) w = std::min(8.0f, w+0.1f);
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) c = std::max(0.0f, c-0.1f);
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) c = std::min(6.0f, c+0.1f);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        if (!paused) {
            float dt = GetFrameTime();
            float force = F0 * std::cos(w * t);
            float a = (force - c*v - k*x) / m;
            v += a * dt;
            x += v * dt;
            t += dt;
            hist.push_back(x);
            if (hist.size() > 500) hist.pop_front();
        }

        BeginDrawing();
        ClearBackground(Color{6,9,16,255});
        BeginMode3D(camera);

        Vector3 anchor = {-3.4f, 0.5f, 0.0f};
        Vector3 massPos = {-0.4f + x, 0.5f, 0.0f};

        DrawCube(anchor, 0.2f, 1.0f, 1.0f, Color{120,150,190,255});

        const int coils = 16;
        Vector3 prev = anchor;
        for (int i=1;i<=coils*8;++i) {
            float u = static_cast<float>(i)/(coils*8);
            float xx = anchor.x + (massPos.x - anchor.x)*u;
            float yy = anchor.y + 0.12f*std::sin(2.0f*PI*coils*u);
            Vector3 cur = {xx,yy,0};
            DrawLine3D(prev, cur, Color{170,210,255,255});
            prev = cur;
        }

        DrawCube(massPos, 0.45f, 0.45f, 0.45f, Color{255, 200, 120, 255});

        float gx0 = -2.8f;
        float gz = -1.5f;
        float scaleX = 0.012f;
        for (size_t i=1;i<hist.size();++i) {
            float x0 = gx0 + (i-1)*scaleX;
            float x1 = gx0 + i*scaleX;
            float y0 = 0.2f + 0.45f*hist[i-1];
            float y1 = 0.2f + 0.45f*hist[i];
            DrawLine3D({x0,y0,gz}, {x1,y1,gz}, Color{120,220,255,200});
        }

        EndMode3D();

        DrawText("Damped Forced Oscillator (Driven Spring-Mass)", 20, 18, 29, Color{232,238,248,255});
        DrawText("Hold left mouse: orbit | wheel: zoom | [ ] drive freq | +/- damping | P pause | R reset", 20, 54, 18, Color{164,183,210,255});

        float force = F0 * std::cos(w * t);
        std::ostringstream os;
        os << std::fixed << std::setprecision(3)
           << "x=" << x << "  v=" << v << "  F_drive=" << force << "  w=" << w << "  c=" << c;
        if (paused) os << "  [PAUSED]";
        DrawText(os.str().c_str(), 20, 82, 20, Color{126,224,255,255});
        DrawFPS(20, 110);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
