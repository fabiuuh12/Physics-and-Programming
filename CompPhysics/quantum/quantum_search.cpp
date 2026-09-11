#include "raylib.h"
#include "../common/studio.h"
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
    InitWindow(kScreenWidth, kScreenHeight, "Quantum Search (Grover Intuition) 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.0f, 5.0f, 9.0f};
    camera.target = {0.0f, 0.6f, 0.0f};
    camera.up = {0,1,0};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float camYaw=0.84f, camPitch=0.34f, camDistance=13.0f;

    int N = 16;
    int target = 11;
    int iter = 0;
    bool paused = false;
    float t=0.0f;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused=!paused;
        if (IsKeyPressed(KEY_R)) { iter=0; target=11; paused=false; t=0.0f; }
        if (IsKeyPressed(KEY_SPACE)) iter++;
        if (IsKeyPressed(KEY_LEFT_BRACKET)) target = std::max(0, target-1);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) target = std::min(N-1, target+1);

        UpdateOrbitCameraDragOnly(&camera,&camYaw,&camPitch,&camDistance);
        if (!paused) t += GetFrameTime();

        float theta = std::asin(1.0f / std::sqrt(static_cast<float>(N)));
        float pTarget = std::pow(std::sin((2*iter+1)*theta), 2.0f);
        pTarget = std::clamp(pTarget, 0.0f, 1.0f);
        float pOther = (1.0f - pTarget) / (N - 1);

        BeginDrawing();
        ClearBackground(Color{6,9,16,255});
        BeginMode3D(camera);

        for (int i=0;i<N;++i) {
            int row = i / 4;
            int col = i % 4;
            float x = -2.4f + col * 1.6f;
            float z = -2.4f + row * 1.6f;
            float p = (i==target) ? pTarget : pOther;
            float h = 0.2f + 2.0f * p;
            Color c = (i==target) ? Color{255, 180, 120, 255} : Color{120, 200, 255, 230};
            DrawCube({x, 0.1f + h*0.5f, z}, 0.7f, h, 0.7f, c);
        }

        EndMode3D();

        studio::title("Quantum Search (Grover) Probability Amplification", studio::Style::Quantum);
        studio::help("Hold left mouse: orbit | wheel: zoom | SPACE iterate | [ ] target index | P pause | R reset");

        std::ostringstream os;
        os << std::fixed << std::setprecision(4)
           << "N=" << N << "  target=" << target << "  iteration=" << iter
           << "  P(target)~" << pTarget;
        if (paused) os << "  [PAUSED]";
        studio::readout(os.str().c_str());
        studio::fps();

        EndDrawing();
        if (studio::smokeFrame(__FILE__)) break;
    }

    studio::unload();
    CloseWindow();
    return 0;
}
