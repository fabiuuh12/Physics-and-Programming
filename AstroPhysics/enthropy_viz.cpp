#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {
constexpr int kScreenWidth = 1280;
constexpr int kScreenHeight = 820;
struct Dot { Vector3 p; Vector3 v; };

void UpdateOrbitCameraDragOnly(Camera3D* c, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }
    *distance -= GetMouseWheelMove() * 0.6f;
    *distance = std::clamp(*distance, 4.0f, 35.0f);
    float cp = std::cos(*pitch);
    c->position = Vector3Add(c->target, {*distance * cp * std::cos(*yaw), *distance * std::sin(*pitch), *distance * cp * std::sin(*yaw)});
}
}

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Entropy Mixing 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.0f, 5.0f, 9.0f};
    camera.target = {0,0.7f,0};
    camera.up = {0,1,0};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float camYaw=0.84f, camPitch=0.34f, camDistance=13.0f;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> u(-1.0f,1.0f);

    std::vector<Dot> dots;
    auto reset = [&]() {
        dots.clear();
        for (int i=0;i<220;++i) {
            float side = (i < 110) ? -1.0f : 1.0f;
            Vector3 p = {side*1.2f + 0.6f*u(rng), 0.6f + 0.9f*u(rng), 1.2f*u(rng)};
            Vector3 v = {1.2f*u(rng), 1.2f*u(rng), 1.2f*u(rng)};
            dots.push_back({p,v});
        }
    };

    bool wall = true;
    bool paused = false;
    reset();

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused=!paused;
        if (IsKeyPressed(KEY_W)) wall = !wall;
        if (IsKeyPressed(KEY_R)) { reset(); wall=true; paused=false; }

        UpdateOrbitCameraDragOnly(&camera,&camYaw,&camPitch,&camDistance);

        if (!paused) {
            float dt = GetFrameTime();
            for (auto& d : dots) {
                d.p = Vector3Add(d.p, Vector3Scale(d.v, dt));
                if (d.p.x < -2.2f || d.p.x > 2.2f) d.v.x *= -1.0f;
                if (d.p.y < -0.3f || d.p.y > 1.5f) d.v.y *= -1.0f;
                if (d.p.z < -1.6f || d.p.z > 1.6f) d.v.z *= -1.0f;
                if (wall && std::fabs(d.p.x) < 0.03f) d.v.x *= -1.0f;
            }
        }

        int leftCount=0;
        for (auto& d : dots) if (d.p.x < 0.0f) leftCount++;
        float mix = 1.0f - std::fabs(leftCount - 110.0f)/110.0f;

        BeginDrawing();
        ClearBackground(Color{6,9,16,255});
        BeginMode3D(camera);

        DrawGrid(24,0.5f);
        DrawCubeWires({0,0.6f,0}, 4.6f, 2.0f, 3.4f, Color{130,180,255,180});
        if (wall) DrawCube({0,0.6f,0}, 0.05f, 1.9f, 3.2f, Color{170,170,190,130});

        for (int i=0;i<(int)dots.size();++i) {
            Color c = (i<110) ? Color{255,140,120,230} : Color{120,200,255,230};
            DrawSphere(dots[i].p, 0.045f, c);
        }

        EndMode3D();

        DrawText("Entropy and Mixing (Box Gas Model)", 20, 18, 29, Color{232,238,248,255});
        DrawText("Hold left mouse: orbit | wheel: zoom | W toggle partition | P pause | R reset", 20, 54, 18, Color{164,183,210,255});

        std::ostringstream os;
        os << std::fixed << std::setprecision(2) << "left count=" << leftCount << "  mixing index=" << mix;
        if (wall) os << "  [partition ON]";
        if (paused) os << "  [PAUSED]";
        DrawText(os.str().c_str(), 20, 82, 20, Color{126,224,255,255});
        DrawFPS(20,110);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
