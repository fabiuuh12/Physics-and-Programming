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
    *distance = std::clamp(*distance, 4.0f, 36.0f);
    float cp = std::cos(*pitch);
    c->position = Vector3Add(c->target, {*distance * cp * std::cos(*yaw), *distance * std::sin(*pitch), *distance * cp * std::sin(*yaw)});
}

}

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Schrodinger's Cat 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {7.6f, 4.8f, 8.8f};
    camera.target = {0.0f, 0.7f, 0.0f};
    camera.up = {0,1,0};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float camYaw=0.84f, camPitch=0.34f, camDistance=13.0f;

    bool measured=false;
    bool alive=true;
    float pAlive = 0.5f;
    bool paused=false;
    float t=0.0f;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused=!paused;
        if (IsKeyPressed(KEY_R)) { measured=false; alive=true; pAlive=0.5f; paused=false; t=0.0f; }
        if (IsKeyPressed(KEY_M)) {
            measured = true;
            alive = (pAlive >= 0.5f);
        }
        if (!measured) {
            if (IsKeyPressed(KEY_LEFT_BRACKET)) pAlive = std::max(0.0f, pAlive - 0.05f);
            if (IsKeyPressed(KEY_RIGHT_BRACKET)) pAlive = std::min(1.0f, pAlive + 0.05f);
        }

        UpdateOrbitCameraDragOnly(&camera,&camYaw,&camPitch,&camDistance);
        if (!paused) t += GetFrameTime();

        float amp = measured ? 0.0f : 0.5f * (1.0f + std::sin(2.2f*t));
        float shownAlive = measured ? (alive ? 1.0f : 0.0f) : pAlive * amp + (1.0f-amp)*0.5f;
        float shownDead  = measured ? (alive ? 0.0f : 1.0f) : (1.0f-pAlive) * amp + (1.0f-amp)*0.5f;

        BeginDrawing();
        ClearBackground(Color{6,9,16,255});
        BeginMode3D(camera);

        DrawCubeWires({0.0f,0.8f,0.0f}, 4.2f, 2.0f, 2.5f, Color{130,180,255,180});

        Vector3 alivePos = {-0.9f, 0.65f, 0.0f};
        Vector3 deadPos  = { 0.9f, 0.65f, 0.0f};
        DrawSphere(alivePos, 0.35f + 0.1f*shownAlive, Color{120,255,160, static_cast<unsigned char>(70 + 185*shownAlive)});
        DrawSphere(deadPos,  0.35f + 0.1f*shownDead,  Color{255,140,140, static_cast<unsigned char>(70 + 185*shownDead)});

        DrawLine3D({-1.5f,1.3f,0.0f},{1.5f,1.3f,0.0f},Color{200,210,230,180});

        EndMode3D();

        DrawText("Schrodinger's Cat: Superposition to Measurement", 20, 18, 29, Color{232,238,248,255});
        DrawText("Hold left mouse: orbit | wheel: zoom | [ ] prior P(alive) | M measure | P pause | R reset", 20, 54, 18, Color{164,183,210,255});

        std::ostringstream os;
        os << std::fixed << std::setprecision(2)
           << "P(alive)=" << pAlive << "  P(dead)=" << (1.0f-pAlive)
           << "  state=" << (measured ? (alive ? "measured alive" : "measured dead") : "superposed");
        if (paused) os << "  [PAUSED]";
        DrawText(os.str().c_str(), 20, 82, 20, Color{126,224,255,255});
        DrawFPS(20, 110);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
