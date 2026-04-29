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

float Gamma(float beta) {
    beta = std::clamp(beta, 0.0f, 0.999f);
    return 1.0f / std::sqrt(1.0f - beta * beta);
}

void DrawClock3D(Vector3 center, float radius, float timeVal, Color ring, Color hand) {
    DrawCylinderWires({center.x, center.y - 0.05f, center.z}, radius, radius, 0.1f, 36, ring);
    DrawSphere({center.x, center.y, center.z}, 0.02f, ring);

    float a = -2.0f * PI * std::fmod(timeVal, 1.0f);
    Vector3 tip = {center.x + radius * 0.85f * std::cos(a), center.y, center.z + radius * 0.85f * std::sin(a)};
    DrawLine3D(center, tip, hand);
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Relativistic Time Dilation 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.0f, 5.2f, 8.8f};
    camera.target = {0.0f, 0.6f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.84f, camPitch = 0.34f, camDistance = 13.0f;

    float beta = 0.65f;
    bool paused = false;
    float tLab = 0.0f;
    float tShip = 0.0f;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) { beta = 0.65f; paused = false; tLab = 0.0f; tShip = 0.0f; }
        if (IsKeyPressed(KEY_LEFT_BRACKET)) beta = std::max(0.0f, beta - 0.02f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) beta = std::min(0.99f, beta + 0.02f);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        float g = Gamma(beta);
        if (!paused) {
            float dt = GetFrameTime();
            tLab += dt;
            tShip += dt / g;
        }

        Vector3 labClock = {-2.1f, 0.7f, 0.0f};
        Vector3 shipClock = {2.1f, 0.7f, 0.0f};

        BeginDrawing();
        ClearBackground(Color{7, 10, 16, 255});

        BeginMode3D(camera);

        DrawCube({-2.1f, 0.3f, 0.0f}, 1.3f, 0.6f, 1.3f, Color{120, 170, 230, 120});
        DrawCube({2.1f, 0.3f, 0.0f}, 1.6f, 0.6f, 1.0f, Color{255, 180, 120, 120});

        DrawClock3D(labClock, 0.45f, tLab * 0.6f, Color{130, 210, 255, 255}, Color{130, 230, 255, 255});
        DrawClock3D(shipClock, 0.45f, tShip * 0.6f, Color{255, 190, 120, 255}, Color{255, 220, 140, 255});

        float shipX = 2.1f + 2.2f * std::sin(0.4f * tLab);
        DrawLine3D({-3.5f, 0.05f, 0.0f}, {3.5f, 0.05f, 0.0f}, Color{110, 130, 170, 120});
        DrawSphere({shipX, 0.12f, 0.0f}, 0.12f, Color{255, 160, 120, 230});

        EndMode3D();

        DrawText("Relativistic Time Dilation (Twin Clock Concept)", 20, 18, 29, Color{235, 240, 250, 255});
        DrawText("Hold left mouse: orbit | wheel: zoom | [ ] velocity beta=v/c | P pause | R reset", 20, 54, 18, Color{170, 184, 204, 255});

        std::ostringstream os;
        os << std::fixed << std::setprecision(3)
           << "beta=" << beta
           << "  gamma=" << g
           << "  lab time=" << tLab
           << "  ship proper time=" << tShip;
        if (paused) os << "  [PAUSED]";
        DrawText(os.str().c_str(), 20, 82, 20, Color{200, 220, 255, 255});
        DrawFPS(20, 110);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
