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
    *distance = std::clamp(*distance, 4.0f, 35.0f);
    float cp = std::cos(*pitch);
    c->position = Vector3Add(c->target, {*distance * cp * std::cos(*yaw), *distance * std::sin(*pitch), *distance * cp * std::sin(*yaw)});
}
} // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Particle Entanglement Correlation 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {7.8f, 4.9f, 8.8f};
    camera.target = {0,0.6f,0};
    camera.up = {0,1,0};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float camYaw=0.84f, camPitch=0.34f, camDistance=13.0f;

    float a = 0.0f;
    float b = PI/4.0f;
    bool paused = false;
    float t = 0.0f;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) { a=0.0f; b=PI/4.0f; paused=false; t=0.0f; }
        if (IsKeyPressed(KEY_LEFT)) a -= 0.06f;
        if (IsKeyPressed(KEY_RIGHT)) a += 0.06f;
        if (IsKeyPressed(KEY_DOWN)) b -= 0.06f;
        if (IsKeyPressed(KEY_UP)) b += 0.06f;

        UpdateOrbitCameraDragOnly(&camera,&camYaw,&camPitch,&camDistance);
        if (!paused) t += GetFrameTime();

        float corr = -std::cos(2.0f*(a-b));
        float pSame = 0.5f * (1.0f + corr);
        float pDiff = 1.0f - pSame;

        BeginDrawing();
        ClearBackground(Color{6,9,16,255});
        BeginMode3D(camera);
        Vector3 p1 = {-1.8f, 0.6f, 0.0f};
        Vector3 p2 = { 1.8f, 0.6f, 0.0f};
        DrawSphere(p1, 0.22f, Color{120,220,255,255});
        DrawSphere(p2, 0.22f, Color{255,170,120,255});
        DrawLine3D(p1, p2, Color{170,200,255,120});

        DrawLine3D(p1, Vector3Add(p1, {std::cos(a), 0.0f, std::sin(a)}), Color{130,220,255,255});
        DrawLine3D(p2, Vector3Add(p2, {std::cos(b), 0.0f, std::sin(b)}), Color{255,180,120,255});

        DrawCube({0.0f, 0.25f + pSame, -1.5f}, 0.5f, 2.0f*pSame, 0.4f, Color{120,220,255,220});
        DrawCube({0.8f, 0.25f + pDiff, -1.5f}, 0.5f, 2.0f*pDiff, 0.4f, Color{255,170,120,220});

        EndMode3D();

        DrawText("Entanglement Correlation (Singlet-like Model)", 20, 18, 29, Color{232,238,248,255});
        DrawText("Hold left mouse: orbit | wheel: zoom | LEFT/RIGHT set analyzer A | UP/DOWN set analyzer B | P pause | R reset", 20, 54, 18, Color{164,183,210,255});

        std::ostringstream os;
        os << std::fixed << std::setprecision(3) << "A=" << a << " rad  B=" << b << " rad  corr~" << corr << "  P(same)~" << pSame;
        if (paused) os << "  [PAUSED]";
        DrawText(os.str().c_str(), 20, 82, 20, Color{126,224,255,255});
        DrawFPS(20, 110);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
