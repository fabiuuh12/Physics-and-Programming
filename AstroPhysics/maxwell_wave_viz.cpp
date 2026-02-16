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
    *distance = std::clamp(*distance, 4.0f, 32.0f);
    float cp = std::cos(*pitch);
    c->position = Vector3Add(c->target, {*distance * cp * std::cos(*yaw), *distance * std::sin(*pitch), *distance * cp * std::sin(*yaw)});
}

void DrawArrow(Vector3 a, Vector3 b, Color c) {
    DrawLine3D(a, b, c);
    Vector3 d = Vector3Normalize(Vector3Subtract(b, a));
    Vector3 s = Vector3Normalize(Vector3CrossProduct(d, {0.0f, 1.0f, 0.0f}));
    if (Vector3Length(s) < 1e-4f) s = {1.0f, 0.0f, 0.0f};
    DrawLine3D(b, Vector3Add(b, Vector3Add(Vector3Scale(d, -0.18f), Vector3Scale(s, 0.09f))), c);
    DrawLine3D(b, Vector3Add(b, Vector3Add(Vector3Scale(d, -0.18f), Vector3Scale(s, -0.09f))), c);
}
} // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Maxwell Electromagnetic Wave 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {9.0f, 5.4f, 9.0f};
    camera.target = {0.0f, 0.8f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.82f, camPitch = 0.34f, camDistance = 14.0f;

    float amp = 1.0f;
    float k = 1.2f;
    float omega = 2.0f;
    bool paused = false;
    float t = 0.0f;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) { amp = 1.0f; k = 1.2f; omega = 2.0f; paused = false; t = 0.0f; }
        if (IsKeyPressed(KEY_LEFT_BRACKET)) amp = std::max(0.2f, amp - 0.1f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) amp = std::min(2.0f, amp + 0.1f);
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) omega = std::max(0.3f, omega - 0.1f);
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) omega = std::min(6.0f, omega + 0.1f);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);
        if (!paused) t += GetFrameTime();

        BeginDrawing();
        ClearBackground(Color{6, 9, 16, 255});
        BeginMode3D(camera);

        DrawGrid(28, 0.5f);
        DrawLine3D({-6.0f, 0.0f, 0.0f}, {6.0f, 0.0f, 0.0f}, Color{120, 140, 180, 130});

        const int N = 60;
        for (int i = 0; i < N; ++i) {
            float x = -5.5f + 11.0f * static_cast<float>(i) / (N - 1);
            float ph = k * x - omega * t;
            float Ey = amp * std::sin(ph);
            float Bz = Ey;

            Vector3 base = {x, 0.0f, 0.0f};
            DrawArrow(base, {x, Ey, 0.0f}, Color{120, 220, 255, 255});
            DrawArrow(base, {x, 0.0f, Bz}, Color{255, 180, 120, 255});
        }

        EndMode3D();

        DrawText("Maxwell Wave: E and B Orthogonal Fields", 20, 18, 29, Color{232, 238, 248, 255});
        DrawText("Hold left mouse: orbit | wheel: zoom | [ ] amplitude | +/- omega | P pause | R reset", 20, 54, 18, Color{164, 183, 210, 255});

        std::ostringstream os;
        os << std::fixed << std::setprecision(2) << "A=" << amp << "  k=" << k << "  omega=" << omega;
        if (paused) os << "  [PAUSED]";
        DrawText(os.str().c_str(), 20, 82, 20, Color{126, 224, 255, 255});
        DrawText("Blue arrows: Electric field E  |  Orange arrows: Magnetic field B", 20, 110, 18, Color{190, 205, 225, 255});
        DrawFPS(20, 138);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
