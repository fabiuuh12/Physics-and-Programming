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
constexpr float kMass = 1.0f;

void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }

    *distance -= GetMouseWheelMove() * 0.6f;
    *distance = std::clamp(*distance, 3.5f, 28.0f);

    const float cp = std::cos(*pitch);
    const Vector3 offset = {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}

std::string Hud(float r, float omega, float I, float L, bool paused, bool autoMode) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(3)
       << "r=" << r
       << "  omega=" << omega
       << "  I=" << I
       << "  L=" << L;
    if (autoMode) os << "  [AUTO]";
    if (paused) os << "  [PAUSED]";
    return os.str();
}

void DrawArrow(const Vector3& from, const Vector3& to, Color color) {
    DrawLine3D(from, to, color);
    Vector3 dir = Vector3Normalize(Vector3Subtract(to, from));
    Vector3 side = Vector3Normalize(Vector3CrossProduct(dir, {0.0f, 1.0f, 0.0f}));
    if (Vector3Length(side) < 1e-4f) {
        side = {1.0f, 0.0f, 0.0f};
    }
    Vector3 back = Vector3Scale(dir, -0.25f);
    DrawLine3D(to, Vector3Add(Vector3Add(to, back), Vector3Scale(side, 0.12f)), color);
    DrawLine3D(to, Vector3Add(Vector3Add(to, back), Vector3Scale(side, -0.12f)), color);
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Angular Momentum Conservation 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {7.0f, 4.5f, 8.0f};
    camera.target = {0.0f, 0.6f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.82f;
    float camPitch = 0.33f;
    float camDistance = 12.0f;

    float r = 2.2f;
    float angle = 0.0f;
    float L = 8.0f;

    bool paused = false;
    bool autoMode = true;
    float t = 0.0f;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_A)) autoMode = !autoMode;
        if (IsKeyPressed(KEY_R)) {
            r = 2.2f;
            angle = 0.0f;
            L = 8.0f;
            autoMode = true;
            paused = false;
            t = 0.0f;
        }

        if (!autoMode) {
            if (IsKeyDown(KEY_LEFT_BRACKET)) r -= 1.25f * GetFrameTime();
            if (IsKeyDown(KEY_RIGHT_BRACKET)) r += 1.25f * GetFrameTime();
        }

        r = std::clamp(r, 0.45f, 2.8f);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        if (!paused) {
            t += GetFrameTime();

            if (autoMode) {
                r = 1.6f + 1.05f * std::sin(0.95f * t);
                r = std::clamp(r, 0.45f, 2.8f);
            }

            float I = 2.0f * kMass * r * r;
            float omega = L / std::max(0.001f, I);
            angle += omega * GetFrameTime();
        }

        const float I = 2.0f * kMass * r * r;
        const float omega = L / std::max(0.001f, I);

        Vector3 p1 = {r * std::cos(angle), 0.6f, r * std::sin(angle)};
        Vector3 p2 = {-p1.x, 0.6f, -p1.z};

        BeginDrawing();
        ClearBackground(Color{6, 9, 17, 255});

        BeginMode3D(camera);

        DrawCylinder({0.0f, 0.6f, 0.0f}, 0.10f, 0.10f, 0.8f, 24, Color{180, 190, 215, 220});
        DrawLine3D(p1, p2, Color{180, 215, 255, 255});

        DrawSphere(p1, 0.18f, Color{255, 180, 120, 255});
        DrawSphere(p2, 0.18f, Color{120, 210, 255, 255});

        DrawArrow({0.0f, 0.1f, 0.0f}, {0.0f, 0.1f + 0.45f + 0.3f * omega, 0.0f}, Color{255, 210, 120, 255});
        DrawArrow({0.35f, 0.1f, 0.0f}, {0.35f, 0.1f + 0.45f + 0.08f * L, 0.0f}, Color{130, 220, 255, 255});

        EndMode3D();

        DrawText("Angular Momentum Conservation (L = I * omega)", 20, 18, 29, Color{232, 238, 248, 255});
        DrawText("Hold left mouse: orbit | wheel: zoom | A auto/manual radius | [ ] radius (manual) | P pause | R reset", 20, 54, 18, Color{164, 183, 210, 255});

        std::string hud = Hud(r, omega, I, L, paused, autoMode);
        DrawText(hud.c_str(), 20, 82, 20, Color{126, 224, 255, 255});
        DrawText("Yellow arrow: omega  |  Blue arrow: angular momentum L (constant)", 20, 110, 18, Color{185, 198, 215, 255});

        DrawFPS(20, 138);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
