#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>

namespace {
constexpr int kScreenWidth = 1280;
constexpr int kScreenHeight = 820;
constexpr float kPi = 3.14159265358979323846f;

void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0034f;
        *pitch += d.y * 0.0034f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }
    *distance -= GetMouseWheelMove() * 0.7f;
    *distance = std::clamp(*distance, 6.0f, 55.0f);
    float cp = std::cos(*pitch);
    camera->position = Vector3Add(camera->target, {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw)
    });
}

void BuildOrthoBasis(Vector3 axis, Vector3* u, Vector3* v) {
    Vector3 ref = (std::abs(axis.y) < 0.98f) ? Vector3{0.0f, 1.0f, 0.0f} : Vector3{1.0f, 0.0f, 0.0f};
    *u = Vector3Normalize(Vector3CrossProduct(axis, ref));
    *v = Vector3Normalize(Vector3CrossProduct(axis, *u));
}
}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Pulsar Beam Sweep + Timing 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {10.0f, 6.0f, 10.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float camYaw = 0.82f;
    float camPitch = 0.35f;
    float camDistance = 18.0f;

    float spinHz = 2.8f;
    float tiltDeg = 28.0f;
    float beamWidthDeg = 8.0f;
    bool paused = false;
    float phase = 0.0f;
    std::deque<float> pulseHistory(360, 0.0f);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            spinHz = 2.8f;
            tiltDeg = 28.0f;
            beamWidthDeg = 8.0f;
            paused = false;
            phase = 0.0f;
            pulseHistory.assign(360, 0.0f);
        }
        if (IsKeyDown(KEY_EQUAL)) spinHz = std::min(18.0f, spinHz + 4.0f * GetFrameTime());
        if (IsKeyDown(KEY_MINUS)) spinHz = std::max(0.2f, spinHz - 4.0f * GetFrameTime());
        if (IsKeyDown(KEY_RIGHT_BRACKET)) tiltDeg = std::min(80.0f, tiltDeg + 35.0f * GetFrameTime());
        if (IsKeyDown(KEY_LEFT_BRACKET)) tiltDeg = std::max(2.0f, tiltDeg - 35.0f * GetFrameTime());
        if (IsKeyDown(KEY_PERIOD)) beamWidthDeg = std::min(28.0f, beamWidthDeg + 22.0f * GetFrameTime());
        if (IsKeyDown(KEY_COMMA)) beamWidthDeg = std::max(2.0f, beamWidthDeg - 22.0f * GetFrameTime());

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        if (!paused) phase += GetFrameTime() * spinHz * 2.0f * kPi;

        float tilt = tiltDeg * DEG2RAD;
        Vector3 spinAxis = {0.0f, 1.0f, 0.0f};
        Vector3 magnetic = Vector3Normalize({std::sin(tilt) * std::cos(phase), std::cos(tilt), std::sin(tilt) * std::sin(phase)});
        Vector3 antiMagnetic = Vector3Negate(magnetic);
        Vector3 observerDir = Vector3Normalize({1.0f, 0.1f, 0.0f});

        float sigma = beamWidthDeg * DEG2RAD;
        float a1 = std::acos(std::clamp(Vector3DotProduct(magnetic, observerDir), -1.0f, 1.0f));
        float a2 = std::acos(std::clamp(Vector3DotProduct(antiMagnetic, observerDir), -1.0f, 1.0f));
        float intensity = std::exp(-(a1 * a1) / (2.0f * sigma * sigma)) + std::exp(-(a2 * a2) / (2.0f * sigma * sigma));
        intensity = std::min(1.0f, intensity);
        pulseHistory.push_back(intensity);
        if (pulseHistory.size() > 360) pulseHistory.pop_front();

        BeginDrawing();
        ClearBackground(Color{6, 9, 16, 255});
        BeginMode3D(camera);

        DrawGrid(20, 1.0f);
        DrawSphere({0.0f, 0.0f, 0.0f}, 0.55f, Color{138, 204, 255, 255});
        DrawSphereWires({0.0f, 0.0f, 0.0f}, 0.7f, 10, 10, Fade(SKYBLUE, 0.45f));

        DrawLine3D({0.0f, -2.5f, 0.0f}, {0.0f, 2.5f, 0.0f}, Color{120, 170, 255, 230});  // Spin axis.
        DrawLine3D({0.0f, 0.0f, 0.0f}, Vector3Scale(magnetic, 3.2f), Color{255, 188, 130, 255});
        DrawLine3D({0.0f, 0.0f, 0.0f}, Vector3Scale(antiMagnetic, 3.2f), Color{255, 188, 130, 180});
        DrawLine3D({0.0f, 0.0f, 0.0f}, Vector3Scale(observerDir, 4.2f), Color{170, 255, 190, 255});

        Vector3 u{};
        Vector3 v{};
        BuildOrthoBasis(magnetic, &u, &v);
        float coneHalf = beamWidthDeg * DEG2RAD;
        for (int i = 0; i < 48; ++i) {
            float ang = (2.0f * kPi * i) / 48.0f;
            Vector3 rim = Vector3Normalize(Vector3Add(Vector3Scale(magnetic, std::cos(coneHalf)),
                                                      Vector3Scale(Vector3Add(Vector3Scale(u, std::cos(ang)), Vector3Scale(v, std::sin(ang))),
                                                                   std::sin(coneHalf))));
            DrawLine3D(Vector3Scale(magnetic, 0.45f), Vector3Scale(rim, 5.2f), Fade(Color{255, 210, 138, 255}, 0.55f));
        }
        BuildOrthoBasis(antiMagnetic, &u, &v);
        for (int i = 0; i < 48; ++i) {
            float ang = (2.0f * kPi * i) / 48.0f;
            Vector3 rim = Vector3Normalize(Vector3Add(Vector3Scale(antiMagnetic, std::cos(coneHalf)),
                                                      Vector3Scale(Vector3Add(Vector3Scale(u, std::cos(ang)), Vector3Scale(v, std::sin(ang))),
                                                                   std::sin(coneHalf))));
            DrawLine3D(Vector3Scale(antiMagnetic, 0.45f), Vector3Scale(rim, 5.2f), Fade(Color{255, 195, 132, 255}, 0.34f));
        }
        EndMode3D();

        DrawRectangle(874, 516, 384, 236, Fade(Color{18, 26, 42, 255}, 0.92f));
        DrawText("Pulse Profile (observer)", 894, 536, 22, Color{220, 230, 244, 255});
        for (int i = 1; i < static_cast<int>(pulseHistory.size()); ++i) {
            int x0 = 900 + i - 1;
            int x1 = 900 + i;
            int y0 = 730 - static_cast<int>(pulseHistory[i - 1] * 162.0f);
            int y1 = 730 - static_cast<int>(pulseHistory[i] * 162.0f);
            DrawLine(x0, y0, x1, y1, Color{130, 240, 186, 255});
        }

        DrawText("Pulsar Beam Sweep + Observer Timing", 20, 18, 30, Color{232, 238, 248, 255});
        DrawText("Mouse orbit | wheel zoom | +/- spin Hz | [ ] tilt | , . beam width | P pause | R reset",
                 20, 54, 18, Color{164, 183, 210, 255});
        char status[220];
        std::snprintf(status, sizeof(status), "spin=%.2f Hz  tilt=%.1f deg  beam=%.1f deg  pulse=%.3f%s",
                      spinHz, tiltDeg, beamWidthDeg, intensity, paused ? " [PAUSED]" : "");
        DrawText(status, 20, 84, 20, Color{126, 224, 255, 255});
        DrawFPS(20, 112);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
