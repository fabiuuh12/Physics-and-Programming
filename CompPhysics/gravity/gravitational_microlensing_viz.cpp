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
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }
    *distance -= GetMouseWheelMove() * 0.7f;
    *distance = std::clamp(*distance, 5.0f, 55.0f);
    float cp = std::cos(*pitch);
    camera->position = Vector3Add(camera->target, {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw)
    });
}

float Magnification(float u) {
    u = std::max(u, 0.0001f);
    return (u * u + 2.0f) / (u * std::sqrt(u * u + 4.0f));
}
}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Gravitational Microlensing Event 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {9.5f, 6.2f, 10.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float camYaw = 0.85f;
    float camPitch = 0.35f;
    float camDistance = 19.0f;

    float lensMass = 1.0f;
    float impact = 0.34f;
    float crossingTime = 5.2f;
    float driftSpeed = 1.0f;
    bool paused = false;

    float t = -7.0f;
    std::deque<float> magHistory(360, 1.0f);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            lensMass = 1.0f;
            impact = 0.34f;
            crossingTime = 5.2f;
            driftSpeed = 1.0f;
            t = -7.0f;
            paused = false;
            magHistory.assign(360, 1.0f);
        }
        if (IsKeyDown(KEY_UP)) lensMass = std::min(5.0f, lensMass + 1.0f * GetFrameTime());
        if (IsKeyDown(KEY_DOWN)) lensMass = std::max(0.2f, lensMass - 1.0f * GetFrameTime());
        if (IsKeyDown(KEY_RIGHT_BRACKET)) impact = std::min(1.8f, impact + 0.7f * GetFrameTime());
        if (IsKeyDown(KEY_LEFT_BRACKET)) impact = std::max(0.03f, impact - 0.7f * GetFrameTime());
        if (IsKeyDown(KEY_RIGHT)) crossingTime = std::min(15.0f, crossingTime + 2.0f * GetFrameTime());
        if (IsKeyDown(KEY_LEFT)) crossingTime = std::max(1.2f, crossingTime - 2.0f * GetFrameTime());
        if (IsKeyDown(KEY_EQUAL)) driftSpeed = std::min(4.0f, driftSpeed + 1.0f * GetFrameTime());
        if (IsKeyDown(KEY_MINUS)) driftSpeed = std::max(0.2f, driftSpeed - 1.0f * GetFrameTime());

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        float thetaE = std::sqrt(lensMass);
        if (!paused) {
            t += GetFrameTime() * driftSpeed;
            if (t > 7.0f) t = -7.0f;
        }

        float ySky = thetaE * (t / crossingTime);
        float zSky = thetaE * impact;
        float u = std::sqrt((ySky * ySky + zSky * zSky) / (thetaE * thetaE));
        float mag = Magnification(u);
        mag = std::min(mag, 30.0f);
        magHistory.push_back(mag);
        if (magHistory.size() > 360) magHistory.pop_front();

        Vector3 source = {-8.0f, ySky, zSky};
        Vector3 lens = {0.0f, 0.0f, 0.0f};
        Vector3 observer = {8.0f, 0.0f, 0.0f};

        Vector3 skyDir = {0.0f, ySky, zSky};
        float skyLen = std::sqrt(ySky * ySky + zSky * zSky);
        Vector3 unitSky = (skyLen < 0.0001f) ? Vector3{0.0f, 1.0f, 0.0f} : Vector3Scale(skyDir, 1.0f / skyLen);
        float sqrtTerm = std::sqrt(u * u + 4.0f);
        float thetaPlus = 0.5f * (u + sqrtTerm) * thetaE;
        float thetaMinus = 0.5f * (u - sqrtTerm) * thetaE;
        Vector3 image1 = {0.0f, unitSky.y * thetaPlus, unitSky.z * thetaPlus};
        Vector3 image2 = {0.0f, unitSky.y * thetaMinus, unitSky.z * thetaMinus};

        BeginDrawing();
        ClearBackground(Color{6, 10, 18, 255});
        BeginMode3D(camera);

        DrawGrid(20, 1.0f);

        // Observer-lens-source axis.
        DrawLine3D({-9.0f, 0.0f, 0.0f}, {9.0f, 0.0f, 0.0f}, Fade(LIGHTGRAY, 0.35f));
        DrawSphere(lens, 0.35f, Color{255, 214, 120, 255});
        DrawSphere(source, 0.20f + 0.06f * std::min(mag, 8.0f), Color{170, 220, 255, 255});
        DrawSphere(observer, 0.28f, Color{220, 235, 255, 255});

        for (int i = 0; i < 80; ++i) {
            float a0 = (2.0f * kPi * i) / 80.0f;
            float a1 = (2.0f * kPi * (i + 1)) / 80.0f;
            Vector3 p0 = {0.0f, thetaE * std::cos(a0), thetaE * std::sin(a0)};
            Vector3 p1 = {0.0f, thetaE * std::cos(a1), thetaE * std::sin(a1)};
            DrawLine3D(p0, p1, Fade(Color{240, 220, 150, 255}, 0.45f));
        }

        DrawSphere(image1, 0.10f, Color{255, 235, 190, 255});
        DrawSphere(image2, 0.08f, Color{255, 200, 150, 255});

        DrawLine3D(source, image1, Fade(SKYBLUE, 0.7f));
        DrawLine3D(source, image2, Fade(SKYBLUE, 0.45f));
        DrawLine3D(image1, observer, Fade(Color{255, 220, 130, 255}, 0.72f));
        DrawLine3D(image2, observer, Fade(Color{255, 180, 120, 255}, 0.48f));

        EndMode3D();

        DrawRectangle(870, 516, 390, 238, Fade(Color{18, 26, 44, 255}, 0.92f));
        DrawText("Magnification Light Curve", 892, 536, 22, Color{220, 230, 244, 255});
        for (int i = 1; i < static_cast<int>(magHistory.size()); ++i) {
            float m0 = std::min(8.0f, magHistory[i - 1]);
            float m1 = std::min(8.0f, magHistory[i]);
            int x0 = 900 + i - 1;
            int x1 = 900 + i;
            int y0 = 732 - static_cast<int>(((m0 - 1.0f) / 7.0f) * 168.0f);
            int y1 = 732 - static_cast<int>(((m1 - 1.0f) / 7.0f) * 168.0f);
            DrawLine(x0, y0, x1, y1, Color{128, 240, 188, 255});
        }

        DrawText("Gravitational Microlensing Event", 20, 18, 30, Color{232, 238, 248, 255});
        DrawText("Mouse orbit | wheel zoom | Up/Down mass | [ ] impact | Left/Right crossing | +/- drift | P pause | R reset",
                 20, 54, 18, Color{164, 183, 210, 255});

        char status[230];
        std::snprintf(status, sizeof(status), "M_lens=%.2f  u0=%.2f  tE=%.2f  A=%.3f%s",
                      lensMass, impact, crossingTime, mag, paused ? " [PAUSED]" : "");
        DrawText(status, 20, 84, 20, Color{126, 224, 255, 255});
        DrawFPS(20, 112);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
