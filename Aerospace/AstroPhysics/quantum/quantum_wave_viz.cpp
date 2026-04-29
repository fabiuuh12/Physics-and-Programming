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

void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }

    *distance -= GetMouseWheelMove() * 0.6f;
    *distance = std::clamp(*distance, 4.5f, 35.0f);

    float cp = std::cos(*pitch);
    Vector3 offset = {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}

float PsiReal(float x, float z, float t, float kx, float kz, float omega, float phase) {
    return std::cos(kx * x + kz * z - omega * t + phase);
}

float Envelope(float x, float z, float sigma) {
    float r2 = x * x + z * z;
    return std::exp(-r2 / (2.0f * sigma * sigma));
}

std::string Hud(float amp1, float amp2, float omega, bool paused) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(2)
       << "A1=" << amp1
       << "  A2=" << amp2
       << "  omega=" << omega;
    if (paused) os << "  [PAUSED]";
    return os.str();
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Quantum Wave Superposition 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.5f, 5.6f, 9.2f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.86f;
    float camPitch = 0.34f;
    float camDistance = 14.5f;

    float amp1 = 1.0f;
    float amp2 = 0.85f;
    float omega = 2.2f;
    float phase2 = 0.8f;
    float timeScale = 1.0f;

    bool paused = false;
    float t = 0.0f;

    constexpr int NX = 86;
    constexpr int NZ = 86;
    constexpr float XMIN = -6.0f;
    constexpr float XMAX = 6.0f;
    constexpr float ZMIN = -6.0f;
    constexpr float ZMAX = 6.0f;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            amp1 = 1.0f;
            amp2 = 0.85f;
            omega = 2.2f;
            phase2 = 0.8f;
            timeScale = 1.0f;
            t = 0.0f;
        }

        if (IsKeyPressed(KEY_LEFT_BRACKET)) amp2 = std::max(0.0f, amp2 - 0.05f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) amp2 = std::min(2.0f, amp2 + 0.05f);
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) omega = std::max(0.2f, omega - 0.1f);
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) omega = std::min(8.0f, omega + 0.1f);
        if (IsKeyPressed(KEY_COMMA)) phase2 -= 0.08f;
        if (IsKeyPressed(KEY_PERIOD)) phase2 += 0.08f;
        if (IsKeyPressed(KEY_SEMICOLON)) timeScale = std::max(0.2f, timeScale - 0.2f);
        if (IsKeyPressed(KEY_APOSTROPHE)) timeScale = std::min(5.0f, timeScale + 0.2f);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        if (!paused) {
            t += GetFrameTime() * timeScale;
        }

        BeginDrawing();
        ClearBackground(Color{6, 9, 17, 255});

        BeginMode3D(camera);

        for (int ix = 0; ix < NX - 1; ++ix) {
            for (int iz = 0; iz < NZ - 1; ++iz) {
                float x0 = XMIN + (XMAX - XMIN) * static_cast<float>(ix) / static_cast<float>(NX - 1);
                float x1 = XMIN + (XMAX - XMIN) * static_cast<float>(ix + 1) / static_cast<float>(NX - 1);
                float z0 = ZMIN + (ZMAX - ZMIN) * static_cast<float>(iz) / static_cast<float>(NZ - 1);
                float z1 = ZMIN + (ZMAX - ZMIN) * static_cast<float>(iz + 1) / static_cast<float>(NZ - 1);

                float e00 = Envelope(x0, z0, 4.9f);
                float e10 = Envelope(x1, z0, 4.9f);
                float e01 = Envelope(x0, z1, 4.9f);

                float psi00 = e00 * (amp1 * PsiReal(x0, z0, t, 1.3f, 0.5f, omega, 0.0f) + amp2 * PsiReal(x0, z0, t, -0.7f, 1.1f, omega * 1.1f, phase2));
                float psi10 = e10 * (amp1 * PsiReal(x1, z0, t, 1.3f, 0.5f, omega, 0.0f) + amp2 * PsiReal(x1, z0, t, -0.7f, 1.1f, omega * 1.1f, phase2));
                float psi01 = e01 * (amp1 * PsiReal(x0, z1, t, 1.3f, 0.5f, omega, 0.0f) + amp2 * PsiReal(x0, z1, t, -0.7f, 1.1f, omega * 1.1f, phase2));

                Vector3 p00 = {x0, 0.9f * psi00, z0};
                Vector3 p10 = {x1, 0.9f * psi10, z0};
                Vector3 p01 = {x0, 0.9f * psi01, z1};

                float prob = std::clamp(0.28f * psi00 * psi00, 0.0f, 1.0f);
                Color c = Color{
                    static_cast<unsigned char>(70 + 120 * prob),
                    static_cast<unsigned char>(110 + 130 * prob),
                    static_cast<unsigned char>(180 + 70 * prob),
                    static_cast<unsigned char>(60 + 110 * prob)
                };

                DrawTriangle3D(p00, p10, p01, c);
                DrawLine3D(p00, p10, Color{90, 130, 200, 70});
                DrawLine3D(p00, p01, Color{90, 130, 200, 70});
            }
        }

        for (int i = 0; i < 140; ++i) {
            float x = -5.8f + 11.6f * static_cast<float>(i) / 139.0f;
            float z = -4.8f;
            float env = Envelope(x, z, 4.9f);
            float psi = env * (amp1 * PsiReal(x, z, t, 1.3f, 0.5f, omega, 0.0f) + amp2 * PsiReal(x, z, t, -0.7f, 1.1f, omega * 1.1f, phase2));
            float p = psi * psi;
            DrawSphere({x, 0.05f + 0.35f * p, z}, 0.03f + 0.05f * p, Color{255, 200, 120, 190});
        }

        EndMode3D();

        DrawText("Quantum Wave Superposition (3D)", 20, 18, 29, Color{232, 238, 248, 255});
        DrawText("Hold left mouse: orbit | wheel: zoom | [ ] A2 | +/- omega | , . phase | ; ' time | P pause | R reset", 20, 54, 18, Color{164, 183, 210, 255});

        std::string hud = Hud(amp1, amp2, omega, paused);
        DrawText(hud.c_str(), 20, 82, 20, Color{126, 224, 255, 255});

        DrawFPS(20, 112);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
