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
    *distance = std::clamp(*distance, 4.0f, 26.0f);

    float cp = std::cos(*pitch);
    Vector3 offset = {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}

float IntensityFromPhase(float phaseDiff) {
    return 0.5f * (1.0f + std::cos(phaseDiff));
}

std::string Hud(float strainAmp, float gwFreq, float lX, float lZ, bool paused) {
    std::ostringstream os;
    os << std::scientific << std::setprecision(2)
       << "strain=" << strainAmp
       << "  freq=" << std::fixed << std::setprecision(2) << gwFreq << "Hz"
       << "  Lx=" << std::setprecision(3) << lX
       << "  Lz=" << lZ;
    if (paused) os << "  [PAUSED]";
    return os.str();
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Interferometer GW Visualization 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {6.5f, 4.2f, 6.8f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.82f;
    float camPitch = 0.33f;
    float camDistance = 11.0f;

    const float L0 = 3.2f;
    const float lambda = 0.12f;

    float strainAmp = 6.0e-3f;
    float gwFreq = 0.75f;
    float timeScale = 1.0f;
    bool paused = false;
    float t = 0.0f;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            strainAmp = 6.0e-3f;
            gwFreq = 0.75f;
            timeScale = 1.0f;
            t = 0.0f;
        }

        if (IsKeyPressed(KEY_LEFT_BRACKET)) strainAmp = std::max(0.0e-3f, strainAmp - 0.5e-3f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) strainAmp = std::min(20.0e-3f, strainAmp + 0.5e-3f);
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) gwFreq = std::max(0.1f, gwFreq - 0.05f);
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) gwFreq = std::min(3.0f, gwFreq + 0.05f);
        if (IsKeyPressed(KEY_COMMA)) timeScale = std::max(0.2f, timeScale - 0.2f);
        if (IsKeyPressed(KEY_PERIOD)) timeScale = std::min(6.0f, timeScale + 0.2f);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        if (!paused) {
            t += GetFrameTime() * timeScale;
        }

        float h = strainAmp * std::sin(2.0f * PI * gwFreq * t);
        float Lx = L0 * (1.0f + h);
        float Lz = L0 * (1.0f - h);

        float phaseDiff = (2.0f * PI / lambda) * (2.0f * (Lx - Lz));
        float intensity = IntensityFromPhase(phaseDiff);

        Vector3 splitter = {0.0f, 0.2f, 0.0f};
        Vector3 xMirror = {Lx, 0.2f, 0.0f};
        Vector3 zMirror = {0.0f, 0.2f, Lz};
        Vector3 laserIn = {-1.6f, 0.2f, 0.0f};
        Vector3 detector = {0.0f, 0.2f, -1.6f};

        BeginDrawing();
        ClearBackground(Color{6, 9, 17, 255});

        BeginMode3D(camera);

        DrawGrid(24, 0.5f);

        DrawCube({0.0f, 0.0f, 0.0f}, 8.0f, 0.04f, 8.0f, Color{30, 36, 48, 255});

        DrawCube(splitter, 0.16f, 0.16f, 0.16f, Color{220, 220, 230, 255});
        DrawCube(xMirror, 0.14f, 0.32f, 0.42f, Color{170, 210, 255, 255});
        DrawCube(zMirror, 0.42f, 0.32f, 0.14f, Color{170, 210, 255, 255});
        DrawCube(detector, 0.20f, 0.20f, 0.20f, Color{255, 210, 130, 255});

        DrawLine3D(laserIn, splitter, Color{255, 90, 90, 255});

        DrawLine3D(splitter, xMirror, Color{255, 70, 70, 220});
        DrawLine3D(xMirror, splitter, Color{255, 130, 90, 180});

        DrawLine3D(splitter, zMirror, Color{255, 70, 70, 220});
        DrawLine3D(zMirror, splitter, Color{255, 130, 90, 180});

        DrawLine3D(splitter, detector, Color{255, static_cast<unsigned char>(80 + 160 * intensity), 80, 255});

        for (int i = 1; i <= 5; ++i) {
            float r = 0.7f * i;
            Color c = Color{120, 170, 255, static_cast<unsigned char>(80 - i * 10)};
            int segs = 80;
            for (int j = 0; j < segs; ++j) {
                float a0 = 2.0f * PI * static_cast<float>(j) / static_cast<float>(segs);
                float a1 = 2.0f * PI * static_cast<float>(j + 1) / static_cast<float>(segs);
                DrawLine3D(
                    {r * std::cos(a0), 0.0f, r * std::sin(a0)},
                    {r * std::cos(a1), 0.0f, r * std::sin(a1)},
                    c
                );
            }
        }

        EndMode3D();

        DrawText("Gravitational-Wave Interferometer (L-shaped)", 20, 18, 29, Color{232, 238, 248, 255});
        DrawText("Hold left mouse: orbit | wheel: zoom | [ ] strain | +/- GW freq | , . time scale | P pause | R reset", 20, 54, 18, Color{164, 183, 210, 255});

        std::string hud = Hud(strainAmp, gwFreq, Lx, Lz, paused);
        DrawText(hud.c_str(), 20, 82, 20, Color{126, 224, 255, 255});

        DrawText("Detector intensity", 20, 114, 18, Color{210, 220, 230, 255});
        DrawRectangle(20, 138, 340, 24, Color{35, 45, 65, 255});
        DrawRectangle(20, 138, static_cast<int>(340.0f * intensity), 24, Color{255, static_cast<unsigned char>(100 + 140 * intensity), 90, 255});

        DrawFPS(20, 172);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
