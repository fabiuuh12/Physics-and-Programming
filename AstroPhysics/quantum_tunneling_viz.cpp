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
    *distance = std::clamp(*distance, 4.5f, 32.0f);

    float cp = std::cos(*pitch);
    Vector3 offset = {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}

float Gaussian(float x, float center, float sigma) {
    float u = (x - center) / sigma;
    return std::exp(-0.5f * u * u);
}

std::string Hud(float barrierH, float packetE, float transP, bool paused) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(2)
       << "Barrier=" << barrierH
       << "  PacketE=" << packetE
       << "  TransProb~" << transP;
    if (paused) os << "  [PAUSED]";
    return os.str();
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Quantum Tunneling 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {9.0f, 5.6f, 9.2f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.86f;
    float camPitch = 0.35f;
    float camDistance = 14.0f;

    float barrierCenter = 0.0f;
    float barrierWidth = 1.0f;
    float barrierHeight = 1.15f;

    float packetEnergy = 0.80f;
    float packetCenter = -5.8f;
    float packetSpeed = 1.25f;
    float sigma = 0.95f;

    float timeScale = 1.0f;
    bool paused = false;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            barrierHeight = 1.15f;
            packetEnergy = 0.80f;
            packetCenter = -5.8f;
            packetSpeed = 1.25f;
            timeScale = 1.0f;
            paused = false;
        }

        if (IsKeyPressed(KEY_LEFT_BRACKET)) barrierHeight = std::max(0.35f, barrierHeight - 0.05f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) barrierHeight = std::min(2.5f, barrierHeight + 0.05f);
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) packetEnergy = std::max(0.25f, packetEnergy - 0.05f);
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) packetEnergy = std::min(2.4f, packetEnergy + 0.05f);
        if (IsKeyPressed(KEY_COMMA)) timeScale = std::max(0.2f, timeScale - 0.2f);
        if (IsKeyPressed(KEY_PERIOD)) timeScale = std::min(5.0f, timeScale + 0.2f);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        if (!paused) {
            packetCenter += packetSpeed * GetFrameTime() * timeScale;
            if (packetCenter > 8.0f) {
                packetCenter = -5.8f;
            }
        }

        float decay = std::sqrt(std::max(0.0f, barrierHeight - packetEnergy)) * barrierWidth;
        float transProb = std::exp(-2.0f * decay);

        BeginDrawing();
        ClearBackground(Color{6, 9, 17, 255});

        BeginMode3D(camera);

        DrawCube({barrierCenter, barrierHeight * 0.5f, 0.0f}, barrierWidth, barrierHeight, 3.6f, Color{200, 120, 130, 120});
        DrawCubeWires({barrierCenter, barrierHeight * 0.5f, 0.0f}, barrierWidth, barrierHeight, 3.6f, Color{255, 170, 180, 200});

        for (int i = 0; i < 96; ++i) {
            float x = -8.0f + 16.0f * static_cast<float>(i) / 95.0f;

            float incident = Gaussian(x, packetCenter, sigma);
            float reflected = (1.0f - transProb) * Gaussian(x, -packetCenter - 0.8f, sigma * 1.08f);
            float transmitted = transProb * Gaussian(x, packetCenter - 1.0f, sigma * 1.15f);

            float envelope = 0.0f;
            if (x < barrierCenter - barrierWidth * 0.5f) {
                envelope = incident + reflected;
            } else if (x > barrierCenter + barrierWidth * 0.5f) {
                envelope = transmitted;
            } else {
                envelope = incident * std::exp(-1.9f * std::fabs(x - barrierCenter));
            }

            float y = 0.05f + 1.8f * envelope;
            float phase = 8.0f * x - 4.0f * packetCenter;
            float z = 0.45f * std::sin(phase) * envelope;

            Color c = (x > barrierCenter + barrierWidth * 0.5f)
                ? Color{110, 230, 255, 220}
                : Color{160, 170, 255, 220};

            DrawSphere({x, y, z}, 0.045f + 0.05f * envelope, c);
        }

        for (int i = 0; i < 100; ++i) {
            float x0 = -8.0f + 16.0f * static_cast<float>(i) / 100.0f;
            float x1 = -8.0f + 16.0f * static_cast<float>(i + 1) / 100.0f;

            float y0 = 0.02f + 0.25f * ((x0 < barrierCenter) ? 0.0f : transProb);
            float y1 = 0.02f + 0.25f * ((x1 < barrierCenter) ? 0.0f : transProb);
            DrawLine3D({x0, y0, -1.8f}, {x1, y1, -1.8f}, Color{255, 210, 120, 140});
        }

        EndMode3D();

        DrawText("Quantum Tunneling (Wave Packet vs Barrier)", 20, 18, 29, Color{232, 238, 248, 255});
        DrawText("Hold left mouse: orbit | wheel: zoom | [ ] barrier | +/- packet energy | , . time | P pause | R reset", 20, 54, 18, Color{164, 183, 210, 255});

        std::string hud = Hud(barrierHeight, packetEnergy, transProb, paused);
        DrawText(hud.c_str(), 20, 82, 20, Color{126, 224, 255, 255});

        DrawFPS(20, 112);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
