#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kScreenWidth = 1280;
constexpr int kScreenHeight = 820;

struct Traveler {
    float x;
    float z;
    float coupling;
    float baseSpeed;
    Color color;
    std::string label;
};

void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }

    *distance -= GetMouseWheelMove() * 0.6f;
    *distance = std::clamp(*distance, 6.0f, 30.0f);

    float cp = std::cos(*pitch);
    Vector3 offset = {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}

float HiggsFieldValue(float x, float z, float t, float vev, bool pulseActive, float pulseTime) {
    float background = 0.06f * std::sin(0.8f * x + 1.1f * z - 1.9f * t);
    float pulse = 0.0f;
    if (pulseActive) {
        float r = std::sqrt(x * x + z * z);
        float envelope = 0.65f * std::exp(-0.9f * pulseTime) * std::exp(-0.45f * r);
        pulse = envelope * std::sin(8.5f * r - 6.0f * pulseTime);
    }
    return std::max(0.1f, vev + background + pulse);
}

std::string HudLine(float strongCoupling, float masslessSpeed, float massiveSpeed, bool paused, bool pulseActive) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(2)
       << "strong coupling=" << strongCoupling
       << "  speed(gamma-like)=" << masslessSpeed
       << "  speed(W-like)=" << massiveSpeed;
    if (pulseActive) os << "  [field excitation active]";
    if (paused) os << "  [PAUSED]";
    return os.str();
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Higgs Field Visualization 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.8f, 6.0f, 9.5f};
    camera.target = {0.0f, 0.8f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.84f;
    float camPitch = 0.40f;
    float camDistance = 14.0f;

    bool paused = false;
    bool pulseActive = false;
    float t = 0.0f;
    float pulseTime = 0.0f;

    float strongCoupling = 1.05f;
    const float vev = 1.0f;
    const float trackMinX = -5.3f;
    const float trackMaxX = 5.3f;

    std::vector<Traveler> travelers = {
        {trackMinX, -1.2f, 0.0f, 2.9f, Color{125, 215, 255, 255}, "gamma-like (no Higgs coupling)"},
        {trackMinX, 1.2f, strongCoupling, 2.9f, Color{255, 175, 105, 255}, "W-like (strong Higgs coupling)"},
    };

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_SPACE)) {
            pulseActive = true;
            pulseTime = 0.0f;
        }
        if (IsKeyPressed(KEY_R)) {
            paused = false;
            pulseActive = false;
            pulseTime = 0.0f;
            t = 0.0f;
            strongCoupling = 1.05f;
            travelers[0].x = trackMinX;
            travelers[1].x = trackMinX;
        }
        if (IsKeyPressed(KEY_UP)) strongCoupling = std::min(2.2f, strongCoupling + 0.1f);
        if (IsKeyPressed(KEY_DOWN)) strongCoupling = std::max(0.2f, strongCoupling - 0.1f);
        travelers[1].coupling = strongCoupling;

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        float gammaSpeed = travelers[0].baseSpeed;
        float massiveSpeed = travelers[1].baseSpeed;
        if (!paused) {
            float dt = GetFrameTime();
            t += dt;
            if (pulseActive) {
                pulseTime += dt;
                if (pulseTime > 4.2f) pulseActive = false;
            }

            for (Traveler& traveler : travelers) {
                float field = HiggsFieldValue(traveler.x, traveler.z, t, vev, pulseActive, pulseTime);
                float effectiveMass = traveler.coupling * field;
                float speed = traveler.baseSpeed / (1.0f + 2.2f * effectiveMass);
                traveler.x += speed * dt;
                if (traveler.x > trackMaxX) traveler.x = trackMinX;
                if (traveler.coupling < 0.01f) gammaSpeed = speed;
                else massiveSpeed = speed;
            }
        }

        BeginDrawing();
        ClearBackground(Color{8, 10, 18, 255});

        BeginMode3D(camera);

        DrawPlane({0.0f, 0.0f, 0.0f}, {13.0f, 13.0f}, Color{20, 28, 38, 255});

        const int fieldN = 24;
        const float span = 5.4f;
        for (int iz = 0; iz <= fieldN; ++iz) {
            for (int ix = 0; ix <= fieldN; ++ix) {
                float x = -span + 2.0f * span * static_cast<float>(ix) / static_cast<float>(fieldN);
                float z = -span + 2.0f * span * static_cast<float>(iz) / static_cast<float>(fieldN);
                float phi = HiggsFieldValue(x, z, t, vev, pulseActive, pulseTime);
                float norm = std::clamp((phi - 0.45f) / 1.35f, 0.0f, 1.0f);
                float y = 0.12f + 0.30f * phi;
                float r = 0.04f + 0.03f * norm;
                Color c = {
                    static_cast<unsigned char>(70 + 95 * norm),
                    static_cast<unsigned char>(120 + 80 * norm),
                    static_cast<unsigned char>(210 + 35 * (1.0f - norm)),
                    185
                };
                DrawSphere({x, y, z}, r, c);
            }
        }

        DrawLine3D({trackMinX, 0.10f, -1.2f}, {trackMaxX, 0.10f, -1.2f}, Color{110, 180, 230, 150});
        DrawLine3D({trackMinX, 0.10f, 1.2f}, {trackMaxX, 0.10f, 1.2f}, Color{230, 160, 110, 150});

        for (const Traveler& traveler : travelers) {
            float phi = HiggsFieldValue(traveler.x, traveler.z, t, vev, pulseActive, pulseTime);
            float y = 0.18f + 0.30f * phi;
            DrawSphere({traveler.x, y, traveler.z}, 0.16f, traveler.color);
            DrawSphere({traveler.x, y, traveler.z}, 0.23f, Color{traveler.color.r, traveler.color.g, traveler.color.b, 40});
        }

        DrawCubeWires({0.0f, 0.8f, 0.0f}, 11.0f, 1.8f, 11.0f, Color{120, 160, 210, 80});
        EndMode3D();

        DrawText("Higgs Field (Conceptual)", 20, 18, 30, Color{235, 240, 252, 255});
        DrawText("Non-zero field fills space. Particles that couple to it move as if they have inertia (mass).", 20, 54, 19, Color{168, 186, 214, 255});
        DrawText("Mouse drag: orbit | wheel: zoom | UP/DOWN: coupling | SPACE: excite field | P: pause | R: reset", 20, 80, 18, Color{168, 186, 214, 255});

        DrawText("blue: no coupling (stays fast)", 20, 110, 19, Color{125, 215, 255, 255});
        DrawText("orange: strong coupling (slower = larger effective mass)", 20, 134, 19, Color{255, 175, 105, 255});

        std::string hud = HudLine(strongCoupling, gammaSpeed, massiveSpeed, paused, pulseActive);
        DrawText(hud.c_str(), 20, 164, 20, Color{255, 220, 130, 255});
        DrawFPS(20, 194);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
