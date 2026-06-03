#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <vector>

namespace {
constexpr int kScreenWidth = 1280;
constexpr int kScreenHeight = 820;
constexpr float kPi = 3.14159265358979323846f;

struct EjectaParticle {
    Vector3 dir;
    float speed;
    float phase;
};

void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0034f;
        *pitch += d.y * 0.0034f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }
    *distance -= GetMouseWheelMove() * 0.8f;
    *distance = std::clamp(*distance, 6.0f, 65.0f);
    float cp = std::cos(*pitch);
    camera->position = Vector3Add(camera->target, {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw)
    });
}

std::vector<EjectaParticle> BuildEjectaField(int n) {
    std::vector<EjectaParticle> particles;
    particles.reserve(n);
    for (int i = 0; i < n; ++i) {
        float t = (i + 0.5f) / static_cast<float>(n);
        float y = 1.0f - 2.0f * t;
        float r = std::sqrt(std::max(0.0f, 1.0f - y * y));
        float phi = kPi * (3.0f - std::sqrt(5.0f)) * static_cast<float>(i);
        Vector3 d = {r * std::cos(phi), y, r * std::sin(phi)};
        float lat = std::abs(d.y);
        float speed = 1.4f + 0.8f * (1.0f - lat);  // Slightly faster near the orbital plane.
        particles.push_back({d, speed, 0.17f * static_cast<float>(i)});
    }
    return particles;
}
}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Neutron Star Merger + Kilonova 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {12.0f, 8.0f, 12.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float camYaw = 0.82f;
    float camPitch = 0.35f;
    float camDistance = 22.0f;

    float m1 = 1.45f;
    float m2 = 1.30f;
    float inspiralRate = 0.27f;
    float spin = 0.8f;
    bool paused = false;

    float sep = 6.0f;
    float phase = 0.0f;
    bool merged = false;
    float postMergerTime = 0.0f;
    float simTime = 0.0f;
    std::deque<float> luminosityHistory(360, 0.04f);
    std::vector<EjectaParticle> ejecta = BuildEjectaField(950);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            m1 = 1.45f;
            m2 = 1.30f;
            inspiralRate = 0.27f;
            spin = 0.8f;
            sep = 6.0f;
            phase = 0.0f;
            merged = false;
            postMergerTime = 0.0f;
            simTime = 0.0f;
            paused = false;
            luminosityHistory.assign(360, 0.04f);
        }
        if (IsKeyDown(KEY_UP)) m2 = std::min(2.4f, m2 + 0.5f * GetFrameTime());
        if (IsKeyDown(KEY_DOWN)) m2 = std::max(0.8f, m2 - 0.5f * GetFrameTime());
        if (IsKeyDown(KEY_RIGHT)) inspiralRate = std::min(0.9f, inspiralRate + 0.25f * GetFrameTime());
        if (IsKeyDown(KEY_LEFT)) inspiralRate = std::max(0.05f, inspiralRate - 0.25f * GetFrameTime());
        if (IsKeyDown(KEY_RIGHT_BRACKET)) spin = std::min(2.2f, spin + 0.8f * GetFrameTime());
        if (IsKeyDown(KEY_LEFT_BRACKET)) spin = std::max(0.1f, spin - 0.8f * GetFrameTime());

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        float luminosity = 0.0f;
        if (!paused) {
            float dt = GetFrameTime();
            simTime += dt;

            if (!merged) {
                sep -= inspiralRate * dt * (0.4f + 1.9f / (sep + 0.25f));
                sep = std::max(0.62f, sep);
                phase += dt * (2.3f + 8.0f / std::pow(sep + 0.2f, 1.5f));
                if (sep <= 0.66f) {
                    merged = true;
                    postMergerTime = 0.0f;
                }
                luminosity = 0.015f + 0.2f / std::pow(sep + 0.25f, 2.8f);
            } else {
                postMergerTime += dt;
                float flash = 2.3f * std::exp(-std::pow((postMergerTime - 0.22f) / 0.09f, 2.0f));
                float kilonova = 1.5f * std::exp(-postMergerTime / 7.5f);
                luminosity = 0.05f + flash + kilonova;
            }

            luminosityHistory.push_back(luminosity);
            if (luminosityHistory.size() > 360) luminosityHistory.pop_front();
        }

        float totalMass = m1 + m2;
        float r1 = sep * (m2 / totalMass);
        float r2 = sep * (m1 / totalMass);
        Vector3 p1 = {r1 * std::cos(phase), 0.0f, r1 * std::sin(phase)};
        Vector3 p2 = {-r2 * std::cos(phase), 0.0f, -r2 * std::sin(phase)};

        BeginDrawing();
        ClearBackground(Color{5, 8, 15, 255});
        BeginMode3D(camera);

        DrawGrid(24, 1.0f);

        if (!merged) {
            DrawSphere(p1, 0.34f, Color{130, 190, 255, 255});
            DrawSphere(p2, 0.32f, Color{255, 190, 145, 255});
            DrawLine3D(p1, p2, Fade(Color{180, 210, 240, 255}, 0.35f));

            Vector3 spin1 = Vector3Scale(Vector3Normalize({std::cos(phase * spin), 0.45f, std::sin(phase * spin)}), 1.25f);
            Vector3 spin2 = Vector3Scale(Vector3Normalize({-std::cos(phase * spin * 0.92f), -0.35f, -std::sin(phase * spin * 0.92f)}), 1.15f);
            DrawLine3D(p1, Vector3Add(p1, spin1), Color{100, 220, 255, 220});
            DrawLine3D(p2, Vector3Add(p2, spin2), Color{255, 175, 120, 220});
        } else {
            float remnantRadius = 0.55f + 0.1f * std::sin(simTime * 7.0f);
            DrawSphere({0.0f, 0.0f, 0.0f}, remnantRadius, Color{168, 208, 255, 255});
            DrawSphereWires({0.0f, 0.0f, 0.0f}, remnantRadius + 0.15f, 16, 16, Fade(SKYBLUE, 0.4f));

            for (const EjectaParticle& e : ejecta) {
                float travel = e.speed * std::pow(postMergerTime + 0.03f, 0.9f);
                float ripple = 1.0f + 0.18f * std::sin(5.0f * postMergerTime + e.phase);
                Vector3 pos = Vector3Scale(e.dir, travel * ripple);
                float heat = std::exp(-postMergerTime / 9.0f);
                Color c = Color{
                    static_cast<unsigned char>(180 + 70 * heat),
                    static_cast<unsigned char>(120 + 90 * heat),
                    static_cast<unsigned char>(90 + 140 * heat),
                    190
                };
                DrawPoint3D(pos, c);
            }
        }
        EndMode3D();

        DrawRectangle(876, 518, 382, 236, Fade(Color{18, 28, 44, 255}, 0.92f));
        DrawText("Luminosity (GW + Kilonova)", 896, 536, 22, Color{220, 230, 244, 255});
        for (int i = 1; i < static_cast<int>(luminosityHistory.size()); ++i) {
            int x0 = 900 + i - 1;
            int x1 = 900 + i;
            float l0 = std::min(3.2f, luminosityHistory[i - 1]);
            float l1 = std::min(3.2f, luminosityHistory[i]);
            int y0 = 730 - static_cast<int>((l0 / 3.2f) * 164.0f);
            int y1 = 730 - static_cast<int>((l1 / 3.2f) * 164.0f);
            DrawLine(x0, y0, x1, y1, Color{130, 240, 188, 255});
        }

        DrawText("Neutron Star Merger + Kilonova", 20, 18, 30, Color{232, 238, 248, 255});
        DrawText("Mouse drag orbit | wheel zoom | Up/Down m2 | Left/Right inspiral | [ ] spin | P pause | R reset",
                 20, 54, 18, Color{164, 183, 210, 255});

        char status[220];
        std::snprintf(status, sizeof(status),
                      "m1=%.2f Msun  m2=%.2f Msun  sep=%.2f  state=%s%s",
                      m1, m2, sep, merged ? "post-merger" : "inspiral", paused ? " [PAUSED]" : "");
        DrawText(status, 20, 84, 20, Color{126, 224, 255, 255});
        DrawFPS(20, 112);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
