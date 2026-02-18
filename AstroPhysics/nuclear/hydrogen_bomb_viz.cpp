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

struct Plasma {
    Vector3 pos;
    Vector3 vel;
    float life;
};

void UpdateOrbitCameraDragOnly(Camera3D* c, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }
    *distance -= GetMouseWheelMove() * 0.6f;
    *distance = std::clamp(*distance, 4.0f, 38.0f);
    float cp = std::cos(*pitch);
    c->position = Vector3Add(c->target, {*distance * cp * std::cos(*yaw), *distance * std::sin(*pitch), *distance * cp * std::sin(*yaw)});
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Hydrogen Bomb Two-Stage Concept 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.0f, 5.0f, 9.0f};
    camera.target = {0.0f, 0.3f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.84f, camPitch = 0.34f, camDistance = 13.5f;

    std::vector<Plasma> plasma;
    bool paused = false;
    float t = 0.0f;
    float stage1 = 3.0f;
    float stage2 = 7.0f;

    auto reset = [&]() {
        t = 0.0f;
        plasma.clear();
    };

    reset();

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) { reset(); paused = false; }

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        if (!paused) {
            float dt = GetFrameTime();
            t += dt;

            if (t > stage1 && t < stage2) {
                for (int i = 0; i < 6; ++i) {
                    float a = 2.0f * PI * (0.07f * i + t * 0.4f);
                    Vector3 dir = {std::cos(a), 0.35f * std::sin(2.0f * a), std::sin(a)};
                    plasma.push_back({{0.0f, 0.0f, 0.0f}, Vector3Scale(dir, 2.4f), 2.0f});
                }
            }
            if (t > stage2) {
                for (int i = 0; i < 12; ++i) {
                    float a = 2.0f * PI * (0.11f * i + t * 0.6f);
                    Vector3 dir = {std::cos(a), 0.25f * std::sin(3.0f * a), std::sin(a)};
                    plasma.push_back({{0.0f, 0.0f, 0.0f}, Vector3Scale(dir, 4.8f), 2.6f});
                }
            }

            for (Plasma& p : plasma) {
                p.pos = Vector3Add(p.pos, Vector3Scale(p.vel, dt));
                p.vel = Vector3Scale(p.vel, 0.985f);
                p.life -= dt;
            }
            plasma.erase(std::remove_if(plasma.begin(), plasma.end(), [](const Plasma& p) { return p.life <= 0.0f; }), plasma.end());
        }

        BeginDrawing();
        ClearBackground(Color{8, 9, 13, 255});

        BeginMode3D(camera);

        Color primaryC = (t < stage1) ? Color{120, 220, 255, 220} : Color{255, 160, 90, 240};
        DrawSphere({-1.2f, 0.0f, 0.0f}, 0.32f, primaryC);

        Color secondaryC = (t < stage2) ? Color{120, 180, 255, 160} : Color{255, 230, 120, 240};
        DrawSphere({1.2f, 0.0f, 0.0f}, 0.42f, secondaryC);

        if (t >= stage1 && t < stage2) {
            DrawCylinder({0.0f, 0.0f, 0.0f}, 0.14f, 0.14f, 2.5f, 18, Color{255, 160, 90, 120});
        }

        for (const Plasma& p : plasma) {
            unsigned char a = static_cast<unsigned char>(std::clamp(90.0f * p.life, 20.0f, 255.0f));
            DrawSphere(p.pos, 0.04f + 0.01f * p.life, Color{255, 180, 100, a});
        }

        float blast = (t > stage2) ? std::min(9.0f, (t - stage2) * 2.2f) : 0.0f;
        if (blast > 0.05f) {
            DrawSphere({0.0f, 0.0f, 0.0f}, blast, Color{255, 170, 100, 20});
        }

        EndMode3D();

        std::string phase = (t < stage1) ? "pre-detonation" : ((t < stage2) ? "fission primary" : "fusion secondary");

        DrawText("Hydrogen Bomb Two-Stage Sequence (Conceptual)", 20, 18, 29, Color{235, 240, 250, 255});
        DrawText("Hold left mouse: orbit | wheel: zoom | P pause | R reset", 20, 54, 19, Color{170, 184, 204, 255});

        std::ostringstream os;
        os << std::fixed << std::setprecision(2) << "t=" << t << "  phase=" << phase << "  plasma=" << plasma.size();
        if (paused) os << "  [PAUSED]";
        DrawText(os.str().c_str(), 20, 82, 20, Color{255, 210, 150, 255});
        DrawFPS(20, 110);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
