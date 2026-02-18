#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kScreenWidth = 1280;
constexpr int kScreenHeight = 820;

struct GasParticle {
    Vector3 pos;
    Vector3 vel;
};

void UpdateOrbitCameraDragOnly(Camera3D* c, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }
    *distance -= GetMouseWheelMove() * 0.6f;
    *distance = std::clamp(*distance, 4.0f, 34.0f);
    float cp = std::cos(*pitch);
    c->position = Vector3Add(c->target, {*distance * cp * std::cos(*yaw), *distance * std::sin(*pitch), *distance * cp * std::sin(*yaw)});
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Thermodynamics Gas Laws 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.0f, 5.0f, 9.0f};
    camera.target = {0.0f, 0.8f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.85f, camPitch = 0.34f, camDistance = 13.0f;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> ur(-1.0f, 1.0f);

    float halfX = 2.4f;
    float halfY = 1.6f;
    float halfZ = 1.8f;
    float temperature = 1.0f;

    std::vector<GasParticle> gas;
    gas.reserve(240);

    auto reset = [&]() {
        gas.clear();
        for (int i = 0; i < 240; ++i) {
            Vector3 p = {ur(rng) * halfX * 0.95f, ur(rng) * halfY * 0.95f + 0.8f, ur(rng) * halfZ * 0.95f};
            Vector3 v = {ur(rng), ur(rng), ur(rng)};
            v = Vector3Scale(Vector3Normalize(v), 1.2f * std::sqrt(temperature));
            gas.push_back({p, v});
        }
    };

    reset();
    bool paused = false;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) { temperature = 1.0f; reset(); paused = false; }
        if (IsKeyPressed(KEY_LEFT_BRACKET)) temperature = std::max(0.2f, temperature - 0.1f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) temperature = std::min(4.0f, temperature + 0.1f);

        for (GasParticle& g : gas) {
            float current = Vector3Length(g.vel);
            if (current > 1e-4f) {
                g.vel = Vector3Scale(g.vel, (1.2f * std::sqrt(temperature)) / current);
            }
        }

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        float momentumWall = 0.0f;
        if (!paused) {
            float dt = GetFrameTime();
            for (GasParticle& gp : gas) {
                gp.pos = Vector3Add(gp.pos, Vector3Scale(gp.vel, dt));

                if (gp.pos.x < -halfX) { gp.pos.x = -halfX; momentumWall += 2.0f * std::fabs(gp.vel.x); gp.vel.x *= -1.0f; }
                if (gp.pos.x > halfX)  { gp.pos.x = halfX;  momentumWall += 2.0f * std::fabs(gp.vel.x); gp.vel.x *= -1.0f; }
                if (gp.pos.y < 0.8f - halfY) { gp.pos.y = 0.8f - halfY; momentumWall += 2.0f * std::fabs(gp.vel.y); gp.vel.y *= -1.0f; }
                if (gp.pos.y > 0.8f + halfY) { gp.pos.y = 0.8f + halfY; momentumWall += 2.0f * std::fabs(gp.vel.y); gp.vel.y *= -1.0f; }
                if (gp.pos.z < -halfZ) { gp.pos.z = -halfZ; momentumWall += 2.0f * std::fabs(gp.vel.z); gp.vel.z *= -1.0f; }
                if (gp.pos.z > halfZ)  { gp.pos.z = halfZ;  momentumWall += 2.0f * std::fabs(gp.vel.z); gp.vel.z *= -1.0f; }
            }
        }

        float volume = (2.0f * halfX) * (2.0f * halfY) * (2.0f * halfZ);
        float n = static_cast<float>(gas.size());
        float pIdeal = n * temperature / std::max(0.1f, volume);

        BeginDrawing();
        ClearBackground(Color{7, 10, 16, 255});

        BeginMode3D(camera);

        DrawCubeWires({0.0f, 0.8f, 0.0f}, 2.0f * halfX, 2.0f * halfY, 2.0f * halfZ, Color{130, 180, 255, 180});

        for (const GasParticle& gp : gas) {
            float sp = Vector3Length(gp.vel);
            float heat = std::clamp(sp / 3.0f, 0.0f, 1.0f);
            Color c = Color{static_cast<unsigned char>(100 + 155 * heat), static_cast<unsigned char>(120 + 80 * (1.0f - heat)), 255, 230};
            DrawSphere(gp.pos, 0.05f, c);
        }

        EndMode3D();

        DrawText("Thermodynamics Laws: Ideal Gas in a Box", 20, 18, 29, Color{235, 240, 250, 255});
        DrawText("Hold left mouse: orbit | wheel: zoom | [ ] temperature | P pause | R reset", 20, 54, 18, Color{170, 184, 204, 255});

        std::ostringstream os;
        os << std::fixed << std::setprecision(3)
           << "N=" << gas.size()
           << "  T=" << temperature
           << "  V=" << volume
           << "  P~" << pIdeal
           << "  PV/(NT)~" << (pIdeal * volume / std::max(0.001f, n * temperature));
        if (paused) os << "  [PAUSED]";
        DrawText(os.str().c_str(), 20, 82, 20, Color{200, 220, 255, 255});
        DrawFPS(20, 110);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
