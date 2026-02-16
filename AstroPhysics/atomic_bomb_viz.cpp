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

struct Fuel {
    Vector3 pos;
    bool active;
};

struct Neutron {
    Vector3 pos;
    Vector3 vel;
    float life;
};

struct Debris {
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
    *distance = std::clamp(*distance, 4.0f, 36.0f);
    float cp = std::cos(*pitch);
    c->position = Vector3Add(c->target, {*distance * cp * std::cos(*yaw), *distance * std::sin(*pitch), *distance * cp * std::sin(*yaw)});
}

std::string Hud(int activeFuel, int nNeutrons, float yield, bool paused) {
    std::ostringstream os;
    os << "active nuclei=" << activeFuel << "  neutrons=" << nNeutrons << "  yield=" << std::fixed << std::setprecision(1) << yield;
    if (paused) os << "  [PAUSED]";
    return os.str();
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Atomic Bomb Chain Reaction 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {7.0f, 4.6f, 8.0f};
    camera.target = {0.0f, 0.2f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.86f, camPitch = 0.34f, camDistance = 12.5f;

    std::vector<Fuel> fuel;
    std::vector<Neutron> neutrons;
    std::vector<Debris> debris;

    auto reset = [&]() {
        fuel.clear();
        neutrons.clear();
        debris.clear();
        for (int x = -4; x <= 4; ++x) {
            for (int y = -2; y <= 2; ++y) {
                for (int z = -4; z <= 4; ++z) {
                    if ((x + y + z) % 2 == 0) {
                        fuel.push_back({{0.42f * x, 0.35f * y, 0.42f * z}, true});
                    }
                }
            }
        }
        neutrons.push_back({{-2.8f, 0.0f, 0.0f}, {4.5f, 0.0f, 0.0f}, 4.0f});
    };

    reset();
    bool paused = false;
    float yield = 0.0f;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) { reset(); yield = 0.0f; paused = false; }

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        float dt = GetFrameTime();
        if (!paused) {
            for (Neutron& n : neutrons) {
                n.pos = Vector3Add(n.pos, Vector3Scale(n.vel, dt));
                n.life -= dt;
            }

            for (Neutron& n : neutrons) {
                for (Fuel& f : fuel) {
                    if (!f.active) continue;
                    if (Vector3Distance(n.pos, f.pos) < 0.16f) {
                        f.active = false;
                        yield += 1.0f;

                        for (int k = 0; k < 2; ++k) {
                            float a = 2.0f * PI * (0.17f * static_cast<float>(k) + yield * 0.13f);
                            Vector3 dir = Vector3Normalize({std::cos(a), 0.35f * std::sin(2.0f * a), std::sin(a)});
                            neutrons.push_back({f.pos, Vector3Scale(dir, 4.0f + 0.6f * static_cast<float>(k)), 2.4f});
                        }

                        for (int d = 0; d < 4; ++d) {
                            float a = 2.0f * PI * static_cast<float>(d) / 4.0f;
                            Vector3 dir = {std::cos(a), 0.2f * (d - 1.5f), std::sin(a)};
                            debris.push_back({f.pos, Vector3Scale(dir, 2.0f), 1.6f});
                        }
                        break;
                    }
                }
            }

            for (Debris& d : debris) {
                d.pos = Vector3Add(d.pos, Vector3Scale(d.vel, dt));
                d.life -= dt;
                d.vel = Vector3Scale(d.vel, 0.985f);
            }

            neutrons.erase(std::remove_if(neutrons.begin(), neutrons.end(), [](const Neutron& n) { return n.life <= 0.0f; }), neutrons.end());
            debris.erase(std::remove_if(debris.begin(), debris.end(), [](const Debris& d) { return d.life <= 0.0f; }), debris.end());
        }

        int activeCount = 0;
        for (const Fuel& f : fuel) if (f.active) activeCount++;

        BeginDrawing();
        ClearBackground(Color{8, 9, 13, 255});

        BeginMode3D(camera);

        float shockR = std::min(8.0f, 0.15f * yield);
        if (shockR > 0.1f) DrawSphere({0.0f, 0.0f, 0.0f}, shockR, Color{255, 150, 90, 20});

        for (const Fuel& f : fuel) {
            if (f.active) DrawSphere(f.pos, 0.09f, Color{120, 220, 255, 230});
        }
        for (const Neutron& n : neutrons) DrawSphere(n.pos, 0.04f, Color{255, 240, 160, 255});
        for (const Debris& d : debris) DrawSphere(d.pos, 0.03f, Color{255, 120, 90, static_cast<unsigned char>(std::max(0.0f, d.life) * 140.0f)});

        EndMode3D();

        DrawText("Atomic Bomb Chain Reaction (Conceptual)", 20, 18, 29, Color{235, 240, 250, 255});
        DrawText("Hold left mouse: orbit | wheel: zoom | P pause | R reset", 20, 54, 19, Color{170, 184, 204, 255});
        DrawText(Hud(activeCount, static_cast<int>(neutrons.size()), yield, paused).c_str(), 20, 82, 20, Color{255, 210, 150, 255});
        DrawFPS(20, 110);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
