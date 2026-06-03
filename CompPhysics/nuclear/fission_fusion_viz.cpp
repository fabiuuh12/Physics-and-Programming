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

struct Fragment {
    Vector3 pos;
    Vector3 vel;
    float life;
    Color color;
};

void SpawnEnergyBurst(std::vector<Fragment>* frags, Vector3 origin, int count, float speed, float life, Color color) {
    for (int i = 0; i < count; ++i) {
        float a = 2.0f * PI * static_cast<float>(i) / static_cast<float>(count);
        Vector3 dir = {std::cos(a), 0.24f * std::sin(2.4f * a), std::sin(a)};
        frags->push_back({origin, Vector3Scale(dir, speed), life, color});
    }
}

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
    InitWindow(kScreenWidth, kScreenHeight, "Fission vs Fusion 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {7.2f, 4.6f, 8.0f};
    camera.target = {0.0f, 0.2f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.84f, camPitch = 0.33f, camDistance = 12.4f;

    bool fusionMode = false;
    bool paused = false;
    std::vector<Fragment> frags;

    float reactionTime = 0.0f;
    bool reactionActive = false;
    bool energyReleased = false;

    float neutronX = -3.0f;
    float splitOffset = 0.0f;
    float fusionSeparation = 1.8f;

    auto resetReaction = [&]() {
        frags.clear();
        reactionTime = 0.0f;
        reactionActive = false;
        energyReleased = false;
        neutronX = -3.0f;
        splitOffset = 0.0f;
        fusionSeparation = 1.8f;
    };

    auto trigger = [&]() {
        resetReaction();
        reactionActive = true;
    };

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_M)) {
            fusionMode = !fusionMode;
            resetReaction();
        }
        if (IsKeyPressed(KEY_SPACE)) trigger();
        if (IsKeyPressed(KEY_R)) {
            resetReaction();
            paused = false;
        }

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        if (!paused) {
            float dt = GetFrameTime();

            if (reactionActive) {
                reactionTime += dt;

                if (!fusionMode) {
                    const float impactTime = 0.95f;
                    if (reactionTime < impactTime) {
                        neutronX = -3.0f + (reactionTime / impactTime) * 3.0f;
                    } else {
                        splitOffset = std::min((reactionTime - impactTime) * 1.35f, 1.35f);
                        if (!energyReleased) {
                            SpawnEnergyBurst(&frags, {0.0f, 0.0f, 0.0f}, 22, 3.1f, 2.1f, Color{255, 175, 100, 255});
                            energyReleased = true;
                        }
                    }
                } else {
                    const float mergeTime = 1.15f;
                    if (reactionTime < mergeTime) {
                        fusionSeparation = std::max(0.0f, 1.8f - (reactionTime / mergeTime) * 1.8f);
                    } else if (!energyReleased) {
                        SpawnEnergyBurst(&frags, {0.0f, 0.0f, 0.0f}, 30, 4.0f, 2.5f, Color{255, 225, 120, 255});
                        energyReleased = true;
                    }
                }
            }

            for (Fragment& f : frags) {
                f.pos = Vector3Add(f.pos, Vector3Scale(f.vel, dt));
                f.vel = Vector3Scale(f.vel, 0.985f);
                f.life -= dt;
            }
            frags.erase(std::remove_if(frags.begin(), frags.end(), [](const Fragment& f) { return f.life <= 0.0f; }), frags.end());
        }

        BeginDrawing();
        ClearBackground(Color{7, 10, 16, 255});

        BeginMode3D(camera);

        if (!fusionMode) {
            const bool hasSplit = splitOffset > 0.0f;

            if (!hasSplit) {
                DrawSphere({0.0f, 0.0f, 0.0f}, 0.48f, Color{120, 210, 255, 230});
                DrawSphere({0.0f, 0.0f, 0.0f}, 0.66f, Color{120, 180, 255, 70});
            } else {
                float daughterDist = 0.35f + splitOffset;
                DrawSphere({-daughterDist, 0.06f, 0.0f}, 0.30f, Color{130, 215, 255, 235});
                DrawSphere({daughterDist, -0.06f, 0.0f}, 0.30f, Color{130, 215, 255, 235});
            }

            if (reactionActive && !energyReleased) {
                DrawSphere({neutronX, 0.0f, 0.0f}, 0.11f, Color{255, 120, 120, 255});
                DrawLine3D({neutronX - 0.35f, 0.0f, 0.0f}, {neutronX - 0.12f, 0.0f, 0.0f}, Color{255, 150, 150, 180});
            }
        } else {
            const bool merged = energyReleased;

            if (!merged) {
                DrawSphere({-fusionSeparation, 0.0f, 0.0f}, 0.30f, Color{120, 210, 255, 230});
                DrawSphere({fusionSeparation, 0.0f, 0.0f}, 0.30f, Color{120, 210, 255, 230});
                DrawLine3D({-fusionSeparation, 0.0f, 0.0f}, {fusionSeparation, 0.0f, 0.0f}, Color{255, 190, 120, 120});
            } else {
                float pulse = 1.0f + 0.18f * std::exp(-2.0f * std::max(0.0f, reactionTime - 1.15f)) *
                                         std::sin(15.0f * std::max(0.0f, reactionTime - 1.15f));
                DrawSphere({0.0f, 0.0f, 0.0f}, 0.42f * pulse, Color{150, 225, 255, 240});
                DrawSphere({0.0f, 0.0f, 0.0f}, 0.62f * pulse, Color{170, 205, 255, 70});
            }
        }

        for (const Fragment& f : frags) {
            Color c = f.color;
            c.a = static_cast<unsigned char>(std::clamp(120.0f * f.life, 20.0f, 255.0f));
            DrawSphere(f.pos, 0.05f, c);
        }

        EndMode3D();

        DrawText("Fission vs Fusion (Reaction Mechanics)", 20, 18, 29, Color{235, 240, 250, 255});
        DrawText("Hold left mouse: orbit | wheel: zoom | M mode toggle | SPACE trigger | P pause | R reset", 20, 54, 18, Color{170, 184, 204, 255});

        std::ostringstream os;
        if (fusionMode) {
            os << "mode=fusion: two nuclei merge into one + energy";
        } else {
            os << "mode=fission: neutron strikes nucleus, splits into two + energy";
        }
        os << "  particles=" << frags.size();
        if (paused) os << "  [PAUSED]";
        DrawText(os.str().c_str(), 20, 82, 20, Color{255, 210, 150, 255});
        DrawFPS(20, 110);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
