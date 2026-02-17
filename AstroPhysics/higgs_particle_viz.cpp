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

enum class Stage {
    kIdle,
    kIncomingBeams,
    kHiggsCreated,
    kDecayProducts
};

struct DecayProduct {
    Vector3 pos;
    Vector3 vel;
    float life;
    float radius;
    Color color;
    bool isPhoton;
};

void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }

    *distance -= GetMouseWheelMove() * 0.6f;
    *distance = std::clamp(*distance, 6.0f, 36.0f);

    float cp = std::cos(*pitch);
    Vector3 offset = {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}

void DrawColliderRing(float radius, float y, Color color) {
    constexpr int kSegments = 92;
    for (int i = 0; i < kSegments; ++i) {
        float a0 = 2.0f * PI * static_cast<float>(i) / static_cast<float>(kSegments);
        float a1 = 2.0f * PI * static_cast<float>(i + 1) / static_cast<float>(kSegments);
        Vector3 p0 = {radius * std::cos(a0), y, radius * std::sin(a0)};
        Vector3 p1 = {radius * std::cos(a1), y, radius * std::sin(a1)};
        DrawLine3D(p0, p1, color);
    }
}

void DrawBeamBunch(float x, Color color) {
    for (int i = 0; i < 7; ++i) {
        float a = 2.0f * PI * static_cast<float>(i) / 7.0f;
        float z = 0.18f * std::cos(a);
        float y = 0.12f * std::sin(a);
        DrawSphere({x, y, z}, 0.06f, color);
    }
    DrawSphere({x, 0.0f, 0.0f}, 0.07f, color);
}

std::string StageName(Stage s) {
    switch (s) {
        case Stage::kIdle: return "idle";
        case Stage::kIncomingBeams: return "incoming beams";
        case Stage::kHiggsCreated: return "higgs created";
        case Stage::kDecayProducts: return "decay products";
    }
    return "unknown";
}

std::string Hud(Stage stage, bool autoCycle, int productCount, bool paused) {
    std::ostringstream os;
    os << "stage=" << StageName(stage)
       << "  auto=" << (autoCycle ? "on" : "off")
       << "  products=" << productCount;
    if (paused) os << "  [PAUSED]";
    return os.str();
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Higgs Particle Visualization 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.7f, 5.6f, 8.0f};
    camera.target = {0.0f, 0.3f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.87f;
    float camPitch = 0.34f;
    float camDistance = 13.0f;

    bool paused = false;
    bool autoCycle = true;
    Stage stage = Stage::kIdle;
    float stageClock = 0.0f;
    float leftBunchX = -5.0f;
    float rightBunchX = 5.0f;
    std::vector<DecayProduct> products;

    auto resetEvent = [&]() {
        stage = Stage::kIdle;
        stageClock = 0.0f;
        leftBunchX = -5.0f;
        rightBunchX = 5.0f;
        products.clear();
    };

    auto spawnDecayProducts = [&]() {
        products.clear();
        products.push_back({{0.0f, 0.0f, 0.0f}, {3.8f, 0.4f, 1.2f}, 2.4f, 0.05f, Color{255, 225, 120, 255}, true});
        products.push_back({{0.0f, 0.0f, 0.0f}, {-3.8f, -0.4f, -1.2f}, 2.4f, 0.05f, Color{255, 225, 120, 255}, true});

        for (int i = 0; i < 4; ++i) {
            float a = 0.4f + static_cast<float>(i) * (PI * 0.5f);
            float vy = (i % 2 == 0) ? 0.9f : -0.9f;
            Color c = (i % 2 == 0) ? Color{130, 220, 255, 255} : Color{255, 150, 220, 255};
            products.push_back({{0.0f, 0.0f, 0.0f}, {2.0f * std::cos(a), vy, 2.0f * std::sin(a)}, 2.8f, 0.07f, c, false});
        }
    };

    auto triggerEvent = [&]() {
        stage = Stage::kIncomingBeams;
        stageClock = 0.0f;
        leftBunchX = -5.0f;
        rightBunchX = 5.0f;
        products.clear();
    };

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_A)) autoCycle = !autoCycle;
        if (IsKeyPressed(KEY_SPACE)) triggerEvent();
        if (IsKeyPressed(KEY_R)) {
            paused = false;
            autoCycle = true;
            resetEvent();
        }

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        if (!paused) {
            float dt = GetFrameTime();
            stageClock += dt;

            if (stage == Stage::kIdle && autoCycle && stageClock > 1.2f) {
                triggerEvent();
            } else if (stage == Stage::kIncomingBeams) {
                float u = std::clamp(stageClock / 1.15f, 0.0f, 1.0f);
                leftBunchX = -5.0f + 5.0f * u;
                rightBunchX = 5.0f - 5.0f * u;
                if (stageClock >= 1.15f) {
                    stage = Stage::kHiggsCreated;
                    stageClock = 0.0f;
                    leftBunchX = 0.0f;
                    rightBunchX = 0.0f;
                }
            } else if (stage == Stage::kHiggsCreated) {
                if (stageClock >= 0.75f) {
                    stage = Stage::kDecayProducts;
                    stageClock = 0.0f;
                    spawnDecayProducts();
                }
            } else if (stage == Stage::kDecayProducts) {
                for (DecayProduct& p : products) {
                    p.pos = Vector3Add(p.pos, Vector3Scale(p.vel, dt));
                    p.vel = Vector3Scale(p.vel, 0.992f);
                    p.life -= dt;
                }
                products.erase(std::remove_if(products.begin(), products.end(),
                                              [](const DecayProduct& p) { return p.life <= 0.0f; }),
                               products.end());
                if (products.empty() && stageClock > 2.8f) {
                    stage = Stage::kIdle;
                    stageClock = 0.0f;
                    leftBunchX = -5.0f;
                    rightBunchX = 5.0f;
                }
            }
        }

        BeginDrawing();
        ClearBackground(Color{7, 10, 17, 255});

        BeginMode3D(camera);
        DrawPlane({0.0f, -0.22f, 0.0f}, {14.0f, 14.0f}, Color{18, 24, 34, 255});
        DrawColliderRing(4.2f, -0.08f, Color{95, 140, 200, 120});
        DrawColliderRing(4.0f, -0.10f, Color{95, 140, 200, 90});
        DrawLine3D({-5.4f, 0.0f, 0.0f}, {5.4f, 0.0f, 0.0f}, Color{120, 170, 230, 140});

        if (stage == Stage::kIncomingBeams || stage == Stage::kHiggsCreated || stage == Stage::kDecayProducts) {
            DrawBeamBunch(leftBunchX, Color{120, 200, 255, 255});
            DrawBeamBunch(rightBunchX, Color{255, 145, 145, 255});
        }

        if (stage == Stage::kHiggsCreated) {
            float pulse = 0.24f + 0.07f * std::sin(13.0f * stageClock);
            DrawSphere({0.0f, 0.0f, 0.0f}, pulse, Color{255, 235, 145, 255});
            DrawSphere({0.0f, 0.0f, 0.0f}, pulse * 1.7f, Color{255, 220, 130, 65});
        }

        if (stage == Stage::kDecayProducts) {
            DrawSphere({0.0f, 0.0f, 0.0f}, 0.14f, Color{255, 225, 130, 120});
            for (const DecayProduct& p : products) {
                Color lineColor = p.color;
                lineColor.a = static_cast<unsigned char>(std::clamp(90.0f * p.life, 30.0f, 210.0f));
                DrawLine3D({0.0f, 0.0f, 0.0f}, p.pos, lineColor);
                DrawSphere(p.pos, p.radius, p.color);
                if (p.isPhoton) {
                    DrawSphere(p.pos, p.radius * 2.2f, Color{255, 235, 135, 55});
                }
            }
        }

        EndMode3D();

        DrawText("Higgs Particle Event (Conceptual Collider View)", 20, 18, 30, Color{236, 241, 250, 255});
        DrawText("Two proton bunches collide, briefly form a Higgs boson, then decay into detectable products.", 20, 54, 19, Color{166, 186, 212, 255});
        DrawText("Mouse drag: orbit | wheel: zoom | SPACE: trigger event | A: auto cycle | P: pause | R: reset", 20, 80, 18, Color{166, 186, 212, 255});
        DrawText("Yellow lines: photons (gamma gamma) | cyan/magenta: other charged decay products", 20, 108, 19, Color{255, 213, 140, 255});

        std::string hud = Hud(stage, autoCycle, static_cast<int>(products.size()), paused);
        DrawText(hud.c_str(), 20, 138, 20, Color{132, 224, 255, 255});
        DrawFPS(20, 168);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
