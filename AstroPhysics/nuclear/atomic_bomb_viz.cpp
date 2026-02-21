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
constexpr float kCoreRevealStartDistance = 3.2f;
constexpr float kCoreRevealEndDistance = 2.0f;
constexpr float kDetonationButtonX = 1080.0f;
constexpr float kDetonationButtonY = 18.0f;
constexpr float kDetonationButtonW = 178.0f;
constexpr float kDetonationButtonH = 42.0f;

struct Fuel {
    Vector3 pos;
    bool active;
    float cooldown;
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

struct FissionEvent {
    Vector3 pos;
    Vector3 axis;
    float life;
    float totalLife;
};

float SmoothStep(float x) {
    x = std::clamp(x, 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x);
}

float Rand01() {
    return static_cast<float>(GetRandomValue(0, 10000)) / 10000.0f;
}

void SpawnSeedNeutron(std::vector<Neutron>* neutrons) {
    float a = 2.0f * PI * Rand01();
    float y = 2.0f * Rand01() - 1.0f;
    Vector3 dir = Vector3Normalize({std::cos(a), 0.35f * y, std::sin(a)});
    Vector3 start = Vector3Scale(dir, 1.30f + 0.22f * Rand01());
    Vector3 inward = Vector3Normalize(Vector3Negate(start));
    Vector3 tangent = Vector3Normalize(Vector3CrossProduct(inward, {0.0f, 1.0f, 0.0f}));
    if (Vector3Length(tangent) < 0.001f) tangent = {1.0f, 0.0f, 0.0f};
    Vector3 vel = Vector3Normalize(Vector3Add(Vector3Scale(inward, 0.92f), Vector3Scale(tangent, 0.28f)));
    neutrons->push_back({start, Vector3Scale(vel, 3.1f + Rand01()), 2.0f + 1.1f * Rand01()});
}

void UpdateOrbitCameraDragOnly(Camera3D* c, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }
    *distance -= GetMouseWheelMove() * 0.6f;
    *distance = std::clamp(*distance, 1.3f, 38.0f);
    float cp = std::cos(*pitch);
    c->position = Vector3Add(c->target, {*distance * cp * std::cos(*yaw), *distance * std::sin(*pitch), *distance * cp * std::sin(*yaw)});
}

void DrawBombExterior(unsigned char alpha) {
    Color shellBase = {156, 165, 178, alpha};
    Color shellDark = {96, 105, 116, alpha};
    Color shellBright = {218, 224, 232, static_cast<unsigned char>(std::max(30, alpha - 10))};
    Color ring = {225, 231, 239, static_cast<unsigned char>(std::max(34, alpha / 2))};
    Color rivet = {80, 87, 97, static_cast<unsigned char>(std::max(30, alpha - 28))};

    DrawCylinder({0.0f, 0.0f, 0.0f}, 1.12f, 1.18f, 6.2f, 42, shellBase);
    DrawCylinder({0.0f, 0.0f, 0.0f}, 1.16f, 1.22f, 6.16f, 42, Color{200, 206, 216, static_cast<unsigned char>(alpha / 3)});
    DrawCylinder({0.0f, 0.0f, 0.0f}, 1.07f, 1.13f, 6.0f, 42, Color{108, 116, 126, static_cast<unsigned char>(alpha / 3)});
    DrawCylinder({0.0f, 3.52f, 0.0f}, 0.0f, 1.08f, 1.08f, 42, shellBase);
    DrawCylinder({0.0f, -3.25f, 0.58f}, 0.18f, 0.26f, 0.72f, 18, shellDark);
    DrawCylinder({0.0f, -3.25f, -0.58f}, 0.18f, 0.26f, 0.72f, 18, shellDark);

    DrawCube({1.24f, -2.96f, 0.0f}, 0.06f, 0.56f, 0.8f, shellDark);
    DrawCube({-1.24f, -2.96f, 0.0f}, 0.06f, 0.56f, 0.8f, shellDark);
    DrawCube({0.0f, -2.96f, 1.24f}, 0.8f, 0.56f, 0.06f, shellDark);
    DrawCube({0.0f, -2.96f, -1.24f}, 0.8f, 0.56f, 0.06f, shellDark);
    DrawCube({1.09f, 0.3f, 0.0f}, 0.03f, 5.8f, 0.75f, shellBright);
    DrawCube({-1.09f, -0.2f, 0.0f}, 0.03f, 5.4f, 0.55f, Color{133, 142, 154, static_cast<unsigned char>(alpha / 2)});
    DrawCube({0.0f, 0.2f, 1.08f}, 0.58f, 5.3f, 0.03f, Color{127, 136, 148, static_cast<unsigned char>(alpha / 2)});
    DrawCube({0.0f, -0.3f, -1.08f}, 0.44f, 5.0f, 0.03f, shellBright);

    DrawCylinderWires({0.0f, 2.4f, 0.0f}, 1.16f, 1.16f, 0.14f, 42, ring);
    DrawCylinderWires({0.0f, 0.1f, 0.0f}, 1.16f, 1.16f, 0.14f, 42, ring);
    DrawCylinderWires({0.0f, -2.2f, 0.0f}, 1.16f, 1.16f, 0.14f, 42, ring);
    DrawCylinderWires({0.0f, 3.2f, 0.0f}, 0.95f, 1.05f, 0.08f, 32, shellBright);
    DrawCylinderWires({0.0f, -3.05f, 0.0f}, 1.02f, 1.10f, 0.08f, 32, rivet);
}

std::string Hud(int activeFuel, int nNeutrons, float totalFissions, float camDistance, bool paused, bool detonated) {
    std::ostringstream os;
    os << "core=fission running  active nuclei=" << activeFuel
       << "  neutrons=" << nNeutrons
       << "  total fissions=" << std::fixed << std::setprecision(0) << totalFissions
       << "  zoom=" << std::setprecision(2) << camDistance;
    if (detonated) os << "  [DETONATED]";
    if (paused) os << "  [PAUSED]";
    return os.str();
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Atomic Bomb Core Explorer 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.82f, camPitch = 0.24f, camDistance = 11.8f;

    std::vector<Fuel> fuel;
    std::vector<Neutron> neutrons;
    std::vector<Debris> debris;
    std::vector<FissionEvent> fissionEvents;

    float totalFissions = 0.0f;
    float neutronSeedTimer = 0.0f;
    bool detonated = false;
    float detonationTime = 0.0f;

    auto reset = [&]() {
        camYaw = 0.82f;
        camPitch = 0.24f;
        camDistance = 11.8f;

        fuel.clear();
        neutrons.clear();
        debris.clear();
        fissionEvents.clear();
        totalFissions = 0.0f;
        neutronSeedTimer = 0.0f;
        detonated = false;
        detonationTime = 0.0f;

        for (int x = -4; x <= 4; ++x) {
            for (int y = -3; y <= 3; ++y) {
                for (int z = -4; z <= 4; ++z) {
                    if ((x + y + z) % 2 == 0) {
                        fuel.push_back({{0.24f * x, 0.20f * y, 0.24f * z}, true, 0.0f});
                    }
                }
            }
        }

        for (int i = 0; i < 4; ++i) SpawnSeedNeutron(&neutrons);
    };

    reset();
    bool paused = false;

    while (!WindowShouldClose()) {
        Rectangle detonateButton = {kDetonationButtonX, kDetonationButtonY, kDetonationButtonW, kDetonationButtonH};
        Vector2 mouse = GetMousePosition();
        bool detonateHover = CheckCollisionPointRec(mouse, detonateButton);
        if (!detonated && detonateHover && IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            detonated = true;
            detonationTime = 0.0f;
            neutronSeedTimer = 0.0f;
            for (int i = 0; i < 60; ++i) SpawnSeedNeutron(&neutrons);
        }

        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) { reset(); paused = false; }

        camera.target = {0.0f, 0.0f, 0.0f};
        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        float dt = GetFrameTime();
        if (!paused) {
            if (detonated) detonationTime += dt;

            neutronSeedTimer += dt;
            float neutronSeedStep = detonated ? 0.045f : 0.28f;
            while (neutronSeedTimer >= neutronSeedStep) {
                SpawnSeedNeutron(&neutrons);
                neutronSeedTimer -= neutronSeedStep;
            }

            for (Fuel& f : fuel) {
                if (!f.active) {
                    f.cooldown -= dt;
                    if (f.cooldown <= 0.0f) {
                        f.cooldown = 0.0f;
                        f.active = true;
                    }
                }
            }

            for (Neutron& n : neutrons) {
                n.pos = Vector3Add(n.pos, Vector3Scale(n.vel, dt));
                n.life -= dt;
            }

            for (Neutron& n : neutrons) {
                if (n.life <= 0.0f) continue;
                for (Fuel& f : fuel) {
                    if (!f.active) continue;
                    if (Vector3Distance(n.pos, f.pos) < 0.11f) {
                        f.active = false;
                        f.cooldown = detonated ? (0.09f + 0.32f * Rand01()) : (0.75f + 1.15f * Rand01());
                        n.life = 0.0f;
                        totalFissions += 1.0f;

                        float spin = 2.0f * PI * (0.073f * totalFissions + Rand01());
                        Vector3 splitAxis = Vector3Normalize({std::cos(spin), 0.42f * std::sin(1.6f * spin), std::sin(spin)});
                        fissionEvents.push_back({f.pos, splitAxis, 0.56f, 0.56f});

                        for (int k = 0; k < 2; ++k) {
                            float a = 2.0f * PI * (Rand01() + 0.37f * static_cast<float>(k));
                            Vector3 dir = Vector3Normalize({std::cos(a), 0.30f * (2.0f * Rand01() - 1.0f), std::sin(a)});
                            neutrons.push_back({f.pos, Vector3Scale(dir, 2.8f + 1.4f * Rand01()), 1.8f + 1.0f * Rand01()});
                        }

                        for (int d = 0; d < 4; ++d) {
                            float a = 2.0f * PI * static_cast<float>(d) / 4.0f;
                            Vector3 dir = {std::cos(a), 0.2f * (static_cast<float>(d) - 1.5f), std::sin(a)};
                            debris.push_back({f.pos, Vector3Scale(dir, 1.75f), 1.2f});
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
            for (FissionEvent& e : fissionEvents) e.life -= dt;

            neutrons.erase(std::remove_if(neutrons.begin(), neutrons.end(), [](const Neutron& n) { return n.life <= 0.0f; }), neutrons.end());
            debris.erase(std::remove_if(debris.begin(), debris.end(), [](const Debris& d) { return d.life <= 0.0f; }), debris.end());
            fissionEvents.erase(std::remove_if(fissionEvents.begin(), fissionEvents.end(), [](const FissionEvent& e) { return e.life <= 0.0f; }), fissionEvents.end());

            if (neutrons.size() < 2) SpawnSeedNeutron(&neutrons);
        }

        int activeCount = 0;
        for (const Fuel& f : fuel) if (f.active) activeCount++;

        float revealT = (kCoreRevealStartDistance - camDistance) / (kCoreRevealStartDistance - kCoreRevealEndDistance);
        float coreReveal = SmoothStep(revealT);
        if (detonated) {
            float detReveal = SmoothStep(std::clamp(detonationTime / 0.65f, 0.0f, 1.0f));
            coreReveal = std::max(coreReveal, detReveal);
        }
        bool drawCore = coreReveal > 0.015f;

        float detShellFade = detonated ? SmoothStep(std::clamp(detonationTime / 1.15f, 0.0f, 1.0f)) : 0.0f;
        float shellAlphaF = (255.0f - 135.0f * coreReveal) * (1.0f - detShellFade);
        unsigned char shellAlpha = static_cast<unsigned char>(std::clamp(shellAlphaF, 0.0f, 255.0f));
        unsigned char chamberAlpha = static_cast<unsigned char>(std::clamp(220.0f * coreReveal + 30.0f * detShellFade, 0.0f, 220.0f));
        unsigned char nucleusAlpha = static_cast<unsigned char>(std::clamp(255.0f * coreReveal, 0.0f, 255.0f));

        BeginDrawing();
        ClearBackground(Color{8, 9, 13, 255});

        BeginMode3D(camera);

        if (detonated) {
            float blast = SmoothStep(std::clamp(detonationTime / 1.5f, 0.0f, 1.0f));
            float blastR = 1.1f + 24.0f * blast;
            DrawSphere({0.0f, 0.0f, 0.0f}, blastR, Color{255, 145, 88, static_cast<unsigned char>(80.0f * (1.0f - blast))});
        }

        if (drawCore) {
            float heat = std::clamp(static_cast<float>(neutrons.size()) / 60.0f, 0.0f, 1.0f);
            float shockR = 0.16f + 0.12f * heat + 0.04f * std::sin(4.0f * static_cast<float>(GetTime()));
            DrawSphere({0.0f, 0.0f, 0.0f}, shockR, Color{255, 150, 90, static_cast<unsigned char>(32.0f * coreReveal)});

            for (const Fuel& f : fuel) {
                if (f.active) {
                    DrawSphere(f.pos, 0.058f, Color{125, 220, 255, nucleusAlpha});
                } else {
                    float cool = std::clamp(f.cooldown / 1.9f, 0.0f, 1.0f);
                    DrawSphere(f.pos, 0.046f, Color{255, 138, 110, static_cast<unsigned char>(55.0f * cool * coreReveal)});
                }
            }

            for (const Neutron& n : neutrons) DrawSphere(n.pos, 0.034f, Color{255, 240, 160, static_cast<unsigned char>(255.0f * coreReveal)});
            for (const Debris& d : debris) {
                DrawSphere(d.pos, 0.027f, Color{255, 120, 90, static_cast<unsigned char>(std::max(0.0f, d.life) * 145.0f * coreReveal)});
            }

            for (const FissionEvent& e : fissionEvents) {
                float life01 = std::clamp(e.life / e.totalLife, 0.0f, 1.0f);
                float split = 0.035f + 0.16f * SmoothStep(1.0f - life01);
                Vector3 da = Vector3Scale(e.axis, split);
                Vector3 p1 = Vector3Add(e.pos, da);
                Vector3 p2 = Vector3Subtract(e.pos, da);
                unsigned char a = static_cast<unsigned char>((60.0f + 185.0f * life01) * coreReveal);

                DrawSphere(e.pos, 0.11f * life01, Color{255, 245, 170, static_cast<unsigned char>(95.0f * life01 * coreReveal)});
                DrawSphere(p1, 0.043f, Color{255, 170, 115, a});
                DrawSphere(p2, 0.043f, Color{255, 150, 100, a});
                DrawCylinderEx(p1, p2, 0.011f * life01, 0.004f, 12, Color{255, 235, 170, static_cast<unsigned char>(120.0f * life01 * coreReveal)});
            }
        }

        if (chamberAlpha > 2) {
            DrawSphere({0.0f, 0.0f, 0.0f}, 0.98f, Color{140, 195, 245, chamberAlpha});
            DrawSphereWires({0.0f, 0.0f, 0.0f}, 1.06f, 13, 18, Color{200, 225, 255, static_cast<unsigned char>(chamberAlpha / 2)});
        }
        if (shellAlpha > 2) DrawBombExterior(shellAlpha);

        EndMode3D();

        DrawText("Atomic Bomb Core Explorer (Conceptual)", 20, 18, 29, Color{235, 240, 250, 255});
        DrawText("Hold left mouse: orbit | wheel: zoom | P pause | R reset", 20, 54, 19, Color{170, 184, 204, 255});
        DrawText(Hud(activeCount, static_cast<int>(neutrons.size()), totalFissions, camDistance, paused, detonated).c_str(), 20, 82, 20, Color{255, 210, 150, 255});
        Color buttonFill = detonated
            ? Color{118, 44, 36, 255}
            : (detonateHover ? Color{214, 86, 64, 255} : Color{176, 58, 43, 255});
        Color buttonBorder = detonated ? Color{172, 120, 110, 255} : Color{255, 188, 160, 255};
        DrawRectangleRec(detonateButton, buttonFill);
        DrawRectangleLinesEx(detonateButton, 2.0f, buttonBorder);
        const char* buttonText = detonated ? "DETONATED" : "DETONATE";
        int tw = MeasureText(buttonText, 22);
        DrawText(buttonText, static_cast<int>(detonateButton.x + (detonateButton.width - static_cast<float>(tw)) * 0.5f),
                 static_cast<int>(detonateButton.y + 10.0f), 22, Color{255, 239, 232, 255});
        DrawFPS(20, 110);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
