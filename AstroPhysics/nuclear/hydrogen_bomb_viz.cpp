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
constexpr float kExteriorDuration = 2.9f;
constexpr float kZoomDuration = 3.1f;
constexpr float kPrimaryDuration = 3.0f;
constexpr float kTransferDuration = 2.0f;

struct Plasma {
    Vector3 pos;
    Vector3 vel;
    float life;
    float radius;
    Color color;
};

enum class SequencePhase {
    Exterior,
    ZoomToCore,
    PrimaryFission,
    RadiationTransfer,
    SecondaryFusion
};

float SmoothStep(float x) {
    x = std::clamp(x, 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x);
}

SequencePhase GetPhase(float sequenceTime) {
    if (sequenceTime < kExteriorDuration) return SequencePhase::Exterior;
    if (sequenceTime < kExteriorDuration + kZoomDuration) return SequencePhase::ZoomToCore;

    float coreT = sequenceTime - (kExteriorDuration + kZoomDuration);
    if (coreT < kPrimaryDuration) return SequencePhase::PrimaryFission;
    if (coreT < kPrimaryDuration + kTransferDuration) return SequencePhase::RadiationTransfer;
    return SequencePhase::SecondaryFusion;
}

float GetCoreTime(float sequenceTime) {
    return std::max(0.0f, sequenceTime - (kExteriorDuration + kZoomDuration));
}

void ApplyScriptedCamera(Camera3D* c, float sequenceTime) {
    const Vector3 exteriorPos = {11.8f, 5.4f, 12.6f};
    const Vector3 exteriorTarget = {0.0f, 0.4f, 0.0f};
    const Vector3 corePos = {3.1f, 0.9f, 3.4f};
    const Vector3 coreTarget = {0.0f, -0.5f, 0.0f};

    SequencePhase phase = GetPhase(sequenceTime);
    if (phase == SequencePhase::Exterior) {
        float drift = 0.5f * std::sin(sequenceTime * 0.72f);
        c->position = Vector3Add(exteriorPos, {drift, 0.08f * drift, -0.22f * drift});
        c->target = exteriorTarget;
        return;
    }

    float u = (sequenceTime - kExteriorDuration) / kZoomDuration;
    float s = SmoothStep(u);
    c->position = Vector3Lerp(exteriorPos, corePos, s);
    c->target = Vector3Lerp(exteriorTarget, coreTarget, s);
}

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

void DrawBombExterior(unsigned char alpha) {
    Color shell = {176, 184, 196, alpha};
    Color shellDark = {116, 124, 138, alpha};
    Color stripe = {214, 220, 230, static_cast<unsigned char>(std::max(40, alpha / 2))};

    DrawCylinder({0.0f, 0.0f, 0.0f}, 1.22f, 1.30f, 7.0f, 44, shell);
    DrawCylinder({0.0f, 4.0f, 0.0f}, 0.0f, 1.16f, 1.3f, 44, shell);
    DrawCylinder({0.0f, -3.6f, 0.0f}, 0.62f, 0.9f, 0.65f, 32, shellDark);

    DrawCube({1.35f, -3.35f, 0.0f}, 0.06f, 0.8f, 1.0f, shellDark);
    DrawCube({-1.35f, -3.35f, 0.0f}, 0.06f, 0.8f, 1.0f, shellDark);
    DrawCube({0.0f, -3.35f, 1.35f}, 1.0f, 0.8f, 0.06f, shellDark);
    DrawCube({0.0f, -3.35f, -1.35f}, 1.0f, 0.8f, 0.06f, shellDark);

    DrawCylinderWires({0.0f, 2.6f, 0.0f}, 1.26f, 1.26f, 0.18f, 44, stripe);
    DrawCylinderWires({0.0f, 0.0f, 0.0f}, 1.26f, 1.26f, 0.18f, 44, stripe);
    DrawCylinderWires({0.0f, -2.6f, 0.0f}, 1.26f, 1.26f, 0.18f, 44, stripe);
}

const char* PhaseText(SequencePhase phase) {
    switch (phase) {
        case SequencePhase::Exterior: return "intact shell";
        case SequencePhase::ZoomToCore: return "zooming to stages";
        case SequencePhase::PrimaryFission: return "primary fission";
        case SequencePhase::RadiationTransfer: return "x-ray compression";
        case SequencePhase::SecondaryFusion: return "secondary fusion";
    }
    return "sequence";
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

    float camYaw = 0.82f, camPitch = 0.28f, camDistance = 4.8f;

    std::vector<Plasma> plasma;
    bool paused = false;
    float sequenceTime = 0.0f;
    Vector3 primary = {0.0f, -1.25f, 0.0f};
    Vector3 secondary = {0.0f, 1.25f, 0.0f};

    auto reset = [&]() {
        sequenceTime = 0.0f;
        camYaw = 0.82f;
        camPitch = 0.28f;
        camDistance = 4.8f;
        plasma.clear();
    };

    reset();

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) { reset(); paused = false; }
        SequencePhase phase = GetPhase(sequenceTime);

        if (!paused) {
            float dt = GetFrameTime();
            sequenceTime += dt;
            phase = GetPhase(sequenceTime);
            float coreT = GetCoreTime(sequenceTime);

            if (phase == SequencePhase::PrimaryFission) {
                for (int i = 0; i < 6; ++i) {
                    float a = 2.0f * PI * (0.09f * static_cast<float>(i) + coreT * 0.5f);
                    Vector3 dir = {std::cos(a), 0.3f * std::sin(2.2f * a), std::sin(a)};
                    plasma.push_back({primary, Vector3Scale(dir, 2.6f), 1.8f, 0.038f, Color{255, 170, 96, 255}});
                }
            }

            if (phase == SequencePhase::RadiationTransfer) {
                float transferT = std::clamp((coreT - kPrimaryDuration) / kTransferDuration, 0.0f, 1.0f);
                for (int i = 0; i < 7; ++i) {
                    float p = std::clamp(transferT + 0.08f * static_cast<float>(i), 0.0f, 1.0f);
                    Vector3 channelPoint = Vector3Lerp(primary, secondary, p);
                    float swirl = 0.07f * std::sin(18.0f * p + coreT * 7.0f + static_cast<float>(i));
                    Vector3 tangent = {swirl, 0.0f, -swirl};
                    plasma.push_back({channelPoint, Vector3Scale(Vector3Normalize(Vector3Add({0.0f, 1.0f, 0.0f}, tangent)), 3.0f), 1.5f, 0.032f, Color{255, 190, 110, 255}});
                }
            }

            if (phase == SequencePhase::SecondaryFusion) {
                for (int i = 0; i < 12; ++i) {
                    float a = 2.0f * PI * (0.11f * static_cast<float>(i) + coreT * 0.68f);
                    Vector3 dir = {std::cos(a), 0.28f * std::sin(3.1f * a), std::sin(a)};
                    plasma.push_back({secondary, Vector3Scale(dir, 4.8f), 2.4f, 0.043f, Color{255, 232, 130, 255}});
                }
            }

            for (Plasma& p : plasma) {
                p.pos = Vector3Add(p.pos, Vector3Scale(p.vel, dt));
                p.vel = Vector3Scale(p.vel, 0.985f);
                p.life -= dt;
            }
            plasma.erase(std::remove_if(plasma.begin(), plasma.end(), [](const Plasma& p) { return p.life <= 0.0f; }), plasma.end());
        }

        if (phase == SequencePhase::Exterior || phase == SequencePhase::ZoomToCore) {
            ApplyScriptedCamera(&camera, sequenceTime);
        } else {
            float coreT = GetCoreTime(sequenceTime);
            Vector3 baseTarget = primary;
            if (phase == SequencePhase::RadiationTransfer) {
                float transferT = std::clamp((coreT - kPrimaryDuration) / kTransferDuration, 0.0f, 1.0f);
                baseTarget = Vector3Lerp(primary, secondary, transferT);
            } else if (phase == SequencePhase::SecondaryFusion) {
                baseTarget = secondary;
            }
            camera.target = baseTarget;
            UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);
        }

        BeginDrawing();
        ClearBackground(Color{8, 9, 13, 255});

        BeginMode3D(camera);

        float zoom = SmoothStep((sequenceTime - kExteriorDuration) / kZoomDuration);
        unsigned char shellAlpha = static_cast<unsigned char>(std::clamp(255.0f - 162.0f * zoom, 72.0f, 255.0f));
        DrawBombExterior(shellAlpha);

        unsigned char chamberAlpha = static_cast<unsigned char>(std::clamp(60.0f + 130.0f * zoom, 35.0f, 230.0f));
        DrawCylinder({0.0f, 0.0f, 0.0f}, 0.72f, 0.72f, 5.8f, 30, Color{105, 150, 208, chamberAlpha});
        DrawCylinderWires({0.0f, 0.0f, 0.0f}, 0.8f, 0.8f, 5.8f, 30, Color{175, 205, 240, static_cast<unsigned char>(chamberAlpha / 2)});

        float coreT = GetCoreTime(sequenceTime);
        float transferT = std::clamp((coreT - kPrimaryDuration) / kTransferDuration, 0.0f, 1.0f);
        float compression = 0.0f;
        if (phase == SequencePhase::RadiationTransfer) compression = transferT;
        if (phase == SequencePhase::SecondaryFusion) compression = 1.0f;

        Color primaryC = (phase == SequencePhase::PrimaryFission || phase == SequencePhase::RadiationTransfer || phase == SequencePhase::SecondaryFusion)
                             ? Color{255, 160, 92, 240}
                             : Color{120, 220, 255, 210};
        Color secondaryC = (phase == SequencePhase::SecondaryFusion)
                               ? Color{255, 232, 128, 245}
                               : Color{130, 182, 255, 170};

        DrawSphere(primary, 0.35f, primaryC);
        DrawSphere(secondary, 0.48f - 0.14f * compression, secondaryC);
        DrawCylinderEx(primary, secondary, 0.11f + 0.05f * compression, 0.15f + 0.06f * compression, 22, Color{255, 178, 105, static_cast<unsigned char>(70 + 115 * compression)});

        if (phase == SequencePhase::RadiationTransfer) {
            DrawSphere(Vector3Lerp(primary, secondary, transferT), 0.07f, Color{255, 200, 130, 230});
        }

        for (const Plasma& p : plasma) {
            unsigned char a = static_cast<unsigned char>(std::clamp(90.0f * p.life, 20.0f, 255.0f));
            Color c = p.color;
            c.a = a;
            DrawSphere(p.pos, p.radius + 0.008f * p.life, c);
        }

        float blast = (phase == SequencePhase::SecondaryFusion) ? std::min(9.0f, (coreT - (kPrimaryDuration + kTransferDuration)) * 2.2f) : 0.0f;
        if (blast > 0.05f) {
            DrawSphere(secondary, blast, Color{255, 176, 105, 22});
        }

        EndMode3D();

        DrawText("Hydrogen Bomb Two-Stage Core Sequence (Conceptual)", 20, 18, 29, Color{235, 240, 250, 255});
        DrawText("Auto fly-in to casing/core | in core: hold left mouse orbit, wheel zoom | P pause | R reset", 20, 54, 19, Color{170, 184, 204, 255});

        std::ostringstream os;
        os << std::fixed << std::setprecision(2) << "t=" << sequenceTime
           << "  phase=" << PhaseText(phase)
           << "  plasma=" << plasma.size();
        if (paused) os << "  [PAUSED]";
        DrawText(os.str().c_str(), 20, 82, 20, Color{255, 210, 150, 255});
        DrawFPS(20, 110);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
