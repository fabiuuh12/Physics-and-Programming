#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <iomanip>
#include <sstream>
#include <string>

namespace {

constexpr int kScreenWidth = 1280;
constexpr int kScreenHeight = 820;
constexpr float kSoftening = 0.05f;
constexpr int kTrailMax = 1000;

struct LagrangePoints {
    Vector3 l1;
    Vector3 l2;
    Vector3 l3;
    Vector3 l4;
    Vector3 l5;
};

struct Probe {
    bool active = false;
    Vector3 pos = {0.0f, 0.0f, 0.0f};
    Vector3 vel = {0.0f, 0.0f, 0.0f};
    std::deque<Vector3> trail;
};

void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 delta = GetMouseDelta();
        *yaw -= delta.x * 0.0035f;
        *pitch += delta.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }

    *distance -= GetMouseWheelMove() * 0.7f;
    *distance = std::clamp(*distance, 4.5f, 28.0f);

    float cp = std::cos(*pitch);
    Vector3 offset = {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}

float PotentialHeight(float x, float z, float mu, float sheetScale) {
    float x1 = -mu;
    float x2 = 1.0f - mu;
    float r1 = std::sqrt((x - x1) * (x - x1) + z * z + kSoftening * kSoftening);
    float r2 = std::sqrt((x - x2) * (x - x2) + z * z + kSoftening * kSoftening);
    float pseudo = (1.0f - mu) / r1 + mu / r2 + 0.17f * (x * x + z * z);
    return std::max(-4.4f, -sheetScale * pseudo);
}

void DrawPotentialSheet(float mu, float sheetScale) {
    constexpr float kExtent = 3.2f;
    constexpr int kGrid = 58;

    for (int i = 0; i < kGrid; ++i) {
        float z = -kExtent + 2.0f * kExtent * static_cast<float>(i) / static_cast<float>(kGrid - 1);
        for (int j = 0; j < kGrid - 1; ++j) {
            float x0 = -kExtent + 2.0f * kExtent * static_cast<float>(j) / static_cast<float>(kGrid - 1);
            float x1 = -kExtent + 2.0f * kExtent * static_cast<float>(j + 1) / static_cast<float>(kGrid - 1);
            Vector3 p0 = {x0, PotentialHeight(x0, z, mu, sheetScale), z};
            Vector3 p1 = {x1, PotentialHeight(x1, z, mu, sheetScale), z};
            float glow = 1.0f - std::min(1.0f, std::fabs(p0.y) / 4.4f);
            Color c = {
                static_cast<unsigned char>(45 + 60 * glow),
                static_cast<unsigned char>(95 + 90 * glow),
                static_cast<unsigned char>(160 + 85 * glow),
                static_cast<unsigned char>(70 + 100 * glow),
            };
            DrawLine3D(p0, p1, c);
        }
    }

    for (int j = 0; j < kGrid; ++j) {
        float x = -kExtent + 2.0f * kExtent * static_cast<float>(j) / static_cast<float>(kGrid - 1);
        for (int i = 0; i < kGrid - 1; ++i) {
            float z0 = -kExtent + 2.0f * kExtent * static_cast<float>(i) / static_cast<float>(kGrid - 1);
            float z1 = -kExtent + 2.0f * kExtent * static_cast<float>(i + 1) / static_cast<float>(kGrid - 1);
            Vector3 p0 = {x, PotentialHeight(x, z0, mu, sheetScale), z0};
            Vector3 p1 = {x, PotentialHeight(x, z1, mu, sheetScale), z1};
            float glow = 1.0f - std::min(1.0f, std::fabs(p0.y) / 4.4f);
            Color c = {
                static_cast<unsigned char>(40 + 55 * glow),
                static_cast<unsigned char>(85 + 82 * glow),
                static_cast<unsigned char>(145 + 90 * glow),
                static_cast<unsigned char>(56 + 86 * glow),
            };
            DrawLine3D(p0, p1, c);
        }
    }
}

void RotatingFrameAcceleration(float x, float z, float vx, float vz, float mu, float* ax, float* az) {
    float x1 = -mu;
    float x2 = 1.0f - mu;
    float r1sq = (x - x1) * (x - x1) + z * z + kSoftening * kSoftening;
    float r2sq = (x - x2) * (x - x2) + z * z + kSoftening * kSoftening;
    float r1 = std::sqrt(r1sq);
    float r2 = std::sqrt(r2sq);
    float r13 = r1 * r1 * r1;
    float r23 = r2 * r2 * r2;

    float dOmegaDx = x - (1.0f - mu) * (x - x1) / r13 - mu * (x - x2) / r23;
    float dOmegaDz = z - (1.0f - mu) * z / r13 - mu * z / r23;

    *ax = 2.0f * vz + dOmegaDx;
    *az = -2.0f * vx + dOmegaDz;
}

float DOmegaDxAxis(float x, float mu) {
    float ax = 0.0f;
    float az = 0.0f;
    RotatingFrameAcceleration(x, 0.0f, 0.0f, 0.0f, mu, &ax, &az);
    return ax;
}

bool FindRootBisection(float mu, float lo, float hi, float* outX) {
    float fLo = DOmegaDxAxis(lo, mu);
    float fHi = DOmegaDxAxis(hi, mu);
    if (fLo * fHi > 0.0f) return false;

    for (int i = 0; i < 80; ++i) {
        float mid = 0.5f * (lo + hi);
        float fMid = DOmegaDxAxis(mid, mu);
        if (std::fabs(fMid) < 1e-6f) {
            *outX = mid;
            return true;
        }
        if (fLo * fMid <= 0.0f) {
            hi = mid;
            fHi = fMid;
        } else {
            lo = mid;
            fLo = fMid;
        }
    }

    *outX = 0.5f * (lo + hi);
    return true;
}

LagrangePoints ComputeLagrangePoints(float mu) {
    float x1 = -mu;
    float x2 = 1.0f - mu;
    float root1 = 0.5f * (x1 + x2);
    float root2 = x2 + 0.4f;
    float root3 = x1 - 0.4f;

    FindRootBisection(mu, x1 + 0.02f, x2 - 0.02f, &root1);
    FindRootBisection(mu, x2 + 0.02f, 3.0f, &root2);
    FindRootBisection(mu, -3.0f, x1 - 0.02f, &root3);

    float triX = 0.5f - mu;
    float triZ = std::sqrt(3.0f) * 0.5f;

    return {
        {root1, 0.08f, 0.0f},
        {root2, 0.08f, 0.0f},
        {root3, 0.08f, 0.0f},
        {triX, 0.08f, triZ},
        {triX, 0.08f, -triZ},
    };
}

void ResetProbe(Probe* probe) {
    probe->active = false;
    probe->pos = {0.0f, 0.0f, 0.0f};
    probe->vel = {0.0f, 0.0f, 0.0f};
    probe->trail.clear();
}

void LaunchProbeAt(const Vector3& l, int idx, Probe* probe) {
    static const std::array<Vector3, 5> kOffsets = {
        Vector3{0.0f, 0.08f, 0.03f},
        Vector3{0.0f, 0.08f, -0.03f},
        Vector3{0.0f, 0.08f, 0.025f},
        Vector3{0.03f, 0.08f, 0.0f},
        Vector3{-0.03f, 0.08f, 0.0f},
    };
    static const std::array<Vector3, 5> kVels = {
        Vector3{0.0f, 0.0f, 0.05f},
        Vector3{0.0f, 0.0f, -0.04f},
        Vector3{0.0f, 0.0f, 0.035f},
        Vector3{-0.04f, 0.0f, 0.0f},
        Vector3{0.04f, 0.0f, 0.0f},
    };

    probe->active = true;
    probe->pos = Vector3Add(l, kOffsets[idx]);
    probe->vel = kVels[idx];
    probe->trail.clear();
    probe->trail.push_back(probe->pos);
}

void StepProbe(Probe* probe, float dt, float mu) {
    if (!probe->active) return;

    int substeps = std::max(1, static_cast<int>(std::ceil(dt / 0.003f)));
    float h = dt / static_cast<float>(substeps);

    for (int i = 0; i < substeps; ++i) {
        float x = probe->pos.x;
        float z = probe->pos.z;
        float vx = probe->vel.x;
        float vz = probe->vel.z;

        auto deriv = [mu](float px, float pz, float pvx, float pvz) {
            float ax = 0.0f;
            float az = 0.0f;
            RotatingFrameAcceleration(px, pz, pvx, pvz, mu, &ax, &az);
            return std::array<float, 4>{pvx, pvz, ax, az};
        };

        auto k1 = deriv(x, z, vx, vz);
        auto k2 = deriv(x + 0.5f * h * k1[0], z + 0.5f * h * k1[1], vx + 0.5f * h * k1[2], vz + 0.5f * h * k1[3]);
        auto k3 = deriv(x + 0.5f * h * k2[0], z + 0.5f * h * k2[1], vx + 0.5f * h * k2[2], vz + 0.5f * h * k2[3]);
        auto k4 = deriv(x + h * k3[0], z + h * k3[1], vx + h * k3[2], vz + h * k3[3]);

        x += h * (k1[0] + 2.0f * k2[0] + 2.0f * k3[0] + k4[0]) / 6.0f;
        z += h * (k1[1] + 2.0f * k2[1] + 2.0f * k3[1] + k4[1]) / 6.0f;
        vx += h * (k1[2] + 2.0f * k2[2] + 2.0f * k3[2] + k4[2]) / 6.0f;
        vz += h * (k1[3] + 2.0f * k2[3] + 2.0f * k3[3] + k4[3]) / 6.0f;

        probe->pos = {x, 0.08f, z};
        probe->vel = {vx, 0.0f, vz};
    }

    probe->trail.push_back(probe->pos);
    if (static_cast<int>(probe->trail.size()) > kTrailMax) probe->trail.pop_front();

    float r = std::sqrt(probe->pos.x * probe->pos.x + probe->pos.z * probe->pos.z);
    if (r > 4.5f) probe->active = false;
}

void DrawTrail(const std::deque<Vector3>& trail, Color color) {
    if (trail.size() < 2) return;
    for (size_t i = 1; i < trail.size(); ++i) {
        float fade = static_cast<float>(i) / static_cast<float>(trail.size());
        Color c = color;
        c.a = static_cast<unsigned char>(25 + 170 * fade);
        DrawLine3D(trail[i - 1], trail[i], c);
    }
}

std::string Hud(float mu, float speed, float sheetScale, bool paused, bool activeProbe) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(2)
       << "mu=" << mu
       << "  speed=" << speed << "x"
       << "  warp=" << sheetScale
       << "  probe=" << (activeProbe ? "active" : "idle");
    if (paused) os << "  [PAUSED]";
    return os.str();
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Lagrange Points + Gravity Potential Sheet 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.8f, 5.2f, 8.2f};
    camera.target = {0.0f, -0.6f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.80f;
    float camPitch = 0.44f;
    float camDistance = 12.0f;

    float mu = 0.18f;
    float speed = 1.0f;
    float sheetScale = 1.0f;
    bool paused = false;
    bool showSheet = true;
    bool showHelp = true;
    bool showTrails = true;

    Probe probe{};
    LagrangePoints lp = ComputeLagrangePoints(mu);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) ResetProbe(&probe);
        if (IsKeyPressed(KEY_H)) showHelp = !showHelp;
        if (IsKeyPressed(KEY_W)) showSheet = !showSheet;
        if (IsKeyPressed(KEY_T)) showTrails = !showTrails;
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) speed = std::max(0.25f, speed - 0.25f);
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) speed = std::min(8.0f, speed + 0.25f);
        if (IsKeyPressed(KEY_COMMA)) sheetScale = std::max(0.45f, sheetScale - 0.05f);
        if (IsKeyPressed(KEY_PERIOD)) sheetScale = std::min(1.9f, sheetScale + 0.05f);
        if (IsKeyPressed(KEY_LEFT_BRACKET)) mu = std::max(0.05f, mu - 0.01f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) mu = std::min(0.45f, mu + 0.01f);

        lp = ComputeLagrangePoints(mu);
        if (IsKeyPressed(KEY_ONE)) LaunchProbeAt(lp.l1, 0, &probe);
        if (IsKeyPressed(KEY_TWO)) LaunchProbeAt(lp.l2, 1, &probe);
        if (IsKeyPressed(KEY_THREE)) LaunchProbeAt(lp.l3, 2, &probe);
        if (IsKeyPressed(KEY_FOUR)) LaunchProbeAt(lp.l4, 3, &probe);
        if (IsKeyPressed(KEY_FIVE)) LaunchProbeAt(lp.l5, 4, &probe);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        if (!paused) {
            float dt = GetFrameTime() * speed;
            StepProbe(&probe, dt, mu);
        }

        Vector3 p1 = {-mu, 0.12f, 0.0f};
        Vector3 p2 = {1.0f - mu, 0.12f, 0.0f};
        float m1 = 1.0f - mu;
        float m2 = mu;
        float r1 = 0.16f + 0.28f * std::pow(m1, 0.33f);
        float r2 = 0.16f + 0.28f * std::pow(m2, 0.33f);

        BeginDrawing();
        ClearBackground(Color{5, 8, 18, 255});

        BeginMode3D(camera);

        if (showSheet) DrawPotentialSheet(mu, sheetScale);

        DrawLine3D({p1.x, PotentialHeight(p1.x, p1.z, mu, sheetScale), p1.z}, p1, Color{255, 196, 120, 120});
        DrawLine3D({p2.x, PotentialHeight(p2.x, p2.z, mu, sheetScale), p2.z}, p2, Color{150, 210, 255, 120});

        DrawSphere({p1.x, PotentialHeight(p1.x, p1.z, mu, sheetScale), p1.z}, 0.08f, Color{255, 182, 95, 90});
        DrawSphere({p2.x, PotentialHeight(p2.x, p2.z, mu, sheetScale), p2.z}, 0.06f, Color{120, 175, 255, 100});

        DrawSphere(p1, r1, Color{255, 200, 105, 255});
        DrawSphereWires(p1, r1 * 1.2f, 10, 10, Color{255, 220, 150, 120});
        DrawSphere(p2, r2, Color{100, 165, 255, 255});
        DrawSphereWires(p2, r2 * 1.2f, 10, 10, Color{180, 220, 255, 120});

        std::array<Vector3, 5> points = {lp.l1, lp.l2, lp.l3, lp.l4, lp.l5};
        std::array<Color, 5> colors = {
            Color{255, 175, 110, 255}, Color{255, 175, 110, 255}, Color{255, 175, 110, 255},
            Color{140, 255, 180, 255}, Color{140, 255, 180, 255},
        };
        for (int i = 0; i < 5; ++i) {
            DrawSphere(points[i], 0.055f, colors[i]);
            DrawSphereWires(points[i], 0.075f, 8, 8, Color{220, 240, 255, 130});
        }

        if (showTrails) DrawTrail(probe.trail, Color{140, 230, 255, 255});
        if (probe.active) DrawSphere(probe.pos, 0.045f, Color{255, 95, 130, 255});

        EndMode3D();

        std::array<const char*, 5> labels = {"L1", "L2", "L3", "L4", "L5"};
        for (int i = 0; i < 5; ++i) {
            Vector2 s = GetWorldToScreen(points[i], camera);
            DrawText(labels[i], static_cast<int>(s.x) - 8, static_cast<int>(s.y) - 10, 18, Color{225, 235, 250, 240});
        }

        DrawText("Lagrange Points in a Rotating Two-Body Gravity Field", 20, 18, 30, Color{232, 238, 248, 255});
        if (showHelp) {
            DrawText("Hold left mouse: orbit | wheel: zoom | 1..5 launch probe | [ ] mass ratio | +/- speed | , . warp | W sheet | T trails | P pause | R reset | H help",
                     20, 54, 18, Color{164, 183, 210, 255});
        } else {
            DrawText("Press H to show controls", 20, 54, 18, Color{164, 183, 210, 255});
        }

        std::string hud = Hud(mu, speed, sheetScale, paused, probe.active);
        DrawText(hud.c_str(), 20, 82, 21, Color{126, 224, 255, 255});
        DrawText("L4/L5 are generally stable, L1/L2/L3 are saddle points", 20, 110, 18, Color{192, 206, 226, 255});
        DrawFPS(20, 138);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
