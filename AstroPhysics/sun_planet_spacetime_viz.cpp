#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <iomanip>
#include <sstream>
#include <string>

namespace {

constexpr int kScreenWidth = 1280;
constexpr int kScreenHeight = 820;

constexpr float kPi = 3.14159265358979323846f;

struct SimState {
    float angle;
    float speed;
    bool paused;
    bool showHelp;
    float sheetScale;
};

void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 delta = GetMouseDelta();
        *yaw -= delta.x * 0.0035f;
        *pitch += delta.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }

    *distance -= GetMouseWheelMove() * 0.7f;
    *distance = std::clamp(*distance, 6.0f, 40.0f);

    float cp = std::cos(*pitch);
    Vector3 offset = {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}

float WarpContribution(float dx, float dz, float strength, float core) {
    float r2 = dx * dx + dz * dz + core * core;
    return -strength / std::sqrt(r2);
}

float SpacetimeHeight(float x, float z, Vector3 sunPos, Vector3 planetPos, float sheetScale) {
    float sunWell = WarpContribution(x - sunPos.x, z - sunPos.z, 2.3f, 0.55f);
    float planetWell = WarpContribution(x - planetPos.x, z - planetPos.z, 0.8f, 0.28f);
    float h = (sunWell + planetWell) * sheetScale;
    return std::max(-4.6f, h);
}

void DrawSpacetimeSheet(Vector3 sunPos, Vector3 planetPos, float sheetScale) {
    constexpr int kGrid = 54;
    constexpr float kExtent = 11.0f;

    for (int i = 0; i < kGrid; ++i) {
        float z = -kExtent + 2.0f * kExtent * static_cast<float>(i) / static_cast<float>(kGrid - 1);
        for (int j = 0; j < kGrid - 1; ++j) {
            float x0 = -kExtent + 2.0f * kExtent * static_cast<float>(j) / static_cast<float>(kGrid - 1);
            float x1 = -kExtent + 2.0f * kExtent * static_cast<float>(j + 1) / static_cast<float>(kGrid - 1);
            Vector3 p0 = {x0, SpacetimeHeight(x0, z, sunPos, planetPos, sheetScale), z};
            Vector3 p1 = {x1, SpacetimeHeight(x1, z, sunPos, planetPos, sheetScale), z};
            float glow = 1.0f - std::min(1.0f, std::fabs(p0.y) / 4.6f);
            Color c = {
                static_cast<unsigned char>(50 + 60 * glow),
                static_cast<unsigned char>(110 + 80 * glow),
                static_cast<unsigned char>(185 + 50 * glow),
                static_cast<unsigned char>(120 + 80 * glow),
            };
            DrawLine3D(p0, p1, c);
        }
    }

    for (int j = 0; j < kGrid; ++j) {
        float x = -kExtent + 2.0f * kExtent * static_cast<float>(j) / static_cast<float>(kGrid - 1);
        for (int i = 0; i < kGrid - 1; ++i) {
            float z0 = -kExtent + 2.0f * kExtent * static_cast<float>(i) / static_cast<float>(kGrid - 1);
            float z1 = -kExtent + 2.0f * kExtent * static_cast<float>(i + 1) / static_cast<float>(kGrid - 1);
            Vector3 p0 = {x, SpacetimeHeight(x, z0, sunPos, planetPos, sheetScale), z0};
            Vector3 p1 = {x, SpacetimeHeight(x, z1, sunPos, planetPos, sheetScale), z1};
            float glow = 1.0f - std::min(1.0f, std::fabs(p0.y) / 4.6f);
            Color c = {
                static_cast<unsigned char>(45 + 55 * glow),
                static_cast<unsigned char>(100 + 85 * glow),
                static_cast<unsigned char>(170 + 65 * glow),
                static_cast<unsigned char>(95 + 65 * glow),
            };
            DrawLine3D(p0, p1, c);
        }
    }
}

void DrawOrbitTrail(const std::deque<Vector3>& trail) {
    if (trail.size() < 2) return;
    for (size_t i = 1; i < trail.size(); ++i) {
        float fade = static_cast<float>(i) / static_cast<float>(trail.size());
        Color c = {130, 205, 255, static_cast<unsigned char>(35 + 180 * fade)};
        DrawLine3D(trail[i - 1], trail[i], c);
    }
}

std::string Hud(const SimState& s, float period, float radiusA, float radiusB) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(2)
       << "speed=" << s.speed << "x"
       << "  period~" << period
       << "s  orbit=(" << radiusA << "," << radiusB << ")"
       << "  warp=" << s.sheetScale;
    if (s.paused) os << "  [PAUSED]";
    return os.str();
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Sun-Planet Spacetime Curvature 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {12.0f, 8.5f, 11.0f};
    camera.target = {0.0f, -1.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.77f;
    float camPitch = 0.44f;
    float camDistance = 17.0f;

    SimState sim{};
    sim.angle = 0.0f;
    sim.speed = 1.0f;
    sim.paused = false;
    sim.showHelp = true;
    sim.sheetScale = 1.0f;

    constexpr float kOrbitA = 6.2f;
    constexpr float kOrbitB = 5.3f;
    constexpr float kOmega = 0.42f;
    constexpr int kTrailMax = 900;

    std::deque<Vector3> trail;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) sim.paused = !sim.paused;
        if (IsKeyPressed(KEY_R)) {
            sim.angle = 0.0f;
            sim.speed = 1.0f;
            sim.sheetScale = 1.0f;
            sim.paused = false;
            trail.clear();
        }
        if (IsKeyPressed(KEY_H)) sim.showHelp = !sim.showHelp;
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) sim.speed = std::max(0.25f, sim.speed - 0.25f);
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) sim.speed = std::min(8.0f, sim.speed + 0.25f);
        if (IsKeyPressed(KEY_LEFT_BRACKET)) sim.sheetScale = std::max(0.45f, sim.sheetScale - 0.05f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) sim.sheetScale = std::min(1.65f, sim.sheetScale + 0.05f);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        float dt = GetFrameTime() * sim.speed;
        if (!sim.paused) {
            sim.angle += kOmega * dt;
        }

        Vector3 sunPos = {0.0f, 0.58f, 0.0f};
        Vector3 planetPos = {
            kOrbitA * std::cos(sim.angle),
            0.22f,
            kOrbitB * std::sin(sim.angle),
        };

        trail.push_back(planetPos);
        if (static_cast<int>(trail.size()) > kTrailMax) trail.pop_front();

        float sunSheetY = SpacetimeHeight(sunPos.x, sunPos.z, sunPos, planetPos, sim.sheetScale);
        float planetSheetY = SpacetimeHeight(planetPos.x, planetPos.z, sunPos, planetPos, sim.sheetScale);

        BeginDrawing();
        ClearBackground(Color{5, 8, 18, 255});

        BeginMode3D(camera);

        DrawSpacetimeSheet(sunPos, planetPos, sim.sheetScale);
        DrawOrbitTrail(trail);

        DrawLine3D({sunPos.x, sunSheetY, sunPos.z}, sunPos, Color{255, 195, 115, 120});
        DrawLine3D({planetPos.x, planetSheetY, planetPos.z}, planetPos, Color{140, 210, 255, 130});

        DrawSphere({sunPos.x, sunSheetY, sunPos.z}, 0.15f, Color{255, 180, 75, 70});
        DrawSphere({planetPos.x, planetSheetY, planetPos.z}, 0.08f, Color{100, 165, 240, 90});

        DrawSphere(sunPos, 0.78f, Color{255, 196, 95, 255});
        DrawSphereWires(sunPos, 0.93f, 16, 16, Color{255, 210, 130, 90});

        DrawSphere(planetPos, 0.26f, Color{95, 160, 255, 255});
        DrawSphereWires(planetPos, 0.31f, 10, 10, Color{180, 220, 255, 100});

        EndMode3D();

        DrawText("Planet Orbiting the Sun with Spacetime Curvature", 20, 18, 30, Color{232, 238, 248, 255});
        if (sim.showHelp) {
            DrawText("Hold left mouse: orbit camera | wheel: zoom | +/- speed | [ ] warp | P pause | R reset | H help",
                     20, 56, 19, Color{164, 183, 210, 255});
        } else {
            DrawText("Press H to show controls", 20, 56, 19, Color{164, 183, 210, 255});
        }

        float orbitalPeriod = 2.0f * kPi / kOmega;
        std::string hud = Hud(sim, orbitalPeriod, kOrbitA, kOrbitB);
        DrawText(hud.c_str(), 20, 84, 21, Color{126, 224, 255, 255});
        DrawFPS(20, 116);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
