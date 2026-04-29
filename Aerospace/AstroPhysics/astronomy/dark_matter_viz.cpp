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

struct Star {
    float r;
    float theta;
    float size;
    Color color;
};

void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }

    *distance -= GetMouseWheelMove() * 0.6f;
    *distance = std::clamp(*distance, 5.0f, 40.0f);

    float cp = std::cos(*pitch);
    Vector3 offset = {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}

float BaryonicMass(float r) {
    return 9.0f * (1.0f - std::exp(-r / 1.5f));
}

float HaloMass(float r, float haloStrength) {
    return haloStrength * 2.4f * r;
}

float OrbitSpeed(float r, float enclosedMass) {
    return std::sqrt(std::max(0.001f, enclosedMass / r));
}

void DrawCircleXZ(float radius, int segs, Color c) {
    for (int i = 0; i < segs; ++i) {
        float a0 = 2.0f * PI * static_cast<float>(i) / static_cast<float>(segs);
        float a1 = 2.0f * PI * static_cast<float>(i + 1) / static_cast<float>(segs);
        DrawLine3D(
            {radius * std::cos(a0), 0.0f, radius * std::sin(a0)},
            {radius * std::cos(a1), 0.0f, radius * std::sin(a1)},
            c
        );
    }
}

std::string Hud(float haloStrength, float timeScale, bool paused) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(2)
       << "halo=" << haloStrength
       << "  timeScale=" << timeScale << "x";
    if (paused) os << "  [PAUSED]";
    return os.str();
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Dark Matter Rotation Curves 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {10.0f, 6.0f, 11.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.82f;
    float camPitch = 0.33f;
    float camDistance = 16.0f;

    std::vector<Star> stars;
    stars.reserve(420);
    for (int i = 0; i < 420; ++i) {
        float t = static_cast<float>(i) / 419.0f;
        float r = 0.7f + 8.6f * std::pow(t, 0.62f);
        float theta = 2.0f * PI * std::fmod(i * 0.6180339f, 1.0f);
        float size = 0.018f + 0.022f * (1.0f - t);
        unsigned char c = static_cast<unsigned char>(140 + 100 * (1.0f - t));
        stars.push_back({r, theta, size, Color{c, c, 255, 240}});
    }

    float haloStrength = 1.0f;
    float timeScale = 1.0f;
    bool paused = false;
    bool showBaryonicGhost = true;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            haloStrength = 1.0f;
            timeScale = 1.0f;
        }
        if (IsKeyPressed(KEY_G)) showBaryonicGhost = !showBaryonicGhost;

        if (IsKeyPressed(KEY_LEFT_BRACKET)) haloStrength = std::max(0.0f, haloStrength - 0.08f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) haloStrength = std::min(2.5f, haloStrength + 0.08f);

        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) timeScale = std::max(0.2f, timeScale - 0.2f);
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) timeScale = std::min(6.0f, timeScale + 0.2f);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        float dt = GetFrameTime() * timeScale;
        if (!paused) {
            for (Star& s : stars) {
                float mBar = BaryonicMass(s.r);
                float mDM = HaloMass(s.r, haloStrength);
                float vTotal = OrbitSpeed(s.r, mBar + mDM);
                s.theta += (vTotal / s.r) * dt;
            }
        }

        BeginDrawing();
        ClearBackground(Color{5, 8, 16, 255});

        BeginMode3D(camera);

        DrawSphere({0.0f, 0.0f, 0.0f}, 0.45f, Color{255, 215, 140, 255});
        DrawSphere({0.0f, 0.0f, 0.0f}, 3.2f + 2.3f * haloStrength, Color{90, 130, 220, 25});

        for (float r = 1.0f; r <= 9.0f; r += 1.0f) {
            DrawCircleXZ(r, 96, Color{70, 95, 140, 45});
        }

        for (const Star& s : stars) {
            Vector3 p = {s.r * std::cos(s.theta), 0.0f, s.r * std::sin(s.theta)};
            DrawSphere(p, s.size, s.color);

            if (showBaryonicGhost) {
                float vBar = OrbitSpeed(s.r, BaryonicMass(s.r));
                float thetaBar = s.theta - 0.85f * (OrbitSpeed(s.r, BaryonicMass(s.r) + HaloMass(s.r, haloStrength)) - vBar) / s.r;
                Vector3 pb = {s.r * std::cos(thetaBar), 0.0f, s.r * std::sin(thetaBar)};
                DrawSphere(pb, s.size * 0.7f, Color{255, 110, 110, 85});
            }
        }

        EndMode3D();

        DrawText("Dark Matter: Galaxy Rotation Curves (3D)", 20, 18, 29, Color{232, 238, 248, 255});
        DrawText("Hold left mouse: orbit | wheel: zoom | [ ] halo | +/- time | G ghost(baryonic) | P pause | R reset", 20, 54, 19, Color{164, 183, 210, 255});
        std::string hud = Hud(haloStrength, timeScale, paused);
        DrawText(hud.c_str(), 20, 82, 21, Color{126, 224, 255, 255});
        DrawText("Blue stars: with dark matter halo | Red ghosts: baryonic-only speed", 20, 110, 18, Color{200, 180, 180, 255});
        DrawFPS(20, 138);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
