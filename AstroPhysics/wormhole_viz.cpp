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

struct FlowParticle {
    float u;
    float theta;
    float speed;
    float swirl;
    Color color;
};

void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 delta = GetMouseDelta();
        *yaw -= delta.x * 0.0035f;
        *pitch += delta.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }

    *distance -= GetMouseWheelMove() * 0.6f;
    *distance = std::clamp(*distance, 4.0f, 30.0f);

    float cp = std::cos(*pitch);
    Vector3 offset = {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}

float RadiusProfile(float u, float throatRadius, float flare) {
    return throatRadius + flare * (u * u);
}

Vector3 WormholePoint(float u, float theta, float throatRadius, float flare) {
    float r = RadiusProfile(u, throatRadius, flare);
    return {r * std::cos(theta), r * std::sin(theta), u};
}

void DrawWormholeSurface(float throatRadius, float flare) {
    const int rings = 52;
    const int segs = 64;
    const float uMin = -4.8f;
    const float uMax = 4.8f;

    for (int i = 0; i < rings - 1; ++i) {
        float u0 = uMin + (uMax - uMin) * static_cast<float>(i) / static_cast<float>(rings - 1);
        float u1 = uMin + (uMax - uMin) * static_cast<float>(i + 1) / static_cast<float>(rings - 1);
        for (int j = 0; j < segs; ++j) {
            float t0 = 2.0f * PI * static_cast<float>(j) / static_cast<float>(segs);
            float t1 = 2.0f * PI * static_cast<float>(j + 1) / static_cast<float>(segs);

            Vector3 p00 = WormholePoint(u0, t0, throatRadius, flare);
            Vector3 p01 = WormholePoint(u0, t1, throatRadius, flare);
            Vector3 p10 = WormholePoint(u1, t0, throatRadius, flare);

            float glow = 0.22f + 0.58f * (1.0f - std::fabs(u0) / 4.8f);
            Color c = Color{
                static_cast<unsigned char>(70 + 70 * glow),
                static_cast<unsigned char>(110 + 90 * glow),
                static_cast<unsigned char>(170 + 70 * glow),
                static_cast<unsigned char>(40 + 80 * glow)
            };

            DrawTriangle3D(p00, p10, p01, c);
        }
    }

    for (int k = 0; k < 10; ++k) {
        float u = -4.8f + 9.6f * static_cast<float>(k) / 9.0f;
        int segCount = 80;
        for (int j = 0; j < segCount; ++j) {
            float a0 = 2.0f * PI * static_cast<float>(j) / static_cast<float>(segCount);
            float a1 = 2.0f * PI * static_cast<float>(j + 1) / static_cast<float>(segCount);
            Vector3 p0 = WormholePoint(u, a0, throatRadius, flare);
            Vector3 p1 = WormholePoint(u, a1, throatRadius, flare);
            DrawLine3D(p0, p1, Color{90, 150, 220, 70});
        }
    }
}

std::string Hud(float throatRadius, float flare, int particles, bool paused) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(2)
       << "throat=" << throatRadius
       << "  flare=" << flare
       << "  particles=" << particles;
    if (paused) os << "  [PAUSED]";
    return os.str();
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Wormhole 3D Visualization - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.5f, 4.8f, 8.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.82f;
    float camPitch = 0.34f;
    float camDistance = 13.5f;

    float throatRadius = 0.85f;
    float flare = 0.22f;
    bool paused = false;

    std::vector<FlowParticle> flow;
    flow.reserve(320);
    for (int i = 0; i < 320; ++i) {
        float u = -4.7f + 9.4f * (static_cast<float>(i) / 319.0f);
        float theta = 2.0f * PI * std::fmod(i * 0.6180339f, 1.0f);
        float speed = 0.4f + 0.9f * std::fmod(i * 0.371f, 1.0f);
        float swirl = 0.8f + 1.4f * std::fmod(i * 0.529f, 1.0f);
        Color c = Color{
            static_cast<unsigned char>(90 + (i * 17) % 120),
            static_cast<unsigned char>(160 + (i * 11) % 90),
            static_cast<unsigned char>(220 + (i * 7) % 35),
            230
        };
        flow.push_back({u, theta, speed, swirl, c});
    }

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            throatRadius = 0.85f;
            flare = 0.22f;
        }
        if (IsKeyPressed(KEY_LEFT_BRACKET)) throatRadius = std::max(0.45f, throatRadius - 0.05f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) throatRadius = std::min(1.8f, throatRadius + 0.05f);
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) flare = std::max(0.08f, flare - 0.02f);
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) flare = std::min(0.55f, flare + 0.02f);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        float dt = GetFrameTime();
        if (!paused) {
            for (FlowParticle& p : flow) {
                p.u += dt * (0.35f + p.speed);
                p.theta += dt * p.swirl;
                if (p.u > 4.8f) p.u = -4.8f;
            }
        }

        BeginDrawing();
        ClearBackground(Color{5, 8, 18, 255});

        BeginMode3D(camera);


        DrawWormholeSurface(throatRadius, flare);

        DrawSphere({0.0f, 0.0f, -4.8f}, RadiusProfile(-4.8f, throatRadius, flare), Color{80, 120, 190, 35});
        DrawSphere({0.0f, 0.0f, 4.8f}, RadiusProfile(4.8f, throatRadius, flare), Color{80, 120, 190, 35});

        for (const FlowParticle& p : flow) {
            Vector3 pos = WormholePoint(p.u, p.theta, throatRadius, flare);
            DrawSphere(pos, 0.03f, p.color);
        }

        EndMode3D();

        DrawText("Wormhole Tunnel (Morris-Thorne Style Visual)", 20, 18, 29, Color{232, 238, 248, 255});
        DrawText("Hold left mouse: orbit | wheel: zoom | [ ] throat | +/- flare | P pause | R reset", 20, 54, 19, Color{164, 183, 210, 255});
        std::string hud = Hud(throatRadius, flare, static_cast<int>(flow.size()), paused);
        DrawText(hud.c_str(), 20, 82, 21, Color{126, 224, 255, 255});
        DrawFPS(20, 114);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
