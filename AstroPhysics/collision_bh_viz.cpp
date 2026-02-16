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

struct BurstParticle {
    Vector3 dir;
    float speed;
    float age;
    float life;
};

void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }

    *distance -= GetMouseWheelMove() * 0.6f;
    *distance = std::clamp(*distance, 4.0f, 32.0f);

    float cp = std::cos(*pitch);
    Vector3 offset = {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}

void ResetBurst(std::vector<BurstParticle>* burst) {
    burst->clear();
    burst->reserve(180);
    for (int i = 0; i < 180; ++i) {
        float a = 2.0f * PI * static_cast<float>(i) / 180.0f;
        float z = -0.25f + 0.5f * std::fmod(i * 0.61803f, 1.0f);
        float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
        Vector3 dir = {r * std::cos(a), z, r * std::sin(a)};
        float speed = 1.4f + 2.0f * std::fmod(i * 0.371f, 1.0f);
        float life = 1.2f + 1.8f * std::fmod(i * 0.529f, 1.0f);
        burst->push_back({dir, speed, 0.0f, life});
    }
}

std::string Hud(float t, float speed, bool paused, bool merged) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(2)
       << "t=" << t
       << "  speed=" << speed << "x"
       << "  phase=" << (merged ? "ringdown" : "inspiral");
    if (paused) os << "  [PAUSED]";
    return os.str();
}

void DrawWaveRing(float radius, float amp, Color c) {
    int segs = 128;
    for (int i = 0; i < segs; ++i) {
        float a0 = 2.0f * PI * static_cast<float>(i) / static_cast<float>(segs);
        float a1 = 2.0f * PI * static_cast<float>(i + 1) / static_cast<float>(segs);

        float wob0 = amp * std::sin(2.0f * a0);
        float wob1 = amp * std::sin(2.0f * a1);

        Vector3 p0 = {(radius + wob0) * std::cos(a0), 0.0f, (radius + wob0) * std::sin(a0)};
        Vector3 p1 = {(radius + wob1) * std::cos(a1), 0.0f, (radius + wob1) * std::sin(a1)};
        DrawLine3D(p0, p1, c);
    }
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Black Hole Collision 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {9.5f, 5.2f, 9.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.82f;
    float camPitch = 0.36f;
    float camDistance = 14.0f;

    float simT = 0.0f;
    float speed = 1.0f;
    bool paused = false;

    bool merged = false;
    float mergeTime = 0.0f;

    std::vector<BurstParticle> burst;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            simT = 0.0f;
            speed = 1.0f;
            paused = false;
            merged = false;
            mergeTime = 0.0f;
            burst.clear();
        }
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) speed = std::max(0.25f, speed - 0.25f);
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) speed = std::min(6.0f, speed + 0.25f);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        float dt = GetFrameTime() * speed;
        if (!paused) {
            simT += dt;

            if (!merged && simT > 9.0f) {
                merged = true;
                mergeTime = simT;
                ResetBurst(&burst);
            }

            if (merged) {
                for (BurstParticle& p : burst) {
                    p.age += dt;
                }
            }
        }

        float sep = 0.0f;
        float omega = 0.0f;
        float ang = 0.0f;

        Vector3 bh1{0.0f, 0.0f, 0.0f};
        Vector3 bh2{0.0f, 0.0f, 0.0f};

        if (!merged) {
            float tau = std::clamp(simT / 9.0f, 0.0f, 1.0f);
            sep = 3.2f * std::pow(1.0f - tau, 0.58f) + 0.45f;
            omega = 0.55f + 3.6f * tau * tau;
            ang = simT * omega;

            bh1 = {-0.5f * sep * std::cos(ang), 0.12f * std::sin(ang * 0.6f), -0.5f * sep * std::sin(ang)};
            bh2 = { 0.5f * sep * std::cos(ang), -0.12f * std::sin(ang * 0.6f),  0.5f * sep * std::sin(ang)};
        }

        BeginDrawing();
        ClearBackground(Color{4, 7, 16, 255});

        BeginMode3D(camera);

        DrawGrid(28, 0.5f);

        if (!merged) {
            DrawSphere(bh1, 0.34f, BLACK);
            DrawSphere(bh2, 0.32f, BLACK);

            DrawSphere(bh1, 0.48f, Color{130, 200, 255, 30});
            DrawSphere(bh2, 0.46f, Color{130, 200, 255, 30});

            float waveBase = 1.3f + simT * 1.25f;
            DrawWaveRing(waveBase, 0.08f + 0.05f * std::sin(simT * 3.0f), Color{120, 170, 255, 120});
            DrawWaveRing(waveBase + 1.6f, 0.06f, Color{120, 170, 255, 85});
            DrawWaveRing(waveBase + 3.2f, 0.04f, Color{120, 170, 255, 60});

            DrawLine3D(bh1, bh2, Color{90, 130, 200, 80});
        } else {
            float ringdownT = simT - mergeTime;
            float remnantR = 0.44f + 0.05f * std::exp(-1.8f * ringdownT) * std::sin(15.0f * ringdownT);
            DrawSphere({0.0f, 0.0f, 0.0f}, remnantR, BLACK);
            DrawSphere({0.0f, 0.0f, 0.0f}, remnantR + 0.17f, Color{255, 190, 110, 40});

            float w1 = 1.2f + ringdownT * 2.1f;
            float decay = std::exp(-0.85f * ringdownT);
            DrawWaveRing(w1, 0.14f * decay, Color{255, 210, 140, 170});
            DrawWaveRing(w1 + 2.0f, 0.09f * decay, Color{190, 220, 255, 120});
            DrawWaveRing(w1 + 4.0f, 0.06f * decay, Color{150, 200, 255, 80});

            for (const BurstParticle& p : burst) {
                if (p.age > p.life) continue;
                float a = 1.0f - p.age / p.life;
                Vector3 pos = Vector3Scale(p.dir, p.speed * p.age);
                Color c = Color{255, static_cast<unsigned char>(180 + 50 * a), 90, static_cast<unsigned char>(220 * a)};
                DrawSphere(pos, 0.03f + 0.04f * a, c);
            }
        }

        EndMode3D();

        DrawText("Black Hole Merger (Inspiral -> Ringdown)", 20, 18, 30, Color{232, 238, 248, 255});
        DrawText("Hold left mouse: orbit | wheel: zoom | +/- speed | P pause | R reset", 20, 56, 20, Color{164, 183, 210, 255});
        std::string hud = Hud(simT, speed, paused, merged);
        DrawText(hud.c_str(), 20, 86, 21, Color{126, 224, 255, 255});
        DrawFPS(20, 118);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
