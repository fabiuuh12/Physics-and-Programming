#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {
constexpr int kScreenWidth = 1280;
constexpr int kScreenHeight = 820;
constexpr float kG = 7.5f;
constexpr float kPi = 3.14159265358979323846f;

struct CoreBody {
    Vector3 pos;
    Vector3 vel;
    float mass;
};

struct StarParticle {
    Vector3 pos;
    Vector3 vel;
    int galaxyId;
};

void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0034f;
        *pitch += d.y * 0.0034f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }
    *distance -= GetMouseWheelMove() * 1.0f;
    *distance = std::clamp(*distance, 20.0f, 120.0f);
    float cp = std::cos(*pitch);
    camera->position = Vector3Add(camera->target, {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw)
    });
}

float RandRange(std::mt19937* rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    return dist(*rng);
}

void InitSystem(std::vector<StarParticle>* stars, CoreBody* c1, CoreBody* c2, float massRatio, float encounterSpeed, float diskScale) {
    stars->clear();
    stars->reserve(900);

    c1->mass = 22.0f;
    c2->mass = c1->mass * massRatio;
    c1->pos = {-12.0f, 0.0f, -2.5f};
    c2->pos = {12.0f, 0.0f, 2.5f};
    c1->vel = {0.7f * encounterSpeed, 0.0f, 0.0f};
    c2->vel = {-0.7f * encounterSpeed, 0.0f, 0.0f};

    std::mt19937 rng(42);
    const int nPerGalaxy = 420;
    const float soft = 0.6f;

    for (int g = 0; g < 2; ++g) {
        CoreBody core = (g == 0) ? *c1 : *c2;
        float diskTilt = (g == 0) ? 0.28f : -0.22f;
        float cTilt = std::cos(diskTilt);
        float sTilt = std::sin(diskTilt);
        for (int i = 0; i < nPerGalaxy; ++i) {
            float r = diskScale * std::sqrt(RandRange(&rng, 0.02f, 1.0f));
            float ang = RandRange(&rng, 0.0f, 2.0f * kPi);
            float y = RandRange(&rng, -0.22f, 0.22f);

            float lx = r * std::cos(ang);
            float lz = r * std::sin(ang);
            Vector3 local = {lx, y, lz};
            Vector3 tilted = {local.x, local.y * cTilt - local.z * sTilt, local.y * sTilt + local.z * cTilt};
            Vector3 pos = Vector3Add(core.pos, tilted);

            float vCirc = std::sqrt(kG * core.mass / std::max(0.8f, r + soft));
            Vector3 tangent = {-std::sin(ang), 0.0f, std::cos(ang)};
            Vector3 tangentTilted = {tangent.x, tangent.y * cTilt - tangent.z * sTilt, tangent.y * sTilt + tangent.z * cTilt};
            Vector3 vel = Vector3Add(core.vel, Vector3Scale(tangentTilted, vCirc));

            stars->push_back({pos, vel, g});
        }
    }
}
}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Galaxy Merger (Toy N-body) 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {40.0f, 30.0f, 40.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float camYaw = 0.82f;
    float camPitch = 0.42f;
    float camDistance = 65.0f;

    float massRatio = 1.0f;
    float encounterSpeed = 1.0f;
    float diskScale = 8.0f;
    float simSpeed = 1.0f;
    bool paused = false;

    CoreBody c1{};
    CoreBody c2{};
    std::vector<StarParticle> stars;
    InitSystem(&stars, &c1, &c2, massRatio, encounterSpeed, diskScale);

    while (!WindowShouldClose()) {
        bool needsReset = false;
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            massRatio = 1.0f;
            encounterSpeed = 1.0f;
            diskScale = 8.0f;
            simSpeed = 1.0f;
            paused = false;
            needsReset = true;
        }
        if (IsKeyDown(KEY_UP)) {
            massRatio = std::min(3.0f, massRatio + 0.8f * GetFrameTime());
            needsReset = true;
        }
        if (IsKeyDown(KEY_DOWN)) {
            massRatio = std::max(0.25f, massRatio - 0.8f * GetFrameTime());
            needsReset = true;
        }
        if (IsKeyDown(KEY_RIGHT)) {
            encounterSpeed = std::min(2.6f, encounterSpeed + 0.8f * GetFrameTime());
            needsReset = true;
        }
        if (IsKeyDown(KEY_LEFT)) {
            encounterSpeed = std::max(0.3f, encounterSpeed - 0.8f * GetFrameTime());
            needsReset = true;
        }
        if (IsKeyDown(KEY_RIGHT_BRACKET)) {
            diskScale = std::min(13.0f, diskScale + 2.5f * GetFrameTime());
            needsReset = true;
        }
        if (IsKeyDown(KEY_LEFT_BRACKET)) {
            diskScale = std::max(4.0f, diskScale - 2.5f * GetFrameTime());
            needsReset = true;
        }
        if (IsKeyDown(KEY_EQUAL)) simSpeed = std::min(5.0f, simSpeed + 1.2f * GetFrameTime());
        if (IsKeyDown(KEY_MINUS)) simSpeed = std::max(0.2f, simSpeed - 1.2f * GetFrameTime());

        if (needsReset) InitSystem(&stars, &c1, &c2, massRatio, encounterSpeed, diskScale);
        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        if (!paused) {
            float dt = GetFrameTime() * simSpeed;
            Vector3 d = Vector3Subtract(c2.pos, c1.pos);
            float r = std::max(1.4f, Vector3Length(d));
            Vector3 a1 = Vector3Scale(d, kG * c2.mass / (r * r * r));
            Vector3 a2 = Vector3Scale(d, -kG * c1.mass / (r * r * r));
            c1.vel = Vector3Add(c1.vel, Vector3Scale(a1, dt));
            c2.vel = Vector3Add(c2.vel, Vector3Scale(a2, dt));
            c1.pos = Vector3Add(c1.pos, Vector3Scale(c1.vel, dt));
            c2.pos = Vector3Add(c2.pos, Vector3Scale(c2.vel, dt));

            const float eps = 0.9f;
            for (StarParticle& s : stars) {
                Vector3 r1 = Vector3Subtract(c1.pos, s.pos);
                Vector3 r2 = Vector3Subtract(c2.pos, s.pos);
                float d1 = std::sqrt(Vector3LengthSqr(r1) + eps * eps);
                float d2 = std::sqrt(Vector3LengthSqr(r2) + eps * eps);
                Vector3 a = {0.0f, 0.0f, 0.0f};
                a = Vector3Add(a, Vector3Scale(r1, kG * c1.mass / (d1 * d1 * d1)));
                a = Vector3Add(a, Vector3Scale(r2, kG * c2.mass / (d2 * d2 * d2)));
                s.vel = Vector3Add(s.vel, Vector3Scale(a, dt));
                s.pos = Vector3Add(s.pos, Vector3Scale(s.vel, dt));
            }
        }

        BeginDrawing();
        ClearBackground(Color{5, 8, 14, 255});
        BeginMode3D(camera);
        DrawGrid(40, 2.0f);

        for (const StarParticle& s : stars) {
            Color c = (s.galaxyId == 0) ? Color{130, 205, 255, 210} : Color{255, 170, 130, 210};
            DrawPoint3D(s.pos, c);
        }
        DrawSphere(c1.pos, 0.65f, Color{125, 220, 255, 255});
        DrawSphere(c2.pos, 0.65f, Color{255, 180, 130, 255});
        DrawLine3D(c1.pos, c2.pos, Fade(Color{220, 220, 240, 255}, 0.35f));

        EndMode3D();

        DrawText("Galaxy Merger (Toy N-body)", 20, 18, 30, Color{232, 238, 248, 255});
        DrawText("Mouse orbit | wheel zoom | Up/Down mass ratio | Left/Right encounter speed | [ ] disk size | +/- sim speed | P pause | R reset",
                 20, 54, 18, Color{164, 183, 210, 255});

        char status[220];
        std::snprintf(status, sizeof(status),
                      "M2/M1=%.2f  v_enc=%.2f  disk=%.1f  stars=%zu%s",
                      massRatio, encounterSpeed, diskScale, stars.size(), paused ? " [PAUSED]" : "");
        DrawText(status, 20, 84, 20, Color{126, 224, 255, 255});
        DrawFPS(20, 112);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
