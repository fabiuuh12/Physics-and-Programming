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

constexpr float kG = 1.0f;
constexpr float kSoftening = 0.02f;
constexpr int kTrailMax = 1200;

struct Body {
    float mass;
    float radius;
    Vector3 pos;
    Vector3 vel;
    Color color;
};

Vector3 AccelFrom(const Body& self, const Body& other) {
    Vector3 r = Vector3Subtract(other.pos, self.pos);
    float d2 = Vector3DotProduct(r, r) + kSoftening;
    float invD = 1.0f / std::sqrt(d2);
    float invD3 = invD * invD * invD;
    return Vector3Scale(r, kG * other.mass * invD3);
}

void StepRK4(Body* a, Body* b, float dt) {
    struct Deriv { Vector3 dp; Vector3 dv; };

    auto eval = [](const Body& aa, const Body& bb) {
        Deriv da{};
        Deriv db{};
        da.dp = aa.vel;
        db.dp = bb.vel;
        da.dv = AccelFrom(aa, bb);
        db.dv = AccelFrom(bb, aa);
        return std::pair<Deriv, Deriv>{da, db};
    };

    Body a0 = *a;
    Body b0 = *b;

    auto [k1a, k1b] = eval(a0, b0);

    Body a1 = a0;
    Body b1 = b0;
    a1.pos = Vector3Add(a0.pos, Vector3Scale(k1a.dp, dt * 0.5f));
    b1.pos = Vector3Add(b0.pos, Vector3Scale(k1b.dp, dt * 0.5f));
    a1.vel = Vector3Add(a0.vel, Vector3Scale(k1a.dv, dt * 0.5f));
    b1.vel = Vector3Add(b0.vel, Vector3Scale(k1b.dv, dt * 0.5f));
    auto [k2a, k2b] = eval(a1, b1);

    Body a2 = a0;
    Body b2 = b0;
    a2.pos = Vector3Add(a0.pos, Vector3Scale(k2a.dp, dt * 0.5f));
    b2.pos = Vector3Add(b0.pos, Vector3Scale(k2b.dp, dt * 0.5f));
    a2.vel = Vector3Add(a0.vel, Vector3Scale(k2a.dv, dt * 0.5f));
    b2.vel = Vector3Add(b0.vel, Vector3Scale(k2b.dv, dt * 0.5f));
    auto [k3a, k3b] = eval(a2, b2);

    Body a3 = a0;
    Body b3 = b0;
    a3.pos = Vector3Add(a0.pos, Vector3Scale(k3a.dp, dt));
    b3.pos = Vector3Add(b0.pos, Vector3Scale(k3b.dp, dt));
    a3.vel = Vector3Add(a0.vel, Vector3Scale(k3a.dv, dt));
    b3.vel = Vector3Add(b0.vel, Vector3Scale(k3b.dv, dt));
    auto [k4a, k4b] = eval(a3, b3);

    auto weighted = [dt](Vector3 k1, Vector3 k2, Vector3 k3, Vector3 k4) {
        Vector3 s = Vector3Add(k1, Vector3Scale(k2, 2.0f));
        s = Vector3Add(s, Vector3Scale(k3, 2.0f));
        s = Vector3Add(s, k4);
        return Vector3Scale(s, dt / 6.0f);
    };

    a->pos = Vector3Add(a0.pos, weighted(k1a.dp, k2a.dp, k3a.dp, k4a.dp));
    b->pos = Vector3Add(b0.pos, weighted(k1b.dp, k2b.dp, k3b.dp, k4b.dp));
    a->vel = Vector3Add(a0.vel, weighted(k1a.dv, k2a.dv, k3a.dv, k4a.dv));
    b->vel = Vector3Add(b0.vel, weighted(k1b.dv, k2b.dv, k3b.dv, k4b.dv));
}

void DrawTrail(const std::deque<Vector3>& trail, Color c) {
    if (trail.size() < 2) return;
    for (size_t i = 1; i < trail.size(); ++i) {
        float fade = static_cast<float>(i) / static_cast<float>(trail.size());
        Color cc = c;
        cc.a = static_cast<unsigned char>(25 + 170 * fade);
        DrawLine3D(trail[i - 1], trail[i], cc);
    }
}

float Kinetic(const Body& b) {
    return 0.5f * b.mass * Vector3DotProduct(b.vel, b.vel);
}

float Potential(const Body& a, const Body& b) {
    float r = std::sqrt(Vector3DotProduct(Vector3Subtract(a.pos, b.pos), Vector3Subtract(a.pos, b.pos)) + kSoftening);
    return -kG * a.mass * b.mass / r;
}

std::string HudText(float t, float speed, const Body& a, const Body& b, bool paused) {
    float e = Kinetic(a) + Kinetic(b) + Potential(a, b);
    std::ostringstream os;
    os << std::fixed << std::setprecision(3)
       << "t=" << t << "  speed=" << speed << "x"
       << "  E=" << e;
    if (paused) os << "  [PAUSED]";
    return os.str();
}

void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 delta = GetMouseDelta();
        *yaw -= delta.x * 0.0035f;
        *pitch += delta.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }

    *distance -= GetMouseWheelMove() * 0.6f;
    *distance = std::clamp(*distance, 3.0f, 28.0f);

    const float cp = std::cos(*pitch);
    const Vector3 offset = {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}

void ResetSystem(Body* a, Body* b, std::deque<Vector3>* trailA, std::deque<Vector3>* trailB) {
    a->mass = 6.0f;
    a->radius = 0.28f;
    a->pos = {-1.2f, 0.0f, 0.0f};
    a->color = Color{255, 196, 110, 255};

    b->mass = 1.8f;
    b->radius = 0.18f;
    b->pos = {2.0f, 0.0f, 0.0f};
    b->color = Color{115, 208, 255, 255};

    float r = Vector3Distance(a->pos, b->pos);
    float vrel = std::sqrt(kG * (a->mass + b->mass) / r) * 0.96f;
    Vector3 tangent = {0.0f, 0.0f, 1.0f};

    a->vel = Vector3Scale(tangent, -vrel * (b->mass / (a->mass + b->mass)));
    b->vel = Vector3Scale(tangent,  vrel * (a->mass / (a->mass + b->mass)));

    trailA->clear();
    trailB->clear();
    trailA->push_back(a->pos);
    trailB->push_back(b->pos);
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Two Body Orbit 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.0f, 4.6f, 8.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    Body a{};
    Body b{};
    std::deque<Vector3> trailA;
    std::deque<Vector3> trailB;

    ResetSystem(&a, &b, &trailA, &trailB);

    float simTime = 0.0f;
    float speed = 1.0f;
    bool paused = false;
    float camYaw = 0.78f;
    float camPitch = 0.42f;
    float camDistance = 12.0f;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            simTime = 0.0f;
            ResetSystem(&a, &b, &trailA, &trailB);
        }
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) speed = std::min(6.0f, speed + 0.25f);
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) speed = std::max(0.25f, speed - 0.25f);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        if (!paused) {
            float frameDt = GetFrameTime() * speed;
            int steps = std::max(1, static_cast<int>(std::ceil(frameDt / 0.006f)));
            float dt = frameDt / static_cast<float>(steps);
            for (int i = 0; i < steps; ++i) {
                StepRK4(&a, &b, dt);
                simTime += dt;
            }

            trailA.push_back(a.pos);
            trailB.push_back(b.pos);
            if (static_cast<int>(trailA.size()) > kTrailMax) trailA.pop_front();
            if (static_cast<int>(trailB.size()) > kTrailMax) trailB.pop_front();
        }

        BeginDrawing();
        ClearBackground(Color{5, 8, 16, 255});

        BeginMode3D(camera);

        DrawTrail(trailA, a.color);
        DrawTrail(trailB, b.color);

        DrawSphere(a.pos, a.radius, a.color);
        DrawSphere(b.pos, b.radius, b.color);

        Vector3 com = Vector3Scale(Vector3Add(Vector3Scale(a.pos, a.mass), Vector3Scale(b.pos, b.mass)), 1.0f / (a.mass + b.mass));
        DrawSphere(com, 0.05f, Color{220, 220, 220, 230});

        EndMode3D();

        DrawText("Two Body Orbit (Mutual Gravity)", 20, 18, 30, Color{232, 238, 248, 255});
        DrawText("Hold left mouse: orbit | wheel: zoom | P pause | +/- speed | R reset", 20, 56, 20, Color{164, 183, 210, 255});
        std::string hud = HudText(simTime, speed, a, b, paused);
        DrawText(hud.c_str(), 20, 86, 21, Color{126, 224, 255, 255});
        DrawFPS(20, 118);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
