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

constexpr float kG = 9.81f;
constexpr float kL1 = 1.35f;
constexpr float kL2 = 1.25f;
constexpr float kM1 = 1.0f;
constexpr float kM2 = 1.0f;
constexpr int kTrailMax = 1800;

struct State {
    float t1;
    float w1;
    float t2;
    float w2;
};

struct Deriv {
    float dt1;
    float dw1;
    float dt2;
    float dw2;
};

Deriv Eval(const State& s) {
    const float d = s.t1 - s.t2;
    const float den = 2.0f * kM1 + kM2 - kM2 * std::cos(2.0f * s.t1 - 2.0f * s.t2);

    Deriv out{};
    out.dt1 = s.w1;
    out.dt2 = s.w2;

    out.dw1 = (
        -kG * (2.0f * kM1 + kM2) * std::sin(s.t1)
        -kM2 * kG * std::sin(s.t1 - 2.0f * s.t2)
        -2.0f * std::sin(d) * kM2 * (s.w2 * s.w2 * kL2 + s.w1 * s.w1 * kL1 * std::cos(d))
    ) / (kL1 * den);

    out.dw2 = (
        2.0f * std::sin(d) * (
            s.w1 * s.w1 * kL1 * (kM1 + kM2)
            +kG * (kM1 + kM2) * std::cos(s.t1)
            +s.w2 * s.w2 * kL2 * kM2 * std::cos(d)
        )
    ) / (kL2 * den);

    return out;
}

State StepRK4(const State& s, float dt) {
    const Deriv k1 = Eval(s);

    const State s2 = {
        s.t1 + 0.5f * dt * k1.dt1,
        s.w1 + 0.5f * dt * k1.dw1,
        s.t2 + 0.5f * dt * k1.dt2,
        s.w2 + 0.5f * dt * k1.dw2,
    };
    const Deriv k2 = Eval(s2);

    const State s3 = {
        s.t1 + 0.5f * dt * k2.dt1,
        s.w1 + 0.5f * dt * k2.dw1,
        s.t2 + 0.5f * dt * k2.dt2,
        s.w2 + 0.5f * dt * k2.dw2,
    };
    const Deriv k3 = Eval(s3);

    const State s4 = {
        s.t1 + dt * k3.dt1,
        s.w1 + dt * k3.dw1,
        s.t2 + dt * k3.dt2,
        s.w2 + dt * k3.dw2,
    };
    const Deriv k4 = Eval(s4);

    State out{};
    out.t1 = s.t1 + (dt / 6.0f) * (k1.dt1 + 2.0f * k2.dt1 + 2.0f * k3.dt1 + k4.dt1);
    out.w1 = s.w1 + (dt / 6.0f) * (k1.dw1 + 2.0f * k2.dw1 + 2.0f * k3.dw1 + k4.dw1);
    out.t2 = s.t2 + (dt / 6.0f) * (k1.dt2 + 2.0f * k2.dt2 + 2.0f * k3.dt2 + k4.dt2);
    out.w2 = s.w2 + (dt / 6.0f) * (k1.dw2 + 2.0f * k2.dw2 + 2.0f * k3.dw2 + k4.dw2);
    return out;
}

void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 delta = GetMouseDelta();
        *yaw -= delta.x * 0.0035f;
        *pitch += delta.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }

    *distance -= GetMouseWheelMove() * 0.6f;
    *distance = std::clamp(*distance, 2.5f, 20.0f);

    const float cp = std::cos(*pitch);
    const Vector3 offset = {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}

void Positions(const State& s, Vector3* p1, Vector3* p2) {
    const float x1 = kL1 * std::sin(s.t1);
    const float y1 = -kL1 * std::cos(s.t1);
    const float x2 = x1 + kL2 * std::sin(s.t2);
    const float y2 = y1 - kL2 * std::cos(s.t2);

    *p1 = {x1, y1, 0.0f};
    *p2 = {x2, y2, 0.0f};
}

void DrawTrail(const std::deque<Vector3>& trail) {
    if (trail.size() < 2) return;
    for (size_t i = 1; i < trail.size(); ++i) {
        float a = static_cast<float>(i) / static_cast<float>(trail.size());
        Color c = Color{120, 220, 255, static_cast<unsigned char>(20 + 200 * a)};
        DrawLine3D(trail[i - 1], trail[i], c);
    }
}

float TotalEnergy(const State& s) {
    const float y1 = -kL1 * std::cos(s.t1);
    const float y2 = y1 - kL2 * std::cos(s.t2);

    const float v1sq = kL1 * kL1 * s.w1 * s.w1;
    const float v2sq = v1sq + kL2 * kL2 * s.w2 * s.w2 + 2.0f * kL1 * kL2 * s.w1 * s.w2 * std::cos(s.t1 - s.t2);

    const float T = 0.5f * kM1 * v1sq + 0.5f * kM2 * v2sq;
    const float V = kM1 * kG * y1 + kM2 * kG * y2;
    return T + V;
}

std::string HudText(float t, float speed, const State& s, bool paused) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(3)
       << "t=" << t
       << "  speed=" << speed << "x"
       << "  E=" << TotalEnergy(s);
    if (paused) os << "  [PAUSED]";
    return os.str();
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Double Pendulum Chaos 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {5.0f, 2.5f, 6.5f};
    camera.target = {0.0f, -1.2f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    State s{2.0f, 0.0f, 1.65f, 0.0f};
    float simTime = 0.0f;
    float speed = 1.0f;
    bool paused = false;

    float camYaw = 0.92f;
    float camPitch = 0.30f;
    float camDistance = 8.8f;

    std::deque<Vector3> trail;
    Vector3 p1{};
    Vector3 p2{};
    Positions(s, &p1, &p2);
    trail.push_back(p2);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            s = {2.0f, 0.0f, 1.65f, 0.0f};
            simTime = 0.0f;
            trail.clear();
            Positions(s, &p1, &p2);
            trail.push_back(p2);
        }
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) speed = std::min(8.0f, speed + 0.25f);
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) speed = std::max(0.25f, speed - 0.25f);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        if (!paused) {
            float frameDt = GetFrameTime() * speed;
            int steps = std::max(1, static_cast<int>(std::ceil(frameDt / 0.004f)));
            float dt = frameDt / static_cast<float>(steps);
            for (int i = 0; i < steps; ++i) {
                s = StepRK4(s, dt);
                simTime += dt;
            }
        }

        Positions(s, &p1, &p2);
        trail.push_back(p2);
        if (static_cast<int>(trail.size()) > kTrailMax) trail.pop_front();

        BeginDrawing();
        ClearBackground(Color{6, 9, 16, 255});

        BeginMode3D(camera);

        const Vector3 pivot = {0.0f, 0.0f, 0.0f};
        DrawSphere(pivot, 0.06f, Color{230, 230, 240, 255});

        DrawTrail(trail);

        DrawLine3D(pivot, p1, Color{240, 200, 120, 255});
        DrawLine3D(p1, p2, Color{130, 205, 255, 255});

        DrawSphere(p1, 0.11f, Color{255, 190, 100, 255});
        DrawSphere(p2, 0.12f, Color{110, 220, 255, 255});

        EndMode3D();

        DrawText("Double Pendulum Chaos (3D)", 20, 18, 30, Color{232, 238, 248, 255});
        DrawText("Hold left mouse: orbit | wheel: zoom | P pause | +/- speed | R reset", 20, 56, 20, Color{164, 183, 210, 255});
        std::string hud = HudText(simTime, speed, s, paused);
        DrawText(hud.c_str(), 20, 86, 21, Color{126, 224, 255, 255});
        DrawFPS(20, 118);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
