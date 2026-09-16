#include "raylib.h"
#include "../common/studio.h"
#include "../common/physics_models.h"
#include <deque>
#include "raymath.h"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kScreenWidth = 1280;
constexpr int kScreenHeight = 820;

constexpr float kMass = 1.0f;
constexpr float kSpringK = 4.0f;


constexpr float kAnchorX = -2.4f;
constexpr float kBaseMassX = -0.2f;

struct State {
    float x;
    float v;
};

float Accel(float x) {
    return -(kSpringK / kMass) * x;
}

State Rk4Step(const State& s, float dt) {
    const float k1x = s.v;
    const float k1v = Accel(s.x);

    const float k2x = s.v + 0.5f * dt * k1v;
    const float k2v = Accel(s.x + 0.5f * dt * k1x);

    const float k3x = s.v + 0.5f * dt * k2v;
    const float k3v = Accel(s.x + 0.5f * dt * k2x);

    const float k4x = s.v + dt * k3v;
    const float k4v = Accel(s.x + dt * k3x);

    State out{};
    out.x = s.x + (dt / 6.0f) * (k1x + 2.0f * k2x + 2.0f * k3x + k4x);
    out.v = s.v + (dt / 6.0f) * (k1v + 2.0f * k2v + 2.0f * k3v + k4v);
    return out;
}

void DrawSpring(const Vector3& from, const Vector3& to, int coils, float radius, Color color) {
    Vector3 axis = Vector3Subtract(to, from);
    float len = Vector3Length(axis);
    if (len < 0.001f) {
        return;
    }

    Vector3 tangent = Vector3Scale(axis, 1.0f / len);
    Vector3 ref = (std::fabs(tangent.y) < 0.9f) ? Vector3{0.0f, 1.0f, 0.0f} : Vector3{1.0f, 0.0f, 0.0f};
    Vector3 u = Vector3Normalize(Vector3CrossProduct(tangent, ref));
    Vector3 v = Vector3Normalize(Vector3CrossProduct(tangent, u));

    const int segments = coils * 16;
    Vector3 prev = from;
    for (int i = 1; i <= segments; ++i) {
        const float a = static_cast<float>(i) / segments;
        const float phase = a * coils * 2.0f * PI;
        Vector3 p = Vector3Add(from, Vector3Scale(axis, a));
        Vector3 off = Vector3Add(Vector3Scale(u, std::cos(phase) * radius), Vector3Scale(v, std::sin(phase) * radius));
        p = Vector3Add(p, off);
        DrawLine3D(prev, p, color);
        prev = p;
    }
}

std::string FormatHud(const State& s, float t, bool paused, float speed) {
    const float energy = 0.5f * kMass * s.v * s.v + 0.5f * kSpringK * s.x * s.x;

    std::ostringstream os;
    os << std::fixed << std::setprecision(3);
    os << "t=" << t << "  x=" << s.x << "  v=" << s.v << "  E=" << energy;
    if (paused) {
        os << "  [PAUSED]";
    }
    os << "  speed=" << speed << "x";
    return os.str();
}

}  // namespace

int main() {
    if (std::getenv("COMPPHYSICS_SMOKE_FRAMES")) SetConfigFlags(FLAG_WINDOW_HIDDEN);
    InitWindow(kScreenWidth, kScreenHeight, "3D Simple Harmonic Oscillator - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {3.5f, 2.2f, 6.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    State s{1.0f, 0.0f};
    float t = 0.0f;
    bool paused = false;
    float speed = 1.0f;

    std::deque<float> history;
    physics::Clock clock;
    int samples=0;
    float yaw=1.05f,pitch=0.36f,distance=8;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) {
            paused = !paused;
        }
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) {
            speed = std::min(4.0f, speed + 0.25f);
        }
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) {
            speed = std::max(0.25f, speed - 0.25f);
        }
        if (IsKeyPressed(KEY_R)) {
            s = {1.0f, 0.0f};
            t = 0.0f;
            history.clear(); clock.reset(); samples=0;
        }

        studio::orbit(camera,yaw,pitch,distance);

        if (!paused) clock.advance(std::min(GetFrameTime(),0.025f)*speed,[&](double dt) {
            s=Rk4Step(s,float(dt)); t+=float(dt);
            if (++samples%4==0) {
                history.push_back(s.x); if (history.size()>900) history.pop_front();
            }
        });

        const float massX = kBaseMassX + s.x;
        const Vector3 anchor = {kAnchorX, 0.0f, 0.0f};
        const Vector3 massPos = {massX, 0.0f, 0.0f};

        BeginDrawing();
        ClearBackground(Color{7, 10, 18, 255});

        BeginMode3D(camera);

        DrawCube({kAnchorX - 0.2f, 0.0f, 0.0f}, 0.15f, 1.1f, 1.1f, Color{90, 120, 180, 255});
        DrawLine3D({kAnchorX - 0.2f, 0.0f, -0.55f}, {kAnchorX - 0.2f, 0.0f, 0.55f}, Color{200, 220, 255, 130});

        DrawSpring(anchor, massPos, 18, 0.12f, Color{170, 210, 255, 255});

        DrawCube(massPos, 0.45f, 0.45f, 0.45f, Color{255, 210, 120, 255});
        DrawCubeWires(massPos, 0.45f, 0.45f, 0.45f, Color{255, 240, 190, 255});

        DrawLine3D({-3.5f, 0.0f, 0.0f}, {3.5f, 0.0f, 0.0f}, Color{90, 130, 170, 120});

        EndMode3D();

        studio::title("3D Simple Harmonic Oscillator", studio::Style::Instrument);
        studio::help("P: pause  R: reset  +/-: speed  Mouse drag: orbit / wheel: zoom");

        const std::string hud = FormatHud(s, t, paused, speed);
        studio::readout(hud.c_str());

        studio::plot({26,float(GetScreenHeight()-278),420,210},"Displacement / last 15 s",history,-1.2f,1.2f,Color{135,214,230,255});
        studio::note("Fixed 240 Hz RK4 / mouse-driven camera / no automatic orbit");
        studio::fps();

        EndDrawing();
        if (studio::smokeFrame(__FILE__)) break;
    }

    studio::unload();
    CloseWindow();
    return 0;
}
