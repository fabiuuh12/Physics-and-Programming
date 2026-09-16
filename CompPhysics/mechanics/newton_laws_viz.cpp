#include "raylib.h"
#include "../common/studio.h"
#include "../common/physics_models.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

namespace {

constexpr int kScreenWidth = 1280;
constexpr int kScreenHeight = 820;

void UpdateOrbitCameraDragOnly(Camera3D* c, float* yaw, float* pitch, float* distance) {
    studio::pan(c, *yaw, *pitch, *distance);
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON) && !studio::panGesture()) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }
    *distance -= GetMouseWheelMove() * 0.6f;
    *distance = std::clamp(*distance, 4.0f, 30.0f);
    float cp = std::cos(*pitch);
    c->position = Vector3Add(c->target, {*distance * cp * std::cos(*yaw), *distance * std::sin(*pitch), *distance * cp * std::sin(*yaw)});
}

void DrawArrow(Vector3 a, Vector3 b, Color c) {
    DrawLine3D(a, b, c);
    Vector3 dir = Vector3Normalize(Vector3Subtract(b, a));
    Vector3 side = Vector3Normalize(Vector3CrossProduct(dir, {0.0f, 1.0f, 0.0f}));
    if (Vector3Length(side) < 1e-5f) side = {1.0f, 0.0f, 0.0f};
    DrawLine3D(b, Vector3Add(b, Vector3Add(Vector3Scale(dir, -0.25f), Vector3Scale(side, 0.12f))), c);
    DrawLine3D(b, Vector3Add(b, Vector3Add(Vector3Scale(dir, -0.25f), Vector3Scale(side, -0.12f))), c);
}

} // namespace

int main() {
    if (std::getenv("COMPPHYSICS_SMOKE_FRAMES")) SetConfigFlags(FLAG_WINDOW_HIDDEN);
    InitWindow(kScreenWidth, kScreenHeight, "Newton's Laws 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.2f, 4.8f, 8.5f};
    camera.target = {0.0f, 0.6f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.84f, camPitch = 0.34f, camDistance = 13.0f;

    float mass = 2.0f;
    float forceMag = 4.0f;
    bool paused = false;
    physics::Clock clock;

    Vector3 pos = {-3.5f, 0.35f, 0.0f};
    Vector3 vel = {0.0f, 0.0f, 0.0f};

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) { clock.reset(); pos = {-3.5f, 0.35f, 0.0f}; vel = {0,0,0}; paused = false; mass = 2.0f; forceMag = 4.0f; }
        if (IsKeyPressed(KEY_LEFT_BRACKET)) forceMag = std::max(0.0f, forceMag - 0.5f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) forceMag = std::min(12.0f, forceMag + 0.5f);
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) mass = std::max(0.5f, mass - 0.1f);
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) mass = std::min(6.0f, mass + 0.1f);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        Vector3 force = {forceMag, 0.0f, 0.0f};
        Vector3 accel = Vector3Scale(force, 1.0f / mass);

        if (!paused) clock.advance(GetFrameTime(),[&](double step) {
            float dt=float(step);
            pos=Vector3Add(pos,Vector3Add(Vector3Scale(vel,dt),Vector3Scale(accel,0.5f*dt*dt)));
            vel=Vector3Add(vel,Vector3Scale(accel,dt));
            if (pos.x > 4.2f) {
                pos.x = 4.2f;
                vel.x *= -0.8f;
            }
            if (pos.x < -4.2f) {
                pos.x = -4.2f;
                vel.x *= -0.8f;
            }
        });

        BeginDrawing();
        ClearBackground(Color{6, 9, 16, 255});
        BeginMode3D(camera);


        DrawCube({0.0f, -0.02f, 0.0f}, 10.0f, 0.02f, 3.0f, Color{50, 65, 90, 255});
        DrawCube(pos, 0.6f, 0.6f, 0.6f, Color{120, 210, 255, 255});

        DrawArrow(pos, Vector3Add(pos, Vector3Scale(force, 0.12f)), Color{255, 180, 120, 255});
        DrawArrow(pos, Vector3Add(pos, Vector3Scale(Vector3Normalize(vel), std::min(1.4f, 0.18f*Vector3Length(vel)))), Color{130, 220, 255, 255});
        DrawArrow(pos, Vector3Add(pos, Vector3Scale(accel, 0.12f)), Color{220, 255, 140, 255});

        EndMode3D();

        studio::title("Newton's Laws: F = m a (and inertia / reaction)", studio::Style::Instrument);
        studio::help("Hold left mouse: orbit | wheel: zoom | [ ] force | +/- mass | P pause | R reset");

        std::ostringstream os;
        os << std::fixed << std::setprecision(3)
           << "mass=" << mass << "  force=" << forceMag << "  accel=" << accel.x << "  vx=" << vel.x;
        if (paused) os << "  [PAUSED]";
        studio::readout(os.str().c_str());
        studio::note("Orange: force / blue: velocity / green: acceleration / walls: restitution 0.8");
        studio::fps();

        EndDrawing();
        if (studio::smokeFrame(__FILE__)) break;
    }

    studio::unload();
    CloseWindow();
    return 0;
}
