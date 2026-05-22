#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

namespace {

constexpr int kScreenWidth = 1280;
constexpr int kScreenHeight = 820;
constexpr float kExtent = 4.8f;
constexpr int kLines = 7;
constexpr int kSegments = 42;

void UpdateOrbitCamera(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.25f, 1.25f);
    }

    *distance -= GetMouseWheelMove() * 0.75f;
    *distance = std::clamp(*distance, 8.0f, 26.0f);

    float cp = std::cos(*pitch);
    camera->position = Vector3Add(camera->target, {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    });
}

Color WithAlpha(Color c, unsigned char alpha) {
    c.a = alpha;
    return c;
}

float Coordinate(int i) {
    return -kExtent + 2.0f * kExtent * static_cast<float>(i) / static_cast<float>(kLines - 1);
}

Vector3 WarpedPoint(Vector3 p, Vector3 massPos, float mass, float core, float pulse) {
    Vector3 toMass = Vector3Subtract(massPos, p);
    float r = Vector3Length(toMass);
    if (r < 0.001f) return p;

    Vector3 dir = Vector3Scale(toMass, 1.0f / r);
    float pull = mass / (r * r + core * core);
    pull = std::min(pull, r * 0.82f);

    float wave = 0.10f * std::sin(4.8f * r - pulse) * std::exp(-0.18f * r);
    return Vector3Add(p, Vector3Scale(dir, pull + wave));
}

Color WarpColor(Vector3 p, Vector3 massPos) {
    float r = Vector3Distance(p, massPos);
    float near = std::clamp(1.0f - r / 5.8f, 0.0f, 1.0f);
    return Color{
        static_cast<unsigned char>(0 + 20 * near),
        static_cast<unsigned char>(72 + 165 * near),
        static_cast<unsigned char>(170 + 60 * (1.0f - near)),
        static_cast<unsigned char>(70 + 125 * near),
    };
}

void DrawWarpedLine(Vector3 a, Vector3 b, Vector3 massPos, float mass, float core, float pulse) {
    Vector3 prev = WarpedPoint(a, massPos, mass, core, pulse);
    for (int i = 1; i <= kSegments; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(kSegments);
        Vector3 base = Vector3Lerp(a, b, t);
        Vector3 current = WarpedPoint(base, massPos, mass, core, pulse);
        DrawLine3D(prev, current, WarpColor(current, massPos));
        prev = current;
    }
}

void DrawWarpedCubeGrid(Vector3 massPos, float mass, float core, float pulse) {
    for (int iy = 0; iy < kLines; ++iy) {
        float y = Coordinate(iy);
        for (int iz = 0; iz < kLines; ++iz) {
            float z = Coordinate(iz);
            DrawWarpedLine({-kExtent, y, z}, {kExtent, y, z}, massPos, mass, core, pulse);
        }
    }

    for (int ix = 0; ix < kLines; ++ix) {
        float x = Coordinate(ix);
        for (int iz = 0; iz < kLines; ++iz) {
            float z = Coordinate(iz);
            DrawWarpedLine({x, -kExtent, z}, {x, kExtent, z}, massPos, mass, core, pulse);
        }
    }

    for (int ix = 0; ix < kLines; ++ix) {
        float x = Coordinate(ix);
        for (int iy = 0; iy < kLines; ++iy) {
            float y = Coordinate(iy);
            DrawWarpedLine({x, y, -kExtent}, {x, y, kExtent}, massPos, mass, core, pulse);
        }
    }
}

std::string Hud(float mass, float core, bool animate, bool paused) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(2)
       << "mass=" << mass
       << "  core radius=" << core
       << "  warp: points displaced toward mass";
    if (animate) os << "  [flow pulse]";
    if (paused) os << "  [PAUSED]";
    return os.str();
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "3D Gravity Grid Warp - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float camYaw = 0.72f;
    float camPitch = 0.28f;
    float camDistance = 15.5f;

    float mass = 2.7f;
    float core = 1.05f;
    float time = 0.0f;
    bool paused = false;
    bool animate = true;
    Vector3 massPos = {2.45f, 0.0f, 0.45f};

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_SPACE)) animate = !animate;
        if (IsKeyPressed(KEY_R)) {
            mass = 2.7f;
            core = 1.05f;
            time = 0.0f;
            paused = false;
            animate = true;
            massPos = {2.45f, 0.0f, 0.45f};
        }

        float dt = GetFrameTime();
        if (IsKeyDown(KEY_UP)) mass = std::min(5.2f, mass + 1.4f * dt);
        if (IsKeyDown(KEY_DOWN)) mass = std::max(0.6f, mass - 1.4f * dt);
        if (IsKeyDown(KEY_RIGHT)) core = std::min(2.4f, core + 0.75f * dt);
        if (IsKeyDown(KEY_LEFT)) core = std::max(0.45f, core - 0.75f * dt);
        if (IsKeyDown(KEY_W)) massPos.y = std::min(3.6f, massPos.y + 1.8f * dt);
        if (IsKeyDown(KEY_S)) massPos.y = std::max(-3.6f, massPos.y - 1.8f * dt);
        if (IsKeyDown(KEY_A)) massPos.x = std::max(-3.6f, massPos.x - 1.8f * dt);
        if (IsKeyDown(KEY_D)) massPos.x = std::min(3.6f, massPos.x + 1.8f * dt);
        if (IsKeyDown(KEY_Q)) massPos.z = std::max(-3.6f, massPos.z - 1.8f * dt);
        if (IsKeyDown(KEY_E)) massPos.z = std::min(3.6f, massPos.z + 1.8f * dt);

        UpdateOrbitCamera(&camera, &camYaw, &camPitch, &camDistance);

        if (!paused && animate) {
            time += dt * 4.2f;
        }

        BeginDrawing();
        ClearBackground(BLACK);

        BeginMode3D(camera);
        DrawWarpedCubeGrid(massPos, mass, core, time);

        DrawSphere(massPos, 0.22f, WHITE);
        DrawSphereWires(massPos, 0.34f, 18, 12, Color{210, 245, 255, 140});
        DrawSphere(massPos, 0.55f, Color{120, 220, 255, 45});

        Vector3 axisTip = WarpedPoint({-3.4f, -3.4f, -3.4f}, massPos, mass, core, time);
        DrawLine3D(axisTip, massPos, Color{40, 220, 145, 115});
        EndMode3D();

        DrawText("3D Gravity Grid Warp", 20, 18, 30, Color{238, 242, 252, 255});
        DrawText("A mass pulls a full 3D coordinate lattice inward, like space curving around the object.", 20, 54, 18, Color{166, 184, 214, 255});
        DrawText("Mouse drag: orbit | wheel: zoom | UP/DOWN mass | LEFT/RIGHT core | WASD/QE move mass | SPACE pulse | P pause | R reset", 20, 80, 18, Color{166, 184, 214, 255});
        std::string hud = Hud(mass, core, animate, paused);
        DrawText(hud.c_str(), 20, 112, 20, Color{255, 220, 120, 255});
        DrawText("Visual analogy: nearby grid lines bend more strongly; green means stronger local distortion.", 20, 142, 20, Color{126, 224, 255, 255});
        DrawFPS(20, 174);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
