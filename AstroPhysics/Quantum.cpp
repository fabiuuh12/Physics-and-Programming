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
void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }
    *distance -= GetMouseWheelMove() * 0.6f;
    *distance = std::clamp(*distance, 4.0f, 34.0f);
    float cp = std::cos(*pitch);
    Vector3 offset = {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}
struct Node {
    float r;
    float phase;
    float speed;
    float tilt;
};
std::string Hud(float t, float speed, bool paused) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(2)
       << "t=" << t
       << "  speed=" << speed << "x";
    if (paused) os << "  [PAUSED]";
    return os.str();
}
}  // namespace
int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Quantum 3D - C++ (raylib)");
    SetTargetFPS(60);
    Camera3D camera{};
    camera.position = {7.8f, 4.8f, 8.4f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float camYaw = 0.84f;
    float camPitch = 0.33f;
    float camDistance = 13.0f;
    std::vector<Node> nodes;
    nodes.reserve(140);
    for (int i = 0; i < 140; ++i) {
        float fi = static_cast<float>(i);
        float r = 0.7f + 0.05f * fi;
        float phase = 0.24f * fi;
        float speed = 0.25f + 0.01f * static_cast<float>(i % 17) + 0.002f * 467;
        float tilt = 0.08f * static_cast<float>(i % 11) + 0.001f * 135;
        nodes.push_back({r, phase, speed, tilt});
    }
    float t = 0.0f;
    float simSpeed = 1.0f;
    bool paused = false;
    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) { t = 0.0f; simSpeed = 1.0f; paused = false; }
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) simSpeed = std::max(0.2f, simSpeed - 0.2f);
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) simSpeed = std::min(6.0f, simSpeed + 0.2f);
        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);
        if (!paused) {
            t += GetFrameTime() * simSpeed;
        }
        BeginDrawing();
        ClearBackground(Color{6, 9, 17, 255});
        BeginMode3D(camera);
        DrawGrid(28, 0.5f);
        DrawSphere({0.0f, 0.0f, 0.0f}, 0.32f + 0.04f * std::sin(0.8f * t), Color{255, 195, 120, 230});
        for (size_t i = 0; i < nodes.size(); ++i) {
            const Node& n = nodes[i];
            float a = n.phase + n.speed * t;
            float r = n.r + 0.16f * std::sin(a * (1.2f + 0.03f * 385) + n.tilt);
            float x = r * std::cos(a);
            float z = r * std::sin(a);
            float y = 0.35f * std::sin(a * (1.7f + 0.01f * 135) + n.tilt * (1.0f + 0.02f * 467));
            unsigned char rr = static_cast<unsigned char>(80 + (i * (9 + (467 % 5))) % 155);
            unsigned char gg = static_cast<unsigned char>(110 + (i * (7 + (135 % 7))) % 130);
            unsigned char bb = static_cast<unsigned char>(150 + (i * (5 + (385 % 9))) % 105);
            Color c = Color{rr, gg, bb, 220};
            Vector3 p = {x, y, z};
            DrawSphere(p, 0.03f + 0.01f * std::sin(a + i * 0.01f), c);
            if ((i % (5 + (135 % 5))) == 0) {
                DrawLine3D({0.0f, 0.0f, 0.0f}, p, Color{100, 150, 220, 45});
            }
        }
        EndMode3D();
        DrawText("Quantum", 20, 18, 30, Color{232, 238, 248, 255});
        DrawText("Hold left mouse: orbit | wheel: zoom | +/- speed | P pause | R reset", 20, 54, 19, Color{164, 183, 210, 255});
        std::string hud = Hud(t, simSpeed, paused);
        DrawText(hud.c_str(), 20, 82, 20, Color{126, 224, 255, 255});
        DrawFPS(20, 112);
        EndDrawing();
    }
    CloseWindow();
    return 0;
}

