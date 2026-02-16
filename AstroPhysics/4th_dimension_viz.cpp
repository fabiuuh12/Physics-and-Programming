#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kScreenWidth = 1280;
constexpr int kScreenHeight = 820;

struct V4 {
    float x, y, z, w;
};

void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }
    *distance -= GetMouseWheelMove() * 0.6f;
    *distance = std::clamp(*distance, 4.0f, 36.0f);

    float cp = std::cos(*pitch);
    Vector3 offset = {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}

V4 RotateZW(const V4& p, float a) {
    return {p.x, p.y, p.z * std::cos(a) - p.w * std::sin(a), p.z * std::sin(a) + p.w * std::cos(a)};
}

V4 RotateXW(const V4& p, float a) {
    return {p.x * std::cos(a) - p.w * std::sin(a), p.y, p.z, p.x * std::sin(a) + p.w * std::cos(a)};
}

V4 RotateYZ(const V4& p, float a) {
    return {p.x, p.y * std::cos(a) - p.z * std::sin(a), p.y * std::sin(a) + p.z * std::cos(a), p.w};
}

Vector3 Project4Dto3D(const V4& p, float wCamera, float scale) {
    float f = scale / (wCamera - p.w);
    return {p.x * f, p.y * f, p.z * f};
}

std::string Hud(float speed4D, bool paused) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(2) << "4D rotation speed=" << speed4D;
    if (paused) os << "  [PAUSED]";
    return os.str();
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "4D Hypercube Projection 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.0f, 5.2f, 8.8f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.83f;
    float camPitch = 0.32f;
    float camDistance = 13.8f;

    std::vector<V4> vertices;
    vertices.reserve(16);
    for (int i = 0; i < 16; ++i) {
        vertices.push_back({
            (i & 1) ? 1.0f : -1.0f,
            (i & 2) ? 1.0f : -1.0f,
            (i & 4) ? 1.0f : -1.0f,
            (i & 8) ? 1.0f : -1.0f,
        });
    }

    std::vector<std::pair<int, int>> edges;
    for (int i = 0; i < 16; ++i) {
        for (int j = i + 1; j < 16; ++j) {
            int diff = i ^ j;
            if (diff && ((diff & (diff - 1)) == 0)) {
                edges.push_back({i, j});
            }
        }
    }

    float t = 0.0f;
    float speed4D = 1.0f;
    bool paused = false;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            t = 0.0f;
            speed4D = 1.0f;
            paused = false;
        }
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) speed4D = std::max(0.1f, speed4D - 0.1f);
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) speed4D = std::min(5.0f, speed4D + 0.1f);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        if (!paused) {
            t += GetFrameTime() * speed4D;
        }

        std::vector<Vector3> projected(16);
        for (int i = 0; i < 16; ++i) {
            V4 p = vertices[i];
            p = RotateZW(p, t * 0.9f);
            p = RotateXW(p, t * 0.7f);
            p = RotateYZ(p, t * 0.6f);
            projected[i] = Project4Dto3D(p, 3.4f, 2.4f);
        }

        BeginDrawing();
        ClearBackground(Color{6, 9, 17, 255});

        BeginMode3D(camera);

        DrawGrid(26, 0.5f);

        for (const auto& e : edges) {
            int a = e.first;
            int b = e.second;
            float wa = vertices[a].w;
            float wb = vertices[b].w;
            Color c = (wa * wb > 0) ? Color{120, 190, 255, 170} : Color{255, 170, 120, 165};
            DrawLine3D(projected[a], projected[b], c);
        }

        for (int i = 0; i < 16; ++i) {
            float w = vertices[i].w;
            Color c = (w > 0.0f) ? Color{120, 230, 255, 255} : Color{255, 170, 130, 255};
            DrawSphere(projected[i], 0.08f, c);
        }

        EndMode3D();

        DrawText("4D Tesseract Projection into 3D", 20, 18, 29, Color{232, 238, 248, 255});
        DrawText("Hold left mouse: orbit | wheel: zoom | +/- 4D speed | P pause | R reset", 20, 54, 18, Color{164, 183, 210, 255});
        std::string hud = Hud(speed4D, paused);
        DrawText(hud.c_str(), 20, 82, 20, Color{126, 224, 255, 255});
        DrawText("Blue/orange vertices indicate opposite w-hyperplanes", 20, 110, 18, Color{185, 198, 215, 255});
        DrawFPS(20, 138);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
