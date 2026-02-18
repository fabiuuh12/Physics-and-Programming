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

void UpdateOrbitCameraDragOnly(Camera3D* c, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }
    *distance -= GetMouseWheelMove() * 0.6f;
    *distance = std::clamp(*distance, 4.0f, 34.0f);
    float cp = std::cos(*pitch);
    c->position = Vector3Add(c->target, {*distance * cp * std::cos(*yaw), *distance * std::sin(*pitch), *distance * cp * std::sin(*yaw)});
}

}

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Electric Charges Interaction 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {7.5f, 4.8f, 8.4f};
    camera.target = {0.0f, 0.6f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw=0.84f, camPitch=0.34f, camDistance=13.0f;

    float q1 = 1.0f, q2 = -1.0f;
    float m1 = 1.0f, m2 = 1.0f;
    Vector3 p1 = {-1.8f, 0.6f, 0.0f};
    Vector3 p2 = {1.8f, 0.6f, 0.0f};
    Vector3 v1 = {0.0f, 0.0f, 0.25f};
    Vector3 v2 = {0.0f, 0.0f,-0.25f};

    std::deque<Vector3> trail1, trail2;
    trail1.push_back(p1); trail2.push_back(p2);

    bool paused = false;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            q1 = 1.0f; q2 = -1.0f;
            p1 = {-1.8f,0.6f,0.0f}; p2 = {1.8f,0.6f,0.0f};
            v1 = {0,0,0.25f}; v2 = {0,0,-0.25f};
            trail1.clear(); trail2.clear(); trail1.push_back(p1); trail2.push_back(p2);
            paused = false;
        }
        if (IsKeyPressed(KEY_ONE)) q1 *= -1.0f;
        if (IsKeyPressed(KEY_TWO)) q2 *= -1.0f;

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        if (!paused) {
            float dt = GetFrameTime();
            Vector3 r12 = Vector3Subtract(p2, p1);
            float d2 = Vector3DotProduct(r12, r12) + 0.1f;
            float invR = 1.0f / std::sqrt(d2);
            float invR3 = invR * invR * invR;
            float k = 4.5f;

            Vector3 fOn1 = Vector3Scale(r12, k * q1 * q2 * invR3);
            Vector3 fOn2 = Vector3Negate(fOn1);

            v1 = Vector3Add(v1, Vector3Scale(fOn1, dt / m1));
            v2 = Vector3Add(v2, Vector3Scale(fOn2, dt / m2));

            p1 = Vector3Add(p1, Vector3Scale(v1, dt));
            p2 = Vector3Add(p2, Vector3Scale(v2, dt));

            p1.x = std::clamp(p1.x, -4.5f, 4.5f); p1.z = std::clamp(p1.z, -4.5f, 4.5f);
            p2.x = std::clamp(p2.x, -4.5f, 4.5f); p2.z = std::clamp(p2.z, -4.5f, 4.5f);

            trail1.push_back(p1); trail2.push_back(p2);
            if (trail1.size() > 900) trail1.pop_front();
            if (trail2.size() > 900) trail2.pop_front();
        }

        BeginDrawing();
        ClearBackground(Color{6,9,16,255});
        BeginMode3D(camera);

        for (size_t i=1; i<trail1.size(); ++i) DrawLine3D(trail1[i-1], trail1[i], Color{255,140,140,120});
        for (size_t i=1; i<trail2.size(); ++i) DrawLine3D(trail2[i-1], trail2[i], Color{140,180,255,120});

        DrawSphere(p1, 0.2f, (q1 > 0) ? Color{255,120,120,255} : Color{120,160,255,255});
        DrawSphere(p2, 0.2f, (q2 > 0) ? Color{255,120,120,255} : Color{120,160,255,255});

        DrawLine3D(p1, p2, Color{180,200,240,90});

        EndMode3D();

        DrawText("Two Electric Charges Interaction", 20, 18, 29, Color{232,238,248,255});
        DrawText("Hold left mouse: orbit | wheel: zoom | 1/2 flip sign | P pause | R reset", 20, 54, 18, Color{164,183,210,255});

        std::ostringstream os;
        os << std::fixed << std::setprecision(2)
           << "q1=" << q1 << "  q2=" << q2 << "  distance=" << Vector3Distance(p1,p2)
           << "  (same sign repel, opposite attract)";
        if (paused) os << "  [PAUSED]";
        DrawText(os.str().c_str(), 20, 82, 20, Color{126,224,255,255});
        DrawFPS(20, 110);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
