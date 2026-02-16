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
    *distance = std::clamp(*distance, 4.0f, 35.0f);
    float cp = std::cos(*pitch);
    c->position = Vector3Add(c->target, {*distance * cp * std::cos(*yaw), *distance * std::sin(*pitch), *distance * cp * std::sin(*yaw)});
}

void DrawTrail(const std::deque<Vector3>& tr, Color c) {
    for (size_t i = 1; i < tr.size(); ++i) DrawLine3D(tr[i - 1], tr[i], c);
}

} // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Projectile Motion with Drag 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {10.0f, 6.0f, 8.0f};
    camera.target = {2.0f, 1.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.8f, camPitch = 0.33f, camDistance = 15.0f;

    auto resetState = [&](Vector3* pNo, Vector3* vNo, Vector3* pDr, Vector3* vDr, std::deque<Vector3>* trNo, std::deque<Vector3>* trDr, float speed, float angleDeg) {
        float ang = angleDeg * PI / 180.0f;
        Vector3 v0 = {speed * std::cos(ang), speed * std::sin(ang), 0.0f};
        *pNo = {0.0f, 0.15f, -0.3f}; *vNo = v0;
        *pDr = {0.0f, 0.15f, 0.3f};  *vDr = v0;
        trNo->clear(); trDr->clear();
        trNo->push_back(*pNo); trDr->push_back(*pDr);
    };

    float speed = 14.0f;
    float angleDeg = 44.0f;
    float dragK = 0.08f;
    bool paused = false;

    Vector3 pNo{}, vNo{}, pDr{}, vDr{};
    std::deque<Vector3> trNo, trDr;
    resetState(&pNo, &vNo, &pDr, &vDr, &trNo, &trDr, speed, angleDeg);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) { paused = false; resetState(&pNo, &vNo, &pDr, &vDr, &trNo, &trDr, speed, angleDeg); }
        if (IsKeyPressed(KEY_LEFT_BRACKET)) dragK = std::max(0.0f, dragK - 0.01f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) dragK = std::min(0.3f, dragK + 0.01f);
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) angleDeg = std::max(10.0f, angleDeg - 1.0f);
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) angleDeg = std::min(80.0f, angleDeg + 1.0f);
        if (IsKeyPressed(KEY_COMMA)) speed = std::max(4.0f, speed - 0.5f);
        if (IsKeyPressed(KEY_PERIOD)) speed = std::min(30.0f, speed + 0.5f);

        if (IsKeyPressed(KEY_SPACE)) resetState(&pNo, &vNo, &pDr, &vDr, &trNo, &trDr, speed, angleDeg);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        if (!paused) {
            float dt = GetFrameTime();
            Vector3 g = {0.0f, -9.81f, 0.0f};

            if (pNo.y > 0.0f) {
                vNo = Vector3Add(vNo, Vector3Scale(g, dt));
                pNo = Vector3Add(pNo, Vector3Scale(vNo, dt));
                trNo.push_back(pNo);
            }

            if (pDr.y > 0.0f) {
                Vector3 drag = Vector3Scale(vDr, -dragK * Vector3Length(vDr));
                vDr = Vector3Add(vDr, Vector3Scale(Vector3Add(g, drag), dt));
                pDr = Vector3Add(pDr, Vector3Scale(vDr, dt));
                trDr.push_back(pDr);
            }

            if (trNo.size() > 1500) trNo.pop_front();
            if (trDr.size() > 1500) trDr.pop_front();
        }

        BeginDrawing();
        ClearBackground(Color{6, 9, 16, 255});
        BeginMode3D(camera);

        DrawGrid(32, 0.5f);
        DrawCube({8.0f, -0.02f, 0.0f}, 20.0f, 0.02f, 5.0f, Color{50, 70, 95, 255});

        DrawTrail(trNo, Color{130, 220, 255, 255});
        DrawTrail(trDr, Color{255, 170, 110, 255});

        DrawSphere(pNo, 0.11f, Color{130, 220, 255, 255});
        DrawSphere(pDr, 0.11f, Color{255, 170, 110, 255});

        EndMode3D();

        DrawText("Projectile Motion: Vacuum vs Air Drag", 20, 18, 29, Color{232, 238, 248, 255});
        DrawText("Hold left mouse: orbit | wheel: zoom | [ ] drag | +/- angle | , . speed | SPACE relaunch | P pause | R reset", 20, 54, 18, Color{164, 183, 210, 255});

        std::ostringstream os;
        os << std::fixed << std::setprecision(2)
           << "drag=" << dragK << "  angle=" << angleDeg << "deg  speed=" << speed
           << "  range(no drag)=" << std::max(0.0f, pNo.x)
           << "  range(drag)=" << std::max(0.0f, pDr.x);
        if (paused) os << "  [PAUSED]";
        DrawText(os.str().c_str(), 20, 82, 20, Color{126, 224, 255, 255});
        DrawFPS(20, 110);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
