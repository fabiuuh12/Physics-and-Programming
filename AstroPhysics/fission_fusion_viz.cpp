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

struct Fragment {
    Vector3 pos;
    Vector3 vel;
    float life;
    Color color;
};

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

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Fission vs Fusion 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {7.2f, 4.6f, 8.0f};
    camera.target = {0.0f, 0.2f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.84f, camPitch = 0.33f, camDistance = 12.4f;

    bool fusionMode = false;
    bool paused = false;
    float t = 0.0f;
    std::vector<Fragment> frags;

    auto trigger = [&]() {
        frags.clear();
        if (!fusionMode) {
            Vector3 src = {0.0f, 0.0f, 0.0f};
            for (int i = 0; i < 16; ++i) {
                float a = 2.0f * PI * static_cast<float>(i) / 16.0f;
                Vector3 dir = {std::cos(a), 0.24f * std::sin(2.0f * a), std::sin(a)};
                frags.push_back({src, Vector3Scale(dir, 2.8f), 2.2f, Color{255, 170, 110, 255}});
            }
        } else {
            Vector3 s1 = {-1.1f, 0.0f, 0.0f};
            Vector3 s2 = {1.1f, 0.0f, 0.0f};
            for (int i = 0; i < 22; ++i) {
                float a = 2.0f * PI * static_cast<float>(i) / 22.0f;
                Vector3 dir = {std::cos(a), 0.2f * std::sin(3.0f * a), std::sin(a)};
                frags.push_back({{0.0f, 0.0f, 0.0f}, Vector3Scale(dir, 4.1f), 2.6f, Color{255, 220, 120, 255}});
            }
            frags.push_back({s1, {2.0f, 0.0f, 0.0f}, 0.6f, Color{130, 210, 255, 220}});
            frags.push_back({s2, {-2.0f, 0.0f, 0.0f}, 0.6f, Color{130, 210, 255, 220}});
        }
    };

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_M)) { fusionMode = !fusionMode; frags.clear(); t = 0.0f; }
        if (IsKeyPressed(KEY_SPACE)) trigger();
        if (IsKeyPressed(KEY_R)) { frags.clear(); t = 0.0f; paused = false; }

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        if (!paused) {
            float dt = GetFrameTime();
            t += dt;
            for (Fragment& f : frags) {
                f.pos = Vector3Add(f.pos, Vector3Scale(f.vel, dt));
                f.vel = Vector3Scale(f.vel, 0.985f);
                f.life -= dt;
            }
            frags.erase(std::remove_if(frags.begin(), frags.end(), [](const Fragment& f) { return f.life <= 0.0f; }), frags.end());
        }

        BeginDrawing();
        ClearBackground(Color{7, 10, 16, 255});

        BeginMode3D(camera);

        if (!fusionMode) {
            DrawSphere({0.0f, 0.0f, 0.0f}, 0.45f, Color{120, 210, 255, 230});
            DrawSphere({0.0f, 0.0f, 0.0f}, 0.62f, Color{120, 180, 255, 70});
        } else {
            DrawSphere({-1.1f, 0.0f, 0.0f}, 0.28f, Color{120, 210, 255, 230});
            DrawSphere({1.1f, 0.0f, 0.0f}, 0.28f, Color{120, 210, 255, 230});
            DrawLine3D({-1.1f, 0.0f, 0.0f}, {1.1f, 0.0f, 0.0f}, Color{255, 190, 120, 120});
        }

        for (const Fragment& f : frags) {
            Color c = f.color;
            c.a = static_cast<unsigned char>(std::clamp(120.0f * f.life, 20.0f, 255.0f));
            DrawSphere(f.pos, 0.05f, c);
        }

        EndMode3D();

        DrawText("Fission vs Fusion (Conceptual Energy Release)", 20, 18, 29, Color{235, 240, 250, 255});
        DrawText("Hold left mouse: orbit | wheel: zoom | M mode toggle | SPACE trigger | P pause | R reset", 20, 54, 18, Color{170, 184, 204, 255});

        std::ostringstream os;
        os << "mode=" << (fusionMode ? "fusion" : "fission") << "  fragments=" << frags.size();
        if (paused) os << "  [PAUSED]";
        DrawText(os.str().c_str(), 20, 82, 20, Color{255, 210, 150, 255});
        DrawFPS(20, 110);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
