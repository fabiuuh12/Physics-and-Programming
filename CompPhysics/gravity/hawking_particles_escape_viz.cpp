#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {
constexpr int kScreenWidth = 1280;
constexpr int kScreenHeight = 820;
struct Particle { Vector3 p; Vector3 v; float life; };

void UpdateOrbitCameraDragOnly(Camera3D* c, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }
    *distance -= GetMouseWheelMove() * 0.6f;
    *distance = std::clamp(*distance, 4.0f, 36.0f);
    float cp = std::cos(*pitch);
    c->position = Vector3Add(c->target, {*distance * cp * std::cos(*yaw), *distance * std::sin(*pitch), *distance * cp * std::sin(*yaw)});
}
}

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Hawking Particles Escape 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.0f, 5.0f, 9.0f};
    camera.target = {0,0.5f,0};
    camera.up = {0,1,0};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float camYaw=0.84f, camPitch=0.34f, camDistance=13.5f;

    std::vector<Particle> pairs;
    float rate = 14.0f;
    bool paused=false;
    float timer=0.0f;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused=!paused;
        if (IsKeyPressed(KEY_R)) { pairs.clear(); rate=14.0f; paused=false; timer=0.0f; }
        if (IsKeyPressed(KEY_LEFT_BRACKET)) rate = std::max(2.0f, rate-1.0f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) rate = std::min(60.0f, rate+1.0f);

        UpdateOrbitCameraDragOnly(&camera,&camYaw,&camPitch,&camDistance);

        if (!paused) {
            float dt = GetFrameTime();
            timer += dt;
            float period = 1.0f / rate;
            while (timer > period) {
                timer -= period;
                float a = 2.0f*PI*std::fmod((float)pairs.size()*0.618f,1.0f);
                Vector3 n = {std::cos(a), 0.2f*std::sin(2*a), std::sin(a)};
                n = Vector3Normalize(n);
                pairs.push_back({Vector3Scale(n,1.05f), Vector3Scale(n,2.5f), 2.4f}); // escaping
                pairs.push_back({Vector3Scale(n,1.05f), Vector3Scale(n,-1.2f), 1.2f}); // infalling partner
            }

            for (auto& p : pairs) {
                p.p = Vector3Add(p.p, Vector3Scale(p.v, dt));
                p.life -= dt;
                float r = Vector3Length(p.p);
                if (Vector3DotProduct(p.p,p.v) < 0.0f) {
                    p.v = Vector3Scale(p.v, 1.0f + 0.7f*dt);
                } else {
                    p.v = Vector3Scale(p.v, 1.0f - 0.08f*dt);
                }
                if (r < 0.8f) p.life = 0.0f;
            }
            pairs.erase(std::remove_if(pairs.begin(), pairs.end(), [](const Particle& p){ return p.life <= 0.0f; }), pairs.end());
        }

        BeginDrawing();
        ClearBackground(Color{6,9,16,255});
        BeginMode3D(camera);
        DrawSphere({0,0.5f,0}, 0.8f, BLACK);
        DrawSphere({0,0.5f,0}, 1.02f, Color{120,170,230,25});

        for (size_t i=0;i<pairs.size();++i) {
            bool escape = (i % 2 == 0);
            Color c = escape ? Color{255,200,120,230} : Color{120,160,255,200};
            DrawSphere(Vector3Add(pairs[i].p,{0,0.5f,0}), 0.05f, c);
        }

        EndMode3D();

        DrawText("Hawking Pair Production: Escape vs Infall", 20, 18, 29, Color{232,238,248,255});
        DrawText("Hold left mouse: orbit | wheel: zoom | [ ] pair rate | P pause | R reset", 20, 54, 18, Color{164,183,210,255});

        char buf[160];
        snprintf(buf,sizeof(buf),"pair rate=%.1f  active particles=%zu%s", rate, pairs.size(), paused ? "  [PAUSED]" : "");
        DrawText(buf, 20, 82, 20, Color{126,224,255,255});
        DrawFPS(20,110);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
