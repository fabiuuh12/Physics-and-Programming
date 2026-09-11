#include "raylib.h"
#include "../common/studio.h"
#include "../common/physics_models.h"
#include <random>
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
    *distance = std::clamp(*distance, 4.0f, 35.0f);
    float cp = std::cos(*pitch);
    c->position = Vector3Add(c->target, {*distance * cp * std::cos(*yaw), *distance * std::sin(*pitch), *distance * cp * std::sin(*yaw)});
}
} // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Particle Entanglement Correlation 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {7.8f, 4.9f, 8.8f};
    camera.target = {0,0.6f,0};
    camera.up = {0,1,0};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float camYaw=0.84f, camPitch=0.34f, camDistance=13.0f;

    float a = 0.0f;
    float b = PI/4.0f;
    bool paused = false;
    physics::Clock clock;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uniform(0,1);
    int trials=0,same=0,resultA=0,resultB=0,steps=0;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) { a=0.0f; b=PI/4.0f; paused=false; trials=same=steps=0; resultA=resultB=0; clock.reset(); rng.seed(42); }
        float previousA=a,previousB=b;
        if (IsKeyPressed(KEY_LEFT)) a -= 0.06f;
        if (IsKeyPressed(KEY_RIGHT)) a += 0.06f;
        if (IsKeyPressed(KEY_DOWN)) b -= 0.06f;
        if (IsKeyPressed(KEY_UP)) b += 0.06f;

        UpdateOrbitCameraDragOnly(&camera,&camYaw,&camPitch,&camDistance);
        if (a!=previousA || b!=previousB) { trials=same=0; resultA=resultB=0; }

        float corr = -std::cos(2.0f*(a-b));
        float pSame = 0.5f * (1.0f + corr);
        float pDiff = 1.0f - pSame;
        if (!paused) clock.advance(GetFrameTime(),[&](double) {
            if (++steps%24!=0) return;
            resultA=uniform(rng)<0.5f ? -1 : 1;
            bool match=uniform(rng)<pSame;
            resultB=match ? resultA : -resultA;
            ++trials; if (match) ++same;
        });

        BeginDrawing();
        ClearBackground(Color{6,9,16,255});
        BeginMode3D(camera);
        Vector3 p1 = {-1.8f, 0.6f, 0.0f};
        Vector3 p2 = { 1.8f, 0.6f, 0.0f};
        DrawSphere(p1, 0.22f, Color{120,220,255,255});
        DrawSphere(p2, 0.22f, Color{255,170,120,255});
        DrawLine3D(p1, p2, Color{170,200,255,120});

        DrawLine3D(p1, Vector3Add(p1, {std::cos(a), 0.0f, std::sin(a)}), Color{130,220,255,255});
        DrawLine3D(p2, Vector3Add(p2, {std::cos(b), 0.0f, std::sin(b)}), Color{255,180,120,255});

        DrawCube({0.0f, 0.25f + pSame, -1.5f}, 0.5f, 2.0f*pSame, 0.4f, Color{120,220,255,220});
        DrawCube({0.8f, 0.25f + pDiff, -1.5f}, 0.5f, 2.0f*pDiff, 0.4f, Color{255,170,120,220});

        EndMode3D();

        studio::title("Polarization singlet / paired measurements", studio::Style::Quantum);
        studio::help("Hold left mouse: orbit | wheel: zoom | LEFT/RIGHT set analyzer A | UP/DOWN set analyzer B | P pause | R reset");

        std::ostringstream os;
        os << std::fixed << std::setprecision(3) << "A=" << a << " rad  B=" << b << " rad  corr~" << corr << "  P(same)~" << pSame;
        if (paused) os << "  [PAUSED]";
        studio::readout(os.str().c_str());
        studio::note("Photon polarization: E(A,B) = -cos(2(A-B)) / each local result is unbiased");
        studio::panel({28,180,338,120},Color{29,29,46,245});
        studio::text(TextFormat("A  %+d       B  %+d",resultA,resultB),46,196,26,Color{226,214,248,255});
        studio::text(TextFormat("%d pairs / same %d",trials,same),46,234,18,Color{179,191,210,255});
        studio::text(TextFormat("Measured correlation %.3f",trials ? 2.0*same/trials-1 : 0),46,263,16,Color{179,191,210,255});
        studio::fps();

        EndDrawing();
        if (studio::smokeFrame(__FILE__)) break;
    }

    studio::unload();
    CloseWindow();
    return 0;
}
