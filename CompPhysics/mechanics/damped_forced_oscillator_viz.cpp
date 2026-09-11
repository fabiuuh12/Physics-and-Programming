#include "raylib.h"
#include "../common/studio.h"
#include "../common/physics_models.h"
#include <array>
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
    studio::pan(c, *yaw, *pitch, *distance);
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON) && !studio::panGesture()) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }
    *distance -= GetMouseWheelMove() * 0.6f;
    *distance = std::clamp(*distance, 4.0f, 32.0f);
    float cp = std::cos(*pitch);
    c->position = Vector3Add(c->target, {*distance * cp * std::cos(*yaw), *distance * std::sin(*pitch), *distance * cp * std::sin(*yaw)});
}

}

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Damped Forced Oscillator 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.0f, 5.0f, 8.8f};
    camera.target = {0.0f, 0.5f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float camYaw=0.84f, camPitch=0.34f, camDistance=13.0f;

    float m=1.0f, k=10.0f, c=1.2f;
    float F0=4.0f, w=2.4f;
    double x=1.0, v=0.0;
    physics::Clock clock;
    int samples=0;
    double t=0.0;
    bool paused=false;
    std::deque<float> hist;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused=!paused;
        if (IsKeyPressed(KEY_R)) { x=1.0f; v=0.0f; t=0.0f; paused=false; F0=4.0f; w=2.4f; c=1.2f; hist.clear(); clock.reset(); samples=0; }
        if (IsKeyPressed(KEY_LEFT_BRACKET)) w = std::max(0.2f, w-0.1f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) w = std::min(8.0f, w+0.1f);
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) c = std::max(0.0f, c-0.1f);
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) c = std::min(6.0f, c+0.1f);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        if (!paused) clock.advance(GetFrameTime(),[&](double dt) {
            physics::oscillatorStep(x,v,t,dt,m,k,c,F0,w);
            t+=dt;
            if (++samples%4==0) {
                hist.push_back(float(x));
                if (hist.size()>480) hist.pop_front();
            }
        });
        std::array<float,161> response{};
        for (int i=0;i<161;++i) {
            double omega=i*0.05;
            response[i]=F0/std::max(1e-6,std::hypot(k-m*omega*omega,c*omega));
        }

        BeginDrawing();
        ClearBackground(Color{6,9,16,255});
        BeginMode3D(camera);

        Vector3 anchor = {-3.4f, 0.5f, 0.0f};
        Vector3 massPos = {-0.4f + float(x), 0.5f, 0.0f};

        DrawCube(anchor, 0.2f, 1.0f, 1.0f, Color{120,150,190,255});

        const int coils = 16;
        Vector3 prev = anchor;
        for (int i=1;i<=coils*8;++i) {
            float u = static_cast<float>(i)/(coils*8);
            float xx = anchor.x + (massPos.x - anchor.x)*u;
            float yy = anchor.y + 0.12f*std::sin(2.0f*PI*coils*u);
            Vector3 cur = {xx,yy,0};
            DrawLine3D(prev, cur, Color{170,210,255,255});
            prev = cur;
        }

        DrawCube(massPos, 0.45f, 0.45f, 0.45f, Color{255, 200, 120, 255});

        EndMode3D();

        studio::title("Damped Forced Oscillator (Driven Spring-Mass)", studio::Style::Instrument);
        studio::help("Hold left mouse: orbit | wheel: zoom | [ ] drive freq | +/- damping | P pause | R reset");

        float force = F0 * std::cos(w * t);
        std::ostringstream os;
        os << std::fixed << std::setprecision(3)
           << "x=" << x << "  v=" << v << "  F_drive=" << force << "  w=" << w << "  c=" << c;
        if (paused) os << "  [PAUSED]";
        studio::readout(os.str().c_str());
        studio::note(TextFormat("Natural frequency %.3f rad/s / energy %.3f / fixed 240 Hz RK4",std::sqrt(k/m),0.5*m*v*v+0.5*k*x*x));
        studio::plot({float(GetScreenWidth()-388),175,360,205},"Steady amplitude / drive 0 to 8 rad/s",response,0,5,Color{230,190,125,255});
        studio::plot({float(GetScreenWidth()-388),float(GetScreenHeight()-278),360,210},"Displacement / last 8 s",hist,-4,4,Color{134,211,226,255});
        studio::fps();

        EndDrawing();
        if (studio::smokeFrame(__FILE__)) break;
    }

    studio::unload();
    CloseWindow();
    return 0;
}
