#include "raylib.h"
#include "../common/studio.h"
#include "../common/physics_models.h"
#include <deque>
#include <array>
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {
constexpr int kScreenWidth = 1280;
constexpr int kScreenHeight = 820;
struct Dot { Vector3 p; Vector3 v; };

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
}

int main() {
    if (std::getenv("COMPPHYSICS_SMOKE_FRAMES")) SetConfigFlags(FLAG_WINDOW_HIDDEN);
    InitWindow(kScreenWidth, kScreenHeight, "Entropy Mixing 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.0f, 5.0f, 9.0f};
    camera.target = {0,0.7f,0};
    camera.up = {0,1,0};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float camYaw=0.84f, camPitch=0.34f, camDistance=13.0f;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> u(-1.0f,1.0f);

    std::vector<Dot> dots;
    auto reset = [&]() {
        dots.clear();
        for (int i=0;i<220;++i) {
            float side = (i < 110) ? -1.0f : 1.0f;
            Vector3 p = {side*1.2f + 0.6f*u(rng), 0.6f + 0.9f*u(rng), 1.2f*u(rng)};
            Vector3 v = {1.2f*u(rng), 1.2f*u(rng), 1.2f*u(rng)};
            dots.push_back({p,v});
        }
    };

    physics::Clock clock;
    std::deque<float> mixingHistory;
    int samples=0;
    bool wall = true;
    bool paused = false;
    reset();

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused=!paused;
        if (IsKeyPressed(KEY_W)) wall = !wall;
        if (IsKeyPressed(KEY_R)) { reset(); wall=true; paused=false; clock.reset(); mixingHistory.clear(); samples=0; }

        UpdateOrbitCameraDragOnly(&camera,&camYaw,&camPitch,&camDistance);

        auto mixingIndex=[&]() {
            std::array<std::array<int,2>,16> bins{};
            for (size_t i=0;i<dots.size();++i) {
                const auto& p=dots[i].p;
                int bx=std::clamp(int((p.x+2.2f)/4.4f*4),0,3);
                int by=std::clamp(int((p.y+0.3f)/1.8f*2),0,1);
                int bz=std::clamp(int((p.z+1.6f)/3.2f*2),0,1);
                bins[bx+4*by+8*bz][i<110 ? 0 : 1]++;
            }
            double mix=0;
            for (const auto& bin : bins)
                mix+=(bin[0]+bin[1])*physics::binaryEntropy(bin[0],bin[1]);
            return float(mix/dots.size());
        };
        if (!paused) clock.advance(GetFrameTime(),[&](double dt) {
            for (auto& d : dots) {
                auto bounce=[&](float& p,float& v,double lo,double hi) {
                    double pp=p,vv=v;
                    physics::reflect(pp,vv,lo,hi,dt);
                    p=float(pp); v=float(vv);
                };
                double lo=-2.2, hi=2.2;
                if (wall) { if (d.p.x<0) hi=0; else lo=0; }
                bounce(d.p.x,d.v.x,lo,hi);
                bounce(d.p.y,d.v.y,-0.3,1.5);
                bounce(d.p.z,d.v.z,-1.6,1.6);
            }
            if (++samples%8==0) {
                mixingHistory.push_back(mixingIndex());
                if (mixingHistory.size()>240) mixingHistory.pop_front();
            }
        });
        float mix=mixingIndex();
        int leftCount=0;
        for (const auto& d : dots) if (d.p.x<0) ++leftCount;

        BeginDrawing();
        ClearBackground(Color{6,9,16,255});
        BeginMode3D(camera);
        DrawCubeWires({0,0.6f,0}, 4.6f, 2.0f, 3.4f, Color{130,180,255,180});
        if (wall) DrawCube({0,0.6f,0}, 0.05f, 1.9f, 3.2f, Color{170,170,190,130});

        for (int i=0;i<(int)dots.size();++i) {
            Color c = (i<110) ? Color{255,140,120,230} : Color{120,200,255,230};
            DrawSphere(dots[i].p, 0.045f, c);
        }

        EndMode3D();

        studio::title("Entropy and Mixing (Box Gas Model)", studio::Style::Instrument);
        studio::help("Hold left mouse: orbit | wheel: zoom | W toggle partition | P pause | R reset");

        std::ostringstream os;
        os << std::fixed << std::setprecision(2) << "left count=" << leftCount << "  local color mixing=" << mix;
        if (wall) os << "  [partition ON]";
        if (paused) os << "  [PAUSED]";
        studio::readout(os.str().c_str());
        studio::note("Local color entropy in 16 bins / finite sampling / noninteracting gas");
        studio::plot({float(GetScreenWidth()-388),float(GetScreenHeight()-278),360,210},
                     "Color mixing / last 8 s",mixingHistory,0,1,Color{139,220,190,255});
        studio::fps();

        EndDrawing();
        if (studio::smokeFrame(__FILE__)) break;
    }

    studio::unload();
    CloseWindow();
    return 0;
}
