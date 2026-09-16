#include "raylib.h"
#include "../common/studio.h"
#include "../common/physics_models.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>

namespace {
constexpr int kW = 1280;
constexpr int kH = 820;
constexpr float kG = 1.0f;

void UpdateOrbitCameraDragOnly(Camera3D* c, float* yaw, float* pitch, float* dist) {
    studio::pan(c, *yaw, *pitch, *dist);
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON) && !studio::panGesture()) {
        Vector2 d = GetMouseDelta();
        *yaw -= d.x * 0.0035f;
        *pitch += d.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.3f, 1.3f);
    }
    *dist -= GetMouseWheelMove() * 0.6f;
    *dist = std::clamp(*dist, 5.0f, 40.0f);
    float cp = std::cos(*pitch);
    c->position = Vector3Add(c->target, {*dist * cp * std::cos(*yaw), *dist * std::sin(*pitch), *dist * cp * std::sin(*yaw)});
}

float TransitFlux(Vector3 planetPos, float starR, float planetR) {
    // Observer fixed on +X axis looking toward origin.
    if (planetPos.x < 0.0f) return 1.0f;
    float d = std::sqrt(planetPos.y * planetPos.y + planetPos.z * planetPos.z);
    return float(1.0-physics::circleOverlap(d,starR,planetR)/(physics::pi*starR*starR));
}
}

int main() {
    if (std::getenv("COMPPHYSICS_SMOKE_FRAMES")) SetConfigFlags(FLAG_WINDOW_HIDDEN);
    InitWindow(kW, kH, "Exoplanet Transit Lab 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D cam{};
    cam.position = {8.8f, 4.8f, 8.8f};
    cam.target = {0, 0, 0};
    cam.up = {0, 1, 0};
    cam.fovy = 45.0f;
    cam.projection = CAMERA_PERSPECTIVE;
    float yaw = 0.8f, pitch = 0.33f, dist = 14.5f;

    float starMass = 130.0f;
    float starR = 1.2f;
    float planetR = 0.26f;
    bool paused = false;
    physics::Clock clock;
    int samples=0;

    Vector3 p = {0.0f, 0.0f, 4.5f};
    Vector3 v = {5.1f, 0.0f, 0.0f};
    std::deque<float> fluxHistory(360, 1.0f);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            p = {0.0f, 0.0f, 4.5f};
            v = {5.1f, 0.0f, 0.0f};
            fluxHistory.assign(360, 1.0f);
            starMass = 130.0f;
            planetR = 0.26f;
            paused = false; clock.reset(); samples=0;
        }
        if (IsKeyDown(KEY_UP)) starMass = std::min(260.0f, starMass + 45.0f * GetFrameTime());
        if (IsKeyDown(KEY_DOWN)) starMass = std::max(35.0f, starMass - 45.0f * GetFrameTime());
        if (IsKeyDown(KEY_RIGHT_BRACKET)) planetR = std::min(0.56f, planetR + 0.32f * GetFrameTime());
        if (IsKeyDown(KEY_LEFT_BRACKET)) planetR = std::max(0.08f, planetR - 0.32f * GetFrameTime());

        UpdateOrbitCameraDragOnly(&cam, &yaw, &pitch, &dist);

        if (!paused) clock.advance(GetFrameTime(),[&](double step) {
            float dt=float(step);
            auto acceleration=[&](Vector3 position) {
                float radius=std::max(0.35f,Vector3Length(position));
                return Vector3Scale(position,-kG*starMass/(radius*radius*radius));
            };
            Vector3 a=acceleration(p);
            p=Vector3Add(p,Vector3Add(Vector3Scale(v,dt),Vector3Scale(a,0.5f*dt*dt)));
            v=Vector3Add(v,Vector3Scale(Vector3Add(a,acceleration(p)),0.5f*dt));
            if (++samples%4==0) {
                fluxHistory.push_back(TransitFlux(p,starR,planetR));
                if (fluxHistory.size()>360) fluxHistory.pop_front();
            }
        });

        BeginDrawing();
        ClearBackground(Color{7, 10, 18, 255});
        BeginMode3D(cam);
        DrawSphere({0, 0, 0}, starR, Color{255, 196, 96, 255});
        DrawSphere(p, planetR, Color{130, 200, 255, 255});
        DrawLine3D({0, 0, 0}, p, Fade(SKYBLUE, 0.3f));
        DrawGrid(20, 0.8f);
        EndMode3D();

        studio::plot({float(GetScreenWidth()-398),float(GetScreenHeight()-293),370,225},
                     "Relative stellar flux / last 6 s",fluxHistory,0.75f,1.01f,Color{142,226,185,255});

        studio::title("Exoplanet Transit Lab (3D gravity orbit)", studio::Style::Observatory);
        studio::help("Mouse drag orbit | wheel zoom | Up/Down star mass | [ ] planet radius | P pause | R reset");
        char s[220];
        std::snprintf(s, sizeof(s), "M*=%.1f  Rp=%.2f  flux=%.4f%s", starMass, planetR, fluxHistory.back(), paused ? "  [PAUSED]" : "");
        studio::readout(s);
        studio::note("Uniform stellar disk / exact projected overlap / observer on +X / Verlet orbit");
        studio::fps();
        EndDrawing();
        if (studio::smokeFrame(__FILE__)) break;
    }

    studio::unload();
    CloseWindow();
    return 0;
}
