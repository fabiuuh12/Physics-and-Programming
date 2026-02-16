#include "raylib.h"
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

void DrawPotentialWell(float a, float w, float V0) {
    const int n=180;
    for (int i=1;i<n;++i) {
        float x0 = -4.5f + 9.0f*(i-1)/(n-1.0f);
        float x1 = -4.5f + 9.0f*i/(n-1.0f);
        float v0 = V0*((x0*x0)/(a*a) + w*std::sin(2*x0));
        float v1 = V0*((x1*x1)/(a*a) + w*std::sin(2*x1));
        DrawLine3D({x0,0.2f+0.2f*v0,-1.4f}, {x1,0.2f+0.2f*v1,-1.4f}, Color{255,170,120,220});
    }
}
}

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Quantum Potential + States 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.3f, 5.2f, 8.8f};
    camera.target = {0,0.6f,0};
    camera.up = {0,1,0};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float camYaw=0.84f, camPitch=0.34f, camDistance=13.5f;

    float a=2.0f;
    float w=0.35f;
    float V0=0.8f;
    int nState=2;
    bool paused=false;
    float t=0.0f;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused=!paused;
        if (IsKeyPressed(KEY_R)) { a=2.0f; w=0.35f; V0=0.8f; nState=2; paused=false; t=0.0f; }
        if (IsKeyPressed(KEY_LEFT_BRACKET)) nState = std::max(1, nState-1);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) nState = std::min(8, nState+1);
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) V0 = std::max(0.2f, V0-0.05f);
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) V0 = std::min(2.0f, V0+0.05f);

        UpdateOrbitCameraDragOnly(&camera,&camYaw,&camPitch,&camDistance);
        if (!paused) t += GetFrameTime();

        BeginDrawing();
        ClearBackground(Color{6,9,16,255});
        BeginMode3D(camera);

        DrawGrid(24,0.5f);
        DrawPotentialWell(a,w,V0);

        const int n=200;
        for (int i=1;i<n;++i) {
            float x0 = -4.0f + 8.0f*(i-1)/(n-1.0f);
            float x1 = -4.0f + 8.0f*i/(n-1.0f);
            float psi0 = std::sin(nState*PI*(x0+4.0f)/8.0f);
            float psi1 = std::sin(nState*PI*(x1+4.0f)/8.0f);
            float y0 = 0.8f + 0.6f*psi0*std::cos(2.0f*t);
            float y1 = 0.8f + 0.6f*psi1*std::cos(2.0f*t);
            DrawLine3D({x0,y0,0.0f}, {x1,y1,0.0f}, Color{130,220,255,255});
            DrawLine3D({x0,0.35f+0.45f*psi0*psi0,1.3f}, {x1,0.35f+0.45f*psi1*psi1,1.3f}, Color{255,210,130,220});
        }

        EndMode3D();

        DrawText("Quantum States in a Potential (Conceptual)", 20, 18, 29, Color{232,238,248,255});
        DrawText("Hold left mouse: orbit | wheel: zoom | [ ] quantum state n | +/- potential depth | P pause | R reset", 20, 54, 18, Color{164,183,210,255});

        std::ostringstream os;
        os << std::fixed << std::setprecision(2) << "n=" << nState << "  V0=" << V0 << "  (cyan: wavefunction, yellow: probability)";
        if (paused) os << "  [PAUSED]";
        DrawText(os.str().c_str(), 20, 82, 20, Color{126,224,255,255});
        DrawFPS(20,110);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
