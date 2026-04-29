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

struct Charge {
    Vector3 pos;
    float q;
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

Vector3 EFieldAt(Vector3 p, const std::vector<Charge>& charges) {
    Vector3 e{0,0,0};
    for (const Charge& c : charges) {
        Vector3 r = Vector3Subtract(p, c.pos);
        float d2 = Vector3DotProduct(r, r) + 0.08f;
        float invR = 1.0f / std::sqrt(d2);
        float invR3 = invR * invR * invR;
        e = Vector3Add(e, Vector3Scale(r, c.q * invR3));
    }
    return e;
}

void DrawArrow(Vector3 a, Vector3 b, Color c) {
    DrawLine3D(a,b,c);
    Vector3 d = Vector3Normalize(Vector3Subtract(b,a));
    Vector3 s = Vector3Normalize(Vector3CrossProduct(d, {0,1,0}));
    if (Vector3Length(s) < 1e-4f) s = {1,0,0};
    DrawLine3D(b, Vector3Add(b, Vector3Add(Vector3Scale(d,-0.12f), Vector3Scale(s,0.06f))), c);
    DrawLine3D(b, Vector3Add(b, Vector3Add(Vector3Scale(d,-0.12f), Vector3Scale(s,-0.06f))), c);
}

}

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Electric Field 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {7.6f, 4.8f, 8.4f};
    camera.target = {0,0.6f,0};
    camera.up = {0,1,0};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw=0.84f, camPitch=0.34f, camDistance=13.0f;

    std::vector<Charge> charges = {
        {{-1.4f, 0.6f, 0.0f}, +1.0f},
        {{ 1.4f, 0.6f, 0.0f}, -1.0f}
    };

    bool paused = false;
    float t = 0.0f;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            charges[0].q = +1.0f; charges[1].q = -1.0f;
            charges[0].pos = {-1.4f,0.6f,0}; charges[1].pos = {1.4f,0.6f,0};
            paused = false; t = 0.0f;
        }
        if (IsKeyPressed(KEY_ONE)) charges[0].q *= -1.0f;
        if (IsKeyPressed(KEY_TWO)) charges[1].q *= -1.0f;
        if (IsKeyDown(KEY_LEFT)) charges[1].pos.x -= 0.9f * GetFrameTime();
        if (IsKeyDown(KEY_RIGHT)) charges[1].pos.x += 0.9f * GetFrameTime();
        charges[1].pos.x = std::clamp(charges[1].pos.x, -3.0f, 3.0f);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);
        if (!paused) t += GetFrameTime();

        BeginDrawing();
        ClearBackground(Color{6,9,16,255});
        BeginMode3D(camera);

        for (int ix=-6; ix<=6; ++ix) {
            for (int iz=-6; iz<=6; ++iz) {
                Vector3 p = {0.5f*ix, 0.6f, 0.5f*iz};
                Vector3 e = EFieldAt(p, charges);
                float m = Vector3Length(e);
                if (m < 0.05f) continue;
                Vector3 d = Vector3Scale(Vector3Normalize(e), std::min(0.35f, 0.12f + 0.08f*m));
                Color c = Color{static_cast<unsigned char>(120 + std::min(130.0f, 40.0f*m)), 180, 255, 220};
                DrawArrow(p, Vector3Add(p,d), c);
            }
        }

        for (const Charge& c : charges) {
            Color col = (c.q > 0) ? Color{255,120,120,255} : Color{120,160,255,255};
            DrawSphere(c.pos, 0.18f, col);
        }

        EndMode3D();

        DrawText("Electric Field of Point Charges", 20, 18, 29, Color{232,238,248,255});
        DrawText("Hold left mouse: orbit | wheel: zoom | 1/2 flip charge sign | arrows move charge 2 | P pause | R reset", 20, 54, 18, Color{164,183,210,255});

        std::ostringstream os;
        os << std::fixed << std::setprecision(2)
           << "q1=" << charges[0].q << "  q2=" << charges[1].q << "  x2=" << charges[1].pos.x;
        if (paused) os << "  [PAUSED]";
        DrawText(os.str().c_str(), 20, 82, 20, Color{126,224,255,255});
        DrawFPS(20,110);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
