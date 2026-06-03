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

struct WindParticle {
    float angle;
    float radius;
    float speed;
    float y;
};

void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 delta = GetMouseDelta();
        *yaw -= delta.x * 0.0035f;
        *pitch += delta.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }

    *distance -= GetMouseWheelMove() * 0.6f;
    *distance = std::clamp(*distance, 4.0f, 30.0f);

    float cp = std::cos(*pitch);
    Vector3 offset = {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}

Vector3 MagneticAxis(float spinAngle, float tilt) {
    return Vector3Normalize({
        std::sin(tilt) * std::cos(spinAngle),
        std::cos(tilt),
        std::sin(tilt) * std::sin(spinAngle),
    });
}

void DrawBeam(Vector3 axis, float length, float opening, Color color) {
    Vector3 start = Vector3Scale(axis, 0.42f);
    Vector3 end = Vector3Scale(axis, length);
    DrawCylinderEx(start, end, opening * 0.25f, opening, 20, color);
}

void DrawMagneticGuideRings(Vector3 axis) {
    Vector3 ref = (std::fabs(axis.y) < 0.9f) ? Vector3{0.0f, 1.0f, 0.0f} : Vector3{1.0f, 0.0f, 0.0f};
    Vector3 e1 = Vector3Normalize(Vector3CrossProduct(axis, ref));
    Vector3 e2 = Vector3Normalize(Vector3CrossProduct(axis, e1));

    for (int ring = 0; ring < 4; ++ring) {
        float r = 0.8f + 0.55f * static_cast<float>(ring);
        for (int i = 0; i < 120; ++i) {
            float a0 = 2.0f * PI * static_cast<float>(i) / 120.0f;
            float a1 = 2.0f * PI * static_cast<float>(i + 1) / 120.0f;
            Vector3 p0 = Vector3Add(Vector3Scale(e1, r * std::cos(a0)), Vector3Scale(e2, r * std::sin(a0)));
            Vector3 p1 = Vector3Add(Vector3Scale(e1, r * std::cos(a1)), Vector3Scale(e2, r * std::sin(a1)));
            DrawLine3D(p0, p1, Color{130, 210, 255, static_cast<unsigned char>(45 - ring * 8)});
        }
    }
}

std::string Hud(float simTime, float spinRate, float tiltDeg, float pulse, bool paused) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(2)
       << "t=" << simTime
       << "  spin=" << spinRate << "x"
       << "  tilt=" << tiltDeg << " deg"
       << "  pulse=" << pulse;
    if (paused) os << "  [PAUSED]";
    return os.str();
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Pulsar 3D Visualization - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.0f, 4.8f, 8.5f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.82f;
    float camPitch = 0.35f;
    float camDistance = 13.0f;

    float simTime = 0.0f;
    float spinAngle = 0.0f;
    float spinRate = 1.0f;
    float tilt = 0.55f;
    float beamLength = 8.0f;
    bool paused = false;
    bool showGuides = true;

    WindParticle wind[180];
    for (int i = 0; i < 180; ++i) {
        wind[i].angle = 2.0f * PI * static_cast<float>(i) / 180.0f;
        wind[i].radius = 0.9f + 0.03f * static_cast<float>(i % 7);
        wind[i].speed = 1.0f + 0.6f * static_cast<float>((i * 11) % 13) / 12.0f;
        wind[i].y = -0.14f + 0.28f * static_cast<float>((i * 17) % 19) / 18.0f;
    }

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            simTime = 0.0f;
            spinAngle = 0.0f;
            spinRate = 1.0f;
            tilt = 0.55f;
            beamLength = 8.0f;
            paused = false;
        }
        if (IsKeyPressed(KEY_G)) showGuides = !showGuides;
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) spinRate = std::min(8.0f, spinRate + 0.25f);
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) spinRate = std::max(0.25f, spinRate - 0.25f);
        if (IsKeyPressed(KEY_LEFT_BRACKET)) tilt = std::max(0.15f, tilt - 0.03f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) tilt = std::min(1.25f, tilt + 0.03f);
        if (IsKeyPressed(KEY_COMMA)) beamLength = std::max(3.5f, beamLength - 0.3f);
        if (IsKeyPressed(KEY_PERIOD)) beamLength = std::min(12.0f, beamLength + 0.3f);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        float dt = GetFrameTime();
        if (!paused) {
            simTime += dt;
            spinAngle += dt * spinRate * 8.0f;
            for (WindParticle& p : wind) {
                p.radius += dt * p.speed;
                p.angle += dt * (0.7f + p.speed * 0.3f);
                if (p.radius > 10.5f) p.radius = 0.85f;
            }
        }

        Vector3 magAxis = MagneticAxis(spinAngle, tilt);
        Vector3 obsDir = Vector3Normalize(camera.position);
        float beamAlign = std::max(std::fabs(Vector3DotProduct(magAxis, obsDir)), 0.0f);
        float pulse = std::pow(beamAlign, 8.0f);

        BeginDrawing();
        ClearBackground(Color{4, 7, 17, 255});

        BeginMode3D(camera);

        if (showGuides) {
            DrawLine3D(Vector3Scale(magAxis, -2.5f), Vector3Scale(magAxis, 2.5f), Color{130, 220, 255, 170});
            DrawMagneticGuideRings(magAxis);
        }

        Color beamCore = Color{
            static_cast<unsigned char>(180 + 75 * pulse),
            static_cast<unsigned char>(220 + 35 * pulse),
            255,
            static_cast<unsigned char>(75 + 130 * pulse),
        };
        Color beamGlow = Color{
            static_cast<unsigned char>(100 + 90 * pulse),
            static_cast<unsigned char>(160 + 70 * pulse),
            255,
            static_cast<unsigned char>(35 + 80 * pulse),
        };

        DrawBeam(magAxis, beamLength, 0.28f, beamCore);
        DrawBeam(Vector3Negate(magAxis), beamLength, 0.28f, beamCore);
        DrawBeam(magAxis, beamLength + 1.6f, 0.62f, beamGlow);
        DrawBeam(Vector3Negate(magAxis), beamLength + 1.6f, 0.62f, beamGlow);

        DrawSphere({0.0f, 0.0f, 0.0f}, 0.45f, Color{175, 210, 255, 255});
        DrawSphereWires({0.0f, 0.0f, 0.0f}, 0.52f, 14, 14, Color{205, 235, 255, 130});
        DrawSphere({0.0f, 0.0f, 0.0f}, 0.2f, Color{250, 250, 255, static_cast<unsigned char>(170 + 80 * pulse)});

        for (const WindParticle& p : wind) {
            Vector3 pos = {p.radius * std::cos(p.angle), p.y, p.radius * std::sin(p.angle)};
            unsigned char a = static_cast<unsigned char>(170 * (1.0f - p.radius / 10.5f));
            DrawSphere(pos, 0.03f, Color{150, 215, 255, a});
        }

        EndMode3D();

        DrawText("Pulsar (Rotating Magnetized Neutron Star)", 20, 18, 30, Color{232, 238, 248, 255});
        DrawText("Hold left mouse: orbit | wheel: zoom | +/- spin | [ ] tilt | , . beam | G guides | P pause | R reset",
                 20, 54, 19, Color{164, 183, 210, 255});
        std::string hud = Hud(simTime, spinRate, tilt * 180.0f / PI, pulse, paused);
        DrawText(hud.c_str(), 20, 82, 21, Color{126, 224, 255, 255});
        DrawFPS(20, 114);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
