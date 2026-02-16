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

constexpr float kSourceZ = -6.5f;
constexpr float kLensZ = 0.0f;
constexpr float kObserverZ = 6.5f;

struct RayPath {
    Vector3 source;
    Vector3 impact;
    Vector3 end;
    bool toObserver;
};

void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 delta = GetMouseDelta();
        *yaw -= delta.x * 0.0035f;
        *pitch += delta.y * 0.0035f;
        *pitch = std::clamp(*pitch, -1.35f, 1.35f);
    }

    *distance -= GetMouseWheelMove() * 0.6f;
    *distance = std::clamp(*distance, 4.0f, 28.0f);

    float cp = std::cos(*pitch);
    Vector3 offset = {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}

std::vector<RayPath> BuildRays(Vector3 sourcePos, float lensStrength, int nRays) {
    std::vector<RayPath> rays;
    rays.reserve(nRays);

    const Vector3 observer = {0.0f, 0.0f, kObserverZ};

    for (int i = 0; i < nRays; ++i) {
        float a = 2.0f * PI * static_cast<float>(i) / static_cast<float>(nRays);

        float ring = 0.45f + 0.85f * (0.5f + 0.5f * std::sin(a * 3.0f));
        Vector3 impact = {
            ring * std::cos(a + 0.6f),
            ring * std::sin(a + 0.6f),
            kLensZ,
        };

        Vector3 bVec = {impact.x, impact.y, 0.0f};
        float b = std::sqrt(impact.x * impact.x + impact.y * impact.y) + 0.06f;

        float alpha = lensStrength / (b * b + 0.08f);
        Vector3 towardLens = Vector3Normalize(Vector3Negate(bVec));

        Vector3 dir = Vector3Normalize(Vector3Subtract(observer, impact));
        dir = Vector3Normalize(Vector3Add(dir, Vector3Scale(towardLens, alpha)));

        float t = (kObserverZ - impact.z) / std::max(0.001f, dir.z);
        Vector3 end = Vector3Add(impact, Vector3Scale(dir, t));

        bool hitsObserver = (std::fabs(end.x) < 0.45f && std::fabs(end.y) < 0.45f);
        if (!hitsObserver) {
            end = Vector3Add(impact, Vector3Scale(dir, 7.5f));
        }

        rays.push_back({sourcePos, impact, end, hitsObserver});
    }

    return rays;
}

void DrawCircleXY(float z, float radius, Color c, int segs) {
    for (int i = 0; i < segs; ++i) {
        float a0 = 2.0f * PI * static_cast<float>(i) / static_cast<float>(segs);
        float a1 = 2.0f * PI * static_cast<float>(i + 1) / static_cast<float>(segs);
        DrawLine3D(
            {radius * std::cos(a0), radius * std::sin(a0), z},
            {radius * std::cos(a1), radius * std::sin(a1), z},
            c
        );
    }
}

std::string Hud(float lensStrength, Vector3 sourcePos, int hitCount) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(2)
       << "lens=" << lensStrength
       << "  source=(" << sourcePos.x << "," << sourcePos.y << ")"
       << "  focused rays=" << hitCount;
    return os.str();
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Gravitational Lensing 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {7.0f, 4.8f, 9.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.9f;
    float camPitch = 0.35f;
    float camDistance = 13.0f;

    float lensStrength = 0.85f;
    Vector3 sourcePos = {1.3f, 0.3f, kSourceZ};
    bool paused = false;
    float t = 0.0f;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            lensStrength = 0.85f;
            sourcePos = {1.3f, 0.3f, kSourceZ};
            t = 0.0f;
        }
        if (IsKeyPressed(KEY_LEFT_BRACKET)) lensStrength = std::max(0.15f, lensStrength - 0.08f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) lensStrength = std::min(2.2f, lensStrength + 0.08f);

        float move = 1.7f * GetFrameTime();
        if (IsKeyDown(KEY_LEFT)) sourcePos.x -= move;
        if (IsKeyDown(KEY_RIGHT)) sourcePos.x += move;
        if (IsKeyDown(KEY_UP)) sourcePos.y += move;
        if (IsKeyDown(KEY_DOWN)) sourcePos.y -= move;
        sourcePos.x = std::clamp(sourcePos.x, -3.0f, 3.0f);
        sourcePos.y = std::clamp(sourcePos.y, -3.0f, 3.0f);

        if (!paused) {
            t += GetFrameTime();
            sourcePos.x += 0.12f * std::cos(t * 0.7f) * GetFrameTime();
            sourcePos.y += 0.10f * std::sin(t * 0.9f) * GetFrameTime();
            sourcePos.x = std::clamp(sourcePos.x, -3.0f, 3.0f);
            sourcePos.y = std::clamp(sourcePos.y, -3.0f, 3.0f);
        }

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        std::vector<RayPath> rays = BuildRays(sourcePos, lensStrength, 44);
        int hitCount = 0;
        for (const RayPath& r : rays) {
            if (r.toObserver) hitCount++;
        }

        BeginDrawing();
        ClearBackground(Color{6, 9, 17, 255});

        BeginMode3D(camera);

        DrawCircleXY(kSourceZ, 0.7f, Color{120, 170, 255, 60}, 80);
        DrawCircleXY(kLensZ, 1.2f, Color{255, 190, 90, 60}, 90);
        DrawCircleXY(kObserverZ, 0.55f, Color{160, 230, 255, 70}, 80);

        DrawSphere(sourcePos, 0.18f, Color{130, 185, 255, 255});
        DrawSphere({0.0f, 0.0f, kLensZ}, 0.44f, BLACK);
        DrawSphere({0.0f, 0.0f, kLensZ}, 0.64f, Color{255, 165, 90, 50});
        DrawSphere({0.0f, 0.0f, kObserverZ}, 0.16f, Color{140, 230, 255, 255});

        for (const RayPath& r : rays) {
            Color cIn = Color{130, 170, 255, 110};
            Color cOut = r.toObserver ? Color{255, 235, 170, 190} : Color{255, 140, 110, 95};
            DrawLine3D(r.source, r.impact, cIn);
            DrawLine3D(r.impact, r.end, cOut);
            DrawSphere(r.impact, 0.03f, Color{255, 180, 120, 170});
        }

        EndMode3D();

        DrawText("Gravitational Lensing (Thin Lens Approximation)", 20, 18, 29, Color{232, 238, 248, 255});
        DrawText("Hold left mouse: orbit | wheel: zoom | arrows: move source | [ ] lens mass | P pause | R reset", 20, 54, 19, Color{164, 183, 210, 255});
        std::string hud = Hud(lensStrength, sourcePos, hitCount);
        DrawText(hud.c_str(), 20, 82, 21, Color{126, 224, 255, 255});
        DrawFPS(20, 114);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
