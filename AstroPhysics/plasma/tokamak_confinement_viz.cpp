#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace {

constexpr int kScreenWidth = 1440;
constexpr int kScreenHeight = 900;
constexpr float kMajorRadius = 5.8f;
constexpr float kMinorRadius = 1.55f;
constexpr int kPlasmaParticleCount = 680;
constexpr int kFieldLineCount = 12;
constexpr int kRingSegments = 92;

struct OrbitCameraState {
    float yaw = 0.78f;
    float pitch = 0.34f;
    float distance = 18.0f;
};

struct PlasmaParticle {
    float toroidal = 0.0f;
    float poloidal = 0.0f;
    float toroidalSpeed = 0.0f;
    float poloidalSpeed = 0.0f;
    float radialOffset = 0.0f;
    float size = 0.0f;
    float heat = 0.0f;
};

float RandRange(std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    return dist(rng);
}

Color LerpColor(Color a, Color b, float t) {
    const float u = std::clamp(t, 0.0f, 1.0f);
    return Color{
        static_cast<unsigned char>(a.r + (b.r - a.r) * u),
        static_cast<unsigned char>(a.g + (b.g - a.g) * u),
        static_cast<unsigned char>(a.b + (b.b - a.b) * u),
        static_cast<unsigned char>(a.a + (b.a - a.a) * u),
    };
}

void UpdateOrbitCameraDragOnly(Camera3D* camera, OrbitCameraState* orbit) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        const Vector2 delta = GetMouseDelta();
        orbit->yaw -= delta.x * 0.0039f;
        orbit->pitch += delta.y * 0.0039f;
        orbit->pitch = std::clamp(orbit->pitch, -1.28f, 1.28f);
    }

    orbit->distance -= GetMouseWheelMove() * 0.8f;
    orbit->distance = std::clamp(orbit->distance, 8.0f, 32.0f);

    const float cp = std::cos(orbit->pitch);
    camera->target = {0.0f, 0.0f, 0.0f};
    camera->position = Vector3Add(camera->target, {
        orbit->distance * cp * std::cos(orbit->yaw),
        orbit->distance * std::sin(orbit->pitch),
        orbit->distance * cp * std::sin(orbit->yaw),
    });
}

Vector3 TokamakSurfacePoint(float theta, float phi, float minorRadius) {
    const float cphi = std::cos(phi);
    const float sphi = std::sin(phi);
    const float ctheta = std::cos(theta);
    const float stheta = std::sin(theta);
    const float tube = kMajorRadius + minorRadius * cphi;
    return {tube * ctheta, minorRadius * sphi, tube * stheta};
}

Vector3 PlasmaPoint(const PlasmaParticle& p, float time) {
    const float theta = p.toroidal + time * p.toroidalSpeed;
    const float phi = p.poloidal + time * p.poloidalSpeed;
    const float radius = kMinorRadius * (0.28f + 0.62f * p.radialOffset);
    return TokamakSurfacePoint(theta, phi, radius);
}

std::vector<PlasmaParticle> MakeParticles() {
    std::mt19937 rng(2024);
    std::vector<PlasmaParticle> particles;
    particles.reserve(kPlasmaParticleCount);

    for (int i = 0; i < kPlasmaParticleCount; ++i) {
        particles.push_back({
            RandRange(rng, 0.0f, 2.0f * PI),
            RandRange(rng, 0.0f, 2.0f * PI),
            RandRange(rng, 0.55f, 1.55f),
            RandRange(rng, 2.2f, 5.4f),
            RandRange(rng, 0.18f, 1.0f),
            RandRange(rng, 0.025f, 0.085f),
            RandRange(rng, 0.0f, 1.0f),
        });
    }

    return particles;
}

void DrawTokamakFrame(float time) {
    for (int band = 0; band < 7; ++band) {
        const float phi = (2.0f * PI * band) / 7.0f + time * 0.03f;
        for (int i = 0; i < kRingSegments; ++i) {
            const float a0 = (2.0f * PI * i) / kRingSegments;
            const float a1 = (2.0f * PI * (i + 1)) / kRingSegments;
            const Vector3 p0 = TokamakSurfacePoint(a0, phi, kMinorRadius);
            const Vector3 p1 = TokamakSurfacePoint(a1, phi, kMinorRadius);
            DrawLine3D(p0, p1, Fade(Color{94, 116, 168, 255}, 0.42f));
        }
    }

    for (int i = 0; i < 18; ++i) {
        const float theta = (2.0f * PI * i) / 18.0f;
        for (int s = 0; s < kRingSegments; ++s) {
            const float p0 = (2.0f * PI * s) / kRingSegments;
            const float p1 = (2.0f * PI * (s + 1)) / kRingSegments;
            const Vector3 a = TokamakSurfacePoint(theta, p0, kMinorRadius);
            const Vector3 b = TokamakSurfacePoint(theta, p1, kMinorRadius);
            DrawLine3D(a, b, Fade(Color{86, 104, 148, 255}, 0.24f));
        }
    }
}

void DrawFieldLines(float time) {
    for (int line = 0; line < kFieldLineCount; ++line) {
        const float basePoloidal = (2.0f * PI * line) / kFieldLineCount;
        Vector3 prev{};
        bool hasPrev = false;

        for (int i = 0; i <= 220; ++i) {
            const float u = static_cast<float>(i) / 220.0f;
            const float theta = 2.0f * PI * 2.0f * u + time * 0.20f;
            const float phi = basePoloidal + theta * 4.8f;
            const Vector3 p = TokamakSurfacePoint(theta, phi, kMinorRadius * 0.72f);
            if (hasPrev) {
                DrawLine3D(prev, p, Fade(Color{105, 224, 255, 255}, 0.30f));
            }
            prev = p;
            hasPrev = true;
        }
    }
}

void DrawReactorShell() {
    for (int i = 0; i < 26; ++i) {
        const float theta = (2.0f * PI * i) / 26.0f;
        const Vector3 p = TokamakSurfacePoint(theta, 0.0f, kMinorRadius + 0.52f);
        DrawCylinderEx({p.x, -2.2f, p.z}, {p.x, 2.2f, p.z}, 0.06f, 0.06f, 8, Fade(Color{86, 96, 122, 255}, 0.75f));
    }

    for (int i = 0; i < 40; ++i) {
        const float a0 = (2.0f * PI * i) / 40.0f;
        const float a1 = (2.0f * PI * (i + 1)) / 40.0f;
        const Vector3 top0 = {(kMajorRadius + 2.8f) * std::cos(a0), 2.2f, (kMajorRadius + 2.8f) * std::sin(a0)};
        const Vector3 top1 = {(kMajorRadius + 2.8f) * std::cos(a1), 2.2f, (kMajorRadius + 2.8f) * std::sin(a1)};
        const Vector3 bot0 = {(kMajorRadius + 2.8f) * std::cos(a0), -2.2f, (kMajorRadius + 2.8f) * std::sin(a0)};
        const Vector3 bot1 = {(kMajorRadius + 2.8f) * std::cos(a1), -2.2f, (kMajorRadius + 2.8f) * std::sin(a1)};
        DrawLine3D(top0, top1, Fade(Color{118, 130, 166, 255}, 0.30f));
        DrawLine3D(bot0, bot1, Fade(Color{118, 130, 166, 255}, 0.30f));
    }
}

}  // namespace

int main() {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(kScreenWidth, kScreenHeight, "Tokamak Confinement 3D - C++ (raylib)");
    SetWindowMinSize(980, 640);
    SetTargetFPS(120);

    Camera3D camera{};
    camera.position = {12.0f, 6.0f, 11.5f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 46.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    OrbitCameraState orbit{};
    std::vector<PlasmaParticle> particles = MakeParticles();

    while (!WindowShouldClose()) {
        const float time = static_cast<float>(GetTime());
        UpdateOrbitCameraDragOnly(&camera, &orbit);

        BeginDrawing();
        ClearBackground(Color{5, 8, 14, 255});
        BeginMode3D(camera);

        DrawGrid(28, 1.0f);
        DrawReactorShell();
        DrawTokamakFrame(time);
        DrawFieldLines(time);

        for (const PlasmaParticle& particle : particles) {
            const Vector3 p = PlasmaPoint(particle, time);
            const Color core = LerpColor(Color{255, 110, 60, 255}, Color{120, 235, 255, 255}, particle.heat);
            DrawSphere(p, particle.size * 1.8f, Fade(core, 0.10f));
            DrawSphere(p, particle.size, Fade(core, 0.80f));
        }

        DrawSphere({0.0f, 0.0f, 0.0f}, 2.0f, Fade(Color{82, 182, 255, 255}, 0.05f));
        DrawSphere({0.0f, 0.0f, 0.0f}, 1.2f, Fade(Color{255, 176, 86, 255}, 0.04f));

        EndMode3D();

        DrawRectangle(14, 14, 420, 92, Fade(BLACK, 0.28f));
        DrawText("Tokamak Confinement", 26, 24, 30, Color{234, 241, 252, 255});
        DrawText("Plasma spirals around the torus while magnetic field lines wrap the chamber.", 26, 58, 19, Color{170, 192, 223, 255});
        DrawText("Mouse orbit | wheel zoom", 26, 82, 18, Color{132, 220, 255, 255});
        DrawFPS(GetScreenWidth() - 96, 18);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
