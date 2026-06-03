#include "raylib.h"
#include "raymath.h"

#include <array>
#include <cmath>
#include <random>
#include <vector>

namespace {

constexpr int kScreenWidth = 1200;
constexpr int kScreenHeight = 800;
constexpr int kNucleons = 40;
constexpr int kElectrons = 12;
constexpr float kNucleusRadius = 0.25f;
constexpr std::array<float, 3> kShellRadii = {1.1f, 1.65f, 2.2f};

struct Nucleon {
    Vector3 pos;
    Color color;
};

struct Electron {
    float baseRadius;
    float angle0;
    float omega;
    float jitterAmp;
    float jitterPhase;
    float jitterSpeed;
    Vector3 basisU;
    Vector3 basisV;
};

float RandRange(std::mt19937& rng, float minV, float maxV) {
    std::uniform_real_distribution<float> dist(minV, maxV);
    return dist(rng);
}

Vector3 RandomUnit(std::mt19937& rng) {
    const float z = RandRange(rng, -1.0f, 1.0f);
    const float t = RandRange(rng, 0.0f, 2.0f * PI);
    const float r = std::sqrt(1.0f - z * z);
    return {r * std::cos(t), r * std::sin(t), z};
}

void BuildPlaneBasis(const Vector3& normal, Vector3* outU, Vector3* outV) {
    const Vector3 ref = (std::fabs(normal.y) < 0.9f) ? Vector3{0.0f, 1.0f, 0.0f} : Vector3{1.0f, 0.0f, 0.0f};
    *outU = Vector3Normalize(Vector3CrossProduct(normal, ref));
    *outV = Vector3Normalize(Vector3CrossProduct(normal, *outU));
}

Vector3 ElectronPosition(const Electron& e, float t) {
    const float radius = e.baseRadius + e.jitterAmp * std::sin(e.jitterPhase + t * e.jitterSpeed);
    const float angle = e.angle0 + e.omega * t;
    const float x = radius * std::cos(angle);
    const float y = radius * std::sin(angle);

    Vector3 p = Vector3Scale(e.basisU, x);
    p = Vector3Add(p, Vector3Scale(e.basisV, y));
    return p;
}

std::vector<Nucleon> MakeNucleus(std::mt19937& rng) {
    std::vector<Nucleon> out;
    out.reserve(kNucleons);

    for (int i = 0; i < kNucleons; ++i) {
        const float r = kNucleusRadius * std::cbrt(RandRange(rng, 0.0f, 1.0f));
        const Vector3 dir = RandomUnit(rng);
        const Vector3 p = Vector3Scale(dir, r);

        const bool proton = RandRange(rng, 0.0f, 1.0f) < 0.5f;
        const Color c = proton ? Color{255, 100, 80, 255} : Color{120, 170, 255, 255};
        out.push_back({p, c});
    }

    return out;
}

std::vector<Electron> MakeElectrons(std::mt19937& rng) {
    std::vector<Electron> out;
    out.reserve(kElectrons);

    std::uniform_int_distribution<int> shellDist(0, static_cast<int>(kShellRadii.size() - 1));

    for (int i = 0; i < kElectrons; ++i) {
        const float baseRadius = kShellRadii[shellDist(rng)];
        const float angle0 = RandRange(rng, 0.0f, 2.0f * PI);
        const float omegaMag = RandRange(rng, 0.6f, 1.6f);
        const float omega = (RandRange(rng, 0.0f, 1.0f) < 0.5f) ? omegaMag : -omegaMag;

        const float jitterAmp = RandRange(rng, 0.04f, 0.12f);
        const float jitterPhase = RandRange(rng, 0.0f, 2.0f * PI);
        const float jitterSpeed = RandRange(rng, 0.7f, 1.8f);

        const Vector3 normal = RandomUnit(rng);
        Vector3 u{};
        Vector3 v{};
        BuildPlaneBasis(normal, &u, &v);

        out.push_back({baseRadius, angle0, omega, jitterAmp, jitterPhase, jitterSpeed, u, v});
    }

    return out;
}

void DrawShellGuides() {
    constexpr int segments = 80;
    for (float radius : kShellRadii) {
        for (int i = 0; i < segments; ++i) {
            const float a0 = (2.0f * PI * i) / segments;
            const float a1 = (2.0f * PI * (i + 1)) / segments;

            DrawLine3D(
                {radius * std::cos(a0), radius * std::sin(a0), 0.0f},
                {radius * std::cos(a1), radius * std::sin(a1), 0.0f},
                Color{80, 170, 220, 80}
            );

            DrawLine3D(
                {0.0f, radius * std::cos(a0), radius * std::sin(a0)},
                {0.0f, radius * std::cos(a1), radius * std::sin(a1)},
                Color{80, 170, 220, 45}
            );
        }
    }
}

void DrawElectronTrail(const Electron& e, float tNow, float duration, int samples, Color color) {
    Vector3 prev = ElectronPosition(e, tNow - duration);
    for (int i = 1; i <= samples; ++i) {
        const float a = static_cast<float>(i) / samples;
        const float t = tNow - duration + duration * a;
        const Vector3 cur = ElectronPosition(e, t);
        DrawLine3D(prev, cur, color);
        prev = cur;
    }
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "3D Atom Visualization - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {5.0f, 3.2f, 5.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    std::mt19937 rng(7);
    const std::vector<Nucleon> nucleus = MakeNucleus(rng);
    const std::vector<Electron> electrons = MakeElectrons(rng);

    while (!WindowShouldClose()) {
        UpdateCamera(&camera, CAMERA_ORBITAL);
        const float t = static_cast<float>(GetTime());

        BeginDrawing();
        ClearBackground(Color{5, 8, 16, 255});

        BeginMode3D(camera);

        DrawShellGuides();
        DrawSphere({0.0f, 0.0f, 0.0f}, kNucleusRadius * 1.35f, Color{255, 190, 100, 40});

        for (const Nucleon& n : nucleus) {
            DrawSphere(n.pos, 0.06f, n.color);
        }

        for (const Electron& e : electrons) {
            DrawElectronTrail(e, t, 1.3f, 32, Color{130, 210, 255, 45});
            const Vector3 p = ElectronPosition(e, t);
            DrawSphere(p, 0.065f, Color{100, 230, 255, 255});
        }

        EndMode3D();

        DrawText("3D Atom Visualization (intuitive model)", 20, 20, 24, Color{230, 236, 245, 255});
        DrawText("Mouse drag: orbit camera | Mouse wheel: zoom | ESC: exit", 20, 54, 18, Color{170, 184, 204, 255});
        DrawFPS(20, 80);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
