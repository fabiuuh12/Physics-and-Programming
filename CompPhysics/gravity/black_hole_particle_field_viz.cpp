#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace {

constexpr int kScreenWidth = 1440;
constexpr int kScreenHeight = 900;
constexpr int kDiskParticleCount = 2400;
constexpr int kOrbitalStarCount = 180;
constexpr int kGridHalfCount = 15;
constexpr float kGridSpacing = 1.2f;
constexpr float kEventHorizonRadius = 1.0f;
constexpr float kPhotonRingRadius = 1.34f;
constexpr float kDiskInnerRadius = 1.8f;
constexpr float kDiskOuterRadius = 8.6f;

struct OrbitCameraState {
    float yaw = 0.52f;
    float pitch = -0.34f;
    float distance = 20.0f;
};

struct DiskParticle {
    float radius = 0.0f;
    float angle = 0.0f;
    float angularSpeed = 0.0f;
    float height = 0.0f;
    float size = 0.0f;
    float heat = 0.0f;
};

struct OrbitalStar {
    float orbitRadius = 0.0f;
    float phase = 0.0f;
    float speed = 0.0f;
    float incline = 0.0f;
    float yaw = 0.0f;
    float lift = 0.0f;
    float size = 0.0f;
    Color color{};
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

void UpdateOrbitCamera360(Camera3D* camera, OrbitCameraState* orbit) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        const Vector2 delta = GetMouseDelta();
        orbit->yaw -= delta.x * 0.0052f;
        orbit->pitch += delta.y * 0.0052f;
    }

    orbit->distance -= GetMouseWheelMove() * 1.15f;
    orbit->distance = std::clamp(orbit->distance, 5.0f, 52.0f);

    const float cp = std::cos(orbit->pitch);
    const Vector3 offset = {
        orbit->distance * cp * std::cos(orbit->yaw),
        orbit->distance * std::sin(orbit->pitch),
        orbit->distance * cp * std::sin(orbit->yaw),
    };

    camera->target = {0.0f, 0.0f, 0.0f};
    camera->position = Vector3Add(camera->target, offset);

    const Vector3 forward = Vector3Normalize(Vector3Subtract(camera->target, camera->position));
    Vector3 refUp = {0.0f, 1.0f, 0.0f};
    if (std::fabs(Vector3DotProduct(forward, refUp)) > 0.985f) {
        refUp = {0.0f, 0.0f, 1.0f};
    }
    const Vector3 right = Vector3Normalize(Vector3CrossProduct(refUp, forward));
    camera->up = Vector3Normalize(Vector3CrossProduct(forward, right));
}

std::vector<DiskParticle> MakeDiskParticles() {
    std::mt19937 rng(1337);
    std::vector<DiskParticle> particles;
    particles.reserve(kDiskParticleCount);

    for (int i = 0; i < kDiskParticleCount; ++i) {
        const float radius = RandRange(rng, kDiskInnerRadius, kDiskOuterRadius);
        const float band = 0.5f + 0.5f * std::sin(radius * 2.8f + RandRange(rng, 0.0f, 6.28f));
        particles.push_back({
            radius,
            RandRange(rng, 0.0f, 2.0f * PI),
            0.22f / std::pow(radius, 0.92f) + RandRange(rng, -0.0025f, 0.0025f),
            RandRange(rng, -0.10f, 0.10f) * (0.3f + radius / kDiskOuterRadius),
            RandRange(rng, 0.024f, 0.085f),
            0.30f + band * 0.70f,
        });
    }

    return particles;
}

std::vector<OrbitalStar> MakeOrbitalStars() {
    std::mt19937 rng(4242);
    std::vector<OrbitalStar> stars;
    stars.reserve(kOrbitalStarCount);

    for (int i = 0; i < kOrbitalStarCount; ++i) {
        const float orbitRadius = RandRange(rng, 7.0f, 16.0f);
        const float temp = RandRange(rng, 0.0f, 1.0f);
        stars.push_back({
            orbitRadius,
            RandRange(rng, 0.0f, 2.0f * PI),
            0.045f / std::pow(orbitRadius, 0.70f),
            RandRange(rng, -1.25f, 1.25f),
            RandRange(rng, 0.0f, 2.0f * PI),
            RandRange(rng, -1.6f, 1.6f),
            RandRange(rng, 0.07f, 0.18f),
            LerpColor(Color{150, 198, 255, 255}, Color{255, 233, 176, 255}, temp),
        });
    }

    return stars;
}

Vector3 DiskParticlePosition(const DiskParticle& p, float time) {
    const float a = p.angle + time * p.angularSpeed;
    const float x = std::cos(a) * p.radius;
    const float z = std::sin(a) * p.radius;
    const float y = p.height + 0.06f * std::sin(a * 3.0f + time * 1.5f);
    Vector3 pos = {x, y, z};
    pos = Vector3RotateByAxisAngle(pos, {1.0f, 0.0f, 0.0f}, 0.92f);
    pos = Vector3RotateByAxisAngle(pos, {0.0f, 0.0f, 1.0f}, -0.36f);
    return pos;
}

Vector3 OrbitalStarPosition(const OrbitalStar& s, float time) {
    const float a = s.phase + time * s.speed;
    Vector3 pos = {std::cos(a) * s.orbitRadius, s.lift, std::sin(a) * s.orbitRadius};
    pos = Vector3RotateByAxisAngle(pos, {1.0f, 0.0f, 0.0f}, s.incline);
    pos = Vector3RotateByAxisAngle(pos, {0.0f, 1.0f, 0.0f}, s.yaw);
    return pos;
}

Vector3 WarpGridPoint(Vector3 p, float time) {
    const float r = std::sqrt(p.x * p.x + p.z * p.z);
    const float clampedR = std::max(r, kEventHorizonRadius + 0.2f);
    const float sink = 3.0f / (clampedR * clampedR + 0.8f);
    const float ripple = 0.14f * std::sin(time * 1.5f + clampedR * 2.4f);
    p.y -= sink + ripple / (1.0f + clampedR * 0.22f);

    if (r > 0.0001f) {
        const float radialPull = 0.14f / (clampedR + 0.5f);
        p.x -= (p.x / r) * radialPull;
        p.z -= (p.z / r) * radialPull;
    }

    return p;
}

void DrawWarpedGrid(float time) {
    for (int i = -kGridHalfCount; i <= kGridHalfCount; ++i) {
        for (int j = -kGridHalfCount; j < kGridHalfCount; ++j) {
            Vector3 a = WarpGridPoint({i * kGridSpacing, -2.0f, j * kGridSpacing}, time);
            Vector3 b = WarpGridPoint({i * kGridSpacing, -2.0f, (j + 1) * kGridSpacing}, time);
            const float radius = 0.5f * (std::sqrt(a.x * a.x + a.z * a.z) + std::sqrt(b.x * b.x + b.z * b.z));
            const float fade = std::clamp(1.0f - radius / 18.0f, 0.10f, 0.90f);
            DrawLine3D(a, b, Fade(Color{92, 126, 212, 255}, 0.32f * fade));
        }
    }

    for (int j = -kGridHalfCount; j <= kGridHalfCount; ++j) {
        for (int i = -kGridHalfCount; i < kGridHalfCount; ++i) {
            Vector3 a = WarpGridPoint({i * kGridSpacing, -2.0f, j * kGridSpacing}, time);
            Vector3 b = WarpGridPoint({(i + 1) * kGridSpacing, -2.0f, j * kGridSpacing}, time);
            const float radius = 0.5f * (std::sqrt(a.x * a.x + a.z * a.z) + std::sqrt(b.x * b.x + b.z * b.z));
            const float fade = std::clamp(1.0f - radius / 18.0f, 0.10f, 0.90f);
            DrawLine3D(a, b, Fade(Color{92, 126, 212, 255}, 0.32f * fade));
        }
    }
}

void DrawBackdropStars(const Camera3D& camera) {
    std::mt19937 rng(9001);
    for (int i = 0; i < 320; ++i) {
        const float theta = RandRange(rng, 0.0f, 2.0f * PI);
        const float phi = RandRange(rng, -0.55f * PI, 0.55f * PI);
        const float radius = RandRange(rng, 34.0f, 50.0f);
        Vector3 p = {
            radius * std::cos(phi) * std::cos(theta),
            radius * std::sin(phi),
            radius * std::cos(phi) * std::sin(theta),
        };
        p = Vector3Add(p, Vector3Scale(Vector3Normalize(Vector3Subtract(camera.position, camera.target)), 8.0f));
        DrawSphere(p, RandRange(rng, 0.015f, 0.045f), Fade(RAYWHITE, RandRange(rng, 0.35f, 0.95f)));
    }
}

}  // namespace

int main() {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(kScreenWidth, kScreenHeight, "Black Hole Particle Field - C++ (raylib)");
    SetWindowMinSize(980, 620);
    SetTargetFPS(120);

    Camera3D camera{};
    camera.position = {16.0f, 6.5f, 12.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 48.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    OrbitCameraState orbit{};
    std::vector<DiskParticle> disk = MakeDiskParticles();
    std::vector<OrbitalStar> stars = MakeOrbitalStars();

    while (!WindowShouldClose()) {
        const float time = static_cast<float>(GetTime());
        UpdateOrbitCamera360(&camera, &orbit);

        BeginDrawing();
        ClearBackground(Color{4, 6, 12, 255});
        BeginMode3D(camera);

        DrawBackdropStars(camera);
        DrawWarpedGrid(time);

        for (const OrbitalStar& star : stars) {
            const Vector3 p = OrbitalStarPosition(star, time);
            DrawSphere(p, star.size * 1.9f, Fade(star.color, 0.08f));
            DrawSphere(p, star.size, star.color);
        }

        for (const DiskParticle& particle : disk) {
            const Vector3 p = DiskParticlePosition(particle, time);
            const float heat = particle.heat;
            const Color c = LerpColor(Color{255, 116, 42, 255}, Color{255, 239, 170, 255}, heat);
            DrawSphere(p, particle.size * 1.65f, Fade(c, 0.12f + 0.10f * heat));
            DrawSphere(p, particle.size, Fade(c, 0.75f));
        }

        DrawSphere({0.0f, 0.0f, 0.0f}, kPhotonRingRadius * 1.55f, Fade(Color{98, 144, 255, 255}, 0.05f));
        DrawSphere({0.0f, 0.0f, 0.0f}, kPhotonRingRadius, Fade(Color{255, 194, 120, 255}, 0.20f));
        DrawSphere({0.0f, 0.0f, 0.0f}, kEventHorizonRadius, BLACK);
        DrawSphereWires({0.0f, 0.0f, 0.0f}, kPhotonRingRadius, 18, 18, Fade(Color{180, 214, 255, 255}, 0.18f));

        EndMode3D();

        DrawRectangle(12, 12, 350, 86, Fade(BLACK, 0.28f));
        DrawText("Black Hole Particle Field", 24, 24, 28, Color{234, 240, 252, 255});
        DrawText("Mouse drag: 360 orbit   Wheel: zoom", 24, 58, 20, Color{162, 184, 220, 255});
        DrawFPS(GetScreenWidth() - 98, 18);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
