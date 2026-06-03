#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace {

constexpr int kScreenWidth = 1440;
constexpr int kScreenHeight = 900;
constexpr float kSystemExtent = 34.0f;
constexpr int kBackgroundStarCount = 180;
constexpr int kWindParticleCount = 820;

struct OrbitCameraState {
    float yaw = 0.72f;
    float pitch = 0.28f;
    float distance = 34.0f;
};

struct Planet {
    const char* name = "";
    float orbitRadius = 0.0f;
    float orbitOmega = 0.0f;
    float radius = 0.0f;
    float phase = 0.0f;
    float incline = 0.0f;
    float magnetosphere = 0.0f;
    bool strongTail = false;
    Color color{};
    Vector3 pos{};
};

struct WindParticle {
    Vector3 dir{};
    Vector3 pos{};
    float speed = 0.0f;
    float size = 0.0f;
    float band = 0.0f;
};

struct BackdropStar {
    Vector3 pos{};
    float size = 0.0f;
    float alpha = 0.0f;
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
        orbit->yaw -= delta.x * 0.0038f;
        orbit->pitch += delta.y * 0.0038f;
        orbit->pitch = std::clamp(orbit->pitch, -1.24f, 1.24f);
    }

    orbit->distance -= GetMouseWheelMove() * 1.1f;
    orbit->distance = std::clamp(orbit->distance, 12.0f, 68.0f);

    const float cp = std::cos(orbit->pitch);
    camera->target = {0.0f, 0.0f, 0.0f};
    camera->position = Vector3Add(camera->target, {
        orbit->distance * cp * std::cos(orbit->yaw),
        orbit->distance * std::sin(orbit->pitch),
        orbit->distance * cp * std::sin(orbit->yaw),
    });
}

std::vector<Planet> MakePlanets() {
    return {
        {"Mercury", 4.0f, 1.65f, 0.30f, 0.1f, 0.02f, 0.18f, false, Color{208, 198, 186, 255}},
        {"Venus", 6.1f, 1.20f, 0.42f, 1.3f, -0.03f, 0.26f, false, Color{238, 196, 118, 255}},
        {"Earth", 8.6f, 0.92f, 0.46f, 2.4f, 0.06f, 0.78f, true, Color{92, 174, 255, 255}},
        {"Mars", 11.2f, 0.72f, 0.36f, 0.9f, -0.02f, 0.22f, false, Color{242, 112, 78, 255}},
        {"Jupiter", 17.0f, 0.36f, 1.02f, 2.9f, 0.04f, 1.15f, true, Color{230, 184, 138, 255}},
    };
}

void UpdatePlanets(std::vector<Planet>* planets, float time) {
    for (Planet& planet : *planets) {
        const float a = planet.phase + time * planet.orbitOmega;
        Vector3 p = {
            std::cos(a) * planet.orbitRadius,
            std::sin(a * 1.6f) * planet.incline,
            std::sin(a) * planet.orbitRadius,
        };
        planet.pos = p;
    }
}

std::vector<BackdropStar> MakeBackdropStars() {
    std::mt19937 rng(7123);
    std::vector<BackdropStar> stars;
    stars.reserve(kBackgroundStarCount);

    for (int i = 0; i < kBackgroundStarCount; ++i) {
        const float theta = RandRange(rng, 0.0f, 2.0f * PI);
        const float phi = RandRange(rng, -0.46f * PI, 0.46f * PI);
        const float radius = RandRange(rng, 42.0f, 58.0f);
        stars.push_back({
            {
                radius * std::cos(phi) * std::cos(theta),
                radius * std::sin(phi),
                radius * std::cos(phi) * std::sin(theta),
            },
            RandRange(rng, 0.03f, 0.11f),
            RandRange(rng, 0.18f, 0.84f),
        });
    }

    return stars;
}

std::vector<WindParticle> MakeWindParticles() {
    std::mt19937 rng(9182);
    std::vector<WindParticle> particles;
    particles.reserve(kWindParticleCount);

    for (int i = 0; i < kWindParticleCount; ++i) {
        const int shell = i % 18;
        const float theta = RandRange(rng, 0.0f, 2.0f * PI);
        const float elev = RandRange(rng, -0.42f, 0.42f);
        const Vector3 dir = Vector3Normalize({
            std::cos(theta) * std::cos(elev),
            std::sin(elev) * 0.7f,
            std::sin(theta) * std::cos(elev),
        });
        const float radius = RandRange(rng, 0.0f, kSystemExtent) + shell * 1.2f;
        particles.push_back({
            dir,
            Vector3Scale(dir, std::fmod(radius, kSystemExtent)),
            RandRange(rng, 4.6f, 9.6f),
            RandRange(rng, 0.028f, 0.072f),
            static_cast<float>(shell),
        });
    }

    return particles;
}

void RespawnWindParticle(WindParticle* particle, std::mt19937* rng) {
    const float theta = RandRange(*rng, 0.0f, 2.0f * PI);
    const float elev = RandRange(*rng, -0.42f, 0.42f);
    particle->dir = Vector3Normalize({
        std::cos(theta) * std::cos(elev),
        std::sin(elev) * 0.7f,
        std::sin(theta) * std::cos(elev),
    });
    const float radius = RandRange(*rng, 0.2f, 2.4f) + particle->band * 0.26f;
    particle->pos = Vector3Scale(particle->dir, radius);
    particle->speed = RandRange(*rng, 4.6f, 9.6f);
}

void UpdateWindParticles(std::vector<WindParticle>* wind, const std::vector<Planet>& planets, float dt) {
    static std::mt19937 rng(9917);

    for (WindParticle& particle : *wind) {
        particle.pos = Vector3Add(particle.pos, Vector3Scale(particle.dir, particle.speed * dt));

        bool respawn = Vector3Length(particle.pos) > kSystemExtent;

        for (const Planet& planet : planets) {
            const Vector3 rel = Vector3Subtract(particle.pos, planet.pos);
            const float dist = Vector3Length(rel);
            const float influence = planet.radius + planet.magnetosphere * 2.8f;
            const float shield = planet.radius + planet.magnetosphere * 1.45f;

            if (dist < shield * 0.72f) {
                respawn = true;
                break;
            }

            if (dist < influence && dist > 0.0001f) {
                const Vector3 relDir = Vector3Scale(rel, 1.0f / dist);
                const Vector3 solarDir = Vector3Normalize(particle.pos);
                Vector3 tangent = Vector3CrossProduct(relDir, Vector3CrossProduct(solarDir, relDir));
                if (Vector3Length(tangent) < 1.0e-4f) tangent = {0.0f, 1.0f, 0.0f};
                tangent = Vector3Normalize(tangent);

                const float strength = (1.0f - dist / influence) * planet.magnetosphere;
                particle.pos = Vector3Add(particle.pos, Vector3Scale(tangent, strength * dt * 14.0f));

                if (planet.strongTail) {
                    const Vector3 awayFromSun = Vector3Normalize(planet.pos);
                    particle.pos = Vector3Add(particle.pos, Vector3Scale(awayFromSun, strength * dt * 11.0f));
                }
            }
        }

        if (respawn) {
            RespawnWindParticle(&particle, &rng);
        }
    }
}

Color WindParticleColor(const Vector3& pos, const std::vector<Planet>& planets) {
    Color c = Color{162, 220, 255, 255};

    for (const Planet& planet : planets) {
        const float dist = Vector3Distance(pos, planet.pos);
        const float influence = planet.radius + planet.magnetosphere * 2.0f;
        if (dist < influence) {
            const float t = 1.0f - dist / influence;
            c = LerpColor(c, planet.strongTail ? Color{92, 255, 190, 255} : planet.color, t * 0.85f);
        }
    }

    return c;
}

void DrawOrbitRing(float radius, Color color) {
    constexpr int kSegments = 160;
    for (int i = 0; i < kSegments; ++i) {
        const float a0 = (2.0f * PI * i) / kSegments;
        const float a1 = (2.0f * PI * (i + 1)) / kSegments;
        DrawLine3D(
            {radius * std::cos(a0), 0.0f, radius * std::sin(a0)},
            {radius * std::cos(a1), 0.0f, radius * std::sin(a1)},
            color
        );
    }
}

void DrawPlanetInteraction(const Planet& planet, float time) {
    const float tailLen = planet.strongTail ? 4.8f : 2.1f;
    const Vector3 awayFromSun = Vector3Normalize(planet.pos);
    const Vector3 tailEnd = Vector3Add(planet.pos, Vector3Scale(awayFromSun, tailLen));

    for (int i = 0; i < 4; ++i) {
        const float ring = planet.radius + planet.magnetosphere * (0.9f + i * 0.55f);
        DrawSphereWires(planet.pos, ring, 12, 12, Fade(Color{122, 206, 255, 255}, 0.10f));
    }

    if (planet.magnetosphere > 0.4f) {
        for (int i = 0; i < 3; ++i) {
            const float wobble = 0.15f * std::sin(time * 2.0f + i * 0.9f);
            DrawLine3D(
                Vector3Add(planet.pos, {0.0f, wobble, 0.0f}),
                Vector3Add(tailEnd, {0.0f, wobble * 1.8f, 0.0f}),
                Fade(Color{90, 255, 198, 255}, 0.20f)
            );
        }
    }
}

void DrawBackdropStars(const std::vector<BackdropStar>& stars) {
    for (size_t i = 0; i < stars.size(); ++i) {
        const BackdropStar& star = stars[i];
        const Color c = Fade(RAYWHITE, star.alpha);
        if (i % 11 == 0) {
            DrawSphere(star.pos, star.size * 0.75f, c);
        } else {
            DrawPoint3D(star.pos, c);
        }
    }
}

}  // namespace

int main() {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(kScreenWidth, kScreenHeight, "Solar System Solar Wind 3D - C++ (raylib)");
    SetWindowMinSize(1024, 660);
    SetTargetFPS(120);

    Camera3D camera{};
    camera.position = {26.0f, 10.0f, 22.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 46.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    OrbitCameraState orbit{};
    std::vector<Planet> planets = MakePlanets();
    const std::vector<BackdropStar> backdrop = MakeBackdropStars();
    std::vector<WindParticle> wind = MakeWindParticles();
    float simSpeed = 1.0f;
    float simTime = 0.0f;
    bool paused = false;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) simSpeed = std::max(0.1f, simSpeed - 0.2f);
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) simSpeed = std::min(6.0f, simSpeed + 0.2f);
        if (IsKeyPressed(KEY_R)) {
            wind = MakeWindParticles();
            simSpeed = 1.0f;
            simTime = 0.0f;
            paused = false;
        }

        const float dt = paused ? 0.0f : GetFrameTime() * simSpeed;
        simTime += dt;
        UpdateOrbitCameraDragOnly(&camera, &orbit);
        UpdatePlanets(&planets, simTime);
        UpdateWindParticles(&wind, planets, dt);

        BeginDrawing();
        ClearBackground(Color{4, 6, 12, 255});
        BeginMode3D(camera);

        DrawBackdropStars(backdrop);

        for (const Planet& planet : planets) {
            DrawOrbitRing(planet.orbitRadius, Fade(Color{90, 112, 150, 255}, 0.18f));
        }

        DrawSphere({0.0f, 0.0f, 0.0f}, 2.05f, Fade(Color{255, 184, 82, 255}, 0.10f));
        DrawSphere({0.0f, 0.0f, 0.0f}, 1.65f, Color{255, 210, 98, 255});
        DrawSphere({0.0f, 0.0f, 0.0f}, 1.28f, Color{255, 236, 164, 255});
        DrawSphereWires({0.0f, 0.0f, 0.0f}, 1.66f, 16, 16, Fade(Color{255, 248, 210, 255}, 0.18f));

        for (const WindParticle& particle : wind) {
            const Vector3 p = particle.pos;
            const Color c = WindParticleColor(p, planets);
            const Vector3 tail = Vector3Subtract(p, Vector3Scale(Vector3Normalize(p), 0.22f + particle.size * 1.8f));
            DrawLine3D(tail, p, Fade(c, 0.22f));
            DrawPoint3D(p, Fade(c, 0.92f));
        }

        for (const Planet& planet : planets) {
            DrawPlanetInteraction(planet, simTime);
            DrawSphere(planet.pos, planet.radius, planet.color);
            DrawSphereWires(planet.pos, planet.radius * 1.02f, 12, 12, Fade(RAYWHITE, 0.16f));
        }

        EndMode3D();

        DrawRectangle(14, 14, 560, 92, Fade(BLACK, 0.28f));
        DrawText("Solar System + Solar Wind", 26, 24, 30, Color{234, 241, 252, 255});
        DrawText("The Sun emits pulsed particle bands while planets bend, shield, or trail the flow.", 26, 58, 19, Color{170, 192, 223, 255});
        DrawText(TextFormat("Mouse orbit | wheel zoom | - / + speed | P pause | R reset | speed %.1fx%s", simSpeed, paused ? " [PAUSED]" : ""), 26, 82, 18, Color{132, 220, 255, 255});
        DrawFPS(GetScreenWidth() - 96, 18);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
