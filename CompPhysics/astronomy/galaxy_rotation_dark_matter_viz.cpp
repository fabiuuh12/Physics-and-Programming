#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kScreenWidth = 1520;
constexpr int kScreenHeight = 920;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kGalaxyRadius = 20.0f;
constexpr int kStarCount = 1150;
constexpr int kTracerCount = 18;

enum class GravityModel {
    kBaryonsOnly = 0,
    kDarkMatter = 1,
    kMond = 2,
};

struct GalaxyParams {
    float baryonMass = 980.0f;
    float diskScale = 3.6f;
    float bulgeScale = 1.15f;
    float haloStrength = 1.10f;
    float haloCore = 3.4f;
    float mondA0 = 5.6f;
    float timeScale = 0.90f;
    float probeRadius = 11.0f;
};

struct StarParticle {
    float radius;
    float theta;
    float initialTheta;
    float zAmplitude;
    float zPhase;
    float size;
    float brightness;
    float temperature;
    int arm;
};

struct BackgroundStar {
    Vector2 pos;
    float radius;
    Color color;
};

struct TracerState {
    float theta;
};

Color Brighten(Color c, float amount) {
    amount = std::clamp(amount, 0.0f, 1.0f);
    return Color{
        static_cast<unsigned char>(c.r + (255 - c.r) * amount),
        static_cast<unsigned char>(c.g + (255 - c.g) * amount),
        static_cast<unsigned char>(c.b + (255 - c.b) * amount),
        255
    };
}

Color LerpColor(Color a, Color b, float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return Color{
        static_cast<unsigned char>(a.r + (b.r - a.r) * t),
        static_cast<unsigned char>(a.g + (b.g - a.g) * t),
        static_cast<unsigned char>(a.b + (b.b - a.b) * t),
        static_cast<unsigned char>(a.a + (b.a - a.a) * t)
    };
}

float ClampRadius(float r) {
    return std::max(0.22f, r);
}

float DiskMassProfile(float r, const GalaxyParams& params) {
    float x = r / params.diskScale;
    return 0.72f * params.baryonMass * (1.0f - std::exp(-x) * (1.0f + x));
}

float BulgeMassProfile(float r, const GalaxyParams& params) {
    float x3 = r * r * r;
    float s3 = params.bulgeScale * params.bulgeScale * params.bulgeScale;
    return 0.28f * params.baryonMass * x3 / (x3 + s3);
}

float BaryonicEnclosedMass(float r, const GalaxyParams& params) {
    r = ClampRadius(r);
    return DiskMassProfile(r, params) + BulgeMassProfile(r, params);
}

float HaloEnclosedMass(float r, const GalaxyParams& params) {
    r = ClampRadius(r);
    float rr = r * r;
    float core2 = params.haloCore * params.haloCore;
    return params.haloStrength * 130.0f * r * rr / (rr + core2);
}

float NewtonianAcceleration(float enclosedMass, float r) {
    r = ClampRadius(r);
    return enclosedMass / (r * r);
}

float MondAcceleration(float gNewton, float a0) {
    return 0.5f * (gNewton + std::sqrt(gNewton * gNewton + 4.0f * a0 * gNewton));
}

float RotationSpeed(GravityModel model, float r, const GalaxyParams& params) {
    r = ClampRadius(r);
    float baryonicMass = BaryonicEnclosedMass(r, params);
    float gNewton = NewtonianAcceleration(baryonicMass, r);

    if (model == GravityModel::kBaryonsOnly) {
        return std::sqrt(std::max(0.001f, gNewton * r));
    }
    if (model == GravityModel::kDarkMatter) {
        float totalMass = baryonicMass + HaloEnclosedMass(r, params);
        return std::sqrt(std::max(0.001f, NewtonianAcceleration(totalMass, r) * r));
    }
    return std::sqrt(std::max(0.001f, MondAcceleration(gNewton, params.mondA0) * r));
}

Color ModelColor(GravityModel model) {
    switch (model) {
        case GravityModel::kBaryonsOnly: return Color{255, 110, 104, 255};
        case GravityModel::kDarkMatter: return Color{100, 214, 255, 255};
        case GravityModel::kMond: return Color{255, 204, 108, 255};
    }
    return WHITE;
}

const char* ModelName(GravityModel model) {
    switch (model) {
        case GravityModel::kBaryonsOnly: return "Visible Matter Only";
        case GravityModel::kDarkMatter: return "Dark Matter Halo";
        case GravityModel::kMond: return "MOND";
    }
    return "";
}

void UpdateOrbitCamera(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 delta = GetMouseDelta();
        *yaw -= delta.x * 0.0032f;
        *pitch += delta.y * 0.0030f;
        *pitch = std::clamp(*pitch, -1.38f, 1.38f);
    }

    *distance -= GetMouseWheelMove() * 1.1f;
    *distance = std::clamp(*distance, 10.0f, 65.0f);

    float cp = std::cos(*pitch);
    Vector3 offset = {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}

std::vector<BackgroundStar> BuildBackgroundStars() {
    std::vector<BackgroundStar> stars;
    stars.reserve(260);

    std::mt19937 rng(8);
    std::uniform_real_distribution<float> xDist(0.0f, static_cast<float>(kScreenWidth));
    std::uniform_real_distribution<float> yDist(0.0f, static_cast<float>(kScreenHeight));
    std::uniform_real_distribution<float> sDist(0.6f, 2.3f);
    std::uniform_real_distribution<float> mixDist(0.0f, 1.0f);

    for (int i = 0; i < 260; ++i) {
        float mix = mixDist(rng);
        Color cool = Color{176, 204, 255, 255};
        Color warm = Color{255, 221, 182, 255};
        stars.push_back({{xDist(rng), yDist(rng)}, sDist(rng), LerpColor(cool, warm, mix)});
    }
    return stars;
}

std::vector<StarParticle> BuildGalaxyStars() {
    std::vector<StarParticle> stars;
    stars.reserve(kStarCount);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    std::normal_distribution<float> armNoise(0.0f, 0.28f);
    std::normal_distribution<float> heightNoise(0.0f, 1.0f);
    constexpr int armCount = 4;

    for (int i = 0; i < kStarCount; ++i) {
        float u = unit(rng);
        float radius = 0.55f + kGalaxyRadius * std::pow(u, 0.62f);
        int arm = i % armCount;
        float armBase = (2.0f * kPi * static_cast<float>(arm)) / static_cast<float>(armCount);
        float armTwist = radius * 0.34f;
        float theta = armBase + armTwist + armNoise(rng) + 0.35f * std::sin(radius * 0.45f);
        float brightness = 0.38f + 0.62f * std::pow(1.0f - std::min(radius / kGalaxyRadius, 1.0f), 0.35f);
        float temperature = std::clamp(0.24f + 0.66f * (radius / kGalaxyRadius) + 0.08f * armNoise(rng), 0.0f, 1.0f);
        float size = 0.026f + 0.070f * brightness;
        float zAmplitude = 0.03f + 0.10f * std::abs(heightNoise(rng)) + 0.10f * (radius / kGalaxyRadius);
        float zPhase = 2.0f * kPi * unit(rng);

        stars.push_back({radius, theta, theta, zAmplitude, zPhase, size, brightness, temperature, arm});
    }

    return stars;
}

std::array<TracerState, kTracerCount> BuildTracerRing(float seedPhase) {
    std::array<TracerState, kTracerCount> tracers{};
    for (int i = 0; i < kTracerCount; ++i) {
        tracers[i].theta = seedPhase + 2.0f * kPi * static_cast<float>(i) / static_cast<float>(kTracerCount);
    }
    return tracers;
}

Vector3 GalaxyPositionFromStar(const StarParticle& star, float timeSeconds) {
    float wobble = 0.45f * std::sin(timeSeconds * 0.7f + star.zPhase + star.radius * 0.12f);
    float y = star.zAmplitude * wobble;
    return {star.radius * std::cos(star.theta), y, star.radius * std::sin(star.theta)};
}

Color StarColor(const StarParticle& star, float timeSeconds) {
    Color warm = Color{255, 208, 154, 255};
    Color cool = Color{156, 206, 255, 255};
    float twinkle = 0.5f + 0.5f * std::sin(timeSeconds * (1.2f + 0.8f * star.brightness) + star.zPhase);
    Color base = LerpColor(warm, cool, star.temperature);
    return Brighten(base, 0.16f + 0.26f * twinkle * star.brightness);
}

void ResetGalaxy(std::vector<StarParticle>* stars) {
    for (StarParticle& star : *stars) {
        star.theta = star.initialTheta;
    }
}

void UpdateGalaxy(
    std::vector<StarParticle>* stars,
    std::array<std::array<TracerState, kTracerCount>, 3>* tracers,
    GravityModel activeModel,
    const GalaxyParams& params,
    float dt
) {
    for (StarParticle& star : *stars) {
        float speed = RotationSpeed(activeModel, star.radius, params);
        float angular = speed / ClampRadius(star.radius);
        angular *= 0.90f + 0.18f * star.brightness + 0.05f * std::sin(star.zPhase + star.radius * 0.11f);
        star.theta += angular * dt;
    }

    for (int modelIdx = 0; modelIdx < 3; ++modelIdx) {
        GravityModel model = static_cast<GravityModel>(modelIdx);
        float speed = RotationSpeed(model, params.probeRadius, params);
        float angular = speed / ClampRadius(params.probeRadius);
        for (TracerState& tracer : (*tracers)[modelIdx]) {
            tracer.theta += angular * dt;
        }
    }
}

void DrawBackground(const std::vector<BackgroundStar>& stars, float timeSeconds) {
    DrawRectangleGradientV(0, 0, kScreenWidth, kScreenHeight, Color{7, 10, 24, 255}, Color{1, 3, 10, 255});
    DrawCircleGradient(210, 140, 280.0f, Fade(Color{38, 68, 122, 255}, 0.22f), BLANK);
    DrawCircleGradient(kScreenWidth - 220, 180, 260.0f, Fade(Color{88, 48, 110, 255}, 0.18f), BLANK);
    DrawCircleGradient(kScreenWidth / 2, kScreenHeight - 60, 340.0f, Fade(Color{28, 48, 86, 255}, 0.11f), BLANK);

    for (size_t i = 0; i < stars.size(); ++i) {
        float pulse = 0.55f + 0.45f * std::sin(timeSeconds * 0.6f + static_cast<float>(i) * 0.37f);
        DrawCircleV(stars[i].pos, stars[i].radius + 0.6f * pulse, Fade(stars[i].color, 0.10f));
        DrawCircleV(stars[i].pos, stars[i].radius, Fade(stars[i].color, 0.78f));
    }
}

void DrawGalaxyPlane(const GalaxyParams& params, float timeSeconds) {
    DrawCylinder({0.0f, 0.0f, 0.0f}, kGalaxyRadius * 0.97f, kGalaxyRadius * 0.97f, 0.05f, 48, Color{22, 34, 56, 230});
    DrawCylinder({0.0f, 0.0f, 0.0f}, kGalaxyRadius * 0.86f, kGalaxyRadius * 0.86f, 0.032f, 48, Color{36, 58, 88, 205});

    for (int i = 0; i < 6; ++i) {
        float ringRadius = 2.7f + static_cast<float>(i) * 2.85f;
        float pulse = 0.20f + 0.08f * std::sin(timeSeconds * 0.4f + i);
        DrawCircle3D({0.0f, 0.0f, 0.0f}, ringRadius, {1.0f, 0.0f, 0.0f}, 90.0f, Fade(Color{82, 112, 156, 255}, pulse));
    }

    DrawSphere({0.0f, 0.0f, 0.0f}, 1.55f, Color{255, 210, 146, 255});
    DrawSphere({0.0f, 0.0f, 0.0f}, 2.30f, Fade(Color{255, 184, 100, 255}, 0.20f));
    DrawSphere({0.0f, 0.0f, 0.0f}, 3.30f, Fade(Color{255, 214, 154, 255}, 0.10f));
    DrawSphereWires({0.0f, 0.0f, 0.0f}, 3.40f, 16, 16, Fade(Color{255, 233, 190, 255}, 0.14f));

    if (params.haloStrength > 0.01f) {
        float haloRadius = 8.0f + 7.0f * params.haloStrength;
        DrawSphereWires({0.0f, 0.0f, 0.0f}, haloRadius, 26, 26, Fade(Color{90, 196, 255, 255}, 0.13f));
        DrawSphereWires({0.0f, 0.0f, 0.0f}, haloRadius * 1.28f, 22, 22, Fade(Color{90, 196, 255, 255}, 0.07f));
    }
}

void DrawGalaxyStars(const std::vector<StarParticle>& stars, float timeSeconds) {
    for (const StarParticle& star : stars) {
        Vector3 pos = GalaxyPositionFromStar(star, timeSeconds);
        Color color = StarColor(star, timeSeconds);
        float glow = star.size * (1.4f + 0.9f * star.brightness);
        DrawSphere(pos, glow, Fade(color, 0.10f));
        DrawSphere(pos, star.size, color);
    }
}

void DrawProbeRing(const GalaxyParams& params, const std::array<std::array<TracerState, kTracerCount>, 3>& tracers, GravityModel activeModel) {
    DrawCircle3D({0.0f, 0.0f, 0.0f}, params.probeRadius, {1.0f, 0.0f, 0.0f}, 90.0f, Fade(Color{235, 240, 250, 255}, 0.26f));

    for (int modelIdx = 0; modelIdx < 3; ++modelIdx) {
        GravityModel model = static_cast<GravityModel>(modelIdx);
        Color color = ModelColor(model);
        float yOffset = (modelIdx - 1) * 0.14f;
        for (const TracerState& tracer : tracers[modelIdx]) {
            Vector3 pos = {
                params.probeRadius * std::cos(tracer.theta),
                yOffset,
                params.probeRadius * std::sin(tracer.theta)
            };
            float radius = (model == activeModel) ? 0.16f : 0.11f;
            DrawSphere(pos, radius * 1.8f, Fade(color, model == activeModel ? 0.16f : 0.08f));
            DrawSphere(pos, radius, Brighten(color, model == activeModel ? 0.18f : 0.0f));
        }
    }
}

void DrawCurvePanel(Rectangle panel, const GalaxyParams& params, GravityModel activeModel, float timeSeconds) {
    DrawRectangleRounded(panel, 0.06f, 16, Fade(Color{8, 15, 30, 255}, 0.88f));
    DrawRectangleRoundedLinesEx(panel, 0.06f, 16, 1.5f, Fade(Color{118, 146, 186, 255}, 0.38f));

    Rectangle plot = {panel.x + 56.0f, panel.y + 54.0f, panel.width - 86.0f, panel.height - 112.0f};
    DrawRectangleRounded(plot, 0.03f, 8, Fade(Color{15, 26, 46, 255}, 0.92f));

    const float maxRadius = kGalaxyRadius;
    float maxSpeed = 0.0f;
    for (int modelIdx = 0; modelIdx < 3; ++modelIdx) {
        GravityModel model = static_cast<GravityModel>(modelIdx);
        for (int i = 1; i <= 160; ++i) {
            float r = maxRadius * static_cast<float>(i) / 160.0f;
            maxSpeed = std::max(maxSpeed, RotationSpeed(model, r, params));
        }
    }
    maxSpeed *= 1.08f;

    for (int i = 0; i <= 5; ++i) {
        float y = plot.y + plot.height * static_cast<float>(i) / 5.0f;
        DrawLineEx({plot.x, y}, {plot.x + plot.width, y}, 1.0f, Fade(Color{110, 136, 170, 255}, 0.16f));
    }
    for (int i = 0; i <= 5; ++i) {
        float x = plot.x + plot.width * static_cast<float>(i) / 5.0f;
        DrawLineEx({x, plot.y}, {x, plot.y + plot.height}, 1.0f, Fade(Color{110, 136, 170, 255}, 0.16f));
    }

    auto mapPoint = [&](float r, float v) {
        float x = plot.x + plot.width * (r / maxRadius);
        float y = plot.y + plot.height * (1.0f - v / maxSpeed);
        return Vector2{x, y};
    };

    for (int modelIdx = 0; modelIdx < 3; ++modelIdx) {
        GravityModel model = static_cast<GravityModel>(modelIdx);
        Color color = ModelColor(model);
        color = Brighten(color, model == activeModel ? 0.18f : 0.0f);

        Vector2 prev = mapPoint(0.25f, RotationSpeed(model, 0.25f, params));
        for (int i = 1; i <= 220; ++i) {
            float r = 0.25f + (maxRadius - 0.25f) * static_cast<float>(i) / 220.0f;
            Vector2 next = mapPoint(r, RotationSpeed(model, r, params));
            DrawLineEx(prev, next, model == activeModel ? 3.2f : 2.0f, color);
            prev = next;
        }
    }

    for (int i = 0; i < 14; ++i) {
        float r = 1.2f + 1.25f * static_cast<float>(i);
        float vDark = RotationSpeed(GravityModel::kDarkMatter, r, params);
        float vMond = RotationSpeed(GravityModel::kMond, r, params);
        float observed = 0.55f * vDark + 0.45f * vMond + 0.12f * std::sin(timeSeconds * 0.8f + i * 0.8f);
        DrawCircleV(mapPoint(r, observed), 3.1f, Color{238, 243, 250, 235});
    }

    Vector2 probeLineStart = mapPoint(params.probeRadius, 0.0f);
    Vector2 probeLineEnd = mapPoint(params.probeRadius, maxSpeed);
    DrawLineEx(probeLineStart, probeLineEnd, 1.6f, Fade(Color{240, 246, 255, 255}, 0.38f));

    for (int modelIdx = 0; modelIdx < 3; ++modelIdx) {
        GravityModel model = static_cast<GravityModel>(modelIdx);
        float v = RotationSpeed(model, params.probeRadius, params);
        DrawCircleV(mapPoint(params.probeRadius, v), 4.8f, Brighten(ModelColor(model), 0.14f));
    }

    DrawText("Rotation Curves", static_cast<int>(panel.x + 18), static_cast<int>(panel.y + 14), 28, Color{235, 240, 248, 255});
    DrawText("toy galaxy: same luminous matter, different gravity assumptions", static_cast<int>(panel.x + 18), static_cast<int>(panel.y + 42), 17, Color{158, 184, 220, 255});
    DrawText("radius", static_cast<int>(plot.x + plot.width - 42), static_cast<int>(plot.y + plot.height + 10), 16, Color{188, 202, 224, 255});
    DrawText("speed", static_cast<int>(plot.x - 42), static_cast<int>(plot.y - 6), 16, Color{188, 202, 224, 255});

    for (int i = 0; i <= 5; ++i) {
        float r = maxRadius * static_cast<float>(i) / 5.0f;
        float v = maxSpeed * (1.0f - static_cast<float>(i) / 5.0f);
        char xText[32];
        char yText[32];
        std::snprintf(xText, sizeof(xText), "%.0f", r);
        std::snprintf(yText, sizeof(yText), "%.1f", v);
        DrawText(xText, static_cast<int>(plot.x + plot.width * static_cast<float>(i) / 5.0f - 8.0f), static_cast<int>(plot.y + plot.height + 8.0f), 15, Color{176, 188, 206, 255});
        DrawText(yText, static_cast<int>(plot.x - 38.0f), static_cast<int>(plot.y + plot.height * static_cast<float>(i) / 5.0f - 8.0f), 15, Color{176, 188, 206, 255});
    }

    int legendY = static_cast<int>(panel.y + panel.height - 44.0f);
    int legendX = static_cast<int>(panel.x + 20.0f);
    for (int modelIdx = 0; modelIdx < 3; ++modelIdx) {
        GravityModel model = static_cast<GravityModel>(modelIdx);
        Color color = ModelColor(model);
        DrawCircle(legendX, legendY, 7.0f, color);
        DrawText(ModelName(model), legendX + 16, legendY - 9, 18, model == activeModel ? Brighten(color, 0.20f) : Color{224, 232, 244, 255});
        legendX += 180;
    }
    DrawCircle(legendX, legendY, 6.0f, Color{240, 244, 250, 255});
    DrawText("toy observations", legendX + 14, legendY - 9, 18, Color{224, 232, 244, 255});
}

void DrawInfoPanel(const GalaxyParams& params, GravityModel activeModel, bool paused) {
    Rectangle panel = {24.0f, 20.0f, 490.0f, 162.0f};
    DrawRectangleRounded(panel, 0.06f, 16, Fade(Color{8, 15, 30, 255}, 0.78f));
    DrawRectangleRoundedLinesEx(panel, 0.06f, 16, 1.5f, Fade(Color{118, 146, 186, 255}, 0.32f));

    DrawText("Dark Matter vs MOND: Galaxy Rotation", 42, 36, 31, Color{235, 240, 248, 255});
    DrawText("3D toy comparison of flat rotation curves", 42, 70, 18, Color{154, 186, 226, 255});
    DrawText("Mouse orbit | wheel zoom | 1 baryons | 2 dark halo | 3 MOND | Left/Right probe | Q/A halo | W/S MOND a0 | +/- time | P pause | R reset",
             42, 98, 18, Color{176, 193, 216, 255});

    char status[256];
    std::snprintf(
        status,
        sizeof(status),
        "active=%s   probe=%.1f   halo=%.2f   a0=%.2f   time=%.2fx%s",
        ModelName(activeModel),
        params.probeRadius,
        params.haloStrength,
        params.mondA0,
        params.timeScale,
        paused ? "   [PAUSED]" : ""
    );
    DrawText(status, 42, 128, 19, Brighten(ModelColor(activeModel), 0.14f));
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Dark Matter vs MOND Galaxy Rotation - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {28.0f, 15.0f, 24.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 42.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float cameraYaw = 0.82f;
    float cameraPitch = 0.34f;
    float cameraDistance = 33.0f;

    GalaxyParams params;
    GravityModel activeModel = GravityModel::kDarkMatter;
    bool paused = false;

    std::vector<StarParticle> galaxyStars = BuildGalaxyStars();
    std::vector<BackgroundStar> backgroundStars = BuildBackgroundStars();
    std::array<std::array<TracerState, kTracerCount>, 3> tracers = {
        BuildTracerRing(0.0f),
        BuildTracerRing(0.8f),
        BuildTracerRing(1.6f)
    };

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_ONE)) activeModel = GravityModel::kBaryonsOnly;
        if (IsKeyPressed(KEY_TWO)) activeModel = GravityModel::kDarkMatter;
        if (IsKeyPressed(KEY_THREE)) activeModel = GravityModel::kMond;
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            params = GalaxyParams{};
            activeModel = GravityModel::kDarkMatter;
            paused = false;
            ResetGalaxy(&galaxyStars);
            tracers = {
                BuildTracerRing(0.0f),
                BuildTracerRing(0.8f),
                BuildTracerRing(1.6f)
            };
        }

        if (IsKeyDown(KEY_RIGHT)) params.probeRadius = std::min(kGalaxyRadius - 1.0f, params.probeRadius + 7.0f * GetFrameTime());
        if (IsKeyDown(KEY_LEFT)) params.probeRadius = std::max(2.0f, params.probeRadius - 7.0f * GetFrameTime());
        if (IsKeyDown(KEY_Q)) params.haloStrength = std::min(2.2f, params.haloStrength + 0.65f * GetFrameTime());
        if (IsKeyDown(KEY_A)) params.haloStrength = std::max(0.0f, params.haloStrength - 0.65f * GetFrameTime());
        if (IsKeyDown(KEY_W)) params.mondA0 = std::min(11.0f, params.mondA0 + 3.0f * GetFrameTime());
        if (IsKeyDown(KEY_S)) params.mondA0 = std::max(0.6f, params.mondA0 - 3.0f * GetFrameTime());
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) params.timeScale = std::min(3.0f, params.timeScale + 0.10f);
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) params.timeScale = std::max(0.20f, params.timeScale - 0.10f);

        UpdateOrbitCamera(&camera, &cameraYaw, &cameraPitch, &cameraDistance);

        float dt = GetFrameTime() * params.timeScale;
        if (!paused) {
            UpdateGalaxy(&galaxyStars, &tracers, activeModel, params, dt);
        }

        float timeSeconds = static_cast<float>(GetTime());

        BeginDrawing();
        DrawBackground(backgroundStars, timeSeconds);

        BeginMode3D(camera);
        DrawGalaxyPlane(params, timeSeconds);
        DrawGalaxyStars(galaxyStars, timeSeconds);
        DrawProbeRing(params, tracers, activeModel);
        EndMode3D();

        DrawInfoPanel(params, activeModel, paused);
        DrawCurvePanel({940.0f, 28.0f, 548.0f, 360.0f}, params, activeModel, timeSeconds);

        DrawRectangleRounded({940.0f, 406.0f, 548.0f, 120.0f}, 0.06f, 12, Fade(Color{8, 15, 30, 255}, 0.80f));
        DrawRectangleRoundedLinesEx({940.0f, 406.0f, 548.0f, 120.0f}, 0.06f, 12, 1.5f, Fade(Color{118, 146, 186, 255}, 0.30f));
        DrawText("Interpretation", 960, 422, 25, Color{234, 240, 248, 255});
        DrawText("Red falls away when only luminous matter gravitates.", 960, 456, 18, Color{255, 170, 162, 255});
        DrawText("Blue stays flatter by adding an unseen halo around the galaxy.", 960, 482, 18, Color{126, 214, 255, 255});
        DrawText("Gold stays flatter by modifying the low-acceleration law instead.", 960, 508, 18, Color{255, 215, 132, 255});

        DrawFPS(28, kScreenHeight - 38);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
