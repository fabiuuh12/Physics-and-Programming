#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace {

constexpr int kScreenWidth = 1440;
constexpr int kScreenHeight = 900;
constexpr float kEarthRadius = 3.0f;
constexpr int kStarCount = 240;
constexpr int kWindParticleCount = 900;

struct OrbitCameraState {
    float yaw = 0.72f;
    float pitch = 0.30f;
    float distance = 13.5f;
};

struct WindParticle {
    float lat = 0.0f;
    float lon = 0.0f;
    float speed = 0.0f;
    float layer = 0.0f;
};

struct Storm {
    float lat = 0.0f;
    float lon = 0.0f;
    float radius = 0.0f;
    float spin = 1.0f;
    float drift = 0.0f;
};

struct Star {
    Vector3 pos{};
    float radius = 0.0f;
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

Vector3 GlobePoint(float lat, float lon, float radius = kEarthRadius) {
    const float c = std::cos(lat);
    return {
        radius * c * std::cos(lon),
        radius * std::sin(lat),
        radius * c * std::sin(lon),
    };
}

float WrapPi(float angle) {
    while (angle > PI) angle -= 2.0f * PI;
    while (angle < -PI) angle += 2.0f * PI;
    return angle;
}

void UpdateOrbitCamera(Camera3D* camera, OrbitCameraState* orbit) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        const Vector2 delta = GetMouseDelta();
        orbit->yaw -= delta.x * 0.0038f;
        orbit->pitch += delta.y * 0.0038f;
        orbit->pitch = std::clamp(orbit->pitch, -1.25f, 1.25f);
    }

    orbit->distance -= GetMouseWheelMove() * 0.7f;
    orbit->distance = std::clamp(orbit->distance, 7.0f, 24.0f);

    const float cp = std::cos(orbit->pitch);
    camera->target = {0.0f, 0.0f, 0.0f};
    camera->position = Vector3Add(camera->target, {
        orbit->distance * cp * std::cos(orbit->yaw),
        orbit->distance * std::sin(orbit->pitch),
        orbit->distance * cp * std::sin(orbit->yaw),
    });
}

std::vector<Star> MakeStars() {
    std::mt19937 rng(9031);
    std::vector<Star> stars;
    stars.reserve(kStarCount);

    for (int i = 0; i < kStarCount; ++i) {
        const float lon = RandRange(rng, -PI, PI);
        const float lat = RandRange(rng, -0.48f * PI, 0.48f * PI);
        const float radius = RandRange(rng, 26.0f, 42.0f);
        stars.push_back({GlobePoint(lat, lon, radius), RandRange(rng, 0.015f, 0.055f), RandRange(rng, 0.25f, 0.92f)});
    }

    return stars;
}

std::vector<WindParticle> MakeWindParticles() {
    std::mt19937 rng(2319);
    std::vector<WindParticle> particles;
    particles.reserve(kWindParticleCount);

    for (int i = 0; i < kWindParticleCount; ++i) {
        particles.push_back({
            RandRange(rng, -1.22f, 1.22f),
            RandRange(rng, -PI, PI),
            RandRange(rng, 0.25f, 1.0f),
            RandRange(rng, 0.0f, 1.0f),
        });
    }

    return particles;
}

std::vector<Storm> MakeStorms() {
    return {
        {0.42f, -1.25f, 0.54f, 1.0f, 0.22f},
        {0.20f, 1.95f, 0.45f, 1.0f, 0.18f},
        {-0.36f, 0.45f, 0.50f, -1.0f, 0.20f},
        {-0.62f, -2.36f, 0.38f, -1.0f, 0.15f},
    };
}

float PressureField(float lat, float lon, float time) {
    const float equatorialWave = 0.42f * std::sin(3.0f * lon - time * 0.55f) * std::cos(lat * 1.25f);
    const float rossbyWave = 0.34f * std::sin(5.0f * lon + 2.6f * lat + time * 0.22f);
    const float polar = 0.32f * std::sin(std::fabs(lat) * 3.2f - time * 0.16f);
    return equatorialWave + rossbyWave + polar;
}

float TemperatureField(float lat, float lon, float time) {
    const float latitudeGradient = 1.0f - std::pow(std::fabs(lat) / (0.5f * PI), 1.28f);
    const float dayWave = 0.15f * std::cos(lon - time * 0.18f) * std::cos(lat);
    return std::clamp(latitudeGradient + dayWave, 0.0f, 1.0f);
}

Color TemperatureColor(float value) {
    if (value < 0.35f) return LerpColor(Color{34, 92, 184, 255}, Color{92, 214, 238, 255}, value / 0.35f);
    if (value < 0.68f) return LerpColor(Color{92, 214, 238, 255}, Color{236, 218, 102, 255}, (value - 0.35f) / 0.33f);
    return LerpColor(Color{236, 218, 102, 255}, Color{238, 88, 58, 255}, (value - 0.68f) / 0.32f);
}

Color PressureColor(float value) {
    const float t = std::clamp((value + 1.0f) * 0.5f, 0.0f, 1.0f);
    if (t < 0.5f) return LerpColor(Color{42, 98, 214, 255}, Color{236, 240, 244, 255}, t * 2.0f);
    return LerpColor(Color{236, 240, 244, 255}, Color{222, 66, 78, 255}, (t - 0.5f) * 2.0f);
}

void DrawGlobeGrid() {
    for (int latStep = -6; latStep <= 6; ++latStep) {
        const float lat = latStep * (PI / 18.0f);
        Vector3 prev = GlobePoint(lat, -PI, kEarthRadius * 1.006f);
        for (int i = 1; i <= 144; ++i) {
            const float lon = -PI + 2.0f * PI * static_cast<float>(i) / 144.0f;
            const Vector3 p = GlobePoint(lat, lon, kEarthRadius * 1.006f);
            DrawLine3D(prev, p, Fade(Color{190, 220, 235, 255}, 0.14f));
            prev = p;
        }
    }

    for (int lonStep = 0; lonStep < 24; ++lonStep) {
        const float lon = -PI + 2.0f * PI * static_cast<float>(lonStep) / 24.0f;
        Vector3 prev = GlobePoint(-0.48f * PI, lon, kEarthRadius * 1.007f);
        for (int i = 1; i <= 96; ++i) {
            const float lat = -0.48f * PI + 0.96f * PI * static_cast<float>(i) / 96.0f;
            const Vector3 p = GlobePoint(lat, lon, kEarthRadius * 1.007f);
            DrawLine3D(prev, p, Fade(Color{190, 220, 235, 255}, 0.10f));
            prev = p;
        }
    }
}

void DrawWeatherCells(float time, bool showTemperature, bool showPressure) {
    constexpr int kLatBands = 28;
    constexpr int kLonBands = 56;

    for (int iy = 0; iy < kLatBands; ++iy) {
        const float lat = -0.46f * PI + 0.92f * PI * (static_cast<float>(iy) + 0.5f) / kLatBands;
        for (int ix = 0; ix < kLonBands; ++ix) {
            const float lon = -PI + 2.0f * PI * (static_cast<float>(ix) + 0.5f) / kLonBands;
            Color color = Color{68, 138, 196, 255};
            float alpha = 0.11f;

            if (showTemperature) {
                color = TemperatureColor(TemperatureField(lat, lon, time));
                alpha = 0.23f;
            }
            if (showPressure) {
                color = showTemperature ? LerpColor(color, PressureColor(PressureField(lat, lon, time)), 0.42f)
                                        : PressureColor(PressureField(lat, lon, time));
                alpha = showTemperature ? 0.26f : 0.21f;
            }

            DrawSphere(GlobePoint(lat, lon, kEarthRadius * 1.013f), 0.045f + 0.035f * std::cos(lat) * std::cos(lat), Fade(color, alpha));
        }
    }
}

void DrawJetStreams(float time) {
    for (float baseLat : {-0.82f, 0.82f}) {
        Vector3 prev{};
        bool hasPrev = false;
        for (int i = 0; i <= 220; ++i) {
            const float lon = -PI + 2.0f * PI * static_cast<float>(i) / 220.0f;
            const float lat = baseLat + 0.10f * std::sin(4.0f * lon + time * 0.65f);
            const Vector3 p = GlobePoint(lat, lon, kEarthRadius * 1.085f);
            if (hasPrev) DrawLine3D(prev, p, Fade(Color{120, 230, 255, 255}, 0.58f));
            prev = p;
            hasPrev = true;
        }
    }
}

void UpdateWindParticles(std::vector<WindParticle>* particles, float dt, float time) {
    for (WindParticle& particle : *particles) {
        const float jet = 0.35f + 1.15f * std::exp(-std::pow((std::fabs(particle.lat) - 0.82f) / 0.18f, 2.0f));
        const float trade = -0.33f * std::cos(particle.lat * 3.0f);
        const float wave = 0.12f * std::sin(4.0f * particle.lon + particle.lat * 2.0f + time);
        particle.lon = WrapPi(particle.lon + (trade + jet + wave) * particle.speed * dt * 0.18f);
        particle.lat += 0.025f * std::sin(2.0f * particle.lon - time * 0.55f + particle.layer * 6.0f) * dt;
        particle.lat = std::clamp(particle.lat, -1.35f, 1.35f);
    }
}

void DrawWindParticles(const std::vector<WindParticle>& particles, float time) {
    for (const WindParticle& particle : particles) {
        const Vector3 p = GlobePoint(particle.lat, particle.lon, kEarthRadius * 1.12f);
        const Vector3 tail = GlobePoint(particle.lat, particle.lon - 0.030f - 0.018f * particle.speed, kEarthRadius * 1.115f);
        const float alpha = 0.26f + 0.20f * std::sin(time * 2.0f + particle.layer * 8.0f);
        DrawLine3D(tail, p, Fade(Color{226, 246, 255, 255}, alpha));
        DrawSphere(p, 0.018f + particle.speed * 0.012f, Fade(Color{210, 244, 255, 255}, 0.62f));
    }
}

void DrawCloudBands(float time) {
    for (int band = -5; band <= 5; ++band) {
        const float baseLat = band * 0.22f;
        Vector3 prev{};
        bool hasPrev = false;
        for (int i = 0; i <= 190; ++i) {
            const float lon = -PI + 2.0f * PI * static_cast<float>(i) / 190.0f;
            const float lat = baseLat + 0.035f * std::sin(3.0f * lon + time * 0.40f + band);
            const Vector3 p = GlobePoint(lat, lon + time * 0.035f * (band % 2 == 0 ? 1.0f : -1.0f), kEarthRadius * 1.055f);
            if (hasPrev) DrawLine3D(prev, p, Fade(Color{236, 242, 244, 255}, 0.20f));
            prev = p;
            hasPrev = true;
        }
    }
}

void DrawStorms(const std::vector<Storm>& storms, float time) {
    for (const Storm& storm : storms) {
        const float centerLon = WrapPi(storm.lon + storm.drift * time * 0.07f);
        const Vector3 eye = GlobePoint(storm.lat, centerLon, kEarthRadius * 1.15f);
        DrawSphere(eye, 0.060f, Color{245, 250, 255, 225});

        for (int arm = 0; arm < 3; ++arm) {
            Vector3 prev = eye;
            for (int i = 1; i <= 38; ++i) {
                const float t = static_cast<float>(i) / 38.0f;
                const float angle = storm.spin * (t * 4.6f + arm * 2.0f * PI / 3.0f + time * 0.72f);
                const float lat = storm.lat + storm.radius * t * 0.38f * std::sin(angle);
                const float lon = centerLon + storm.radius * t * std::cos(angle) / std::max(0.25f, std::cos(storm.lat));
                const Vector3 p = GlobePoint(lat, lon, kEarthRadius * (1.11f + 0.025f * t));
                DrawLine3D(prev, p, Fade(Color{245, 250, 255, 255}, 0.38f * (1.0f - t * 0.28f)));
                prev = p;
            }
        }
    }
}

void DrawContinents() {
    const std::vector<std::vector<Vector2>> shapes = {
        {{-2.85f, 0.88f}, {-2.45f, 1.08f}, {-2.10f, 0.78f}, {-2.18f, 0.38f}, {-2.48f, 0.10f}, {-2.70f, 0.34f}},
        {{-1.72f, 0.22f}, {-1.34f, 0.02f}, {-1.18f, -0.42f}, {-1.38f, -0.98f}, {-1.70f, -0.72f}, {-1.92f, -0.18f}},
        {{-0.22f, 0.72f}, {0.45f, 0.90f}, {1.10f, 0.66f}, {1.45f, 0.18f}, {0.78f, -0.06f}, {0.05f, 0.12f}},
        {{0.12f, 0.05f}, {0.56f, -0.20f}, {0.50f, -0.86f}, {0.18f, -1.05f}, {-0.12f, -0.42f}},
        {{1.45f, -0.45f}, {1.82f, -0.35f}, {2.12f, -0.62f}, {1.92f, -0.92f}, {1.52f, -0.78f}},
    };

    for (const auto& shape : shapes) {
        for (size_t i = 0; i < shape.size(); ++i) {
            const Vector2 a = shape[i];
            const Vector2 b = shape[(i + 1) % shape.size()];
            DrawLine3D(GlobePoint(a.y, a.x, kEarthRadius * 1.018f), GlobePoint(b.y, b.x, kEarthRadius * 1.018f), Fade(Color{92, 188, 118, 255}, 0.42f));
        }
    }
}

}  // namespace

int main() {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(kScreenWidth, kScreenHeight, "Earth Weather Globe - Meteorology Visualization");
    SetWindowMinSize(980, 640);
    SetTargetFPS(120);

    Camera3D camera{};
    camera.position = {10.0f, 4.0f, 8.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    OrbitCameraState orbit{};
    std::vector<Star> stars = MakeStars();
    std::vector<WindParticle> wind = MakeWindParticles();
    const std::vector<Storm> storms = MakeStorms();

    bool showTemperature = true;
    bool showPressure = true;
    bool showWind = true;
    bool paused = false;
    float simTime = 0.0f;
    float timeScale = 1.0f;

    while (!WindowShouldClose()) {
        const float dt = std::max(1.0e-4f, GetFrameTime());
        UpdateOrbitCamera(&camera, &orbit);

        if (IsKeyPressed(KEY_ONE)) showTemperature = !showTemperature;
        if (IsKeyPressed(KEY_TWO)) showPressure = !showPressure;
        if (IsKeyPressed(KEY_THREE)) showWind = !showWind;
        if (IsKeyPressed(KEY_SPACE)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            simTime = 0.0f;
            wind = MakeWindParticles();
        }
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) timeScale = std::min(4.0f, timeScale + 0.25f);
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) timeScale = std::max(0.25f, timeScale - 0.25f);

        if (!paused) {
            simTime += dt * timeScale;
            UpdateWindParticles(&wind, dt * timeScale, simTime);
        }

        BeginDrawing();
        ClearBackground(Color{4, 8, 18, 255});
        BeginMode3D(camera);

        for (const Star& star : stars) {
            DrawSphere(star.pos, star.radius, Fade(Color{220, 232, 255, 255}, star.alpha));
        }

        DrawSphere({0.0f, 0.0f, 0.0f}, kEarthRadius * 1.035f, Fade(Color{80, 166, 255, 255}, 0.08f));
        DrawSphere({0.0f, 0.0f, 0.0f}, kEarthRadius, Color{21, 64, 120, 255});
        DrawContinents();
        DrawWeatherCells(simTime, showTemperature, showPressure);
        DrawGlobeGrid();
        DrawCloudBands(simTime);
        DrawStorms(storms, simTime);
        DrawJetStreams(simTime);
        if (showWind) DrawWindParticles(wind, simTime);

        EndMode3D();

        DrawRectangle(14, 14, 560, 146, Fade(BLACK, 0.32f));
        DrawText("Earth Weather Globe", 28, 26, 32, Color{238, 244, 252, 255});
        DrawText("Synthetic meteorology: pressure waves, temperature bands, jet streams, cyclones, cloud cover.", 28, 62, 18, Color{176, 198, 224, 255});
        DrawText("Mouse orbit | wheel zoom | 1 temperature | 2 pressure | 3 wind | +/- speed | Space pause | R reset", 28, 90, 18, Color{132, 226, 255, 255});

        DrawText(TextFormat("Temp %s   Pressure %s   Wind %s   Time x%.2f",
                            showTemperature ? "on" : "off",
                            showPressure ? "on" : "off",
                            showWind ? "on" : "off",
                            timeScale),
                 28, 120, 18, Color{230, 238, 246, 255});

        DrawRectangle(GetScreenWidth() - 280, 18, 252, 116, Fade(BLACK, 0.28f));
        DrawText("Layer Readout", GetScreenWidth() - 260, 30, 22, Color{238, 244, 252, 255});
        DrawText("Blue/red: low/high pressure", GetScreenWidth() - 260, 60, 17, Color{210, 224, 238, 255});
        DrawText("Cyan lines: jet streams", GetScreenWidth() - 260, 84, 17, Color{160, 232, 255, 255});
        DrawText("White spirals: storm systems", GetScreenWidth() - 260, 108, 17, Color{238, 244, 252, 255});

        DrawFPS(GetScreenWidth() - 96, GetScreenHeight() - 34);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
