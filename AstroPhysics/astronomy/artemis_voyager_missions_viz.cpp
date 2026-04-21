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

constexpr int kScreenWidth = 1500;
constexpr int kScreenHeight = 920;
constexpr float kSceneExtent = 56.0f;
constexpr int kStarCount = 520;
constexpr int kTrailSamples = 220;

struct OrbitCameraState {
    float yaw = 0.74f;
    float pitch = 0.30f;
    float distance = 64.0f;
    Vector3 target = {0.0f, 0.0f, 0.0f};
};

enum class FocusMode {
    kOverview = 0,
    kArtemis = 1,
    kVoyager1 = 2,
    kVoyager2 = 3,
};

struct Planet {
    const char* name = "";
    float orbitRadius = 0.0f;
    float orbitOmega = 0.0f;
    float eccentricity = 0.0f;
    float radius = 0.0f;
    float phase = 0.0f;
    float tilt = 0.0f;
    float warpStrength = 0.0f;
    float atmosphere = 0.0f;
    float ringInner = 0.0f;
    float ringOuter = 0.0f;
    bool gasGiant = false;
    Color color{};
    Color accent{};
    Vector3 pos{};
};

struct BackdropStar {
    Vector3 pos{};
    float size = 0.0f;
    float twinkle = 0.0f;
};

struct MissionSample {
    Vector3 pos{};
    Vector3 tangent{};
    const char* phase = "";
    float localProgress = 0.0f;
};

float Clamp01(float v) {
    return std::clamp(v, 0.0f, 1.0f);
}

float SmoothStep(float t) {
    const float u = Clamp01(t);
    return u * u * (3.0f - 2.0f * u);
}

float SegmentT(float t, float a, float b) {
    if (b <= a) return 0.0f;
    return Clamp01((t - a) / (b - a));
}

float RandRange(std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    return dist(rng);
}

Color LerpColor(Color a, Color b, float t) {
    const float u = Clamp01(t);
    return Color{
        static_cast<unsigned char>(a.r + (b.r - a.r) * u),
        static_cast<unsigned char>(a.g + (b.g - a.g) * u),
        static_cast<unsigned char>(a.b + (b.b - a.b) * u),
        static_cast<unsigned char>(a.a + (b.a - a.a) * u),
    };
}

Vector3 Bezier3(Vector3 a, Vector3 b, Vector3 c, Vector3 d, float t) {
    const float u = Clamp01(t);
    const float s = 1.0f - u;
    const float s2 = s * s;
    const float u2 = u * u;
    return {
        s2 * s * a.x + 3.0f * s2 * u * b.x + 3.0f * s * u2 * c.x + u2 * u * d.x,
        s2 * s * a.y + 3.0f * s2 * u * b.y + 3.0f * s * u2 * c.y + u2 * u * d.y,
        s2 * s * a.z + 3.0f * s2 * u * b.z + 3.0f * s * u2 * c.z + u2 * u * d.z,
    };
}

void UpdateOrbitCameraDragOnly(Camera3D* camera, OrbitCameraState* orbit, Vector3 desiredTarget) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        const Vector2 delta = GetMouseDelta();
        orbit->yaw -= delta.x * 0.0038f;
        orbit->pitch += delta.y * 0.0038f;
        orbit->pitch = std::clamp(orbit->pitch, -1.24f, 1.24f);
    }

    orbit->distance -= GetMouseWheelMove() * 1.4f;
    orbit->distance = std::clamp(orbit->distance, 8.0f, 96.0f);
    orbit->target = Vector3Lerp(orbit->target, desiredTarget, 0.08f);

    const float cp = std::cos(orbit->pitch);
    camera->target = orbit->target;
    camera->position = Vector3Add(orbit->target, {
        orbit->distance * cp * std::cos(orbit->yaw),
        orbit->distance * std::sin(orbit->pitch),
        orbit->distance * cp * std::sin(orbit->yaw),
    });
}

std::vector<BackdropStar> MakeBackdropStars() {
    std::mt19937 rng(71241);
    std::vector<BackdropStar> stars;
    stars.reserve(kStarCount);
    for (int i = 0; i < kStarCount; ++i) {
        const float theta = RandRange(rng, 0.0f, 2.0f * PI);
        const float phi = RandRange(rng, -0.47f * PI, 0.47f * PI);
        const float radius = RandRange(rng, 62.0f, 92.0f);
        stars.push_back({
            {
                radius * std::cos(phi) * std::cos(theta),
                radius * std::sin(phi),
                radius * std::cos(phi) * std::sin(theta),
            },
            RandRange(rng, 0.02f, 0.09f),
            RandRange(rng, 0.5f, 4.0f),
        });
    }
    return stars;
}

std::vector<Planet> MakePlanets() {
    return {
        {"Mercury", 5.2f, 1.05f, 0.16f, 0.22f, 0.8f, 0.01f, 0.12f, 0.02f, 0.0f, 0.0f, false, Color{188, 178, 164, 255}, Color{216, 204, 188, 255}},
        {"Venus", 7.4f, 0.82f, 0.04f, 0.42f, 1.6f, -0.02f, 0.18f, 0.05f, 0.0f, 0.0f, false, Color{224, 188, 116, 255}, Color{244, 220, 156, 255}},
        {"Earth", 10.0f, 0.62f, 0.02f, 0.68f, 2.5f, 0.12f, 0.26f, 0.08f, 0.0f, 0.0f, false, Color{96, 176, 255, 255}, Color{136, 224, 255, 255}},
        {"Mars", 13.2f, 0.46f, 0.09f, 0.48f, 0.6f, 0.07f, 0.18f, 0.03f, 0.0f, 0.0f, false, Color{220, 120, 92, 255}, Color{246, 168, 122, 255}},
        {"Jupiter", 20.4f, 0.19f, 0.05f, 1.55f, 0.4f, 0.07f, 0.70f, 0.12f, 0.0f, 0.0f, true, Color{228, 186, 146, 255}, Color{246, 220, 184, 255}},
        {"Saturn", 29.5f, 0.13f, 0.04f, 1.32f, 1.5f, 0.05f, 0.58f, 0.11f, 2.05f, 2.95f, true, Color{224, 198, 128, 255}, Color{248, 228, 172, 255}},
        {"Uranus", 38.8f, 0.09f, 0.03f, 1.08f, 2.1f, -0.04f, 0.42f, 0.08f, 0.0f, 0.0f, true, Color{146, 226, 236, 255}, Color{198, 248, 255, 255}},
        {"Neptune", 47.0f, 0.07f, 0.02f, 1.04f, 2.9f, 0.03f, 0.44f, 0.08f, 0.0f, 0.0f, true, Color{92, 146, 242, 255}, Color{144, 186, 255, 255}},
    };
}

void UpdatePlanets(std::vector<Planet>* planets, float time) {
    for (Planet& planet : *planets) {
        const float a = planet.phase + time * planet.orbitOmega;
        const float radial = planet.orbitRadius * (1.0f - planet.eccentricity * planet.eccentricity) /
            std::max(0.25f, 1.0f + planet.eccentricity * std::cos(a));
        planet.pos = {
            radial * std::cos(a),
            planet.tilt * std::sin(a * 1.8f),
            radial * std::sin(a),
        };
    }
}

const Planet& FindPlanet(const std::vector<Planet>& planets, const char* name) {
    for (const Planet& planet : planets) {
        if (std::string(planet.name) == name) return planet;
    }
    return planets.front();
}

float SpacetimeHeightAtPoint(float x, float z, const std::vector<Planet>& planets, float time) {
    float h = -0.55f;
    const float sunR2 = x * x + z * z;
    h -= 7.6f / (7.0f + 0.10f * sunR2);
    h += 0.16f * std::sin(0.09f * std::sqrt(sunR2 + 1.0f) - 0.9f * time);

    for (const Planet& planet : planets) {
        const float dx = x - planet.pos.x;
        const float dz = z - planet.pos.z;
        const float d2 = dx * dx + dz * dz;
        h -= planet.warpStrength / (1.5f + 1.4f * d2);
    }
    return h;
}

void AnchorPlanetsToSpacetime(std::vector<Planet>* planets, float time) {
    for (Planet& planet : *planets) {
        const float base = SpacetimeHeightAtPoint(planet.pos.x, planet.pos.z, *planets, time);
        planet.pos.y = base + planet.radius + 0.14f + planet.tilt * std::sin(time * planet.orbitOmega * 2.3f + planet.phase);
    }
}

Vector3 MakeMoonPosition(const Planet& earth, float time) {
    const float moonRadius = 3.4f;
    const float a = time * 2.4f + 1.1f;
    Vector3 moon = Vector3Add(earth.pos, {
        moonRadius * std::cos(a),
        0.55f * std::sin(a * 1.7f),
        moonRadius * std::sin(a),
    });
    moon.y = earth.pos.y + 0.28f + 0.55f * std::sin(a * 1.7f);
    return moon;
}

void DrawSpacetimeGrid(const std::vector<Planet>& planets, float time, float extent) {
    constexpr int kGridLines = 34;
    constexpr int kSegments = 84;
    const float step = (2.0f * extent) / static_cast<float>(kGridLines - 1);
    const Color cool = Color{42, 104, 168, 255};
    const Color bright = Color{98, 188, 255, 255};

    for (int i = 0; i < kGridLines; ++i) {
        const float x = -extent + step * i;
        Vector3 prev = {x, SpacetimeHeightAtPoint(x, -extent, planets, time), -extent};
        for (int s = 1; s < kSegments; ++s) {
            const float z = -extent + (2.0f * extent * s) / static_cast<float>(kSegments - 1);
            Vector3 cur = {x, SpacetimeHeightAtPoint(x, z, planets, time), z};
            const float glow = Clamp01((2.0f + cur.y) * 0.20f);
            DrawLine3D(prev, cur, Fade(LerpColor(cool, bright, glow), i % 4 == 0 ? 0.34f : 0.20f));
            prev = cur;
        }
    }

    for (int i = 0; i < kGridLines; ++i) {
        const float z = -extent + step * i;
        Vector3 prev = {-extent, SpacetimeHeightAtPoint(-extent, z, planets, time), z};
        for (int s = 1; s < kSegments; ++s) {
            const float x = -extent + (2.0f * extent * s) / static_cast<float>(kSegments - 1);
            Vector3 cur = {x, SpacetimeHeightAtPoint(x, z, planets, time), z};
            const float glow = Clamp01((2.0f + cur.y) * 0.20f);
            DrawLine3D(prev, cur, Fade(LerpColor(cool, bright, glow), i % 4 == 0 ? 0.34f : 0.20f));
            prev = cur;
        }
    }
}

void DrawTimelineBar(float progress, float artemisDay, float voyager1Year, float voyager2Year) {
    const Rectangle frame = {18.0f, static_cast<float>(kScreenHeight - 114), 976.0f, 82.0f};
    DrawRectangleRounded(frame, 0.10f, 12, Fade(Color{8, 14, 26, 255}, 0.94f));
    DrawRectangleRoundedLinesEx(frame, 0.10f, 12, 1.0f, Fade(Color{74, 104, 144, 255}, 0.80f));
    DrawText("Mission timeline control", 34, kScreenHeight - 102, 22, Color{228, 236, 246, 255});

    const int x0 = 34;
    const int x1 = 960;
    const int yArtemis = kScreenHeight - 72;
    const int yV1 = kScreenHeight - 54;
    const int yV2 = kScreenHeight - 36;

    DrawLine(x0, yArtemis, x1, yArtemis, Fade(Color{132, 224, 255, 255}, 0.60f));
    DrawLine(x0, yV1, x1, yV1, Fade(Color{255, 214, 126, 255}, 0.60f));
    DrawLine(x0, yV2, x1, yV2, Fade(Color{132, 255, 214, 255}, 0.60f));

    const std::array<float, 7> artemisTicks = {0.0f, 0.12f, 0.28f, 0.52f, 0.66f, 0.94f, 1.0f};
    const std::array<float, 6> voyager1Ticks = {0.0f, 0.12f, 0.42f, 0.56f, 0.80f, 0.90f};
    const std::array<float, 9> voyager2Ticks = {0.0f, 0.08f, 0.28f, 0.36f, 0.52f, 0.60f, 0.74f, 0.82f, 0.92f};

    for (float t : artemisTicks) {
        const int x = x0 + static_cast<int>((x1 - x0) * t);
        DrawLine(x, yArtemis - 6, x, yArtemis + 6, Color{132, 224, 255, 255});
    }
    for (float t : voyager1Ticks) {
        const int x = x0 + static_cast<int>((x1 - x0) * t);
        DrawLine(x, yV1 - 6, x, yV1 + 6, Color{255, 214, 126, 255});
    }
    for (float t : voyager2Ticks) {
        const int x = x0 + static_cast<int>((x1 - x0) * t);
        DrawLine(x, yV2 - 6, x, yV2 + 6, Color{132, 255, 214, 255});
    }

    const int px = x0 + static_cast<int>((x1 - x0) * progress);
    DrawLine(px, yArtemis - 16, px, yV2 + 14, Color{255, 255, 255, 255});
    DrawCircle(px, yArtemis, 5.0f, Color{132, 224, 255, 255});
    DrawCircle(px, yV1, 5.0f, Color{255, 214, 126, 255});
    DrawCircle(px, yV2, 5.0f, Color{132, 255, 214, 255});

    DrawText(TextFormat("Artemis II  day %4.1f / 10.5", artemisDay), 708, kScreenHeight - 102, 18, Color{132, 224, 255, 255});
    DrawText(TextFormat("Voyager 1  year %.1f", voyager1Year), 708, kScreenHeight - 78, 18, Color{255, 214, 126, 255});
    DrawText(TextFormat("Voyager 2  year %.1f", voyager2Year), 708, kScreenHeight - 54, 18, Color{132, 255, 214, 255});
}

Vector3 EvaluateArtemisPosition(float t, const Planet& earth, Vector3 moonPos) {
    const Vector3 launchPad = Vector3Add(earth.pos, {0.0f, earth.radius + 0.15f, 0.0f});
    const Vector3 parkingA = Vector3Add(earth.pos, {2.4f, 0.8f, 0.0f});
    const Vector3 transLunarStart = Vector3Add(earth.pos, {2.6f, 0.2f, 0.0f});
    const Vector3 lunarApproach = Vector3Add(moonPos, {-1.7f, 0.6f, -0.9f});
    const Vector3 flybyExit = Vector3Add(moonPos, {1.4f, -0.2f, 0.9f});
    const Vector3 returnTarget = Vector3Add(earth.pos, {-2.7f, -0.2f, 0.8f});
    const Vector3 splashdown = Vector3Add(earth.pos, {-0.8f, earth.radius + 0.1f, -0.6f});

    if (t < 0.12f) {
        const float u = SmoothStep(SegmentT(t, 0.0f, 0.12f));
        return Bezier3(
            launchPad,
            Vector3Add(earth.pos, {0.7f, 1.8f, 0.3f}),
            Vector3Add(earth.pos, {1.4f, 1.1f, 1.4f}),
            parkingA,
            u
        );
    }
    if (t < 0.28f) {
        const float u = SegmentT(t, 0.12f, 0.28f);
        const float theta = 4.0f * PI * u;
        return Vector3Add(earth.pos, {
            2.65f * std::cos(theta),
            0.6f + 0.35f * std::sin(theta * 0.5f),
            2.15f * std::sin(theta),
        });
    }
    if (t < 0.52f) {
        const float u = SmoothStep(SegmentT(t, 0.28f, 0.52f));
        return Bezier3(
            transLunarStart,
            Vector3Add(earth.pos, {6.0f, 1.6f, 3.5f}),
            Vector3Add(moonPos, {-4.2f, 0.9f, -2.6f}),
            lunarApproach,
            u
        );
    }
    if (t < 0.66f) {
        const float u = SegmentT(t, 0.52f, 0.66f);
        const float theta = -0.9f * PI + 1.5f * PI * u;
        return Vector3Add(moonPos, {
            1.6f * std::cos(theta),
            0.55f * std::sin(theta * 1.7f),
            1.15f * std::sin(theta),
        });
    }
    if (t < 0.94f) {
        const float u = SmoothStep(SegmentT(t, 0.66f, 0.94f));
        return Bezier3(
            flybyExit,
            Vector3Add(moonPos, {5.8f, -0.8f, 3.0f}),
            Vector3Add(earth.pos, {-6.4f, 0.5f, -3.2f}),
            returnTarget,
            u
        );
    }
    const float u = SmoothStep(SegmentT(t, 0.94f, 1.0f));
    return Bezier3(
        returnTarget,
        Vector3Add(earth.pos, {-1.8f, 1.4f, 0.8f}),
        Vector3Add(earth.pos, {-1.0f, 0.8f, -0.1f}),
        splashdown,
        u
    );
}

MissionSample EvaluateArtemisMission(float t, const Planet& earth, Vector3 moonPos) {
    MissionSample sample{};
    sample.pos = EvaluateArtemisPosition(t, earth, moonPos);
    if (t < 0.12f) {
        sample.phase = "Launch + ascent";
        sample.localProgress = SmoothStep(SegmentT(t, 0.0f, 0.12f));
    } else if (t < 0.28f) {
        sample.phase = "High Earth orbit checkout";
        sample.localProgress = SegmentT(t, 0.12f, 0.28f);
    } else if (t < 0.52f) {
        sample.phase = "Trans-lunar injection + coast";
        sample.localProgress = SmoothStep(SegmentT(t, 0.28f, 0.52f));
    } else if (t < 0.66f) {
        sample.phase = "Lunar flyby";
        sample.localProgress = SegmentT(t, 0.52f, 0.66f);
    } else if (t < 0.94f) {
        sample.phase = "Free-return toward Earth";
        sample.localProgress = SmoothStep(SegmentT(t, 0.66f, 0.94f));
    } else {
        sample.phase = "Re-entry + recovery";
        sample.localProgress = SmoothStep(SegmentT(t, 0.94f, 1.0f));
    }
    const float eps = 0.0025f;
    const Vector3 p1 = EvaluateArtemisPosition(Clamp01(t + eps), earth, moonPos);
    const Vector3 p0 = EvaluateArtemisPosition(Clamp01(t - eps), earth, moonPos);
    sample.tangent = Vector3Normalize(Vector3Subtract(p1, p0));
    if (Vector3Length(sample.tangent) < 0.001f) sample.tangent = {1.0f, 0.0f, 0.0f};
    return sample;
}

Vector3 EvaluateVoyager1Position(float t, const Planet& earth, const Planet& jupiter, const Planet& saturn) {
    const Vector3 launch = Vector3Add(earth.pos, {0.0f, earth.radius + 0.25f, 0.0f});
    const Vector3 earthDeparture = Vector3Add(earth.pos, {2.8f, 0.7f, 1.5f});
    const Vector3 jApproach = Vector3Add(jupiter.pos, {-3.3f, 1.1f, -1.5f});
    const Vector3 jExit = Vector3Add(jupiter.pos, {3.6f, 0.5f, 1.4f});
    const Vector3 sApproach = Vector3Add(saturn.pos, {-3.8f, 0.8f, -2.1f});
    const Vector3 sExit = Vector3Add(saturn.pos, {4.2f, 1.1f, -1.3f});
    const Vector3 interstellar = {50.0f, 11.0f, -33.0f};

    if (t < 0.12f) {
        const float u = SmoothStep(SegmentT(t, 0.0f, 0.12f));
        return Bezier3(
            launch,
            Vector3Add(earth.pos, {1.2f, 1.9f, 0.2f}),
            Vector3Add(earth.pos, {2.0f, 1.2f, 1.2f}),
            earthDeparture,
            u
        );
    }
    if (t < 0.42f) {
        const float u = SmoothStep(SegmentT(t, 0.12f, 0.42f));
        return Bezier3(
            earthDeparture,
            Vector3Add(earth.pos, {7.5f, 2.0f, 4.0f}),
            Vector3Add(jupiter.pos, {-8.2f, 1.7f, -5.0f}),
            jApproach,
            u
        );
    }
    if (t < 0.56f) {
        const float u = SegmentT(t, 0.42f, 0.56f);
        const float theta = PI * (0.18f + 1.15f * u);
        return Vector3Add(jupiter.pos, {
            3.6f * std::cos(theta),
            1.0f * std::sin(theta * 1.35f),
            2.3f * std::sin(theta),
        });
    }
    if (t < 0.80f) {
        const float u = SmoothStep(SegmentT(t, 0.56f, 0.80f));
        return Bezier3(
            jExit,
            Vector3Add(jupiter.pos, {8.6f, 1.6f, 3.4f}),
            Vector3Add(saturn.pos, {-9.2f, 1.2f, -5.4f}),
            sApproach,
            u
        );
    }
    if (t < 0.90f) {
        const float u = SegmentT(t, 0.80f, 0.90f);
        const float theta = -0.65f * PI + 1.05f * PI * u;
        return Vector3Add(saturn.pos, {
            4.1f * std::cos(theta),
            0.9f * std::sin(theta * 1.2f),
            3.0f * std::sin(theta),
        });
    }
    const float u = SmoothStep(SegmentT(t, 0.90f, 1.0f));
    return Bezier3(
        sExit,
        Vector3Add(saturn.pos, {8.0f, 3.0f, -6.0f}),
        {42.0f, 9.0f, -24.0f},
        interstellar,
        u
    );
}

MissionSample EvaluateVoyager1Mission(float t, const Planet& earth, const Planet& jupiter, const Planet& saturn) {
    MissionSample sample{};
    sample.pos = EvaluateVoyager1Position(t, earth, jupiter, saturn);
    if (t < 0.12f) {
        sample.phase = "Launch from Earth";
        sample.localProgress = SmoothStep(SegmentT(t, 0.0f, 0.12f));
    } else if (t < 0.42f) {
        sample.phase = "Cruise to Jupiter";
        sample.localProgress = SmoothStep(SegmentT(t, 0.12f, 0.42f));
    } else if (t < 0.56f) {
        sample.phase = "Jupiter gravity assist";
        sample.localProgress = SegmentT(t, 0.42f, 0.56f);
    } else if (t < 0.80f) {
        sample.phase = "Cruise to Saturn";
        sample.localProgress = SmoothStep(SegmentT(t, 0.56f, 0.80f));
    } else if (t < 0.90f) {
        sample.phase = "Saturn + Titan shaping";
        sample.localProgress = SegmentT(t, 0.80f, 0.90f);
    } else {
        sample.phase = "Interstellar cruise";
        sample.localProgress = SmoothStep(SegmentT(t, 0.90f, 1.0f));
    }
    const float eps = 0.0025f;
    const Vector3 p1 = EvaluateVoyager1Position(Clamp01(t + eps), earth, jupiter, saturn);
    const Vector3 p0 = EvaluateVoyager1Position(Clamp01(t - eps), earth, jupiter, saturn);
    sample.tangent = Vector3Normalize(Vector3Subtract(p1, p0));
    if (Vector3Length(sample.tangent) < 0.001f) sample.tangent = {1.0f, 0.0f, 0.0f};
    return sample;
}

Vector3 EvaluateVoyager2Position(float t, const Planet& earth, const Planet& jupiter, const Planet& saturn, const Planet& uranus, const Planet& neptune) {
    const Vector3 launch = Vector3Add(earth.pos, {0.0f, earth.radius + 0.18f, 0.0f});
    const Vector3 earthDeparture = Vector3Add(earth.pos, {-2.7f, 0.6f, 1.6f});
    const Vector3 jApproach = Vector3Add(jupiter.pos, {-3.8f, -0.6f, 2.0f});
    const Vector3 jExit = Vector3Add(jupiter.pos, {3.4f, 0.2f, -2.1f});
    const Vector3 sApproach = Vector3Add(saturn.pos, {-4.6f, 0.6f, 2.4f});
    const Vector3 sExit = Vector3Add(saturn.pos, {4.5f, -0.3f, -1.8f});
    const Vector3 uApproach = Vector3Add(uranus.pos, {-4.0f, 0.5f, 1.6f});
    const Vector3 uExit = Vector3Add(uranus.pos, {4.5f, 0.3f, -0.8f});
    const Vector3 nApproach = Vector3Add(neptune.pos, {-4.6f, 0.1f, 1.2f});
    const Vector3 nExit = Vector3Add(neptune.pos, {4.8f, 0.4f, -1.0f});
    const Vector3 interstellar = {35.0f, -13.0f, -47.0f};

    if (t < 0.08f) {
        const float u = SmoothStep(SegmentT(t, 0.0f, 0.08f));
        return Bezier3(
            launch,
            Vector3Add(earth.pos, {-1.0f, 1.5f, 0.6f}),
            Vector3Add(earth.pos, {-2.0f, 1.0f, 1.6f}),
            earthDeparture,
            u
        );
    }
    if (t < 0.28f) {
        const float u = SmoothStep(SegmentT(t, 0.08f, 0.28f));
        return Bezier3(
            earthDeparture,
            Vector3Add(earth.pos, {-7.0f, 1.8f, 4.6f}),
            Vector3Add(jupiter.pos, {-8.8f, -0.2f, 5.0f}),
            jApproach,
            u
        );
    }
    if (t < 0.36f) {
        const float u = SegmentT(t, 0.28f, 0.36f);
        const float theta = 0.95f * PI + 1.05f * PI * u;
        return Vector3Add(jupiter.pos, {
            3.8f * std::cos(theta),
            1.1f * std::sin(theta * 1.4f),
            2.5f * std::sin(theta),
        });
    }
    if (t < 0.52f) {
        const float u = SmoothStep(SegmentT(t, 0.36f, 0.52f));
        return Bezier3(
            jExit,
            Vector3Add(jupiter.pos, {8.2f, 1.0f, -4.5f}),
            Vector3Add(saturn.pos, {-9.4f, 0.9f, 4.6f}),
            sApproach,
            u
        );
    }
    if (t < 0.60f) {
        const float u = SegmentT(t, 0.52f, 0.60f);
        const float theta = 1.10f * PI + 0.95f * PI * u;
        return Vector3Add(saturn.pos, {
            4.2f * std::cos(theta),
            0.7f * std::sin(theta * 1.3f),
            2.8f * std::sin(theta),
        });
    }
    if (t < 0.74f) {
        const float u = SmoothStep(SegmentT(t, 0.60f, 0.74f));
        return Bezier3(
            sExit,
            Vector3Add(saturn.pos, {8.8f, 0.5f, -3.4f}),
            Vector3Add(uranus.pos, {-9.5f, 0.6f, 3.0f}),
            uApproach,
            u
        );
    }
    if (t < 0.82f) {
        const float u = SegmentT(t, 0.74f, 0.82f);
        const float theta = 1.05f * PI + 1.00f * PI * u;
        return Vector3Add(uranus.pos, {
            3.8f * std::cos(theta),
            0.6f * std::sin(theta * 1.25f),
            2.2f * std::sin(theta),
        });
    }
    if (t < 0.92f) {
        const float u = SmoothStep(SegmentT(t, 0.82f, 0.92f));
        return Bezier3(
            uExit,
            Vector3Add(uranus.pos, {7.8f, -0.4f, -2.6f}),
            Vector3Add(neptune.pos, {-8.8f, 0.2f, 3.6f}),
            nApproach,
            u
        );
    }
    if (t < 0.97f) {
        const float u = SegmentT(t, 0.92f, 0.97f);
        const float theta = 1.16f * PI + 1.10f * PI * u;
        return Vector3Add(neptune.pos, {
            4.0f * std::cos(theta),
            0.5f * std::sin(theta * 1.2f),
            2.6f * std::sin(theta),
        });
    }
    const float u = SmoothStep(SegmentT(t, 0.97f, 1.0f));
    return Bezier3(
        nExit,
        Vector3Add(neptune.pos, {7.0f, -2.5f, -5.0f}),
        {28.0f, -9.0f, -38.0f},
        interstellar,
        u
    );
}

MissionSample EvaluateVoyager2Mission(float t, const Planet& earth, const Planet& jupiter, const Planet& saturn, const Planet& uranus, const Planet& neptune) {
    MissionSample sample{};
    sample.pos = EvaluateVoyager2Position(t, earth, jupiter, saturn, uranus, neptune);
    if (t < 0.08f) {
        sample.phase = "Launch from Earth";
        sample.localProgress = SmoothStep(SegmentT(t, 0.0f, 0.08f));
    } else if (t < 0.28f) {
        sample.phase = "Cruise to Jupiter";
        sample.localProgress = SmoothStep(SegmentT(t, 0.08f, 0.28f));
    } else if (t < 0.36f) {
        sample.phase = "Jupiter gravity assist";
        sample.localProgress = SegmentT(t, 0.28f, 0.36f);
    } else if (t < 0.52f) {
        sample.phase = "Cruise to Saturn";
        sample.localProgress = SmoothStep(SegmentT(t, 0.36f, 0.52f));
    } else if (t < 0.60f) {
        sample.phase = "Saturn flyby";
        sample.localProgress = SegmentT(t, 0.52f, 0.60f);
    } else if (t < 0.74f) {
        sample.phase = "Cruise to Uranus";
        sample.localProgress = SmoothStep(SegmentT(t, 0.60f, 0.74f));
    } else if (t < 0.82f) {
        sample.phase = "Uranus flyby";
        sample.localProgress = SegmentT(t, 0.74f, 0.82f);
    } else if (t < 0.92f) {
        sample.phase = "Cruise to Neptune";
        sample.localProgress = SmoothStep(SegmentT(t, 0.82f, 0.92f));
    } else if (t < 0.97f) {
        sample.phase = "Neptune + Triton geometry";
        sample.localProgress = SegmentT(t, 0.92f, 0.97f);
    } else {
        sample.phase = "Southbound interstellar cruise";
        sample.localProgress = SmoothStep(SegmentT(t, 0.97f, 1.0f));
    }
    const float eps = 0.0025f;
    const Vector3 p1 = EvaluateVoyager2Position(Clamp01(t + eps), earth, jupiter, saturn, uranus, neptune);
    const Vector3 p0 = EvaluateVoyager2Position(Clamp01(t - eps), earth, jupiter, saturn, uranus, neptune);
    sample.tangent = Vector3Normalize(Vector3Subtract(p1, p0));
    if (Vector3Length(sample.tangent) < 0.001f) sample.tangent = {1.0f, 0.0f, 0.0f};
    return sample;
}

template <typename EvalFn>
void DrawMissionTrail(EvalFn eval, float currentT, Color pastColor, Color futureColor) {
    Vector3 prev = eval(0.0f).pos;
    for (int i = 1; i < kTrailSamples; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(kTrailSamples - 1);
        const Vector3 cur = eval(t).pos;
        const float fadeT = std::pow(t, 0.72f);
        const Color color = t <= currentT
            ? Fade(pastColor, 0.25f + 0.65f * fadeT)
            : Fade(futureColor, 0.06f + 0.16f * (1.0f - fadeT));
        DrawLine3D(prev, cur, color);
        prev = cur;
    }
}

void DrawOrbitRing(float radius, Color color) {
    constexpr int kSegments = 180;
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

void MakeBasis(Vector3 forward, Vector3* right, Vector3* up) {
    const Vector3 worldUp = std::fabs(forward.y) > 0.92f ? Vector3{0.0f, 0.0f, 1.0f} : Vector3{0.0f, 1.0f, 0.0f};
    *right = Vector3Normalize(Vector3CrossProduct(worldUp, forward));
    *up = Vector3Normalize(Vector3CrossProduct(forward, *right));
}

void DrawRingPlane(Vector3 center, Vector3 right, Vector3 up, float radius, Color color) {
    constexpr int kSegments = 28;
    Vector3 prev = Vector3Add(center, Vector3Scale(right, radius));
    for (int i = 1; i <= kSegments; ++i) {
        const float a = (2.0f * PI * i) / kSegments;
        const Vector3 point = Vector3Add(
            center,
            Vector3Add(Vector3Scale(right, radius * std::cos(a)), Vector3Scale(up, radius * std::sin(a)))
        );
        DrawLine3D(prev, point, color);
        prev = point;
    }
}

void DrawPlanetVisual(const Planet& planet, float time) {
    DrawSphere(planet.pos, planet.radius, planet.color);
    DrawSphereWires(planet.pos, planet.radius * 1.08f, 12, 12, Fade(planet.accent, 0.18f));
    if (planet.atmosphere > 0.001f) {
        DrawSphereWires(planet.pos, planet.radius + planet.atmosphere, 14, 14, Fade(planet.accent, 0.20f));
        DrawSphereWires(planet.pos, planet.radius + planet.atmosphere * 1.8f, 14, 14, Fade(planet.accent, 0.08f));
    }

    if (planet.gasGiant) {
        for (int i = -3; i <= 3; ++i) {
            const float lat = static_cast<float>(i) / 3.0f;
            const float bandR = planet.radius * (0.42f + 0.48f * (1.0f - std::fabs(lat)));
            const float yOff = lat * planet.radius * 0.58f;
            DrawRingPlane(
                Vector3Add(planet.pos, {0.0f, yOff, 0.0f}),
                {1.0f, 0.0f, 0.0f},
                {0.0f, 0.0f, 1.0f},
                bandR,
                Fade(i % 2 == 0 ? planet.accent : planet.color, 0.16f)
            );
        }
    }

    if (planet.ringOuter > 0.0f) {
        const float ringTilt = 0.24f + 0.08f * std::sin(time * 0.7f + planet.phase);
        DrawRingPlane(planet.pos, {1.0f, 0.0f, 0.0f}, {0.0f, ringTilt, 1.0f}, planet.ringInner, Fade(planet.accent, 0.26f));
        DrawRingPlane(planet.pos, {1.0f, 0.0f, 0.0f}, {0.0f, ringTilt, 1.0f}, planet.ringOuter, Fade(planet.accent, 0.34f));
        for (int i = 0; i < 3; ++i) {
            const float r = planet.ringInner + (planet.ringOuter - planet.ringInner) * (static_cast<float>(i) / 2.0f);
            DrawRingPlane(planet.pos, {1.0f, 0.0f, 0.0f}, {0.0f, ringTilt, 1.0f}, r, Fade(planet.color, 0.14f));
        }
    }
}

void DrawProjectedOrbit(const Planet& planet, const std::vector<Planet>& planets, float time) {
    constexpr int kSegments = 180;
    Vector3 prev{};
    for (int i = 0; i <= kSegments; ++i) {
        const float a = (2.0f * PI * i) / kSegments;
        const float angle = planet.phase + a;
        const float radial = planet.orbitRadius * (1.0f - planet.eccentricity * planet.eccentricity) /
            std::max(0.25f, 1.0f + planet.eccentricity * std::cos(angle));
        const float x = radial * std::cos(angle);
        const float z = radial * std::sin(angle);
        Vector3 cur = {x, SpacetimeHeightAtPoint(x, z, planets, time) + 0.02f, z};
        if (i > 0) DrawLine3D(prev, cur, Fade(planet.accent, 0.12f));
        prev = cur;
    }
}

void DrawVoyagerProbe(Vector3 pos, Vector3 tangent, Color color) {
    const Vector3 forward = Vector3Normalize(tangent);
    Vector3 right{}, up{};
    MakeBasis(forward, &right, &up);

    const Vector3 dishCenter = Vector3Add(pos, Vector3Scale(forward, 0.28f));
    const Vector3 busCenter = Vector3Add(pos, Vector3Scale(forward, -0.14f));
    const Vector3 rtgArmA = Vector3Add(busCenter, Vector3Scale(right, 0.34f));
    const Vector3 rtgArmB = Vector3Add(busCenter, Vector3Scale(right, -0.22f));
    const Vector3 rtgEndA = Vector3Add(rtgArmA, Vector3Add(Vector3Scale(up, 0.16f), Vector3Scale(forward, -0.10f)));
    const Vector3 rtgEndB = Vector3Add(rtgArmB, Vector3Add(Vector3Scale(up, -0.14f), Vector3Scale(forward, -0.08f)));
    const Vector3 magBoomEnd = Vector3Add(busCenter, Vector3Scale(forward, -0.92f));

    DrawSphere(busCenter, 0.12f, color);
    DrawLine3D(busCenter, dishCenter, Fade(color, 0.95f));
    DrawRingPlane(dishCenter, right, up, 0.22f, Fade(RAYWHITE, 0.85f));
    DrawLine3D(dishCenter, Vector3Add(dishCenter, Vector3Scale(forward, 0.26f)), Fade(RAYWHITE, 0.75f));
    DrawLine3D(busCenter, rtgArmA, Fade(color, 0.90f));
    DrawLine3D(busCenter, rtgArmB, Fade(color, 0.90f));
    DrawLine3D(rtgArmA, rtgEndA, Fade(ORANGE, 0.85f));
    DrawLine3D(rtgArmB, rtgEndB, Fade(ORANGE, 0.85f));
    DrawSphere(rtgEndA, 0.05f, Fade(ORANGE, 0.95f));
    DrawSphere(rtgEndB, 0.05f, Fade(ORANGE, 0.95f));
    DrawLine3D(busCenter, magBoomEnd, Fade(SKYBLUE, 0.85f));
}

void DrawOrionCraft(Vector3 pos, Vector3 tangent, Color color) {
    const Vector3 forward = Vector3Normalize(tangent);
    Vector3 right{}, up{};
    MakeBasis(forward, &right, &up);

    const Vector3 capsule = Vector3Add(pos, Vector3Scale(forward, 0.16f));
    const Vector3 service = Vector3Add(pos, Vector3Scale(forward, -0.20f));
    const Vector3 arrayRootA = Vector3Add(service, Vector3Scale(right, 0.18f));
    const Vector3 arrayRootB = Vector3Add(service, Vector3Scale(right, -0.18f));
    const Vector3 arrayEndA = Vector3Add(arrayRootA, Vector3Scale(right, 0.62f));
    const Vector3 arrayEndB = Vector3Add(arrayRootB, Vector3Scale(right, -0.62f));
    const Vector3 antenna = Vector3Add(capsule, Vector3Scale(forward, 0.28f));

    DrawSphere(capsule, 0.16f, color);
    DrawSphere(service, 0.12f, Fade(SKYBLUE, 0.9f));
    DrawLine3D(service, capsule, Fade(RAYWHITE, 0.85f));
    DrawRingPlane(capsule, right, up, 0.22f, Fade(RAYWHITE, 0.5f));
    DrawLine3D(arrayRootA, arrayEndA, Fade(Color{112, 188, 255, 255}, 0.95f));
    DrawLine3D(arrayRootA, Vector3Add(arrayEndA, Vector3Scale(up, 0.16f)), Fade(Color{112, 188, 255, 255}, 0.55f));
    DrawLine3D(arrayRootA, Vector3Add(arrayEndA, Vector3Scale(up, -0.16f)), Fade(Color{112, 188, 255, 255}, 0.55f));
    DrawLine3D(arrayRootB, arrayEndB, Fade(Color{112, 188, 255, 255}, 0.95f));
    DrawLine3D(arrayRootB, Vector3Add(arrayEndB, Vector3Scale(up, 0.16f)), Fade(Color{112, 188, 255, 255}, 0.55f));
    DrawLine3D(arrayRootB, Vector3Add(arrayEndB, Vector3Scale(up, -0.16f)), Fade(Color{112, 188, 255, 255}, 0.55f));
    DrawLine3D(capsule, antenna, Fade(RAYWHITE, 0.8f));
}

void DrawEncounterPulse(Vector3 center, float radius, float time, Color color) {
    for (int i = 0; i < 3; ++i) {
        const float ring = radius + 0.35f * i + 0.18f * std::sin(time * 2.2f + i * 1.3f);
        DrawSphereWires(center, ring, 12, 12, Fade(color, 0.16f));
    }
}

void DrawPlanetLabel(const Camera3D& camera, Vector3 pos, const char* text, Color color) {
    const Vector2 screen = GetWorldToScreen(pos, camera);
    if (screen.x < 0.0f || screen.x > static_cast<float>(kScreenWidth) || screen.y < 0.0f || screen.y > static_cast<float>(kScreenHeight)) return;
    DrawText(text, static_cast<int>(screen.x) + 6, static_cast<int>(screen.y) - 6, 18, color);
}

void SetFocus(FocusMode focus, OrbitCameraState* orbit) {
    switch (focus) {
        case FocusMode::kOverview:
            orbit->distance = 64.0f;
            orbit->yaw = 0.74f;
            orbit->pitch = 0.30f;
            break;
        case FocusMode::kArtemis:
            orbit->distance = 15.0f;
            orbit->yaw = 0.92f;
            orbit->pitch = 0.42f;
            break;
        case FocusMode::kVoyager1:
            orbit->distance = 18.0f;
            orbit->yaw = 0.34f;
            orbit->pitch = 0.28f;
            break;
        case FocusMode::kVoyager2:
            orbit->distance = 20.0f;
            orbit->yaw = 1.24f;
            orbit->pitch = 0.22f;
            break;
    }
}

const char* FocusName(FocusMode focus) {
    switch (focus) {
        case FocusMode::kOverview: return "Overview";
        case FocusMode::kArtemis: return "Artemis II";
        case FocusMode::kVoyager1: return "Voyager 1";
        case FocusMode::kVoyager2: return "Voyager 2";
    }
    return "";
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Artemis II + Voyager 1 & 2 Mission Observatory - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {48.0f, 28.0f, 48.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    OrbitCameraState orbit{};
    std::vector<Planet> planets = MakePlanets();
    const std::vector<BackdropStar> stars = MakeBackdropStars();

    float sceneTime = 0.0f;
    float masterProgress = 0.0f;
    float timeScale = 0.08f;
    bool paused = false;
    FocusMode focus = FocusMode::kOverview;
    SetFocus(focus, &orbit);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE)) paused = !paused;
        if (IsKeyPressed(KEY_R)) {
            sceneTime = 0.0f;
            masterProgress = 0.0f;
            timeScale = 0.08f;
            paused = false;
            focus = FocusMode::kOverview;
            SetFocus(focus, &orbit);
        }
        if (IsKeyPressed(KEY_ZERO)) {
            focus = FocusMode::kOverview;
            SetFocus(focus, &orbit);
        }
        if (IsKeyPressed(KEY_ONE)) {
            focus = FocusMode::kArtemis;
            SetFocus(focus, &orbit);
        }
        if (IsKeyPressed(KEY_TWO)) {
            focus = FocusMode::kVoyager1;
            SetFocus(focus, &orbit);
        }
        if (IsKeyPressed(KEY_THREE)) {
            focus = FocusMode::kVoyager2;
            SetFocus(focus, &orbit);
        }
        if (IsKeyPressed(KEY_TAB)) {
            focus = static_cast<FocusMode>((static_cast<int>(focus) + 1) % 4);
            SetFocus(focus, &orbit);
        }
        if (IsKeyDown(KEY_LEFT)) masterProgress = std::max(0.0f, masterProgress - 0.18f * GetFrameTime());
        if (IsKeyDown(KEY_RIGHT)) masterProgress = std::min(1.0f, masterProgress + 0.18f * GetFrameTime());
        if (IsKeyDown(KEY_Q)) masterProgress = std::max(0.0f, masterProgress - 0.04f * GetFrameTime());
        if (IsKeyDown(KEY_E)) masterProgress = std::min(1.0f, masterProgress + 0.04f * GetFrameTime());
        if (IsKeyPressed(KEY_LEFT_BRACKET)) timeScale = std::max(0.005f, timeScale - 0.015f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) timeScale = std::min(0.40f, timeScale + 0.015f);
        if (IsKeyPressed(KEY_COMMA)) timeScale = std::max(0.005f, timeScale * 0.5f);
        if (IsKeyPressed(KEY_PERIOD)) timeScale = std::min(0.40f, timeScale * 2.0f);
        if (IsKeyPressed(KEY_HOME)) masterProgress = 0.0f;
        if (IsKeyPressed(KEY_END)) masterProgress = 1.0f;

        if (!paused) {
            const float dt = GetFrameTime();
            sceneTime += dt;
            masterProgress += dt * timeScale;
            if (masterProgress > 1.0f) masterProgress -= 1.0f;
        }

        UpdatePlanets(&planets, sceneTime * 0.65f);
        AnchorPlanetsToSpacetime(&planets, sceneTime);
        const Planet& earth = FindPlanet(planets, "Earth");
        const Planet& mercury = FindPlanet(planets, "Mercury");
        const Planet& venus = FindPlanet(planets, "Venus");
        const Planet& mars = FindPlanet(planets, "Mars");
        const Planet& jupiter = FindPlanet(planets, "Jupiter");
        const Planet& saturn = FindPlanet(planets, "Saturn");
        const Planet& uranus = FindPlanet(planets, "Uranus");
        const Planet& neptune = FindPlanet(planets, "Neptune");
        const Vector3 moonPos = MakeMoonPosition(earth, sceneTime * 0.75f);

        const MissionSample artemis = EvaluateArtemisMission(masterProgress, earth, moonPos);
        const MissionSample voyager1 = EvaluateVoyager1Mission(masterProgress, earth, jupiter, saturn);
        const MissionSample voyager2 = EvaluateVoyager2Mission(masterProgress, earth, jupiter, saturn, uranus, neptune);

        Vector3 desiredTarget = {0.0f, 0.0f, 0.0f};
        if (focus == FocusMode::kArtemis) desiredTarget = Vector3Lerp(earth.pos, moonPos, 0.22f);
        if (focus == FocusMode::kVoyager1) desiredTarget = voyager1.pos;
        if (focus == FocusMode::kVoyager2) desiredTarget = voyager2.pos;
        UpdateOrbitCameraDragOnly(&camera, &orbit, desiredTarget);

        const bool artemisFlyby = masterProgress >= 0.52f && masterProgress < 0.66f;
        const bool voyager1Jupiter = masterProgress >= 0.42f && masterProgress < 0.56f;
        const bool voyager1Saturn = masterProgress >= 0.80f && masterProgress < 0.90f;
        const bool voyager2Jupiter = masterProgress >= 0.28f && masterProgress < 0.36f;
        const bool voyager2Saturn = masterProgress >= 0.52f && masterProgress < 0.60f;
        const bool voyager2Uranus = masterProgress >= 0.74f && masterProgress < 0.82f;
        const bool voyager2Neptune = masterProgress >= 0.92f && masterProgress < 0.97f;
        const float artemisDay = 10.5f * masterProgress;
        const float voyager1Year = 1977.0f + (2012.0f - 1977.0f) * masterProgress;
        const float voyager2Year = 1977.0f + (2018.0f - 1977.0f) * masterProgress;

        BeginDrawing();
        ClearBackground(Color{4, 7, 14, 255});
        DrawRectangleGradientV(0, 0, kScreenWidth, kScreenHeight, Color{5, 10, 22, 255}, Color{1, 3, 10, 255});
        DrawRectangleGradientH(0, 0, kScreenWidth, kScreenHeight, Fade(Color{14, 32, 54, 255}, 0.18f), BLANK);

        BeginMode3D(camera);

        for (const BackdropStar& star : stars) {
            const float pulse = 0.68f + 0.32f * std::sin(sceneTime * star.twinkle);
            DrawSphere(star.pos, star.size * pulse, Fade(WHITE, 0.16f + 0.60f * pulse));
        }

        DrawSphere({0.0f, 0.0f, 0.0f}, 2.5f, Color{255, 208, 112, 255});
        for (int i = 0; i < 5; ++i) {
            DrawSphereWires({0.0f, 0.0f, 0.0f}, 2.9f + 0.9f * i, 16, 16, Fade(Color{255, 170, 62, 255}, 0.05f));
        }

        DrawSpacetimeGrid(planets, sceneTime, kSceneExtent - 4.0f);
        for (const Planet& planet : planets) DrawProjectedOrbit(planet, planets, sceneTime);

        for (const Planet& planet : planets) DrawPlanetVisual(planet, sceneTime);

        DrawSphere(moonPos, 0.26f, Color{208, 212, 220, 255});
        DrawSphereWires(earth.pos, 3.4f, 18, 18, Fade(SKYBLUE, 0.14f));

        DrawMissionTrail(
            [&](float t) { return EvaluateArtemisMission(t, earth, moonPos); },
            masterProgress,
            Color{120, 220, 255, 255},
            Color{120, 220, 255, 255}
        );
        DrawMissionTrail(
            [&](float t) { return EvaluateVoyager1Mission(t, earth, jupiter, saturn); },
            masterProgress,
            Color{255, 216, 112, 255},
            Color{255, 216, 112, 255}
        );
        DrawMissionTrail(
            [&](float t) { return EvaluateVoyager2Mission(t, earth, jupiter, saturn, uranus, neptune); },
            masterProgress,
            Color{112, 255, 204, 255},
            Color{112, 255, 204, 255}
        );

        if (artemisFlyby) DrawEncounterPulse(moonPos, 1.7f, sceneTime, Color{122, 220, 255, 255});
        if (voyager1Jupiter || voyager2Jupiter) DrawEncounterPulse(jupiter.pos, 4.2f, sceneTime, Color{255, 196, 116, 255});
        if (voyager1Saturn || voyager2Saturn) DrawEncounterPulse(saturn.pos, 4.5f, sceneTime, Color{255, 216, 132, 255});
        if (voyager2Uranus) DrawEncounterPulse(uranus.pos, 4.0f, sceneTime, Color{152, 238, 240, 255});
        if (voyager2Neptune) DrawEncounterPulse(neptune.pos, 4.0f, sceneTime, Color{112, 168, 255, 255});

        DrawOrionCraft(artemis.pos, artemis.tangent, Color{212, 242, 255, 255});
        DrawVoyagerProbe(voyager1.pos, voyager1.tangent, Color{255, 214, 126, 255});
        DrawVoyagerProbe(voyager2.pos, voyager2.tangent, Color{132, 255, 214, 255});

        DrawLine3D(earth.pos, moonPos, Fade(Color{124, 170, 224, 255}, 0.08f));

        EndMode3D();

        DrawText("Artemis II + Voyager 1 & 2 Mission Observatory", 20, 18, 30, Color{236, 242, 248, 255});
        DrawText("Mouse drag: orbit | wheel: zoom | 0-3 focus | Tab cycle focus | Left/Right scrub | Q/E fine scrub | [ ] / , . rate | Space pause | R reset", 20, 54, 18, Color{166, 184, 210, 255});

        DrawRectangleRounded(Rectangle{16, 112, 428, 228}, 0.08f, 12, Fade(Color{10, 16, 28, 255}, 0.92f));
        DrawRectangleRoundedLinesEx(Rectangle{16, 112, 428, 228}, 0.08f, 12, 1.0f, Fade(Color{74, 104, 144, 255}, 0.85f));
        DrawText(TextFormat("Focus: %s", FocusName(focus)), 32, 128, 22, Color{226, 234, 246, 255});
        DrawText(TextFormat("Mission clock: %5.1f%%", masterProgress * 100.0f), 32, 156, 20, Color{198, 212, 234, 255});
        DrawText(TextFormat("Playback rate: %.2fx%s", timeScale / 0.08f, paused ? "  [PAUSED]" : ""), 32, 182, 20, Color{198, 212, 234, 255});
        DrawText(TextFormat("Artemis mission day: %4.1f / 10.5", artemisDay), 32, 208, 20, Color{198, 212, 234, 255});
        DrawText("Artemis II", 32, 236, 20, Color{132, 224, 255, 255});
        DrawText(artemis.phase, 154, 236, 20, Color{222, 232, 244, 255});
        DrawText("Voyager 1", 32, 264, 20, Color{255, 214, 126, 255});
        DrawText(voyager1.phase, 154, 264, 20, Color{222, 232, 244, 255});
        DrawText("Voyager 2", 32, 292, 20, Color{132, 255, 214, 255});
        DrawText(voyager2.phase, 154, 292, 20, Color{222, 232, 244, 255});
        DrawText("Visual note: geometry is scaled for readability while preserving mission relationships.", 32, 320, 18, Color{152, 170, 194, 255});

        DrawRectangleRounded(Rectangle{1094, 112, 390, 292}, 0.08f, 12, Fade(Color{8, 14, 24, 255}, 0.92f));
        DrawRectangleRoundedLinesEx(Rectangle{1094, 112, 390, 292}, 0.08f, 12, 1.0f, Fade(Color{74, 104, 144, 255}, 0.85f));
        DrawText("Mission architecture", 1114, 130, 24, Color{230, 236, 246, 255});
        DrawText("Artemis II", 1114, 170, 20, Color{132, 224, 255, 255});
        DrawText("Crewed translunar loop, lunar flyby, free-return Earth recovery.", 1226, 170, 18, Color{198, 212, 234, 255});
        DrawText("Voyager 1", 1114, 206, 20, Color{255, 214, 126, 255});
        DrawText("Earth -> Jupiter assist -> Saturn/Titan shaping -> interstellar northward escape.", 1226, 206, 18, Color{198, 212, 234, 255});
        DrawText("Voyager 2", 1114, 242, 20, Color{132, 255, 214, 255});
        DrawText("Earth -> Jupiter -> Saturn -> Uranus -> Neptune -> interstellar southward cruise.", 1226, 242, 18, Color{198, 212, 234, 255});
        DrawText("Encounter pulses highlight gravity-assist windows and major geometry changes.", 1114, 292, 18, Color{152, 170, 194, 255});
        DrawText("Spacetime grid is a stylized gravity well sheet, not a relativistic simulation.", 1114, 320, 18, Color{152, 170, 194, 255});
        DrawText("Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune are all in the scene.", 1114, 348, 18, Color{152, 170, 194, 255});

        DrawPlanetLabel(camera, mercury.pos, "Mercury", Color{210, 202, 188, 255});
        DrawPlanetLabel(camera, venus.pos, "Venus", Color{238, 214, 164, 255});
        DrawPlanetLabel(camera, earth.pos, "Earth", Color{178, 214, 255, 255});
        DrawPlanetLabel(camera, moonPos, "Moon", Color{214, 220, 226, 255});
        DrawPlanetLabel(camera, mars.pos, "Mars", Color{246, 168, 122, 255});
        DrawPlanetLabel(camera, jupiter.pos, "Jupiter", Color{236, 204, 170, 255});
        DrawPlanetLabel(camera, saturn.pos, "Saturn", Color{236, 218, 154, 255});
        DrawPlanetLabel(camera, uranus.pos, "Uranus", Color{178, 238, 244, 255});
        DrawPlanetLabel(camera, neptune.pos, "Neptune", Color{138, 182, 255, 255});
        DrawPlanetLabel(camera, artemis.pos, "Artemis II", Color{132, 224, 255, 255});
        DrawPlanetLabel(camera, voyager1.pos, "Voyager 1", Color{255, 214, 126, 255});
        DrawPlanetLabel(camera, voyager2.pos, "Voyager 2", Color{132, 255, 214, 255});

        DrawTimelineBar(masterProgress, artemisDay, voyager1Year, voyager2Year);
        DrawFPS(20, kScreenHeight - 34);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
