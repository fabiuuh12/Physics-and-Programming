#include "../vision/hand_tracking_scene_shared.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>
#include <vector>

using namespace astro_hand;

namespace {

constexpr int kScreenWidth = 1480;
constexpr int kScreenHeight = 920;
constexpr float kPi = 3.14159265358979323846f;

float Hash01(unsigned int n) {
    n ^= 2747636419u;
    n *= 2654435769u;
    n ^= n >> 16;
    n *= 2654435769u;
    n ^= n >> 16;
    n *= 2654435769u;
    return static_cast<float>(n & 0x00FFFFFFu) / static_cast<float>(0x01000000u);
}

struct MouthState {
    Vector3 center{};
    float radius = 1.4f;
    float targetX = 0.0f;
    float targetRadius = 1.4f;
    bool centerControlled = false;
    bool centerLatched = false;
    Vector3 gripStart{};
    float centerStartX = 0.0f;
    float radiusStart = 1.4f;
};

struct FlowParticle {
    float x = 0.0f;
    float theta = 0.0f;
    float radial = 0.5f;
    float speed = 1.0f;
    float glow = 1.0f;
    float phase = 0.0f;
};

struct TransitParticle {
    bool active = false;
    int stage = 0;
    int direction = 1;
    Vector3 pos{};
    Vector3 vel{};
    float x = 0.0f;
    float theta = 0.0f;
    float radial = 0.4f;
    float speed = 1.0f;
    float size = 0.08f;
    Color color{};
};

struct RimMote {
    int side = -1;
    float theta = 0.0f;
    float ringFraction = 0.0f;
    float speed = 1.0f;
    float lift = 0.0f;
    float phase = 0.0f;
    float size = 0.05f;
};

struct DistantStar {
    Vector3 position{};
    float size = 0.1f;
    float tint = 0.0f;
    int side = -1;
};

struct NebulaBlob {
    Vector3 position{};
    float radius = 2.0f;
    float alpha = 0.05f;
    float tint = 0.0f;
    int side = -1;
};

struct EnvironmentPalette {
    Color mouth{};
    Color inner{};
    Color fog{};
    Color star{};
    Color nebula{};
    Color accent{};
};

float RandomFloat(float minValue, float maxValue) {
    return minValue + (maxValue - minValue) * (static_cast<float>(GetRandomValue(0, 10000)) / 10000.0f);
}

Color LerpColor(Color a, Color b, float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return Color{
        static_cast<unsigned char>(a.r + (b.r - a.r) * t),
        static_cast<unsigned char>(a.g + (b.g - a.g) * t),
        static_cast<unsigned char>(a.b + (b.b - a.b) * t),
        static_cast<unsigned char>(a.a + (b.a - a.a) * t),
    };
}

float HandPalmScale(const HandGeometry& g) {
    return std::max(0.18f, Vector3Distance(g.landmarks[0], g.landmarks[9]));
}

float MouthRadiusAt(float x, float leftX, float rightX, float leftRadius, float rightRadius) {
    const float span = std::max(0.25f, rightX - leftX);
    const float t = std::clamp((x - leftX) / span, 0.0f, 1.0f);
    const float edge = leftRadius + (rightRadius - leftRadius) * t;
    const float throatDip = 0.58f * std::sin(t * kPi);
    return std::max(0.32f, edge - throatDip * std::min(leftRadius, rightRadius));
}

float VisualMouthRadiusAt(float x, const MouthState& left, const MouthState& right, float time) {
    const float span = std::max(0.25f, right.center.x - left.center.x);
    const float u = std::clamp((x - left.center.x) / span, 0.0f, 1.0f);
    const float base = MouthRadiusAt(x, left.center.x, right.center.x, left.radius, right.radius);
    const float edgeBoost = 0.35f + 0.65f * std::pow(std::fabs(u - 0.5f) * 2.0f, 0.8f);
    const float breathing = base * 0.055f * edgeBoost * std::sin(time * 1.55f + u * 5.5f);
    const float ripple = base * 0.032f * std::sin(time * 2.8f + u * 11.0f);
    return std::max(0.32f, base + breathing + ripple);
}

Vector3 WormholePoint(float x, float theta, float radialFraction, const MouthState& left, const MouthState& right) {
    const float r = MouthRadiusAt(x, left.center.x, right.center.x, left.radius, right.radius) * radialFraction;
    const float y = 1.8f + r * std::cos(theta);
    const float z = r * std::sin(theta);
    const float swirl = 0.10f * std::sin(theta * 3.0f + x * 0.75f);
    return {x, y + swirl, z};
}

Vector3 VisualWormholePoint(float x, float theta, float radialFraction, const MouthState& left, const MouthState& right, float time) {
    const float span = std::max(0.25f, right.center.x - left.center.x);
    const float u = std::clamp((x - left.center.x) / span, 0.0f, 1.0f);
    const float r = VisualMouthRadiusAt(x, left, right, time) * radialFraction;
    const float twist = theta + time * (0.60f + 0.22f * radialFraction) + u * 5.8f;
    const float y = 1.8f + r * std::cos(twist);
    const float z = r * std::sin(twist);
    const float drift = 0.08f * std::sin(time * 1.9f + u * 8.5f + theta * 2.4f);
    return {x, y + drift, z + drift * 0.45f};
}

Color TubeColorX(float x, const MouthState& left, const MouthState& right, const EnvironmentPalette& nearPalette, const EnvironmentPalette& farPalette, float pulse) {
    const float span = std::max(0.25f, right.center.x - left.center.x);
    const float u = std::clamp((x - left.center.x) / span, 0.0f, 1.0f);
    Color base = LerpColor(nearPalette.inner, farPalette.inner, u);
    return LerpColor(base, Color{255, 255, 255, 255}, pulse * 0.10f);
}

Vector3 DistortAroundMouthX(Vector3 p, const MouthState& mouth, float distortion) {
    const float dy = p.y - mouth.center.y;
    const float dz = p.z - mouth.center.z;
    const float planeDistance = std::abs(p.x - mouth.center.x);
    const float radial = std::sqrt(dy * dy + dz * dz);
    constexpr float influence = 10.5f;
    if (planeDistance > 7.0f || radial > influence) return p;

    float falloff = (1.0f - radial / influence) * std::exp(-planeDistance * 0.55f);
    falloff = std::max(falloff, 0.0f);
    const float twist = distortion * 0.46f * falloff;
    const float c = std::cos(twist);
    const float s = std::sin(twist);
    const float y2 = dy * c - dz * s;
    const float z2 = dy * s + dz * c;
    const float pull = 1.0f - distortion * 0.16f * falloff;
    return {
        p.x - (p.x - mouth.center.x) * distortion * 0.06f * falloff,
        mouth.center.y + y2 * pull,
        mouth.center.z + z2 * pull,
    };
}

void InitializeParticles(std::vector<FlowParticle>& particles) {
    particles.clear();
    particles.reserve(24);
    for (int i = 0; i < 24; ++i) {
        particles.push_back(FlowParticle{
            -3.0f + 6.0f * static_cast<float>(GetRandomValue(0, 10000)) / 10000.0f,
            2.0f * kPi * static_cast<float>(GetRandomValue(0, 10000)) / 10000.0f,
            0.18f + 0.72f * static_cast<float>(GetRandomValue(0, 10000)) / 10000.0f,
            1.2f + 2.0f * static_cast<float>(GetRandomValue(0, 10000)) / 10000.0f,
            0.45f + 0.55f * static_cast<float>(GetRandomValue(0, 10000)) / 10000.0f,
            2.0f * kPi * static_cast<float>(GetRandomValue(0, 10000)) / 10000.0f,
        });
    }
}

void InitializeTransit(std::vector<TransitParticle>& particles) {
    particles.assign(36, TransitParticle{});
}

void InitializeRimMotes(std::vector<RimMote>& motes) {
    motes.clear();
    motes.reserve(180);
    for (int side : {-1, 1}) {
        for (int i = 0; i < 90; ++i) {
            motes.push_back(RimMote{
                side,
                RandomFloat(0.0f, 2.0f * kPi),
                RandomFloat(0.15f, 1.0f),
                RandomFloat(0.6f, 1.9f),
                RandomFloat(-0.3f, 0.3f),
                RandomFloat(0.0f, 2.0f * kPi),
                RandomFloat(0.03f, 0.09f),
            });
        }
    }
}

void InitializeStars(std::vector<DistantStar>& stars) {
    stars.clear();
    stars.reserve(170);
    for (int side : {-1, 1}) {
        for (int i = 0; i < 85; ++i) {
            const float theta = RandomFloat(0.0f, 2.0f * kPi);
            const float radial = RandomFloat(4.0f, 20.0f);
            const float x = side * RandomFloat(8.0f, 24.0f);
            stars.push_back(DistantStar{
                {x, 1.8f + radial * std::cos(theta), radial * std::sin(theta) * RandomFloat(0.45f, 1.1f)},
                RandomFloat(0.05f, 0.17f),
                RandomFloat(0.0f, 1.0f),
                side,
            });
        }
    }
}

void InitializeNebula(std::vector<NebulaBlob>& blobs) {
    blobs.clear();
    blobs.reserve(24);
    for (int side : {-1, 1}) {
        for (int i = 0; i < 12; ++i) {
            const float theta = RandomFloat(0.0f, 2.0f * kPi);
            const float radial = RandomFloat(3.0f, 12.0f);
            const float x = side * RandomFloat(9.0f, 18.0f);
            blobs.push_back(NebulaBlob{
                {x, 1.8f + radial * std::cos(theta), radial * std::sin(theta)},
                RandomFloat(1.8f, 4.6f),
                RandomFloat(0.03f, 0.10f),
                RandomFloat(0.0f, 1.0f),
                side,
            });
        }
    }
}

float MouthDepthRadius(const MouthState& mouth, float u, float t, bool rightSide, float innerScale) {
    const float taper = 1.06f - 0.62f * std::pow(u, 0.82f);
    const float ripple = 0.035f * std::sin(t * (2.1f + 0.18f * innerScale) + u * 12.0f + (rightSide ? 0.9f : 0.0f));
    const float swirl = 0.018f * std::sin(t * 3.3f + u * 20.0f + innerScale * 4.0f);
    return std::max(0.14f, mouth.radius * innerScale * (taper + ripple + swirl));
}

Vector3 MouthDepthPoint(const MouthState& mouth, bool rightSide, float u, float theta, float t, float innerScale) {
    const float direction = rightSide ? -1.0f : 1.0f;
    const float x = mouth.center.x + direction * (0.04f + 1.18f * u);
    const float radius = MouthDepthRadius(mouth, u, t, rightSide, innerScale);
    const float wobble = theta + t * (1.15f + 0.14f * innerScale) + u * 3.8f;
    const float y = mouth.center.y + radius * std::cos(wobble);
    const float z = mouth.center.z + radius * std::sin(wobble);
    return {x, y, z};
}

void DrawDeepSpaceField(float t) {
    for (int i = 0; i < 260; ++i) {
        const float rx = Hash01(17u + static_cast<unsigned int>(i) * 92821u);
        const float ry = Hash01(31u + static_cast<unsigned int>(i) * 68917u);
        const float rz = Hash01(47u + static_cast<unsigned int>(i) * 42131u);
        const float tw = Hash01(53u + static_cast<unsigned int>(i) * 53197u);
        const float side = (i % 2 == 0) ? -1.0f : 1.0f;
        Vector3 p{
            side * (8.0f + 18.0f * rx),
            -3.0f + 10.0f * ry,
            -12.0f + 24.0f * rz,
        };
        if (std::fabs(p.z) < 3.0f) p.z += (p.z < 0.0f ? -3.0f : 3.0f);
        const float pulse = 0.55f + 0.45f * std::sin(t * (0.4f + tw) + rx * 9.0f);
        const float size = 0.02f + 0.08f * tw;
        const Color c = (i % 5 == 0) ? Color{196, 220, 255, 255} : Color{244, 246, 255, 255};
        DrawSphere(p, size, Fade(c, 0.35f + 0.55f * pulse));
    }

    for (int i = 0; i < 26; ++i) {
        const float rx = Hash01(71u + static_cast<unsigned int>(i) * 11113u);
        const float ry = Hash01(97u + static_cast<unsigned int>(i) * 23131u);
        const float rz = Hash01(131u + static_cast<unsigned int>(i) * 39119u);
        const float scale = Hash01(149u + static_cast<unsigned int>(i) * 51787u);
        const float side = (i % 2 == 0) ? -1.0f : 1.0f;
        const Color nebula = (i % 3 == 0) ? Color{88, 128, 255, 255} : Color{168, 212, 255, 255};
        DrawSphere({side * (9.0f + 11.0f * rx), 0.5f + 5.0f * ry, -10.0f + 20.0f * rz},
                   0.35f + 0.75f * scale,
                   Fade(nebula, 0.035f));
    }
}

void DrawWarpedRingX(float x, float baseRadius, Color color, float wobble, float time, float phase) {
    const int segments = 96;
    for (int i = 0; i < segments; ++i) {
        const float a0 = 2.0f * kPi * static_cast<float>(i) / static_cast<float>(segments);
        const float a1 = 2.0f * kPi * static_cast<float>(i + 1) / static_cast<float>(segments);
        const float r0 = baseRadius + wobble * std::sin(a0 * 6.0f + time * 2.1f + phase);
        const float r1 = baseRadius + wobble * std::sin(a1 * 6.0f + time * 2.1f + phase);
        DrawLine3D({x, 1.8f + r0 * std::cos(a0), r0 * std::sin(a0)},
                   {x, 1.8f + r1 * std::cos(a1), r1 * std::sin(a1)},
                   color);
    }
}

void DrawMouthDepth(const MouthState& mouth, bool rightSide, float t, const EnvironmentPalette& palette) {
    const Color lip = palette.mouth;
    const Color glow = palette.accent;
    const int depthSlices = 22;
    const int segs = 54;
    for (int i = 0; i < depthSlices - 1; ++i) {
        const float u0 = static_cast<float>(i) / static_cast<float>(depthSlices - 1);
        const float u1 = static_cast<float>(i + 1) / static_cast<float>(depthSlices - 1);
        for (int j = 0; j < segs; ++j) {
            const float a0 = 2.0f * kPi * static_cast<float>(j) / static_cast<float>(segs);
            const float a1 = 2.0f * kPi * static_cast<float>(j + 1) / static_cast<float>(segs);

            const Vector3 p00 = MouthDepthPoint(mouth, rightSide, u0, a0, t, 1.0f);
            const Vector3 p01 = MouthDepthPoint(mouth, rightSide, u0, a1, t, 1.0f);
            const Vector3 p10 = MouthDepthPoint(mouth, rightSide, u1, a0, t, 1.0f);
            const Vector3 p11 = MouthDepthPoint(mouth, rightSide, u1, a1, t, 1.0f);

            const Vector3 i00 = MouthDepthPoint(mouth, rightSide, u0, a0, t, 0.72f);
            const Vector3 i01 = MouthDepthPoint(mouth, rightSide, u0, a1, t, 0.72f);
            const Vector3 i10 = MouthDepthPoint(mouth, rightSide, u1, a0, t, 0.72f);
            const Vector3 i11 = MouthDepthPoint(mouth, rightSide, u1, a1, t, 0.72f);

            const float fade = 1.0f - u0;
            const float shimmer = 0.62f + 0.38f * std::sin(t * 2.7f + u0 * 6.0f + a0 * 3.0f);
            const Color shell = Color{
                static_cast<unsigned char>(lip.r * (0.62f + 0.22f * shimmer)),
                static_cast<unsigned char>(lip.g * (0.64f + 0.18f * shimmer)),
                static_cast<unsigned char>(lip.b * (0.70f + 0.16f * shimmer)),
                static_cast<unsigned char>(22 + 34 * fade),
            };
            const Color inner = Color{
                static_cast<unsigned char>(glow.r * (0.72f + 0.20f * shimmer)),
                static_cast<unsigned char>(glow.g * (0.72f + 0.20f * shimmer)),
                static_cast<unsigned char>(glow.b * (0.82f + 0.18f * shimmer)),
                static_cast<unsigned char>(12 + 26 * fade),
            };
            DrawTriangle3D(p00, p10, p01, shell);
            DrawTriangle3D(p01, p10, p11, shell);
            DrawTriangle3D(i00, i10, i01, inner);
            DrawTriangle3D(i01, i10, i11, inner);
        }
    }

    const float direction = rightSide ? -1.0f : 1.0f;
    for (int i = 0; i < 10; ++i) {
        const float u0 = static_cast<float>(i) / 10.0f;
        const float u1 = static_cast<float>(i + 1) / 10.0f;
        const float x0 = mouth.center.x + direction * (0.02f + 1.05f * u0);
        const float x1 = mouth.center.x + direction * (0.10f + 1.05f * u1);
        const float r0 = MouthDepthRadius(mouth, u0 * 0.92f, t, rightSide, 0.54f);
        const float r1 = MouthDepthRadius(mouth, u1 * 0.92f, t, rightSide, 0.24f);
        DrawCylinderEx({x0, mouth.center.y, mouth.center.z},
                       {x1, mouth.center.y, mouth.center.z},
                       r0,
                       std::max(0.08f, r1),
                       44,
                       Color{2, 3, 7, 235});
    }

    for (int i = 0; i < 4; ++i) {
        const float u = static_cast<float>(i) / 3.0f;
        const float x = MouthDepthPoint(mouth, rightSide, 0.18f + u * 0.24f, 0.0f, t, 1.0f).x;
        const float radius = MouthDepthRadius(mouth, 0.18f + u * 0.24f, t, rightSide, 0.84f);
        DrawWarpedRingX(x, radius, Fade(lip, 0.34f - 0.05f * static_cast<float>(i)), 0.05f + 0.03f * u, t * 1.3f, i * 0.8f + (rightSide ? 0.6f : 0.0f));
    }

    for (int ribbon = 0; ribbon < 3; ++ribbon) {
        Vector3 prev = MouthDepthPoint(mouth, rightSide, 0.02f, 0.7f * static_cast<float>(ribbon), t, 0.66f + 0.05f * static_cast<float>(ribbon));
        for (int i = 1; i <= 28; ++i) {
            const float u = static_cast<float>(i) / 28.0f;
            const float theta = t * (1.8f + 0.25f * ribbon) + u * 5.8f + ribbon * 1.7f;
            const Vector3 curr = MouthDepthPoint(mouth, rightSide, u * 0.76f, theta, t, 0.58f + 0.06f * static_cast<float>(ribbon));
            DrawLine3D(prev, curr, Fade(glow, 0.10f + 0.02f * ribbon));
            prev = curr;
        }
    }

    const Vector3 sink = MouthDepthPoint(mouth, rightSide, 0.94f, t * 2.2f, t, 0.28f);
    DrawSphere(sink,
               mouth.radius * 0.10f,
               Fade(glow, 0.08f));
}

void DrawMouth(const MouthState& mouth, bool rightSide, float t, const EnvironmentPalette& palette, float pulseBoost) {
    const Color rim = palette.mouth;
    const Color glow = palette.accent;
    const float pulse = 0.92f + 0.08f * std::sin(t * 2.0f + (rightSide ? 1.2f : 0.4f)) + 0.10f * pulseBoost;

    DrawCylinderEx(
        {mouth.center.x - 0.12f, mouth.center.y, mouth.center.z},
        {mouth.center.x + 0.12f, mouth.center.y, mouth.center.z},
        mouth.radius * 1.04f,
        mouth.radius * 1.04f,
        36,
        Fade(rim, 0.12f));

    DrawCylinderWiresEx(
        {mouth.center.x - 0.02f, mouth.center.y, mouth.center.z},
        {mouth.center.x + 0.02f, mouth.center.y, mouth.center.z},
        mouth.radius * pulse,
        mouth.radius * pulse,
        36,
        Fade(rim, 0.85f));

    DrawMouthDepth(mouth, rightSide, t, palette);

    DrawWarpedRingX(mouth.center.x, mouth.radius * 1.10f, Fade(rim, 0.74f + pulseBoost * 0.12f), 0.10f + mouth.radius * 0.05f + pulseBoost * 0.06f, t, rightSide ? 0.9f : 0.1f);
    DrawWarpedRingX(mouth.center.x, mouth.radius * 1.34f, Fade(glow, 0.18f), 0.14f + mouth.radius * 0.06f + pulseBoost * 0.06f, t, rightSide ? 2.4f : 1.4f);
    DrawWarpedRingX(mouth.center.x, mouth.radius * 1.58f, Fade(palette.accent, 0.08f), 0.18f + mouth.radius * 0.08f + pulseBoost * 0.08f, t, rightSide ? 3.8f : 2.7f);

    const int petals = 24;
    for (int i = 0; i < petals; ++i) {
        const float a = 2.0f * kPi * static_cast<float>(i) / static_cast<float>(petals);
        const float wave = 0.10f * std::sin(t * 2.8f + a * 4.0f);
        const Vector3 p0 = {mouth.center.x, mouth.center.y + std::cos(a) * mouth.radius * (1.03f + wave), mouth.center.z + std::sin(a) * mouth.radius * (1.03f + wave)};
        const Vector3 p1 = {mouth.center.x, mouth.center.y + std::cos(a) * mouth.radius * (1.22f + wave * 0.6f), mouth.center.z + std::sin(a) * mouth.radius * (1.22f + wave * 0.6f)};
        DrawLine3D(p0, p1, Fade(glow, 0.42f));
    }
}

void DrawTunnelSurface(const MouthState& left, const MouthState& right, float t, const EnvironmentPalette& nearPalette, const EnvironmentPalette& farPalette, float pulseValue, float swirlIntensity) {
    const int slices = 68;
    const int segments = 56;
    for (int i = 0; i < slices - 1; ++i) {
        const float x0 = left.center.x + (right.center.x - left.center.x) * static_cast<float>(i) / static_cast<float>(slices - 1);
        const float x1 = left.center.x + (right.center.x - left.center.x) * static_cast<float>(i + 1) / static_cast<float>(slices - 1);
        for (int j = 0; j < segments; ++j) {
            const float a0 = 2.0f * kPi * static_cast<float>(j) / static_cast<float>(segments);
            const float a1 = 2.0f * kPi * static_cast<float>(j + 1) / static_cast<float>(segments);
            const Vector3 p00 = VisualWormholePoint(x0, a0, 1.0f, left, right, t);
            const Vector3 p10 = VisualWormholePoint(x1, a0, 1.0f, left, right, t);
            const Vector3 p01 = VisualWormholePoint(x0, a1, 1.0f, left, right, t);
            const Vector3 p11 = VisualWormholePoint(x1, a1, 1.0f, left, right, t);
            const Vector3 i00 = VisualWormholePoint(x0, a0, 0.82f, left, right, t);
            const Vector3 i10 = VisualWormholePoint(x1, a0, 0.82f, left, right, t);
            const Vector3 i01 = VisualWormholePoint(x0, a1, 0.82f, left, right, t);
            const Vector3 i11 = VisualWormholePoint(x1, a1, 0.82f, left, right, t);

            const float blend = static_cast<float>(i) / static_cast<float>(slices - 1);
            const float centerGlow = 1.0f - std::fabs(blend - 0.5f) * 2.0f;
            const float shimmer = 0.30f + 0.20f * std::sin(t * 2.4f + blend * 8.0f + a0 * 2.0f);
            const float mouthFade = std::sin(std::clamp(blend, 0.0f, 1.0f) * kPi);
            Color c = TubeColorX(x0, left, right, nearPalette, farPalette, pulseValue);
            c = LerpColor(c, Color{255, 255, 255, 255}, 0.10f * shimmer + 0.10f * centerGlow);
            c.a = static_cast<unsigned char>((18 + 52 * centerGlow + 20 * pulseValue) * (0.20f + 0.80f * mouthFade));
            Color inner = TubeColorX(x0, left, right, nearPalette, farPalette, pulseValue);
            inner = LerpColor(inner, Color{255, 255, 255, 255}, 0.18f + 0.12f * shimmer);
            inner.a = static_cast<unsigned char>((8 + 24 * centerGlow + 12 * pulseValue) * (0.10f + 0.90f * mouthFade));
            DrawTriangle3D(p00, p10, p01, c);
            DrawTriangle3D(p01, p10, p11, c);
            DrawTriangle3D(i00, i10, i01, inner);
            DrawTriangle3D(i01, i10, i11, inner);
        }
    }

    for (int ring = 0; ring < 16; ++ring) {
        const float u = static_cast<float>(ring) / 15.0f;
        const float x = left.center.x + (right.center.x - left.center.x) * u;
        const float radius = VisualMouthRadiusAt(x, left, right, t);
        const float ringGlow = 0.08f + 0.06f * std::sin(t * 1.7f + u * 4.8f);
        DrawWarpedRingX(x, radius, Fade(TubeColorX(x, left, right, nearPalette, farPalette, pulseValue), ringGlow + 0.04f * pulseValue), 0.08f + swirlIntensity * 0.03f, t * 0.8f, u * 3.0f);
    }

    for (int ribbon = 0; ribbon < 4; ++ribbon) {
        Vector3 prev = VisualWormholePoint(left.center.x, 0.0f, 0.50f + 0.08f * static_cast<float>(ribbon), left, right, t);
        for (int i = 1; i <= 80; ++i) {
            const float u = static_cast<float>(i) / 80.0f;
            const float x = left.center.x + (right.center.x - left.center.x) * u;
            const float theta = u * 7.6f + t * (0.85f + 0.16f * static_cast<float>(ribbon)) + ribbon * 0.95f;
            const Vector3 curr = VisualWormholePoint(x, theta, 0.50f + 0.08f * static_cast<float>(ribbon), left, right, t);
            DrawLine3D(prev, curr, Fade(LerpColor(nearPalette.accent, farPalette.accent, u), 0.05f + 0.02f * static_cast<float>(ribbon)));
            prev = curr;
        }
    }
}

void DrawThroatCore(const MouthState& left, const MouthState& right, float t, const EnvironmentPalette& nearPalette, const EnvironmentPalette& farPalette, float pulseValue) {
    for (int i = 0; i < 44; ++i) {
        const float u = static_cast<float>(i) / 43.0f;
        const float x = left.center.x + (right.center.x - left.center.x) * u;
        const float r = MouthRadiusAt(x, left.center.x, right.center.x, left.radius, right.radius);
        const float pulse = 0.45f + 0.55f * std::sin(t * 3.0f + u * 10.0f);
        DrawSphere({x, 1.8f, 0.0f}, r * (0.08f + 0.02f * pulse), Fade(TubeColorX(x, left, right, nearPalette, farPalette, pulseValue), 0.05f + 0.09f * pulse));
    }
}

void DrawFlowParticles(std::vector<FlowParticle>& particles, const MouthState& left, const MouthState& right, float dt, float t, const EnvironmentPalette& nearPalette, const EnvironmentPalette& farPalette, float pulseValue, float swirlIntensity) {
    const float leftX = left.center.x;
    const float rightX = right.center.x;
    const float span = std::max(0.8f, rightX - leftX);

    for (FlowParticle& particle : particles) {
        particle.x += dt * particle.speed * (1.1f + 1.3f / span);
        particle.theta += dt * (1.6f + particle.glow * 1.2f);
        if (particle.x > rightX) {
            particle.x = leftX;
            particle.theta = 2.0f * kPi * static_cast<float>(GetRandomValue(0, 10000)) / 10000.0f;
            particle.radial = 0.16f + 0.78f * static_cast<float>(GetRandomValue(0, 10000)) / 10000.0f;
            particle.phase = 2.0f * kPi * static_cast<float>(GetRandomValue(0, 10000)) / 10000.0f;
        }
        const float radius = VisualMouthRadiusAt(particle.x, left, right, t);
        const float lane = radius * (0.22f + particle.radial * 0.58f + 0.06f * std::sin(t * 2.6f + particle.phase));
        const float theta = particle.theta + swirlIntensity * (particle.x - leftX) * 0.18f;
        const Vector3 p = VisualWormholePoint(particle.x, theta, lane / std::max(radius, 0.1f), left, right, t);
        const Vector3 prev = VisualWormholePoint(particle.x - dt * particle.speed * 0.55f, theta - dt * (1.2f + particle.glow), lane / std::max(radius, 0.1f), left, right, t);
        const float centerFactor = 1.0f - std::fabs(((particle.x - leftX) / std::max(span, 0.1f)) - 0.5f) * 2.0f;
        Color c = TubeColorX(particle.x, left, right, nearPalette, farPalette, pulseValue);
        c = LerpColor(c, Color{255, 255, 255, 255}, particle.glow * 0.25f + centerFactor * 0.22f);
        const float alpha = 0.16f + 0.18f * particle.glow;
        DrawLine3D(prev, p, Fade(c, 0.26f));
        DrawSphere(p, 0.03f + 0.018f * particle.glow + 0.01f * centerFactor, Fade(c, alpha));
    }
}

void UpdateAndDrawTransitParticles(std::vector<TransitParticle>& particles, const MouthState& left, const MouthState& right, float dt) {
    for (TransitParticle& particle : particles) {
        if (!particle.active) continue;
        if (particle.stage == 0) {
            const MouthState& source = particle.direction > 0 ? left : right;
            const Vector3 target = {
                source.center.x,
                source.center.y + 0.08f * source.radius,
                source.center.z,
            };
            const Vector3 pull = Vector3Subtract(target, particle.pos);
            particle.vel = Vector3Add(particle.vel, Vector3Scale(pull, dt * 1.9f));
            particle.vel.y -= 1.2f * dt;
            particle.vel = Vector3Scale(particle.vel, std::exp(-0.14f * dt));
            particle.pos = Vector3Add(particle.pos, Vector3Scale(particle.vel, dt));

            const float dx = std::fabs(particle.pos.x - source.center.x);
            const float radial = std::sqrt((particle.pos.y - source.center.y) * (particle.pos.y - source.center.y) + (particle.pos.z - source.center.z) * (particle.pos.z - source.center.z));
            if (dx < 0.18f && radial < source.radius * 0.84f) {
                particle.stage = 1;
                particle.x = source.center.x;
                particle.theta = std::atan2(particle.pos.z - source.center.z, particle.pos.y - source.center.y);
                particle.radial = std::clamp(radial / std::max(source.radius, 0.2f), 0.10f, 0.78f);
                particle.speed = std::max(4.4f, particle.speed);
            }
        } else if (particle.stage == 1) {
            particle.x += dt * particle.speed * static_cast<float>(particle.direction);
            particle.theta += dt * (1.8f + particle.speed * 0.16f);
            particle.pos = WormholePoint(particle.x, particle.theta, particle.radial, left, right);
            if (particle.direction > 0 && particle.x >= right.center.x) {
                particle.stage = 2;
                particle.pos = {
                    right.center.x + 0.06f,
                    right.center.y + std::cos(particle.theta) * right.radius * particle.radial,
                    right.center.z + std::sin(particle.theta) * right.radius * particle.radial,
                };
                particle.vel = {2.4f, 0.45f, 0.30f * std::sin(particle.theta)};
            }
            if (particle.direction < 0 && particle.x <= left.center.x) {
                particle.stage = 2;
                particle.pos = {
                    left.center.x - 0.06f,
                    left.center.y + std::cos(particle.theta) * left.radius * particle.radial,
                    left.center.z + std::sin(particle.theta) * left.radius * particle.radial,
                };
                particle.vel = {-2.4f, 0.45f, 0.30f * std::sin(particle.theta)};
            }
        } else {
            particle.vel.y -= 0.9f * dt;
            particle.vel = Vector3Scale(particle.vel, std::exp(-0.08f * dt));
            particle.pos = Vector3Add(particle.pos, Vector3Scale(particle.vel, dt));
            if (std::fabs(particle.pos.x) > 10.5f || std::fabs(particle.pos.y) > 8.0f || std::fabs(particle.pos.z) > 8.0f) {
                particle.active = false;
                continue;
            }
        }

        DrawSphere(particle.pos, particle.size, particle.color);
        DrawSphere(particle.pos, particle.size * 2.0f, Fade(particle.color, 0.12f));
    }
}

bool IsFistGesture(const HandGeometry& g) {
    const float palmScale = HandPalmScale(g);
    const float thumb = Vector3Distance(g.landmarks[4], g.palmCenter) / palmScale;
    const float index = Vector3Distance(g.landmarks[8], g.palmCenter) / palmScale;
    const float middle = Vector3Distance(g.landmarks[12], g.palmCenter) / palmScale;
    const float ring = Vector3Distance(g.landmarks[16], g.palmCenter) / palmScale;
    const float pinky = Vector3Distance(g.landmarks[20], g.palmCenter) / palmScale;
    return thumb < 1.44f && index < 1.18f && middle < 1.14f && ring < 1.10f && pinky < 1.08f;
}

bool IsPinchGesture(const HandGeometry& g) {
    const float palmScale = HandPalmScale(g);
    const float pinch = Vector3Distance(g.landmarks[4], g.landmarks[8]) / palmScale;
    return pinch < 0.44f;
}

bool HandNearMouth(Vector3 point, const MouthState& mouth, float dxLimit, float radialLimit) {
    const float dx = std::fabs(point.x - mouth.center.x);
    const float radial = std::sqrt((point.y - mouth.center.y) * (point.y - mouth.center.y) + (point.z - mouth.center.z) * (point.z - mouth.center.z));
    return dx < dxLimit && radial < radialLimit;
}

Vector3 FistEmitterPoint(const HandGeometry& g) {
    return Average({g.landmarks[5], g.landmarks[9], g.landmarks[13], g.landmarks[17]});
}

bool FistNearMouth(const HandGeometry& g, const HandControlState& hand, const MouthState& mouth) {
    if (!hand.active || hand.pinched) return false;
    const Vector3 fist = FistEmitterPoint(g);
    const Vector3 palm = g.palmCenter;
    return HandNearMouth(fist, mouth, 2.6f, mouth.radius * 2.00f) || HandNearMouth(palm, mouth, 2.8f, mouth.radius * 2.10f);
}

bool FistInFireLane(const HandGeometry& g, const HandControlState& hand, const MouthState& mouth, bool rightSide) {
    if (!hand.active || hand.pinched) return false;
    const Vector3 fist = FistEmitterPoint(g);
    const float sideDx = rightSide ? (fist.x - mouth.center.x) : (mouth.center.x - fist.x);
    const float radial = std::sqrt((fist.y - mouth.center.y) * (fist.y - mouth.center.y) + (fist.z - mouth.center.z) * (fist.z - mouth.center.z));
    return sideDx > 0.6f && sideDx < 6.6f && radial < mouth.radius * 2.8f;
}

void EmitTransitBurst(std::vector<TransitParticle>& particles, Vector3 start, const MouthState& source, bool fromLeft) {
    const int direction = fromLeft ? 1 : -1;
    const Color tint = fromLeft ? Color{132, 224, 255, 255} : Color{255, 214, 132, 255};
    int emitted = 0;
    for (TransitParticle& particle : particles) {
        if (particle.active) continue;
        const float seed = static_cast<float>(GetRandomValue(0, 10000)) / 10000.0f;
        particle.active = true;
        particle.stage = 0;
        particle.direction = direction;
        particle.pos = start;
        const Vector3 toMouth = Vector3Subtract(source.center, start);
        particle.vel = Vector3Add(Vector3Scale(SafeNormalize(toMouth, {direction > 0 ? 1.0f : -1.0f, 0.0f, 0.0f}), 1.6f + 1.2f * seed), Vector3{0.0f, 0.8f + 0.4f * seed, 0.0f});
        particle.x = source.center.x + (fromLeft ? 0.05f : -0.05f);
        particle.theta = seed * 2.0f * kPi;
        particle.radial = 0.18f + 0.56f * seed;
        particle.speed = 4.0f + 2.2f * seed;
        particle.size = 0.035f + 0.03f * seed;
        particle.color = tint;
        if (++emitted >= 5) break;
    }
}

void UpdateHandControlForMouth(const HandControlState& hand, MouthState& mouth, bool rightSide, float dt) {
    mouth.centerControlled = false;
    if (!hand.active || !hand.pinched) {
        mouth.centerLatched = false;
        return;
    }

    const Vector3 pinch = hand.pinchPoint;
    const float dxNow = std::fabs(pinch.x - mouth.center.x);
    const float radialNow = std::sqrt((pinch.y - mouth.center.y) * (pinch.y - mouth.center.y) + (pinch.z - mouth.center.z) * (pinch.z - mouth.center.z));
    const bool nearMouth = dxNow < 1.8f && radialNow < mouth.radius * 1.65f;

    if (mouth.centerLatched || nearMouth) {
        if (!mouth.centerLatched) {
            mouth.gripStart = pinch;
            mouth.centerStartX = mouth.center.x;
            mouth.radiusStart = mouth.radius;
        }
        mouth.centerLatched = true;
        const float dx = pinch.x - mouth.gripStart.x;
        const float dy = pinch.y - mouth.gripStart.y;
        mouth.targetX = std::clamp(
            mouth.centerStartX + dx,
            rightSide ? 0.9f : -6.8f,
            rightSide ? 6.8f : -0.9f);
        mouth.targetRadius = std::clamp(mouth.radiusStart + dy * 1.15f, 0.75f, 2.9f);
        mouth.centerControlled = true;
    } else {
        mouth.centerLatched = false;
    }

    mouth.center.x = LerpFloat(mouth.center.x, mouth.targetX, 1.0f - std::exp(-10.0f * dt));
    mouth.radius = LerpFloat(mouth.radius, mouth.targetRadius, 1.0f - std::exp(-8.0f * dt));
}

}  // namespace

int main() {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(kScreenWidth, kScreenHeight, "Wormhole Hand Lab");
    SetTargetFPS(120);
    SetWindowMinSize(1080, 720);

    const EnvironmentPalette nearPalette{
        Color{110, 224, 255, 255},
        Color{72, 132, 255, 255},
        Color{44, 98, 170, 255},
        Color{180, 220, 255, 255},
        Color{66, 120, 214, 255},
        Color{150, 244, 255, 255},
    };
    const EnvironmentPalette farPalette{
        Color{255, 180, 118, 255},
        Color{255, 110, 78, 255},
        Color{184, 74, 54, 255},
        Color{255, 224, 196, 255},
        Color{214, 86, 72, 255},
        Color{255, 212, 140, 255},
    };

    Camera3D camera{};
    camera.target = {0.0f, 1.8f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 42.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float camYaw = 0.72f;
    float camPitch = 0.18f;
    float camDistance = 15.8f;

    HandSceneBridge bridge;
    bridge.Start();

    MouthState left{{-3.6f, 1.8f, 0.0f}, 1.42f, -3.6f, 1.42f};
    MouthState right{{3.6f, 1.8f, 0.0f}, 1.42f, 3.6f, 1.42f};
    std::vector<FlowParticle> particles;
    InitializeParticles(particles);
    std::vector<TransitParticle> transit;
    InitializeTransit(transit);
    std::vector<RimMote> rimMotes;
    InitializeRimMotes(rimMotes);
    std::vector<DistantStar> stars;
    InitializeStars(stars);
    std::vector<NebulaBlob> nebulaBlobs;
    InitializeNebula(nebulaBlobs);
    std::array<float, 2> fireCooldown = {0.0f, 0.0f};
    std::array<float, 2> mouthPulse = {0.0f, 0.0f};
    const float swirlIntensity = 1.35f;
    const float distortion = 0.85f;

    while (!WindowShouldClose()) {
        const float dt = std::max(GetFrameTime(), 1.0e-4f);
        const float t = static_cast<float>(GetTime());

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);
        bridge.Update(camera, t, dt);

        const HandControlState& leftHand = bridge.Control(false);
        const HandControlState& rightHand = bridge.Control(true);
        const bool leftFistRaw = bridge.LeftTracked() && IsFistGesture(bridge.Geometry(false));
        const bool rightFistRaw = bridge.RightTracked() && IsFistGesture(bridge.Geometry(true));
        const bool leftPinch = bridge.LeftTracked() && IsPinchGesture(bridge.Geometry(false)) && !leftFistRaw;
        const bool rightPinch = bridge.RightTracked() && IsPinchGesture(bridge.Geometry(true)) && !rightFistRaw;
        const bool leftFist = leftFistRaw;
        const bool rightFist = rightFistRaw;
        fireCooldown[0] = std::max(0.0f, fireCooldown[0] - dt);
        fireCooldown[1] = std::max(0.0f, fireCooldown[1] - dt);
        mouthPulse[0] = std::max(0.0f, mouthPulse[0] - dt);
        mouthPulse[1] = std::max(0.0f, mouthPulse[1] - dt);

        HandControlState leftControl = leftHand;
        HandControlState rightControl = rightHand;
        leftControl.pinched = leftPinch;
        rightControl.pinched = rightPinch;
        UpdateHandControlForMouth(leftControl, left, false, dt);
        UpdateHandControlForMouth(rightControl, right, true, dt);
        if (leftControl.pinched && left.centerControlled) mouthPulse[0] = std::max(mouthPulse[0], 0.35f);
        if (rightControl.pinched && right.centerControlled) mouthPulse[1] = std::max(mouthPulse[1], 0.35f);

        if (leftFist && fireCooldown[0] <= 0.0f &&
            (FistNearMouth(bridge.Geometry(false), leftControl, left) || FistInFireLane(bridge.Geometry(false), leftControl, left, false))) {
            EmitTransitBurst(transit, FistEmitterPoint(bridge.Geometry(false)), left, true);
            fireCooldown[0] = 0.07f;
            mouthPulse[0] = 1.05f;
        }
        if (rightFist && fireCooldown[1] <= 0.0f &&
            (FistNearMouth(bridge.Geometry(true), rightControl, right) || FistInFireLane(bridge.Geometry(true), rightControl, right, true))) {
            EmitTransitBurst(transit, FistEmitterPoint(bridge.Geometry(true)), right, false);
            fireCooldown[1] = 0.07f;
            mouthPulse[1] = 1.05f;
        }

        left.center.y = right.center.y = 1.8f;
        left.center.z = right.center.z = 0.0f;
        left.radius = std::clamp(left.radius, 0.75f, 2.9f);
        right.radius = std::clamp(right.radius, 0.75f, 2.9f);
        left.center.x = std::clamp(left.center.x, -6.8f, -0.9f);
        right.center.x = std::clamp(right.center.x, 0.9f, 6.8f);
        const float throatPulse = 0.5f + 0.5f * std::sin(t * 1.35f);
        const float pulseValue = throatPulse * 0.75f + std::max(mouthPulse[0], mouthPulse[1]) * 0.25f;

        for (RimMote& mote : rimMotes) mote.theta += mote.speed * dt * (0.9f + 0.25f * pulseValue);

        BeginDrawing();
        ClearBackground(BLACK);

        BeginMode3D(camera);
        for (const NebulaBlob& blob : nebulaBlobs) {
            const EnvironmentPalette& palette = blob.side < 0 ? nearPalette : farPalette;
            const MouthState& mouth = blob.side < 0 ? left : right;
            Vector3 warped = DistortAroundMouthX(blob.position, mouth, distortion);
            const Color c = LerpColor(palette.nebula, palette.fog, blob.tint);
            DrawSphere(warped, blob.radius, Fade(c, blob.alpha));
        }

        for (const DistantStar& star : stars) {
            const EnvironmentPalette& palette = star.side < 0 ? nearPalette : farPalette;
            const MouthState& mouth = star.side < 0 ? left : right;
            Vector3 warped = DistortAroundMouthX(star.position, mouth, distortion);
            const Color c = LerpColor(palette.star, Color{255, 255, 255, 255}, star.tint);
            DrawSphere(warped, star.size, Fade(c, 0.92f));
        }

        DrawDeepSpaceField(t);
        DrawTunnelSurface(left, right, t, nearPalette, farPalette, pulseValue, swirlIntensity);
        DrawThroatCore(left, right, t, nearPalette, farPalette, pulseValue);
        DrawFlowParticles(particles, left, right, dt, t, nearPalette, farPalette, pulseValue, swirlIntensity);
        UpdateAndDrawTransitParticles(transit, left, right, dt);
        DrawMouth(left, false, t, nearPalette, mouthPulse[0]);
        DrawMouth(right, true, t, farPalette, mouthPulse[1]);

        for (const RimMote& mote : rimMotes) {
            const MouthState& mouth = mote.side < 0 ? left : right;
            const EnvironmentPalette& palette = mote.side < 0 ? nearPalette : farPalette;
            const float localPulse = mote.side < 0 ? mouthPulse[0] : mouthPulse[1];
            const float ringRadius = mouth.radius + 0.2f + mote.ringFraction * 1.6f;
            Vector3 position{
                mouth.center.x + mote.lift + 0.18f * std::sin(t * 2.0f + mote.phase),
                mouth.center.y + ringRadius * std::cos(mote.theta),
                mouth.center.z + ringRadius * std::sin(mote.theta),
            };
            DrawSphere(position, mote.size, Fade(palette.accent, 0.12f + 0.20f * mote.ringFraction + 0.08f * localPulse));
        }

        for (int i = 0; i < 18; ++i) {
            const float u = static_cast<float>(i) / 17.0f;
            const float x = left.center.x + (right.center.x - left.center.x) * u;
            const float r = MouthRadiusAt(x, left.center.x, right.center.x, left.radius, right.radius);
            const Color c = TubeColorX(x, left, right, nearPalette, farPalette, pulseValue);
            DrawSphere({x, 1.8f, 0.0f}, 0.05f + 0.03f * std::sin(u * kPi), Fade(c, 0.16f));
            DrawSphere({x, 1.8f, 0.0f}, r * 0.08f, Fade(c, 0.08f));
        }

        if (bridge.AnyTracked()) bridge.DrawHands(false);
        EndMode3D();

        const Vector2 leftMouthScreen = GetWorldToScreen(left.center, camera);
        const Vector2 rightMouthScreen = GetWorldToScreen(right.center, camera);
        DrawCircleGradient(static_cast<int>(leftMouthScreen.x), static_cast<int>(leftMouthScreen.y),
                           72.0f + mouthPulse[0] * 24.0f,
                           Fade(nearPalette.mouth, 0.04f + mouthPulse[0] * 0.03f),
                           Fade(Color{0, 0, 0, 0}, 0.0f));
        DrawCircleGradient(static_cast<int>(rightMouthScreen.x), static_cast<int>(rightMouthScreen.y),
                           72.0f + mouthPulse[1] * 24.0f,
                           Fade(farPalette.mouth, 0.04f + mouthPulse[1] * 0.03f),
                           Fade(Color{0, 0, 0, 0}, 0.0f));
        DrawCircleGradient(GetScreenWidth() / 2, GetScreenHeight() / 2, 320.0f,
                           Fade(LerpColor(nearPalette.fog, farPalette.fog, 0.5f), 0.015f),
                           Fade(Color{0, 0, 0, 0}, 0.0f));

        DrawText("Wormhole Hand Lab", 20, 18, 34, Color{236, 240, 248, 255});
        DrawText("Pinch near a mouth to reshape it. Hold a fist on that side to stream matter into the wormhole.", 20, 56, 20, Color{182, 198, 226, 255});
        DrawBridgeStatus(bridge, 20, 84);
        bridge.DrawPreviewPanel({static_cast<float>(GetScreenWidth() - 392), 20.0f, 360.0f, 220.0f}, "Python Webcam Feed");
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
