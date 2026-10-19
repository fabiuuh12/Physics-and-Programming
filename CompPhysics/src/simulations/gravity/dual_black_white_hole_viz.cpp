#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr int kScreenWidth = 1480;
constexpr int kScreenHeight = 900;
constexpr int kWindowMinWidth = 980;
constexpr int kWindowMinHeight = 620;

constexpr int kStarCount = 420;
constexpr int kDiskParticlesPerHole = 1700;
constexpr int kBridgeParticleCount = 900;
constexpr int kWhiteJetParticleCount = 760;
constexpr int kWormholeParticleCount = 980;

constexpr float kDiskInnerScale = 1.35f;
constexpr float kDiskOuterScale = 7.4f;
constexpr float kBridgeBounds = 28.0f;
constexpr float kMaxParticleSpeed = 18.0f;

struct Star {
    Vector3 position;
    float radius;
    float twinkle;
    float phase;
    Color color;
};

struct Hole {
    std::string name;
    Vector3 center;
    float massMsun = 100.0f;
    float horizon = 1.2f;
    bool white = false;
    Color core;
    Color halo;
    Color accent;
};

struct DiskParticle {
    int holeIndex = 0;
    float radius = 2.0f;
    float angle = 0.0f;
    float angularSpeed = 0.0f;
    float drift = 0.0f;
    float height = 0.0f;
    float phase = 0.0f;
    float heat = 0.0f;
    float size = 0.06f;
    Vector3 position{};
    Vector3 velocity{};
};

struct BridgeParticle {
    Vector3 position{};
    Vector3 prev{};
    Vector3 velocity{};
    float life = 0.0f;
    float size = 0.03f;
};

struct JetParticle {
    float lane = 0.0f;
    float y = 0.0f;
    float prevY = 0.0f;
    float speed = 3.0f;
    float radius = 0.1f;
    float swirl = 1.0f;
    float phase = 0.0f;
    float heat = 1.0f;
};

struct Wormhole {
    Vector3 center{};
    float throatRadius = 1.0f;
    float flare = 0.65f;
    float halfLength = 4.6f;
    Color colorA{};
    Color colorB{};
};

struct WormParticle {
    float u = 0.0f;
    float theta = 0.0f;
    float speed = 1.0f;
    float swirl = 1.0f;
    float heat = 1.0f;
    float size = 0.03f;
    Vector3 prev{};
    Vector3 position{};
};

struct CameraRig {
    float yaw = -0.82f;
    float pitch = 0.28f;
    float distance = 31.0f;
    Vector3 target = {0.0f, 0.6f, 0.0f};
};

struct RenderQuality {
    int starStride = 1;
    int diskStride = 1;
    int bridgeStride = 1;
    int jetStride = 1;
    int wormStride = 1;
    int wormRings = 52;
    int wormSegs = 58;
    int wormWireModulo = 4;
    bool drawTails = true;
    bool pointMode = false;
    const char* label = "HIGH";
};

RenderQuality BuildRenderQuality(float fpsSmoothed) {
    RenderQuality q;
    if (fpsSmoothed >= 57.0f) {
        q.label = "HIGH";
    } else if (fpsSmoothed >= 48.0f) {
        q.starStride = 1;
        q.diskStride = 2;
        q.bridgeStride = 2;
        q.jetStride = 2;
        q.wormStride = 2;
        q.wormRings = 42;
        q.wormSegs = 46;
        q.wormWireModulo = 6;
        q.label = "MED";
    } else if (fpsSmoothed >= 38.0f) {
        q.starStride = 2;
        q.diskStride = 3;
        q.bridgeStride = 3;
        q.jetStride = 3;
        q.wormStride = 3;
        q.wormRings = 30;
        q.wormSegs = 34;
        q.wormWireModulo = 0;
        q.drawTails = false;
        q.pointMode = true;
        q.label = "LOW";
    } else {
        q.starStride = 2;
        q.diskStride = 4;
        q.bridgeStride = 4;
        q.jetStride = 4;
        q.wormStride = 4;
        q.wormRings = 22;
        q.wormSegs = 24;
        q.wormWireModulo = 0;
        q.drawTails = false;
        q.pointMode = true;
        q.label = "ECO";
    }
    return q;
}

float Rand01() {
    return static_cast<float>(GetRandomValue(0, 10000)) / 10000.0f;
}

float RandRange(float lo, float hi) {
    return lo + (hi - lo) * Rand01();
}

float Clamp01(float v) {
    return std::clamp(v, 0.0f, 1.0f);
}

float Mix(float a, float b, float t) {
    return a + (b - a) * Clamp01(t);
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

Color WithAlpha(Color c, unsigned char alpha) {
    c.a = alpha;
    return c;
}

void UpdateHoleScale(Hole* hole) {
    // Stylized horizon size from mass in solar masses.
    hole->horizon = 0.62f + std::sqrt(std::max(1.0f, hole->massMsun)) * 0.082f;
}

std::vector<Star> BuildStars() {
    std::vector<Star> stars;
    stars.reserve(kStarCount);
    for (int i = 0; i < kStarCount; ++i) {
        const float theta = RandRange(0.0f, 2.0f * PI);
        const float phi = RandRange(-0.42f * PI, 0.42f * PI);
        const float radius = RandRange(95.0f, 130.0f);
        const float cp = std::cos(phi);
        const Vector3 p = {
            radius * cp * std::cos(theta),
            radius * std::sin(phi),
            radius * cp * std::sin(theta),
        };
        stars.push_back({
            p,
            RandRange(0.03f, 0.15f),
            RandRange(0.5f, 1.6f),
            RandRange(0.0f, 2.0f * PI),
            LerpColor(Color{160, 195, 255, 255}, Color{255, 220, 170, 255}, Rand01()),
        });
    }
    return stars;
}

void ResetDiskParticle(DiskParticle* p, const Hole& hole) {
    const float inner = hole.horizon * kDiskInnerScale;
    const float outer = hole.horizon * kDiskOuterScale;
    const bool white = hole.white;

    const float u = std::pow(Rand01(), white ? 0.72f : 0.55f);
    p->radius = Mix(inner, outer, u);
    p->angle = RandRange(0.0f, 2.0f * PI);
    p->phase = RandRange(0.0f, 2.0f * PI);
    p->height = RandRange(-0.20f, 0.20f);
    p->heat = 1.0f - Clamp01((p->radius - inner) / std::max(0.01f, (outer - inner)));
    p->size = Mix(0.025f, 0.08f, Rand01());

    const float grav = std::sqrt(hole.massMsun / std::max(0.2f, p->radius));
    p->angularSpeed = (white ? 0.045f : 0.070f) * grav * RandRange(0.85f, 1.15f);
    p->drift = (white ? 0.40f : 0.28f) * RandRange(0.75f, 1.25f);
}

std::vector<DiskParticle> BuildDiskParticles(const std::vector<Hole>& holes) {
    std::vector<DiskParticle> particles;
    particles.reserve(kDiskParticlesPerHole * static_cast<int>(holes.size()));
    for (int holeIndex = 0; holeIndex < static_cast<int>(holes.size()); ++holeIndex) {
        for (int i = 0; i < kDiskParticlesPerHole; ++i) {
            DiskParticle p;
            p.holeIndex = holeIndex;
            ResetDiskParticle(&p, holes[holeIndex]);
            particles.push_back(p);
        }
    }
    return particles;
}

void ResetBridgeParticle(BridgeParticle* p) {
    p->position = {
        RandRange(-9.0f, 9.0f),
        RandRange(-3.0f, 3.0f),
        RandRange(-8.5f, 8.5f),
    };
    p->prev = p->position;
    p->velocity = {
        RandRange(-0.8f, 0.8f),
        RandRange(-0.35f, 0.35f),
        RandRange(-0.8f, 0.8f),
    };
    p->life = RandRange(4.0f, 11.0f);
    p->size = RandRange(0.018f, 0.052f);
}

std::vector<BridgeParticle> BuildBridgeParticles() {
    std::vector<BridgeParticle> particles;
    particles.reserve(kBridgeParticleCount);
    for (int i = 0; i < kBridgeParticleCount; ++i) {
        BridgeParticle p;
        ResetBridgeParticle(&p);
        particles.push_back(p);
    }
    return particles;
}

void ResetJetParticle(JetParticle* p) {
    p->lane = Rand01() < 0.5f ? -1.0f : 1.0f;
    p->y = RandRange(0.0f, 22.0f);
    p->prevY = p->y;
    p->speed = RandRange(4.0f, 9.5f);
    p->radius = RandRange(0.09f, 0.62f);
    p->swirl = RandRange(0.6f, 2.2f);
    p->phase = RandRange(0.0f, 2.0f * PI);
    p->heat = RandRange(0.35f, 1.0f);
}

std::vector<JetParticle> BuildWhiteJetParticles() {
    std::vector<JetParticle> particles;
    particles.reserve(kWhiteJetParticleCount);
    for (int i = 0; i < kWhiteJetParticleCount; ++i) {
        JetParticle p;
        ResetJetParticle(&p);
        particles.push_back(p);
    }
    return particles;
}

float WormRadiusAt(const Wormhole& worm, float uNorm) {
    return worm.throatRadius + worm.flare * (uNorm * uNorm);
}

Vector3 WormPoint(const Wormhole& worm, float uNorm, float theta) {
    const float r = WormRadiusAt(worm, uNorm);
    return {
        worm.center.x + r * std::cos(theta),
        worm.center.y + r * std::sin(theta),
        worm.center.z + uNorm * worm.halfLength,
    };
}

void ResetWormParticle(WormParticle* p, bool startAtInlet) {
    p->u = startAtInlet ? -1.02f : RandRange(-1.0f, 1.0f);
    p->theta = RandRange(0.0f, 2.0f * PI);
    p->speed = RandRange(0.45f, 1.25f);
    p->swirl = RandRange(1.2f, 3.2f);
    p->heat = RandRange(0.25f, 1.0f);
    p->size = RandRange(0.016f, 0.05f);
}

std::vector<WormParticle> BuildWormParticles(const Wormhole& worm) {
    std::vector<WormParticle> particles;
    particles.reserve(kWormholeParticleCount);
    for (int i = 0; i < kWormholeParticleCount; ++i) {
        WormParticle p;
        ResetWormParticle(&p, false);
        p.position = WormPoint(worm, p.u, p.theta);
        p.prev = p.position;
        particles.push_back(p);
    }
    return particles;
}

void DrawWormholeSurface(const Wormhole& worm, float time, const RenderQuality& quality) {
    const int rings = std::max(10, quality.wormRings);
    const int segs = std::max(12, quality.wormSegs);
    for (int i = 0; i < rings - 1; ++i) {
        const float u0 = Mix(-1.0f, 1.0f, static_cast<float>(i) / static_cast<float>(rings - 1));
        const float u1 = Mix(-1.0f, 1.0f, static_cast<float>(i + 1) / static_cast<float>(rings - 1));
        const float glow = 1.0f - std::fabs(u0);
        const float pulse = 0.45f + 0.55f * (0.5f + 0.5f * std::sin(time * 2.0f + u0 * 5.0f));
        const Color shell = LerpColor(worm.colorA, worm.colorB, 0.45f + 0.5f * glow);
        const Color c = WithAlpha(shell, static_cast<unsigned char>(35 + 80 * glow * pulse));
        const Color wire = WithAlpha(LerpColor(shell, WHITE, 0.18f), static_cast<unsigned char>(70 + 120 * glow));

        for (int j = 0; j < segs; ++j) {
            const float t0 = 2.0f * PI * static_cast<float>(j) / static_cast<float>(segs);
            const float t1 = 2.0f * PI * static_cast<float>(j + 1) / static_cast<float>(segs);

            const Vector3 p00 = WormPoint(worm, u0, t0);
            const Vector3 p01 = WormPoint(worm, u0, t1);
            const Vector3 p10 = WormPoint(worm, u1, t0);
            const Vector3 p11 = WormPoint(worm, u1, t1);

            DrawTriangle3D(p00, p10, p01, c);
            DrawTriangle3D(p01, p10, p11, c);
            if (quality.wormWireModulo > 0 && (j % quality.wormWireModulo) == 0) DrawLine3D(p00, p10, wire);
        }
    }

    for (int k = 0; k < 8; ++k) {
        const float u = Mix(-1.0f, 1.0f, static_cast<float>(k) / 7.0f);
        const float ringRadius = WormRadiusAt(worm, u);
        const Vector3 center = {worm.center.x, worm.center.y, worm.center.z + u * worm.halfLength};
        const Color ring = WithAlpha(LerpColor(worm.colorA, worm.colorB, 0.5f + 0.5f * u), static_cast<unsigned char>(70 + 70 * (1.0f - std::fabs(u))));
        DrawCircle3D(center, ringRadius, {0.0f, 0.0f, 1.0f}, 0.0f, ring);
    }

    const Vector3 mouthIn = {worm.center.x, worm.center.y, worm.center.z - worm.halfLength};
    const Vector3 mouthOut = {worm.center.x, worm.center.y, worm.center.z + worm.halfLength};
    DrawSphereWires(mouthIn, worm.throatRadius * 1.06f, 22, 22, WithAlpha(worm.colorA, 180));
    DrawSphereWires(mouthOut, worm.throatRadius * 1.06f, 22, 22, WithAlpha(worm.colorB, 180));
}

float UpdateWormParticles(std::vector<WormParticle>* particles, const Wormhole& worm, float dt, float simSpeed) {
    float speedSum = 0.0f;
    for (WormParticle& p : *particles) {
        p.prev = p.position;
        const float speedScale = (0.7f + 0.6f * p.heat) * (1.0f + 0.2f / std::max(0.15f, 0.25f + std::fabs(p.u)));
        p.u += p.speed * dt * simSpeed * speedScale;
        p.theta += p.swirl * dt * simSpeed * (1.5f - 0.7f * std::fabs(p.u));

        if (p.u > 1.05f) ResetWormParticle(&p, true);
        p.position = WormPoint(worm, p.u, p.theta);
        speedSum += Vector3Length(Vector3Scale(Vector3Subtract(p.position, p.prev), 1.0f / std::max(1e-4f, dt)));
    }
    return particles->empty() ? 0.0f : speedSum / static_cast<float>(particles->size());
}

void DrawWormParticles(const std::vector<WormParticle>& particles, const RenderQuality& quality) {
    for (std::size_t i = 0; i < particles.size(); i += static_cast<std::size_t>(quality.wormStride)) {
        const WormParticle& p = particles[i];
        const float centerBoost = 1.0f - Clamp01(std::fabs(p.u));
        Color c = LerpColor(Color{130, 180, 255, 255}, Color{180, 255, 255, 255}, p.heat);
        c = WithAlpha(c, static_cast<unsigned char>(70 + 160 * centerBoost));
        if (quality.drawTails) DrawLine3D(p.prev, p.position, WithAlpha(c, 110));
        if (quality.pointMode && (i % 3 != 0)) {
            DrawPoint3D(p.position, c);
        } else {
            DrawSphere(p.position, p.size * (0.65f + 0.8f * centerBoost), c);
        }
    }
}

void UpdateOrbitCamera(Camera3D* camera, CameraRig* rig, bool autoOrbit, float dt) {
    if (autoOrbit) rig->yaw += dt * 0.12f;
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        const Vector2 delta = GetMouseDelta();
        rig->yaw -= delta.x * 0.0033f;
        rig->pitch += delta.y * 0.0033f;
    }

    rig->pitch = std::clamp(rig->pitch, -1.28f, 1.10f);
    rig->distance -= GetMouseWheelMove() * 1.2f;
    rig->distance = std::clamp(rig->distance, 12.0f, 58.0f);

    const float cp = std::cos(rig->pitch);
    const Vector3 offset = {
        rig->distance * cp * std::cos(rig->yaw),
        rig->distance * std::sin(rig->pitch),
        rig->distance * cp * std::sin(rig->yaw),
    };
    camera->position = Vector3Add(rig->target, offset);
    camera->target = rig->target;
}

void DrawStars(const std::vector<Star>& stars, float time, const RenderQuality& quality) {
    for (std::size_t i = 0; i < stars.size(); i += static_cast<std::size_t>(quality.starStride)) {
        const Star& s = stars[i];
        const float glow = 0.55f + 0.45f * std::sin(time * s.twinkle + s.phase);
        const Color c = WithAlpha(s.color, static_cast<unsigned char>(80 + 170 * glow));
        if (quality.pointMode && (i % 4 != 0)) {
            DrawPoint3D(s.position, c);
        } else {
            DrawSphere(s.position, s.radius * glow, c);
        }
    }
}

void DrawPhotonRings(const Hole& hole, float time) {
    const float pulse = 0.65f + 0.35f * std::sin(time * 2.2f + (hole.white ? 1.5f : 0.0f));
    const float base = hole.horizon * 2.9f;
    const Color ringColor = WithAlpha(hole.accent, static_cast<unsigned char>(120 + 90 * pulse));
    DrawCircle3D(hole.center, base, {1.0f, 0.0f, 0.0f}, 90.0f, ringColor);
    DrawCircle3D(hole.center, base * 1.08f, {1.0f, 0.0f, 0.0f}, 90.0f, WithAlpha(ringColor, 95));
    DrawCircle3D(hole.center, base * 0.92f, {0.95f, 0.15f, 0.20f}, 90.0f, WithAlpha(ringColor, 80));
}

void DrawHoleBody(const Hole& hole, float time) {
    const float pulse = 0.5f + 0.5f * std::sin(time * 2.4f + (hole.white ? 1.3f : 0.0f));
    DrawSphere(hole.center, hole.horizon * 0.92f, hole.core);
    DrawSphere(hole.center, hole.horizon * 1.22f, WithAlpha(hole.halo, static_cast<unsigned char>(40 + 35 * pulse)));
    DrawSphereWires(hole.center, hole.horizon * 1.28f, 22, 22, WithAlpha(hole.accent, static_cast<unsigned char>(90 + 90 * pulse)));
    DrawPhotonRings(hole, time);
}

void UpdateDiskParticles(std::vector<DiskParticle>* particles, const std::vector<Hole>& holes, float dt, float simSpeed, float* blackSpeedOut, float* whiteSpeedOut) {
    float blackSpeedSum = 0.0f;
    float whiteSpeedSum = 0.0f;
    int blackCount = 0;
    int whiteCount = 0;

    for (DiskParticle& p : *particles) {
        const Hole& hole = holes[p.holeIndex];
        const bool white = hole.white;
        const float scaledDt = dt * simSpeed;
        const float radialStrength = white ? 0.50f : 0.58f;
        const float whirl = white ? 0.85f : 1.20f;
        const float inner = hole.horizon * kDiskInnerScale;
        const float outer = hole.horizon * kDiskOuterScale;

        p.angle += p.angularSpeed * scaledDt * whirl * (1.0f + 0.24f / std::max(inner, p.radius));
        if (white) {
            p.radius += p.drift * scaledDt * radialStrength * (0.6f + 0.4f * p.heat);
            if (p.radius > outer) ResetDiskParticle(&p, hole);
        } else {
            p.radius -= p.drift * scaledDt * radialStrength * (0.7f + 0.8f * p.heat);
            if (p.radius < inner * 0.96f) ResetDiskParticle(&p, hole);
        }

        const float yWave = std::sin(p.phase + p.angle * 3.0f) * 0.12f;
        const float y = p.height + yWave;
        const float cs = std::cos(p.angle);
        const float sn = std::sin(p.angle);
        const Vector3 radialDir = {cs, 0.0f, sn};
        const Vector3 tanDir = {-sn, 0.0f, cs};

        p.position = Vector3Add(hole.center, {p.radius * cs, y, p.radius * sn});

        const float vTan = p.angularSpeed * p.radius * 0.95f;
        const float vRad = p.drift * (white ? 0.85f : -0.95f);
        const Vector3 v = Vector3Add(
            Vector3Scale(tanDir, vTan),
            Vector3Add(
                Vector3Scale(radialDir, vRad),
                Vector3{0.0f, 0.18f * std::cos(p.phase + p.angle), 0.0f}
            )
        );
        p.velocity = Vector3Scale(v, simSpeed);

        const float speed = Vector3Length(p.velocity);
        if (white) {
            whiteSpeedSum += speed;
            whiteCount += 1;
        } else {
            blackSpeedSum += speed;
            blackCount += 1;
        }
    }

    *blackSpeedOut = blackCount > 0 ? blackSpeedSum / static_cast<float>(blackCount) : 0.0f;
    *whiteSpeedOut = whiteCount > 0 ? whiteSpeedSum / static_cast<float>(whiteCount) : 0.0f;
}

void DrawDiskParticles(const std::vector<DiskParticle>& particles, const std::vector<Hole>& holes, const RenderQuality& quality) {
    for (std::size_t i = 0; i < particles.size(); i += static_cast<std::size_t>(quality.diskStride)) {
        const DiskParticle& p = particles[i];
        const Hole& hole = holes[p.holeIndex];
        const float inner = hole.horizon * kDiskInnerScale;
        const float outer = hole.horizon * kDiskOuterScale;
        const float hot = 1.0f - Clamp01((p.radius - inner) / std::max(0.01f, (outer - inner)));
        const float boost = Clamp01(Vector3Length(p.velocity) / 8.0f);

        Color c;
        if (hole.white) {
            c = LerpColor(Color{142, 197, 255, 255}, Color{255, 248, 210, 255}, hot);
            c = LerpColor(c, Color{255, 255, 255, 255}, boost * 0.45f);
        } else {
            c = LerpColor(Color{255, 190, 120, 255}, Color{255, 86, 38, 255}, hot);
            c = LerpColor(c, Color{255, 236, 190, 255}, boost * 0.35f);
        }
        c = WithAlpha(c, static_cast<unsigned char>(170 + 80 * hot));

        if (quality.drawTails) {
            const Vector3 tail = Vector3Subtract(p.position, Vector3Scale(p.velocity, 0.018f));
            DrawLine3D(tail, p.position, WithAlpha(c, 130));
        }
        if (quality.pointMode && (i % 3 != 0)) {
            DrawPoint3D(p.position, c);
        } else {
            DrawSphere(p.position, p.size * (0.8f + 0.6f * hot), c);
        }
    }
}

void UpdateBridgeParticles(std::vector<BridgeParticle>* particles, const std::vector<Hole>& holes, const Wormhole& worm, float dt, float simSpeed, int* capturedByBlackOut, int* wormTransfersOut) {
    constexpr float kGravScale = 0.085f;
    int capturedByBlack = 0;
    int wormTransfers = 0;
    const Vector3 mouthIn = {worm.center.x, worm.center.y, worm.center.z - worm.halfLength};
    const Vector3 mouthOut = {worm.center.x, worm.center.y, worm.center.z + worm.halfLength};

    for (BridgeParticle& p : *particles) {
        p.prev = p.position;
        Vector3 accel = {0.0f, 0.0f, 0.0f};

        for (const Hole& hole : holes) {
            const Vector3 toHole = Vector3Subtract(hole.center, p.position);
            const float distance = std::max(0.24f, Vector3Length(toHole));
            const Vector3 dirToHole = Vector3Scale(toHole, 1.0f / distance);

            const float grav = kGravScale * hole.massMsun / (distance * distance + 0.22f);
            accel = Vector3Add(accel, Vector3Scale(dirToHole, grav));

            if (hole.white) {
                const float shell = Clamp01((hole.horizon * 2.35f - distance) / (hole.horizon * 1.5f));
                const float ejection = shell * shell * hole.massMsun * 0.030f / (distance * distance + 0.2f);
                accel = Vector3Add(accel, Vector3Scale(dirToHole, -ejection));
            }
        }

        const Vector3 toIn = Vector3Subtract(mouthIn, p.position);
        const float distIn = std::max(0.10f, Vector3Length(toIn));
        if (distIn < worm.throatRadius * 4.2f) {
            const Vector3 inDir = Vector3Scale(toIn, 1.0f / distIn);
            const float funnel = (worm.throatRadius * 4.2f - distIn) / (worm.throatRadius * 4.2f);
            accel = Vector3Add(accel, Vector3Scale(inDir, 1.6f * funnel * funnel));

            Vector3 swirl = Vector3CrossProduct(inDir, {0.0f, 1.0f, 0.0f});
            const float swirlLen = Vector3Length(swirl);
            if (swirlLen > 1e-4f) swirl = Vector3Scale(swirl, 1.0f / swirlLen);
            accel = Vector3Add(accel, Vector3Scale(swirl, 0.65f * funnel));
        }

        p.velocity = Vector3Add(p.velocity, Vector3Scale(accel, dt * simSpeed));
        p.velocity = Vector3Scale(p.velocity, 1.0f - 0.04f * dt * simSpeed);
        const float speed = Vector3Length(p.velocity);
        if (speed > kMaxParticleSpeed) p.velocity = Vector3Scale(p.velocity, kMaxParticleSpeed / speed);

        p.position = Vector3Add(p.position, Vector3Scale(p.velocity, dt * simSpeed));
        p.life -= dt * 0.55f;

        if (Vector3Distance(p.position, mouthIn) < worm.throatRadius * 0.62f) {
            const float a = RandRange(0.0f, 2.0f * PI);
            const float rr = worm.throatRadius * RandRange(0.4f, 0.92f);
            p.position = {
                mouthOut.x + rr * std::cos(a),
                mouthOut.y + rr * std::sin(a),
                mouthOut.z
            };
            p.prev = p.position;
            const Vector3 outDir = Vector3Normalize(Vector3{std::cos(a), std::sin(a), 1.45f});
            p.velocity = Vector3Scale(outDir, RandRange(4.0f, 8.4f));
            p.life = RandRange(4.5f, 10.5f);
            wormTransfers += 1;
        }

        bool reset = false;
        for (const Hole& hole : holes) {
            const float dist = Vector3Distance(p.position, hole.center);
            if (!hole.white && dist < hole.horizon * 0.86f) {
                capturedByBlack += 1;
                reset = true;
                break;
            }
            if (hole.white && dist < hole.horizon * 0.82f) {
                const Vector3 out = Vector3Normalize(Vector3Subtract(p.position, hole.center));
                p.position = Vector3Add(hole.center, Vector3Scale(out, hole.horizon * 1.05f));
                p.velocity = Vector3Scale(out, RandRange(3.0f, 7.2f));
            }
        }

        if (!reset && (std::fabs(p.position.x) > kBridgeBounds || std::fabs(p.position.y) > kBridgeBounds || std::fabs(p.position.z) > kBridgeBounds)) {
            reset = true;
        }
        if (!reset && p.life <= 0.0f) reset = true;

        if (reset) ResetBridgeParticle(&p);
    }

    *capturedByBlackOut = capturedByBlack;
    *wormTransfersOut = wormTransfers;
}

void DrawBridgeParticles(const std::vector<BridgeParticle>& particles, const RenderQuality& quality) {
    for (std::size_t i = 0; i < particles.size(); i += static_cast<std::size_t>(quality.bridgeStride)) {
        const BridgeParticle& p = particles[i];
        const float speed = Vector3Length(p.velocity);
        const float v = Clamp01(speed / 8.5f);
        Color c = LerpColor(Color{118, 176, 255, 255}, Color{255, 219, 150, 255}, v);
        c = WithAlpha(c, static_cast<unsigned char>(85 + 150 * v));
        if (quality.drawTails) DrawLine3D(p.prev, p.position, WithAlpha(c, 96));
        if (quality.pointMode && (i % 3 != 0)) {
            DrawPoint3D(p.position, c);
        } else {
            DrawSphere(p.position, p.size * (0.85f + 0.7f * v), c);
        }
    }
}

float UpdateWhiteJets(std::vector<JetParticle>* particles, const Hole& whiteHole, float dt, float simSpeed) {
    float speedSum = 0.0f;
    for (JetParticle& p : *particles) {
        p.prevY = p.y;
        p.y += p.speed * dt * simSpeed;
        if (p.y > 24.5f) ResetJetParticle(&p);
        speedSum += p.speed * simSpeed;
    }
    return particles->empty() ? 0.0f : speedSum / static_cast<float>(particles->size());
}

void DrawWhiteJets(const std::vector<JetParticle>& particles, const Hole& whiteHole, float time, const RenderQuality& quality) {
    for (std::size_t i = 0; i < particles.size(); i += static_cast<std::size_t>(quality.jetStride)) {
        const JetParticle& p = particles[i];
        const float spin0 = p.swirl * p.prevY + p.phase + time * 0.5f;
        const float spin1 = p.swirl * p.y + p.phase + time * 0.5f;

        const Vector3 prevPos = {
            whiteHole.center.x + p.radius * std::cos(spin0),
            whiteHole.center.y + p.lane * p.prevY,
            whiteHole.center.z + p.radius * std::sin(spin0),
        };
        const Vector3 pos = {
            whiteHole.center.x + p.radius * std::cos(spin1),
            whiteHole.center.y + p.lane * p.y,
            whiteHole.center.z + p.radius * std::sin(spin1),
        };

        const float fade = 1.0f - Clamp01(p.y / 24.5f);
        Color c = LerpColor(Color{145, 195, 255, 255}, Color{255, 255, 240, 255}, p.heat);
        c = WithAlpha(c, static_cast<unsigned char>(60 + 170 * fade));
        if (quality.drawTails) DrawLine3D(prevPos, pos, WithAlpha(c, static_cast<unsigned char>(90 * fade + 30)));
        if (quality.pointMode && (i % 2 != 0)) {
            DrawPoint3D(pos, c);
        } else {
            DrawSphere(pos, Mix(0.018f, 0.055f, p.heat) * (0.65f + 0.55f * fade), c);
        }
    }
}

void DrawMinimalHud(
    const std::vector<Hole>& holes,
    const Wormhole& worm,
    float blackMeanSpeed,
    float whiteMeanSpeed,
    float whiteJetSpeed,
    float wormFlowSpeed,
    float simSpeed,
    int bridgeCount,
    int captures,
    int wormTransfers,
    bool paused,
    int fps,
    const RenderQuality& quality
) {
    const float cScale = 12.0f;
    const float blackV = blackMeanSpeed / cScale;
    const float whiteV = std::max(whiteMeanSpeed, whiteJetSpeed) / cScale;
    const float wormV = wormFlowSpeed / cScale;

    DrawRectangleRounded(Rectangle{18.0f, 16.0f, 1120.0f, 90.0f}, 0.16f, 8, Color{8, 13, 22, 190});
    DrawText("Black Hole + White Hole + Wormhole", 34, 30, 24, RAYWHITE);

    char lineA[320];
    std::snprintf(
        lineA, sizeof(lineA),
        "Black mass %.1f Msun | <disk speed> %.2f c     White mass %.1f Msun | <outflow speed> %.2f c     Worm flow %.2f c",
        holes[0].massMsun, blackV, holes[1].massMsun, whiteV, wormV
    );
    DrawText(lineA, 34, 62, 19, Color{194, 207, 235, 255});

    char lineB[256];
    std::snprintf(
        lineB, sizeof(lineB),
        "Bridge particles %d | captures/frame %d | worm transfers/frame %d | throat %.2f | sim %.2fx | fps %d | render %s%s",
        bridgeCount, captures, wormTransfers, worm.throatRadius, simSpeed, fps, quality.label, paused ? " | PAUSED" : ""
    );
    DrawText(lineB, 34, 84, 18, Color{170, 188, 222, 255});
}

}  // namespace

int main() {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(kScreenWidth, kScreenHeight, "Dual Black Hole + White Hole 3D - C++");
    SetWindowMinSize(kWindowMinWidth, kWindowMinHeight);
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {27.0f, 8.2f, 23.0f};
    camera.target = {4.0f, 0.6f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 46.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    std::vector<Hole> holes = {
        {"Black Hole", {-6.2f, 0.0f, 0.0f}, 145.0f, 1.4f, false, BLACK, Color{80, 100, 140, 255}, Color{255, 145, 95, 255}},
        {"White Hole", { 6.2f, 0.0f, 0.0f}, 116.0f, 1.3f, true,  Color{245, 248, 255, 255}, Color{180, 210, 255, 255}, Color{170, 215, 255, 255}},
    };
    for (Hole& hole : holes) UpdateHoleScale(&hole);
    Wormhole wormhole{{15.2f, 0.0f, 0.0f}, 1.22f, 0.62f, 4.8f, Color{105, 165, 255, 255}, Color{170, 245, 255, 255}};

    CameraRig rig;
    rig.target = {4.0f, 0.6f, 0.0f};
    const std::vector<Star> stars = BuildStars();
    std::vector<DiskParticle> disk = BuildDiskParticles(holes);
    std::vector<BridgeParticle> bridge = BuildBridgeParticles();
    std::vector<JetParticle> whiteJets = BuildWhiteJetParticles();
    std::vector<WormParticle> wormParticles = BuildWormParticles(wormhole);

    bool paused = false;
    bool autoOrbit = true;
    bool hudVisible = true;
    bool forceEcoRender = false;
    float simSpeed = 1.0f;
    float sceneTime = 0.0f;
    float blackMeanSpeed = 0.0f;
    float whiteMeanSpeed = 0.0f;
    float whiteJetSpeed = 0.0f;
    float wormFlowSpeed = 0.0f;
    int capturesFrame = 0;
    int wormTransfersFrame = 0;
    float fpsSmoothed = 60.0f;
    int fpsNow = 60;
    RenderQuality renderQuality = BuildRenderQuality(fpsSmoothed);

    while (!WindowShouldClose()) {
        const float dt = GetFrameTime();
        if (IsKeyPressed(KEY_SPACE)) paused = !paused;
        if (IsKeyPressed(KEY_A)) autoOrbit = !autoOrbit;
        if (IsKeyPressed(KEY_H)) hudVisible = !hudVisible;
        if (IsKeyPressed(KEY_L)) forceEcoRender = !forceEcoRender;

        if (IsKeyDown(KEY_Q)) simSpeed = std::max(0.25f, simSpeed - dt * 0.8f);
        if (IsKeyDown(KEY_E)) simSpeed = std::min(4.0f, simSpeed + dt * 0.8f);

        if (IsKeyDown(KEY_Z)) holes[0].massMsun = std::max(40.0f, holes[0].massMsun - 52.0f * dt);
        if (IsKeyDown(KEY_X)) holes[0].massMsun = std::min(320.0f, holes[0].massMsun + 52.0f * dt);
        if (IsKeyDown(KEY_C)) holes[1].massMsun = std::max(40.0f, holes[1].massMsun - 52.0f * dt);
        if (IsKeyDown(KEY_V)) holes[1].massMsun = std::min(320.0f, holes[1].massMsun + 52.0f * dt);
        if (IsKeyDown(KEY_B)) wormhole.throatRadius = std::max(0.60f, wormhole.throatRadius - 1.0f * dt);
        if (IsKeyDown(KEY_N)) wormhole.throatRadius = std::min(2.60f, wormhole.throatRadius + 1.0f * dt);

        if (IsKeyPressed(KEY_R)) {
            holes[0].massMsun = 145.0f;
            holes[1].massMsun = 116.0f;
            wormhole.throatRadius = 1.22f;
            simSpeed = 1.0f;
            paused = false;
            autoOrbit = true;
        }

        for (Hole& hole : holes) UpdateHoleScale(&hole);
        UpdateOrbitCamera(&camera, &rig, autoOrbit, dt);

        fpsNow = GetFPS();
        fpsSmoothed = Mix(fpsSmoothed, static_cast<float>(fpsNow), Clamp01(dt * 5.0f));
        renderQuality = forceEcoRender ? BuildRenderQuality(0.0f) : BuildRenderQuality(fpsSmoothed);

        if (!paused) {
            sceneTime += dt * simSpeed;
            UpdateDiskParticles(&disk, holes, dt, simSpeed, &blackMeanSpeed, &whiteMeanSpeed);
            UpdateBridgeParticles(&bridge, holes, wormhole, dt, simSpeed, &capturesFrame, &wormTransfersFrame);
            whiteJetSpeed = UpdateWhiteJets(&whiteJets, holes[1], dt, simSpeed);
            wormFlowSpeed = UpdateWormParticles(&wormParticles, wormhole, dt, simSpeed);
        } else {
            capturesFrame = 0;
            wormTransfersFrame = 0;
        }

        BeginDrawing();
        ClearBackground(Color{4, 7, 14, 255});
        BeginMode3D(camera);

        DrawStars(stars, sceneTime * 0.55f, renderQuality);
        DrawDiskParticles(disk, holes, renderQuality);
        DrawBridgeParticles(bridge, renderQuality);
        DrawWhiteJets(whiteJets, holes[1], sceneTime, renderQuality);
        DrawWormholeSurface(wormhole, sceneTime, renderQuality);
        DrawWormParticles(wormParticles, renderQuality);
        DrawHoleBody(holes[0], sceneTime);
        DrawHoleBody(holes[1], sceneTime);

        EndMode3D();

        if (hudVisible) {
            DrawMinimalHud(holes, wormhole, blackMeanSpeed, whiteMeanSpeed, whiteJetSpeed, wormFlowSpeed, simSpeed, static_cast<int>(bridge.size()), capturesFrame, wormTransfersFrame, paused, fpsNow, renderQuality);
            DrawRectangleRounded(Rectangle{18.0f, static_cast<float>(GetScreenHeight() - 38), 940.0f, 26.0f}, 0.20f, 6, Color{8, 13, 22, 160});
            DrawText("Mouse drag orbit | wheel zoom | Z/X black mass | C/V white mass | B/N wormhole throat | Q/E sim speed | A auto | L eco render | H hide HUD", 28, GetScreenHeight() - 30, 15, Color{188, 202, 231, 255});
        }

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
