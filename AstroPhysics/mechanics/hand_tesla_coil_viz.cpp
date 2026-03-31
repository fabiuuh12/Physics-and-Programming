#include "raylib.h"
#include "raymath.h"
#include "../vision/hand_tracking_scene_shared.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace astro_hand;

constexpr int kScreenWidth = 1440;
constexpr int kScreenHeight = 900;

constexpr float kFloorY = 0.0f;
constexpr Vector3 kCoilLeftTip = {-3.35f, 2.95f, -0.28f};
constexpr Vector3 kCoilRightTip = {3.35f, 2.95f, 0.28f};
constexpr std::array<Vector3, 2> kCoilTips = {kCoilLeftTip, kCoilRightTip};
constexpr Vector3 kCameraTarget = {0.0f, 2.58f, 0.0f};
constexpr int kMaxSecondaryContacts = 3;

float gRenderQuality = 1.0f;

constexpr std::array<const char*, 21> kLandmarkNames = {
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
};

enum class ArcPhase {
    Idle = 0,
    Corona,
    Seeking,
    Contact,
    PlasmaHold,
};

enum class FeatureKind {
    Capsule = 0,
    Triangle,
};

struct ContactFeature {
    FeatureKind kind = FeatureKind::Capsule;
    bool rightHand = true;
    int a = -1;
    int b = -1;
    int c = -1;
    const char* label = "none";
    float bias = 0.0f;
};

struct ContactCandidate {
    bool valid = false;
    ContactFeature feature{};
    Vector3 point{};
    float gap = 1000.0f;
    float score = 1000.0f;
    float surfaceRadius = 0.0f;
};

struct ActiveHandRef {
    bool valid = false;
    bool rightHand = true;
    const HandGeometry* geometry = nullptr;
    const HandControlState* control = nullptr;
    std::array<ContactCandidate, 2> candidates{};
    std::array<std::array<ContactCandidate, kMaxSecondaryContacts>, 2> secondaryCandidates{};
    std::array<int, 2> secondaryCounts = {0, 0};
    float compositeScore = 1000.0f;
};

struct CoilSceneState {
    float voltageTarget = 0.35f;
    float voltage = 0.35f;
    float nearGapObserved = 0.26f;
    float farGapObserved = 2.60f;
    float lockPersistence = 0.62f;
    float lockSlack = 0.12f;
    float audioHarshness = 0.62f;
    float renderQuality = 1.0f;
    std::array<float, 2> requiredVoltage = {0.30f, 0.30f};
    std::array<float, 2> proximity = {0.0f, 0.0f};
    std::array<float, 2> reaction = {0.0f, 0.0f};
    std::array<float, 2> corona = {0.0f, 0.0f};
    std::array<float, 2> phaseTime = {0.0f, 0.0f};
    std::array<float, 2> lockTime = {0.0f, 0.0f};
    std::array<std::string, 2> contactLabel = {"none", "none"};
    std::array<ArcPhase, 2> phase = {ArcPhase::Idle, ArcPhase::Idle};
    std::array<ContactFeature, 2> lockedFeature{};
    std::array<bool, 2> lockedValid = {false, false};
    std::array<bool, 2> active = {false, false};
};

struct CoilAudioEngine {
    bool ready = false;
    AudioStream stream{};
    std::vector<float> buffer{};
    float carrierPhase = 0.0f;
    float buzzPhase = 0.0f;
    float hissPhase = 0.0f;
    uint32_t noiseState = 0x12345u;

    void Start() {
        InitAudioDevice();
        if (!IsAudioDeviceReady()) return;
        SetAudioStreamBufferSizeDefault(1024);
        stream = LoadAudioStream(44100, 32, 1);
        if (!IsAudioStreamValid(stream)) return;
        buffer.assign(1024, 0.0f);
        SetAudioStreamVolume(stream, 0.45f);
        PlayAudioStream(stream);
        ready = true;
    }

    void Shutdown() {
        if (ready) {
            StopAudioStream(stream);
            UnloadAudioStream(stream);
            ready = false;
        }
        if (IsAudioDeviceReady()) CloseAudioDevice();
    }

    float NextNoise() {
        noiseState = 1664525u * noiseState + 1013904223u;
        return static_cast<float>((noiseState >> 8) & 0x00ffffffu) / 8388607.5f - 1.0f;
    }

    void Update(const CoilSceneState& state) {
        if (!ready) return;
        if (!IsAudioStreamPlaying(stream)) PlayAudioStream(stream);
        const float maxReaction = std::max(state.reaction[0], state.reaction[1]);
        const float maxCorona = std::max(state.corona[0], state.corona[1]);
        const ArcPhase audioPhase =
            (state.phase[0] > state.phase[1]) ? state.phase[0] : state.phase[1];

        const float sr = 44100.0f;
        while (IsAudioStreamProcessed(stream)) {
            for (size_t i = 0; i < buffer.size(); ++i) {
                const float phaseGain =
                    (audioPhase == ArcPhase::PlasmaHold) ? 1.00f :
                    (audioPhase == ArcPhase::Contact) ? 0.82f :
                    (audioPhase == ArcPhase::Seeking) ? 0.48f :
                    (audioPhase == ArcPhase::Corona) ? 0.28f : 0.10f;
                const float harsh = state.audioHarshness;
                const float smooth = 1.0f - harsh;

                const float baseFreq = 42.0f + 110.0f * state.voltage * state.voltage + 26.0f * smooth;
                const float buzzFreq = baseFreq * (1.75f + 0.58f * harsh + 0.26f * maxCorona);
                const float hissFreq = 760.0f + 1100.0f * smooth + 2100.0f * harsh * maxCorona + 1600.0f * maxReaction;

                carrierPhase += 2.0f * PI * baseFreq / sr;
                buzzPhase += 2.0f * PI * buzzFreq / sr;
                hissPhase += 2.0f * PI * hissFreq / sr;
                if (carrierPhase > 2.0f * PI) carrierPhase -= 2.0f * PI;
                if (buzzPhase > 2.0f * PI) buzzPhase -= 2.0f * PI;
                if (hissPhase > 2.0f * PI) hissPhase -= 2.0f * PI;

                const float hum = std::sin(carrierPhase);
                const float subHum = std::sin(carrierPhase * 0.5f + 0.6f * std::sin(hissPhase * 0.01f));
                const float buzz = std::sin(buzzPhase) * std::sin(carrierPhase * (0.45f + 0.15f * harsh) + hissPhase * 0.03f);
                const float hiss = NextNoise() * (0.05f + 0.30f * smooth + 0.72f * harsh * maxCorona);
                const float crackleThreshold = 0.90f - 0.48f * harsh - 0.28f * maxReaction;
                const float crackle = (NextNoise() > crackleThreshold) ? (0.18f + 0.82f * maxReaction * (0.40f + 0.60f * harsh)) : 0.0f;

                float sample =
                    hum * (0.020f + 0.020f * smooth + 0.024f * state.voltage) +
                    subHum * (0.006f + 0.018f * smooth) +
                    buzz * (0.008f + 0.018f * smooth + 0.038f * harsh * phaseGain) +
                    hiss * (0.004f + 0.012f * phaseGain + 0.010f * harsh) +
                    crackle * (0.010f + 0.032f * harsh);

                sample *= (0.18f + 0.82f * phaseGain);
                buffer[i] = std::clamp(sample, -0.90f, 0.90f);
            }
            UpdateAudioStream(stream, buffer.data(), static_cast<int>(buffer.size()));
        }
        SetAudioStreamVolume(stream, 0.16f + 0.38f * std::max(state.voltage, maxReaction));
    }
};

struct ArcPalette {
    Color haze{};
    Color outer{};
    Color mid{};
    Color core{};
    Color ember{};
};

float Ease(float x) {
    x = Clamp01(x);
    return x * x * (3.0f - 2.0f * x);
}

const char* PhaseName(ArcPhase phase) {
    switch (phase) {
        case ArcPhase::Idle: return "idle";
        case ArcPhase::Corona: return "corona";
        case ArcPhase::Seeking: return "seeking";
        case ArcPhase::Contact: return "contact";
        case ArcPhase::PlasmaHold: return "plasma_hold";
    }
    return "unknown";
}

bool SameFeature(const ContactFeature& a, const ContactFeature& b) {
    return a.kind == b.kind && a.rightHand == b.rightHand && a.a == b.a && a.b == b.b && a.c == b.c;
}

bool IsTipFeature(const ContactFeature& feature) {
    return feature.kind == FeatureKind::Capsule &&
           (feature.b == 4 || feature.b == 8 || feature.b == 12 || feature.b == 16 || feature.b == 20);
}

Color BlendColor(Color a, Color b, float t) {
    t = Clamp01(t);
    return Color{
        static_cast<unsigned char>(a.r + (b.r - a.r) * t),
        static_cast<unsigned char>(a.g + (b.g - a.g) * t),
        static_cast<unsigned char>(a.b + (b.b - a.b) * t),
        static_cast<unsigned char>(a.a + (b.a - a.a) * t),
    };
}

Color ArcTone(float hueShift, float warmth, float alpha = 255.0f) {
    const Color cold = Color{
        static_cast<unsigned char>(110 + 30 * hueShift),
        static_cast<unsigned char>(200 + 24 * hueShift),
        static_cast<unsigned char>(255),
        static_cast<unsigned char>(alpha),
    };
    const Color warm = Color{
        static_cast<unsigned char>(255),
        static_cast<unsigned char>(206 + 26 * hueShift),
        static_cast<unsigned char>(140 + 30 * hueShift),
        static_cast<unsigned char>(alpha),
    };
    return BlendColor(cold, warm, warmth);
}

ArcPalette PaletteForArc(int seed, float intensity, bool plasmaHold = false) {
    const float hueShift = std::fmod(std::abs(seed) * 0.173f, 1.0f);
    const float warmth = Clamp01((plasmaHold ? 0.42f : 0.18f) + 0.32f * intensity);
    ArcPalette palette;
    palette.haze = Fade(ArcTone(hueShift, warmth * 0.45f), 0.05f + 0.10f * intensity);
    palette.outer = Fade(ArcTone(hueShift, warmth * 0.35f), 0.14f + 0.14f * intensity);
    palette.mid = Fade(ArcTone(hueShift, warmth * 0.62f), 0.30f + 0.18f * intensity);
    palette.core = Fade(BlendColor(WHITE, ArcTone(hueShift, warmth), 0.12f + 0.12f * intensity), 0.70f + 0.18f * intensity);
    palette.ember = Fade(BlendColor(Color{255, 150, 82, 255}, ArcTone(hueShift, 0.85f), 0.35f), 0.15f + 0.12f * intensity);
    return palette;
}

int ScaleDetail(int base, float minScale = 0.55f) {
    const float scale = LerpFloat(minScale, 1.0f, Clamp01(gRenderQuality));
    return std::max(1, static_cast<int>(std::round(base * scale)));
}

Vector3 ClosestPointOnSegment(Vector3 p, Vector3 a, Vector3 b, float* outT = nullptr) {
    const Vector3 ab = Vector3Subtract(b, a);
    const float denom = Vector3DotProduct(ab, ab);
    float t = 0.0f;
    if (denom > 1.0e-6f) {
        t = Vector3DotProduct(Vector3Subtract(p, a), ab) / denom;
        t = Clamp01(t);
    }
    if (outT) *outT = t;
    return Vector3Lerp(a, b, t);
}

Vector3 ClosestPointOnTriangle(Vector3 p, Vector3 a, Vector3 b, Vector3 c) {
    const Vector3 ab = Vector3Subtract(b, a);
    const Vector3 ac = Vector3Subtract(c, a);
    const Vector3 ap = Vector3Subtract(p, a);
    const float d1 = Vector3DotProduct(ab, ap);
    const float d2 = Vector3DotProduct(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;

    const Vector3 bp = Vector3Subtract(p, b);
    const float d3 = Vector3DotProduct(ab, bp);
    const float d4 = Vector3DotProduct(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        const float v = d1 / (d1 - d3);
        return Vector3Add(a, Vector3Scale(ab, v));
    }

    const Vector3 cp = Vector3Subtract(p, c);
    const float d5 = Vector3DotProduct(ab, cp);
    const float d6 = Vector3DotProduct(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;

    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        const float w = d2 / (d2 - d6);
        return Vector3Add(a, Vector3Scale(ac, w));
    }

    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        const Vector3 bc = Vector3Subtract(c, b);
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return Vector3Add(b, Vector3Scale(bc, w));
    }

    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    return Vector3Add(a, Vector3Add(Vector3Scale(ab, v), Vector3Scale(ac, w)));
}

ContactCandidate EvaluateCapsuleFeature(const Vector3& source, const HandGeometry& g, const ContactFeature& feature) {
    ContactCandidate out{};
    out.valid = true;
    out.feature = feature;
    const Vector3 a = g.landmarks[static_cast<size_t>(feature.a)];
    const Vector3 b = g.landmarks[static_cast<size_t>(feature.b)];
    const Vector3 spinePoint = ClosestPointOnSegment(source, a, b);
    const float radius = 0.55f * (g.radii[static_cast<size_t>(feature.a)] + g.radii[static_cast<size_t>(feature.b)]);
    Vector3 outward = Vector3Subtract(source, spinePoint);
    if (Vector3Length(outward) < 1.0e-4f) outward = {1.0f, 0.0f, 0.0f};
    outward = Vector3Normalize(outward);
    out.point = Vector3Add(spinePoint, Vector3Scale(outward, radius));
    out.surfaceRadius = radius;
    out.gap = std::max(0.0f, Vector3Distance(source, spinePoint) - radius);
    out.score = out.gap + feature.bias;
    return out;
}

ContactCandidate EvaluateTriangleFeature(const Vector3& source, const HandGeometry& g, const ContactFeature& feature) {
    ContactCandidate out{};
    out.valid = true;
    out.feature = feature;
    const Vector3 a = g.landmarks[static_cast<size_t>(feature.a)];
    const Vector3 b = g.landmarks[static_cast<size_t>(feature.b)];
    const Vector3 c = g.landmarks[static_cast<size_t>(feature.c)];
    const Vector3 point = ClosestPointOnTriangle(source, a, b, c);
    const Vector3 ab = Vector3Subtract(b, a);
    const Vector3 ac = Vector3Subtract(c, a);
    Vector3 normal = SafeNormalize(Vector3CrossProduct(ab, ac), {0.0f, 0.0f, 1.0f});
    if (Vector3DotProduct(normal, Vector3Subtract(source, point)) < 0.0f) normal = Vector3Scale(normal, -1.0f);
    out.surfaceRadius = 0.05f;
    out.point = Vector3Add(point, Vector3Scale(normal, out.surfaceRadius));
    out.gap = std::max(0.0f, Vector3Distance(source, point) - out.surfaceRadius);
    out.score = out.gap + feature.bias;
    return out;
}

ContactCandidate EvaluateFeature(const Vector3& source, const HandGeometry& g, const ContactFeature& feature) {
    return (feature.kind == FeatureKind::Capsule) ? EvaluateCapsuleFeature(source, g, feature) : EvaluateTriangleFeature(source, g, feature);
}

std::array<ContactCandidate, kMaxSecondaryContacts> CollectSecondaryContacts(
    const Vector3& source,
    const HandGeometry& g,
    bool rightHand,
    const ContactCandidate& primary,
    float voltage,
    int* outCount) {
    std::array<ContactCandidate, kMaxSecondaryContacts> selected{};
    *outCount = 0;

    static constexpr std::array<ContactFeature, 5> tipFeatures = {{
        {FeatureKind::Capsule, false, 3, 4, -1, "thumb_tip", -0.20f},
        {FeatureKind::Capsule, false, 7, 8, -1, "index_tip", -0.28f},
        {FeatureKind::Capsule, false, 11, 12, -1, "middle_tip", -0.26f},
        {FeatureKind::Capsule, false, 15, 16, -1, "ring_tip", -0.18f},
        {FeatureKind::Capsule, false, 19, 20, -1, "pinky_tip", -0.14f},
    }};

    std::array<ContactCandidate, 5> matches{};
    int matchCount = 0;
    const float maxGap = primary.gap + 0.26f + 0.84f * voltage;
    const float maxScore = primary.score + 0.18f + 0.66f * voltage;

    for (ContactFeature feature : tipFeatures) {
        feature.rightHand = rightHand;
        ContactCandidate candidate = EvaluateCapsuleFeature(source, g, feature);
        if (!candidate.valid || SameFeature(candidate.feature, primary.feature)) continue;
        if (candidate.gap > maxGap || candidate.score > maxScore) continue;
        matches[matchCount++] = candidate;
    }

    std::sort(matches.begin(), matches.begin() + matchCount, [](const ContactCandidate& a, const ContactCandidate& b) {
        return a.score < b.score;
    });

    const int count = std::min(matchCount, kMaxSecondaryContacts);
    for (int i = 0; i < count; ++i) selected[static_cast<size_t>(i)] = matches[static_cast<size_t>(i)];
    *outCount = count;
    return selected;
}

ContactCandidate ResolveHandContact(const Vector3& source, const HandGeometry& g, bool rightHand) {
    ContactCandidate best{};
    static constexpr std::array<ContactFeature, 24> capsuleFeatures = {{
        {FeatureKind::Capsule, false, 0, 1, -1, "thumb_base", 0.08f},
        {FeatureKind::Capsule, false, 1, 2, -1, "thumb_mcp", 0.02f},
        {FeatureKind::Capsule, false, 2, 3, -1, "thumb_ip", -0.08f},
        {FeatureKind::Capsule, false, 3, 4, -1, "thumb_tip", -0.20f},
        {FeatureKind::Capsule, false, 0, 5, -1, "index_base", 0.04f},
        {FeatureKind::Capsule, false, 5, 6, -1, "index_lower", -0.02f},
        {FeatureKind::Capsule, false, 6, 7, -1, "index_upper", -0.14f},
        {FeatureKind::Capsule, false, 7, 8, -1, "index_tip", -0.28f},
        {FeatureKind::Capsule, false, 0, 9, -1, "middle_base", 0.06f},
        {FeatureKind::Capsule, false, 9, 10, -1, "middle_lower", -0.03f},
        {FeatureKind::Capsule, false, 10, 11, -1, "middle_upper", -0.16f},
        {FeatureKind::Capsule, false, 11, 12, -1, "middle_tip", -0.26f},
        {FeatureKind::Capsule, false, 0, 13, -1, "ring_base", 0.08f},
        {FeatureKind::Capsule, false, 13, 14, -1, "ring_lower", 0.00f},
        {FeatureKind::Capsule, false, 14, 15, -1, "ring_upper", -0.08f},
        {FeatureKind::Capsule, false, 15, 16, -1, "ring_tip", -0.18f},
        {FeatureKind::Capsule, false, 0, 17, -1, "pinky_base", 0.10f},
        {FeatureKind::Capsule, false, 17, 18, -1, "pinky_lower", 0.02f},
        {FeatureKind::Capsule, false, 18, 19, -1, "pinky_upper", -0.06f},
        {FeatureKind::Capsule, false, 19, 20, -1, "pinky_tip", -0.14f},
        {FeatureKind::Capsule, false, 5, 9, -1, "index_middle_web", 0.12f},
        {FeatureKind::Capsule, false, 9, 13, -1, "mid_palm_bridge", 0.16f},
        {FeatureKind::Capsule, false, 13, 17, -1, "ring_pinky_web", 0.18f},
        {FeatureKind::Capsule, false, 0, 17, -1, "palm_edge", 0.22f},
    }};
    static constexpr std::array<ContactFeature, 6> triangleFeatures = {{
        {FeatureKind::Triangle, false, 0, 1, 5, "thenar_palm", 0.18f},
        {FeatureKind::Triangle, false, 0, 5, 9, "inner_palm", 0.24f},
        {FeatureKind::Triangle, false, 0, 9, 13, "center_palm", 0.28f},
        {FeatureKind::Triangle, false, 0, 13, 17, "outer_palm", 0.24f},
        {FeatureKind::Triangle, false, 1, 5, 9, "thumb_web", 0.16f},
        {FeatureKind::Triangle, false, 9, 13, 17, "metacarpal_pad", 0.26f},
    }};

    auto consider = [&](ContactCandidate candidate) {
        if (!candidate.valid) return;
        if (!best.valid || candidate.score < best.score) best = candidate;
    };

    for (ContactFeature feature : capsuleFeatures) {
        feature.rightHand = rightHand;
        consider(EvaluateCapsuleFeature(source, g, feature));
    }
    for (ContactFeature feature : triangleFeatures) {
        feature.rightHand = rightHand;
        consider(EvaluateTriangleFeature(source, g, feature));
    }
    return best;
}

ActiveHandRef PickActiveHand(const HandSceneBridge& bridge, CoilSceneState* state) {
    ActiveHandRef best{};

    auto considerTrackedHand = [&](bool rightHand) {
        const HandControlState& control = bridge.Control(rightHand);
        if (!control.active) return;
        const HandGeometry& geometry = bridge.Geometry(rightHand);
        std::array<ContactCandidate, 2> candidates{};
        std::array<std::array<ContactCandidate, kMaxSecondaryContacts>, 2> secondaryCandidates{};
        std::array<int, 2> secondaryCounts = {0, 0};
        float compositeScore = 0.0f;
        for (size_t i = 0; i < kCoilTips.size(); ++i) {
            ContactCandidate candidate = ResolveHandContact(kCoilTips[i], geometry, rightHand);
            if (state->lockedValid[i] && state->lockedFeature[i].rightHand == rightHand) {
                ContactCandidate locked = EvaluateFeature(kCoilTips[i], geometry, state->lockedFeature[i]);
                const bool keepLocked =
                    locked.valid &&
                    (state->lockTime[i] < state->lockPersistence ||
                     locked.score <= candidate.score + state->lockSlack ||
                     state->phase[i] == ArcPhase::Contact ||
                     state->phase[i] == ArcPhase::PlasmaHold);
                if (keepLocked) candidate = locked;
            }
            candidates[i] = candidate;
            secondaryCandidates[i] = CollectSecondaryContacts(
                kCoilTips[i], geometry, rightHand, candidate, state->voltage, &secondaryCounts[i]);
            compositeScore += candidate.score;
        }
        compositeScore -= 0.10f * std::min(candidates[0].score, candidates[1].score);
        compositeScore -= 0.05f * static_cast<float>(secondaryCounts[0] + secondaryCounts[1]);

        if (!best.valid || compositeScore < best.compositeScore) {
            best.valid = true;
            best.rightHand = rightHand;
            best.geometry = &geometry;
            best.control = &control;
            best.candidates = candidates;
            best.secondaryCandidates = secondaryCandidates;
            best.secondaryCounts = secondaryCounts;
            best.compositeScore = compositeScore;
        }
    };

    considerTrackedHand(false);
    considerTrackedHand(true);
    return best;
}

void UpdateGapCalibration(CoilSceneState* state, bool handValid, float gap, float dt) {
    if (!handValid) return;
    state->nearGapObserved = std::min(state->nearGapObserved + 0.18f * dt, gap);
    state->farGapObserved = std::max(state->farGapObserved - 0.24f * dt, gap);
    state->nearGapObserved = std::min(state->nearGapObserved, state->farGapObserved - 0.55f);
}

void UpdatePhase(CoilSceneState* state, int coilIndex, bool handValid, float proximityTarget, float reactionTarget, float coronaTarget, float dt) {
    ArcPhase next = ArcPhase::Idle;
    if (handValid && state->voltage > 0.08f) {
        if (reactionTarget > 0.72f && state->reaction[coilIndex] > 0.58f) {
            next = ArcPhase::PlasmaHold;
        } else if (reactionTarget > 0.10f) {
            next = ArcPhase::Contact;
        } else if (coronaTarget > 0.54f && proximityTarget > 0.12f) {
            next = ArcPhase::Seeking;
        } else if (coronaTarget > 0.08f) {
            next = ArcPhase::Corona;
        }
    }

    if (next == state->phase[coilIndex]) {
        state->phaseTime[coilIndex] += dt;
    } else {
        state->phase[coilIndex] = next;
        state->phaseTime[coilIndex] = 0.0f;
    }
    state->active[coilIndex] = (state->phase[coilIndex] == ArcPhase::Contact || state->phase[coilIndex] == ArcPhase::PlasmaHold);
}

void DrawTeslaCoilModel(const Vector3& tip, float voltage, float reaction, float t) {
    const float pulse = 0.50f + 0.50f * std::sin(t * 3.8f);
    const float energized = 0.18f + 0.82f * std::max(voltage, reaction);

    const float sx = tip.x;
    const float sz = tip.z;
    DrawCylinderEx({sx, 0.12f, sz}, {sx, 0.64f, sz}, 1.16f, 0.90f, 28, Color{34, 36, 46, 255});
    DrawCylinderEx({sx, 0.64f, sz}, {sx, 0.94f, sz}, 0.84f, 0.76f, 24, Color{50, 56, 72, 255});
    DrawCylinderEx({sx, 0.94f, sz}, {sx, 2.50f, sz}, 0.34f, 0.30f, 18, Color{82, 94, 128, 255});
    DrawCylinderEx({sx, 2.50f, sz}, {sx, 2.82f, sz}, 0.46f, 0.38f, 18, Color{106, 120, 158, 255});
    DrawCylinderEx({sx, 2.82f, sz}, {sx, 2.90f, sz}, 0.16f, 0.14f, 12, Color{164, 170, 186, 255});

    for (int i = 0; i < 20; ++i) {
        const float y = 0.98f + static_cast<float>(i) * 0.073f;
        const Color copper = ColorLerp(Color{154, 98, 56, 255}, Color{255, 176, 92, 255}, 0.15f + 0.56f * energized);
        DrawCylinderWiresEx({sx, y, sz}, {sx, y + 0.02f, sz}, 0.48f, 0.48f, 30, copper);
    }

    for (int i = 0; i < 28; ++i) {
        const float angle = static_cast<float>(i) / 28.0f * PI * 2.0f;
        const float rx = 0.66f;
        const float rz = 0.36f;
        const Vector3 toroid = {
            tip.x + std::cos(angle) * rx,
            tip.y,
            tip.z + std::sin(angle) * rz,
        };
        DrawSphere(toroid, 0.10f, ColorLerp(Color{174, 180, 194, 255}, Color{255, 228, 150, 255}, 0.12f + 0.56f * energized));
    }

    DrawSphere(tip, 0.11f, Color{255, 248, 234, 255});
    DrawSphere(tip, 0.16f + 0.03f * pulse, Fade(Color{255, 232, 184, 255}, 0.18f + 0.20f * energized));

    const std::array<Vector3, 4> support = {{
        {sx - 0.61f, 0.82f, sz - 0.34f},
        {sx + 0.59f, 0.82f, sz - 0.34f},
        {sx - 0.61f, 0.82f, sz + 0.34f},
        {sx + 0.59f, 0.82f, sz + 0.34f},
    }};
    for (const Vector3& s : support) {
        DrawCylinderEx(s, {sx + 0.18f * (s.x > sx ? 1.0f : -1.0f), 2.62f, sz + (s.z - sz) * 0.35f}, 0.05f, 0.04f, 12, Color{118, 126, 148, 255});
    }
}

void DrawGroundGrid() {
    DrawPlane({0.0f, kFloorY, 0.0f}, {18.0f, 12.0f}, Color{24, 26, 34, 255});
    for (int x = -9; x <= 9; ++x) {
        DrawLine3D({static_cast<float>(x), kFloorY + 0.002f, -6.0f}, {static_cast<float>(x), kFloorY + 0.002f, 6.0f}, Fade(Color{82, 92, 118, 255}, 0.24f));
    }
    for (int z = -6; z <= 6; ++z) {
        DrawLine3D({-9.0f, kFloorY + 0.002f, static_cast<float>(z)}, {9.0f, kFloorY + 0.002f, static_cast<float>(z)}, Fade(Color{82, 92, 118, 255}, 0.24f));
    }
}

void DrawArcPath(Vector3 start, Vector3 end, float intensity, float t, int seed) {
    const Vector3 delta = Vector3Subtract(end, start);
    const float length = Vector3Length(delta);
    if (length < 1.0e-4f) return;

    const Vector3 dir = Vector3Normalize(delta);
    Vector3 side = SafeNormalize(Vector3CrossProduct(dir, {0.0f, 1.0f, 0.0f}), {0.0f, 0.0f, 1.0f});
    Vector3 up = SafeNormalize(Vector3CrossProduct(side, dir), {0.0f, 1.0f, 0.0f});
    const ArcPalette palette = PaletteForArc(seed, intensity);

    const int segments = ScaleDetail(10 + static_cast<int>(length * (2.4f + 1.8f * intensity)), 0.52f);
    const float jag = 0.02f + 0.12f * intensity;
    Vector3 prev = start;
    Vector3 prevOuterA = start;
    Vector3 prevOuterB = start;
    Vector3 prevOuterC = start;
    for (int i = 1; i <= segments; ++i) {
        const float s = static_cast<float>(i) / static_cast<float>(segments);
        Vector3 point = Vector3Lerp(start, end, s);
        if (i < segments) {
            const float env = std::sin(s * PI);
            const float wobbleA = std::sin(t * (7.0f + 0.85f * seed) + 11.0f * s + 6.2831f * std::fmod(seed * 0.173f, 1.0f));
            const float wobbleB = std::cos(t * (8.6f + 0.65f * seed) + 9.0f * s + 6.2831f * std::fmod(seed * 0.287f, 1.0f));
            point = Vector3Add(point, Vector3Scale(side, wobbleA * jag * env));
            point = Vector3Add(point, Vector3Scale(up, wobbleB * jag * 0.70f * env));
        }

        const float shellRadius = (0.010f + 0.050f * intensity) * std::sin(s * PI);
        const float angle = t * (6.6f + 0.20f * seed) + 12.0f * s;
        const Vector3 outerOffsetA = Vector3Add(
            Vector3Scale(side, std::cos(angle) * shellRadius),
            Vector3Scale(up, std::sin(angle) * shellRadius * 0.76f));
        const Vector3 outerOffsetB = Vector3Add(
            Vector3Scale(side, std::cos(angle + 2.2f) * shellRadius * 0.82f),
            Vector3Scale(up, std::sin(angle + 2.2f) * shellRadius));
        const Vector3 outerOffsetC = Vector3Add(
            Vector3Scale(side, std::cos(angle + 4.4f) * shellRadius * 0.65f),
            Vector3Scale(up, std::sin(angle + 4.4f) * shellRadius * 0.90f));
        const Vector3 outerA = Vector3Add(point, outerOffsetA);
        const Vector3 outerB = Vector3Add(point, outerOffsetB);
        const Vector3 outerC = Vector3Add(point, outerOffsetC);

        DrawLine3D(prevOuterA, outerA, palette.outer);
        if (gRenderQuality > 0.56f || intensity > 0.64f) DrawLine3D(prevOuterB, outerB, palette.outer);
        if (gRenderQuality > 0.82f || intensity > 0.82f) DrawLine3D(prevOuterC, outerC, palette.haze);
        DrawLine3D(prev, point, palette.mid);
        DrawLine3D(prev, point, palette.core);

        DrawSphere(point, 0.026f + 0.045f * intensity, palette.haze);
        if (gRenderQuality > 0.60f || intensity > 0.70f) DrawSphere(point, 0.014f + 0.025f * intensity, palette.outer);
        DrawSphere(point, 0.006f + 0.016f * intensity, palette.core);
        if (gRenderQuality > 0.70f && i < segments && (i % 2 == 0 || intensity > 0.72f)) {
            const Vector3 mist = Vector3Add(point, Vector3Scale(outerOffsetA, 1.3f));
            DrawSphere(mist, 0.010f + 0.020f * intensity, palette.haze);
        }

        prev = point;
        prevOuterA = outerA;
        prevOuterB = outerB;
        prevOuterC = outerC;
    }
}

void DrawBranchFilaments(Vector3 start, Vector3 end, float intensity, float t, int seed) {
    const Vector3 delta = Vector3Subtract(end, start);
    const float length = Vector3Length(delta);
    if (length < 1.0e-4f) return;

    const Vector3 dir = Vector3Normalize(delta);
    Vector3 side = SafeNormalize(Vector3CrossProduct(dir, {0.0f, 1.0f, 0.0f}), {0.0f, 0.0f, 1.0f});
    Vector3 up = SafeNormalize(Vector3CrossProduct(side, dir), {0.0f, 1.0f, 0.0f});
    const int branches = ScaleDetail(1 + static_cast<int>(intensity * 6.0f), 0.45f);

    for (int i = 0; i < branches; ++i) {
        const float s = 0.14f + 0.72f * static_cast<float>(i + 1) / static_cast<float>(branches + 1);
        const Vector3 anchor = Vector3Lerp(start, end, s);
        const float angle = t * (4.2f + 0.18f * seed) + 9.0f * s + static_cast<float>(i) * 1.8f;
        Vector3 branchDir = Vector3Add(
            Vector3Scale(side, std::sin(angle)),
            Vector3Scale(up, std::cos(angle * 1.17f)));
        branchDir = SafeNormalize(Vector3Add(branchDir, Vector3Scale(dir, 0.25f)), side);
        const float reach = length * (0.05f + 0.10f * intensity) * (0.75f + 0.25f * std::sin(angle * 0.8f));
        DrawArcPath(anchor, Vector3Add(anchor, Vector3Scale(branchDir, reach)), 0.12f + 0.36f * intensity, t, seed + 200 + i * 11);
    }
}

void DrawArcCluster(Vector3 start, Vector3 end, float intensity, float t, int seed, bool plasmaHold = false) {
    const Vector3 delta = Vector3Subtract(end, start);
    const float length = Vector3Length(delta);
    if (length < 1.0e-4f) return;

    const Vector3 dir = Vector3Normalize(delta);
    Vector3 side = SafeNormalize(Vector3CrossProduct(dir, {0.0f, 1.0f, 0.0f}), {0.0f, 0.0f, 1.0f});
    Vector3 up = SafeNormalize(Vector3CrossProduct(side, dir), {0.0f, 1.0f, 0.0f});

    DrawArcPath(start, end, Clamp01(0.22f + 0.95f * intensity), t, seed);

    const int shells = ScaleDetail(1 + static_cast<int>(intensity * 3.0f) + (plasmaHold ? 2 : 0), 0.50f);
    const float shellScale = plasmaHold ? 0.10f : 0.07f;
    for (int i = 0; i < shells; ++i) {
        const float angle = t * (3.6f + 0.22f * seed) + static_cast<float>(i) * (2.0f * PI / static_cast<float>(std::max(1, shells)));
        const float spread = (0.012f + shellScale * intensity) * (0.65f + 0.35f * std::sin(angle * 1.4f));
        const Vector3 offset = Vector3Add(
            Vector3Scale(side, std::cos(angle) * spread),
            Vector3Scale(up, std::sin(angle) * spread * (plasmaHold ? 1.20f : 0.82f)));
        const Vector3 startOffset = Vector3Add(start, Vector3Scale(offset, 0.26f));
        const Vector3 endOffset = Vector3Add(end, Vector3Scale(offset, -0.72f));
        DrawArcPath(startOffset, endOffset, Clamp01(0.16f + 0.72f * intensity), t, seed + 31 + i * 7);
    }

    if (gRenderQuality > 0.48f || intensity > 0.58f) {
        DrawBranchFilaments(start, end, intensity * (plasmaHold ? 1.15f : 0.85f), t, seed + 91);
    }
}

void DrawArcAtmosphere(Vector3 start, Vector3 end, float intensity, float t, int seed, bool plasmaHold = false) {
    if (gRenderQuality < 0.46f && intensity < 0.72f) return;
    const Vector3 delta = Vector3Subtract(end, start);
    const float length = Vector3Length(delta);
    if (length < 1.0e-4f) return;

    const Vector3 dir = Vector3Normalize(delta);
    Vector3 side = SafeNormalize(Vector3CrossProduct(dir, {0.0f, 1.0f, 0.0f}), {0.0f, 0.0f, 1.0f});
    Vector3 up = SafeNormalize(Vector3CrossProduct(side, dir), {0.0f, 1.0f, 0.0f});
    const ArcPalette palette = PaletteForArc(seed + 13, intensity, plasmaHold);
    const int puffs = ScaleDetail(4 + static_cast<int>(intensity * 7.0f), 0.45f);

    for (int i = 1; i < puffs; ++i) {
        const float s = static_cast<float>(i) / static_cast<float>(puffs);
        const float env = std::sin(s * PI);
        const float angle = t * (1.8f + 0.06f * seed) + 7.0f * s + static_cast<float>(i) * 0.9f;
        const float radius = (0.05f + 0.22f * intensity) * env * (plasmaHold ? 1.35f : 1.0f);
        Vector3 p = Vector3Lerp(start, end, s);
        p = Vector3Add(p, Vector3Scale(side, std::cos(angle) * radius));
        p = Vector3Add(p, Vector3Scale(up, std::sin(angle) * radius * 0.65f));
        DrawSphere(p, 0.016f + 0.030f * intensity * env, palette.haze);
        if ((gRenderQuality > 0.70f || intensity > 0.80f) && i % 2 == 0) {
            DrawSphere(Vector3Lerp(p, end, 0.18f), 0.010f + 0.018f * intensity, palette.outer);
        }
    }
}

void DrawGroundIonReflection(Vector3 point, float intensity, int seed) {
    const float spread = 0.10f + 0.26f * intensity;
    const ArcPalette palette = PaletteForArc(seed, intensity, intensity > 0.72f);
    const int glowCount = (gRenderQuality > 0.68f) ? 3 : 2;
    for (int i = 0; i < glowCount; ++i) {
        const float angle = static_cast<float>(i) * 1.1f + 0.4f * seed;
        const Vector3 glow = {
            point.x + std::cos(angle) * spread * (0.45f + 0.16f * i),
            kFloorY + 0.010f,
            point.z + std::sin(angle) * spread * (0.30f + 0.12f * i),
        };
        DrawSphere(glow, 0.030f + 0.060f * intensity, Fade(palette.haze, 0.65f));
    }
}

void DrawCorona(Vector3 origin, float voltage, float reaction, float t) {
    const int streamerCount = ScaleDetail(3 + static_cast<int>(voltage * 7.0f), 0.55f);
    for (int i = 0; i < streamerCount; ++i) {
        const float az = static_cast<float>(i) / static_cast<float>(streamerCount) * PI * 2.0f;
        const float elev = 0.25f + 0.35f * std::sin(t * 1.7f + static_cast<float>(i));
        const float reach = 0.14f + 0.28f * voltage + 0.24f * reaction;
        const Vector3 tip = {
            origin.x + std::cos(az) * reach * 0.85f,
            origin.y + elev * reach,
            origin.z + std::sin(az) * reach * 0.45f,
        };
        DrawArcCluster(origin, tip, 0.14f + 0.46f * voltage, t, 100 + i);
        if (voltage > 0.36f && (gRenderQuality > 0.58f || reaction > 0.60f)) {
            DrawArcAtmosphere(origin, tip, 0.08f + 0.18f * voltage, t, 140 + i);
        }
    }

    const ArcPalette palette = PaletteForArc(static_cast<int>(origin.x * 20.0f), Clamp01(voltage + reaction));
    DrawSphere(origin, 0.18f + 0.12f * voltage, palette.haze);
}

void DrawImpactHalo(Vector3 point, float reaction, float t) {
    const float pulse = 0.50f + 0.50f * std::sin(t * 8.5f);
    DrawSphere(point, 0.050f + 0.06f * reaction, Fade(Color{255, 214, 152, 255}, 0.14f + 0.10f * reaction));
    DrawSphere(point, 0.020f + 0.018f * reaction, Color{255, 248, 238, 255});
    DrawSphere(point, 0.10f + 0.14f * reaction + 0.02f * pulse, Fade(Color{255, 146, 84, 255}, 0.08f + 0.08f * reaction));
}

void DrawSkinCrawl(const ActiveHandRef& hand, const ContactCandidate& candidate, float reaction, bool active, float t) {
    if (!hand.valid || !active || candidate.feature.kind != FeatureKind::Capsule) return;
    const Vector3 a = hand.geometry->landmarks[static_cast<size_t>(candidate.feature.a)];
    const Vector3 b = hand.geometry->landmarks[static_cast<size_t>(candidate.feature.b)];
    const Vector3 towardRoot = Vector3Lerp(candidate.point, a, 0.68f);
    DrawArcPath(candidate.point, towardRoot, 0.18f + 0.40f * reaction, t, 210);
    if (reaction > 0.46f) {
        const Vector3 towardTip = Vector3Lerp(candidate.point, b, 0.40f + 0.16f * std::sin(t * 5.2f));
        DrawArcPath(candidate.point, towardTip, 0.16f + 0.28f * reaction, t, 260);
    }
}

void DrawHeatOverlay(const ActiveHandRef& hand, const ContactCandidate& candidate, float reaction, bool active) {
    if (!hand.valid || !active) return;
    const float halo = 0.05f + 0.08f * reaction;
    DrawSphere(candidate.point, halo, Fade(Color{255, 154, 86, 255}, 0.22f));
    DrawSphere(candidate.point, halo * 0.45f, Fade(Color{255, 230, 184, 255}, 0.30f));

    if (candidate.feature.kind == FeatureKind::Capsule) {
        const Vector3 a = hand.geometry->landmarks[static_cast<size_t>(candidate.feature.a)];
        const Vector3 b = hand.geometry->landmarks[static_cast<size_t>(candidate.feature.b)];
        DrawLine3D(a, candidate.point, Fade(Color{255, 146, 86, 255}, 0.55f));
        DrawLine3D(candidate.point, b, Fade(Color{255, 120, 76, 255}, 0.25f));
        DrawSphere(a, 0.018f + 0.020f * reaction, Fade(Color{255, 160, 100, 255}, 0.22f));
    }
}

void DrawPlasmaBloom(Vector3 point, float reaction, float t, int seed) {
    const ArcPalette palette = PaletteForArc(seed + 40, reaction, reaction > 0.72f);
    const float pulse = 0.50f + 0.50f * std::sin(t * (6.0f + 0.4f * seed));
    DrawSphere(point, 0.08f + 0.18f * reaction, palette.haze);
    DrawSphere(point, 0.04f + 0.11f * reaction, palette.ember);
    DrawSphere(point, 0.02f + 0.05f * reaction + 0.01f * pulse, palette.core);
}

void DrawIonizedPath(Vector3 start, Vector3 end, float reaction, float t, int seed) {
    const int beads = 5 + static_cast<int>(reaction * 8.0f);
    for (int i = 1; i < beads; ++i) {
        const float s = static_cast<float>(i) / static_cast<float>(beads);
        Vector3 p = Vector3Lerp(start, end, s);
        p.y += 0.04f * std::sin(t * 7.0f + static_cast<float>(seed) + 9.0f * s);
        DrawSphere(p, 0.006f + 0.010f * reaction, Fade(Color{255, 210, 144, 255}, 0.18f));
    }
}

void DrawContactPlasma(Vector3 point, float reaction, float t, int seed) {
    const ArcPalette palette = PaletteForArc(seed + 70, reaction, true);
    const int filaments = ScaleDetail(4 + static_cast<int>(reaction * 8.0f), 0.55f);
    for (int i = 0; i < filaments; ++i) {
        const float angle = static_cast<float>(i) / static_cast<float>(filaments) * 2.0f * PI + t * (1.6f + 0.08f * seed);
        const float lift = 0.18f + 0.36f * std::sin(t * 3.1f + static_cast<float>(i) * 0.8f);
        const float radius = 0.03f + 0.12f * reaction * (0.55f + 0.45f * std::sin(t * 5.7f + static_cast<float>(i)));
        const Vector3 shell = {
            point.x + std::cos(angle) * radius,
            point.y + std::sin(angle * 1.4f) * radius * 0.35f + radius * 0.18f * lift,
            point.z + std::sin(angle) * radius * 0.70f,
        };
        DrawLine3D(point, shell, palette.outer);
        if (gRenderQuality > 0.60f || reaction > 0.72f) DrawSphere(shell, 0.010f + 0.018f * reaction, palette.mid);
    }

    const float halo = 0.08f + 0.18f * reaction;
    DrawSphere(point, halo * 1.20f, palette.haze);
    DrawSphere(point, halo, palette.ember);
    DrawSphere(point, halo * 0.62f, palette.core);
}

void DrawIntercoilField(const ActiveHandRef& hand, const CoilSceneState& state, float t) {
    const float pairReaction = std::min(state.reaction[0], state.reaction[1]);
    if (state.active[0] && state.active[1]) {
        DrawArcCluster(hand.candidates[0].point, hand.candidates[1].point, 0.24f + 0.56f * pairReaction, t, 150, true);
        DrawArcAtmosphere(hand.candidates[0].point, hand.candidates[1].point, 0.16f + 0.44f * pairReaction, t, 157, true);
        DrawArcCluster(kCoilTips[0], hand.candidates[1].point, 0.14f + 0.30f * pairReaction, t, 171);
        DrawArcCluster(kCoilTips[1], hand.candidates[0].point, 0.14f + 0.30f * pairReaction, t, 177);
        DrawPlasmaBloom(Vector3Lerp(hand.candidates[0].point, hand.candidates[1].point, 0.5f), 0.5f * (state.reaction[0] + state.reaction[1]), t, 199);
        return;
    }

    const float bridgeReaction = Clamp01(state.voltage - 0.70f) * (0.35f + 0.65f * std::max(state.corona[0], state.corona[1]));
    if (bridgeReaction > 0.02f) {
        DrawArcCluster(kCoilTips[0], kCoilTips[1], 0.18f + 0.54f * bridgeReaction, t, 233, bridgeReaction > 0.38f);
        DrawArcAtmosphere(kCoilTips[0], kCoilTips[1], 0.10f + 0.28f * bridgeReaction, t, 249, bridgeReaction > 0.38f);
        DrawIonizedPath(kCoilTips[0], kCoilTips[1], bridgeReaction, t, 241);
    }
}

void DrawReactionScene(const ActiveHandRef& hand, const CoilSceneState& state, float t) {
    for (size_t i = 0; i < kCoilTips.size(); ++i) {
        DrawTeslaCoilModel(kCoilTips[i], state.voltage, state.reaction[i], t + static_cast<float>(i) * 0.3f);
        DrawCorona(kCoilTips[i], state.voltage, std::max(state.corona[i], state.reaction[i]), t + static_cast<float>(i) * 0.17f);
    }

    if (!hand.valid) {
        DrawIntercoilField(hand, state, t);
        return;
    }

    for (size_t i = 0; i < kCoilTips.size(); ++i) {
        const ContactCandidate& candidate = hand.candidates[i];
        if (state.phase[i] == ArcPhase::Seeking) {
            const Vector3 seekPoint = Vector3Lerp(kCoilTips[i], candidate.point, 0.48f + 0.08f * std::sin(t * (4.6f + 0.2f * i)));
            DrawArcCluster(kCoilTips[i], seekPoint, 0.16f + 0.24f * state.corona[i], t, 18 + static_cast<int>(i));
            DrawArcAtmosphere(kCoilTips[i], seekPoint, 0.10f + 0.18f * state.corona[i], t, 31 + static_cast<int>(i));
        }
        if (!state.active[i]) continue;

        const bool plasmaHold = state.phase[i] == ArcPhase::PlasmaHold;
        DrawArcCluster(kCoilTips[i], candidate.point, Clamp01(0.24f + 0.92f * state.reaction[i]), t, 1 + static_cast<int>(i) * 12, plasmaHold);
        DrawArcAtmosphere(kCoilTips[i], candidate.point, Clamp01(0.18f + 0.82f * state.reaction[i]), t, 61 + static_cast<int>(i) * 17, plasmaHold);
        DrawIonizedPath(kCoilTips[i], candidate.point, state.reaction[i], t, 20 + static_cast<int>(i));
        if (IsTipFeature(candidate.feature)) {
            const Vector3 tipLead = Vector3Lerp(kCoilTips[i], candidate.point, 0.78f);
            DrawArcCluster(kCoilTips[i], tipLead, Clamp01(0.18f + 0.60f * state.reaction[i]), t, 11 + static_cast<int>(i) * 7);
            DrawArcAtmosphere(kCoilTips[i], tipLead, Clamp01(0.14f + 0.36f * state.reaction[i]), t, 101 + static_cast<int>(i) * 11);
        }
        if (plasmaHold) {
            const Vector3 offsetA = {candidate.point.x, candidate.point.y + 0.10f * state.reaction[i], candidate.point.z + 0.08f * std::sin(t * 5.1f + static_cast<float>(i))};
            const Vector3 offsetB = {candidate.point.x, candidate.point.y - 0.06f, candidate.point.z - 0.10f * std::cos(t * 6.2f + static_cast<float>(i))};
            DrawArcCluster(kCoilTips[i], offsetA, 0.44f * state.reaction[i], t, 2 + static_cast<int>(i) * 9, true);
            DrawArcCluster(kCoilTips[i], offsetB, 0.34f * state.reaction[i], t, 3 + static_cast<int>(i) * 9, true);
            DrawArcAtmosphere(kCoilTips[i], offsetA, 0.24f + 0.34f * state.reaction[i], t, 131 + static_cast<int>(i) * 13, true);
            DrawArcAtmosphere(kCoilTips[i], offsetB, 0.18f + 0.28f * state.reaction[i], t, 151 + static_cast<int>(i) * 13, true);
        }

        DrawSkinCrawl(hand, candidate, state.reaction[i], state.active[i], t);
        DrawImpactHalo(candidate.point, state.reaction[i], t);
        DrawPlasmaBloom(candidate.point, state.reaction[i], t, static_cast<int>(i));
        DrawContactPlasma(candidate.point, state.reaction[i], t, static_cast<int>(i));
        DrawHeatOverlay(hand, candidate, state.reaction[i], state.active[i]);
        DrawGroundIonReflection(candidate.point, state.reaction[i], static_cast<int>(i) + 400);

        const float multiFingerStrength = Clamp01((state.reaction[i] - 0.18f) / 0.72f) * Clamp01(0.25f + state.voltage);
        const int allowedSecondary =
            std::min(hand.secondaryCounts[i], std::max(0, static_cast<int>(std::ceil(multiFingerStrength * static_cast<float>(kMaxSecondaryContacts)))));
        for (int j = 0; j < allowedSecondary; ++j) {
            const ContactCandidate& secondary = hand.secondaryCandidates[i][static_cast<size_t>(j)];
            const float secondaryFalloff = 1.0f - 0.18f * static_cast<float>(j);
            const float secondaryReaction = Clamp01((0.18f + 0.58f * multiFingerStrength) * secondaryFalloff);
            const bool secondaryHold = plasmaHold && j == 0;
            DrawArcCluster(kCoilTips[i], secondary.point, secondaryReaction, t, 320 + static_cast<int>(i) * 31 + j * 7, secondaryHold);
            if (secondaryReaction > 0.24f) {
                DrawArcAtmosphere(kCoilTips[i], secondary.point, 0.12f + 0.34f * secondaryReaction, t, 360 + static_cast<int>(i) * 17 + j * 5, secondaryHold);
            }
            DrawArcCluster(candidate.point, secondary.point, 0.12f + 0.42f * secondaryReaction, t, 390 + static_cast<int>(i) * 13 + j * 9, secondaryHold);
            DrawPlasmaBloom(secondary.point, 0.50f * secondaryReaction, t, 430 + static_cast<int>(i) * 9 + j);
            DrawContactPlasma(secondary.point, 0.34f * secondaryReaction, t, 460 + static_cast<int>(i) * 9 + j);
            DrawHeatOverlay(hand, secondary, 0.42f * secondaryReaction, true);
            if (secondaryReaction > 0.20f) {
                DrawSkinCrawl(hand, secondary, 0.58f * secondaryReaction, true, t);
            }
        }
    }

    DrawIntercoilField(hand, state, t);

    if ((state.active[0] || state.active[1]) && state.voltage > 0.66f && !(state.active[0] && state.active[1])) {
        const size_t hot = state.active[0] ? 0 : 1;
        const size_t cold = hot == 0 ? 1 : 0;
        const Vector3 bridgeMid = Vector3Lerp(hand.candidates[hot].point, kCoilTips[cold], 0.44f);
        DrawArcCluster(hand.candidates[hot].point, bridgeMid, 0.20f + 0.36f * state.reaction[hot], t, 222);
    }
}

void DrawVoltageMeter2D(const CoilSceneState& state) {
    const Rectangle panel = {1360.0f, 332.0f, 34.0f, 228.0f};
    DrawRectangleRounded({panel.x - 20.0f, panel.y - 18.0f, panel.width + 40.0f, panel.height + 36.0f}, 0.16f, 12, Fade(BLACK, 0.40f));
    DrawRectangleLinesEx(panel, 2.0f, Color{104, 120, 158, 255});
    const float fill = Clamp01(state.voltage);
    const int fillH = static_cast<int>((panel.height - 8.0f) * fill);
    if (fillH > 0) {
        DrawRectangleGradientV(
            static_cast<int>(panel.x + 4.0f),
            static_cast<int>(panel.y + panel.height - 4.0f - fillH),
            static_cast<int>(panel.width - 8.0f),
            fillH,
            Color{255, 154, 76, 255},
            Color{116, 220, 255, 255});
    }
    const float requiredY = panel.y + panel.height - panel.height * Clamp01(std::min(state.requiredVoltage[0], state.requiredVoltage[1]));
    const float reactionY = panel.y + panel.height - panel.height * Clamp01(std::max(state.reaction[0], state.reaction[1]));
    DrawLineEx({panel.x - 10.0f, requiredY}, {panel.x + panel.width + 10.0f, requiredY}, 2.0f, Color{255, 198, 124, 255});
    DrawLineEx({panel.x - 10.0f, reactionY}, {panel.x + panel.width + 10.0f, reactionY}, 2.0f, Color{150, 244, 255, 255});
    DrawText("kV", static_cast<int>(panel.x - 4.0f), static_cast<int>(panel.y - 34.0f), 22, Color{228, 236, 248, 255});
}

void DrawHud(const ActiveHandRef& hand, const CoilSceneState& state) {
    DrawRectangleRounded({18.0f, 772.0f, 360.0f, 84.0f}, 0.10f, 12, Fade(BLACK, 0.28f));
    DrawRectangleLinesEx({18.0f, 772.0f, 360.0f, 84.0f}, 1.0f, Color{86, 104, 136, 255});

    const float maxReaction = std::max(state.reaction[0], state.reaction[1]);
    const ArcPhase phase = (state.phase[0] > state.phase[1]) ? state.phase[0] : state.phase[1];
    const std::string handLine = hand.valid ? std::string(hand.rightHand ? "right hand" : "left hand") : "no hand";
    const std::string contactLine = "L " + state.contactLabel[0] + "   R " + state.contactLabel[1];

    std::ostringstream summary;
    summary.setf(std::ios::fixed);
    summary.precision(2);
    summary << "voltage " << state.voltage
            << "   react " << maxReaction
            << "   " << PhaseName(phase);

    DrawText("Dual Coil Plasma Lab", 34, 786, 24, Color{232, 238, 248, 255});
    DrawText(handLine.c_str(), 34, 814, 18, Color{194, 210, 234, 255});
    DrawText(contactLine.c_str(), 138, 814, 18, Color{255, 214, 150, 255});
    DrawText(summary.str().c_str(), 34, 834, 18, Color{138, 224, 255, 255});
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Tesla Coil Hand Bridge Lab");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {8.6f, 5.4f, 8.8f};
    camera.target = kCameraTarget;
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 38.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float yaw = 0.90f;
    float pitch = -0.24f;
    float distance = 12.6f;

    HandSceneBridge bridge;
    bridge.Start();

    CoilAudioEngine audio;
    audio.Start();

    CoilSceneState state{};
    bool showLandmarks = false;

    while (!WindowShouldClose()) {
        const float now = static_cast<float>(GetTime());
        const float dt = std::min(GetFrameTime(), 0.05f);
        const float targetQuality = std::clamp(1.24f - 14.0f * std::max(0.0f, dt - (1.0f / 70.0f)), 0.46f, 1.0f);
        state.renderQuality = LerpFloat(state.renderQuality, targetQuality, 1.0f - std::exp(-3.0f * dt));
        gRenderQuality = state.renderQuality;

        if (IsKeyPressed(KEY_L)) showLandmarks = !showLandmarks;
        if (IsKeyPressed(KEY_EQUAL)) state.voltageTarget = std::clamp(state.voltageTarget + 0.08f, 0.0f, 1.0f);
        if (IsKeyPressed(KEY_MINUS)) state.voltageTarget = std::clamp(state.voltageTarget - 0.08f, 0.0f, 1.0f);
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) state.voltageTarget = std::clamp(state.voltageTarget + 0.02f, 0.0f, 1.0f);
        if (IsKeyPressed(KEY_LEFT_BRACKET)) state.voltageTarget = std::clamp(state.voltageTarget - 0.02f, 0.0f, 1.0f);
        if (IsKeyPressed(KEY_SEMICOLON)) {
            state.lockPersistence = std::clamp(state.lockPersistence - 0.05f, 0.20f, 1.10f);
            state.lockSlack = std::clamp(state.lockSlack - 0.015f, 0.03f, 0.28f);
        }
        if (IsKeyPressed(KEY_APOSTROPHE)) {
            state.lockPersistence = std::clamp(state.lockPersistence + 0.05f, 0.20f, 1.10f);
            state.lockSlack = std::clamp(state.lockSlack + 0.015f, 0.03f, 0.28f);
        }
        if (IsKeyPressed(KEY_COMMA)) state.audioHarshness = std::clamp(state.audioHarshness - 0.08f, 0.0f, 1.0f);
        if (IsKeyPressed(KEY_PERIOD)) state.audioHarshness = std::clamp(state.audioHarshness + 0.08f, 0.0f, 1.0f);
        if (IsKeyPressed(KEY_ZERO)) state.voltageTarget = 0.35f;
        if (IsKeyPressed(KEY_C)) {
            state.nearGapObserved = 0.26f;
            state.farGapObserved = 2.60f;
        }

        bridge.Update(camera, now, dt);

        const ActiveHandRef hand = PickActiveHand(bridge, &state);
        const bool handValid = hand.valid;
        state.voltage = LerpFloat(state.voltage, state.voltageTarget, 1.0f - std::exp(-5.5f * dt));

        for (size_t i = 0; i < kCoilTips.size(); ++i) {
            const float gap = handValid ? hand.candidates[i].gap : state.farGapObserved;
            UpdateGapCalibration(&state, handValid, gap, dt);

            const float nearGap = std::clamp(state.nearGapObserved + 0.02f, 0.04f, 0.55f);
            const float farGap = std::clamp(std::max(state.farGapObserved, nearGap + 0.90f), 1.10f, 3.60f);
            const float gapNorm = handValid ? Clamp01((gap - nearGap) / (farGap - nearGap)) : 1.0f;
            const float proximityTarget = handValid ? (1.0f - gapNorm) : 0.0f;

            state.proximity[i] = LerpFloat(state.proximity[i], proximityTarget, 1.0f - std::exp(-9.0f * dt));
            state.requiredVoltage[i] = 0.12f + 0.82f * std::pow(gapNorm, 1.18f);

            const float coronaTarget = Clamp01((state.voltage - (state.requiredVoltage[i] - 0.22f)) / 0.30f);
            const float reactionTarget = handValid
                ? Clamp01((state.voltage - state.requiredVoltage[i]) / 0.26f) * (0.55f + 0.45f * proximityTarget)
                : 0.0f;

            state.corona[i] = LerpFloat(state.corona[i], coronaTarget, 1.0f - std::exp(-7.0f * dt));
            state.reaction[i] = LerpFloat(state.reaction[i], reactionTarget, 1.0f - std::exp(-11.0f * dt));
            UpdatePhase(&state, static_cast<int>(i), handValid, proximityTarget, reactionTarget, coronaTarget, dt);

            if (handValid) {
                if (state.lockedValid[i] && SameFeature(state.lockedFeature[i], hand.candidates[i].feature)) {
                    state.lockTime[i] += dt;
                } else {
                    state.lockedFeature[i] = hand.candidates[i].feature;
                    state.lockedValid[i] = true;
                    state.lockTime[i] = 0.0f;
                }
                state.contactLabel[i] = hand.candidates[i].feature.label;
            } else {
                state.lockedValid[i] = false;
                state.lockTime[i] = 0.0f;
                state.contactLabel[i] = "none";
            }
        }

        Vector3 desiredTarget = kCameraTarget;
        if (handValid) {
            const Vector3 center = Vector3Scale(Vector3Add(hand.candidates[0].point, hand.candidates[1].point), 0.5f);
            desiredTarget = Vector3Lerp(kCameraTarget, center, 0.34f);
            desiredTarget.y = LerpFloat(kCameraTarget.y, desiredTarget.y, 0.68f);
            desiredTarget.z = LerpFloat(kCameraTarget.z, desiredTarget.z, 0.68f);
        }
        camera.target = Vector3Lerp(camera.target, desiredTarget, 1.0f - std::exp(-3.8f * dt));
        UpdateOrbitCameraDragOnly(&camera, &yaw, &pitch, &distance);

        audio.Update(state);

        BeginDrawing();
        ClearBackground(Color{8, 10, 16, 255});
        DrawRectangleGradientV(0, 0, kScreenWidth, kScreenHeight, Color{10, 16, 26, 255}, Color{3, 4, 8, 255});
        DrawStarfieldBackdrop(static_cast<int>(80 + 80 * state.renderQuality), 0x5AC31Fu, now * 0.35f, Color{188, 216, 255, 255});
        DrawCircleGradient(212, 182, 250.0f, Fade(Color{46, 86, 174, 255}, 0.16f + 0.06f * state.corona[0]), Fade(BLACK, 0.0f));
        DrawCircleGradient(1228, 182, 250.0f, Fade(Color{58, 104, 192, 255}, 0.16f + 0.06f * state.corona[1]), Fade(BLACK, 0.0f));
        DrawCircleGradient(720, 260, 280.0f, Fade(Color{255, 124, 70, 255}, 0.02f + 0.05f * std::max(state.reaction[0], state.reaction[1])), Fade(BLACK, 0.0f));

        BeginMode3D(camera);
        DrawGroundGrid();
        bridge.DrawHands(showLandmarks);
        DrawReactionScene(hand, state, now);
        EndMode3D();

        bridge.DrawPreviewPanel({1002.0f, 18.0f, 406.0f, 288.0f}, "Python Bridge Preview");
        DrawVoltageMeter2D(state);
        DrawHud(hand, state);
        EndDrawing();
    }

    audio.Shutdown();
    bridge.Shutdown();
    CloseWindow();
    return 0;
}
