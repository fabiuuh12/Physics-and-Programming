#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kScreenWidth = 1440;
constexpr int kScreenHeight = 900;
constexpr int kGridX = 92;
constexpr int kGridZ = 92;
constexpr float kXMin = -9.0f;
constexpr float kXMax = 9.0f;
constexpr float kZMin = -9.0f;
constexpr float kZMax = 9.0f;

enum class DemoMode {
    kVacuum = 0,
    kTraveling = 1,
    kCollision = 2,
};

enum class PacketStage {
    kForming = 0,
    kStable = 1,
    kDecaying = 2,
};

struct OrbitCameraState {
    float yaw = 0.86f;
    float pitch = 0.42f;
    float distance = 20.0f;
};

struct CameraPreset {
    float yaw = 0.0f;
    float pitch = 0.0f;
    float distance = 0.0f;
};

struct TravelingExcitation {
    Vector2 pos{};
    Vector2 vel{};
    float amplitude = 0.0f;
    float sigma = 0.0f;
    float wavelength = 0.0f;
    float omega = 0.0f;
    float phase = 0.0f;
    float life = 0.0f;
    float age = 0.0f;
    float trailPhase = 0.0f;
    Color color{};
};

struct StablePacket {
    Vector2 pos{};
    Vector2 drift{};
    float amplitude = 0.0f;
    float sigma = 0.0f;
    float omega = 0.0f;
    float phase = 0.0f;
    float life = 0.0f;
    float age = 0.0f;
    float stageAge = 0.0f;
    float decayDuration = 0.0f;
    float transitionGlow = 0.0f;
    int level = 0;
    PacketStage stage = PacketStage::kForming;
    Color color{};
};

struct ShockRing {
    Vector2 center{};
    float radius = 0.0f;
    float speed = 0.0f;
    float width = 0.0f;
    float amplitude = 0.0f;
    float life = 0.0f;
    float age = 0.0f;
    Color color{};
};

struct Spark {
    Vector3 pos{};
    Vector3 vel{};
    float size = 0.0f;
    float life = 0.0f;
    float age = 0.0f;
    Color color{};
};

struct VacuumFluctuation {
    Vector2 center{};
    float amplitude = 0.0f;
    float sigma = 0.0f;
    float omega = 0.0f;
    float phase = 0.0f;
    float drift = 0.0f;
};

struct QuantizedPacketPreset {
    float amplitude = 0.0f;
    float sigma = 0.0f;
    float life = 0.0f;
    float coreRadius = 0.0f;
    float haloRadius = 0.0f;
    int orbitCount = 0;
    Color color{};
    const char* label = "";
};

struct Metrics {
    float primaryEnergy = 0.0f;
    float secondaryEnergy = 0.0f;
    float transfer = 0.0f;
    float localizedEnergy = 0.0f;
};

struct InspectInfo {
    bool hasPacket = false;
    int level = 0;
    PacketStage stage = PacketStage::kStable;
    float amplitude = 0.0f;
    float sigma = 0.0f;
    float stageWeight = 0.0f;
};

constexpr std::array<QuantizedPacketPreset, 3> kPacketPresets = {{
    {1.35f, 0.86f, 11.0f, 0.18f, 0.48f, 2, Color{255, 210, 120, 255}, "n=1"},
    {1.85f, 1.06f, 14.0f, 0.22f, 0.62f, 3, Color{255, 156, 218, 255}, "n=2"},
    {2.35f, 1.26f, 17.0f, 0.27f, 0.80f, 4, Color{124, 244, 255, 255}, "n=3"},
}};

CameraPreset PresetForMode(DemoMode mode) {
    switch (mode) {
        case DemoMode::kVacuum: return {0.70f, 0.54f, 24.0f};
        case DemoMode::kTraveling: return {0.90f, 0.40f, 18.5f};
        case DemoMode::kCollision: return {1.02f, 0.26f, 13.8f};
    }
    return {0.86f, 0.42f, 20.0f};
}

void SnapCameraToPreset(OrbitCameraState* orbit, DemoMode mode) {
    const CameraPreset preset = PresetForMode(mode);
    orbit->yaw = preset.yaw;
    orbit->pitch = preset.pitch;
    orbit->distance = preset.distance;
}

float PacketFormationDuration(int level) {
    return 1.05f + 0.18f * static_cast<float>(level);
}

float PacketDecayDuration(int level) {
    return 1.45f + 0.24f * static_cast<float>(level);
}

int FieldIndex(int ix, int iz) {
    return iz * kGridX + ix;
}

float RandRange(std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    return dist(rng);
}

float Saturate(float x) {
    return std::clamp(x, 0.0f, 1.0f);
}

Color LerpColor(Color a, Color b, float t) {
    const float u = Saturate(t);
    return Color{
        static_cast<unsigned char>(a.r + (b.r - a.r) * u),
        static_cast<unsigned char>(a.g + (b.g - a.g) * u),
        static_cast<unsigned char>(a.b + (b.b - a.b) * u),
        static_cast<unsigned char>(a.a + (b.a - a.a) * u),
    };
}

float PacketStageWeight(const StablePacket& packet) {
    if (packet.stage == PacketStage::kForming) {
        return Saturate(packet.stageAge / PacketFormationDuration(packet.level));
    }
    if (packet.stage == PacketStage::kDecaying) {
        return 1.0f - Saturate(packet.stageAge / packet.decayDuration);
    }
    return 1.0f;
}

float WrappedPhaseDelta(float a, float b) {
    return std::atan2(std::sin(a - b), std::cos(a - b));
}

float CollisionResonanceScore(const TravelingExcitation& a, const TravelingExcitation& b) {
    const float phaseAlign = 1.0f - std::fabs(WrappedPhaseDelta(a.phase, b.phase + PI)) / PI;
    const Vector2 dirA = Vector2Normalize(a.vel);
    const Vector2 dirB = Vector2Normalize(b.vel);
    const float opposing = Saturate((-Vector2DotProduct(dirA, dirB) + 1.0f) * 0.5f);
    const float energyMatch = 1.0f - Saturate(std::fabs(a.amplitude - b.amplitude) / 1.4f);
    return 0.45f * phaseAlign + 0.35f * opposing + 0.20f * energyMatch;
}

float PacketAbsorptionScore(const StablePacket& packet, const TravelingExcitation& excitation) {
    const float phaseAlign = 1.0f - std::fabs(WrappedPhaseDelta(packet.phase, excitation.phase)) / PI;
    const float scaleMatch = 1.0f - Saturate(std::fabs(packet.sigma - excitation.sigma) / 1.3f);
    return 0.62f * phaseAlign + 0.38f * scaleMatch;
}

void UpdateOrbitCameraDragOnly(Camera3D* camera, OrbitCameraState* orbit) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        const Vector2 delta = GetMouseDelta();
        orbit->yaw -= delta.x * 0.0036f;
        orbit->pitch += delta.y * 0.0036f;
        orbit->pitch = std::clamp(orbit->pitch, -1.30f, 1.30f);
    }

    orbit->distance -= GetMouseWheelMove() * 0.9f;
    orbit->distance = std::clamp(orbit->distance, 7.0f, 34.0f);

    const float cp = std::cos(orbit->pitch);
    camera->target = {0.0f, 0.0f, 0.0f};
    camera->position = Vector3Add(camera->target, {
        orbit->distance * cp * std::cos(orbit->yaw),
        orbit->distance * std::sin(orbit->pitch),
        orbit->distance * cp * std::sin(orbit->yaw),
    });
}

const char* DemoModeName(DemoMode mode) {
    switch (mode) {
        case DemoMode::kVacuum: return "Vacuum";
        case DemoMode::kTraveling: return "Traveling Excitation";
        case DemoMode::kCollision: return "Collision -> Localized Packet";
    }
    return "Unknown";
}

const char* DemoModeSummary(DemoMode mode) {
    switch (mode) {
        case DemoMode::kVacuum: return "Low-amplitude vacuum fluctuations keep both fields alive.";
        case DemoMode::kTraveling: return "Energy moves as packets across the field surface.";
        case DemoMode::kCollision: return "Collisions localize energy into discrete packet states.";
    }
    return "";
}

float Gaussian2D(const Vector2& delta, float sigma) {
    const float r2 = delta.x * delta.x + delta.y * delta.y;
    return std::exp(-r2 / (2.0f * sigma * sigma));
}

float VacuumPrimary(float x, float z, float time) {
    const float radial = std::sqrt(x * x + z * z);
    const float breathing = 0.05f * std::sin(0.55f * radial - 1.2f * time);
    const float lattice =
        0.05f * std::sin(0.72f * x - 1.1f * time) +
        0.04f * std::cos(0.63f * z + 0.95f * time) +
        0.03f * std::sin(0.42f * (x + z) + 1.8f * time) +
        0.02f * std::cos(1.10f * x - 0.84f * z - 2.1f * time);
    return breathing + lattice;
}

float VacuumSecondary(float x, float z, float time) {
    const float radial = std::sqrt(x * x + z * z);
    const float breathing = 0.04f * std::cos(0.48f * radial + 1.05f * time);
    const float lattice =
        0.04f * std::cos(0.56f * x + 1.25f * time) +
        0.03f * std::sin(0.74f * z - 0.82f * time) +
        0.03f * std::sin(0.36f * (x - z) + 1.5f * time);
    return breathing + lattice;
}

std::vector<VacuumFluctuation> MakeVacuumFluctuations() {
    std::mt19937 rng(4242);
    std::vector<VacuumFluctuation> fluctuations;
    fluctuations.reserve(18);

    for (int i = 0; i < 18; ++i) {
        fluctuations.push_back({
            {RandRange(rng, -7.4f, 7.4f), RandRange(rng, -7.4f, 7.4f)},
            RandRange(rng, 0.035f, 0.085f),
            RandRange(rng, 0.55f, 1.35f),
            RandRange(rng, 1.6f, 4.6f),
            RandRange(rng, 0.0f, 2.0f * PI),
            RandRange(rng, 0.15f, 0.85f),
        });
    }

    return fluctuations;
}

float VacuumLocalizedContribution(const VacuumFluctuation& fluctuation, float x, float z, float time, bool coupled) {
    const float wobbleX = 0.38f * std::sin(time * fluctuation.drift + fluctuation.phase);
    const float wobbleZ = 0.38f * std::cos(time * (0.82f * fluctuation.drift) - fluctuation.phase * 0.6f);
    const Vector2 center = {
        fluctuation.center.x + wobbleX,
        fluctuation.center.y + wobbleZ,
    };

    const Vector2 delta = {x - center.x, z - center.y};
    const float envelope = Gaussian2D(delta, fluctuation.sigma);
    const float radial = std::sqrt(delta.x * delta.x + delta.y * delta.y);
    const float phase = fluctuation.omega * time + fluctuation.phase + (coupled ? 0.8f : 0.0f);
    const float ripple = std::sin(4.8f * radial - phase) + 0.55f * std::cos(2.8f * radial + 1.2f * phase);
    return fluctuation.amplitude * envelope * ripple;
}

float TravelingContribution(const TravelingExcitation& excitation, float x, float z, float time, bool coupled) {
    const Vector2 sample = {x, z};
    const Vector2 delta = Vector2Subtract(sample, excitation.pos);
    const float envelope = Gaussian2D(delta, excitation.sigma);
    Vector2 dir = excitation.vel;
    if (Vector2Length(dir) < 1.0e-5f) dir = {1.0f, 0.0f};
    dir = Vector2Normalize(dir);

    const float proj = Vector2DotProduct(delta, dir);
    const float cross = delta.x * (-dir.y) + delta.y * dir.x;
    const float k = 2.0f * PI / excitation.wavelength;
    const float ageFade = Saturate(1.0f - excitation.age / excitation.life);
    const float phaseShift = coupled ? 0.7f : 0.0f;
    const float coupledSkew = coupled ? 0.22f * std::sin(cross * 1.8f + 0.9f * time) : 0.0f;
    return excitation.amplitude * ageFade * envelope *
           std::cos(k * proj - excitation.omega * time + excitation.phase + phaseShift + coupledSkew);
}

float StableContribution(const StablePacket& packet, float x, float z, float time, bool coupled) {
    const Vector2 sample = {x, z};
    const Vector2 delta = Vector2Subtract(sample, packet.pos);
    const float envelope = Gaussian2D(delta, packet.sigma);
    const float radial = std::sqrt(delta.x * delta.x + delta.y * delta.y);
    const float stageWeight = PacketStageWeight(packet);
    const float breathing = 0.74f + 0.26f * std::sin(packet.omega * time + packet.phase);
    const float phaseOffset = coupled ? 0.9f : 0.0f;
    const float ripple = std::cos((3.4f + 0.5f * packet.level) * radial - 1.08f * packet.omega * time + packet.phase + phaseOffset);
    const float transitionBoost = 1.0f + 0.35f * packet.transitionGlow;
    const float coupledScale = coupled ? 0.90f + 0.24f * packet.level + 0.30f * packet.transitionGlow : 1.0f;

    float formationShell = 0.0f;
    if (packet.stage == PacketStage::kForming) {
        const float formT = Saturate(packet.stageAge / PacketFormationDuration(packet.level));
        const float shellRadius = 1.4f - 0.9f * formT;
        formationShell = (1.0f - formT) * std::exp(-((radial - shellRadius) * (radial - shellRadius)) / (2.0f * 0.14f * 0.14f));
    } else if (packet.stage == PacketStage::kDecaying) {
        const float decayT = Saturate(packet.stageAge / packet.decayDuration);
        formationShell = decayT * std::exp(-((radial - (0.5f + 1.0f * decayT)) * (radial - (0.5f + 1.0f * decayT))) / (2.0f * 0.18f * 0.18f));
    }

    return packet.amplitude * coupledScale * transitionBoost *
           ((0.28f + 0.72f * stageWeight) * breathing * envelope * (0.78f + 0.22f * ripple) +
            0.46f * formationShell);
}

float RingContribution(const ShockRing& ring, float x, float z, bool coupled) {
    const Vector2 sample = {x, z};
    const Vector2 delta = Vector2Subtract(sample, ring.center);
    const float radial = std::sqrt(delta.x * delta.x + delta.y * delta.y);
    const float width = coupled ? ring.width * 1.22f : ring.width;
    const float shell = std::exp(-((radial - ring.radius) * (radial - ring.radius)) / (2.0f * width * width));
    const float fade = Saturate(1.0f - ring.age / ring.life);
    const float phase = coupled ? 1.0f : 0.0f;
    return ring.amplitude * fade * shell * std::sin(4.2f * radial - 2.0f * ring.age + phase);
}

TravelingExcitation MakeTravelingExcitationSafe(Vector2 pos, Vector2 vel, Color color, float phase, float amplitudeScale, std::mt19937* rng) {
    TravelingExcitation excitation{};
    excitation.pos = pos;
    excitation.vel = vel;
    excitation.amplitude = 1.10f * amplitudeScale;
    excitation.sigma = RandRange(*rng, 0.95f, 1.18f);
    excitation.wavelength = RandRange(*rng, 1.55f, 1.92f);
    excitation.omega = RandRange(*rng, 5.1f, 6.2f);
    excitation.phase = phase;
    excitation.life = RandRange(*rng, 7.6f, 9.4f);
    excitation.age = 0.0f;
    excitation.trailPhase = RandRange(*rng, 0.0f, 2.0f * PI);
    excitation.color = color;
    return excitation;
}

StablePacket MakeQuantizedPacket(Vector2 center, int level, Color seedColor, std::mt19937* rng) {
    const QuantizedPacketPreset& preset = kPacketPresets[std::clamp(level, 0, 2)];
    StablePacket packet{};
    packet.pos = center;
    packet.drift = {RandRange(*rng, -0.04f, 0.04f), RandRange(*rng, -0.04f, 0.04f)};
    packet.amplitude = preset.amplitude;
    packet.sigma = preset.sigma;
    packet.omega = 4.2f + 0.75f * static_cast<float>(level);
    packet.phase = RandRange(*rng, 0.0f, 2.0f * PI);
    packet.life = preset.life;
    packet.age = 0.0f;
    packet.stageAge = 0.0f;
    packet.decayDuration = PacketDecayDuration(level);
    packet.transitionGlow = 1.0f;
    packet.level = level;
    packet.stage = PacketStage::kForming;
    packet.color = LerpColor(seedColor, preset.color, 0.65f);
    return packet;
}

void AddShockBurst(std::vector<ShockRing>* rings, std::vector<Spark>* sparks, Vector2 center, int level, std::mt19937* rng);

void PromotePacket(StablePacket* packet, Color inputColor) {
    if (packet->level < 2) ++packet->level;
    const QuantizedPacketPreset& preset = kPacketPresets[packet->level];
    packet->amplitude = preset.amplitude;
    packet->sigma = preset.sigma;
    packet->life = preset.life;
    packet->decayDuration = PacketDecayDuration(packet->level);
    packet->stage = PacketStage::kForming;
    packet->stageAge = 0.0f;
    packet->age = 0.0f;
    packet->transitionGlow = 1.0f;
    packet->color = LerpColor(packet->color, LerpColor(inputColor, preset.color, 0.7f), 0.65f);
}

bool DemotePacket(StablePacket* packet, std::vector<TravelingExcitation>* traveling, std::vector<ShockRing>* rings, std::vector<Spark>* sparks, std::mt19937* rng) {
    if (packet->level > 0) {
        --packet->level;
        const QuantizedPacketPreset& preset = kPacketPresets[packet->level];
        packet->amplitude = preset.amplitude;
        packet->sigma = preset.sigma;
        packet->life = preset.life * 0.7f;
        packet->decayDuration = PacketDecayDuration(packet->level);
        packet->stage = PacketStage::kForming;
        packet->stageAge = 0.0f;
        packet->age = 0.0f;
        packet->transitionGlow = 0.9f;
        packet->color = LerpColor(packet->color, preset.color, 0.55f);
        return false;
    }

    const float baseAngle = RandRange(*rng, 0.0f, 2.0f * PI);
    for (int i = 0; i < 2; ++i) {
        const float angle = baseAngle + i * PI;
        const Vector2 vel = {std::cos(angle) * RandRange(*rng, 2.2f, 3.0f), std::sin(angle) * RandRange(*rng, 2.2f, 3.0f)};
        traveling->push_back(MakeTravelingExcitationSafe(packet->pos, vel, packet->color, RandRange(*rng, 0.0f, 2.0f * PI), 0.82f, rng));
    }
    AddShockBurst(rings, sparks, packet->pos, 0, rng);
    return true;
}

void SpawnCollisionPair(std::vector<TravelingExcitation>* traveling, std::mt19937* rng, float offset) {
    traveling->push_back(MakeTravelingExcitationSafe({kXMin + 1.0f, offset}, {3.0f, 0.0f}, Color{64, 232, 255, 255}, 0.0f, 1.0f, rng));
    traveling->push_back(MakeTravelingExcitationSafe({kXMax - 1.0f, -offset}, {-3.0f, 0.0f}, Color{255, 86, 214, 255}, PI, 1.0f, rng));
}

void SpawnDiagonalPair(std::vector<TravelingExcitation>* traveling, std::mt19937* rng) {
    traveling->push_back(MakeTravelingExcitationSafe({-7.0f, -6.2f}, {2.5f, 2.0f}, Color{112, 255, 178, 255}, 0.5f, 0.95f, rng));
    traveling->push_back(MakeTravelingExcitationSafe({6.8f, 6.0f}, {-2.3f, -2.1f}, Color{255, 180, 96, 255}, 2.8f, 0.95f, rng));
}

void SpawnTravelingTrain(std::vector<TravelingExcitation>* traveling, std::mt19937* rng) {
    const float z = RandRange(*rng, -4.6f, 4.6f);
    const float direction = RandRange(*rng, 0.0f, 1.0f) > 0.5f ? 1.0f : -1.0f;
    const float x = direction > 0.0f ? kXMin + 0.8f : kXMax - 0.8f;
    const Vector2 velocity = {direction * RandRange(*rng, 2.4f, 3.3f), RandRange(*rng, -0.3f, 0.3f)};
    traveling->push_back(MakeTravelingExcitationSafe({x, z}, velocity, Color{90, 226, 255, 255}, RandRange(*rng, 0.0f, 2.0f * PI), 0.9f, rng));
}

void AddShockBurst(std::vector<ShockRing>* rings, std::vector<Spark>* sparks, Vector2 center, int level, std::mt19937* rng) {
    const float levelScale = 1.0f + 0.25f * static_cast<float>(level);
    rings->push_back({center, 0.0f, 4.8f * levelScale, 0.28f, 0.75f * levelScale, 2.6f, 0.0f, Color{92, 230, 255, 255}});
    rings->push_back({center, 0.0f, 6.2f * levelScale, 0.42f, 0.62f * levelScale, 3.0f, 0.0f, Color{255, 110, 214, 255}});
    rings->push_back({center, 0.0f, 7.4f * levelScale, 0.54f, 0.42f * levelScale, 2.4f, 0.0f, Color{255, 214, 128, 255}});

    const int sparkCount = 28 + level * 10;
    for (int i = 0; i < sparkCount; ++i) {
        const float angle = RandRange(*rng, 0.0f, 2.0f * PI);
        const float speed = RandRange(*rng, 1.2f, 6.8f) * levelScale;
        sparks->push_back({
            {center.x, 0.18f, center.y},
            {std::cos(angle) * speed, RandRange(*rng, 0.7f, 2.7f), std::sin(angle) * speed},
            RandRange(*rng, 0.05f, 0.16f),
            RandRange(*rng, 1.0f, 2.3f),
            0.0f,
            i % 2 == 0 ? Color{90, 238, 255, 255} : Color{255, 148, 214, 255},
        });
    }
}

void ResetSceneForMode(DemoMode mode,
                       std::vector<TravelingExcitation>* traveling,
                       std::vector<StablePacket>* stable,
                       std::vector<ShockRing>* rings,
                       std::vector<Spark>* sparks,
                       std::mt19937* rng,
                       int* mergeCount,
                       float* cycleTimer,
                       float* simTime) {
    traveling->clear();
    stable->clear();
    rings->clear();
    sparks->clear();
    *mergeCount = 0;
    *cycleTimer = 0.0f;
    *simTime = 0.0f;

    if (mode == DemoMode::kTraveling) {
        SpawnTravelingTrain(traveling, rng);
        SpawnDiagonalPair(traveling, rng);
    } else if (mode == DemoMode::kCollision) {
        SpawnCollisionPair(traveling, rng, 1.3f);
    }
}

void ApplyStablePacketInteractions(std::vector<StablePacket>* stable, float dt) {
    for (size_t i = 0; i < stable->size(); ++i) {
        for (size_t j = i + 1; j < stable->size(); ++j) {
            StablePacket& a = stable->at(i);
            StablePacket& b = stable->at(j);
            if (a.stage != PacketStage::kStable || b.stage != PacketStage::kStable) continue;

            const Vector2 delta = Vector2Subtract(b.pos, a.pos);
            const float dist = std::max(0.001f, Vector2Length(delta));
            if (dist > 4.8f) continue;

            const Vector2 dir = Vector2Scale(delta, 1.0f / dist);
            const float repulsion = (0.7f + 0.25f * (a.level + b.level)) / (0.9f + dist * dist);
            const float swirl = (a.level == b.level ? 0.08f : 0.04f) / (1.2f + dist);
            const Vector2 tangent = {-dir.y, dir.x};

            a.drift = Vector2Add(a.drift, Vector2Scale(Vector2Subtract(Vector2Scale(tangent, swirl), Vector2Scale(dir, repulsion)), dt));
            b.drift = Vector2Add(b.drift, Vector2Scale(Vector2Add(Vector2Scale(dir, repulsion), Vector2Scale(tangent, swirl)), dt));
        }
    }

    for (StablePacket& packet : *stable) {
        packet.drift = Vector2Scale(packet.drift, 1.0f - 0.22f * dt);
    }
}

void BuildFieldCaches(const std::vector<TravelingExcitation>& traveling,
                      const std::vector<StablePacket>& stable,
                      const std::vector<ShockRing>& rings,
                      const std::vector<VacuumFluctuation>& fluctuations,
                      DemoMode mode,
                      float time,
                      std::vector<float>* primary,
                      std::vector<float>* secondary,
                      Metrics* metrics) {
    primary->assign(kGridX * kGridZ, 0.0f);
    secondary->assign(kGridX * kGridZ, 0.0f);

    float primaryEnergy = 0.0f;
    float secondaryEnergy = 0.0f;
    float transfer = 0.0f;
    float localizedEnergy = 0.0f;

    for (int ix = 0; ix < kGridX; ++ix) {
        for (int iz = 0; iz < kGridZ; ++iz) {
            const float x = kXMin + (kXMax - kXMin) * static_cast<float>(ix) / static_cast<float>(kGridX - 1);
            const float z = kZMin + (kZMax - kZMin) * static_cast<float>(iz) / static_cast<float>(kGridZ - 1);

            float p = VacuumPrimary(x, z, time);
            float s = VacuumSecondary(x, z, time);
            const float fluctuationScale = mode == DemoMode::kVacuum ? 1.0f : 0.22f;
            for (const VacuumFluctuation& fluctuation : fluctuations) {
                p += fluctuationScale * VacuumLocalizedContribution(fluctuation, x, z, time, false);
                s += fluctuationScale * 0.82f * VacuumLocalizedContribution(fluctuation, x, z, time, true);
            }
            for (const TravelingExcitation& excitation : traveling) {
                p += TravelingContribution(excitation, x, z, time, false);
                s += 0.46f * TravelingContribution(excitation, x, z, time, true);
            }
            for (const StablePacket& packet : stable) {
                p += StableContribution(packet, x, z, time, false);
                s += 1.18f * StableContribution(packet, x, z, time, true);
                const Vector2 delta = {x - packet.pos.x, z - packet.pos.y};
                const float radial = std::sqrt(delta.x * delta.x + delta.y * delta.y);
                const float angular = std::atan2(delta.y, delta.x);
                const float stageWeight = PacketStageWeight(packet);
                const float filament = std::exp(-radial * (1.4f - 0.12f * packet.level)) *
                                       std::sin((3.0f + packet.level) * angular - 1.3f * time + packet.phase);
                s += (0.18f + 0.10f * packet.level + 0.18f * packet.transitionGlow) * stageWeight * filament;
                localizedEnergy += packet.amplitude * packet.amplitude * stageWeight;
            }
            for (const ShockRing& ring : rings) {
                p += RingContribution(ring, x, z, false);
                s += 0.70f * RingContribution(ring, x, z, true);
            }

            primary->at(FieldIndex(ix, iz)) = 0.68f * p;
            secondary->at(FieldIndex(ix, iz)) = 0.52f * s;
            primaryEnergy += p * p;
            secondaryEnergy += s * s;
            transfer += std::fabs(p - s);
        }
    }

    const float denom = static_cast<float>(kGridX * kGridZ);
    metrics->primaryEnergy = primaryEnergy / denom;
    metrics->secondaryEnergy = secondaryEnergy / denom;
    metrics->transfer = transfer / denom;
    metrics->localizedEnergy = localizedEnergy / std::max(1.0f, static_cast<float>(stable.size()));
}

void DrawPrimaryFieldSurface(const std::vector<float>& primary) {
    for (int ix = 0; ix < kGridX - 1; ++ix) {
        for (int iz = 0; iz < kGridZ - 1; ++iz) {
            const float x0 = kXMin + (kXMax - kXMin) * static_cast<float>(ix) / static_cast<float>(kGridX - 1);
            const float x1 = kXMin + (kXMax - kXMin) * static_cast<float>(ix + 1) / static_cast<float>(kGridX - 1);
            const float z0 = kZMin + (kZMax - kZMin) * static_cast<float>(iz) / static_cast<float>(kGridZ - 1);
            const float z1 = kZMin + (kZMax - kZMin) * static_cast<float>(iz + 1) / static_cast<float>(kGridZ - 1);

            const float f00 = primary[FieldIndex(ix, iz)];
            const float f10 = primary[FieldIndex(ix + 1, iz)];
            const float f01 = primary[FieldIndex(ix, iz + 1)];
            const float f11 = primary[FieldIndex(ix + 1, iz + 1)];

            const Vector3 p00 = {x0, f00, z0};
            const Vector3 p10 = {x1, f10, z0};
            const Vector3 p01 = {x0, f01, z1};
            const Vector3 p11 = {x1, f11, z1};

            const float intensity = Saturate(0.12f + 0.42f * (std::fabs(f00) + std::fabs(f10) + std::fabs(f01) + std::fabs(f11)));
            const Color cool = LerpColor(Color{18, 24, 60, 255}, Color{86, 238, 255, 255}, intensity);
            const Color hot = LerpColor(cool, Color{255, 88, 214, 255}, Saturate(intensity * 0.8f));

            DrawTriangle3D(p00, p10, p01, Fade(cool, 0.16f + 0.22f * intensity));
            DrawTriangle3D(p10, p11, p01, Fade(hot, 0.14f + 0.20f * intensity));

            if ((ix + iz) % 2 == 0) {
                DrawLine3D(p00, p10, Fade(Color{92, 214, 255, 255}, 0.14f + 0.22f * intensity));
                DrawLine3D(p00, p01, Fade(Color{255, 106, 216, 255}, 0.10f + 0.16f * intensity));
            }
        }
    }
}

void DrawCoupledFieldOverlay(const std::vector<float>& secondary) {
    for (int ix = 0; ix < kGridX - 1; ix += 2) {
        for (int iz = 0; iz < kGridZ - 1; iz += 2) {
            const float x0 = kXMin + (kXMax - kXMin) * static_cast<float>(ix) / static_cast<float>(kGridX - 1);
            const float x1 = kXMin + (kXMax - kXMin) * static_cast<float>(ix + 1) / static_cast<float>(kGridX - 1);
            const float z0 = kZMin + (kZMax - kZMin) * static_cast<float>(iz) / static_cast<float>(kGridZ - 1);
            const float z1 = kZMin + (kZMax - kZMin) * static_cast<float>(iz + 1) / static_cast<float>(kGridZ - 1);

            const float f00 = secondary[FieldIndex(ix, iz)] + 0.12f;
            const float f10 = secondary[FieldIndex(ix + 1, iz)] + 0.12f;
            const float f01 = secondary[FieldIndex(ix, iz + 1)] + 0.12f;
            const float f11 = secondary[FieldIndex(ix + 1, iz + 1)] + 0.12f;

            const Vector3 p00 = {x0, f00, z0};
            const Vector3 p10 = {x1, f10, z0};
            const Vector3 p01 = {x0, f01, z1};
            const Vector3 p11 = {x1, f11, z1};

            const float intensity = Saturate(0.10f + 0.52f * (std::fabs(f00) + std::fabs(f10) + std::fabs(f01) + std::fabs(f11)));
            const Color c0 = LerpColor(Color{255, 162, 94, 255}, Color{255, 224, 158, 255}, intensity);
            const Color c1 = LerpColor(Color{255, 120, 220, 255}, Color{255, 200, 122, 255}, intensity);

            DrawLine3D(p00, p10, Fade(c0, 0.08f + 0.18f * intensity));
            DrawLine3D(p10, p11, Fade(c1, 0.08f + 0.18f * intensity));
            DrawLine3D(p11, p01, Fade(c0, 0.08f + 0.18f * intensity));
            DrawLine3D(p01, p00, Fade(c1, 0.08f + 0.18f * intensity));
        }
    }
}

void DrawShockRings(const std::vector<ShockRing>& rings) {
    for (const ShockRing& ring : rings) {
        const float fade = Saturate(1.0f - ring.age / ring.life);
        const float height = 0.10f + 0.10f * std::sin(ring.age * 4.0f);
        Vector3 previous{};
        bool hasPrevious = false;

        for (int i = 0; i <= 96; ++i) {
            const float angle = (2.0f * PI * i) / 96.0f;
            const Vector3 point = {
                ring.center.x + ring.radius * std::cos(angle),
                height,
                ring.center.y + ring.radius * std::sin(angle),
            };
            if (hasPrevious) {
                DrawLine3D(previous, point, Fade(ring.color, 0.32f * fade));
            }
            previous = point;
            hasPrevious = true;
        }
    }
}

void DrawTravelingExcitations(const std::vector<TravelingExcitation>& traveling, float time) {
    for (const TravelingExcitation& excitation : traveling) {
        const float fade = Saturate(1.0f - excitation.age / excitation.life);
        const float pulse = 0.5f + 0.5f * std::sin(excitation.omega * time + excitation.phase);
        const float y = 0.44f + 0.18f * pulse;
        const Vector3 center = {excitation.pos.x, y, excitation.pos.y};
        const float core = 0.12f + 0.05f * fade;
        const Color halo = LerpColor(excitation.color, Color{255, 255, 255, 255}, 0.18f);

        for (int g = 1; g <= 4; ++g) {
            const float ghostT = static_cast<float>(g) / 4.0f;
            const Vector3 ghost = {
                center.x - excitation.vel.x * 0.20f * g,
                center.y - 0.03f * g,
                center.z - excitation.vel.y * 0.20f * g,
            };
            DrawSphere(ghost, core * (1.4f - 0.18f * g), Fade(excitation.color, 0.04f * fade * (1.0f - ghostT)));
        }

        DrawSphere(center, core * 3.2f, Fade(halo, 0.06f + 0.04f * fade));
        DrawSphere(center, core * 1.7f, Fade(excitation.color, 0.12f + 0.08f * fade));
        DrawSphere(center, core, Fade(excitation.color, 0.95f * fade));

        Vector2 dir2 = excitation.vel;
        if (Vector2Length(dir2) < 1.0e-5f) dir2 = {1.0f, 0.0f};
        dir2 = Vector2Normalize(dir2);
        for (int i = 1; i <= 5; ++i) {
            const float trailT = static_cast<float>(i) / 5.0f;
            const Vector3 tail = {
                center.x - dir2.x * (0.45f + 0.45f * trailT),
                center.y - 0.02f * i,
                center.z - dir2.y * (0.45f + 0.45f * trailT),
            };
            DrawLine3D(center, tail, Fade(excitation.color, 0.08f * fade * (1.0f - trailT)));
        }
    }
}

void DrawStablePackets(const std::vector<StablePacket>& stable, float time) {
    for (const StablePacket& packet : stable) {
        const QuantizedPacketPreset& preset = kPacketPresets[packet.level];
        const float fade = PacketStageWeight(packet);
        const float pulse = 0.55f + 0.45f * std::sin(packet.omega * time + packet.phase);
        const Vector3 center = {packet.pos.x, 0.62f + 0.28f * pulse, packet.pos.y};
        const float transitionBoost = 1.0f + 0.45f * packet.transitionGlow;
        for (int g = 1; g <= 3; ++g) {
            const float ghostT = static_cast<float>(g) / 3.0f;
            const Vector3 ghostCenter = {
                center.x - packet.drift.x * 1.3f * static_cast<float>(g),
                center.y - 0.04f * g,
                center.z - packet.drift.y * 1.3f * static_cast<float>(g),
            };
            DrawSphere(ghostCenter, preset.haloRadius * (1.1f - 0.12f * g), Fade(packet.color, 0.03f * fade * (1.0f - ghostT)));
        }

        if (packet.stage == PacketStage::kForming) {
            const float formT = Saturate(packet.stageAge / PacketFormationDuration(packet.level));
            const float shellRadius = 1.3f - 0.88f * formT + 0.08f * pulse;
            const float beamHeight = 1.4f - 0.5f * formT;
            DrawCylinderEx({packet.pos.x, 0.02f, packet.pos.y},
                           {packet.pos.x, beamHeight, packet.pos.y},
                           0.08f + 0.03f * packet.level,
                           0.02f,
                           10,
                           Fade(packet.color, 0.14f + 0.12f * (1.0f - formT)));
            Vector3 previous{};
            bool hasPrevious = false;
            for (int seg = 0; seg <= 60; ++seg) {
                const float angle = (2.0f * PI * seg) / 60.0f + time * 1.2f;
                const Vector3 point = {
                    packet.pos.x + shellRadius * std::cos(angle),
                    0.20f + 0.28f * (1.0f - formT),
                    packet.pos.y + shellRadius * std::sin(angle),
                };
                if (hasPrevious) {
                    DrawLine3D(previous, point, Fade(packet.color, 0.22f * (1.0f - formT)));
                }
                previous = point;
                hasPrevious = true;
            }
        }

        DrawCylinderEx({packet.pos.x, 0.02f, packet.pos.y},
                       {packet.pos.x, 0.68f + 0.18f * pulse, packet.pos.y},
                       0.03f + 0.02f * packet.level,
                       0.01f,
                       8,
                       Fade(packet.color, 0.18f * fade * transitionBoost));

        DrawSphere(center, preset.haloRadius * 1.65f * (1.0f + 0.20f * pulse), Fade(packet.color, 0.04f + 0.08f * fade));
        DrawSphere(center, preset.haloRadius * (1.0f + 0.18f * pulse), Fade(packet.color, 0.10f + 0.08f * fade));
        DrawSphere(center, preset.haloRadius * 0.58f * (1.0f + 0.16f * pulse), Fade(LerpColor(packet.color, WHITE, 0.35f), 0.22f + 0.14f * fade));
        DrawSphere(center, preset.coreRadius * (1.0f + 0.24f * pulse) * transitionBoost, Fade(LerpColor(packet.color, WHITE, 0.50f), 0.98f * fade));

        for (int orbit = 0; orbit < preset.orbitCount; ++orbit) {
            const float radius = 0.48f + 0.22f * orbit + 0.04f * pulse;
            const float elevation = 0.08f + 0.08f * orbit;
            Vector3 previous{};
            bool hasPrevious = false;
            for (int seg = 0; seg <= 52; ++seg) {
                const float angle = (2.0f * PI * seg) / 52.0f + time * (0.4f + 0.18f * orbit);
                const Vector3 point = {
                    packet.pos.x + radius * std::cos(angle),
                    elevation,
                    packet.pos.y + radius * std::sin(angle),
                };
                if (hasPrevious) {
                    DrawLine3D(previous, point, Fade(packet.color, (packet.stage == PacketStage::kDecaying ? 0.06f : 0.11f) * fade));
                }
                previous = point;
                hasPrevious = true;
            }

            for (int sat = 0; sat < 2 + orbit; ++sat) {
                const float a = time * (1.1f + 0.2f * orbit) + sat * (2.0f * PI / (2.0f + orbit));
                const Vector3 satellite = {
                    packet.pos.x + radius * std::cos(a),
                    elevation,
                    packet.pos.y + radius * std::sin(a),
                };
                DrawSphere(satellite, 0.03f + 0.008f * orbit, Fade(packet.color, 0.68f * fade));
            }
        }

        if (packet.stage == PacketStage::kDecaying) {
            const float decayT = Saturate(packet.stageAge / packet.decayDuration);
            Vector3 previous{};
            bool hasPrevious = false;
            const float radius = 0.55f + 1.15f * decayT;
            for (int seg = 0; seg <= 48; ++seg) {
                const float angle = (2.0f * PI * seg) / 48.0f - time * 0.8f;
                const Vector3 point = {
                    packet.pos.x + radius * std::cos(angle),
                    0.12f + 0.20f * decayT,
                    packet.pos.y + radius * std::sin(angle),
                };
                if (hasPrevious) {
                    DrawLine3D(previous, point, Fade(LerpColor(packet.color, Color{255, 214, 130, 255}, 0.5f), 0.18f * (1.0f - decayT)));
                }
                previous = point;
                hasPrevious = true;
            }
        }
    }
}

void DrawSparks(const std::vector<Spark>& sparks) {
    for (const Spark& spark : sparks) {
        const float fade = Saturate(1.0f - spark.age / spark.life);
        const Vector3 tail = Vector3Subtract(spark.pos, Vector3Scale(spark.vel, 0.05f));
        DrawLine3D(tail, spark.pos, Fade(spark.color, 0.20f * fade));
        DrawSphere(spark.pos, spark.size, Fade(spark.color, 0.85f * fade));
    }
}

InspectInfo BuildInspectInfo(const std::vector<StablePacket>& stable) {
    InspectInfo info{};
    if (stable.empty()) return info;

    const StablePacket* best = &stable.front();
    for (const StablePacket& packet : stable) {
        if (packet.level > best->level || (packet.level == best->level && packet.transitionGlow > best->transitionGlow)) {
            best = &packet;
        }
    }

    info.hasPacket = true;
    info.level = best->level;
    info.stage = best->stage;
    info.amplitude = best->amplitude;
    info.sigma = best->sigma;
    info.stageWeight = PacketStageWeight(*best);
    return info;
}

const char* PacketStageName(PacketStage stage) {
    switch (stage) {
        case PacketStage::kForming: return "forming";
        case PacketStage::kStable: return "stable";
        case PacketStage::kDecaying: return "de-exciting";
    }
    return "";
}

void DrawLabelTag(const char* text, Vector3 world, const Camera3D& camera, Color color) {
    const Vector2 screen = GetWorldToScreen(world, camera);
    if (screen.x < 30.0f || screen.x > static_cast<float>(GetScreenWidth() - 30) ||
        screen.y < 30.0f || screen.y > static_cast<float>(GetScreenHeight() - 30)) {
        return;
    }

    const int fontSize = 16;
    const int width = MeasureText(text, fontSize) + 18;
    const int x = static_cast<int>(screen.x) - width / 2;
    const int y = static_cast<int>(screen.y) - 28;
    DrawRectangleRounded({static_cast<float>(x), static_cast<float>(y), static_cast<float>(width), 24.0f}, 0.28f, 8, Fade(BLACK, 0.52f));
    DrawText(text, x + 9, y + 4, fontSize, color);
}

void DrawSceneLabels(const Camera3D& camera,
                     DemoMode mode,
                     const std::vector<TravelingExcitation>& traveling,
                     const std::vector<StablePacket>& stable,
                     const std::vector<ShockRing>& rings) {
    DrawLabelTag("field surface", {5.4f, -0.55f, 5.6f}, camera, Color{112, 232, 255, 255});
    DrawLabelTag("coupled field", {-5.0f, 0.32f, -5.6f}, camera, Color{255, 184, 118, 255});

    if (!traveling.empty() && mode != DemoMode::kVacuum) {
        const TravelingExcitation& excitation = traveling.front();
        DrawLabelTag("traveling excitation", {excitation.pos.x, 0.88f, excitation.pos.y}, camera, excitation.color);
    }
    if (!stable.empty()) {
        const StablePacket& packet = stable.front();
        DrawLabelTag("localized particle-like state", {packet.pos.x, 1.45f, packet.pos.y}, camera, packet.color);
    }
    if (!rings.empty()) {
        const ShockRing& ring = rings.front();
        DrawLabelTag("energy injection", {ring.center.x + ring.radius, 0.35f, ring.center.y}, camera, ring.color);
    }
}

}  // namespace

int main() {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(kScreenWidth, kScreenHeight, "Field Excitations 3D - C++ (raylib)");
    SetWindowMinSize(1100, 700);
    SetTargetFPS(120);

    Camera3D camera{};
    camera.position = {13.0f, 8.0f, 14.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 46.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    OrbitCameraState orbit{};
    std::vector<TravelingExcitation> traveling;
    std::vector<StablePacket> stable;
    std::vector<ShockRing> rings;
    std::vector<Spark> sparks;
    const std::vector<VacuumFluctuation> fluctuations = MakeVacuumFluctuations();
    std::vector<float> primary(kGridX * kGridZ, 0.0f);
    std::vector<float> secondary(kGridX * kGridZ, 0.0f);

    DemoMode mode = DemoMode::kCollision;
    std::mt19937 rng(9001);
    int mergeCount = 0;
    float simTime = 0.0f;
    float simSpeed = 1.0f;
    float cycleTimer = 0.0f;
    bool paused = false;
    bool slowMotion = false;
    bool autoDrive = true;
    bool inspectOverlay = false;
    Metrics metrics{};

    ResetSceneForMode(mode, &traveling, &stable, &rings, &sparks, &rng, &mergeCount, &cycleTimer, &simTime);
    SnapCameraToPreset(&orbit, mode);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_ONE)) {
            mode = DemoMode::kVacuum;
            ResetSceneForMode(mode, &traveling, &stable, &rings, &sparks, &rng, &mergeCount, &cycleTimer, &simTime);
            SnapCameraToPreset(&orbit, mode);
        }
        if (IsKeyPressed(KEY_TWO)) {
            mode = DemoMode::kTraveling;
            ResetSceneForMode(mode, &traveling, &stable, &rings, &sparks, &rng, &mergeCount, &cycleTimer, &simTime);
            SnapCameraToPreset(&orbit, mode);
        }
        if (IsKeyPressed(KEY_THREE)) {
            mode = DemoMode::kCollision;
            ResetSceneForMode(mode, &traveling, &stable, &rings, &sparks, &rng, &mergeCount, &cycleTimer, &simTime);
            SnapCameraToPreset(&orbit, mode);
        }

        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_T)) slowMotion = !slowMotion;
        if (IsKeyPressed(KEY_A)) autoDrive = !autoDrive;
        if (IsKeyPressed(KEY_I)) inspectOverlay = !inspectOverlay;
        if (IsKeyPressed(KEY_G)) SnapCameraToPreset(&orbit, mode);
        if (IsKeyPressed(KEY_R)) {
            paused = false;
            simSpeed = 1.0f;
            ResetSceneForMode(mode, &traveling, &stable, &rings, &sparks, &rng, &mergeCount, &cycleTimer, &simTime);
            SnapCameraToPreset(&orbit, mode);
        }
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) simSpeed = std::min(4.0f, simSpeed + 0.2f);
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) simSpeed = std::max(0.2f, simSpeed - 0.2f);
        if (IsKeyPressed(KEY_C)) SpawnCollisionPair(&traveling, &rng, RandRange(rng, -2.2f, 2.2f));
        if (IsKeyPressed(KEY_V)) SpawnTravelingTrain(&traveling, &rng);
        if (IsKeyPressed(KEY_SPACE)) {
            const int level = static_cast<int>(RandRange(rng, 0.0f, 2.999f));
            const Vector2 center = {RandRange(rng, -2.6f, 2.6f), RandRange(rng, -2.6f, 2.6f)};
            stable.push_back(MakeQuantizedPacket(center, level, Color{255, 214, 132, 255}, &rng));
            AddShockBurst(&rings, &sparks, center, level, &rng);
            ++mergeCount;
        }

        const float timeScale = slowMotion ? 0.24f : 1.0f;
        const float dt = paused ? 0.0f : GetFrameTime() * simSpeed * timeScale;
        simTime += dt;
        cycleTimer += dt;
        UpdateOrbitCameraDragOnly(&camera, &orbit);

        if (autoDrive && mode == DemoMode::kVacuum && cycleTimer > 2.5f) {
            cycleTimer = 0.0f;
            const Vector2 center = {RandRange(rng, -4.2f, 4.2f), RandRange(rng, -4.2f, 4.2f)};
            rings.push_back({center, 0.0f, RandRange(rng, 1.4f, 2.6f), 0.24f, 0.10f, 1.8f, 0.0f, Color{110, 220, 255, 255}});
        } else if (autoDrive && mode == DemoMode::kTraveling && cycleTimer > 2.2f) {
            cycleTimer = 0.0f;
            SpawnTravelingTrain(&traveling, &rng);
            if (RandRange(rng, 0.0f, 1.0f) > 0.62f) SpawnDiagonalPair(&traveling, &rng);
        } else if (autoDrive && mode == DemoMode::kCollision && cycleTimer > 3.7f) {
            cycleTimer = 0.0f;
            if (RandRange(rng, 0.0f, 1.0f) > 0.4f) {
                SpawnCollisionPair(&traveling, &rng, 1.8f * std::sin(simTime * 0.7f));
            } else {
                SpawnDiagonalPair(&traveling, &rng);
            }
        }

        for (TravelingExcitation& excitation : traveling) {
            excitation.pos = Vector2Add(excitation.pos, Vector2Scale(excitation.vel, dt));
            excitation.age += dt;
        }

        for (StablePacket& packet : stable) {
            packet.pos = Vector2Add(packet.pos, Vector2Scale(packet.drift, dt));
            packet.stageAge += dt;
            packet.transitionGlow = std::max(0.0f, packet.transitionGlow - dt * 0.8f);
            if (packet.stage == PacketStage::kForming) {
                if (packet.stageAge >= PacketFormationDuration(packet.level)) {
                    packet.stage = PacketStage::kStable;
                    packet.stageAge = 0.0f;
                    packet.age = 0.0f;
                }
            } else if (packet.stage == PacketStage::kStable) {
                packet.age += dt;
                if (packet.age >= packet.life) {
                    packet.stage = PacketStage::kDecaying;
                    packet.stageAge = 0.0f;
                    packet.decayDuration = PacketDecayDuration(packet.level);
                    AddShockBurst(&rings, &sparks, packet.pos, packet.level, &rng);
                }
            }
        }

        ApplyStablePacketInteractions(&stable, dt);

        for (ShockRing& ring : rings) {
            ring.radius += ring.speed * dt;
            ring.age += dt;
        }

        for (Spark& spark : sparks) {
            spark.pos = Vector3Add(spark.pos, Vector3Scale(spark.vel, dt));
            spark.vel.y -= 1.8f * dt;
            spark.age += dt;
        }

        if (mode == DemoMode::kCollision) {
            std::vector<int> consumedByPackets;
            for (size_t i = 0; i < traveling.size(); ++i) {
                if (std::find(consumedByPackets.begin(), consumedByPackets.end(), static_cast<int>(i)) != consumedByPackets.end()) continue;
                for (StablePacket& packet : stable) {
                    if (packet.stage != PacketStage::kStable) continue;
                    const float captureRadius = 0.65f + 0.22f * packet.level;
                    if (Vector2Distance(traveling[i].pos, packet.pos) < captureRadius) {
                        const float absorption = PacketAbsorptionScore(packet, traveling[i]);
                        if (absorption > 0.62f) {
                            PromotePacket(&packet, traveling[i].color);
                            AddShockBurst(&rings, &sparks, packet.pos, packet.level, &rng);
                            consumedByPackets.push_back(static_cast<int>(i));
                            ++mergeCount;
                            break;
                        }

                        if (absorption < 0.32f) {
                            traveling[i].vel = Vector2Scale(traveling[i].vel, -0.85f);
                            traveling[i].phase += PI * 0.4f;
                            rings.push_back({packet.pos, 0.0f, 2.6f, 0.16f, 0.12f, 0.8f, 0.0f, Color{255, 170, 110, 255}});
                        }
                    }
                }
            }

            std::vector<int> consumed;
            for (size_t i = 0; i < traveling.size(); ++i) {
                if (std::find(consumedByPackets.begin(), consumedByPackets.end(), static_cast<int>(i)) != consumedByPackets.end()) continue;
                if (std::find(consumed.begin(), consumed.end(), static_cast<int>(i)) != consumed.end()) continue;
                for (size_t j = i + 1; j < traveling.size(); ++j) {
                    if (std::find(consumedByPackets.begin(), consumedByPackets.end(), static_cast<int>(j)) != consumedByPackets.end()) continue;
                    if (std::find(consumed.begin(), consumed.end(), static_cast<int>(j)) != consumed.end()) continue;

                    const float dist = Vector2Distance(traveling[i].pos, traveling[j].pos);
                    if (dist < 1.28f) {
                        const Vector2 center = Vector2Scale(Vector2Add(traveling[i].pos, traveling[j].pos), 0.5f);
                        const float energyProxy = std::fabs(traveling[i].amplitude) + std::fabs(traveling[j].amplitude);
                        const float resonance = CollisionResonanceScore(traveling[i], traveling[j]);
                        if (resonance > 0.58f) {
                            const int level = resonance > 0.86f && energyProxy > 2.2f ? 2 : (resonance > 0.70f ? 1 : 0);
                            const Color mergedColor = LerpColor(traveling[i].color, traveling[j].color, 0.5f);
                            stable.push_back(MakeQuantizedPacket(center, level, mergedColor, &rng));
                            AddShockBurst(&rings, &sparks, center, level, &rng);
                            consumed.push_back(static_cast<int>(i));
                            consumed.push_back(static_cast<int>(j));
                            ++mergeCount;
                        } else {
                            traveling[i].vel = Vector2Rotate(traveling[i].vel, 0.45f);
                            traveling[j].vel = Vector2Rotate(traveling[j].vel, -0.45f);
                            traveling[i].phase += 0.7f;
                            traveling[j].phase -= 0.7f;
                            rings.push_back({center, 0.0f, 2.4f, 0.14f, 0.10f, 0.7f, 0.0f, Color{120, 210, 255, 255}});
                        }
                        break;
                    }
                }
            }

            consumed.insert(consumed.end(), consumedByPackets.begin(), consumedByPackets.end());
            std::sort(consumed.begin(), consumed.end());
            consumed.erase(std::unique(consumed.begin(), consumed.end()), consumed.end());

            std::vector<TravelingExcitation> survivors;
            survivors.reserve(traveling.size());
            for (size_t i = 0; i < traveling.size(); ++i) {
                const TravelingExcitation& excitation = traveling[i];
                const bool removeForMerge = std::binary_search(consumed.begin(), consumed.end(), static_cast<int>(i));
                const bool expired = excitation.age >= excitation.life ||
                                     std::fabs(excitation.pos.x) > 11.0f ||
                                     std::fabs(excitation.pos.y) > 11.0f;
                if (!removeForMerge && !expired) survivors.push_back(excitation);
            }
            traveling = std::move(survivors);
        } else {
            traveling.erase(
                std::remove_if(traveling.begin(), traveling.end(), [](const TravelingExcitation& excitation) {
                    return excitation.age >= excitation.life ||
                           std::fabs(excitation.pos.x) > 11.0f ||
                           std::fabs(excitation.pos.y) > 11.0f;
                }),
                traveling.end()
            );
        }

        std::vector<StablePacket> nextStable;
        nextStable.reserve(stable.size());
        for (StablePacket& packet : stable) {
            if (packet.stage == PacketStage::kDecaying && packet.stageAge >= packet.decayDuration) {
                const bool vanished = DemotePacket(&packet, &traveling, &rings, &sparks, &rng);
                if (vanished) {
                    AddShockBurst(&rings, &sparks, packet.pos, 0, &rng);
                    continue;
                }
                AddShockBurst(&rings, &sparks, packet.pos, packet.level, &rng);
            }

            if (std::fabs(packet.pos.x) > 11.0f || std::fabs(packet.pos.y) > 11.0f) continue;
            nextStable.push_back(packet);
        }
        stable = std::move(nextStable);
        rings.erase(
            std::remove_if(rings.begin(), rings.end(), [](const ShockRing& ring) { return ring.age >= ring.life; }),
            rings.end()
        );
        sparks.erase(
            std::remove_if(sparks.begin(), sparks.end(), [](const Spark& spark) { return spark.age >= spark.life; }),
            sparks.end()
        );

        BuildFieldCaches(traveling, stable, rings, fluctuations, mode, simTime, &primary, &secondary, &metrics);

        BeginDrawing();
        ClearBackground(Color{4, 6, 16, 255});
        DrawRectangleGradientV(0, 0, GetScreenWidth(), GetScreenHeight() / 2, Color{8, 12, 28, 255}, Color{4, 6, 16, 255});
        DrawRectangleGradientV(0, GetScreenHeight() / 2, GetScreenWidth(), GetScreenHeight() / 2, Color{4, 6, 16, 255}, Color{3, 4, 10, 255});

        BeginMode3D(camera);
        DrawPrimaryFieldSurface(primary);
        DrawCoupledFieldOverlay(secondary);
        DrawShockRings(rings);
        DrawTravelingExcitations(traveling, simTime);
        DrawStablePackets(stable, simTime);
        DrawSparks(sparks);
        EndMode3D();

        DrawSceneLabels(camera, mode, traveling, stable, rings);

        DrawRectangleRounded({14.0f, 14.0f, 640.0f, 102.0f}, 0.08f, 12, Fade(BLACK, 0.34f));
        DrawText("Field Excitations", 26, 22, 32, Color{236, 242, 252, 255});
        DrawText(TextFormat("%s | %s", DemoModeName(mode), DemoModeSummary(mode)), 26, 58, 18, Color{176, 196, 224, 255});
        DrawText(TextFormat("1 vacuum  2 traveling  3 collision  |  C inject pair  V inject wave  Space quantized burst  |  A auto%s  T slow%s",
                            autoDrive ? " [ON]" : "",
                            slowMotion ? " [ON]" : ""),
                 26, 82, 16, Color{124, 226, 255, 255});

        DrawRectangleRounded({GetScreenWidth() - 280.0f, 14.0f, 266.0f, 122.0f}, 0.08f, 12, Fade(BLACK, 0.36f));
        DrawText("Energy", GetScreenWidth() - 258, 24, 24, Color{236, 242, 252, 255});
        DrawText(TextFormat("primary  %.3f", metrics.primaryEnergy), GetScreenWidth() - 258, 54, 18, Color{102, 232, 255, 255});
        DrawText(TextFormat("coupled  %.3f", metrics.secondaryEnergy), GetScreenWidth() - 258, 76, 18, Color{255, 190, 120, 255});
        DrawText(TextFormat("transfer %.3f  local %.3f", metrics.transfer, metrics.localizedEnergy), GetScreenWidth() - 258, 98, 18, Color{255, 132, 214, 255});
        DrawText(TextFormat("traveling %d   localized %d   events %d%s",
                            static_cast<int>(traveling.size()),
                            static_cast<int>(stable.size()),
                            mergeCount,
                            paused ? "   [PAUSED]" : ""),
                 GetScreenWidth() - 258,
                 120,
                 16,
                 Color{222, 228, 240, 255});

        if (inspectOverlay && paused) {
            const InspectInfo inspect = BuildInspectInfo(stable);
            if (inspect.hasPacket) {
                DrawRectangleRounded({14.0f, static_cast<float>(GetScreenHeight() - 118), 300.0f, 92.0f}, 0.08f, 12, Fade(BLACK, 0.36f));
                DrawText("Inspect", 26, GetScreenHeight() - 108, 22, Color{236, 242, 252, 255});
                DrawText(TextFormat("level %s  stage %s", kPacketPresets[inspect.level].label, PacketStageName(inspect.stage)),
                         26, GetScreenHeight() - 80, 17, Color{126, 228, 255, 255});
                DrawText(TextFormat("amp %.2f  sigma %.2f  coherence %.2f", inspect.amplitude, inspect.sigma, inspect.stageWeight),
                         26, GetScreenHeight() - 58, 16, Color{220, 228, 240, 255});
            }
        }

        DrawFPS(GetScreenWidth() - 96, GetScreenHeight() - 34);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
