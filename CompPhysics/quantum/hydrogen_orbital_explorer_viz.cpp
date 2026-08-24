#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <random>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace {
constexpr int kScreenWidth = 1440;
constexpr int kScreenHeight = 900;
constexpr int kPanelX = 1088;
constexpr int kPanelW = kScreenWidth - kPanelX - 24;
constexpr float kWorldRadius = 8.0f;
constexpr int kMaxPoints = 62000;

struct OrbitCamera {
    float yaw = 0.78f;
    float pitch = 0.34f;
    float distance = 16.0f;
};

struct CloudPoint {
    Vector3 pos{};
    float psi = 0.0f;
    float density = 0.0f;
    float jitter = 0.0f;
    int shell = 1;
    bool valence = false;
};

struct Nucleon {
    Vector3 pos{};
    bool proton = false;
};

struct ReactionAtomModel {
    std::vector<CloudPoint> cloud;
    std::vector<Nucleon> nucleus;
    float scale = 1.0f;
};

struct ReactionModels {
    ReactionAtomModel deuterium;
    ReactionAtomModel tritium;
    ReactionAtomModel helium4;
    ReactionAtomModel fissionParent;
    ReactionAtomModel barium141;
    ReactionAtomModel krypton92;
};

enum class ReactionMode {
    kIdle = 0,
    kFusion = 1,
    kFission = 2,
};

struct DetectionFlash {
    Vector3 pos{};
    float age = 0.0f;
    float life = 0.0f;
};

struct Orbital {
    const char* label;
    const char* caption;
    const char* quantum;
    float scale;
};

struct ElementPreset {
    const char* symbol;
    const char* name;
    const char* config;
    int protons;
    int massNumber;
    float scale;
};

constexpr std::array<Orbital, 8> kOrbitals{{
    {"1s", "spherical ground state", "n=1  l=0  m=0", 2.65f},
    {"2s", "radial node shell", "n=2  l=0  m=0", 3.65f},
    {"2p x", "two-lobed p orbital", "n=2  l=1  real px", 3.35f},
    {"2p z", "vertical p orbital", "n=2  l=1  m=0", 3.35f},
    {"3d z2", "dumbbell plus torus", "n=3  l=2  m=0", 4.45f},
    {"3d xy", "four-lobed diagonal", "n=3  l=2  real dxy", 4.45f},
    {"3d x2-y2", "four-lobed axial", "n=3  l=2  real dx2-y2", 4.45f},
    {"4f z", "higher angular structure", "n=4  l=3  real fz", 5.00f},
}};

constexpr std::array<ElementPreset, 11> kElements{{
    {"H", "Hydrogen", "1s1", 1, 1, 2.65f},
    {"He", "Helium", "1s2", 2, 4, 2.80f},
    {"C", "Carbon", "[He] 2s2 2p2", 6, 12, 3.55f},
    {"O", "Oxygen", "[He] 2s2 2p4", 8, 16, 3.65f},
    {"Ne", "Neon", "[He] 2s2 2p6", 10, 20, 3.85f},
    {"Na", "Sodium", "[Ne] 3s1", 11, 23, 4.25f},
    {"Cl", "Chlorine", "[Ne] 3s2 3p5", 17, 35, 4.55f},
    {"Ar", "Argon", "[Ne] 3s2 3p6", 18, 40, 4.70f},
    {"Fe", "Iron", "[Ar] 3d6 4s2", 26, 56, 5.15f},
    {"Cu", "Copper", "[Ar] 3d10 4s1", 29, 64, 5.25f},
    {"Au", "Gold", "[Xe] 4f14 5d10 6s1", 79, 197, 5.85f},
}};

float Saturate(float x) {
    return std::clamp(x, 0.0f, 1.0f);
}

float RandRange(std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    return dist(rng);
}

float OrbitalPsi(int orbital, Vector3 p) {
    const float r = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z) + 0.0001f;
    switch (orbital) {
        case 0:
            return std::exp(-r);
        case 1:
            return (2.0f - r) * std::exp(-r * 0.5f) * 0.5f;
        case 2:
            return p.x * std::exp(-r * 0.5f);
        case 3:
            return p.z * std::exp(-r * 0.5f);
        case 4:
            return (2.0f * p.z * p.z - p.x * p.x - p.y * p.y) * std::exp(-r / 3.0f) * 0.18f;
        case 5:
            return p.x * p.y * std::exp(-r / 3.0f) * 0.34f;
        case 6:
            return (p.x * p.x - p.y * p.y) * std::exp(-r / 3.0f) * 0.24f;
        default:
            return p.z * (5.0f * p.z * p.z - 3.0f * r * r) * std::exp(-r / 4.0f) * 0.035f;
    }
}

float MaxPsiForOrbital(int orbital) {
    switch (orbital) {
        case 0: return 1.0f;
        case 1: return 1.0f;
        case 2:
        case 3: return 0.75f;
        case 4:
        case 5:
        case 6: return 1.15f;
        default: return 1.6f;
    }
}

bool MouseOverControls(int controlsMode, bool reactionActive = false) {
    const Vector2 mouse = GetMousePosition();
    const Rectangle toggle{static_cast<float>(kScreenWidth - 190), 12.0f, 166.0f, 34.0f};
    if (CheckCollisionPointRec(mouse, toggle)) return true;
    if (reactionActive && mouse.x < kPanelX - 14 && mouse.y >= 640.0f && mouse.y <= 770.0f) return true;
    return controlsMode == 1 && mouse.x >= kPanelX - 14;
}

void UpdateOrbitCamera(Camera3D* camera, OrbitCamera* orbit, int controlsMode, bool reactionActive) {
    static bool orbitDragging = false;

    if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && !MouseOverControls(controlsMode, reactionActive)) {
        orbitDragging = true;
        DisableCursor();
    }
    if (orbitDragging && IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) {
        orbitDragging = false;
        EnableCursor();
    }

    if (orbitDragging) {
        const Vector2 delta = GetMouseDelta();
        orbit->yaw -= delta.x * 0.0036f;
        orbit->pitch += delta.y * 0.0036f;
        orbit->pitch = std::clamp(orbit->pitch, -1.32f, 1.32f);
    }
    orbit->distance -= GetMouseWheelMove() * 0.95f;
    orbit->distance = std::clamp(orbit->distance, 1.35f, 110.0f);

    const float cp = std::cos(orbit->pitch);
    camera->target = {0.0f, 0.0f, 0.0f};
    camera->position = {
        orbit->distance * cp * std::cos(orbit->yaw),
        orbit->distance * std::sin(orbit->pitch),
        orbit->distance * cp * std::sin(orbit->yaw)
    };
}

bool DrawButton(Rectangle r, const char* label, bool active = false) {
    const Vector2 mouse = GetMousePosition();
    const bool hover = CheckCollisionPointRec(mouse, r);
    const Color fill = active ? Color{64, 76, 122, 235} : (hover ? Color{37, 46, 72, 232} : Color{24, 30, 48, 224});
    const Color edge = active ? Color{170, 145, 245, 240} : (hover ? Color{116, 130, 170, 210} : Color{64, 76, 104, 180});
    DrawRectangleRounded(r, 0.10f, 8, fill);
    DrawRectangleRoundedLines(r, 0.10f, 8, edge);
    const int fontSize = 16;
    const int width = MeasureText(label, fontSize);
    DrawText(label, static_cast<int>(r.x + (r.width - width) * 0.5f), static_cast<int>(r.y + 8), fontSize, active ? Color{240, 238, 255, 255} : Color{204, 212, 232, 255});
    return hover && IsMouseButtonPressed(MOUSE_LEFT_BUTTON);
}

bool DrawSlider(Rectangle r, const char* label, float* value, float minValue, float maxValue) {
    const Vector2 mouse = GetMousePosition();
    const bool active = CheckCollisionPointRec(mouse, r) && IsMouseButtonDown(MOUSE_LEFT_BUTTON);
    if (active) {
        const float t = Saturate((mouse.x - r.x) / r.width);
        *value = minValue + (maxValue - minValue) * t;
    }

    std::ostringstream os;
    os << label << "  " << std::fixed << std::setprecision(2) << *value;
    DrawText(os.str().c_str(), static_cast<int>(r.x), static_cast<int>(r.y - 24), 17, {190, 200, 222, 255});
    DrawRectangleRounded(r, 0.5f, 8, {32, 38, 58, 235});
    const float t = Saturate((*value - minValue) / (maxValue - minValue));
    DrawRectangleRounded({r.x, r.y, r.width * t, r.height}, 0.5f, 8, {126, 92, 204, 245});
    DrawCircle(static_cast<int>(r.x + r.width * t), static_cast<int>(r.y + r.height * 0.5f), 10.0f, {230, 222, 255, 255});
    DrawCircleLines(static_cast<int>(r.x + r.width * t), static_cast<int>(r.y + r.height * 0.5f), 10.0f, {116, 130, 170, 210});
    return active;
}

std::vector<CloudPoint> BuildCloud(int orbital, int seed);
std::vector<CloudPoint> BuildElementCloud(int element, int seed);
std::vector<CloudPoint> BuildElementCloudForProtons(int protons, int seed);

bool DrawIntSlider(Rectangle r, const char* label, int* value, int minValue, int maxValue) {
    float f = static_cast<float>(*value);
    const bool active = DrawSlider(r, label, &f, static_cast<float>(minValue), static_cast<float>(maxValue));
    *value = static_cast<int>(std::round(f));
    return active;
}

void SelectOrbital(int nextOrbital,
                   int* orbital,
                   float* scale,
                   std::vector<CloudPoint>* cloud,
                   std::vector<CloudPoint>* morphFrom,
                   float* morphT) {
    if (nextOrbital == *orbital) return;
    *morphFrom = *cloud;
    *orbital = nextOrbital;
    *scale = kOrbitals[*orbital].scale;
    *cloud = BuildCloud(*orbital, GetRandomValue(1, 999999));
    *morphT = 0.0f;
}

void SelectElement(int nextElement,
                   int* element,
                   float* scale,
                   std::vector<CloudPoint>* cloud,
                   std::vector<CloudPoint>* morphFrom,
                   float* morphT) {
    if (nextElement == *element) return;
    *morphFrom = *cloud;
    *element = nextElement;
    *scale = kElements[*element].scale;
    *cloud = BuildElementCloud(*element, GetRandomValue(1, 999999));
    *morphT = 0.0f;
}

std::vector<CloudPoint> BuildCloud(int orbital, int seed) {
    std::mt19937 rng(seed + orbital * 7919);
    std::vector<CloudPoint> points;
    points.reserve(kMaxPoints);

    const float maxPsi = MaxPsiForOrbital(orbital);
    int attempts = 0;
    while (static_cast<int>(points.size()) < kMaxPoints && attempts < kMaxPoints * 90) {
        ++attempts;
        Vector3 p{
            RandRange(rng, -kWorldRadius, kWorldRadius),
            RandRange(rng, -kWorldRadius, kWorldRadius),
            RandRange(rng, -kWorldRadius, kWorldRadius)
        };
        if (Vector3Length(p) > kWorldRadius) continue;

        const float psi = OrbitalPsi(orbital, p);
        const float probability = Saturate((psi * psi) / (maxPsi * maxPsi));
        const float accept = std::pow(probability, 0.42f);
        if (RandRange(rng, 0.0f, 1.0f) > accept) continue;

        points.push_back({p, psi, probability, RandRange(rng, 0.0f, 2.0f * PI), 1, true});
    }

    return points;
}

std::vector<CloudPoint> BuildElementCloud(int element, int seed) {
    return BuildElementCloudForProtons(kElements[element].protons, seed);
}

std::vector<CloudPoint> BuildElementCloudForProtons(int protons, int seed) {
    struct Component {
        int orbital;
        float spread;
        int weight;
        int shell;
        bool valence;
    };

    std::vector<Component> components;
    auto add = [&](int orbital, float spread, int weight, int shell, bool valence) {
        for (int i = 0; i < weight; ++i) components.push_back({orbital, spread, 1, shell, valence});
    };

    add(0, 0.42f, std::min(protons, 2), 1, protons <= 2);
    if (protons >= 3) add(1, 0.72f, std::min(protons - 2, 2), 2, protons <= 4);
    if (protons >= 5) {
        const int pElectrons = std::min(protons - 4, 6);
        add(2, 0.88f, (pElectrons + 1) / 2, 2, protons <= 10);
        add(3, 0.88f, pElectrons / 2, 2, protons <= 10);
    }
    if (protons >= 11) add(1, 1.18f, std::min(protons - 10, 2), 3, protons <= 12);
    if (protons >= 13) {
        const int pElectrons = std::min(protons - 12, 6);
        add(2, 1.32f, (pElectrons + 1) / 2, 3, protons <= 18);
        add(3, 1.32f, pElectrons / 2, 3, protons <= 18);
    }
    if (protons >= 21) {
        const int dElectrons = std::min(protons - 20, 10);
        add(4, 1.08f, std::max(1, dElectrons / 3), 3, false);
        add(5, 1.08f, std::max(1, dElectrons / 4), 3, false);
        add(6, 1.08f, std::max(1, dElectrons / 4), 3, false);
        add(1, 1.78f, protons == 29 ? 1 : 2, 4, true);
    }
    if (protons >= 57) {
        add(7, 1.20f, 8, 4, false);
        add(4, 1.52f, 5, 5, false);
        add(1, 2.16f, 2, 6, true);
    }

    if (components.empty()) components.push_back({0, 1.0f, 1, 1, true});

    std::mt19937 rng(seed + protons * 3571);
    std::uniform_int_distribution<int> pick(0, static_cast<int>(components.size() - 1));
    std::vector<CloudPoint> points;
    points.reserve(kMaxPoints);

    int attempts = 0;
    while (static_cast<int>(points.size()) < kMaxPoints && attempts < kMaxPoints * 120) {
        ++attempts;
        const Component c = components[pick(rng)];
        Vector3 sample{
            RandRange(rng, -kWorldRadius, kWorldRadius),
            RandRange(rng, -kWorldRadius, kWorldRadius),
            RandRange(rng, -kWorldRadius, kWorldRadius)
        };
        if (Vector3Length(sample) > kWorldRadius) continue;

        const float psi = OrbitalPsi(c.orbital, sample);
        const float probability = Saturate((psi * psi) / (MaxPsiForOrbital(c.orbital) * MaxPsiForOrbital(c.orbital)));
        const float accept = std::pow(probability, 0.40f);
        if (RandRange(rng, 0.0f, 1.0f) > accept) continue;

        Vector3 p = Vector3Scale(sample, c.spread);
        if (Vector3Length(p) > kWorldRadius) p = Vector3Scale(Vector3Normalize(p), kWorldRadius * RandRange(rng, 0.92f, 1.0f));
        points.push_back({p, psi, probability, RandRange(rng, 0.0f, 2.0f * PI), c.shell, c.valence});
    }

    return points;
}

Vector3 RandomUnit(std::mt19937& rng) {
    const float z = RandRange(rng, -1.0f, 1.0f);
    const float theta = RandRange(rng, 0.0f, 2.0f * PI);
    const float r = std::sqrt(1.0f - z * z);
    return {r * std::cos(theta), r * std::sin(theta), z};
}

std::vector<Nucleon> BuildNucleus(int protons, int neutrons, int maxDots) {
    std::mt19937 rng(91 + protons * 17 + neutrons * 31 + maxDots);
    std::vector<Nucleon> nucleus;
    const int total = std::max(1, protons + neutrons);
    const int count = std::clamp(maxDots, 1, 32);
    nucleus.reserve(count);

    for (int i = 0; i < count; ++i) {
        const float radius = std::cbrt(RandRange(rng, 0.0f, 1.0f)) * 0.075f;
        Vector3 pos = Vector3Scale(RandomUnit(rng), radius);
        nucleus.push_back({pos, RandRange(rng, 0.0f, 1.0f) < static_cast<float>(protons) / total});
    }

    return nucleus;
}

float SliceCoord(Vector3 p, int axis) {
    if (axis == 0) return p.x;
    if (axis == 1) return p.y;
    return p.z;
}

const char* SliceAxisName(int axis) {
    if (axis == 0) return "X";
    if (axis == 1) return "Y";
    return "Z";
}

float SmoothStep(float x) {
    const float t = Saturate(x);
    return t * t * (3.0f - 2.0f * t);
}

CloudPoint BlendPoint(const CloudPoint& a, const CloudPoint& b, float t) {
    const float u = SmoothStep(t);
    return {
        Vector3Lerp(a.pos, b.pos, u),
        a.psi + (b.psi - a.psi) * u,
        a.density + (b.density - a.density) * u,
        a.jitter + (b.jitter - a.jitter) * u,
        u < 0.5f ? a.shell : b.shell,
        u < 0.5f ? a.valence : b.valence,
    };
}

std::string ScreenshotPath(const char* label) {
    char path[128];
    std::snprintf(path, sizeof(path), "media/atomic_orbital_%s_%05d.png", label, GetRandomValue(10000, 99999));
    std::string out(path);
    std::replace(out.begin(), out.end(), ' ', '_');
    return out;
}

float ReactionPhase(float reactionT, float duration) {
    return std::fmod(std::max(0.0f, reactionT), duration) / duration;
}

std::vector<Vector3> AtomOffsets(int atomCount, float scale) {
    std::vector<Vector3> offsets;
    const int count = std::clamp(atomCount, 1, 12);
    offsets.reserve(count);
    if (count == 1) {
        offsets.push_back({0.0f, 0.0f, 0.0f});
        return offsets;
    }

    const float spacing = scale * 1.75f;
    if (count <= 4) {
        const float start = -0.5f * spacing * static_cast<float>(count - 1);
        for (int i = 0; i < count; ++i) {
            offsets.push_back({start + spacing * i, 0.0f, 0.0f});
        }
        return offsets;
    }

    offsets.push_back({0.0f, 0.0f, 0.0f});
    const int outer = count - 1;
    const float radius = spacing * (outer <= 6 ? 1.0f : 1.55f);
    for (int i = 0; i < outer; ++i) {
        const float a = 2.0f * PI * static_cast<float>(i) / static_cast<float>(outer);
        offsets.push_back({radius * std::cos(a), 0.0f, radius * std::sin(a)});
    }
    return offsets;
}

void DrawReactionLabel(const Camera3D& camera, Vector3 center, float radius, const char* label, const char* detail) {
    const Vector2 screen = GetWorldToScreen(center, camera);
    const int labelWidth = MeasureText(label, 18);
    DrawText(label, static_cast<int>(screen.x - labelWidth * 0.5f), static_cast<int>(screen.y + radius), 18, {238, 242, 252, 245});
    const int detailWidth = MeasureText(detail, 14);
    DrawText(detail, static_cast<int>(screen.x - detailWidth * 0.5f), static_cast<int>(screen.y + radius + 23.0f), 14, {156, 174, 206, 230});
}

void DrawParticleStream(Vector2 from, Vector2 to, Color color, int count, float dotRadius, bool arrowHead = false) {
    const Vector2 delta = Vector2Subtract(to, from);
    const float length = Vector2Length(delta);
    if (length < 0.001f) return;
    const Vector2 direction = Vector2Scale(delta, 1.0f / length);

    for (int i = 0; i < count; ++i) {
        const float t = count == 1 ? 1.0f : static_cast<float>(i) / static_cast<float>(count - 1);
        Color c = color;
        c.a = static_cast<unsigned char>(color.a * (0.38f + 0.62f * t));
        DrawCircleV(Vector2Lerp(from, to, t), dotRadius * (0.72f + 0.28f * t), c);
    }

    if (!arrowHead) return;
    const Vector2 perpendicular{-direction.y, direction.x};
    const Vector2 base = Vector2Subtract(to, Vector2Scale(direction, 17.0f));
    const Vector2 wingA = Vector2Add(base, Vector2Scale(perpendicular, 9.0f));
    const Vector2 wingB = Vector2Subtract(base, Vector2Scale(perpendicular, 9.0f));
    for (int i = 0; i < 5; ++i) {
        const float t = static_cast<float>(i) / 4.0f;
        DrawCircleV(Vector2Lerp(wingA, to, t), dotRadius, color);
        DrawCircleV(Vector2Lerp(wingB, to, t), dotRadius, color);
    }
}

void DrawReactionNeutron(const Camera3D& camera, Vector3 position, Vector3 velocityHint, const char* label) {
    const Vector2 p = GetWorldToScreen(position, camera);
    const Vector2 tail = GetWorldToScreen(Vector3Subtract(position, velocityHint), camera);
    DrawParticleStream(tail, Vector2Subtract(p, {7.0f, 0.0f}), {112, 190, 255, 180}, 10, 1.8f, true);
    for (int i = 0; i < 9; ++i) {
        const float angle = static_cast<float>(i) * 2.399963f;
        const float radius = i == 0 ? 0.0f : 2.0f + 0.72f * static_cast<float>(i);
        const Vector2 dot{p.x + std::cos(angle) * radius, p.y + std::sin(angle) * radius};
        DrawCircleV(dot, i == 0 ? 3.0f : 2.2f, i % 3 == 0 ? Color{160, 220, 255, 255} : Color{92, 174, 255, 245});
    }
    if (label && label[0] != '\0') DrawText(label, static_cast<int>(p.x - 29.0f), static_cast<int>(p.y - 31.0f), 15, {184, 220, 250, 240});
}

Color BlendColor(Color a, Color b, float t) {
    const float u = Saturate(t);
    return {
        static_cast<unsigned char>(a.r + (b.r - a.r) * u),
        static_cast<unsigned char>(a.g + (b.g - a.g) * u),
        static_cast<unsigned char>(a.b + (b.b - a.b) * u),
        static_cast<unsigned char>(a.a + (b.a - a.a) * u),
    };
}

void DrawMassEnergyConversion(const Camera3D& camera, Vector3 center, float progress, float time) {
    if (progress <= 0.0f || progress >= 1.0f) return;
    const Vector2 p = GetWorldToScreen(center, camera);
    const Color energy{255, 222, 118, 245};

    for (int i = 0; i < 14; ++i) {
        const float delay = 0.018f * static_cast<float>(i);
        const float t = SmoothStep(Saturate((progress - delay) / (1.0f - delay)));
        const float direction = i % 2 == 0 ? 1.0f : -1.0f;
        const float lane = static_cast<float>((i / 2) % 4) - 1.5f;
        const float distance = 10.0f + 155.0f * t;
        const float wave = std::sin(time * 3.0f + i * 1.7f + t * 5.0f) * (3.0f + 2.0f * t);
        const Vector2 particle{p.x + direction * distance, p.y + lane * 5.0f + wave};
        const Color matter = i % 3 == 0 ? Color{255, 100, 92, 245} : Color{92, 174, 255, 245};
        Color converted = BlendColor(matter, energy, Saturate(t * 2.2f));
        converted.a = static_cast<unsigned char>(converted.a * (1.0f - 0.72f * Saturate((t - 0.72f) / 0.28f)));

        for (int trail = 3; trail >= 1; --trail) {
            const float trailT = std::max(0.0f, t - trail * 0.035f);
            Color trailColor = converted;
            trailColor.a = static_cast<unsigned char>(converted.a * (0.12f + 0.12f * (4 - trail)));
            DrawCircleV({p.x + direction * (10.0f + 155.0f * trailT), particle.y}, 1.2f, trailColor);
        }
        DrawCircleV(particle, 2.8f - 0.9f * t, converted);
    }

}

void DrawReactionScene(const Camera3D& camera, ReactionMode reaction, float reactionTime, float time) {
    const float duration = reaction == ReactionMode::kFusion ? 5.2f : 4.8f;
    const float phase = ReactionPhase(reactionTime, duration);
    DrawText(reaction == ReactionMode::kFusion ? "FUSION: H-2 + H-3  ->  He-4 + neutron"
                                               : "FISSION: U-235 + neutron  ->  Ba-141 + Kr-92 + 3 neutrons",
             24, 84, 21, reaction == ReactionMode::kFusion ? Color{146, 222, 255, 245} : Color{255, 164, 202, 245});

    if (reaction == ReactionMode::kFusion) {
        const float approach = SmoothStep(Saturate(phase / 0.48f));
        const float separation = 4.3f + (0.50f - 4.3f) * approach;
        const float conversion = Saturate((phase - 0.46f) / 0.38f);

        if (phase < 0.58f) {
            DrawReactionLabel(camera, {-separation, 0.22f, 0.0f}, 62.0f, "Deuterium  H-2", "1 proton + 1 neutron");
            DrawReactionLabel(camera, {separation, -0.22f, 0.0f}, 62.0f, "Tritium  H-3", "1 proton + 2 neutrons");
            const Vector2 left = GetWorldToScreen({-separation + 0.9f, 0.22f, 0.0f}, camera);
            const Vector2 right = GetWorldToScreen({separation - 0.9f, -0.22f, 0.0f}, camera);
            const Vector2 center = Vector2Scale(Vector2Add(left, right), 0.5f);
            DrawParticleStream(left, center, {130, 215, 255, 150}, 13, 1.7f, true);
            DrawParticleStream(right, center, {190, 142, 255, 150}, 13, 1.7f, true);
        }
        DrawMassEnergyConversion(camera, {0.0f, 0.0f, 0.0f}, conversion, time);
        if (phase >= 0.52f) {
            DrawReactionLabel(camera, {0.0f, 0.0f, 0.0f}, 92.0f, "Helium-4", "2 protons + 2 neutrons");
            const float neutronTravel = 0.4f + 3.2f * SmoothStep(Saturate((phase - 0.56f) / 0.28f));
            const Vector3 neutronPos{0.55f * neutronTravel, 0.72f * neutronTravel, 0.35f * neutronTravel};
            DrawReactionNeutron(camera, neutronPos, Vector3Scale(Vector3Normalize(neutronPos), 0.55f), "released neutron");
        }

        const char* step = phase < 0.36f ? "1  Isotopes approach" : (phase < 0.58f ? "2  Nuclei fuse; mass defect forms" : "3  He-4 + neutron + 17.6 MeV");
        DrawText(step, 24, 116, 17, {190, 202, 226, 235});
    } else {
        const float incoming = SmoothStep(Saturate(phase / 0.34f));
        const float split = SmoothStep(Saturate((phase - 0.42f) / 0.32f));
        const float conversion = Saturate((phase - 0.39f) / 0.43f);

        if (phase < 0.49f) {
            DrawReactionLabel(camera, {0.0f, 0.0f, 0.0f}, 112.0f, "Uranium-235", "92 protons + 143 neutrons");
            const float neutronX = -5.4f + 5.0f * incoming;
            DrawReactionNeutron(camera, {neutronX, 0.0f, 0.0f}, {0.65f, 0.0f, 0.0f}, "neutron");
        }
        DrawMassEnergyConversion(camera, {0.0f, 0.0f, 0.0f}, conversion, time);
        if (phase >= 0.42f) {
            const float distance = 0.45f + 3.2f * split;
            DrawReactionLabel(camera, {-distance, 0.48f * split, 0.0f}, 76.0f, "Barium-141", "56 protons + 85 neutrons");
            DrawReactionLabel(camera, {distance, -0.48f * split, 0.0f}, 76.0f, "Krypton-92", "36 protons + 56 neutrons");
            for (int i = 0; i < 3; ++i) {
                const float angle = -0.85f + i * 0.85f;
                const float travel = (0.35f + 3.1f * split);
                Vector3 neutronPos{std::cos(angle) * travel, 1.0f + std::sin(angle) * travel, 0.45f * (i - 1)};
                DrawReactionNeutron(camera, neutronPos, Vector3Scale(Vector3Normalize(neutronPos), 0.55f), i == 1 ? "released neutrons" : "");
            }
        }

        const char* step = phase < 0.30f ? "1  Incoming neutron" : (phase < 0.49f ? "2  U-236 becomes unstable" : "3  Ba-141 + Kr-92 + 3n + about 200 MeV");
        DrawText(step, 24, 116, 17, {190, 202, 226, 235});
    }

    DrawText("The sequence automatically replays", 24, 142, 15, {130, 146, 176, 220});
}

void DrawLegendItem(int x, int y, Color color, const char* label) {
    DrawCircle(x + 4, y + 6, 4.0f, color);
    DrawText(label, x + 14, y, 14, {174, 188, 214, 235});
}

void DrawReactionLegend(bool phaseColor) {
    int x = 24;
    DrawLegendItem(x, 174, {255, 96, 88, 245}, "proton");
    x += 78;
    DrawLegendItem(x, 174, {96, 176, 255, 245}, "neutron");
    x += 88;
    if (phaseColor) {
        DrawLegendItem(x, 174, {78, 198, 255, 235}, "+ phase");
        x += 82;
        DrawLegendItem(x, 174, {255, 142, 90, 235}, "- phase");
        x += 82;
    } else {
        DrawLegendItem(x, 174, {224, 104, 244, 240}, "electron density");
        x += 132;
    }
    DrawLegendItem(x, 174, {255, 222, 118, 245}, "released energy");
}

float ReactionConversionProgress(ReactionMode reaction, float phase) {
    return reaction == ReactionMode::kFusion ? Saturate((phase - 0.46f) / 0.38f)
                                             : Saturate((phase - 0.39f) / 0.43f);
}

void DrawReactionSciencePanel(ReactionMode reaction, float reactionTime) {
    const float duration = reaction == ReactionMode::kFusion ? 5.2f : 4.8f;
    const float phase = ReactionPhase(reactionTime, duration);
    const float conversion = SmoothStep(ReactionConversionProgress(reaction, phase));
    const Rectangle card{24.0f, 204.0f, 440.0f, 158.0f};
    DrawRectangleRounded(card, 0.06f, 8, {10, 16, 30, 222});
    DrawRectangleRoundedLines(card, 0.06f, 8, {82, 98, 134, 170});
    DrawText("CONSERVATION CHECK", 40, 216, 16, {224, 232, 248, 245});

    if (reaction == ReactionMode::kFusion) {
        DrawText("Before   H-2 + H-3        2p  3n  A=5", 40, 242, 15, {166, 184, 214, 240});
        DrawText("After    He-4 + n          2p  3n  A=5", 40, 264, 15, {166, 184, 214, 240});
        DrawText("mass defect 0.0189 u     released energy 17.6 MeV", 40, 290, 15, {255, 220, 142, 245});
    } else {
        DrawText("Before   U-235 + n        92p  144n  A=236", 40, 242, 15, {166, 184, 214, 240});
        DrawText("After    Ba-141 + Kr-92 + 3n   92p  144n  A=236", 40, 264, 15, {166, 184, 214, 240});
        DrawText("mass defect about 0.215 u   energy about 200 MeV", 40, 290, 15, {255, 220, 142, 245});
    }

    DrawText("p and n conserved", 316, 216, 14, {116, 224, 166, 245});
    DrawText("mass defect", 40, 316, 13, {190, 202, 226, 235});
    DrawText("energy   E = dm c^2", 248, 316, 13, {190, 202, 226, 235});
    DrawRectangleRounded({40.0f, 336.0f, 170.0f, 9.0f}, 0.5f, 6, {32, 40, 62, 230});
    DrawRectangleRounded({40.0f, 336.0f, 170.0f * (1.0f - conversion), 9.0f}, 0.5f, 6, {160, 112, 210, 245});
    DrawRectangleRounded({248.0f, 336.0f, 170.0f, 9.0f}, 0.5f, 6, {32, 40, 62, 230});
    DrawRectangleRounded({248.0f, 336.0f, 170.0f * conversion, 9.0f}, 0.5f, 6, {255, 206, 108, 245});
    DrawText("conversion progress (bars not to scale)", 40, 348, 11, {116, 132, 162, 215});
}

Color PointColor(float psi, float density, float pulse, bool phaseColor) {
    if (!phaseColor) {
        const float t = Saturate(std::pow(density * 30.0f, 0.34f) + pulse * 0.035f);
        if (t < 0.14f) return {190, 168, 255, 226};
        if (t < 0.30f) return {232, 124, 255, 234};
        if (t < 0.46f) return {252, 86, 186, 242};
        if (t < 0.62f) return {188, 56, 156, 248};
        if (t < 0.78f) return {112, 38, 142, 252};
        if (t < 0.91f) return {58, 28, 106, 255};
        return {24, 16, 62, 255};
    }

    const Color positive{78, 198, 255, 235};
    const Color negative{255, 142, 90, 235};
    Color c = psi >= 0.0f ? positive : negative;
    const float lift = Saturate(density * 3.0f + pulse * 0.12f);
    c.r = static_cast<unsigned char>(c.r * (0.58f + 0.42f * lift));
    c.g = static_cast<unsigned char>(c.g * (0.58f + 0.42f * lift));
    c.b = static_cast<unsigned char>(c.b * (0.58f + 0.42f * lift));
    return c;
}

Color ElementPointColor(const CloudPoint& p, float pulse, bool phaseColor) {
    if (phaseColor) return PointColor(p.psi, p.density, pulse, true);

    const float densityTone = Saturate(std::pow(p.density * 18.0f, 0.36f) + pulse * 0.02f);
    if (p.valence) {
        if (densityTone < 0.35f) return {242, 132, 255, 236};
        if (densityTone < 0.70f) return {210, 84, 238, 246};
        return {136, 54, 190, 255};
    }
    if (p.shell <= 1) {
        if (densityTone < 0.55f) return {46, 58, 118, 224};
        return {30, 26, 82, 246};
    }
    if (p.shell == 2) {
        if (densityTone < 0.50f) return {94, 64, 172, 232};
        return {66, 34, 128, 250};
    }
    if (p.shell == 3) {
        if (densityTone < 0.50f) return {156, 64, 178, 238};
        return {92, 38, 146, 252};
    }
    if (p.shell == 4) {
        if (densityTone < 0.50f) return {204, 78, 162, 240};
        return {120, 44, 132, 252};
    }
    return densityTone < 0.50f ? Color{230, 96, 160, 242} : Color{146, 48, 126, 255};
}

void DrawProjectedCloud(const std::vector<CloudPoint>& cloud,
                        const std::vector<CloudPoint>& morphFrom,
                        const Camera3D& camera,
                        Vector3 atomOffset,
                        float visibleDensity,
                        float scale,
                        float pointSize,
                        float time,
                        float animationSpeed,
                        bool phaseColor,
                        bool elementMode,
                        bool slice,
                        int sliceAxis,
                        float slicePosition,
                        float sliceThickness,
                        float morphT) {
    const int targetCount = static_cast<int>(cloud.size() * visibleDensity);
    const float invWorld = scale / kWorldRadius;
    const int side = std::max(1, static_cast<int>(std::round(pointSize)));

    for (int i = 0; i < targetCount; ++i) {
        CloudPoint p = cloud[i];
        if (!morphFrom.empty() && i < static_cast<int>(morphFrom.size()) && morphT < 1.0f) {
            p = BlendPoint(morphFrom[i], cloud[i], morphT);
        }
        if (slice && std::fabs(SliceCoord(p.pos, sliceAxis) - slicePosition * kWorldRadius) > sliceThickness * kWorldRadius) continue;

        const float pulse = 0.5f + 0.5f * std::sin(time * 2.0f + p.jitter);
        const float breathe = 1.0f + 0.025f * animationSpeed * std::sin(time * 1.15f + p.jitter);
        const Vector3 world = Vector3Add(atomOffset, Vector3Scale(p.pos, invWorld * breathe));
        const Vector2 screen = GetWorldToScreen(world, camera);
        if (screen.x < -4.0f || screen.x > kScreenWidth + 4.0f || screen.y < -4.0f || screen.y > kScreenHeight + 4.0f) continue;

        Color c = elementMode ? ElementPointColor(p, pulse, phaseColor) : PointColor(p.psi, p.density, pulse, phaseColor);
        DrawRectangle(static_cast<int>(screen.x), static_cast<int>(screen.y), side, side, c);
    }
}

void DrawDetectionFlashes(const std::vector<DetectionFlash>& flashes, const Camera3D& camera, Vector3 atomOffset, float scale) {
    const float invWorld = scale / kWorldRadius;
    for (const DetectionFlash& flash : flashes) {
        const float t = Saturate(flash.age / flash.life);
        const Vector2 screen = GetWorldToScreen(Vector3Add(atomOffset, Vector3Scale(flash.pos, invWorld)), camera);
        const unsigned char alpha = static_cast<unsigned char>(220 * (1.0f - t));
        DrawRectangle(static_cast<int>(screen.x) - 2, static_cast<int>(screen.y) - 2, 5, 5, {235, 210, 255, alpha});
    }
}

void DrawProjectedNucleus(const std::vector<Nucleon>& nucleus,
                          const Camera3D& camera,
                          Vector3 atomOffset,
                          float scale,
                          float time,
                          bool controlsOpen) {
    const float nucleusScale = 0.16f * scale;
    (void)controlsOpen;

    for (const Nucleon& n : nucleus) {
        const float wobble = 0.006f * std::sin(time * 1.8f + n.pos.x * 7.0f + n.pos.z * 5.0f);
        Vector3 p = Vector3Scale(n.pos, nucleusScale * (1.0f + wobble));
        Vector2 screen = GetWorldToScreen(Vector3Add(atomOffset, p), camera);
        Color fill = n.proton ? Color{255, 96, 88, 245} : Color{96, 176, 255, 245};
        DrawRectangle(static_cast<int>(screen.x), static_cast<int>(screen.y), 2, 2, fill);
    }
}

void DrawReactionModel(const ReactionAtomModel& model,
                       const Camera3D& camera,
                       Vector3 offset,
                       float density,
                       float time,
                       float animationSpeed,
                       bool phaseColor) {
    if (density <= 0.005f) return;
    static const std::vector<CloudPoint> noMorph;
    DrawProjectedCloud(model.cloud, noMorph, camera, offset, density, model.scale, 2.0f, time,
                       animationSpeed, phaseColor, true, false, 1, 0.0f, 1.0f, 1.0f);
    DrawProjectedNucleus(model.nucleus, camera, offset, model.scale, time, false);
}

void DrawReactionModels(const ReactionModels& models,
                        const Camera3D& camera,
                        ReactionMode reaction,
                        float reactionTime,
                        float time,
                        float animationSpeed,
                        bool phaseColor) {
    const float duration = reaction == ReactionMode::kFusion ? 5.2f : 4.8f;
    const float phase = ReactionPhase(reactionTime, duration);

    if (reaction == ReactionMode::kFusion) {
        const float approach = SmoothStep(Saturate(phase / 0.48f));
        const float separation = 4.3f + (0.50f - 4.3f) * approach;
        const float merge = SmoothStep(Saturate((phase - 0.52f) / 0.06f));
        if (phase < 0.58f) {
            const float lightDensity = 0.22f * (1.0f - merge);
            DrawReactionModel(models.deuterium, camera, {-separation, 0.22f, 0.0f}, lightDensity, time, animationSpeed, phaseColor);
            DrawReactionModel(models.tritium, camera, {separation, -0.22f, 0.0f}, lightDensity, time + 0.8f, animationSpeed, phaseColor);
        }
        if (phase >= 0.52f) {
            DrawReactionModel(models.helium4, camera, {0.0f, 0.0f, 0.0f}, 0.34f * merge, time, animationSpeed, phaseColor);
        }
        return;
    }

    const float split = SmoothStep(Saturate((phase - 0.42f) / 0.32f));
    const float fragmentReveal = SmoothStep(Saturate((phase - 0.42f) / 0.07f));
    if (phase < 0.49f) {
        DrawReactionModel(models.fissionParent, camera, {0.0f, 0.0f, 0.0f}, 0.34f * (1.0f - fragmentReveal), time, animationSpeed, phaseColor);
    }
    if (phase >= 0.42f) {
        const float distance = 0.45f + 3.2f * split;
        const float daughterDensity = 0.17f * fragmentReveal;
        DrawReactionModel(models.barium141, camera, {-distance, 0.48f * split, 0.0f}, daughterDensity, time, animationSpeed, phaseColor);
        DrawReactionModel(models.krypton92, camera, {distance, -0.48f * split, 0.0f}, daughterDensity, time + 0.5f, animationSpeed, phaseColor);
    }
}

float ReactionDuration(ReactionMode reaction) {
    return reaction == ReactionMode::kFusion ? 5.2f : 4.8f;
}

std::array<float, 4> ReactionStagePositions(ReactionMode reaction) {
    return reaction == ReactionMode::kFusion ? std::array<float, 4>{{0.0f, 0.36f, 0.58f, 0.82f}}
                                             : std::array<float, 4>{{0.0f, 0.30f, 0.49f, 0.78f}};
}

void StepReactionStage(ReactionMode reaction, float* reactionTime, bool* paused, int direction) {
    const float duration = ReactionDuration(reaction);
    const float phase = ReactionPhase(*reactionTime, duration);
    const std::array<float, 4> stages = ReactionStagePositions(reaction);
    int target = direction > 0 ? 3 : 0;
    if (direction > 0) {
        for (int i = 0; i < 4; ++i) {
            if (stages[i] > phase + 0.025f) {
                target = i;
                break;
            }
        }
    } else {
        for (int i = 3; i >= 0; --i) {
            if (stages[i] < phase - 0.025f) {
                target = i;
                break;
            }
        }
    }
    *reactionTime = stages[target] * duration;
    *paused = true;
}

void DrawReactionTimeline(ReactionMode reaction, float* reactionTime, bool* paused, float* animationSpeed) {
    const float duration = ReactionDuration(reaction);
    const std::array<float, 4> stages = ReactionStagePositions(reaction);
    const std::array<const char*, 4> fusionLabels{{"approach", "contact", "conversion", "products"}};
    const std::array<const char*, 4> fissionLabels{{"neutron", "capture", "split", "products"}};
    const auto& labels = reaction == ReactionMode::kFusion ? fusionLabels : fissionLabels;
    const Rectangle card{24.0f, 646.0f, 1040.0f, 116.0f};
    const Rectangle track{40.0f, 716.0f, 1008.0f, 9.0f};

    DrawRectangleRounded(card, 0.05f, 8, {9, 14, 26, 226});
    DrawRectangleRoundedLines(card, 0.05f, 8, {76, 90, 124, 175});
    DrawText("REACTION TIMELINE", 40, 657, 15, {210, 220, 240, 240});
    if (DrawButton({190.0f, 654.0f, 76.0f, 30.0f}, "Replay")) {
        *reactionTime = 0.0f;
        *paused = false;
    }
    if (DrawButton({274.0f, 654.0f, 62.0f, 30.0f}, "Prev")) StepReactionStage(reaction, reactionTime, paused, -1);
    if (DrawButton({344.0f, 654.0f, 70.0f, 30.0f}, *paused ? "Play" : "Pause")) *paused = !*paused;
    if (DrawButton({422.0f, 654.0f, 62.0f, 30.0f}, "Next")) StepReactionStage(reaction, reactionTime, paused, 1);
    DrawText("speed", 506, 662, 14, {142, 158, 188, 230});
    if (DrawButton({552.0f, 654.0f, 58.0f, 30.0f}, "0.5x", std::fabs(*animationSpeed - 0.5f) < 0.05f)) *animationSpeed = 0.5f;
    if (DrawButton({618.0f, 654.0f, 58.0f, 30.0f}, "1.0x", std::fabs(*animationSpeed - 1.0f) < 0.05f)) *animationSpeed = 1.0f;
    if (DrawButton({684.0f, 654.0f, 58.0f, 30.0f}, "2.0x", std::fabs(*animationSpeed - 2.0f) < 0.05f)) *animationSpeed = 2.0f;
    DrawText("drag the track to inspect any moment", 764, 662, 14, {128, 144, 174, 220});

    const Vector2 mouse = GetMousePosition();
    const Rectangle hitbox{track.x, track.y - 10.0f, track.width, 28.0f};
    if (CheckCollisionPointRec(mouse, hitbox) && IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        const float scrubPhase = std::clamp((mouse.x - track.x) / track.width, 0.0f, 0.995f);
        *reactionTime = scrubPhase * duration;
        *paused = true;
    }

    const float phase = ReactionPhase(*reactionTime, duration);
    DrawRectangleRounded(track, 0.5f, 8, {30, 38, 58, 245});
    if (phase > 0.001f) {
        DrawRectangleRounded({track.x, track.y, track.width * phase, track.height}, 0.5f, 8,
                             reaction == ReactionMode::kFusion ? Color{98, 194, 238, 245} : Color{222, 102, 166, 245});
    }
    DrawCircle(static_cast<int>(track.x + track.width * phase), static_cast<int>(track.y + track.height * 0.5f), 7.0f, {244, 238, 255, 255});
    for (int i = 0; i < 4; ++i) {
        const float x = track.x + track.width * stages[i];
        DrawCircle(static_cast<int>(x), static_cast<int>(track.y + 4.0f), 4.0f, {176, 148, 238, 255});
        DrawText(labels[i], static_cast<int>(x - MeasureText(labels[i], 12) * 0.5f), 736, 12, {142, 158, 188, 225});
    }
}

void DrawPanel(int* orbital,
               int* element,
               bool* elementMode,
               float* visibleDensity,
               float* scale,
               float* pointSize,
               float* animationSpeed,
               bool* phaseColor,
               bool* slice,
               bool* paused,
               bool* cinematic,
               bool* detections,
               ReactionMode* reaction,
               float* reactionTime,
               int* sliceAxis,
               float* slicePosition,
               float* sliceThickness,
               int* nucleusCount,
               int* atomCount,
               int* controlsMode,
               int* panelTab,
               std::vector<Nucleon>* nucleus,
               std::vector<CloudPoint>* morphFrom,
               float* morphT,
               std::vector<CloudPoint>* cloud) {
    Rectangle toggle{static_cast<float>(kScreenWidth - 190), 12.0f, 166.0f, 34.0f};
    if (*controlsMode == 0) {
        if (DrawButton(toggle, "Open Controls")) {
            *controlsMode = 1;
            return;
        }
        return;
    }

    DrawRectangle(kPanelX - 14, 0, kScreenWidth - kPanelX + 14, kScreenHeight, {10, 14, 24, 246});
    DrawRectangle(kPanelX - 14, 0, 1, kScreenHeight, {88, 100, 136, 180});
    DrawText("Atom Lab", kPanelX, 20, 26, {232, 236, 248, 255});
    if (DrawButton({static_cast<float>(kPanelX), 58.0f, 146.0f, 30.0f}, "Hide")) {
        *controlsMode = 0;
        return;
    }
    const float tabY = 108.0f;
    const std::array<const char*, 4> tabs{{"Atom", "View", "Slice", "Tools"}};
    for (int i = 0; i < 4; ++i) {
        Rectangle tab{static_cast<float>(kPanelX + i * 78), tabY, 68.0f, 30.0f};
        if (DrawButton(tab, tabs[i], *panelTab == i)) *panelTab = i;
    }

    const float y0 = 168.0f;
    if (*panelTab == 0) {
        DrawText("Mode", kPanelX, static_cast<int>(y0), 20, {226, 232, 244, 255});
        if (DrawButton({static_cast<float>(kPanelX), y0 + 32.0f, 146.0f, 34.0f}, "Orbital", !*elementMode)) {
            if (*elementMode) {
                *elementMode = false;
                *morphFrom = *cloud;
                *cloud = BuildCloud(*orbital, GetRandomValue(1, 999999));
                *scale = kOrbitals[*orbital].scale;
                *morphT = 0.0f;
            }
        }
        if (DrawButton({static_cast<float>(kPanelX + 158), y0 + 32.0f, 146.0f, 34.0f}, "Element", *elementMode)) {
            if (!*elementMode) {
                *elementMode = true;
                *morphFrom = *cloud;
                *cloud = BuildElementCloud(*element, GetRandomValue(1, 999999));
                *scale = kElements[*element].scale;
                *morphT = 0.0f;
            }
        }

        if (*elementMode) {
            DrawText("Element", kPanelX, static_cast<int>(y0 + 98), 20, {226, 232, 244, 255});
            for (int i = 0; i < static_cast<int>(kElements.size()); ++i) {
                const int col = i % 3;
                const int row = i / 3;
                Rectangle r{static_cast<float>(kPanelX + col * 102), y0 + 132.0f + row * 42.0f, 92.0f, 32.0f};
                if (DrawButton(r, kElements[i].symbol, i == *element)) SelectElement(i, element, scale, cloud, morphFrom, morphT);
            }
            DrawText(kElements[*element].name, kPanelX, static_cast<int>(y0 + 318), 18, {154, 168, 198, 255});
        } else {
            DrawText("Orbital", kPanelX, static_cast<int>(y0 + 98), 20, {226, 232, 244, 255});
            for (int i = 0; i < static_cast<int>(kOrbitals.size()); ++i) {
                const int col = i % 2;
                const int row = i / 2;
                Rectangle r{static_cast<float>(kPanelX + col * 154), y0 + 132.0f + row * 42.0f, 142.0f, 32.0f};
                if (DrawButton(r, kOrbitals[i].label, i == *orbital)) SelectOrbital(i, orbital, scale, cloud, morphFrom, morphT);
            }
            DrawText(kOrbitals[*orbital].caption, kPanelX, static_cast<int>(y0 + 318), 18, {154, 168, 198, 255});
        }
    } else if (*panelTab == 1) {
        DrawText("View", kPanelX, static_cast<int>(y0), 20, {226, 232, 244, 255});
        DrawSlider({static_cast<float>(kPanelX), y0 + 64.0f, static_cast<float>(kPanelW), 12.0f}, "Density", visibleDensity, 0.10f, 1.0f);
        DrawSlider({static_cast<float>(kPanelX), y0 + 136.0f, static_cast<float>(kPanelW), 12.0f}, "Scale", scale, 1.7f, 6.2f);
        DrawSlider({static_cast<float>(kPanelX), y0 + 208.0f, static_cast<float>(kPanelW), 12.0f}, "Point size", pointSize, 1.0f, 3.0f);
        DrawSlider({static_cast<float>(kPanelX), y0 + 280.0f, static_cast<float>(kPanelW), 12.0f}, "Animation", animationSpeed, 0.0f, 2.5f);
        DrawIntSlider({static_cast<float>(kPanelX), y0 + 352.0f, static_cast<float>(kPanelW), 12.0f}, "Atoms", atomCount, 1, 12);
    } else if (*panelTab == 2) {
        DrawText("Slice", kPanelX, static_cast<int>(y0), 20, {226, 232, 244, 255});
        if (DrawButton({static_cast<float>(kPanelX), y0 + 36.0f, 146.0f, 34.0f}, *slice ? "On" : "Off", *slice)) *slice = !*slice;
        if (DrawButton({static_cast<float>(kPanelX + 158), y0 + 36.0f, 46.0f, 34.0f}, "X", *sliceAxis == 0)) *sliceAxis = 0;
        if (DrawButton({static_cast<float>(kPanelX + 210), y0 + 36.0f, 46.0f, 34.0f}, "Y", *sliceAxis == 1)) *sliceAxis = 1;
        if (DrawButton({static_cast<float>(kPanelX + 262), y0 + 36.0f, 46.0f, 34.0f}, "Z", *sliceAxis == 2)) *sliceAxis = 2;
        DrawSlider({static_cast<float>(kPanelX), y0 + 132.0f, static_cast<float>(kPanelW), 12.0f}, "Position", slicePosition, -1.0f, 1.0f);
        DrawSlider({static_cast<float>(kPanelX), y0 + 212.0f, static_cast<float>(kPanelW), 12.0f}, "Thickness", sliceThickness, 0.03f, 0.70f);
    } else {
        DrawText("Display", kPanelX, static_cast<int>(y0), 20, {226, 232, 244, 255});
        if (DrawButton({static_cast<float>(kPanelX), y0 + 36.0f, 146.0f, 34.0f}, *phaseColor ? "Phase" : "Heat", *phaseColor)) *phaseColor = !*phaseColor;
        if (DrawButton({static_cast<float>(kPanelX + 158), y0 + 36.0f, 146.0f, 34.0f}, *cinematic ? "Cinema" : "Manual", *cinematic)) *cinematic = !*cinematic;
        if (DrawButton({static_cast<float>(kPanelX), y0 + 84.0f, 146.0f, 34.0f}, *detections ? "Detect" : "Quiet", *detections)) *detections = !*detections;
        if (DrawButton({static_cast<float>(kPanelX + 158), y0 + 84.0f, 146.0f, 34.0f}, *paused ? "Play" : "Pause", *paused)) *paused = !*paused;

        DrawText("Reaction", kPanelX, static_cast<int>(y0 + 150), 20, {226, 232, 244, 255});
        if (DrawButton({static_cast<float>(kPanelX), y0 + 184.0f, 94.0f, 34.0f}, "Idle", *reaction == ReactionMode::kIdle)) {
            *reaction = ReactionMode::kIdle;
            *reactionTime = 0.0f;
        }
        if (DrawButton({static_cast<float>(kPanelX + 104), y0 + 184.0f, 94.0f, 34.0f}, "Fusion", *reaction == ReactionMode::kFusion)) {
            *reaction = ReactionMode::kFusion;
            *reactionTime = 0.0f;
        }
        if (DrawButton({static_cast<float>(kPanelX + 208), y0 + 184.0f, 94.0f, 34.0f}, "Fission", *reaction == ReactionMode::kFission)) {
            *reaction = ReactionMode::kFission;
            *reactionTime = 0.0f;
        }

        DrawText("Actions", kPanelX, static_cast<int>(y0 + 250), 20, {226, 232, 244, 255});
        if (DrawButton({static_cast<float>(kPanelX), y0 + 284.0f, 146.0f, 34.0f}, "Resample")) {
            *morphFrom = *cloud;
            *cloud = *elementMode ? BuildElementCloud(*element, GetRandomValue(1, 999999)) : BuildCloud(*orbital, GetRandomValue(1, 999999));
            *morphT = 0.0f;
        }
        if (DrawButton({static_cast<float>(kPanelX + 158), y0 + 284.0f, 146.0f, 34.0f}, "Screenshot")) {
            const std::string path = ScreenshotPath(*elementMode ? kElements[*element].symbol : kOrbitals[*orbital].label);
            TakeScreenshot(path.c_str());
        }
        DrawIntSlider({static_cast<float>(kPanelX), y0 + 374.0f, static_cast<float>(kPanelW), 12.0f}, "Core dots", nucleusCount, 1, 24);
    }
    const int protons = *elementMode ? kElements[*element].protons : 1;
    const int neutrons = *elementMode ? std::max(0, kElements[*element].massNumber - kElements[*element].protons) : 0;
    *nucleus = BuildNucleus(protons, neutrons, *nucleusCount);

    if (*elementMode) {
        DrawRectangle(kPanelX, 892, 8, 8, {46, 58, 118, 224});
        DrawText("inner", kPanelX + 14, 886, 14, {144, 158, 190, 255});
        DrawRectangle(kPanelX + 74, 892, 8, 8, {156, 64, 178, 238});
        DrawText("middle", kPanelX + 88, 886, 14, {144, 158, 190, 255});
        DrawRectangle(kPanelX + 162, 892, 8, 8, {242, 132, 255, 236});
        DrawText("valence", kPanelX + 176, 886, 14, {144, 158, 190, 255});
    }
}
} // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Atomic Orbital Explorer - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {11.0f, 5.8f, 11.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 38.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    OrbitCamera orbit{};
    int orbital = 0;
    int element = 0;
    bool elementMode = false;
    float visibleDensity = 0.92f;
    float scale = kOrbitals[orbital].scale;
    float pointSize = 2.0f;
    float animationSpeed = 0.65f;
    bool phaseColor = false;
    bool slice = false;
    bool paused = false;
    bool cinematic = false;
    bool detections = true;
    int controlsMode = 0;
    int panelTab = 0;
    ReactionMode reaction = ReactionMode::kIdle;
    float reactionTime = 0.0f;
    int sliceAxis = 1;
    float slicePosition = 0.0f;
    float sliceThickness = 0.28f;
    int nucleusCount = 8;
    int atomCount = 1;
    float morphT = 1.0f;
    float detectionTimer = 0.0f;
    float time = 0.0f;

    std::vector<CloudPoint> cloud = BuildCloud(orbital, 31);
    std::vector<CloudPoint> morphFrom;
    std::vector<Nucleon> nucleus = BuildNucleus(1, 0, nucleusCount);
    std::vector<DetectionFlash> flashes;
    std::mt19937 detectionRng(411);
    ReactionModels reactionModels{
        {BuildElementCloudForProtons(1, 701), BuildNucleus(1, 1, 8), 2.20f},
        {BuildElementCloudForProtons(1, 702), BuildNucleus(1, 2, 9), 2.28f},
        {BuildElementCloudForProtons(2, 703), BuildNucleus(2, 2, 12), 2.94f},
        {BuildElementCloudForProtons(92, 704), BuildNucleus(92, 143, 28), 4.90f},
        {BuildElementCloudForProtons(56, 705), BuildNucleus(56, 85, 24), 3.70f},
        {BuildElementCloudForProtons(36, 706), BuildNucleus(36, 56, 22), 3.45f},
    };

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_S)) slice = !slice;
        if (IsKeyPressed(KEY_C)) phaseColor = !phaseColor;
        if (IsKeyPressed(KEY_V)) cinematic = !cinematic;
        if (IsKeyPressed(KEY_D)) detections = !detections;
        if (IsKeyPressed(KEY_F)) {
            reaction = reaction == ReactionMode::kFusion ? ReactionMode::kIdle : ReactionMode::kFusion;
            reactionTime = 0.0f;
        }
        if (IsKeyPressed(KEY_G)) {
            reaction = reaction == ReactionMode::kFission ? ReactionMode::kIdle : ReactionMode::kFission;
            reactionTime = 0.0f;
        }
        if (reaction != ReactionMode::kIdle) {
            if (IsKeyPressed(KEY_LEFT)) StepReactionStage(reaction, &reactionTime, &paused, -1);
            if (IsKeyPressed(KEY_RIGHT)) StepReactionStage(reaction, &reactionTime, &paused, 1);
            if (IsKeyPressed(KEY_SPACE)) paused = !paused;
        }
        if (IsKeyPressed(KEY_E)) {
            elementMode = !elementMode;
            morphFrom = cloud;
            cloud = elementMode ? BuildElementCloud(element, GetRandomValue(1, 999999)) : BuildCloud(orbital, GetRandomValue(1, 999999));
            scale = elementMode ? kElements[element].scale : kOrbitals[orbital].scale;
            morphT = 0.0f;
            const int protons = elementMode ? kElements[element].protons : 1;
            const int neutrons = elementMode ? std::max(0, kElements[element].massNumber - kElements[element].protons) : 0;
            nucleus = BuildNucleus(protons, neutrons, nucleusCount);
        }
        if (IsKeyPressed(KEY_R)) {
            if (reaction != ReactionMode::kIdle) {
                reactionTime = 0.0f;
                paused = false;
            } else {
                morphFrom = cloud;
                cloud = elementMode ? BuildElementCloud(element, GetRandomValue(1, 999999)) : BuildCloud(orbital, GetRandomValue(1, 999999));
                morphT = 0.0f;
            }
        }
        if (IsKeyPressed(KEY_K)) {
            const std::string path = ScreenshotPath(elementMode ? kElements[element].symbol : kOrbitals[orbital].label);
            TakeScreenshot(path.c_str());
        }
        if (elementMode) {
            if (IsKeyPressed(KEY_ONE)) SelectElement(0, &element, &scale, &cloud, &morphFrom, &morphT);
            if (IsKeyPressed(KEY_TWO)) SelectElement(1, &element, &scale, &cloud, &morphFrom, &morphT);
            if (IsKeyPressed(KEY_THREE)) SelectElement(2, &element, &scale, &cloud, &morphFrom, &morphT);
            if (IsKeyPressed(KEY_FOUR)) SelectElement(3, &element, &scale, &cloud, &morphFrom, &morphT);
            if (IsKeyPressed(KEY_FIVE)) SelectElement(4, &element, &scale, &cloud, &morphFrom, &morphT);
            if (IsKeyPressed(KEY_SIX)) SelectElement(5, &element, &scale, &cloud, &morphFrom, &morphT);
            if (IsKeyPressed(KEY_SEVEN)) SelectElement(6, &element, &scale, &cloud, &morphFrom, &morphT);
            if (IsKeyPressed(KEY_EIGHT)) SelectElement(7, &element, &scale, &cloud, &morphFrom, &morphT);
        } else {
            if (IsKeyPressed(KEY_ONE)) SelectOrbital(0, &orbital, &scale, &cloud, &morphFrom, &morphT);
            if (IsKeyPressed(KEY_TWO)) SelectOrbital(1, &orbital, &scale, &cloud, &morphFrom, &morphT);
            if (IsKeyPressed(KEY_THREE)) SelectOrbital(2, &orbital, &scale, &cloud, &morphFrom, &morphT);
            if (IsKeyPressed(KEY_FOUR)) SelectOrbital(3, &orbital, &scale, &cloud, &morphFrom, &morphT);
            if (IsKeyPressed(KEY_FIVE)) SelectOrbital(4, &orbital, &scale, &cloud, &morphFrom, &morphT);
            if (IsKeyPressed(KEY_SIX)) SelectOrbital(5, &orbital, &scale, &cloud, &morphFrom, &morphT);
            if (IsKeyPressed(KEY_SEVEN)) SelectOrbital(6, &orbital, &scale, &cloud, &morphFrom, &morphT);
            if (IsKeyPressed(KEY_EIGHT)) SelectOrbital(7, &orbital, &scale, &cloud, &morphFrom, &morphT);
        }

        const float dt = GetFrameTime();
        const bool reactionActive = reaction != ReactionMode::kIdle;
        if (cinematic && !paused && !MouseOverControls(controlsMode, reactionActive)) orbit.yaw += dt * 0.22f;
        UpdateOrbitCamera(&camera, &orbit, controlsMode, reactionActive);
        if (!paused) {
            time += dt * animationSpeed;
            if (reaction != ReactionMode::kIdle) reactionTime += dt * animationSpeed;
            morphT = std::min(1.0f, morphT + dt * 0.75f);
            detectionTimer += dt;
            if (detections && detectionTimer > 0.075f && !cloud.empty()) {
                detectionTimer = 0.0f;
                std::uniform_int_distribution<int> dist(0, std::max(0, static_cast<int>(cloud.size() * visibleDensity) - 1));
                for (int i = 0; i < 3; ++i) {
                    const int index = dist(detectionRng);
                    CloudPoint p = cloud[index];
                    if (!morphFrom.empty() && morphT < 1.0f && index < static_cast<int>(morphFrom.size())) p = BlendPoint(morphFrom[index], p, morphT);
                    if (slice && std::fabs(SliceCoord(p.pos, sliceAxis) - slicePosition * kWorldRadius) > sliceThickness * kWorldRadius) continue;
                    flashes.push_back({p.pos, 0.0f, RandRange(detectionRng, 0.22f, 0.42f)});
                }
            }
            for (DetectionFlash& flash : flashes) flash.age += dt;
            flashes.erase(std::remove_if(flashes.begin(), flashes.end(), [](const DetectionFlash& f) { return f.age >= f.life; }), flashes.end());
        }

        BeginDrawing();
        ClearBackground({4, 6, 12, 255});

        if (reaction == ReactionMode::kIdle) {
            const std::vector<Vector3> atomOffsets = AtomOffsets(atomCount, scale);
            const float perAtomDensity = visibleDensity / std::sqrt(static_cast<float>(atomCount));
            for (const Vector3& offset : atomOffsets) {
                DrawProjectedCloud(cloud, morphFrom, camera, offset, perAtomDensity, scale, pointSize, time, animationSpeed, phaseColor, elementMode, slice, sliceAxis, slicePosition, sliceThickness, morphT);
                DrawProjectedNucleus(nucleus, camera, offset, scale, time, controlsMode != 0);
            }
            DrawDetectionFlashes(flashes, camera, atomOffsets.front(), scale);
        } else {
            DrawReactionModels(reactionModels, camera, reaction, reactionTime, time, animationSpeed, phaseColor);
            DrawReactionScene(camera, reaction, reactionTime, time);
            DrawReactionLegend(phaseColor);
            DrawReactionSciencePanel(reaction, reactionTime);
            DrawReactionTimeline(reaction, &reactionTime, &paused, &animationSpeed);
        }

        if (reaction == ReactionMode::kIdle) {
            DrawText(elementMode ? kElements[element].name : kOrbitals[orbital].label, 24, kScreenHeight - 46, 28, {226, 232, 244, 230});
            DrawText(elementMode ? kElements[element].config : kOrbitals[orbital].quantum, 24, kScreenHeight - 76, 18, {172, 184, 210, 220});
            if (elementMode) {
                std::ostringstream info;
                info << "Z=" << kElements[element].protons << "  electrons=" << kElements[element].protons
                     << "  atoms=" << atomCount << "  shell-aware approximate density";
                DrawText(info.str().c_str(), 24, kScreenHeight - 130, 16, {128, 142, 172, 210});
            }
            DrawText("E mode | F fusion | G fission | K shot | V orbit | D detect", 24, kScreenHeight - 104, 16, {128, 142, 172, 210});
        } else {
            DrawText("R replay | LEFT/RIGHT stage | SPACE play/pause | C heat/phase | F/G exit", 24, kScreenHeight - 78, 16, {144, 160, 190, 225});
        }
        if (slice) {
            std::ostringstream os;
            os << "Slice " << SliceAxisName(sliceAxis) << " = " << std::fixed << std::setprecision(2) << slicePosition
               << "   thickness " << sliceThickness;
            DrawText(os.str().c_str(), 24, 58, 18, {204, 214, 236, 220});
        }
        if (cinematic) DrawText("CINEMATIC", 24, 32, 18, {204, 214, 236, 220});
        DrawPanel(&orbital, &element, &elementMode, &visibleDensity, &scale, &pointSize, &animationSpeed, &phaseColor, &slice, &paused, &cinematic, &detections,
                  &reaction, &reactionTime, &sliceAxis, &slicePosition, &sliceThickness, &nucleusCount, &atomCount, &controlsMode, &panelTab, &nucleus, &morphFrom, &morphT, &cloud);
        DrawFPS(14, 12);

        EndDrawing();
    }

    EnableCursor();
    CloseWindow();
    return 0;
}
