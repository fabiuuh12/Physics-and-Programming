#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kScreenWidth = 1380;
constexpr int kScreenHeight = 860;
constexpr float kSimDuration = 4.4f;

enum class ParticleStyle {
    kFermion,
    kPhoton,
    kBoson,
    kGluon,
    kScalar
};

enum class RateModel {
    kResonantSChannel,
    kQEDScattering,
    kWeakSuppressed
};

enum class Stage {
    kIncoming,
    kMediator,
    kOutgoing,
    kComplete
};

struct DiagramEdge {
    int from;
    int to;
    const char* label;
    ParticleStyle style;
    Color color;
    bool showArrow;
    bool forwardArrow;
    int phase;
    Vector2 labelOffset;
};

struct ConservationCheck {
    const char* label;
    bool valid;
};

struct FeynmanProcess {
    const char* name;
    const char* reaction;
    const char* description;
    const char* mediator;
    float energyMin;
    float energyMax;
    float defaultEnergy;
    float resonanceEnergy;
    float resonanceWidth;
    float baseCoupling;
    RateModel rateModel;
    std::vector<Vector2> nodes;
    std::vector<DiagramEdge> edges;
    std::vector<int> vertexNodeIndices;
    std::vector<ConservationCheck> checks;
};

float Saturate(float x) {
    return std::clamp(x, 0.0f, 1.0f);
}

float EaseOutCubic(float x) {
    const float t = 1.0f - Saturate(x);
    return 1.0f - t * t * t;
}

Color WithAlpha(Color c, unsigned char alpha) {
    c.a = alpha;
    return c;
}

std::string FormatFloat(float value, int decimals = 2) {
    std::ostringstream os;
    os.setf(std::ios::fixed);
    os.precision(decimals);
    os << value;
    return os.str();
}

const char* StageLabel(Stage stage) {
    switch (stage) {
        case Stage::kIncoming: return "incoming";
        case Stage::kMediator: return "mediator";
        case Stage::kOutgoing: return "outgoing";
        case Stage::kComplete: return "complete";
    }
    return "unknown";
}

Stage CurrentStage(float simTime) {
    if (simTime < 1.35f) return Stage::kIncoming;
    if (simTime < 2.35f) return Stage::kMediator;
    if (simTime < 4.0f) return Stage::kOutgoing;
    return Stage::kComplete;
}

void UpdateOrbitCameraDragOnly(Camera3D* camera, float* yaw, float* pitch, float* distance) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON) && GetMouseY() > 70) {
        const Vector2 delta = GetMouseDelta();
        *yaw -= delta.x * 0.0038f;
        *pitch += delta.y * 0.0036f;
        *pitch = std::clamp(*pitch, -0.95f, 1.1f);
    }

    *distance -= GetMouseWheelMove() * 0.55f;
    *distance = std::clamp(*distance, 8.0f, 24.0f);

    const float cp = std::cos(*pitch);
    const Vector3 offset = {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw)
    };
    camera->position = Vector3Add(camera->target, offset);
}

Vector3 DiagramToWorld(Vector2 p) {
    return {
        (p.x - 690.0f) * 0.0125f,
        (430.0f - p.y) * 0.0105f,
        0.0f
    };
}

Vector3 WorldLerp(const Vector3& a, const Vector3& b, float t) {
    return Vector3Lerp(a, b, t);
}

Vector3 ArrowBaseDirection(const Vector3& start, const Vector3& end, bool forwardArrow) {
    return forwardArrow ? Vector3Subtract(end, start) : Vector3Subtract(start, end);
}

void DrawTubeSegment(const Vector3& a, const Vector3& b, float radius, Color color) {
    DrawCylinderEx(a, b, radius, radius, 8, color);
}

void DrawArrowHead3D(const Vector3& start, const Vector3& end, bool forwardArrow, float radius, Color color) {
    const float t = forwardArrow ? 0.62f : 0.38f;
    const Vector3 tip = WorldLerp(start, end, t);
    const Vector3 dir = Vector3Normalize(ArrowBaseDirection(start, end, forwardArrow));
    const Vector3 base = Vector3Add(tip, Vector3Scale(dir, -0.34f));
    DrawCylinderEx(base, tip, radius * 1.85f, 0.0f, 10, color);
}

std::vector<Vector3> BuildStyledPath(const DiagramEdge& edge, const std::vector<Vector2>& nodes) {
    const Vector3 start = DiagramToWorld(nodes[edge.from]);
    const Vector3 end = DiagramToWorld(nodes[edge.to]);
    const Vector3 dir = Vector3Normalize(Vector3Subtract(end, start));
    const Vector3 up = {0.0f, 1.0f, 0.0f};
    Vector3 side = Vector3CrossProduct(up, dir);
    if (Vector3Length(side) < 0.001f) side = {1.0f, 0.0f, 0.0f};
    side = Vector3Normalize(side);

    std::vector<Vector3> points;
    int segments = 1;
    float amplitude = 0.0f;
    float cycles = 1.0f;
    float sideMix = 0.0f;

    switch (edge.style) {
        case ParticleStyle::kPhoton:
            segments = 56;
            amplitude = 0.22f;
            cycles = 5.0f;
            break;
        case ParticleStyle::kBoson:
            segments = 44;
            amplitude = 0.13f;
            cycles = 4.5f;
            break;
        case ParticleStyle::kGluon:
            segments = 72;
            amplitude = 0.16f;
            cycles = 11.0f;
            sideMix = 0.08f;
            break;
        case ParticleStyle::kScalar:
        case ParticleStyle::kFermion:
            points.push_back(start);
            points.push_back(end);
            return points;
    }

    points.reserve(segments + 1);
    for (int i = 0; i <= segments; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(segments);
        Vector3 point = WorldLerp(start, end, t);
        const float wave = std::sin(t * 2.0f * PI * cycles) * amplitude;
        point.z += wave;
        if (edge.style == ParticleStyle::kGluon) {
            point = Vector3Add(point, Vector3Scale(side, std::cos(t * 2.0f * PI * cycles) * sideMix));
        }
        points.push_back(point);
    }
    return points;
}

void DrawStyledEdge3D(const DiagramEdge& edge, const std::vector<Vector2>& nodes, bool active) {
    const std::vector<Vector3> points = BuildStyledPath(edge, nodes);
    const float radius = active ? 0.045f : 0.028f;
    Color color = active ? edge.color : WithAlpha(edge.color, 125);

    if (edge.style == ParticleStyle::kScalar) {
        for (std::size_t i = 0; i + 1 < points.size(); ++i) {
            const Vector3 segment = Vector3Subtract(points[i + 1], points[i]);
            const float length = Vector3Length(segment);
            const Vector3 dir = Vector3Normalize(segment);
            float walked = 0.0f;
            while (walked < length) {
                const float dashEnd = std::min(walked + 0.28f, length);
                const Vector3 a = Vector3Add(points[i], Vector3Scale(dir, walked));
                const Vector3 b = Vector3Add(points[i], Vector3Scale(dir, dashEnd));
                DrawTubeSegment(a, b, radius, color);
                walked = dashEnd + 0.16f;
            }
        }
    } else {
        for (std::size_t i = 0; i + 1 < points.size(); ++i) {
            DrawTubeSegment(points[i], points[i + 1], radius, color);
        }
    }

    if (edge.showArrow) {
        DrawArrowHead3D(DiagramToWorld(nodes[edge.from]), DiagramToWorld(nodes[edge.to]), edge.forwardArrow, radius, color);
    }
}

Vector3 EdgePacketPosition(const DiagramEdge& edge, const std::vector<Vector2>& nodes, float progress) {
    const Vector3 start = DiagramToWorld(nodes[edge.from]);
    const Vector3 end = DiagramToWorld(nodes[edge.to]);
    const float t = EaseOutCubic(progress);

    if (edge.style == ParticleStyle::kFermion || edge.style == ParticleStyle::kScalar) {
        return WorldLerp(start, end, t);
    }

    const Vector3 dir = Vector3Normalize(Vector3Subtract(end, start));
    const Vector3 up = {0.0f, 1.0f, 0.0f};
    Vector3 side = Vector3CrossProduct(up, dir);
    if (Vector3Length(side) < 0.001f) side = {1.0f, 0.0f, 0.0f};
    side = Vector3Normalize(side);

    Vector3 point = WorldLerp(start, end, t);
    if (edge.style == ParticleStyle::kPhoton) point.z += std::sin(t * 2.0f * PI * 5.0f) * 0.22f;
    if (edge.style == ParticleStyle::kBoson) point.z += std::sin(t * 2.0f * PI * 4.5f) * 0.13f;
    if (edge.style == ParticleStyle::kGluon) {
        point.z += std::sin(t * 2.0f * PI * 11.0f) * 0.16f;
        point = Vector3Add(point, Vector3Scale(side, std::cos(t * 2.0f * PI * 11.0f) * 0.08f));
    }
    return point;
}

void DrawPacket3D(const DiagramEdge& edge, const std::vector<Vector2>& nodes, float progress) {
    const Vector3 pos = EdgePacketPosition(edge, nodes, progress);
    float radius = 0.13f;
    if (edge.style == ParticleStyle::kFermion) radius = 0.11f;
    if (edge.style == ParticleStyle::kScalar) radius = 0.1f;
    if (edge.style == ParticleStyle::kGluon) radius = 0.14f;

    DrawSphere(pos, radius * 1.8f, WithAlpha(edge.color, 40));
    DrawSphere(pos, radius, edge.color);
    DrawSphereWires(pos, radius * 1.35f, 8, 8, WithAlpha(edge.color, 110));
}

void DrawVertexGlow3D(const Vector3& center, float pulse) {
    const float r = 0.12f + 0.18f * pulse;
    DrawSphere(center, r * 1.85f, WithAlpha(Color{255, 231, 155, 255}, static_cast<unsigned char>(80.0f * pulse)));
    DrawSphere(center, r, Color{255, 241, 184, 255});
    DrawSphereWires(center, r * 1.6f, 10, 10, WithAlpha(Color{255, 214, 130, 255}, static_cast<unsigned char>(180.0f * pulse)));
}

void DrawCircleOutline3D(Vector3 center, float radius, float z, int segments, Color color) {
    for (int i = 0; i < segments; ++i) {
        const float a0 = 2.0f * PI * static_cast<float>(i) / static_cast<float>(segments);
        const float a1 = 2.0f * PI * static_cast<float>(i + 1) / static_cast<float>(segments);
        const Vector3 p0 = {center.x + radius * std::cos(a0), center.y + radius * std::sin(a0), z};
        const Vector3 p1 = {center.x + radius * std::cos(a1), center.y + radius * std::sin(a1), z};
        DrawLine3D(p0, p1, color);
    }
}

float PhaseProgress(int phase, float simTime) {
    if (phase == 0) return Saturate(simTime / 1.35f);
    if (phase == 1) return Saturate((simTime - 1.35f) / 1.0f);
    if (phase == 2) return Saturate((simTime - 2.35f) / 1.65f);
    return 0.0f;
}

bool PhaseActive(int phase, float simTime) {
    if (phase == 0) return simTime <= 1.35f;
    if (phase == 1) return simTime >= 1.2f && simTime <= 2.45f;
    if (phase == 2) return simTime >= 2.25f;
    return false;
}

float RelativeRate(const FeynmanProcess& process, float energyGeV, float couplingScale) {
    float response = 0.0f;
    switch (process.rateModel) {
        case RateModel::kResonantSChannel: {
            const float offset = (energyGeV - process.resonanceEnergy) / std::max(process.resonanceWidth, 1.0f);
            response = 0.30f + 0.70f / (1.0f + offset * offset);
            break;
        }
        case RateModel::kQEDScattering: {
            const float scaled = (energyGeV - process.energyMin) / std::max(process.energyMax - process.energyMin, 1.0f);
            response = 0.62f + 0.28f * (1.0f - 0.55f * scaled);
            break;
        }
        case RateModel::kWeakSuppressed: {
            response = 0.12f + 0.88f * energyGeV / (energyGeV + process.resonanceEnergy * 0.65f);
            break;
        }
    }
    return std::clamp(process.baseCoupling * couplingScale * response, 0.0f, 1.0f);
}

std::vector<FeynmanProcess> BuildProcesses() {
    std::vector<FeynmanProcess> processes;

    processes.push_back({
        "Muon Pair Production",
        "e- + e+ -> gamma*/Z -> mu- + mu+",
        "Neutral-current annihilation with a resonance peak near the Z mass.",
        "virtual gamma / Z",
        20.0f,
        160.0f,
        91.2f,
        91.2f,
        10.0f,
        0.88f,
        RateModel::kResonantSChannel,
        {
            {180.0f, 290.0f}, {180.0f, 570.0f}, {520.0f, 430.0f},
            {860.0f, 430.0f}, {1200.0f, 290.0f}, {1200.0f, 570.0f},
        },
        {
            {0, 2, "e-", ParticleStyle::kFermion, Color{90, 215, 255, 255}, true, true, 0, {-16.0f, -28.0f}},
            {1, 2, "e+", ParticleStyle::kFermion, Color{255, 128, 156, 255}, true, false, 0, {-16.0f, 10.0f}},
            {2, 3, "gamma* / Z", ParticleStyle::kPhoton, Color{255, 219, 102, 255}, false, true, 1, {-28.0f, -28.0f}},
            {3, 4, "mu-", ParticleStyle::kFermion, Color{112, 245, 193, 255}, true, true, 2, {10.0f, -28.0f}},
            {3, 5, "mu+", ParticleStyle::kFermion, Color{210, 170, 255, 255}, true, false, 2, {10.0f, 10.0f}},
        },
        {2, 3},
        {{"charge", true}, {"lepton flow", true}, {"4-momentum", true}},
    });

    processes.push_back({
        "Compton Scattering",
        "e- + gamma -> e- + gamma",
        "QED scattering through a virtual electron propagator.",
        "virtual electron",
        5.0f,
        90.0f,
        24.0f,
        0.0f,
        1.0f,
        0.82f,
        RateModel::kQEDScattering,
        {
            {180.0f, 540.0f}, {180.0f, 260.0f}, {520.0f, 430.0f},
            {860.0f, 430.0f}, {1200.0f, 260.0f}, {1200.0f, 540.0f},
        },
        {
            {0, 2, "e-", ParticleStyle::kFermion, Color{98, 224, 255, 255}, true, true, 0, {-16.0f, 10.0f}},
            {1, 2, "gamma", ParticleStyle::kPhoton, Color{255, 220, 108, 255}, false, true, 0, {-16.0f, -28.0f}},
            {2, 3, "e- virtual", ParticleStyle::kFermion, Color{190, 226, 255, 255}, true, true, 1, {-24.0f, -30.0f}},
            {3, 4, "gamma", ParticleStyle::kPhoton, Color{255, 220, 108, 255}, false, true, 2, {10.0f, -28.0f}},
            {3, 5, "e-", ParticleStyle::kFermion, Color{98, 224, 255, 255}, true, true, 2, {10.0f, 10.0f}},
        },
        {2, 3},
        {{"charge", true}, {"fermion number", true}, {"QED vertex", true}},
    });

    processes.push_back({
        "Beta Decay Vertex",
        "d -> u + W- ; W- -> e- + anti-nu_e",
        "Weak decay with a heavy virtual W suppressing the rate at low scale.",
        "virtual W-",
        1.0f,
        120.0f,
        8.0f,
        80.4f,
        12.0f,
        0.54f,
        RateModel::kWeakSuppressed,
        {
            {180.0f, 430.0f}, {520.0f, 430.0f}, {820.0f, 560.0f},
            {1180.0f, 250.0f}, {1180.0f, 470.0f}, {1180.0f, 650.0f},
        },
        {
            {0, 1, "d", ParticleStyle::kFermion, Color{107, 208, 255, 255}, true, true, 0, {-8.0f, -28.0f}},
            {1, 3, "u", ParticleStyle::kFermion, Color{112, 245, 193, 255}, true, true, 2, {-8.0f, -28.0f}},
            {1, 2, "W-", ParticleStyle::kBoson, Color{255, 177, 94, 255}, false, true, 1, {-42.0f, 4.0f}},
            {2, 4, "e-", ParticleStyle::kFermion, Color{130, 228, 255, 255}, true, true, 2, {10.0f, -10.0f}},
            {2, 5, "anti-nu_e", ParticleStyle::kFermion, Color{220, 196, 255, 255}, true, false, 2, {-44.0f, -14.0f}},
        },
        {1, 2},
        {{"charge", true}, {"baryon number", true}, {"lepton number", true}},
    });

    processes.push_back({
        "Higgs Diphoton Channel",
        "g + g -> H* -> gamma + gamma",
        "Collider-style Higgs production with a scalar resonance around 125 GeV.",
        "virtual Higgs",
        80.0f,
        170.0f,
        125.0f,
        125.0f,
        7.0f,
        0.76f,
        RateModel::kResonantSChannel,
        {
            {180.0f, 290.0f}, {180.0f, 570.0f}, {520.0f, 430.0f},
            {860.0f, 430.0f}, {1200.0f, 290.0f}, {1200.0f, 570.0f},
        },
        {
            {0, 2, "g", ParticleStyle::kGluon, Color{255, 138, 115, 255}, false, true, 0, {-10.0f, -28.0f}},
            {1, 2, "g", ParticleStyle::kGluon, Color{255, 138, 115, 255}, false, true, 0, {-10.0f, 10.0f}},
            {2, 3, "H*", ParticleStyle::kScalar, Color{161, 255, 152, 255}, false, true, 1, {-8.0f, -28.0f}},
            {3, 4, "gamma", ParticleStyle::kPhoton, Color{255, 226, 110, 255}, false, true, 2, {10.0f, -28.0f}},
            {3, 5, "gamma", ParticleStyle::kPhoton, Color{255, 226, 110, 255}, false, true, 2, {10.0f, 10.0f}},
        },
        {2, 3},
        {{"charge", true}, {"color-neutral final", true}, {"scalar resonance", true}},
    });

    return processes;
}

void DrawBackdrop3D() {
    DrawPlane({0.0f, -4.2f, 0.0f}, {28.0f, 18.0f}, Color{8, 12, 20, 255});

    DrawCubeV({0.0f, 0.0f, -0.12f}, {18.4f, 8.6f, 0.1f}, Color{11, 20, 33, 200});
    DrawCubeWiresV({0.0f, 0.0f, -0.12f}, {18.4f, 8.6f, 0.1f}, Color{64, 100, 144, 255});

    for (int i = -8; i <= 8; ++i) {
        const float x = static_cast<float>(i);
        const Color c = (i % 2 == 0) ? Color{28, 48, 72, 110} : Color{22, 36, 56, 70};
        DrawLine3D({x, -4.0f, -0.11f}, {x, 4.0f, -0.11f}, c);
    }
    for (int j = -4; j <= 4; ++j) {
        const float y = static_cast<float>(j);
        const Color c = (j % 2 == 0) ? Color{28, 48, 72, 110} : Color{22, 36, 56, 70};
        DrawLine3D({-8.0f, y, -0.11f}, {8.0f, y, -0.11f}, c);
    }

    DrawCircleOutline3D({0.0f, 0.0f, 0.0f}, 1.52f, -0.35f, 40, Color{50, 95, 145, 130});
    DrawCircleOutline3D({0.0f, 0.0f, 0.0f}, 3.96f, -0.42f, 64, Color{44, 75, 118, 95});
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Feynman Diagram Simulator 3D - C++ (raylib)");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {11.5f, 4.7f, 11.5f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 42.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.79f;
    float camPitch = 0.28f;
    float camDistance = 16.0f;

    std::vector<FeynmanProcess> processes = BuildProcesses();
    int selectedProcess = 0;
    float energyGeV = processes[selectedProcess].defaultEnergy;
    float couplingScale = 1.0f;
    float simTime = 0.0f;
    bool paused = false;
    bool autoReplay = true;
    bool showHelp = false;

    auto selectProcess = [&](int index) {
        selectedProcess = (index + static_cast<int>(processes.size())) % static_cast<int>(processes.size());
        energyGeV = processes[selectedProcess].defaultEnergy;
        simTime = 0.0f;
    };

    while (!WindowShouldClose()) {
        const FeynmanProcess& process = processes[selectedProcess];

        const float tabGap = 10.0f;
        const float tabX = 18.0f;
        const float tabY = 14.0f;
        const float tabWidth = 250.0f;
        const float tabHeight = 34.0f;

        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && GetMouseY() < 60) {
            const Vector2 mouse = GetMousePosition();
            for (int i = 0; i < static_cast<int>(processes.size()); ++i) {
                Rectangle tab{tabX + i * (tabWidth + tabGap), tabY, tabWidth, tabHeight};
                if (CheckCollisionPointRec(mouse, tab)) {
                    selectProcess(i);
                    break;
                }
            }
        }

        if (IsKeyPressed(KEY_ONE)) selectProcess(0);
        if (IsKeyPressed(KEY_TWO) && processes.size() > 1) selectProcess(1);
        if (IsKeyPressed(KEY_THREE) && processes.size() > 2) selectProcess(2);
        if (IsKeyPressed(KEY_FOUR) && processes.size() > 3) selectProcess(3);
        if (IsKeyPressed(KEY_TAB)) selectProcess(selectedProcess + 1);
        if (IsKeyPressed(KEY_SPACE)) simTime = 0.0f;
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_A)) autoReplay = !autoReplay;
        if (IsKeyPressed(KEY_H)) showHelp = !showHelp;
        if (IsKeyPressed(KEY_R)) {
            couplingScale = 1.0f;
            energyGeV = process.defaultEnergy;
            simTime = 0.0f;
            paused = false;
        }

        energyGeV += GetFrameTime() * 38.0f * static_cast<float>(IsKeyDown(KEY_RIGHT) - IsKeyDown(KEY_LEFT));
        energyGeV = std::clamp(energyGeV, process.energyMin, process.energyMax);

        couplingScale += GetFrameTime() * 0.75f * static_cast<float>(IsKeyDown(KEY_UP) - IsKeyDown(KEY_DOWN));
        couplingScale = std::clamp(couplingScale, 0.35f, 1.8f);

        UpdateOrbitCameraDragOnly(&camera, &camYaw, &camPitch, &camDistance);

        if (!paused) {
            simTime += GetFrameTime();
            if (simTime > kSimDuration) simTime = autoReplay ? 0.0f : kSimDuration;
        }

        const float eventRate = RelativeRate(process, energyGeV, couplingScale);
        const Stage stage = CurrentStage(simTime);

        BeginDrawing();
        ClearBackground(Color{6, 9, 16, 255});

        DrawRectangleGradientV(0, 0, kScreenWidth, 140, Color{10, 18, 29, 220}, Color{10, 18, 29, 20});
        DrawRectangleGradientV(0, kScreenHeight - 120, kScreenWidth, 120, Color{8, 12, 20, 20}, Color{8, 12, 20, 210});

        BeginMode3D(camera);
        DrawBackdrop3D();

        for (const DiagramEdge& edge : process.edges) {
            const bool active = PhaseActive(edge.phase, simTime);
            DrawStyledEdge3D(edge, process.nodes, active);
            if (active) DrawPacket3D(edge, process.nodes, PhaseProgress(edge.phase, simTime));
        }

        for (std::size_t i = 0; i < process.vertexNodeIndices.size(); ++i) {
            float pulse = 0.0f;
            if (i == 0) pulse = 1.0f - std::fabs(simTime - 1.35f) / 0.28f;
            if (i == 1) pulse = 1.0f - std::fabs(simTime - 2.35f) / 0.28f;
            pulse = Saturate(pulse);
            DrawVertexGlow3D(DiagramToWorld(process.nodes[process.vertexNodeIndices[i]]), pulse);
        }

        EndMode3D();

        for (int i = 0; i < static_cast<int>(processes.size()); ++i) {
            Rectangle tab{tabX + i * (tabWidth + tabGap), tabY, tabWidth, tabHeight};
            const bool active = i == selectedProcess;
            DrawRectangleRounded(tab, 0.25f, 12, active ? Color{33, 82, 116, 255} : Color{13, 24, 39, 228});
            DrawRectangleRoundedLinesEx(tab, 0.25f, 12, 2.0f, active ? Color{117, 220, 255, 255} : Color{47, 76, 109, 255});
            DrawText(TextFormat("%d", i + 1), static_cast<int>(tab.x + 10), static_cast<int>(tab.y + 7), 18, Color{255, 226, 146, 255});
            DrawText(processes[i].name, static_cast<int>(tab.x + 30), static_cast<int>(tab.y + 7), 18, active ? RAYWHITE : Color{181, 196, 220, 255});
        }

        DrawText(process.name, 24, 64, 26, Color{236, 241, 250, 255});
        DrawText(process.reaction, 24, 94, 18, Color{125, 219, 255, 255});
        DrawText(process.description, 24, kScreenHeight - 62, 16, Color{166, 184, 208, 255});

        Rectangle stats{1088.0f, 78.0f, 266.0f, 130.0f};
        DrawRectangleRounded(stats, 0.08f, 18, Color{11, 19, 31, 214});
        DrawRectangleRoundedLinesEx(stats, 0.08f, 18, 2.0f, Color{49, 79, 113, 255});
        DrawText("stage", 1106, 96, 16, Color{174, 190, 214, 255});
        DrawText(StageLabel(stage), 1188, 96, 16, Color{255, 226, 146, 255});
        DrawText("rate", 1106, 122, 16, Color{174, 190, 214, 255});
        DrawText(FormatFloat(eventRate).c_str(), 1188, 122, 16, Color{126, 230, 188, 255});
        DrawText("energy", 1106, 148, 16, Color{174, 190, 214, 255});
        DrawText((FormatFloat(energyGeV, 1) + " GeV").c_str(), 1188, 148, 16, Color{125, 219, 255, 255});
        DrawText("coupling", 1106, 174, 16, Color{174, 190, 214, 255});
        DrawText(FormatFloat(couplingScale).c_str(), 1188, 174, 16, Color{255, 191, 114, 255});

        Rectangle checks{1088.0f, 220.0f, 266.0f, 104.0f};
        DrawRectangleRounded(checks, 0.08f, 18, Color{11, 19, 31, 198});
        DrawRectangleRoundedLinesEx(checks, 0.08f, 18, 2.0f, Color{49, 79, 113, 255});
        for (std::size_t i = 0; i < process.checks.size(); ++i) {
            const int y = 238 + static_cast<int>(i) * 24;
            DrawCircle(1104, y + 7, 5, process.checks[i].valid ? Color{90, 206, 132, 255} : Color{224, 92, 92, 255});
            DrawText(process.checks[i].label, 1118, y, 15, Color{206, 216, 232, 255});
        }

        for (const DiagramEdge& edge : process.edges) {
            const Vector3 a = DiagramToWorld(process.nodes[edge.from]);
            const Vector3 b = DiagramToWorld(process.nodes[edge.to]);
            Vector2 textPos = GetWorldToScreen(WorldLerp(a, b, 0.5f), camera);
            textPos.x += edge.labelOffset.x * 0.55f;
            textPos.y += edge.labelOffset.y * 0.55f;
            DrawText(edge.label, static_cast<int>(textPos.x), static_cast<int>(textPos.y), 15, WithAlpha(edge.color, 230));
        }

        DrawText("mouse drag orbit  wheel zoom  left/right energy  up/down coupling  space replay", 24, kScreenHeight - 34, 15, Color{156, 176, 201, 255});
        DrawText("A auto  P pause  H help  R reset", 992, kScreenHeight - 34, 15, Color{156, 176, 201, 255});

        if (showHelp) {
            Rectangle help{24.0f, 132.0f, 288.0f, 138.0f};
            DrawRectangleRounded(help, 0.08f, 18, Color{11, 19, 31, 220});
            DrawRectangleRoundedLinesEx(help, 0.08f, 18, 2.0f, Color{49, 79, 113, 255});
            DrawText("3D view", 42, 150, 18, RAYWHITE);
            DrawText("The diagram now sits on a vertical plane.", 42, 178, 15, Color{177, 192, 216, 255});
            DrawText("Colored spheres move along propagators.", 42, 202, 15, Color{177, 192, 216, 255});
            DrawText("Text is intentionally compact at the edges.", 42, 226, 15, Color{177, 192, 216, 255});
            DrawText("Curly = gluon, dashed = scalar.", 42, 250, 15, Color{255, 226, 146, 255});
        }

        DrawFPS(kScreenWidth - 92, 16);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
