#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <array>
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
    kBoson
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

Vector2 Add(Vector2 a, Vector2 b) {
    return {a.x + b.x, a.y + b.y};
}

Vector2 Sub(Vector2 a, Vector2 b) {
    return {a.x - b.x, a.y - b.y};
}

Vector2 Mul(Vector2 a, float s) {
    return {a.x * s, a.y * s};
}

Vector2 NormalizeOrZero(Vector2 v) {
    const float len = std::sqrt(v.x * v.x + v.y * v.y);
    if (len < 1e-5f) return {0.0f, 0.0f};
    return {v.x / len, v.y / len};
}

Vector2 Perp(Vector2 v) {
    return {-v.y, v.x};
}

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
        case Stage::kIncoming: return "incoming legs";
        case Stage::kMediator: return "virtual propagator";
        case Stage::kOutgoing: return "outgoing products";
        case Stage::kComplete: return "event complete";
    }
    return "unknown";
}

Stage CurrentStage(float simTime) {
    if (simTime < 1.35f) return Stage::kIncoming;
    if (simTime < 2.35f) return Stage::kMediator;
    if (simTime < 4.0f) return Stage::kOutgoing;
    return Stage::kComplete;
}

std::array<Vector2, 3> ArrowHead(Vector2 tip, Vector2 direction, float size) {
    const Vector2 dir = NormalizeOrZero(direction);
    const Vector2 side = Perp(dir);
    return {
        tip,
        Add(tip, Add(Mul(dir, -size), Mul(side, size * 0.5f))),
        Add(tip, Add(Mul(dir, -size), Mul(side, -size * 0.5f)))
    };
}

void DrawArrowHead(Vector2 tip, Vector2 direction, float size, Color color) {
    const auto tri = ArrowHead(tip, direction, size);
    DrawTriangle(tri[0], tri[1], tri[2], color);
}

void DrawWavyLine(Vector2 start, Vector2 end, float amplitude, int segments, float thickness, Color color) {
    const Vector2 delta = Sub(end, start);
    const Vector2 dir = NormalizeOrZero(delta);
    const Vector2 normal = Perp(dir);

    Vector2 prev = start;
    for (int i = 1; i <= segments; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(segments);
        const float wave = std::sin(t * 2.0f * PI * 5.0f) * amplitude;
        const Vector2 point = Add(Vector2Lerp(start, end, t), Mul(normal, wave));
        DrawLineEx(prev, point, thickness, color);
        prev = point;
    }
}

void DrawStraightLine(Vector2 start, Vector2 end, float thickness, Color color) {
    DrawLineEx(start, end, thickness, color);
}

void DrawParticleLine(const DiagramEdge& edge, const std::vector<Vector2>& nodes, bool highlight) {
    const Vector2 start = nodes[edge.from];
    const Vector2 end = nodes[edge.to];
    const float thickness = highlight ? 4.2f : 2.6f;
    Color drawColor = edge.color;
    if (!highlight) drawColor = WithAlpha(drawColor, 130);

    if (edge.style == ParticleStyle::kPhoton) {
        DrawWavyLine(start, end, highlight ? 11.0f : 8.0f, 56, thickness, drawColor);
    } else if (edge.style == ParticleStyle::kBoson) {
        DrawWavyLine(start, end, highlight ? 7.0f : 5.0f, 44, thickness, drawColor);
    } else {
        DrawStraightLine(start, end, thickness, drawColor);
    }

    const float arrowT = edge.forwardArrow ? 0.62f : 0.38f;
    const Vector2 tip = Vector2Lerp(start, end, arrowT);
    const Vector2 dir = edge.forwardArrow ? Sub(end, start) : Sub(start, end);
    DrawArrowHead(tip, dir, highlight ? 11.0f : 8.0f, drawColor);

    const Vector2 labelPos = Add(Vector2Lerp(start, end, 0.5f), edge.labelOffset);
    DrawText(edge.label,
             static_cast<int>(labelPos.x),
             static_cast<int>(labelPos.y),
             22,
             highlight ? WithAlpha(RAYWHITE, 240) : WithAlpha(drawColor, 220));
}

void DrawPacket(const DiagramEdge& edge, const std::vector<Vector2>& nodes, float progress) {
    const Vector2 start = nodes[edge.from];
    const Vector2 end = nodes[edge.to];
    const Vector2 pos = Vector2Lerp(start, end, EaseOutCubic(progress));
    const float radius = (edge.style == ParticleStyle::kFermion) ? 9.0f : 10.5f;

    DrawCircleV(pos, radius * 1.9f, WithAlpha(edge.color, 40));
    DrawCircleV(pos, radius, edge.color);
    DrawRing(pos, radius * 1.5f, radius * 1.95f, 0.0f, 360.0f, 36, WithAlpha(edge.color, 90));
}

void DrawVertexGlow(Vector2 center, float pulse, Color color) {
    const float radius = 18.0f + 24.0f * pulse;
    DrawCircleV(center, radius, WithAlpha(color, static_cast<unsigned char>(70.0f * pulse)));
    DrawCircleV(center, 7.0f + 4.0f * pulse, color);
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
        "e- + e+  ->  gamma*/Z  ->  mu- + mu+",
        "Annihilation into a virtual neutral boson. The simulator boosts the rate near the Z resonance and keeps a smaller baseline from photon exchange.",
        "virtual gamma / Z",
        20.0f,
        160.0f,
        91.2f,
        91.2f,
        10.0f,
        0.88f,
        RateModel::kResonantSChannel,
        {
            {180.0f, 290.0f},
            {180.0f, 570.0f},
            {520.0f, 430.0f},
            {860.0f, 430.0f},
            {1200.0f, 290.0f},
            {1200.0f, 570.0f},
        },
        {
            {0, 2, "e-", ParticleStyle::kFermion, Color{90, 215, 255, 255}, true, 0, {-18.0f, -36.0f}},
            {1, 2, "e+", ParticleStyle::kFermion, Color{255, 128, 156, 255}, false, 0, {-18.0f, 12.0f}},
            {2, 3, "gamma* / Z", ParticleStyle::kPhoton, Color{255, 219, 102, 255}, true, 1, {-40.0f, -38.0f}},
            {3, 4, "mu-", ParticleStyle::kFermion, Color{112, 245, 193, 255}, true, 2, {18.0f, -36.0f}},
            {3, 5, "mu+", ParticleStyle::kFermion, Color{210, 170, 255, 255}, false, 2, {18.0f, 12.0f}},
        },
        {2, 3},
        {
            {"charge", true},
            {"lepton family flow", true},
            {"4-momentum at vertices", true},
        },
    });

    processes.push_back({
        "Compton Scattering",
        "e- + gamma  ->  e- + gamma",
        "A QED scattering channel with a virtual electron between two vertices. The rate stays comparatively smooth as you move the energy slider.",
        "virtual electron",
        5.0f,
        90.0f,
        24.0f,
        0.0f,
        1.0f,
        0.82f,
        RateModel::kQEDScattering,
        {
            {180.0f, 540.0f},
            {180.0f, 260.0f},
            {520.0f, 430.0f},
            {860.0f, 430.0f},
            {1200.0f, 260.0f},
            {1200.0f, 540.0f},
        },
        {
            {0, 2, "e-", ParticleStyle::kFermion, Color{98, 224, 255, 255}, true, 0, {-18.0f, 12.0f}},
            {1, 2, "gamma", ParticleStyle::kPhoton, Color{255, 220, 108, 255}, true, 0, {-18.0f, -36.0f}},
            {2, 3, "e- (virtual)", ParticleStyle::kFermion, Color{190, 226, 255, 255}, true, 1, {-38.0f, -40.0f}},
            {3, 4, "gamma", ParticleStyle::kPhoton, Color{255, 220, 108, 255}, true, 2, {14.0f, -36.0f}},
            {3, 5, "e-", ParticleStyle::kFermion, Color{98, 224, 255, 255}, true, 2, {18.0f, 12.0f}},
        },
        {2, 3},
        {
            {"charge", true},
            {"fermion number", true},
            {"gauge vertex structure", true},
        },
    });

    processes.push_back({
        "Beta Decay Vertex",
        "d  ->  u + W-  and  W- -> e- + anti-nu_e",
        "A weak-interaction view of beta decay at the quark level. The heavy W propagator keeps the relative rate suppressed until the energy scale rises.",
        "virtual W-",
        1.0f,
        120.0f,
        8.0f,
        80.4f,
        12.0f,
        0.54f,
        RateModel::kWeakSuppressed,
        {
            {180.0f, 430.0f},
            {520.0f, 430.0f},
            {840.0f, 300.0f},
            {820.0f, 560.0f},
            {1180.0f, 250.0f},
            {1180.0f, 470.0f},
            {1180.0f, 650.0f},
        },
        {
            {0, 1, "d", ParticleStyle::kFermion, Color{107, 208, 255, 255}, true, 0, {-8.0f, -34.0f}},
            {1, 2, "u", ParticleStyle::kFermion, Color{112, 245, 193, 255}, true, 2, {-8.0f, -34.0f}},
            {1, 3, "W-", ParticleStyle::kBoson, Color{255, 177, 94, 255}, true, 1, {-50.0f, 6.0f}},
            {3, 5, "e-", ParticleStyle::kFermion, Color{130, 228, 255, 255}, true, 2, {18.0f, -18.0f}},
            {3, 6, "anti-nu_e", ParticleStyle::kFermion, Color{220, 196, 255, 255}, false, 2, {-60.0f, -16.0f}},
        },
        {1, 3},
        {
            {"charge", true},
            {"baryon number flow", true},
            {"lepton number", true},
        },
    });

    return processes;
}

void DrawBackgroundGrid() {
    const Color grid = Color{25, 36, 54, 255};
    for (int x = 40; x < kScreenWidth; x += 40) {
        DrawLine(x, 0, x, kScreenHeight, WithAlpha(grid, (x % 120 == 0) ? 95 : 52));
    }
    for (int y = 40; y < kScreenHeight; y += 40) {
        DrawLine(0, y, kScreenWidth, y, WithAlpha(grid, (y % 120 == 0) ? 95 : 52));
    }
}

void DrawRateBar(Rectangle bounds, float value, Color fillColor) {
    DrawRectangleRounded(bounds, 0.28f, 16, Color{17, 26, 38, 255});
    DrawRectangleRoundedLinesEx(bounds, 0.28f, 16, 2.0f, Color{58, 84, 116, 255});

    Rectangle fill = bounds;
    fill.width = std::max(0.0f, bounds.width * Saturate(value));
    DrawRectangleRounded(fill, 0.28f, 16, fillColor);
}

}  // namespace

int main() {
    InitWindow(kScreenWidth, kScreenHeight, "Feynman Diagram Simulator - C++ (raylib)");
    SetTargetFPS(60);

    std::vector<FeynmanProcess> processes = BuildProcesses();

    int selectedProcess = 0;
    float energyGeV = processes[selectedProcess].defaultEnergy;
    float couplingScale = 1.0f;
    float simTime = 0.0f;
    bool paused = false;
    bool autoReplay = true;
    bool showOverlay = true;

    auto resetForProcess = [&]() {
        const FeynmanProcess& process = processes[selectedProcess];
        energyGeV = std::clamp(energyGeV, process.energyMin, process.energyMax);
        simTime = 0.0f;
    };

    auto selectProcess = [&](int index) {
        selectedProcess = (index + static_cast<int>(processes.size())) % static_cast<int>(processes.size());
        energyGeV = processes[selectedProcess].defaultEnergy;
        simTime = 0.0f;
    };

    while (!WindowShouldClose()) {
        const FeynmanProcess& process = processes[selectedProcess];

        if (IsKeyPressed(KEY_ONE)) selectProcess(0);
        if (IsKeyPressed(KEY_TWO) && processes.size() > 1) selectProcess(1);
        if (IsKeyPressed(KEY_THREE) && processes.size() > 2) selectProcess(2);
        if (IsKeyPressed(KEY_TAB)) selectProcess(selectedProcess + 1);
        if (IsKeyPressed(KEY_SPACE)) simTime = 0.0f;
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_A)) autoReplay = !autoReplay;
        if (IsKeyPressed(KEY_H)) showOverlay = !showOverlay;
        if (IsKeyPressed(KEY_R)) {
            couplingScale = 1.0f;
            energyGeV = process.defaultEnergy;
            simTime = 0.0f;
            paused = false;
        }

        energyGeV += GetFrameTime() * 38.0f * (IsKeyDown(KEY_RIGHT) - IsKeyDown(KEY_LEFT));
        energyGeV = std::clamp(energyGeV, process.energyMin, process.energyMax);

        couplingScale += GetFrameTime() * 0.75f * (IsKeyDown(KEY_UP) - IsKeyDown(KEY_DOWN));
        couplingScale = std::clamp(couplingScale, 0.35f, 1.8f);

        if (!paused) {
            simTime += GetFrameTime();
            if (simTime > kSimDuration) {
                simTime = autoReplay ? 0.0f : kSimDuration;
            }
        }

        const float eventRate = RelativeRate(process, energyGeV, couplingScale);
        const Stage stage = CurrentStage(simTime);

        BeginDrawing();
        ClearBackground(Color{7, 10, 17, 255});
        DrawBackgroundGrid();

        DrawRectangleGradientV(0, 0, kScreenWidth, 200, Color{9, 16, 28, 180}, Color{9, 16, 28, 20});
        DrawRectangleGradientV(0, kScreenHeight - 220, kScreenWidth, 220, Color{7, 12, 22, 30}, Color{7, 12, 22, 200});

        Rectangle diagramBox{70.0f, 140.0f, 1240.0f, 560.0f};
        DrawRectangleRounded(diagramBox, 0.03f, 24, Color{10, 18, 31, 235});
        DrawRectangleRoundedLinesEx(diagramBox, 0.03f, 24, 2.0f, Color{55, 86, 126, 255});

        for (const DiagramEdge& edge : process.edges) {
            const bool active = PhaseActive(edge.phase, simTime);
            DrawParticleLine(edge, process.nodes, active);

            if (active) {
                DrawPacket(edge, process.nodes, PhaseProgress(edge.phase, simTime));
            }
        }

        for (std::size_t i = 0; i < process.vertexNodeIndices.size(); ++i) {
            const Vector2 center = process.nodes[process.vertexNodeIndices[i]];
            float pulse = 0.0f;
            if (i == 0) pulse = 1.0f - std::fabs(simTime - 1.35f) / 0.28f;
            if (i == 1) pulse = 1.0f - std::fabs(simTime - 2.35f) / 0.28f;
            pulse = Saturate(pulse);
            DrawVertexGlow(center, pulse, Color{255, 239, 160, 255});
        }

        DrawText(process.name, 76, 40, 34, Color{236, 241, 250, 255});
        DrawText(process.reaction, 76, 84, 24, Color{121, 219, 255, 255});
        DrawText(process.description, 76, 714, 20, Color{176, 193, 220, 255});

        DrawText("Conceptual Feynman simulator", 1040, 40, 24, Color{255, 226, 146, 255});
        DrawText("Not a full QFT calculator", 1065, 74, 20, Color{176, 193, 220, 255});

        Rectangle rateBar{78.0f, 762.0f, 430.0f, 28.0f};
        DrawText("relative event rate", 78, 736, 20, Color{210, 221, 236, 255});
        DrawRateBar(rateBar, eventRate, Color{104, 224, 174, 255});

        std::string rateText = "rate=" + FormatFloat(eventRate) + "   mediator=" + std::string(process.mediator);
        DrawText(rateText.c_str(), 78, 798, 20, Color{136, 232, 194, 255});

        Rectangle energyBar{566.0f, 762.0f, 260.0f, 24.0f};
        DrawText("center-of-mass energy", 566, 736, 20, Color{210, 221, 236, 255});
        DrawRateBar(energyBar, (energyGeV - process.energyMin) / (process.energyMax - process.energyMin), Color{91, 180, 255, 255});
        const std::string energyText = FormatFloat(energyGeV, 1) + " GeV";
        DrawText(energyText.c_str(), 842, 760, 21, Color{121, 219, 255, 255});

        Rectangle couplingBar{948.0f, 762.0f, 220.0f, 24.0f};
        DrawText("coupling scale", 948, 736, 20, Color{210, 221, 236, 255});
        DrawRateBar(couplingBar, (couplingScale - 0.35f) / 1.45f, Color{255, 178, 90, 255});
        DrawText(FormatFloat(couplingScale).c_str(), 1184, 760, 21, Color{255, 199, 115, 255});

        DrawText("conservation checks", 1038, 170, 22, Color{230, 236, 245, 255});
        for (std::size_t i = 0; i < process.checks.size(); ++i) {
            const int y = 206 + static_cast<int>(i) * 38;
            const Color chip = process.checks[i].valid ? Color{84, 197, 128, 255} : Color{210, 96, 96, 255};
            DrawRectangleRounded({1038.0f, static_cast<float>(y), 230.0f, 28.0f}, 0.25f, 14, WithAlpha(chip, 38));
            DrawCircle(1054, y + 14, 7, chip);
            DrawText(process.checks[i].label, 1070, y + 5, 20, Color{215, 224, 238, 255});
        }

        std::string stageText = "stage: " + std::string(StageLabel(stage));
        DrawText(stageText.c_str(), 1038, 344, 22, Color{255, 228, 140, 255});
        DrawText(TextFormat("diagram clock: %.2f s / %.1f s", simTime, kSimDuration),
                 1038,
                 378,
                 20,
                 Color{184, 197, 220, 255});

        if (showOverlay) {
            Rectangle help{1018.0f, 432.0f, 270.0f, 206.0f};
            DrawRectangleRounded(help, 0.06f, 20, Color{12, 20, 34, 220});
            DrawRectangleRoundedLinesEx(help, 0.06f, 20, 2.0f, Color{53, 83, 122, 255});
            DrawText("controls", 1042, 452, 22, Color{233, 239, 248, 255});
            DrawText("1/2/3 or TAB  change process", 1042, 486, 19, Color{178, 193, 218, 255});
            DrawText("LEFT/RIGHT    energy", 1042, 514, 19, Color{178, 193, 218, 255});
            DrawText("UP/DOWN       coupling", 1042, 542, 19, Color{178, 193, 218, 255});
            DrawText("SPACE         replay event", 1042, 570, 19, Color{178, 193, 218, 255});
            DrawText("A / P / H     auto pause help", 1042, 598, 19, Color{178, 193, 218, 255});
            DrawText("R             reset sliders", 1042, 626, 19, Color{178, 193, 218, 255});
        } else {
            DrawText("H to show controls", 1040, 452, 20, Color{178, 193, 218, 255});
        }

        DrawText("Legend: straight lines are fermions, wavy lines are gauge bosons, glowing dots mark packets moving through each leg.", 76, 118, 20, Color{173, 190, 214, 255});
        DrawFPS(1240, 22);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
