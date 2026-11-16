#include "collider/model.h"
#include "collider/renderer.h"

#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <cmath>

namespace {
constexpr Color kInk{225, 235, 244, 255};
constexpr Color kMuted{126, 151, 174, 255};
constexpr float kLetterSpacing = .18f;

float uiWidth(Font font, const char* text, float size) {
    return MeasureTextEx(font, text, size, kLetterSpacing).x;
}

void uiText(Font font, const char* text, float x, float y, float size, Color color) {
    DrawTextEx(font, text, {std::round(x), std::round(y)}, size, kLetterSpacing, color);
}

float smooth(float value) {
    value = std::clamp(value, 0.0f, 1.0f);
    return value * value * (3.0f - 2.0f * value);
}

const char* phaseName(float time) {
    if (time < 2.4f) return "BEAMS APPROACHING";
    if (time < 2.75f) return "PROTON COLLISION";
    if (time < 5.0f) return "PARTICLES CROSS THE DETECTOR";
    if (time < 10.6f) return "EVENT RECONSTRUCTION";
    return "NEXT COLLISION";
}

Camera3D cinematicCamera(float time) {
    Camera3D camera{};
    camera.target = {0, .2f, 0};
    camera.up = {0, 1, 0};
    camera.fovy = 39;
    camera.projection = CAMERA_PERSPECTIVE;
    float intro = smooth(time / 2.5f);
    // Mostly side-on so the beam line reads immediately; a slow drift still
    // reveals the detector's depth without collapsing it into an end-on view.
    float orbit = 1.25f + time * .035f;
    float distance = 34.0f - 4.5f * intro + 1.4f * std::sin(time * .21f);
    float pitch = .24f + .10f * std::sin(time * .17f);
    camera.position = {distance * std::cos(pitch) * std::cos(orbit), distance * std::sin(pitch),
                       distance * std::cos(pitch) * std::sin(orbit)};
    return camera;
}

void drawPill(Font font, Rectangle bounds, const char* text, bool active, Color accent) {
    DrawRectangleRounded(bounds, .45f, 10, active ? Fade(accent, .18f) : Color{17, 29, 43, 225});
    DrawRectangleRoundedLinesEx(bounds, .45f, 10, 1, active ? Fade(accent, .60f) : Color{47, 66, 83, 180});
    float width = uiWidth(font, text, 13);
    uiText(font, text, bounds.x + (bounds.width - width) * .5f, bounds.y + 8, 13, active ? accent : kMuted);
}
}  // namespace

int main() {
    bool smoke = std::getenv("COMPPHYSICS_SMOKE_FRAMES") != nullptr;
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE | FLAG_WINDOW_HIGHDPI |
                   (smoke ? FLAG_WINDOW_HIDDEN : 0));
    InitWindow(1440, 900, "Collider Event | Particle Accelerator");
    SetWindowMinSize(1000, 650);
    SetTargetFPS(120);

    auto fontPath = project_paths::root() / "resources/fonts/Inter-Regular.ttf";
    Font uiFont = LoadFontEx(fontPath.string().c_str(), 56, nullptr, 0);
    bool ownsUiFont = uiFont.texture.id != 0;
    if (ownsUiFont) {
        SetTextureFilter(uiFont.texture, TEXTURE_FILTER_BILINEAR);
    } else {
        uiFont = studio::font();
    }

    collider::Renderer renderer;
    if (!renderer.ready()) {
        TraceLog(LOG_ERROR, "Collider lighting shader failed to initialize");
        CloseWindow();
        return 1;
    }

    const char* postFragment = R"GLSL(#version 330
in vec2 fragTexCoord;
in vec4 fragColor;
uniform sampler2D texture0;
uniform vec2 resolution;
uniform float bloom;
out vec4 finalColor;
void main() {
    vec2 uv = fragTexCoord;
    vec2 px = 1.0 / resolution;
    vec3 base = texture(texture0, uv).rgb;
    vec3 glow = vec3(0.0);
    vec2 offsets[8] = vec2[](vec2(2,0),vec2(-2,0),vec2(0,2),vec2(0,-2),
                              vec2(1.4,1.4),vec2(-1.4,1.4),vec2(1.4,-1.4),vec2(-1.4,-1.4));
    for (int i=0;i<8;i++) {
        vec3 sampleColor = texture(texture0, uv + offsets[i]*px).rgb;
        glow += max(sampleColor - vec3(.48), vec3(0.0));
    }
    base += glow * (.085 * bloom);
    base = base / (base + vec3(.72));
    base = pow(base, vec3(.92));
    vec2 q = uv * 2.0 - 1.0;
    float vignette = 1.0 - .30 * smoothstep(.38, 1.35, dot(q,q));
    float grain = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898,78.233))) * 43758.5453) - .5;
    finalColor = vec4(base * vignette + grain/255.0, 1.0);
})GLSL";
    Shader post = LoadShaderFromMemory(nullptr, postFragment);
    int resolutionLoc = GetShaderLocation(post, "resolution");
    int bloomLoc = GetShaderLocation(post, "bloom");

    constexpr float renderScale = .88f;
    int targetWidth = static_cast<int>(GetScreenWidth() * renderScale);
    int targetHeight = static_cast<int>(GetScreenHeight() * renderScale);
    RenderTexture2D target = LoadRenderTexture(targetWidth, targetHeight);
    SetTextureFilter(target.texture, TEXTURE_FILTER_BILINEAR);

    collider::Timeline timeline;
    if (const char* initial = std::getenv("COMPPHYSICS_COLLIDER_TIME")) {
        timeline.time = std::clamp(static_cast<float>(std::atof(initial)), 0.0f, collider::Timeline::duration);
    }
    constexpr float energyOptions[] = {3.5f, 7.0f, 13.6f};
    int energyIndex = 2;
    bool forceHiggs = std::getenv("COMPPHYSICS_FORCE_HIGGS") != nullptr;
    collider::Event event = collider::generateEvent(1, energyOptions[energyIndex], forceHiggs);
    std::uint32_t eventNumber = 1;
    float fieldTesla = 3.8f;
    bool cinematic = true, detectorVisible = true, showTracks = true;
    bool waveView = std::getenv("COMPPHYSICS_WAVE_VIEW") != nullptr;
    bool showHud = false;
    float yaw = .62f, pitch = .30f, distance = 25.0f;
    Camera3D manualCamera{{20, 8, 18}, {0, .2f, 0}, {0, 1, 0}, 39, CAMERA_PERSPECTIVE};
    float previousTimeline = timeline.time, averageMs = 8.0f;

    while (!WindowShouldClose()) {
        float dt = std::min(GetFrameTime(), .05f);
        int width = GetScreenWidth(), height = GetScreenHeight();
        int desiredWidth = static_cast<int>(width * renderScale);
        int desiredHeight = static_cast<int>(height * renderScale);
        if (desiredWidth != targetWidth || desiredHeight != targetHeight) {
            UnloadRenderTexture(target);
            targetWidth = desiredWidth; targetHeight = desiredHeight;
            target = LoadRenderTexture(targetWidth, targetHeight);
            SetTextureFilter(target.texture, TEXTURE_FILTER_BILINEAR);
        }

        if (IsKeyPressed(KEY_SPACE) || IsKeyPressed(KEY_P)) timeline.paused = !timeline.paused;
        if (IsKeyPressed(KEY_R)) timeline.restart();
        if (IsKeyPressed(KEY_N)) { event = collider::generateEvent(++eventNumber, energyOptions[energyIndex]); timeline.restart(); }
        if (IsKeyPressed(KEY_E)) {
            energyIndex = (energyIndex + 1) % 3;
            event = collider::generateEvent(++eventNumber, energyOptions[energyIndex]);
            timeline.restart();
        }
        if (IsKeyPressed(KEY_C)) cinematic = !cinematic;
        if (IsKeyPressed(KEY_D)) detectorVisible = !detectorVisible;
        if (IsKeyPressed(KEY_T)) showTracks = !showTracks;
        if (IsKeyPressed(KEY_W)) waveView = !waveView;
        if (IsKeyPressed(KEY_H)) showHud = !showHud;
        if (IsKeyPressed(KEY_G)) {
            event = collider::generateEvent(++eventNumber, energyOptions[energyIndex], true);
            timeline.restart();
        }
        if (IsKeyPressed(KEY_ONE)) timeline.speed = .35f;
        if (IsKeyPressed(KEY_TWO)) timeline.speed = 1.0f;
        if (IsKeyPressed(KEY_THREE)) timeline.speed = 2.0f;
        if (IsKeyDown(KEY_LEFT)) fieldTesla = std::max(.5f, fieldTesla - dt);
        if (IsKeyDown(KEY_RIGHT)) fieldTesla = std::min(6.0f, fieldTesla + dt);

        timeline.update(dt);
        if (timeline.time < previousTimeline) event = collider::generateEvent(++eventNumber, energyOptions[energyIndex]);
        previousTimeline = timeline.time;
        averageMs += (dt * 1000.0f - averageMs) * .04f;

        if (!cinematic) {
            if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
                Vector2 delta = GetMouseDelta(); yaw -= delta.x * .004f;
                pitch = std::clamp(pitch + delta.y * .004f, -.95f, 1.15f);
            }
            distance = std::clamp(distance * std::exp(-GetMouseWheelMove() * .08f), 13.0f, 46.0f);
            manualCamera.position = {distance * std::cos(pitch) * std::cos(yaw), distance * std::sin(pitch),
                                     distance * std::cos(pitch) * std::sin(yaw)};
        }
        Camera3D camera = cinematic ? cinematicCamera(timeline.time) : manualCamera;

        BeginTextureMode(target);
        ClearBackground(collider::kBackground);
        BeginMode3D(camera);
        renderer.drawScene(camera, event, timeline, fieldTesla, detectorVisible, showTracks, waveView);
        EndMode3D();
        EndTextureMode();

        float resolution[2] = {static_cast<float>(targetWidth), static_cast<float>(targetHeight)};
        float bloom = 1.0f;
        SetShaderValue(post, resolutionLoc, resolution, SHADER_UNIFORM_VEC2);
        SetShaderValue(post, bloomLoc, &bloom, SHADER_UNIFORM_FLOAT);

        BeginDrawing();
        ClearBackground(BLACK);
        BeginShaderMode(post);
        DrawTexturePro(target.texture,
                       {0, 0, static_cast<float>(target.texture.width), -static_cast<float>(target.texture.height)},
                       {0, 0, static_cast<float>(width), static_cast<float>(height)}, {0, 0}, 0, WHITE);
        EndShaderMode();

        if (event.higgsCandidate && timeline.reveal() > .5f) {
            const char* channel = event.higgsChannel == collider::HiggsChannel::GammaGamma ? "H -> gamma gamma" : "H -> ZZ* -> 4 leptons";
            float cardY = showHud ? 116.0f : 35.0f;
            Rectangle card{35, cardY, 252, 72};
            DrawRectangleRounded(card, .16f, 8, Color{13, 22, 36, 226});
            DrawRectangleRoundedLinesEx(card, .16f, 8, 1, Fade(collider::kMagenta, .65f));
            uiText(uiFont, "HIGGS CANDIDATE", 49, cardY + 12, 13, collider::kMagenta);
            uiText(uiFont, channel, 49, cardY + 33, 15, kInk);
            uiText(uiFont, TextFormat("reconstructed mass  %.1f GeV", event.higgsMassGeV), 49, cardY + 53, 12, kMuted);
        }

        if (showHud) {
        DrawRectangleGradientV(0, 0, width, 126, Color{4, 9, 16, 245}, Color{4, 9, 16, 25});
        DrawRectangleGradientV(0, height - 125, width, 125, Color{4, 9, 16, 0}, Color{4, 9, 16, 245});
        uiText(uiFont, "COLLIDER / EVENT DISPLAY", 34, 22, 24, kInk);
        uiText(uiFont, TextFormat("EVENT %06u", event.number), 35, 55, 14, kMuted);

        float controlsStart = static_cast<float>(width - 566);
        float timelineX = 330.0f;
        float timelineWidth = std::max(140.0f, controlsStart - timelineX - 24.0f);
        DrawLineEx({timelineX, 49}, {timelineX + timelineWidth, 49}, 2, Color{44, 65, 83, 220});
        float progress = timeline.time / collider::Timeline::duration;
        DrawLineEx({timelineX, 49}, {timelineX + timelineWidth * progress, 49}, 2, collider::kCyan);
        constexpr float marks[] = {0, 2.55f, 5.0f, 10.65f, collider::Timeline::duration};
        constexpr const char* labels[] = {"INJECT", "COLLIDE", "TRACE", "HOLD", "RESET"};
        for (int i = 0; i < 5; ++i) {
            float x = timelineX + timelineWidth * marks[i] / collider::Timeline::duration;
            DrawCircleV({x, 49}, i == 1 && timeline.time > 2.45f && timeline.time < 2.9f ? 5 : 3,
                        marks[i] <= timeline.time ? collider::kCyan : kMuted);
            uiText(uiFont, labels[i], x - uiWidth(uiFont, labels[i], 11) * .5f, 64, 11, kMuted);
        }

        Rectangle viewButton{static_cast<float>(width - 566), 24, 142, 30};
        Rectangle energyButton{static_cast<float>(width - 414), 24, 96, 30};
        Rectangle cinemaButton{static_cast<float>(width - 308), 24, 168, 30};
        Rectangle nextButton{static_cast<float>(width - 130), 24, 96, 30};
        if (studio::clicked(viewButton)) waveView = !waveView;
        if (studio::clicked(energyButton)) {
            energyIndex = (energyIndex + 1) % 3;
            event = collider::generateEvent(++eventNumber, energyOptions[energyIndex]);
            timeline.restart();
        }
        if (studio::clicked(cinemaButton)) cinematic = !cinematic;
        if (studio::clicked(nextButton)) { event = collider::generateEvent(++eventNumber, energyOptions[energyIndex]); timeline.restart(); }
        drawPill(uiFont, viewButton, waveView ? "BEAMS: WAVES" : "BEAMS: PARTICLES", waveView, collider::kCyan);
        drawPill(uiFont, energyButton, TextFormat("%.1f TeV", energyOptions[energyIndex]), false, collider::kAmber);
        drawPill(uiFont, cinemaButton, cinematic ? "CAMERA: CINEMATIC" : "CAMERA: FREE", cinematic, collider::kViolet);
        drawPill(uiFont, nextButton, "NEW EVENT", false, collider::kCyan);

        uiText(uiFont, phaseName(timeline.time), 34, height - 94, 27, kInk);
        uiText(uiFont, TextFormat("%.0f GeV visible energy", event.totalEnergy), 35, height - 59, 14, collider::kAmber);
        uiText(uiFont, TextFormat("B %.1f T   /   %zu reconstructed tracks   /   %.1f ms", fieldTesla,
                                  event.particles.size(), averageMs), 35, height - 33, 13, kMuted);
        if (timeline.reveal() > .5f && width > 1120) {
            const struct { const char* label; Color color; } legend[] = {
                {"ELECTRON", collider::kCyan}, {"MUON", collider::kViolet},
                {"PHOTON", collider::kAmber}, {"HADRON", Color{105,239,173,255}}
            };
            float lx = width - 420.0f;
            for (const auto& item : legend) {
                DrawCircleV({lx, static_cast<float>(height - 72)}, 3.5f, item.color);
                uiText(uiFont, item.label, lx + 9, height - 79, 11, kMuted);
                lx += uiWidth(uiFont, item.label, 11) + 35;
            }
        }
        const char* help = cinematic ? "Space pause   N new collision   W beam model   C camera   E energy" :
                                       "Drag orbit   Scroll zoom   W beam model   C cinematic   E energy";
        uiText(uiFont, help, width - uiWidth(uiFont, help, 13) - 35, height - 33, 13, kMuted);
        if (timeline.paused) uiText(uiFont, "PAUSED", width - 105, height - 65, 15, collider::kMagenta);
        }
        EndDrawing();
        if (studio::smokeFrame(__FILE__)) break;
    }

    UnloadRenderTexture(target);
    UnloadShader(post);
    if (ownsUiFont) UnloadFont(uiFont);
    studio::unload();
    CloseWindow();
    return 0;
}
