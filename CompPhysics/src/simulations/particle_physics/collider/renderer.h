#pragma once

#include "model.h"
#include "../../../common/studio.h"
#include "rlgl.h"
#include "raymath.h"

#include <array>
#include <cmath>
#include <vector>

namespace collider {

constexpr Color kBackground{5, 10, 18, 255};
constexpr Color kCyan{62, 218, 255, 255};
constexpr Color kAmber{255, 181, 69, 255};
constexpr Color kMagenta{255, 79, 187, 255};
constexpr Color kViolet{151, 112, 255, 255};

inline Vector3 render(Vec3 value) { return {value.x, value.y, value.z}; }
inline Color render(Rgba value) { return {value.r, value.g, value.b, value.a}; }

inline Color mix(Color a, Color b, float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return {
        static_cast<unsigned char>(a.r + (b.r - a.r) * t),
        static_cast<unsigned char>(a.g + (b.g - a.g) * t),
        static_cast<unsigned char>(a.b + (b.b - a.b) * t),
        static_cast<unsigned char>(a.a + (b.a - a.a) * t),
    };
}

inline Matrix trs(Vector3 p, Vector3 s, float rotationX = 0.0f) {
    return MatrixMultiply(MatrixMultiply(MatrixScale(s.x, s.y, s.z), MatrixRotateX(rotationX)),
                          MatrixTranslate(p.x, p.y, p.z));
}

inline Mesh makeCutawayShell() {
    constexpr int segments = 72;
    constexpr int verticesPerSegment = 6;
    Mesh mesh{};
    mesh.vertexCount = segments * verticesPerSegment;
    mesh.triangleCount = mesh.vertexCount / 3;
    mesh.vertices = static_cast<float*>(MemAlloc(mesh.vertexCount * 3 * sizeof(float)));
    mesh.normals = static_cast<float*>(MemAlloc(mesh.vertexCount * 3 * sizeof(float)));
    int cursor = 0;
    auto emit = [&](float x, float angle) {
        mesh.vertices[cursor] = x;
        mesh.normals[cursor++] = 0;
        mesh.vertices[cursor] = std::cos(angle);
        mesh.normals[cursor++] = std::cos(angle);
        mesh.vertices[cursor] = std::sin(angle);
        mesh.normals[cursor++] = std::sin(angle);
    };
    const float start = 1.50f;
    const float span = 2.0f * PI - 1.47f;
    for (int i = 0; i < segments; ++i) {
        float a = start + span * i / segments;
        float b = start + span * (i + 1) / segments;
        emit(-1, a); emit(1, a); emit(1, b);
        emit(-1, a); emit(1, b); emit(-1, b);
    }
    UploadMesh(&mesh, false);
    return mesh;
}

inline Texture2D makeGlowTexture() {
    constexpr int side = 64;
    Image image = GenImageColor(side, side, BLANK);
    auto* pixels = static_cast<Color*>(image.data);
    for (int y = 0; y < side; ++y) {
        for (int x = 0; x < side; ++x) {
            float dx = (x + .5f - side * .5f) / (side * .5f);
            float dy = (y + .5f - side * .5f) / (side * .5f);
            float r2 = dx * dx + dy * dy;
            float alpha = std::pow(std::max(0.0f, 1.0f - r2), 3.5f);
            pixels[y * side + x] = {255, 255, 255, static_cast<unsigned char>(alpha * 255)};
        }
    }
    Texture2D texture = LoadTextureFromImage(image);
    UnloadImage(image);
    SetTextureFilter(texture, TEXTURE_FILTER_BILINEAR);
    return texture;
}

struct LayerBatch {
    std::vector<Matrix> matrices;
    Color color{};
};

class Renderer {
  public:
    Renderer() {
        cube_ = GenMeshCube(1, 1, 1);
        cylinder_ = GenMeshCylinder(1, 1, 32);
        sphere_ = GenMeshSphere(1, 18, 24);
        torus_ = GenMeshTorus(.12f, 1.0f, 20, 64);
        shell_ = makeCutawayShell();
        glow_ = makeGlowTexture();

        const char* instancedVs = R"GLSL(#version 330
in vec3 vertexPosition;
in vec3 vertexNormal;
in mat4 instanceTransform;
uniform mat4 mvp;
uniform mat4 matModel;
uniform vec3 viewPos;
out vec3 worldNormal;
out vec3 worldPosition;
void main() {
    mat4 model = matModel * instanceTransform;
    vec4 world = model * vec4(vertexPosition, 1.0);
    worldPosition = world.xyz;
    worldNormal = normalize(mat3(model) * vertexNormal);
    gl_Position = mvp * instanceTransform * vec4(vertexPosition, 1.0);
})GLSL";
        const char* litFs = R"GLSL(#version 330
in vec3 worldNormal;
in vec3 worldPosition;
uniform vec3 viewPos;
uniform vec4 colDiffuse;
out vec4 finalColor;
void main() {
    vec3 n = normalize(worldNormal);
    vec3 light = normalize(vec3(-0.35, 0.82, 0.46));
    vec3 view = normalize(viewPos - worldPosition);
    float diffuse = max(dot(n, light), 0.0);
    float rim = pow(1.0 - max(dot(n, view), 0.0), 2.4);
    float spec = pow(max(dot(reflect(-light, n), view), 0.0), 48.0);
    vec3 base = colDiffuse.rgb * (0.27 + 0.73 * diffuse);
    base += colDiffuse.rgb * rim * 0.24 + vec3(spec * 0.32);
    finalColor = vec4(base, colDiffuse.a);
})GLSL";
        shader_ = LoadShaderFromMemory(instancedVs, litFs);
        ready_ = shader_.id != 0 && shader_.id != rlGetShaderIdDefault();
        material_ = LoadMaterialDefault();
        shellMaterial_ = LoadMaterialDefault();
        if (ready_) {
            material_.shader = shader_;
            shader_.locs[SHADER_LOC_MATRIX_MVP] = GetShaderLocation(shader_, "mvp");
            shader_.locs[SHADER_LOC_MATRIX_MODEL] = GetShaderLocation(shader_, "matModel");
            shader_.locs[SHADER_LOC_VECTOR_VIEW] = GetShaderLocation(shader_, "viewPos");
            shader_.locs[SHADER_LOC_COLOR_DIFFUSE] = GetShaderLocation(shader_, "colDiffuse");
        }
        buildDetector();
        buildStars();
    }

    ~Renderer() {
        UnloadTexture(glow_);
        UnloadMesh(cube_);
        UnloadMesh(cylinder_);
        UnloadMesh(sphere_);
        UnloadMesh(torus_);
        UnloadMesh(shell_);
        if (ready_) {
            material_.shader = {};
            UnloadShader(shader_);
        }
        UnloadMaterial(material_);
        UnloadMaterial(shellMaterial_);
    }

    bool ready() const { return ready_; }

    void drawScene(const Camera3D& camera, const Event& event, const Timeline& timeline,
                   float fieldTesla, bool detectorVisible, bool showTracks) {
        if (!ready_) return;
        SetShaderValue(shader_, shader_.locs[SHADER_LOC_VECTOR_VIEW], &camera.position.x, SHADER_UNIFORM_VEC3);
        drawEnvironment(camera);
        if (detectorVisible) drawDetector(camera, timeline.reveal());
        drawBeams(camera, timeline);
        if (showTracks) drawEvent(camera, event, timeline, fieldTesla);
    }

  private:
    Mesh cube_{}, cylinder_{}, sphere_{}, torus_{}, shell_{};
    Material material_{};
    Material shellMaterial_{};
    Shader shader_{};
    Texture2D glow_{};
    bool ready_ = false;
    std::array<LayerBatch, 4> barrel_{};
    std::array<LayerBatch, 3> endcaps_{};
    std::vector<Vector3> stars_;

    void addBarrel(LayerBatch& batch, float radius, float halfLength, int slices, int around,
                   Vector3 moduleScale, float stagger = 0.0f) {
        batch.matrices.reserve(batch.matrices.size() + static_cast<size_t>(slices * around));
        for (int x = 0; x < slices; ++x) {
            float px = -halfLength + (2.0f * halfLength) * (x + .5f) / slices;
            for (int a = 0; a < around; ++a) {
                float angle = 2.0f * PI * (a + (x & 1 ? stagger : 0.0f)) / around;
                // Remove the camera-facing quadrant. A real event display is far
                // more readable as a cutaway than through an opaque full barrel.
                if (angle > .03f && angle < 1.48f) continue;
                Vector3 p{px, radius * std::cos(angle), radius * std::sin(angle)};
                batch.matrices.push_back(trs(p, moduleScale, angle));
            }
        }
    }

    void addEndcap(LayerBatch& batch, float x, float inner, float outer, int rings, int around) {
        for (int r = 0; r < rings; ++r) {
            float radius = inner + (outer - inner) * (r + .5f) / rings;
            int count = std::max(16, static_cast<int>(around * radius / outer));
            for (int a = 0; a < count; ++a) {
                float angle = 2.0f * PI * a / count;
                if (x > 0 && angle > -.02f && angle < 1.52f) continue;
                Vector3 p{x, radius * std::cos(angle), radius * std::sin(angle)};
                batch.matrices.push_back(trs(p, {.18f, (outer - inner) * .39f / rings, radius * PI * .82f / count}, angle));
            }
        }
    }

    void buildDetector() {
        barrel_[0].color = {47, 194, 231, 255};
        barrel_[1].color = {230, 170, 54, 255};
        barrel_[2].color = {84, 178, 190, 255};
        barrel_[3].color = {120, 100, 196, 255};
        addBarrel(barrel_[0], 1.05f, 5.7f, 22, 44, {.38f, .08f, .025f}, .5f);
        addBarrel(barrel_[0], 1.55f, 6.2f, 18, 48, {.48f, .10f, .035f});
        addBarrel(barrel_[1], 3.05f, 7.1f, 14, 38, {.72f, .28f, .12f}, .5f);
        addBarrel(barrel_[2], 4.55f, 7.7f, 12, 34, {.84f, .36f, .17f});
        addBarrel(barrel_[3], 7.25f, 8.7f, 8, 28, {1.08f, .46f, .10f}, .5f);

        endcaps_[0].color = {51, 188, 224, 255};
        endcaps_[1].color = {226, 157, 54, 255};
        endcaps_[2].color = {111, 94, 185, 255};
        for (float side : {-1.0f, 1.0f}) {
            addEndcap(endcaps_[0], side * 6.35f, .55f, 2.1f, 5, 64);
            addEndcap(endcaps_[1], side * 7.35f, 1.0f, 5.35f, 6, 58);
            addEndcap(endcaps_[2], side * 8.85f, 1.4f, 8.0f, 6, 52);
        }
    }

    void buildStars() {
        std::uint32_t rng = 0x1234abcdu;
        stars_.reserve(520);
        for (int i = 0; i < 520; ++i) {
            float y = random01(rng) * 2.0f - 1.0f;
            float angle = random01(rng) * 2.0f * PI;
            float radius = 52.0f + random01(rng) * 24.0f;
            stars_.push_back({radius * std::sqrt(1 - y * y) * std::cos(angle), radius * y,
                              radius * std::sqrt(1 - y * y) * std::sin(angle)});
        }
    }

    void batch(const LayerBatch& layer, float alpha = 1.0f) {
        if (layer.matrices.empty()) return;
        material_.maps[MATERIAL_MAP_DIFFUSE].color = Fade(layer.color, alpha);
        DrawMeshInstanced(cube_, material_, layer.matrices.data(), static_cast<int>(layer.matrices.size()));
    }

    void drawEnvironment(const Camera3D& camera) {
        (void)camera;
        BeginBlendMode(BLEND_ADDITIVE);
        rlDisableDepthMask();
        for (size_t i = 0; i < stars_.size(); ++i) {
            float alpha = .12f + .32f * ((i * 17) % 13) / 12.0f;
            DrawPoint3D(stars_[i], Fade(Color{142, 186, 224, 255}, alpha));
        }
        rlEnableDepthMask();
        EndBlendMode();
    }

    void drawDetector(const Camera3D& camera, float reveal) {
        (void)camera;
        // Thin continuous shells give the segmented sensors a coherent form.
        // They are deliberately subtle and do not write depth, so tracks remain
        // readable through the open camera-facing quadrant.
        BeginBlendMode(BLEND_ALPHA);
        rlDisableDepthMask();
        struct Shell { float radius, halfLength; Color color; };
        const Shell shells[] = {
            {1.56f, 6.25f, Color{42, 178, 216, 34}},
            {3.12f, 7.18f, Color{238, 172, 54, 32}},
            {4.63f, 7.78f, Color{76, 184, 194, 27}},
            {7.34f, 8.78f, Color{132, 104, 218, 22}},
        };
        for (const auto& shell : shells) {
            shellMaterial_.maps[MATERIAL_MAP_DIFFUSE].color = shell.color;
            DrawMesh(shell_, shellMaterial_, MatrixScale(shell.halfLength, shell.radius, shell.radius));
        }
        rlEnableDepthMask();
        EndBlendMode();

        // Fade the outer systems in after the collision so tracks remain readable.
        batch(barrel_[3], .34f + .42f * reveal);
        batch(barrel_[2], .48f + .42f * reveal);
        batch(barrel_[1], .58f + .40f * reveal);
        batch(barrel_[0], .74f + .26f * reveal);
        for (const auto& cap : endcaps_) batch(cap, .48f + .42f * reveal);

        // Beam pipe and structural rings use cached meshes and only a few draw calls.
        Material pipe = material_;
        pipe.maps[MATERIAL_MAP_DIFFUSE].color = {90, 131, 153, 255};
        Matrix axis = MatrixMultiply(MatrixRotateZ(PI * .5f), MatrixScale(.20f, 20.0f, .20f));
        DrawMesh(cylinder_, pipe, axis);
        pipe.maps[MATERIAL_MAP_DIFFUSE].color = {50, 72, 90, 255};
        for (float x : {-8.6f, -6.8f, 6.8f, 8.6f}) {
            Matrix ring = MatrixMultiply(MatrixRotateZ(PI * .5f), MatrixMultiply(MatrixScale(7.9f, .28f, 7.9f), MatrixTranslate(x, 0, 0)));
            DrawMesh(torus_, pipe, ring);
        }
    }

    void drawBeams(const Camera3D& camera, const Timeline& timeline) {
        float progress = timeline.beam();
        if (timeline.time > 2.82f) return;
        Vector3 right = Vector3Normalize(Vector3CrossProduct(Vector3Subtract(camera.target, camera.position), camera.up));
        Vector3 up = Vector3Normalize(Vector3CrossProduct(right, Vector3Subtract(camera.target, camera.position)));
        BeginBlendMode(BLEND_ADDITIVE);
        rlDisableDepthMask();
        for (int side : {-1, 1}) {
            float head = side * (11.8f * (1.0f - progress));
            Color color = side < 0 ? kCyan : kMagenta;
            for (int i = 0; i < 70; ++i) {
                float trail = i / 69.0f;
                float x = head + side * trail * 4.1f;
                float swirl = timeline.time * 6.0f + i * 2.39996f;
                Vector3 p{x, .09f * std::cos(swirl) * (1.0f - trail), .09f * std::sin(swirl) * (1.0f - trail)};
                DrawBillboard(camera, glow_, p, .08f + .16f * (1.0f - trail), Fade(color, .95f * (1.0f - trail)));
            }
        }
        rlEnableDepthMask();
        EndBlendMode();
        (void)right;
        (void)up;
    }

    void drawTrackLine(const Particle& particle, float progress, float fieldTesla, float fade, float intensity = 1.0f) {
        constexpr int segments = 84;
        float maxDistance = 17.0f;
        Vector3 previous{};
        bool havePrevious = false;
        for (int i = 1; i <= segments; ++i) {
            float fraction = progress * i / segments;
            Vec3 modelPoint = trackPoint(particle, maxDistance * fraction, fieldTesla);
            Vector3 point = render(modelPoint);
            if (radialDistance(modelPoint) > maxRadius(particle.species) || std::abs(point.x) > 9.3f) break;
            if (havePrevious) {
                float head = i / static_cast<float>(segments);
                Color c = Fade(render(particle.color), fade * intensity * (.32f + .68f * head));
                rlColor4ub(c.r, c.g, c.b, c.a);
                rlVertex3f(previous.x, previous.y, previous.z);
                rlVertex3f(point.x, point.y, point.z);
            }
            previous = point;
            havePrevious = true;
        }
    }

    void drawEvent(const Camera3D& camera, const Event& event, const Timeline& timeline, float fieldTesla) {
        float progress = timeline.tracks();
        float fade = timeline.holdFade();
        if (timeline.collision() <= 0.0f) return;

        BeginBlendMode(BLEND_ADDITIVE);
        rlDisableDepthMask();
        rlSetTexture(0);
        rlBegin(RL_LINES);
        for (const auto& particle : event.particles) drawTrackLine(particle, progress, fieldTesla, fade);
        rlEnd();

        // Give the high-energy pair a soft glow and bright core. Layered line
        // batches are considerably cheaper than dozens of tiny tube meshes.
        if (event.particles.size() >= 2) {
            rlSetLineWidth(4.0f);
            rlBegin(RL_LINES);
            for (size_t i = event.particles.size() - 2; i < event.particles.size(); ++i)
                drawTrackLine(event.particles[i], progress, fieldTesla, fade, .18f);
            rlEnd();
            rlSetLineWidth(2.0f);
            rlBegin(RL_LINES);
            for (size_t i = event.particles.size() - 2; i < event.particles.size(); ++i)
                drawTrackLine(event.particles[i], progress, fieldTesla, fade, 1.0f);
            rlEnd();
            rlSetLineWidth(1.0f);
        }

        // Energy deposits appear at the subsystem reached by each particle.
        for (const auto& particle : event.particles) {
            if (progress < .80f) continue;
            Vector3 hit = render(endpoint(particle, fieldTesla));
            float energy = std::clamp(particle.momentum / 80.0f, .12f, 1.0f);
            float pulse = .70f + .30f * std::sin(GetTime() * 4.0 + particle.phase);
            DrawBillboard(camera, glow_, hit, (.16f + .44f * energy) * pulse, Fade(render(particle.color), .52f * fade));
        }

        float flash = timeline.collision() * (1.0f - std::clamp((timeline.time - 2.72f) / .8f, 0.0f, 1.0f));
        float vertexGlow = (.10f + .10f * std::sin(GetTime() * 3.2f)) * timeline.reveal() * fade;
        DrawBillboard(camera, glow_, {0, 0, 0}, .48f, Fade(WHITE, vertexGlow));
        DrawBillboard(camera, glow_, {0, 0, 0}, 1.15f, Fade(kViolet, vertexGlow * .32f));
        if (flash > .001f) {
            DrawBillboard(camera, glow_, {0, 0, 0}, 1.2f + 3.8f * flash, Fade(WHITE, .92f * flash));
            DrawBillboard(camera, glow_, {0, 0, 0}, 3.0f + 5.0f * flash, Fade(kCyan, .26f * flash));
        }
        rlEnableDepthMask();
        EndBlendMode();
    }
};

}  // namespace collider
