#include "raylib.h"
#include "raymath.h"
#include "../common/studio.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <deque>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kScreenWidth = 1440;
constexpr int kScreenHeight = 900;

constexpr float kG = 14.0f;
constexpr float kSoftening = 0.18f;
constexpr int kTrailMax = 1800;
constexpr float kFixedStep = 1.0f / 240.0f;
constexpr Color kVelocity{103, 220, 244, 255};
constexpr Color kForce{255, 191, 112, 255};
constexpr Color kAcceleration{192, 166, 255, 255};
constexpr Color kText{233, 240, 249, 255};
constexpr Color kMuted{148, 167, 189, 255};
constexpr float kVelocityScale = 0.9f;
constexpr float kForceScale = 0.35f;
constexpr float kAccelerationScale = 1.2f;
constexpr float kArrowMax = 4.5f;
constexpr float kPi = 3.14159265358979323846f;

struct Body {
    const char* name;
    float mass;
    float radius;
    Vector3 pos;
    Vector3 vel;
    Color color;
};

struct Preset {
    const char* name;
    std::array<Body, 3> bodies;
    float suggestedDistance;
    const char* description;
};

struct Derivative {
    std::array<Vector3, 3> dPos{};
    std::array<Vector3, 3> dVel{};
};

struct State {
    std::array<Vector3, 3> pos{};
    std::array<Vector3, 3> vel{};
};

// Pair force uses exactly the same softened potential as the integrator.
Vector3 PairForce(Vector3 from, Vector3 to, float fromMass, float toMass) {
    Vector3 delta = Vector3Subtract(to, from);
    float dist2 = Vector3LengthSqr(delta) + kSoftening * kSoftening;
    return Vector3Scale(delta, kG * fromMass * toMass / (dist2 * std::sqrt(dist2)));
}

Vector3 ComputeAcceleration(const std::array<Vector3, 3>& positions,
                            const std::array<float, 3>& masses, int idx) {
    Vector3 acc{};
    for (int other=0; other<3; ++other) {
        if (other != idx)
            acc = Vector3Add(acc, PairForce(positions[idx], positions[other], 1.0f, masses[other]));
    }
    return acc;
}

Vector3 BodyAcceleration(const std::array<Body, 3>& bodies, int i) {
    std::array<Vector3,3> positions{};
    std::array<float,3> masses{};
    for (int j=0; j<3; ++j) { positions[j]=bodies[j].pos; masses[j]=bodies[j].mass; }
    return ComputeAcceleration(positions, masses, i);
}

Derivative EvaluateDerivative(const State& state, const std::array<float, 3>& masses) {
    Derivative deriv{};
    for (int i = 0; i < 3; ++i) {
        deriv.dPos[i] = state.vel[i];
        deriv.dVel[i] = ComputeAcceleration(state.pos, masses, i);
    }
    return deriv;
}

State AdvanceState(const State& state, const Derivative& deriv, float dt) {
    State next = state;
    for (int i = 0; i < 3; ++i) {
        next.pos[i] = Vector3Add(state.pos[i], Vector3Scale(deriv.dPos[i], dt));
        next.vel[i] = Vector3Add(state.vel[i], Vector3Scale(deriv.dVel[i], dt));
    }
    return next;
}

void StepRK4(std::array<Body, 3>* bodies, float dt) {
    State start{};
    std::array<float, 3> masses{};
    for (int i = 0; i < 3; ++i) {
        start.pos[i] = (*bodies)[i].pos;
        start.vel[i] = (*bodies)[i].vel;
        masses[i] = (*bodies)[i].mass;
    }

    const Derivative k1 = EvaluateDerivative(start, masses);
    const Derivative k2 = EvaluateDerivative(AdvanceState(start, k1, dt * 0.5f), masses);
    const Derivative k3 = EvaluateDerivative(AdvanceState(start, k2, dt * 0.5f), masses);
    const Derivative k4 = EvaluateDerivative(AdvanceState(start, k3, dt), masses);

    for (int i = 0; i < 3; ++i) {
        Vector3 posDelta = Vector3Add(k1.dPos[i], Vector3Scale(k2.dPos[i], 2.0f));
        posDelta = Vector3Add(posDelta, Vector3Scale(k3.dPos[i], 2.0f));
        posDelta = Vector3Add(posDelta, k4.dPos[i]);

        Vector3 velDelta = Vector3Add(k1.dVel[i], Vector3Scale(k2.dVel[i], 2.0f));
        velDelta = Vector3Add(velDelta, Vector3Scale(k3.dVel[i], 2.0f));
        velDelta = Vector3Add(velDelta, k4.dVel[i]);

        (*bodies)[i].pos = Vector3Add((*bodies)[i].pos, Vector3Scale(posDelta, dt / 6.0f));
        (*bodies)[i].vel = Vector3Add((*bodies)[i].vel, Vector3Scale(velDelta, dt / 6.0f));
    }
}

float TotalMass(const std::array<Body, 3>& bodies) {
    return bodies[0].mass + bodies[1].mass + bodies[2].mass;
}

Vector3 ComputeBarycenter(const std::array<Body, 3>& bodies) {
    Vector3 weighted = {0.0f, 0.0f, 0.0f};
    float totalMass = TotalMass(bodies);
    for (const Body& body : bodies) {
        weighted = Vector3Add(weighted, Vector3Scale(body.pos, body.mass));
    }
    return Vector3Scale(weighted, 1.0f / totalMass);
}

Vector3 ComputeLinearMomentum(const std::array<Body, 3>& bodies) {
    Vector3 momentum = {0.0f, 0.0f, 0.0f};
    for (const Body& body : bodies) {
        momentum = Vector3Add(momentum, Vector3Scale(body.vel, body.mass));
    }
    return momentum;
}

float TotalEnergy(const std::array<Body, 3>& bodies) {
    float kinetic = 0.0f;
    float potential = 0.0f;

    for (const Body& body : bodies) {
        kinetic += 0.5f * body.mass * Vector3LengthSqr(body.vel);
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            float dist = std::sqrt(Vector3DistanceSqr(bodies[i].pos, bodies[j].pos) + kSoftening * kSoftening);
            potential -= kG * bodies[i].mass * bodies[j].mass / dist;
        }
    }

    return kinetic + potential;
}

float TotalAngularMomentum(const std::array<Body, 3>& bodies) {
    Vector3 angular = {0.0f, 0.0f, 0.0f};
    for (const Body& body : bodies) {
        angular = Vector3Add(angular, Vector3CrossProduct(body.pos, Vector3Scale(body.vel, body.mass)));
    }
    return Vector3Length(angular);
}

Preset MakeLagrangePreset() {
    constexpr float radius = 4.2f;
    constexpr float mass = 3.4f;
    const float separation2 = 3.0f * radius * radius + kSoftening*kSoftening;
    const float orbitalSpeed = std::sqrt(3*kG*mass*radius*radius / std::pow(separation2,1.5f));
    const Vector3 planeZ = Vector3Normalize({0.0f, -0.18f, 1.0f});

    std::array<Body, 3> bodies{};
    for (int i = 0; i < 3; ++i) {
        float angle = (2.0f * kPi * static_cast<float>(i)) / 3.0f;
        Vector3 pos = Vector3Add({radius * std::cos(angle),0,0}, Vector3Scale(planeZ,radius * std::sin(angle)));
        Vector3 radial = Vector3Normalize(pos);
        Vector3 axis = Vector3Normalize(Vector3{0.0f, 1.0f, 0.18f});
        Vector3 tangent = Vector3Normalize(Vector3CrossProduct(axis, radial));
        bodies[i] = {
            i == 0 ? "Aurelia" : (i == 1 ? "Cerulean" : "Rose"),
            mass,
            0.42f,
            pos,
            Vector3Scale(tangent, orbitalSpeed),
            i == 0 ? Color{255, 196, 104, 255} : (i == 1 ? Color{112, 214, 255, 255} : Color{255, 124, 168, 255}),
        };
    }

    return {
        "Lagrange Triangle",
        bodies,
        18.0f,
        "equal masses in a rotating equilateral configuration"
    };
}

Preset MakeBraidedChaosPreset() {
    std::array<Body, 3> bodies{{
        {"Alpha", 4.2f, 0.45f, {-4.8f, 1.1f, -1.6f}, {0.58f, 0.18f, 1.24f}, Color{255, 200, 120, 255}},
        {"Beta", 3.6f, 0.40f, {4.5f, -0.8f, 1.8f}, {-0.82f, 0.36f, -1.08f}, Color{118, 225, 255, 255}},
        {"Gamma", 2.4f, 0.33f, {0.2f, 0.9f, -5.2f}, {0.36f, -0.92f, 0.41f}, Color{188, 132, 255, 255}},
    }};

    return {
        "Braided Chaos",
        bodies,
        22.0f,
        "generic 3-body interaction with strong out-of-plane motion"
    };
}

Preset MakeBinaryIntruderPreset() {
    std::array<Body, 3> bodies{{
        {"Primary", 6.5f, 0.50f, {-1.9f, 0.0f, 0.0f}, {0.0f, 0.32f, 1.58f}, Color{255, 194, 110, 255}},
        {"Companion", 5.2f, 0.46f, {1.9f, 0.0f, 0.0f}, {0.0f, -0.28f, -1.92f}, Color{120, 213, 255, 255}},
        {"Intruder", 1.5f, 0.28f, {0.0f, 7.0f, -6.5f}, {-0.10f, -1.88f, 1.65f}, Color{255, 118, 166, 255}},
    }};

    return {
        "Binary Capture",
        bodies,
        24.0f,
        "close binary disturbed by a lighter incoming body"
    };
}

std::array<Preset, 3> BuildPresets() {
    return {MakeLagrangePreset(), MakeBraidedChaosPreset(), MakeBinaryIntruderPreset()};
}

void AppendTrails(
    const std::array<Body, 3>& bodies,
    std::array<std::deque<Vector3>, 3>* trails
) {
    for (int i = 0; i < 3; ++i) {
        (*trails)[i].push_back(bodies[i].pos);
        if (static_cast<int>((*trails)[i].size()) > kTrailMax) {
            (*trails)[i].pop_front();
        }
    }
}

void ResetSimulation(
    const Preset& preset,
    std::array<Body, 3>* bodies,
    std::array<std::deque<Vector3>, 3>* trails,
    float* simTime
) {
    *bodies = preset.bodies;
    for (std::deque<Vector3>& trail : *trails) {
        trail.clear();
    }
    AppendTrails(*bodies, trails);
    *simTime = 0.0f;
}

void DrawTrail(const std::deque<Vector3>& trail, Color color) {
    if (trail.size() < 2) {
        return;
    }

    for (size_t i = 1; i < trail.size(); ++i) {
        float alpha = static_cast<float>(i) / static_cast<float>(trail.size());
        Color segColor = color;
        segColor.a = static_cast<unsigned char>(18 + 170 * alpha);
        DrawLine3D(trail[i - 1], trail[i], segColor);
    }
}

void DrawStarfield(const std::vector<Vector3>& stars) {
    for (size_t i = 0; i < stars.size(); ++i) {
        unsigned char alpha = static_cast<unsigned char>(120 + (i % 120));
        DrawPoint3D(stars[i], Color{220, 232, 255, alpha});
    }
}

void UpdateOrbitCamera(Camera3D* camera, float* yaw, float* pitch, float* distance, Vector3 target) {
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON) && GetMouseX() < GetScreenWidth()-350) {
        Vector2 delta = GetMouseDelta();
        *yaw -= delta.x * 0.0034f;
        *pitch += delta.y * 0.0030f;
        *pitch = std::clamp(*pitch, -1.42f, 1.42f);
    }

    if (GetMouseX() < GetScreenWidth()-350) *distance -= GetMouseWheelMove() * 1.4f;
    *distance = std::clamp(*distance, 6.0f, 70.0f);

    camera->target = Vector3Lerp(camera->target, target, 0.09f);
    float cp = std::cos(*pitch);
    Vector3 offset = {
        *distance * cp * std::cos(*yaw),
        *distance * std::sin(*pitch),
        *distance * cp * std::sin(*yaw),
    };
    camera->position = Vector3Add(camera->target, offset);
}

std::vector<Vector3> BuildStarfield() {
    std::vector<Vector3> stars;
    stars.reserve(360);

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> azimuthDist(0.0f, 2.0f * kPi);
    std::uniform_real_distribution<float> heightDist(-0.9f, 0.9f);
    std::uniform_real_distribution<float> radiusDist(42.0f, 70.0f);

    for (int i = 0; i < 360; ++i) {
        float azimuth = azimuthDist(rng);
        float y = heightDist(rng);
        float ring = std::sqrt(std::max(0.0f, 1.0f - y * y));
        float radius = radiusDist(rng);
        stars.push_back({
            radius * ring * std::cos(azimuth),
            radius * y,
            radius * ring * std::sin(azimuth),
        });
    }

    return stars;
}

bool InFront(Vector3 p, Camera3D camera) {
    return Vector3DotProduct(Vector3Subtract(p,camera.position), Vector3Subtract(camera.target,camera.position)) > 0;
}

// CPU-lit sphere: each face has a real surface normal, rather than wire shells.
void DrawBodyWithGlow(const Body& body) {
    constexpr int rings=20, slices=36;
    auto point = [&](int ring,int slice) {
        float latitude = kPi*ring/rings, longitude = 2*kPi*slice/slices;
        return Vector3Add(body.pos, Vector3Scale({std::sin(latitude)*std::cos(longitude),
                              std::cos(latitude),std::sin(latitude)*std::sin(longitude)},body.radius));
    };
    Vector3 light=Vector3Normalize({-0.6f,0.8f,0.5f});
    for (int r=0;r<rings;++r) for (int a=0;a<slices;++a) {
        Vector3 p00=point(r,a), p10=point(r+1,a), p11=point(r+1,a+1), p01=point(r,a+1);
        Vector3 normal=Vector3Normalize(Vector3Subtract(Vector3Scale(Vector3Add(p00,p11),0.5f),body.pos));
        float lighting=0.22f+0.78f*std::max(0.0f,Vector3DotProduct(normal,light));
        Color c{static_cast<unsigned char>(body.color.r*lighting),
                static_cast<unsigned char>(body.color.g*lighting),
                static_cast<unsigned char>(body.color.b*lighting),255};
        DrawTriangle3D(p00,p11,p10,c);
        DrawTriangle3D(p00,p01,p11,c);
    }
}

void ArrowHead(Vector2 tip, Vector2 direction, float size, Color color) {
    Vector2 side{-direction.y,direction.x};
    Vector2 base=Vector2Subtract(tip,Vector2Scale(direction,size));
    DrawTriangle(tip,Vector2Subtract(base,Vector2Scale(side,size*0.45f)),
                 Vector2Add(base,Vector2Scale(side,size*0.45f)),color);
}

std::vector<Rectangle> annotationBounds;
void Annotation(const char* value, float x, float y, Color color, bool panel=true) {
    float width=studio::textWidth(value,14)+16;
    if (x<0 || x>GetScreenWidth()-350 || y<130 || y>GetScreenHeight()-135) return;
    x=std::clamp(x,12.0f,GetScreenWidth()-360.0f-width);
    const float offsets[]={0,28,-28,56,-56,84,-84,112};
    Rectangle bounds{x-6,y-4,width,25};
    for (float offset:offsets) {
        Rectangle candidate{x-6,y-4+offset,width,25};
        if (candidate.y<130 || candidate.y+25>GetScreenHeight()-135) continue;
        bool collides=false;
        for (Rectangle used:annotationBounds) if (CheckCollisionRecs(candidate,used)) { collides=true; break; }
        if (!collides) { bounds=candidate; break; }
    }
    annotationBounds.push_back(bounds);
    if (panel) studio::panel(bounds,Color{11,18,31,235});
    studio::text(value,bounds.x+6,bounds.y+4,14,color);
}

void VectorArrow(Camera3D camera, const Body& body, Vector3 vector, float scale,
                 Color color, const char* label, bool dashed, Vector2 labelOffset) {
    float magnitude=Vector3Length(vector);
    if (magnitude<0.00001f || !InFront(body.pos,camera)) return;
    float length=magnitude*scale;
    bool capped=length>kArrowMax;
    Vector3 tip=Vector3Add(body.pos,Vector3Scale(vector,std::min(length,kArrowMax)/magnitude));
    if (!InFront(tip,camera)) return;
    Vector2 from=GetWorldToScreen(body.pos,camera), to=GetWorldToScreen(tip,camera);
    float screenLength=Vector2Distance(from,to);
    if (screenLength<4) return;
    Vector2 direction=Vector2Normalize(Vector2Subtract(to,from));
    if (dashed) {
        for (float d=0;d<screenLength;d+=13)
            DrawLineEx(Vector2Add(from,Vector2Scale(direction,d)),
                       Vector2Add(from,Vector2Scale(direction,std::min(d+8,screenLength))),2,color);
    } else DrawLineEx(from,to,3,color);
    ArrowHead(to,direction,std::min(11.0f,screenLength*0.4f),color);
    if (label && *label) {
        float x=to.x+labelOffset.x,y=to.y+labelOffset.y;
        std::string value=std::string(label)+(capped ? " *" : "");
        Annotation(value.c_str(),x,y,color);
    }
}

void TrailDirections(const std::deque<Vector3>& trail, Camera3D camera, Color color) {
    // Arrowheads follow increasing sample time, independent of current velocity.
    if (trail.size()<25) return;
    Vector2 last{-10000,-10000};
    int count=0;
    for (int i=int(trail.size())-12;i>=12 && count<7;i-=48) {
        if (!InFront(trail[i],camera) || !InFront(trail[i-10],camera)) continue;
        Vector2 a=GetWorldToScreen(trail[i-10],camera), b=GetWorldToScreen(trail[i],camera);
        if (Vector2Distance(a,b)<3 || Vector2Distance(last,b)<65) continue;
        ArrowHead(b,Vector2Normalize(Vector2Subtract(b,a)),7,Fade(color,0.7f));
        last=b; ++count;
    }
}

void Button(Rectangle r, const char* text, bool on, Color color) {
    DrawRectangleRounded(r,0.15f,8,on ? Color{35,47,68,255} : Color{19,29,45,255});
    DrawRectangleRoundedLinesEx(r,0.15f,8,1,on ? color : Color{48,62,83,255});
    studio::text(text,r.x+12,r.y+11,16,on ? color : kMuted);
}

void DrawDashboard(const std::array<Body,3>& bodies, const Preset& preset, int selected,
                   bool paused, float speed, float time, float initialEnergy, bool velocity,
                   bool forces, bool acceleration, bool trails, float vectorScale) {
    float x=GetScreenWidth()-330.0f;
    studio::text("ORBITAL DYNAMICS / 02",28,24,13,kVelocity);
    studio::text("Three-body problem",27,45,32,kText);
    studio::text(preset.name,29,91,19,kMuted);
    studio::panel({x,20,306,float(GetScreenHeight()-104)},Color{12,20,34,248});
    studio::text("EXPERIMENT",x+20,39,13,kMuted);
    studio::text(paused ? "PAUSED" : "ORBITS ACTIVE",x+20,66,19,paused ? kForce : kVelocity);
    studio::text(TextFormat("t %.2f    /    %.2fx speed",time,speed),x+20,105,16,kText);
    studio::text("INSPECT A BODY",x+20,149,13,kMuted);
    for (int i=0;i<3;++i) Button({x+20+i*92.0f,175,82,38},TextFormat("%c",'A'+i),selected==i,bodies[i].color);
    const Body& b=bodies[selected];
    studio::text(b.name,x+20,233,24,b.color);
    studio::text(TextFormat("Mass   %.2f",b.mass),x+20,270,16,kMuted);
    studio::text(TextFormat("Speed   %.3f",Vector3Length(b.vel)),x+20,302,18,kVelocity);
    Vector3 acc=BodyAcceleration(bodies,selected);
    studio::text(TextFormat("Acceleration   %.3f",Vector3Length(acc)),x+20,334,18,kAcceleration);
    int row=0;
    for (int other=0;other<3;++other) if (other!=selected) {
        Vector3 f=PairForce(b.pos,bodies[other].pos,b.mass,bodies[other].mass);
        studio::text(TextFormat("Pull from %c     %.3f",'A'+other,Vector3Length(f)),x+20,374+row*30,16,kForce);
        ++row;
    }
    Button({x+20,452,128,40},"V  Motion",velocity,kVelocity);
    Button({x+158,452,128,40},"F  Gravity",forces,kForce);
    Button({x+20,502,128,40},"A  Net accel.",acceleration,kAcceleration);
    Button({x+158,502,128,40},"T  Trails",trails,b.color);
    studio::text(TextFormat("Arrow multiplier  %.2fx",vectorScale),x+20,564,16,kText);
    studio::text("[ / ]  shrink / enlarge",x+20,590,14,kMuted);
    studio::text("Separate scales per vector type",x+20,623,13,kMuted);
    studio::text(TextFormat("v %.2f   F %.2f   a %.2f",kVelocityScale*vectorScale,
                                  kForceScale*vectorScale,kAccelerationScale*vectorScale),x+20,646,14,kMuted);
    studio::text("Simulation units; G = 14",x+20,675,14,kMuted);
    studio::text("* Long arrows capped at 4.5 units",x+20,699,13,kMuted);
    float energy=TotalEnergy(bodies);
    float drift=(energy-initialEnergy)/std::max(0.0001f,std::fabs(initialEnergy));
    studio::text(TextFormat("Energy drift    %+.4f%%",drift*100),x+20,743,15,kText);
    studio::text(TextFormat("Momentum |P|    %.4f",Vector3Length(ComputeLinearMomentum(bodies))),x+20,770,14,kMuted);
    studio::panel({24,float(GetScreenHeight()-65),float(GetScreenWidth()-48),43},Color{12,20,34,245});
    studio::fit("Drag: orbit   Wheel: zoom   1 / 2 / 3: presets   Tab: select body   P: pause   R: reset   - / +: speed   B: barycenter   O: top view",
                 {40,float(GetScreenHeight()-53),float(GetScreenWidth()-80),24},15,kMuted);
    studio::text("SOLID: MOTION    DASHED: GRAVITY    PURPLE: NET ACCELERATION",28,GetScreenHeight()-115,13,kMuted);
    studio::text("Trail arrowheads show the direction of travel.",28,GetScreenHeight()-91,14,kMuted);
}

}  // namespace

int main() {
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(kScreenWidth, kScreenHeight, "Three-body problem | Orbital Lab");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.position = {18.0f, 10.0f, 18.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 42.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    float camYaw = 0.85f;
    float camPitch = 0.45f;
    float camDistance = 18.0f;

    const std::array<Preset, 3> presets = BuildPresets();
    int presetIndex = 0;
    std::array<Body, 3> bodies{};
    std::array<std::deque<Vector3>, 3> trails;
    float simTime = 0.0f;
    ResetSimulation(presets[presetIndex], &bodies, &trails, &simTime);

    float speed = 1.0f;
    bool paused = false;
    bool showTrails = true;
    bool showVectors = true;
    bool showForces = true, showAcceleration = false;
    int selected = 0;
    float vectorScale = 1.0f;
    float initialEnergy = TotalEnergy(bodies);
    bool showBarycenter = true;

    const std::vector<Vector3> stars = BuildStarfield();

    while (!WindowShouldClose()) {
        float panelX = GetScreenWidth()-330.0f;
        for (int i=0;i<3;++i) if (studio::clicked({panelX+20+i*92.0f,175,82,38})) selected=i;
        if (IsKeyPressed(KEY_TAB)) selected=(selected+1)%3;
        if (IsKeyPressed(KEY_F) || studio::clicked({panelX+158,452,128,40})) showForces=!showForces;
        if (IsKeyPressed(KEY_A) || studio::clicked({panelX+20,502,128,40})) showAcceleration=!showAcceleration;
        if (IsKeyDown(KEY_LEFT_BRACKET)) vectorScale=std::max(0.15f,vectorScale*std::exp(-GetFrameTime()));
        if (IsKeyDown(KEY_RIGHT_BRACKET)) vectorScale=std::min(4.0f,vectorScale*std::exp(GetFrameTime()));
        if (IsKeyPressed(KEY_O)) { camYaw=1.5708f; camPitch=1.42f; }
        int requestedPreset = presetIndex;
        if (IsKeyPressed(KEY_ONE)) requestedPreset = 0;
        if (IsKeyPressed(KEY_TWO)) requestedPreset = 1;
        if (IsKeyPressed(KEY_THREE)) requestedPreset = 2;

        if (requestedPreset != presetIndex) {
            presetIndex = requestedPreset;
            camDistance = presets[presetIndex].suggestedDistance;
            ResetSimulation(presets[presetIndex], &bodies, &trails, &simTime);
            initialEnergy=TotalEnergy(bodies);
        }

        if (IsKeyPressed(KEY_R)) {
            ResetSimulation(presets[presetIndex], &bodies, &trails, &simTime);
            initialEnergy=TotalEnergy(bodies);
        }
        if (IsKeyPressed(KEY_P)) paused = !paused;
        if (IsKeyPressed(KEY_T) || studio::clicked({panelX+158,502,128,40})) showTrails = !showTrails;
        if (IsKeyPressed(KEY_V) || studio::clicked({panelX+20,452,128,40})) showVectors = !showVectors;
        if (IsKeyPressed(KEY_B)) showBarycenter = !showBarycenter;
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) speed = std::min(6.0f, speed + 0.25f);
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) speed = std::max(0.25f, speed - 0.25f);

        Vector3 barycenter = ComputeBarycenter(bodies);
        UpdateOrbitCamera(&camera, &camYaw, &camPitch, &camDistance, barycenter);

        if (!paused) {
            float frameAdvance = std::min(GetFrameTime(),0.1f) * speed;
            int steps = std::max(1, static_cast<int>(std::ceil(frameAdvance / kFixedStep)));
            float dt = frameAdvance / static_cast<float>(steps);
            for (int i = 0; i < steps; ++i) {
                StepRK4(&bodies, dt);
                simTime += dt;
            }
            AppendTrails(bodies, &trails);
            barycenter = ComputeBarycenter(bodies);
        }

        BeginDrawing();
        ClearBackground(Color{4, 6, 14, 255});

        DrawRectangleGradientV(0, 0, kScreenWidth, kScreenHeight, Color{7, 11, 24, 255}, Color{2, 3, 8, 255});

        BeginMode3D(camera);

        DrawStarfield(stars);

        if (showTrails) {
            for (int i = 0; i < 3; ++i) {
                DrawTrail(trails[i], bodies[i].color);
            }
        }

        if (showBarycenter) {
            DrawSphere(ComputeBarycenter(bodies), 0.14f, Color{245, 245, 255, 235});
        }

        for (const Body& body : bodies) {
            DrawBodyWithGlow(body);
        }

        EndMode3D();

        annotationBounds.clear();
        BeginScissorMode(0,130,GetScreenWidth()-350,GetScreenHeight()-265);
        for (int i=0;i<3;++i) {
            const Body& b=bodies[i];
            if (showTrails) TrailDirections(trails[i],camera,b.color);
            if (showVectors) VectorArrow(camera,b,b.vel,kVelocityScale*vectorScale,kVelocity,
                                         i==selected ? "motion v" : "",false,{10,4});
            if (InFront(b.pos,camera)) {
                Vector2 pos=GetWorldToScreen(b.pos,camera);
                if (i==selected) {
                    Vector3 right=Vector3Normalize(Vector3CrossProduct(Vector3Subtract(camera.target,camera.position),camera.up));
                    float radius=Vector2Distance(pos,GetWorldToScreen(Vector3Add(b.pos,Vector3Scale(right,b.radius)),camera));
                    DrawCircleLines(int(pos.x),int(pos.y),radius+6,Fade(b.color,0.7f));
                }
                Annotation(TextFormat("%c / %s",'A'+i,b.name),pos.x+22,pos.y-27,b.color,false);
            }
        }
        const Body& inspected=bodies[selected];
        int forceRow=0;
        if (showForces) for (int other=0;other<3;++other) if (other!=selected) {
            Vector3 f=PairForce(inspected.pos,bodies[other].pos,inspected.mass,bodies[other].mass);
            VectorArrow(camera,inspected,f,kForceScale*vectorScale,kForce,
                        TextFormat("pull from %c",'A'+other),true,{10,forceRow==0 ? 28.0f : -25.0f});
            ++forceRow;
        }
        if (showAcceleration) VectorArrow(camera,inspected,BodyAcceleration(bodies,selected),
                                kAccelerationScale*vectorScale,kAcceleration,"net a",false,{10,-48});
        EndScissorMode();
        DrawDashboard(bodies,presets[presetIndex],selected,paused,speed,simTime,initialEnergy,
                      showVectors,showForces,showAcceleration,showTrails,vectorScale);

        EndDrawing();
        if (studio::smokeFrame(__FILE__)) break;
    }

    studio::unload();
    CloseWindow();
    return 0;
}
