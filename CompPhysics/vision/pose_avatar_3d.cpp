#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define NOGDI
#define NOUSER
#endif

#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace {

constexpr int kWindowWidth = 1440;
constexpr int kWindowHeight = 900;
constexpr int kPanelWidth = 310;
constexpr int kPosePort = 50525;
constexpr int kLandmarkCount = 33;
constexpr float kLinkTimeout = 0.8f;

constexpr int NOSE = 0;
constexpr int LEFT_EAR = 7;
constexpr int RIGHT_EAR = 8;
constexpr int LEFT_SHOULDER = 11;
constexpr int RIGHT_SHOULDER = 12;
constexpr int LEFT_ELBOW = 13;
constexpr int RIGHT_ELBOW = 14;
constexpr int LEFT_WRIST = 15;
constexpr int RIGHT_WRIST = 16;
constexpr int LEFT_PINKY = 17;
constexpr int RIGHT_PINKY = 18;
constexpr int LEFT_INDEX = 19;
constexpr int RIGHT_INDEX = 20;
constexpr int LEFT_THUMB = 21;
constexpr int RIGHT_THUMB = 22;
constexpr int LEFT_HIP = 23;
constexpr int RIGHT_HIP = 24;
constexpr int LEFT_KNEE = 25;
constexpr int RIGHT_KNEE = 26;
constexpr int LEFT_ANKLE = 27;
constexpr int RIGHT_ANKLE = 28;
constexpr int LEFT_HEEL = 29;
constexpr int RIGHT_HEEL = 30;
constexpr int LEFT_FOOT = 31;
constexpr int RIGHT_FOOT = 32;

struct PosePacket {
    bool valid = false;
    int64_t timestampMs = 0;
    int personIndex = 0;
    std::array<Vector3, kLandmarkCount> landmarks{};
    std::array<float, kLandmarkCount> visibility{};
};

class UdpPoseReceiver {
  public:
    bool Start(uint16_t port) {
#ifdef _WIN32
        WSADATA wsaData{};
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) return false;
        socket_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (socket_ == INVALID_SOCKET) return false;
#else
        socket_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (socket_ < 0) return false;
#endif
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_ANY);
        address.sin_port = htons(port);
        if (bind(socket_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
            Close();
            return false;
        }
#ifdef _WIN32
        u_long nonBlocking = 1;
        if (ioctlsocket(socket_, FIONBIO, &nonBlocking) != 0) {
            Close();
            return false;
        }
#else
        const int flags = fcntl(socket_, F_GETFL, 0);
        if (flags < 0 || fcntl(socket_, F_SETFL, flags | O_NONBLOCK) < 0) {
            Close();
            return false;
        }
#endif
        ready_ = true;
        return true;
    }

    bool Poll(PosePacket& newest, int& packetCount) {
        packetCount = 0;
        if (!ready_) return false;
        bool receivedPrimary = false;
        bool receivedFallback = false;
        PosePacket fallback{};
        std::array<char, 8192> buffer{};
        while (true) {
            const int bytes = recvfrom(
                socket_, buffer.data(), static_cast<int>(buffer.size()) - 1, 0, nullptr, nullptr);
            if (bytes <= 0) break;
            buffer[bytes] = '\0';
            PosePacket parsed{};
            if (!Parse(buffer.data(), parsed)) continue;
            ++packetCount;
            if (!receivedFallback) {
                fallback = parsed;
                receivedFallback = true;
            }
            if (parsed.personIndex == 0) {
                newest = parsed;
                receivedPrimary = true;
            }
        }
        if (!receivedPrimary && receivedFallback) newest = fallback;
        return receivedPrimary || receivedFallback;
    }

    void Close() {
        if (!ready_) return;
#ifdef _WIN32
        closesocket(socket_);
        WSACleanup();
        socket_ = INVALID_SOCKET;
#else
        close(socket_);
        socket_ = -1;
#endif
        ready_ = false;
    }

    ~UdpPoseReceiver() { Close(); }

  private:
    static bool Parse(const char* message, PosePacket& packet) {
        std::vector<std::string> fields;
        std::stringstream stream(message);
        std::string field;
        while (std::getline(stream, field, ',')) fields.push_back(field);
        if (fields.size() < 5 || fields[0] != "POSE" || fields[1] != "1") return false;
        try {
            packet.timestampMs = std::stoll(fields[2]);
            packet.personIndex = std::stoi(fields[3]);
            const int count = std::stoi(fields[4]);
            if (count != kLandmarkCount || fields.size() < 5 + kLandmarkCount * 4) return false;
            for (int i = 0; i < kLandmarkCount; ++i) {
                const size_t offset = 5 + static_cast<size_t>(i) * 4;
                packet.landmarks[i] = {
                    std::stof(fields[offset]),
                    std::stof(fields[offset + 1]),
                    std::stof(fields[offset + 2]),
                };
                packet.visibility[i] = std::stof(fields[offset + 3]);
            }
        } catch (...) {
            return false;
        }
        packet.valid = true;
        return true;
    }

#ifdef _WIN32
    SOCKET socket_ = INVALID_SOCKET;
#else
    int socket_ = -1;
#endif
    bool ready_ = false;
};

struct AvatarState {
    std::array<Vector3, kLandmarkCount> current{};
    std::array<Vector3, kLandmarkCount> target{};
    std::array<float, kLandmarkCount> visibility{};
    bool mirror = true;
};

Vector3 Midpoint(Vector3 a, Vector3 b) {
    return Vector3Scale(Vector3Add(a, b), 0.5f);
}

Vector3 DirectionOr(Vector3 vector, Vector3 fallback) {
    if (Vector3LengthSqr(vector) > 1.0e-5f) return Vector3Normalize(vector);
    if (Vector3LengthSqr(fallback) > 1.0e-5f) return Vector3Normalize(fallback);
    return {0.0f, -1.0f, 0.0f};
}

void PlaceChild(
    std::array<Vector3, kLandmarkCount>& pose,
    const std::array<Vector3, kLandmarkCount>& raw,
    int parent,
    int child,
    float length,
    const std::array<Vector3, kLandmarkCount>& fallback) {
    const Vector3 direction = DirectionOr(
        Vector3Subtract(raw[child], raw[parent]),
        Vector3Subtract(fallback[child], fallback[parent]));
    pose[child] = Vector3Add(pose[parent], Vector3Scale(direction, length));
}

void SetDefaultPose(std::array<Vector3, kLandmarkCount>& pose) {
    pose.fill({0.0f, 3.0f, 0.0f});
    pose[LEFT_HIP] = {-0.34f, 2.75f, 0.0f};
    pose[RIGHT_HIP] = {0.34f, 2.75f, 0.0f};
    pose[LEFT_SHOULDER] = {-0.68f, 4.45f, 0.0f};
    pose[RIGHT_SHOULDER] = {0.68f, 4.45f, 0.0f};
    pose[LEFT_ELBOW] = {-1.25f, 3.95f, 0.0f};
    pose[RIGHT_ELBOW] = {1.25f, 3.95f, 0.0f};
    pose[LEFT_WRIST] = {-1.62f, 3.30f, 0.0f};
    pose[RIGHT_WRIST] = {1.62f, 3.30f, 0.0f};
    pose[LEFT_PINKY] = pose[LEFT_INDEX] = pose[LEFT_THUMB] = {-1.67f, 3.18f, 0.0f};
    pose[RIGHT_PINKY] = pose[RIGHT_INDEX] = pose[RIGHT_THUMB] = {1.67f, 3.18f, 0.0f};
    pose[LEFT_KNEE] = {-0.34f, 1.48f, 0.02f};
    pose[RIGHT_KNEE] = {0.34f, 1.48f, 0.02f};
    pose[LEFT_ANKLE] = {-0.34f, 0.25f, 0.0f};
    pose[RIGHT_ANKLE] = {0.34f, 0.25f, 0.0f};
    pose[LEFT_HEEL] = {-0.34f, 0.12f, 0.02f};
    pose[RIGHT_HEEL] = {0.34f, 0.12f, 0.02f};
    pose[LEFT_FOOT] = {-0.34f, 0.10f, -0.42f};
    pose[RIGHT_FOOT] = {0.34f, 0.10f, -0.42f};
    pose[NOSE] = {0.0f, 5.32f, -0.18f};
    pose[LEFT_EAR] = {-0.25f, 5.28f, 0.0f};
    pose[RIGHT_EAR] = {0.25f, 5.28f, 0.0f};
}

void RetargetPose(const PosePacket& packet, AvatarState& avatar) {
    const Vector3 rawHip = Midpoint(packet.landmarks[LEFT_HIP], packet.landmarks[RIGHT_HIP]);
    const Vector3 rawShoulder = Midpoint(
        packet.landmarks[LEFT_SHOULDER], packet.landmarks[RIGHT_SHOULDER]);
    const float rawTorso = std::max(0.08f, Vector3Distance(rawHip, rawShoulder));
    const float scale = std::clamp(1.75f / rawTorso, 2.6f, 7.0f);

    std::array<Vector3, kLandmarkCount> raw{};
    for (int i = 0; i < kLandmarkCount; ++i) {
        Vector3 relative = Vector3Subtract(packet.landmarks[i], rawHip);
        relative.x *= avatar.mirror ? -scale : scale;
        relative.y *= -scale;
        relative.z *= -scale;
        raw[i] = relative;
        avatar.visibility[i] = packet.visibility[i];
    }

    // Reconstruct a fixed-proportion avatar from the observed joint directions.
    // This prevents noisy depth estimates from stretching arms and legs.
    const Vector3 hipAxis = DirectionOr(
        Vector3Subtract(raw[RIGHT_HIP], raw[LEFT_HIP]),
        Vector3Subtract(avatar.target[RIGHT_HIP], avatar.target[LEFT_HIP]));
    const Vector3 shoulderAxis = DirectionOr(
        Vector3Subtract(raw[RIGHT_SHOULDER], raw[LEFT_SHOULDER]),
        Vector3Subtract(avatar.target[RIGHT_SHOULDER], avatar.target[LEFT_SHOULDER]));
    const Vector3 rawHipMapped = Midpoint(raw[LEFT_HIP], raw[RIGHT_HIP]);
    const Vector3 rawShoulderMapped = Midpoint(raw[LEFT_SHOULDER], raw[RIGHT_SHOULDER]);
    const Vector3 torsoDirection = DirectionOr(
        Vector3Subtract(rawShoulderMapped, rawHipMapped),
        Vector3Subtract(
            Midpoint(avatar.target[LEFT_SHOULDER], avatar.target[RIGHT_SHOULDER]),
            Midpoint(avatar.target[LEFT_HIP], avatar.target[RIGHT_HIP])));
    const Vector3 hipCenter{0.0f, 0.0f, 0.0f};
    const Vector3 shoulderCenter = Vector3Add(hipCenter, Vector3Scale(torsoDirection, 1.75f));
    avatar.target[LEFT_HIP] = Vector3Add(hipCenter, Vector3Scale(hipAxis, -0.34f));
    avatar.target[RIGHT_HIP] = Vector3Add(hipCenter, Vector3Scale(hipAxis, 0.34f));
    avatar.target[LEFT_SHOULDER] = Vector3Add(shoulderCenter, Vector3Scale(shoulderAxis, -0.68f));
    avatar.target[RIGHT_SHOULDER] = Vector3Add(shoulderCenter, Vector3Scale(shoulderAxis, 0.68f));

    PlaceChild(avatar.target, raw, LEFT_SHOULDER, LEFT_ELBOW, 0.88f, avatar.current);
    PlaceChild(avatar.target, raw, LEFT_ELBOW, LEFT_WRIST, 0.78f, avatar.current);
    PlaceChild(avatar.target, raw, RIGHT_SHOULDER, RIGHT_ELBOW, 0.88f, avatar.current);
    PlaceChild(avatar.target, raw, RIGHT_ELBOW, RIGHT_WRIST, 0.78f, avatar.current);
    PlaceChild(avatar.target, raw, LEFT_HIP, LEFT_KNEE, 1.30f, avatar.current);
    PlaceChild(avatar.target, raw, LEFT_KNEE, LEFT_ANKLE, 1.28f, avatar.current);
    PlaceChild(avatar.target, raw, RIGHT_HIP, RIGHT_KNEE, 1.30f, avatar.current);
    PlaceChild(avatar.target, raw, RIGHT_KNEE, RIGHT_ANKLE, 1.28f, avatar.current);

    // Preserve hand and foot articulation relative to their constrained joints.
    for (int index : {LEFT_PINKY, LEFT_INDEX, LEFT_THUMB}) {
        const Vector3 offset = Vector3Scale(
            DirectionOr(Vector3Subtract(raw[index], raw[LEFT_WRIST]), {-1.0f, -0.2f, 0.0f}),
            index == LEFT_THUMB ? 0.24f : 0.30f);
        avatar.target[index] = Vector3Add(avatar.target[LEFT_WRIST], offset);
    }
    for (int index : {RIGHT_PINKY, RIGHT_INDEX, RIGHT_THUMB}) {
        const Vector3 offset = Vector3Scale(
            DirectionOr(Vector3Subtract(raw[index], raw[RIGHT_WRIST]), {1.0f, -0.2f, 0.0f}),
            index == RIGHT_THUMB ? 0.24f : 0.30f);
        avatar.target[index] = Vector3Add(avatar.target[RIGHT_WRIST], offset);
    }
    avatar.target[LEFT_HEEL] = Vector3Add(
        avatar.target[LEFT_ANKLE],
        Vector3Scale(DirectionOr(Vector3Subtract(raw[LEFT_HEEL], raw[LEFT_ANKLE]), {0.0f, -0.3f, 0.2f}), 0.22f));
    avatar.target[RIGHT_HEEL] = Vector3Add(
        avatar.target[RIGHT_ANKLE],
        Vector3Scale(DirectionOr(Vector3Subtract(raw[RIGHT_HEEL], raw[RIGHT_ANKLE]), {0.0f, -0.3f, 0.2f}), 0.22f));
    avatar.target[LEFT_FOOT] = Vector3Add(
        avatar.target[LEFT_ANKLE],
        Vector3Scale(DirectionOr(Vector3Subtract(raw[LEFT_FOOT], raw[LEFT_ANKLE]), {0.0f, -0.1f, -1.0f}), 0.48f));
    avatar.target[RIGHT_FOOT] = Vector3Add(
        avatar.target[RIGHT_ANKLE],
        Vector3Scale(DirectionOr(Vector3Subtract(raw[RIGHT_FOOT], raw[RIGHT_ANKLE]), {0.0f, -0.1f, -1.0f}), 0.48f));

    const Vector3 rawHeadCenter = Midpoint(raw[LEFT_EAR], raw[RIGHT_EAR]);
    const Vector3 headDirection = DirectionOr(
        Vector3Subtract(rawHeadCenter, rawShoulderMapped),
        {0.0f, 1.0f, 0.0f});
    const Vector3 headCenter = Vector3Add(shoulderCenter, Vector3Scale(headDirection, 0.88f));
    const Vector3 earAxis = DirectionOr(
        Vector3Subtract(raw[RIGHT_EAR], raw[LEFT_EAR]), shoulderAxis);
    avatar.target[LEFT_EAR] = Vector3Add(headCenter, Vector3Scale(earAxis, -0.25f));
    avatar.target[RIGHT_EAR] = Vector3Add(headCenter, Vector3Scale(earAxis, 0.25f));
    const Vector3 faceForward = DirectionOr(
        Vector3Subtract(raw[NOSE], rawHeadCenter), {0.0f, 0.0f, -1.0f});
    avatar.target[NOSE] = Vector3Add(headCenter, Vector3Scale(faceForward, 0.34f));

    const float ankleY = std::min(
        std::min(avatar.target[LEFT_ANKLE].y, avatar.target[RIGHT_ANKLE].y),
        std::min(avatar.target[LEFT_FOOT].y, avatar.target[RIGHT_FOOT].y));
    const float floorOffset = 0.25f - ankleY;
    for (Vector3& point : avatar.target) point.y += floorOffset;
}

void UpdateAvatar(AvatarState& avatar, float deltaTime, bool linked) {
    static std::array<Vector3, kLandmarkCount> idle{};
    static bool initialized = false;
    if (!initialized) {
        SetDefaultPose(idle);
        initialized = true;
    }
    if (!linked) avatar.target = idle;
    for (int i = 0; i < kLandmarkCount; ++i) {
        if (linked && avatar.visibility[i] < 0.25f) continue;
        const float error = Vector3Distance(avatar.current[i], avatar.target[i]);
        const float rate = linked ? std::clamp(8.0f + error * 12.0f, 8.0f, 26.0f) : 3.0f;
        const float response = 1.0f - std::exp(-deltaTime * rate);
        avatar.current[i] = Vector3Lerp(avatar.current[i], avatar.target[i], response);
    }
}

void DrawBone(Vector3 start, Vector3 end, float radius, Color color) {
    if (Vector3Distance(start, end) < 0.015f) return;
    DrawCylinderEx(start, end, radius, radius * 0.88f, 12, color);
    DrawSphere(start, radius * 1.06f, color);
    DrawSphere(end, radius * 1.03f, color);
}

void DrawAvatar(const AvatarState& avatar) {
    const auto& p = avatar.current;
    const Color skin = {224, 170, 132, 255};
    const Color shirt = {55, 132, 210, 255};
    const Color shirtDark = {35, 91, 151, 255};
    const Color pants = {44, 52, 66, 255};
    const Color shoes = {28, 31, 36, 255};
    const Color joint = {221, 228, 235, 255};
    const Color hair = {63, 42, 30, 255};

    const Vector3 hipCenter = Midpoint(p[LEFT_HIP], p[RIGHT_HIP]);
    const Vector3 shoulderCenter = Midpoint(p[LEFT_SHOULDER], p[RIGHT_SHOULDER]);
    const Vector3 headCenter = Vector3Add(Midpoint(p[LEFT_EAR], p[RIGHT_EAR]), {0.0f, 0.02f, 0.0f});
    const float headRadius = std::clamp(Vector3Distance(p[LEFT_EAR], p[RIGHT_EAR]) * 0.72f, 0.28f, 0.48f);
    const Vector3 faceForward = DirectionOr(Vector3Subtract(p[NOSE], headCenter), {0.0f, 0.0f, -1.0f});
    const Vector3 faceRight = DirectionOr(Vector3Subtract(p[RIGHT_EAR], p[LEFT_EAR]), {1.0f, 0.0f, 0.0f});

    // Feet and legs.
    DrawBone(p[LEFT_HIP], p[LEFT_KNEE], 0.25f, pants);
    DrawBone(p[RIGHT_HIP], p[RIGHT_KNEE], 0.25f, pants);
    DrawBone(p[LEFT_KNEE], p[LEFT_ANKLE], 0.22f, pants);
    DrawBone(p[RIGHT_KNEE], p[RIGHT_ANKLE], 0.22f, pants);
    DrawBone(p[LEFT_HEEL], p[LEFT_FOOT], 0.20f, shoes);
    DrawBone(p[RIGHT_HEEL], p[RIGHT_FOOT], 0.20f, shoes);

    // Pelvis, torso, and shoulders.
    DrawBone(p[LEFT_HIP], p[RIGHT_HIP], 0.28f, pants);
    DrawCylinderEx(hipCenter, shoulderCenter, 0.48f, 0.68f, 16, shirt);
    DrawBone(p[LEFT_SHOULDER], p[RIGHT_SHOULDER], 0.30f, shirt);
    DrawCylinderEx(
        Vector3Lerp(hipCenter, shoulderCenter, 0.18f),
        Vector3Lerp(hipCenter, shoulderCenter, 0.34f),
        0.50f, 0.54f, 16, shirtDark);

    // Arms and hands.
    DrawBone(p[LEFT_SHOULDER], p[LEFT_ELBOW], 0.21f, shirt);
    DrawBone(p[RIGHT_SHOULDER], p[RIGHT_ELBOW], 0.21f, shirt);
    DrawBone(p[LEFT_ELBOW], p[LEFT_WRIST], 0.17f, skin);
    DrawBone(p[RIGHT_ELBOW], p[RIGHT_WRIST], 0.17f, skin);
    DrawSphere(p[LEFT_WRIST], 0.22f, skin);
    DrawSphere(p[RIGHT_WRIST], 0.22f, skin);
    DrawBone(p[LEFT_WRIST], p[LEFT_INDEX], 0.055f, skin);
    DrawBone(p[LEFT_WRIST], p[LEFT_PINKY], 0.050f, skin);
    DrawBone(p[LEFT_WRIST], p[LEFT_THUMB], 0.060f, skin);
    DrawBone(p[RIGHT_WRIST], p[RIGHT_INDEX], 0.055f, skin);
    DrawBone(p[RIGHT_WRIST], p[RIGHT_PINKY], 0.050f, skin);
    DrawBone(p[RIGHT_WRIST], p[RIGHT_THUMB], 0.060f, skin);

    // Neck and head.
    DrawBone(shoulderCenter, Vector3Add(headCenter, {0.0f, -headRadius * 0.72f, 0.0f}), 0.18f, skin);
    DrawSphere(Vector3Add(headCenter, {0.0f, headRadius * 0.13f, headRadius * 0.05f}), headRadius * 1.02f, hair);
    const Vector3 faceCenter = Vector3Add(
        Vector3Add(headCenter, Vector3Scale(faceForward, headRadius * 0.08f)),
        {0.0f, -headRadius * 0.05f, 0.0f});
    DrawSphere(faceCenter, headRadius * 0.93f, skin);
    DrawSphere(p[LEFT_EAR], headRadius * 0.17f, skin);
    DrawSphere(p[RIGHT_EAR], headRadius * 0.17f, skin);

    const Vector3 eyeBase = Vector3Add(
        headCenter,
        Vector3Add(Vector3Scale(faceForward, headRadius * 0.86f), {0.0f, headRadius * 0.16f, 0.0f}));
    const Vector3 leftEye = Vector3Add(eyeBase, Vector3Scale(faceRight, -headRadius * 0.30f));
    const Vector3 rightEye = Vector3Add(eyeBase, Vector3Scale(faceRight, headRadius * 0.30f));
    DrawSphere(leftEye, headRadius * 0.095f, RAYWHITE);
    DrawSphere(rightEye, headRadius * 0.095f, RAYWHITE);
    DrawSphere(Vector3Add(leftEye, Vector3Scale(faceForward, headRadius * 0.065f)), headRadius * 0.043f, DARKBROWN);
    DrawSphere(Vector3Add(rightEye, Vector3Scale(faceForward, headRadius * 0.065f)), headRadius * 0.043f, DARKBROWN);
    DrawSphere(Vector3Lerp(headCenter, p[NOSE], 0.88f), headRadius * 0.095f, skin);
    const Vector3 mouthCenter = Vector3Add(
        headCenter,
        Vector3Add(Vector3Scale(faceForward, headRadius * 0.88f), {0.0f, -headRadius * 0.28f, 0.0f}));
    DrawLine3D(
        Vector3Add(mouthCenter, Vector3Scale(faceRight, -headRadius * 0.18f)),
        Vector3Add(mouthCenter, Vector3Scale(faceRight, headRadius * 0.18f)),
        {120, 58, 58, 255});

    // Small neutral joint accents make motion easier to read.
    for (int index : {LEFT_ELBOW, RIGHT_ELBOW, LEFT_KNEE, RIGHT_KNEE}) {
        DrawSphere(p[index], 0.11f, joint);
    }
}

void DrawWorld() {
    DrawPlane({0.0f, -0.02f, 0.0f}, {16.0f, 16.0f}, {28, 32, 37, 255});
    for (int i = -8; i <= 8; ++i) {
        const Color line = (i == 0) ? Color{80, 91, 101, 255} : Color{47, 53, 59, 255};
        DrawLine3D({static_cast<float>(i), 0.0f, -8.0f}, {static_cast<float>(i), 0.0f, 8.0f}, line);
        DrawLine3D({-8.0f, 0.0f, static_cast<float>(i)}, {8.0f, 0.0f, static_cast<float>(i)}, line);
    }
}

}  // namespace

int main() {
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE);
    InitWindow(kWindowWidth, kWindowHeight, "Pose Avatar Studio");
    SetTargetFPS(60);

    Camera3D camera{};
    camera.target = {0.0f, 2.7f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float cameraYaw = 0.68f;
    float cameraPitch = 0.22f;
    float cameraDistance = 9.8f;

    AvatarState avatar{};
    SetDefaultPose(avatar.current);
    SetDefaultPose(avatar.target);
    avatar.visibility.fill(1.0f);

    UdpPoseReceiver receiver;
    const bool receiverReady = receiver.Start(kPosePort);
    PosePacket packet{};
    double lastPacketTime = -100.0;
    int packetsThisFrame = 0;
    int packetTotal = 0;
    bool fullscreen = false;

    while (!WindowShouldClose()) {
        const float dt = std::min(GetFrameTime(), 0.1f);
        if (receiver.Poll(packet, packetsThisFrame)) {
            RetargetPose(packet, avatar);
            lastPacketTime = GetTime();
            packetTotal += packetsThisFrame;
        }
        const bool linked = receiverReady && (GetTime() - lastPacketTime) < kLinkTimeout;
        UpdateAvatar(avatar, dt, linked);

        if (IsKeyPressed(KEY_M)) avatar.mirror = !avatar.mirror;
        if (IsKeyPressed(KEY_F)) {
            fullscreen = !fullscreen;
            ToggleFullscreen();
        }
        if (IsKeyPressed(KEY_R)) {
            cameraYaw = 0.68f;
            cameraPitch = 0.22f;
            cameraDistance = 9.8f;
        }

        const Vector2 mouseDelta = GetMouseDelta();
        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
            cameraYaw -= mouseDelta.x * 0.006f;
            cameraPitch = std::clamp(cameraPitch + mouseDelta.y * 0.006f, -0.15f, 0.72f);
        }
        cameraDistance = std::clamp(cameraDistance - GetMouseWheelMove() * 0.65f, 5.5f, 16.0f);
        camera.position = {
            camera.target.x + std::sin(cameraYaw) * std::cos(cameraPitch) * cameraDistance,
            camera.target.y + std::sin(cameraPitch) * cameraDistance,
            camera.target.z + std::cos(cameraYaw) * std::cos(cameraPitch) * cameraDistance,
        };

        BeginDrawing();
        ClearBackground({17, 20, 23, 255});
        BeginMode3D(camera);
        DrawWorld();
        DrawAvatar(avatar);
        EndMode3D();

        const int height = GetScreenHeight();
        DrawRectangle(0, 0, kPanelWidth, height, {24, 28, 32, 244});
        DrawRectangle(0, 0, kPanelWidth, 92, {18, 21, 24, 255});
        DrawText("POSE AVATAR STUDIO", 24, 25, 22, {235, 240, 244, 255});
        DrawText("Procedural humanoid / UDP", 24, 57, 16, {143, 153, 162, 255});

        const Color statusColor = linked ? Color{94, 211, 126, 255} : Color{229, 174, 75, 255};
        DrawCircle(31, 126, 7.0f, statusColor);
        DrawText(linked ? "POSE STREAM CONNECTED" : "WAITING FOR TRACKER", 50, 117, 17, statusColor);
        DrawLine(24, 155, kPanelWidth - 24, 155, {62, 69, 75, 255});

        DrawText("SOURCE", 24, 185, 14, {143, 153, 162, 255});
        DrawText(TextFormat("UDP 127.0.0.1:%d", kPosePort), 24, 211, 18, RAYWHITE);
        DrawText("TRACKED PERSON", 24, 252, 14, {143, 153, 162, 255});
        DrawText(linked ? TextFormat("Person %d", packet.personIndex + 1) : "--", 24, 278, 18, RAYWHITE);
        DrawText("PACKETS RECEIVED", 24, 319, 14, {143, 153, 162, 255});
        DrawText(TextFormat("%d", packetTotal), 24, 345, 18, RAYWHITE);
        DrawText("RENDER RATE", 24, 386, 14, {143, 153, 162, 255});
        DrawText(TextFormat("%d fps", GetFPS()), 24, 412, 18, RAYWHITE);

        DrawLine(24, 455, kPanelWidth - 24, 455, {62, 69, 75, 255});
        DrawText("AVATAR CONTROLS", 24, 481, 14, {143, 153, 162, 255});
        DrawText("Right-drag   Orbit camera", 24, 510, 16, RAYWHITE);
        DrawText("Mouse wheel  Zoom", 24, 538, 16, RAYWHITE);
        DrawText("M            Mirror pose", 24, 566, 16, RAYWHITE);
        DrawText("R            Reset camera", 24, 594, 16, RAYWHITE);
        DrawText("F            Fullscreen", 24, 622, 16, RAYWHITE);
        DrawText("Esc          Quit", 24, 650, 16, RAYWHITE);

        DrawText("Python tracker streams only", 24, height - 76, 14, {143, 153, 162, 255});
        DrawText("3D landmarks over localhost.", 24, height - 54, 14, {143, 153, 162, 255});
        DrawText("No camera frames are sent.", 24, height - 32, 14, {143, 153, 162, 255});
        EndDrawing();
    }

    receiver.Close();
    CloseWindow();
    return 0;
}
