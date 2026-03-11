#include "raylib.h"
#include "raymath.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
#else
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace {

constexpr int kWindowW = 1400;
constexpr int kWindowH = 900;
constexpr int kUdpPort = 50506;
constexpr float kLinkTimeout = 0.75f;
constexpr float kPoseSmooth = 10.0f;

struct TrackingInput {
    bool valid = false;
    float centerX = 0.5f;
    float centerY = 0.6f;
    float torsoH = 0.28f;
    float shoulderSpan = 0.16f;
    float headX = 0.5f;
    float headY = 0.22f;
    float leftHandX = 0.42f;
    float leftHandY = 0.48f;
    float rightHandX = 0.58f;
    float rightHandY = 0.48f;
    float leftFootX = 0.46f;
    float leftFootY = 0.92f;
    float rightFootX = 0.54f;
    float rightFootY = 0.92f;
    bool crouch = false;
    bool armsUp = false;
    int lean = 0;
    double timestamp = 0.0;
};

struct BodyPose {
    Vector3 head{};
    Vector3 chest{};
    Vector3 pelvis{};
    Vector3 leftShoulder{};
    Vector3 rightShoulder{};
    Vector3 leftElbow{};
    Vector3 rightElbow{};
    Vector3 leftHand{};
    Vector3 rightHand{};
    Vector3 leftHip{};
    Vector3 rightHip{};
    Vector3 leftKnee{};
    Vector3 rightKnee{};
    Vector3 leftFoot{};
    Vector3 rightFoot{};
    bool crouch = false;
    bool armsUp = false;
    int lean = 0;
    bool valid = false;
};

class UdpBridgeReceiver {
  public:
    bool Start(uint16_t port) {
#ifdef _WIN32
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) return false;
#endif
        socket_ = static_cast<int>(socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP));
        if (socket_ < 0) return false;

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        addr.sin_port = htons(port);
        if (bind(socket_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            Close();
            return false;
        }

#ifdef _WIN32
        u_long mode = 1;
        if (ioctlsocket(socket_, FIONBIO, &mode) != 0) {
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

    void Close() {
        if (socket_ >= 0) {
#ifdef _WIN32
            closesocket(socket_);
            WSACleanup();
#else
            close(socket_);
#endif
            socket_ = -1;
        }
        ready_ = false;
    }

    bool Poll(TrackingInput& outInput, int& packetsRead) {
        packetsRead = 0;
        if (!ready_) return false;

        bool gotAny = false;
        std::array<char, 512> buffer{};
        while (true) {
            sockaddr_in src{};
            socklen_t srcLen = sizeof(src);
            const int n = recvfrom(socket_, buffer.data(), static_cast<int>(buffer.size()) - 1, 0,
                                   reinterpret_cast<sockaddr*>(&src), &srcLen);
            if (n <= 0) {
#ifdef _WIN32
                if (WSAGetLastError() == WSAEWOULDBLOCK) break;
#else
                if (errno == EWOULDBLOCK || errno == EAGAIN) break;
#endif
                break;
            }

            buffer[static_cast<size_t>(n)] = '\0';
            TrackingInput parsed{};
            int valid = 0;
            int crouch = 0;
            int armsUp = 0;
            int lean = 0;
            const int got = std::sscanf(
                buffer.data(),
                "%lf,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d,%d,%d",
                &parsed.timestamp,
                &valid,
                &parsed.centerX,
                &parsed.centerY,
                &parsed.torsoH,
                &parsed.shoulderSpan,
                &parsed.headX,
                &parsed.headY,
                &parsed.leftHandX,
                &parsed.leftHandY,
                &parsed.rightHandX,
                &parsed.rightHandY,
                &parsed.leftFootX,
                &parsed.leftFootY,
                &parsed.rightFootX,
                &parsed.rightFootY,
                &crouch,
                &armsUp,
                &lean
            );
            if (got == 19) {
                parsed.valid = (valid != 0);
                parsed.crouch = (crouch != 0);
                parsed.armsUp = (armsUp != 0);
                parsed.lean = std::clamp(lean, -1, 1);
                outInput = parsed;
                gotAny = true;
                packetsRead++;
            }
        }

        return gotAny;
    }

    bool ready() const { return ready_; }

  private:
    int socket_ = -1;
    bool ready_ = false;
};

Vector3 ScreenNormToWorld(float x, float y, float z) {
    const float worldX = Lerp(-7.5f, 7.5f, x);
    const float worldY = Lerp(8.8f, 0.6f, y);
    return {worldX, worldY, z};
}

BodyPose MakeBodyPose(const TrackingInput& in) {
    BodyPose out{};
    out.valid = in.valid;
    out.crouch = in.crouch;
    out.armsUp = in.armsUp;
    out.lean = in.lean;

    const Vector3 chest = ScreenNormToWorld(in.centerX, in.centerY - in.torsoH * 0.15f, 0.0f);
    const Vector3 pelvis = ScreenNormToWorld(in.centerX, in.centerY + in.torsoH * 0.16f, 0.0f);
    const float shoulderHalf = std::max(0.65f, in.shoulderSpan * 6.2f);
    const float hipHalf = shoulderHalf * 0.52f;
    const float sideZ = 0.75f;

    out.chest = chest;
    out.pelvis = pelvis;
    out.head = ScreenNormToWorld(in.headX, in.headY, 0.0f);
    out.leftShoulder = Vector3Add(chest, {-shoulderHalf, 0.22f, -sideZ});
    out.rightShoulder = Vector3Add(chest, {shoulderHalf, 0.22f, sideZ});
    out.leftHip = Vector3Add(pelvis, {-hipHalf, -0.12f, -sideZ * 0.55f});
    out.rightHip = Vector3Add(pelvis, {hipHalf, -0.12f, sideZ * 0.55f});
    out.leftHand = ScreenNormToWorld(in.leftHandX, in.leftHandY, -sideZ * 1.45f);
    out.rightHand = ScreenNormToWorld(in.rightHandX, in.rightHandY, sideZ * 1.45f);
    out.leftFoot = ScreenNormToWorld(in.leftFootX, in.leftFootY, -sideZ * 0.75f);
    out.rightFoot = ScreenNormToWorld(in.rightFootX, in.rightFootY, sideZ * 0.75f);

    out.leftElbow = Vector3Add(Vector3Scale(Vector3Add(out.leftShoulder, out.leftHand), 0.5f), {-0.22f, 0.35f, -0.18f});
    out.rightElbow = Vector3Add(Vector3Scale(Vector3Add(out.rightShoulder, out.rightHand), 0.5f), {0.22f, 0.35f, 0.18f});
    out.leftKnee = Vector3Add(Vector3Scale(Vector3Add(out.leftHip, out.leftFoot), 0.5f), {-0.12f, 0.45f, -0.06f});
    out.rightKnee = Vector3Add(Vector3Scale(Vector3Add(out.rightHip, out.rightFoot), 0.5f), {0.12f, 0.45f, 0.06f});

    return out;
}

Vector3 SmoothVec(Vector3 a, Vector3 b, float blend) {
    return Vector3Lerp(a, b, blend);
}

void SmoothBodyPose(BodyPose* curr, const BodyPose& target, float blend) {
    curr->valid = target.valid;
    curr->crouch = target.crouch;
    curr->armsUp = target.armsUp;
    curr->lean = target.lean;
    curr->head = SmoothVec(curr->head, target.head, blend);
    curr->chest = SmoothVec(curr->chest, target.chest, blend);
    curr->pelvis = SmoothVec(curr->pelvis, target.pelvis, blend);
    curr->leftShoulder = SmoothVec(curr->leftShoulder, target.leftShoulder, blend);
    curr->rightShoulder = SmoothVec(curr->rightShoulder, target.rightShoulder, blend);
    curr->leftElbow = SmoothVec(curr->leftElbow, target.leftElbow, blend);
    curr->rightElbow = SmoothVec(curr->rightElbow, target.rightElbow, blend);
    curr->leftHand = SmoothVec(curr->leftHand, target.leftHand, blend);
    curr->rightHand = SmoothVec(curr->rightHand, target.rightHand, blend);
    curr->leftHip = SmoothVec(curr->leftHip, target.leftHip, blend);
    curr->rightHip = SmoothVec(curr->rightHip, target.rightHip, blend);
    curr->leftKnee = SmoothVec(curr->leftKnee, target.leftKnee, blend);
    curr->rightKnee = SmoothVec(curr->rightKnee, target.rightKnee, blend);
    curr->leftFoot = SmoothVec(curr->leftFoot, target.leftFoot, blend);
    curr->rightFoot = SmoothVec(curr->rightFoot, target.rightFoot, blend);
}

void DrawLink(Vector3 a, Vector3 b, float r, Color core, Color glow) {
    DrawCylinderEx(a, b, r * 1.55f, r * 1.25f, 10, Fade(glow, 0.14f));
    DrawCylinderEx(a, b, r, r * 0.9f, 10, core);
}

void DrawJoint(Vector3 p, float r, Color core, Color glow) {
    DrawSphere(p, r * 1.55f, Fade(glow, 0.12f));
    DrawSphere(p, r, core);
}

void DrawExosuit(const BodyPose& pose) {
    if (!pose.valid) return;

    const Color frame = pose.armsUp ? Color{126, 255, 198, 255} : Color{116, 216, 255, 255};
    const Color glow = pose.crouch ? Color{255, 202, 110, 255} : Color{120, 180, 255, 255};
    const Color armor = Color{70, 80, 96, 255};
    const Color visor = pose.lean == 0 ? Color{126, 250, 224, 255} : Color{255, 184, 118, 255};

    DrawLink(pose.leftShoulder, pose.rightShoulder, 0.12f, frame, glow);
    DrawLink(pose.leftHip, pose.rightHip, 0.11f, frame, glow);
    DrawLink(pose.chest, pose.pelvis, 0.16f, armor, glow);
    DrawCubeV(Vector3Scale(Vector3Add(pose.chest, pose.pelvis), 0.5f), {1.15f, 1.55f, 1.00f}, Fade(armor, 0.95f));
    DrawCubeWiresV(Vector3Scale(Vector3Add(pose.chest, pose.pelvis), 0.5f), {1.15f, 1.55f, 1.00f}, Fade(frame, 0.36f));

    DrawLink(pose.leftShoulder, pose.leftElbow, 0.09f, frame, glow);
    DrawLink(pose.leftElbow, pose.leftHand, 0.08f, frame, glow);
    DrawLink(pose.rightShoulder, pose.rightElbow, 0.09f, frame, glow);
    DrawLink(pose.rightElbow, pose.rightHand, 0.08f, frame, glow);
    DrawLink(pose.leftHip, pose.leftKnee, 0.10f, frame, glow);
    DrawLink(pose.leftKnee, pose.leftFoot, 0.09f, frame, glow);
    DrawLink(pose.rightHip, pose.rightKnee, 0.10f, frame, glow);
    DrawLink(pose.rightKnee, pose.rightFoot, 0.09f, frame, glow);

    DrawJoint(pose.head, 0.35f, armor, glow);
    DrawSphereWires(pose.head, 0.37f, 10, 10, Fade(frame, 0.38f));
    DrawCube({pose.head.x, pose.head.y - 0.02f, pose.head.z + 0.23f}, 0.32f, 0.08f, 0.04f, visor);

    for (Vector3 p : {pose.leftShoulder, pose.rightShoulder, pose.leftElbow, pose.rightElbow, pose.leftHand, pose.rightHand,
                      pose.leftHip, pose.rightHip, pose.leftKnee, pose.rightKnee, pose.leftFoot, pose.rightFoot}) {
        DrawJoint(p, 0.11f, frame, glow);
    }

    DrawCube({pose.leftFoot.x, pose.leftFoot.y - 0.09f, pose.leftFoot.z}, 0.42f, 0.08f, 0.24f, armor);
    DrawCube({pose.rightFoot.x, pose.rightFoot.y - 0.09f, pose.rightFoot.z}, 0.42f, 0.08f, 0.24f, armor);
}

}  // namespace

int main() {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(kWindowW, kWindowH, "Body Exosuit Scene (UDP Bridge)");
    SetTargetFPS(120);
    SetWindowMinSize(1040, 680);

    Camera3D cam{};
    cam.position = {0.0f, 7.0f, 16.0f};
    cam.target = {0.0f, 3.6f, 0.0f};
    cam.up = {0.0f, 1.0f, 0.0f};
    cam.fovy = 46.0f;
    cam.projection = CAMERA_PERSPECTIVE;

    UdpBridgeReceiver receiver;
    const bool receiverOk = receiver.Start(static_cast<uint16_t>(kUdpPort));

    TrackingInput tracking{};
    float lastPacketWallClock = -100.0f;
    BodyPose pose{};
    BodyPose targetPose = MakeBodyPose(tracking);
    pose = targetPose;

    while (!WindowShouldClose()) {
        const float dt = std::max(1.0e-4f, GetFrameTime());
        const float now = static_cast<float>(GetTime());

        int packetsRead = 0;
        if (receiver.Poll(tracking, packetsRead)) {
            lastPacketWallClock = now;
        }
        const bool linkLive = receiver.ready() && ((now - lastPacketWallClock) < kLinkTimeout);
        tracking.valid = tracking.valid && linkLive;

        targetPose = MakeBodyPose(tracking);
        const float blend = 1.0f - std::exp(-kPoseSmooth * dt);
        SmoothBodyPose(&pose, targetPose, blend);

        cam.target = Vector3Lerp(cam.target, {pose.chest.x, 3.6f, 0.0f}, 0.06f);
        cam.position = Vector3Lerp(cam.position, {pose.chest.x * 0.15f, 7.0f, 16.0f}, 0.035f);

        BeginDrawing();
        ClearBackground(Color{8, 11, 18, 255});
        BeginMode3D(cam);

        DrawPlane({0.0f, 0.0f, 0.0f}, {32.0f, 24.0f}, Color{22, 28, 40, 255});
        for (int i = -12; i <= 12; ++i) {
            DrawLine3D({static_cast<float>(i), 0.01f, -12.0f}, {static_cast<float>(i), 0.01f, 12.0f}, Color{38, 48, 70, 110});
        }
        for (int j = -10; j <= 10; ++j) {
            DrawLine3D({-12.0f, 0.01f, static_cast<float>(j)}, {12.0f, 0.01f, static_cast<float>(j)}, Color{38, 48, 70, 110});
        }

        DrawExosuit(pose);

        if (pose.valid) {
            DrawCylinder({pose.chest.x, 0.6f, 0.0f}, 0.14f, 0.06f, 1.2f, 12, Fade(Color{120, 240, 220, 255}, 0.22f));
        }

        EndMode3D();

        DrawRectangle(12, 12, 520, 90, Fade(BLACK, 0.28f));
        DrawText("Body Exosuit Scene", 24, 24, 30, Color{232, 239, 250, 255});
        DrawText("Drive the mech with the Python body bridge. Raise arms, crouch, and lean to see it react.", 24, 58, 18, Color{172, 190, 218, 255});
        DrawText(
            TextFormat("%s udp:%d  body:%s  crouch:%s  arms_up:%s  lean:%d",
                       receiverOk ? "bridge:on" : "bridge:failed",
                       kUdpPort,
                       pose.valid ? "tracked" : "none",
                       pose.crouch ? "yes" : "no",
                       pose.armsUp ? "yes" : "no",
                       pose.lean),
            24,
            82,
            18,
            Color{124, 220, 255, 255}
        );
        DrawFPS(GetScreenWidth() - 96, 18);
        EndDrawing();
    }

    receiver.Close();
    CloseWindow();
    return 0;
}
