#include "raylib.h"
#include "raymath.h"
#include "../common/studio.h"
#include "../common/quantum_solver.h"
#include <algorithm>
#include <cmath>
#include <deque>

int main() {
    InitWindow(1360,860,"Quantum tunneling / Wave laboratory");
    SetTargetFPS(60);
    Camera3D camera{{13,9,15},{0,0,0},{0,1,0},43,CAMERA_PERSPECTIVE};
    float yaw=1.22f,pitch=0.60f,distance=27;
    double energy=0.8,barrier=1.15;
    float timeScale=1;
    bool paused=false,phase=true;
    physics::WavePacket packet;
    physics::Clock clock;
    packet.reset(energy,barrier);
    std::deque<float> transmitted;
    int samples=0;
    while (!WindowShouldClose()) {
        bool restart=IsKeyPressed(KEY_R);
        if (IsKeyPressed(KEY_P) || IsKeyPressed(KEY_SPACE)) paused=!paused;
        if (IsKeyPressed(KEY_V)) phase=!phase;
        if (IsKeyPressed(KEY_LEFT_BRACKET)) { barrier=std::max(0.0,barrier-0.1); restart=true; }
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) { barrier=std::min(4.0,barrier+0.1); restart=true; }
        if (IsKeyPressed(KEY_MINUS)) { energy=std::max(0.2,energy-0.1); restart=true; }
        if (IsKeyPressed(KEY_EQUAL)) { energy=std::min(3.0,energy+0.1); restart=true; }
        if (IsKeyPressed(KEY_COMMA)) timeScale=std::max(0.25f,timeScale/2);
        if (IsKeyPressed(KEY_PERIOD)) timeScale=std::min(2.0f,timeScale*2);
        if (restart) { packet.reset(energy,barrier); clock.reset(); transmitted.clear(); samples=0; paused=false; }
        studio::pan(&camera,yaw,pitch,distance);
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) && !studio::panGesture()) {
            Vector2 d=GetMouseDelta(); yaw-=d.x*0.004f;
            pitch=std::clamp(pitch+d.y*0.004f,-1.35f,1.35f);
        }
        distance=std::clamp(distance*std::exp(-GetMouseWheelMove()*0.06f),7.0f,60.0f);
        camera.position=Vector3Add(camera.target,{distance*std::cos(pitch)*std::cos(yaw),distance*std::sin(pitch),distance*std::cos(pitch)*std::sin(yaw)});
        if (!paused) clock.advance(std::min(GetFrameTime(),0.05f)*timeScale,[&](double) {
            packet.step(); packet.step();
            if (++samples%8==0) {
                transmitted.push_back(float(packet.regions()[2]));
                if (transmitted.size()>360) transmitted.pop_front();
            }
        });
        auto probabilities=packet.regions();
        BeginDrawing();
        ClearBackground({14,16,28,255});
        BeginMode3D(camera);
        DrawLine3D({-12,0,0},{12,0,0},Color{131,139,162,180});
        DrawCube({0,float(barrier)*0.5f,-0.65f},1,float(barrier),1.3f,Color{181,135,189,100});
        DrawCubeWires({0,float(barrier)*0.5f,-0.65f},1,float(barrier),1.3f,Color{206,163,222,220});
        for (int i=1;i<physics::WavePacket::count;++i) {
            float x0=packet.x(i-1),x1=packet.x(i);
            float density0=std::norm(packet.psi[i-1]),density1=std::norm(packet.psi[i]);
            Color color=x1<-0.5f ? Color{174,181,247,255} : (x1>0.5f ? Color{117,221,193,255} : Color{239,203,135,255});
            DrawLine3D({x0,5*density0,0},{x1,5*density1,0},color);
            DrawLine3D({x1,0,0},{x1,5*density1,0},Fade(color,0.22f));
            if (phase) {
                DrawLine3D({x0,float(packet.psi[i-1].real()),-2},{x1,float(packet.psi[i].real()),-2},Color{120,185,235,210});
                DrawLine3D({x0,float(packet.psi[i-1].imag()),-3},{x1,float(packet.psi[i].imag()),-3},Color{226,156,192,210});
            }
        }
        for (int i=-12;i<=12;i+=2) DrawLine3D({float(i),-0.07f,0.1f},{float(i),-0.07f,-0.1f},Color{130,144,166,220});
        EndMode3D();
        studio::title("Quantum tunneling / an evolving wavefunction",studio::Style::Quantum);
        studio::readout(TextFormat("Carrier E %.2f   Barrier %.2f   t %.2f   Norm %.8f%s",energy,barrier,packet.time,packet.norm(),paused ? "   PAUSED" : ""));
        studio::note("hbar = m = 1 / density height 5x / cyan real, pink imaginary / reflecting domain endpoints");
        studio::help("Left drag orbit | wheel zoom | [ ] barrier | -/+ carrier energy (restarts) | V phase | , . speed | Space/P pause | R restart");
        const char* labels[]={"LEFT OF BARRIER","INSIDE BARRIER","RIGHT OF BARRIER"};
        const Color colors[]={{174,181,247,255},{239,203,135,255},{117,221,193,255}};
        for (int i=0;i<3;++i) {
            Rectangle r{26.0f+i*185,180,170,83};
            studio::panel(r,Color{27,29,44,245});
            studio::text(labels[i],r.x+13,r.y+12,11,colors[i]);
            studio::text(TextFormat("%.1f %%",100*probabilities[i]),r.x+13,r.y+33,28,WHITE);
        }
        studio::plot({float(GetScreenWidth()-408),float(GetScreenHeight()-283),380,215},
                     "Right-side probability / last 12 s",transmitted,0,1,colors[2]);
        studio::fps();
        EndDrawing();
        if (studio::smokeFrame(__FILE__)) break;
    }
    studio::unload();
    CloseWindow();
    return 0;
}
