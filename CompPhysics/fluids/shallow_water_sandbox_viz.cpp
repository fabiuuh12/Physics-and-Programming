#include "raylib.h"
#include "rlgl.h"
#include "../common/studio.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
constexpr int NX = 64, NZ = 48;
constexpr double DX = 0.25, G = 9.81;
constexpr float LX = NX * DX, LZ = NZ * DX;
const Color ink{19, 35, 40, 255}, muted{103, 121, 117, 255};

struct State {
    double h = 1.0, x = 0.0, z = 0.0; // depth and depth-integrated momenta
};
State operator+(State a, State b) { return {a.h+b.h, a.x+b.x, a.z+b.z}; }
State operator-(State a, State b) { return {a.h-b.h, a.x-b.x, a.z-b.z}; }
State operator*(State a, double b) { return {a.h*b, a.x*b, a.z*b}; }

// First-order finite volumes, local Lax-Friedrichs (Rusanov) flux.
// Flat bed, wet cells, reflecting walls, no viscosity or external forcing.
State flux(State q, bool alongX) {
    double velocity = (alongX ? q.x : q.z) / q.h;
    double pressure = 0.5 * G * q.h * q.h;
    return {alongX ? q.x : q.z,
            q.x * velocity + (alongX ? pressure : 0.0),
            q.z * velocity + (alongX ? 0.0 : pressure)};
}
State reflected(State q, bool alongX) {
    if (alongX) q.x = -q.x; else q.z = -q.z;
    return q;
}
State interfaceFlux(State a, State b, bool alongX) {
    double sa = std::abs((alongX ? a.x : a.z) / a.h) + std::sqrt(G*a.h);
    double sb = std::abs((alongX ? b.x : b.z) / b.h) + std::sqrt(G*b.h);
    return (flux(a, alongX) + flux(b, alongX)) * 0.5
         - (b-a) * (0.5 * std::max(sa, sb));
}

struct Basin {
    std::vector<State> q = std::vector<State>(NX*NZ);
    std::vector<State> delta = std::vector<State>(NX*NZ);
    std::vector<unsigned char> solid = std::vector<unsigned char>(NX*NZ, 0);
    double elapsed = 0, initialVolume = 0;
    int preset = 1;
    static int index(int x, int z) { return z*NX+x; }
    bool wet(int x, int z) const {
        return x >= 0 && x < NX && z >= 0 && z < NZ && !solid[index(x,z)];
    }
    double volume() const {
        double sum = 0;
        for (int i=0; i<NX*NZ; ++i) if (!solid[i]) sum += q[i].h * DX*DX;
        return sum;
    }
    void reset(int choice) {
        preset = choice;
        elapsed = 0;
        std::fill(q.begin(), q.end(), State{});
        std::fill(solid.begin(), solid.end(), 0);
        for (int z=0; z<NZ; ++z) for (int x=0; x<NX; ++x) {
            int i = index(x,z);
            if (choice == 2) q[i].h = x < NX/2 ? 1.6 : 0.4;
            if (choice == 3) {
                solid[i] = (x == NX/2 && std::abs(z-NZ/2) > 4);
                q[i].h = x < NX/2 ? 1.5 : 0.7;
            }
        }
        if (choice == 1) disturb(-2.5f, 0.0f);
        initialVolume = volume();
    }
    // Redistribute existing water: a localized mound plus uniform withdrawal.
    // No mass is injected by clicking. Limit withdrawal to keep all cells wet.
    void disturb(float wx, float wz) {
        double sum=0, minimum=1e30;
        int count=0;
        std::vector<double> bump(NX*NZ, 0.0);
        for (int z=0; z<NZ; ++z) for (int x=0; x<NX; ++x) {
            int i=index(x,z);
            if (solid[i]) continue;
            double rx=(x+0.5)*DX-LX/2-wx, rz=(z+0.5)*DX-LZ/2-wz;
            bump[i]=std::exp(-(rx*rx+rz*rz)/0.45);
            sum += bump[i];
            minimum=std::min(minimum,q[i].h);
            ++count;
        }
        double mean=sum/count;
        double amplitude=std::min(0.65, 0.2*minimum/std::max(mean,1e-12));
        for (int i=0; i<NX*NZ; ++i) if (!solid[i]) q[i].h += amplitude*(bump[i]-mean);
    }
    double stableStep() const {
        double maxX=0, maxZ=0;
        for (int i=0; i<NX*NZ; ++i) if (!solid[i]) {
            double c=std::sqrt(G*q[i].h);
            maxX=std::max(maxX,std::abs(q[i].x/q[i].h)+c);
            maxZ=std::max(maxZ,std::abs(q[i].z/q[i].h)+c);
        }
        return 0.4*DX/(maxX+maxZ);
    }
    void step(double dt) {
        std::fill(delta.begin(), delta.end(), State{0,0,0});
        // Each interior face is evaluated once, with equal/opposite transfers.
        for (int axis=0; axis<2; ++axis) {
            bool alongX=axis==0;
            int rows=alongX ? NZ : NX, cols=alongX ? NX : NZ;
            for (int row=0; row<rows; ++row) for (int face=0; face<=cols; ++face) {
                int ax=alongX ? face-1 : row, az=alongX ? row : face-1;
                int bx=alongX ? face : row, bz=alongX ? row : face;
                bool wa=wet(ax,az), wb=wet(bx,bz);
                if (!wa && !wb) continue;
                State a=wa ? q[index(ax,az)] : reflected(q[index(bx,bz)],alongX);
                State b=wb ? q[index(bx,bz)] : reflected(q[index(ax,az)],alongX);
                State f=interfaceFlux(a,b,alongX)*(dt/DX);
                if (wa) delta[index(ax,az)]=delta[index(ax,az)]-f;
                if (wb) delta[index(bx,bz)]=delta[index(bx,bz)]+f;
            }
        }
        for (int i=0; i<NX*NZ; ++i) if (!solid[i]) q[i]=q[i]+delta[i];
        elapsed += dt;
    }
    bool healthy() const {
        for (int i=0; i<NX*NZ; ++i) if (!solid[i] &&
            (!(q[i].h>0) || !std::isfinite(q[i].h) ||
             !std::isfinite(q[i].x) || !std::isfinite(q[i].z))) return false;
        return true;
    }
};

int selfTest() {
    bool passed=true;
    for (int preset=0; preset<=3; ++preset) {
        Basin b;
        b.reset(preset);
        double worstMassError=0;
        bool healthy=true;
        for (int n=0; n<1200; ++n) {
            if (n%100==0 && preset==1) b.disturb(2.0f,-1.0f);
            b.step(b.stableStep());
            if (!b.healthy()) { healthy=false; break; }
            worstMassError=std::max(worstMassError,std::abs(b.volume()/b.initialVolume-1));
        }
        double restError=0, symmetryError=0;
        for (int z=0; z<NZ; ++z) for (int x=0; x<NX; ++x) {
            State q=b.q[Basin::index(x,z)];
            if (preset==0) restError=std::max(restError,std::abs(q.h-1)+std::abs(q.x)+std::abs(q.z));
            if (preset==2) symmetryError=std::max(symmetryError,
                std::abs(q.h-b.q[Basin::index(x,NZ-1-z)].h)+std::abs(q.z));
        }
        bool evolved=preset!=2 || b.q[Basin::index(NX/2,NZ/2)].h>0.5;
        bool ok=healthy && evolved && worstMassError<1e-10 && restError<1e-12 && symmetryError<1e-12;
        std::printf("%s preset %d: volume error %.3e, rest error %.3e, symmetry error %.3e\n",
                    ok ? "PASS" : "FAIL",preset,worstMassError,restError,symmetryError);
        passed &= ok;
    }
    return passed ? 0 : 1;
}

Color waterColor(double value, bool speed) {
    float t=std::clamp(static_cast<float>(speed ? value/2.5 : (value-0.35)/1.35),0.0f,1.0f);
    Color low{18,58,112,255}, mid{30,173,198,255}, high{244,220,151,255};
    Color a=t<0.5f ? low : mid, b=t<0.5f ? mid : high;
    float s=t<0.5f ? 2*t : 2*t-1;
    return {static_cast<unsigned char>(a.r+(b.r-a.r)*s),
            static_cast<unsigned char>(a.g+(b.g-a.g)*s),
            static_cast<unsigned char>(a.b+(b.b-a.b)*s),255};
}
float surfaceHeight(double h) { return static_cast<float>(0.25+(h-1.0)*1.5); }
void vertex(float x, float y, float z, Color c) {
    rlColor4ub(c.r,c.g,c.b,c.a); rlVertex3f(x,y,z);
}
void drawWater(const Basin& basin, bool speed) {
    // Shared corner heights produce a continuous mesh without allocating a GPU model.
    std::vector<float> heights((NX+1)*(NZ+1));
    for (int z=0; z<=NZ; ++z) for (int x=0; x<=NX; ++x) {
        double h=0; int count=0;
        for (int dz=-1; dz<=0; ++dz) for (int dx=-1; dx<=0; ++dx)
            if (basin.wet(x+dx,z+dz)) { h+=basin.q[Basin::index(x+dx,z+dz)].h; ++count; }
        heights[z*(NX+1)+x]=surfaceHeight(count ? h/count : 1);
    }
    rlDisableBackfaceCulling();
    rlBegin(RL_TRIANGLES);
    for (int z=0; z<NZ; ++z) for (int x=0; x<NX; ++x) {
        int i=Basin::index(x,z);
        if (basin.solid[i]) continue;
        State q=basin.q[i];
        Color c=waterColor(speed ? std::hypot(q.x,q.z)/q.h : q.h,speed);
        float xx=x*DX-LX/2, zz=z*DX-LZ/2, d=DX;
        float a=heights[z*(NX+1)+x], b=heights[z*(NX+1)+x+1];
        float cc=heights[(z+1)*(NX+1)+x], e=heights[(z+1)*(NX+1)+x+1];
        vertex(xx,a,zz,c); vertex(xx,cc,zz+d,c); vertex(xx+d,b,zz,c);
        vertex(xx+d,b,zz,c); vertex(xx,cc,zz+d,c); vertex(xx+d,e,zz+d,c);
    }
    rlEnd();
    rlEnableBackfaceCulling();
}
} // namespace

int main(int argc, char** argv) {
    if (argc>1 && std::strcmp(argv[1],"--self-test")==0) return selfTest();
    bool smoke=argc>1 && std::strcmp(argv[1],"--smoke-test")==0;
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE);
    InitWindow(1360,900,"Shallow Water | Interactive Fluid Lab");
    SetWindowMinSize(1050,720);
    SetTargetFPS(60);
    Camera3D camera{};
    camera.target={0,0,0}; camera.up={0,1,0}; camera.fovy=43;
    camera.projection=CAMERA_PERSPECTIVE;
    float yaw=0.78f, pitch=0.82f, distance=29;
    Basin basin; basin.reset(1);
    bool paused=false, speedColors=false, arrows=false, healthy=true, showHelp=false;
    float timeScale=1;
    int frames=0;
    const char* names[]={"Still water", "Ripple basin", "Dam break", "Narrow gate"};

    while (!WindowShouldClose()) {
        int width=GetScreenWidth(), height=GetScreenHeight();
        const Color paper{240,241,231,255}, dark{35,56,51,255}, teal{39,117,102,255};
        Rectangle sidebar{22,22,246,570};
        Rectangle transport{float(width)/2-170,float(height)-70,340,48};
        Rectangle pauseButton{transport.x+8,transport.y+6,95,36};
        Rectangle resetButton{transport.x+110,transport.y+6,95,36};
        Rectangle helpButton{transport.x+212,transport.y+6,120,36};
        Rectangle colorButton{38,410,103,35}, vectorButton{149,410,103,35};
        bool onUI=studio::over(sidebar) || studio::over(transport) ||
                  studio::over({float(width)-224,float(height)-196,202,108}) ||
                  (showHelp && studio::over({float(width)-368,92,346,278}));
        if (IsKeyPressed(KEY_SPACE) || studio::clicked(pauseButton)) paused=!paused;
        if (IsKeyPressed(KEY_C) || studio::clicked(colorButton)) speedColors=!speedColors;
        if (IsKeyPressed(KEY_V) || studio::clicked(vectorButton)) arrows=!arrows;
        if (IsKeyPressed(KEY_H) || studio::clicked(helpButton)) showHelp=!showHelp;
        if (IsKeyPressed(KEY_MINUS)) timeScale=std::max(0.25f,timeScale/2);
        if (IsKeyPressed(KEY_EQUAL)) timeScale=std::min(2.0f,timeScale*2);
        int reset=-1;
        if (IsKeyPressed(KEY_ONE)) reset=1;
        if (IsKeyPressed(KEY_TWO)) reset=2;
        if (IsKeyPressed(KEY_THREE)) reset=3;
        if (IsKeyPressed(KEY_ZERO)) reset=0;
        if (IsKeyPressed(KEY_R) || studio::clicked(resetButton)) reset=basin.preset;
        for (int i=0;i<4;++i) if (studio::clicked({38,170.0f+i*49,214,41})) reset=i;
        if (reset>=0) { basin.reset(reset); healthy=true; paused=false; }
        if (IsKeyPressed(KEY_F)) { camera.target={0,0,0}; yaw=0.78f; pitch=0.82f; distance=29; }
        if (IsKeyPressed(KEY_T)) pitch=1.5f;
        if (!onUI && IsMouseButtonDown(MOUSE_BUTTON_RIGHT) && !IsKeyDown(KEY_LEFT_SHIFT)) {
            Vector2 d=GetMouseDelta();
            yaw-=d.x*0.005f; pitch=std::clamp(pitch+d.y*0.005f,-1.45f,1.5f);
        }
        if (!onUI) {
            studio::pan(&camera,yaw,pitch,distance);
            if (IsKeyDown(KEY_LEFT_SHIFT) && IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
                Vector2 d=GetMouseDelta();
                float s=distance*0.0012f;
                camera.target.x-=s*(d.x*std::sin(yaw)+d.y*std::cos(yaw));
                camera.target.z+=s*(d.x*std::cos(yaw)-d.y*std::sin(yaw));
            }
            distance=std::clamp(distance*std::exp(-GetMouseWheelMove()*0.08f),3.0f,65.0f);
        }
        float move=std::min(GetFrameTime(),0.05f)*(IsKeyDown(KEY_LEFT_SHIFT) ? 16.0f : 6.0f);
        float forward=float(IsKeyDown(KEY_W))-float(IsKeyDown(KEY_S));
        float right=float(IsKeyDown(KEY_D))-float(IsKeyDown(KEY_A));
        camera.target.x+=move*(-forward*std::cos(yaw)+right*std::sin(yaw));
        camera.target.z+=move*(-forward*std::sin(yaw)-right*std::cos(yaw));
        camera.target.y+=move*(float(IsKeyDown(KEY_E))-float(IsKeyDown(KEY_Q)));
        camera.position={camera.target.x+distance*std::cos(pitch)*std::cos(yaw),
                         camera.target.y+distance*std::sin(pitch),
                         camera.target.z+distance*std::cos(pitch)*std::sin(yaw)};
        // Pick against cell-height boxes to follow the displaced water surface.
        Ray ray=GetScreenToWorldRay(GetMousePosition(),camera);
        bool hover=false;
        Vector3 hit{};
        float nearest=1e30f;
        if (!onUI && !studio::panGesture()) {
            for (int z=0; z<NZ; ++z) for (int x=0; x<NX; ++x) {
                if (!basin.wet(x,z)) continue;
                float xx=x*DX-LX/2, zz=z*DX-LZ/2;
                float yy=surfaceHeight(basin.q[Basin::index(x,z)].h);
                RayCollision c=GetRayCollisionBox(ray,{{xx,yy-0.025f,zz},{xx+float(DX),yy+0.025f,zz+float(DX)}});
                if (c.hit && c.distance<nearest) { nearest=c.distance; hit=c.point; hover=true; }
            }
        }
        if (hover && healthy && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) basin.disturb(hit.x,hit.z);
        if (!paused && healthy) {
            double remaining=std::min(double(GetFrameTime()),0.05)*timeScale;
            int steps=0;
            while (remaining>1e-9 && steps++<100) {
                double dt=std::min(remaining,basin.stableStep());
                basin.step(dt); remaining-=dt;
                if (!basin.healthy()) { healthy=false; paused=true; break; }
            }
        }

        BeginDrawing();
        ClearBackground(ink);
        BeginMode3D(camera);
        DrawCube({0,-1.25f,0},LX+0.2f,0.15f,LZ+0.2f,Color{25,40,56,255});
        drawWater(basin,speedColors);
        DrawCubeWires({0,0,0},LX,2.5f,LZ,Color{89,126,151,150});
        for (int z=0; z<NZ; ++z) for (int x=0; x<NX; ++x) {
            int i=Basin::index(x,z);
            float xx=(x+0.5f)*DX-LX/2, zz=(z+0.5f)*DX-LZ/2;
            if (basin.solid[i]) DrawCube({xx,0,zz},DX,2.5f,DX,Color{145,160,173,255});
            else if (arrows && x%4==0 && z%4==0) {
                State q=basin.q[i];
                float vx=q.x/q.h, vz=q.z/q.h, norm=std::hypot(vx,vz);
                if (norm>0.02f) {
                    float scale=std::min(0.35f,0.6f/norm);
                    Vector3 start{xx,surfaceHeight(q.h)+0.12f,zz};
                    Vector3 end{xx+vx*scale,start.y,zz+vz*scale};
                    DrawLine3D(start,end,WHITE);
                    DrawLine3D(end,{end.x-vx*scale*0.3f-vz*scale*0.2f,end.y,end.z-vz*scale*0.3f+vx*scale*0.2f},WHITE);
                    DrawLine3D(end,{end.x-vx*scale*0.3f+vz*scale*0.2f,end.y,end.z-vz*scale*0.3f-vx*scale*0.2f},WHITE);
                }
            }
        }
        if (hover) DrawSphereWires({hit.x,hit.y+0.05f,hit.z},0.28f,6,12,Color{250,229,161,255});
        EndMode3D();

        studio::panel(sidebar,paper);
        studio::text("THE WATER LAB",38,39,13,teal);
        studio::text("Make waves.",38,62,31,dark);
        studio::text("A shallow-water experiment",38,106,15,muted);
        DrawLine(38,143,252,143,Color{204,213,199,255});
        for (int i=0;i<4;++i)
            studio::button({38,170.0f+i*49,214,41},TextFormat("%d   %s",i,names[i]),basin.preset==i,teal);
        studio::text("SURFACE VIEW",38,378,12,muted);
        studio::button(colorButton,speedColors ? "Speed" : "Depth",true,teal);
        studio::button(vectorButton,"Vectors",arrows,teal);
        studio::text(speedColors ? "Flow speed / m s^-1" : "Water depth / m",38,465,14,dark);
        for (int k=0;k<214;++k) DrawRectangle(38+k,492,1,8,waterColor(speedColors ? 2.5*k/213 : 0.35+1.35*k/213,speedColors));
        studio::text(speedColors ? "0" : "0.35",38,507,12,muted);
        studio::text(speedColors ? "2.5+" : "1.70+",215,507,12,muted);
        studio::text("Click the water to add a ripple.",38,549,13,muted);
        studio::text("SHALLOW WATER / 01",294,30,14,Color{168,194,186,255});
        studio::text(TextFormat("%s  /  %.2f s  /  %.2fx",paused ? "Paused" : "Live",basin.elapsed,timeScale),294,54,18,Color{225,237,225,255});
        studio::panel(transport,paper);
        studio::button(pauseButton,paused ? "Play" : "Pause",!paused,teal);
        studio::button(resetButton,"Reset",false,teal);
        studio::button(helpButton,showHelp ? "Hide help" : "Controls",showHelp,teal);
        double error=std::abs(basin.volume()/basin.initialVolume-1)*100;
        studio::panel({float(width)-224,float(height)-196,202,108},paper);
        studio::text("WATER VOLUME",width-208,height-184,12,muted);
        studio::text(TextFormat("%.2f m^3",basin.volume()),width-208,height-164,26,dark);
        studio::text(TextFormat("Drift %.1e %%",error),width-208,height-127,13,teal);
        if (showHelp) {
            studio::panel({float(width)-368,92,346,278},paper);
            studio::text("Move through the experiment",width-350,109,20,dark);
            const char* lines[]={"W A S D   Move across the basin", "Q / E   Move down / up", "Right drag   Orbit your focus point", "Middle or Shift + drag   Pan", "Wheel   Zoom    Shift   Move faster", "F   Home view    T   View from above", "Space   Pause    R   Reset scene", "- / =   Speed    H   Toggle this guide", "Flat bed / 64 x 48 / height shown at 1.5x"};
            for (int i=0;i<9;++i) studio::text(lines[i],width-350,146+i*23,14,i==8 ? muted : dark);
        }
        if (!healthy) studio::text("Numerical state invalid. Press R to reset.",294,83,18,ORANGE);
        EndDrawing();
        if (smoke && ++frames>=5) {
            Image screenshot=LoadImageFromScreen();
            bool saved=ExportImage(screenshot,"/tmp/shallow_water_sandbox.png");
            UnloadImage(screenshot);
            studio::unload();
            CloseWindow();
            return saved ? 0 : 1;
        }
    }
    studio::unload();
    CloseWindow();
    return 0;
}
