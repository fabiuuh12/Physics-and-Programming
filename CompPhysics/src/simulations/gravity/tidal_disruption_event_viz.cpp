#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include "../../common/studio.h"
#include "../../common/tidal_disruption.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kWidth = 1600;
constexpr int kHeight = 960;
constexpr int kStarParticles = 3200;
constexpr int kDiskParticles = 1500;
constexpr int kBackdropStars = 560;
constexpr float kHorizon = 0.92f;
constexpr float kPhotonRing = 1.34f;
constexpr float kCycle = 17.5f;

struct OrbitCamera { float yaw=0.76f, pitch=-0.48f, distance=14.8f; };
struct Debris { Vector3 seed{}; float phase=0, lane=0, size=0, release=0, heat=0; int fate=0; };
struct DiskDust { float radius=0, angle=0, height=0, speed=0, size=0, heat=0; };
struct SkyStar { Vector3 p{}; float size=0, alpha=0; Color color{}; };

float randomRange(std::mt19937& rng,float lo,float hi){return std::uniform_real_distribution<float>(lo,hi)(rng);}
float clamp01(float x){return std::clamp(x,0.0f,1.0f);}
float smooth(float a,float b,float x){return static_cast<float>(tidal::smoothstep(a,b,x));}
Vector3 mix3(Vector3 a,Vector3 b,float t){return Vector3Lerp(a,b,clamp01(t));}

Color mixColor(Color a,Color b,float t){
    t=clamp01(t); return {static_cast<unsigned char>(a.r+(b.r-a.r)*t),static_cast<unsigned char>(a.g+(b.g-a.g)*t),static_cast<unsigned char>(a.b+(b.b-a.b)*t),static_cast<unsigned char>(a.a+(b.a-a.a)*t)};
}

Vector3 starCenter(float t){
    float u=std::min(t,4.7f);
    return {-11.2f+2.02f*u,0.32f*std::sin(u*0.75f),2.65f-0.43f*u-0.045f*u*u};
}

std::vector<Debris> makeDebris(){
    std::mt19937 rng(74631); std::vector<Debris> out; out.reserve(kStarParticles);
    for(int i=0;i<kStarParticles;i++){
        float z=randomRange(rng,-1,1), a=randomRange(rng,0,2*PI), radial=std::cbrt(randomRange(rng,0,1));
        float xy=std::sqrt(std::max(0.0f,1-z*z));
        Vector3 seed={radial*xy*std::cos(a),radial*z,radial*xy*std::sin(a)};
        float selector=randomRange(rng,0,1); int fate=selector<0.27f?0:(selector<0.86f?1:2);
        out.push_back({seed,randomRange(rng,0,2*PI),randomRange(rng,-1,1),randomRange(rng,.022f,.061f),
                       3.35f+randomRange(rng,-.48f,.55f)+seed.x*.20f,randomRange(rng,0,1),fate});
    }
    return out;
}

std::vector<DiskDust> makeDisk(){
    std::mt19937 rng(9155); std::vector<DiskDust> out; out.reserve(kDiskParticles);
    for(int i=0;i<kDiskParticles;i++){
        float r=randomRange(rng,1.62f,5.7f); out.push_back({r,randomRange(rng,0,2*PI),randomRange(rng,-.12f,.12f)*(r/5.7f),
            .60f/std::pow(r,1.32f),randomRange(rng,.007f,.024f),1-r/6.2f});
    } return out;
}

std::vector<SkyStar> makeSky(){
    std::mt19937 rng(2727); std::vector<SkyStar> out; out.reserve(kBackdropStars);
    for(int i=0;i<kBackdropStars;i++){
        float a=randomRange(rng,0,2*PI), z=randomRange(rng,-.82f,.82f), r=randomRange(rng,31,47), q=std::sqrt(1-z*z);
        float temp=randomRange(rng,0,1); Color c=mixColor({135,181,255,255},{255,231,188,255},temp);
        out.push_back({{r*q*std::cos(a),r*z,r*q*std::sin(a)},randomRange(rng,.012f,.047f),randomRange(rng,.25f,.95f),c});
    } return out;
}

Vector3 diskPosition(const DiskDust& p,float t){
    float a=p.angle+t*p.speed; Vector3 v={p.radius*std::cos(a),p.height+.035f*std::sin(a*4+t),p.radius*std::sin(a)};
    v=Vector3RotateByAxisAngle(v,{1,0,0},.16f); return v;
}

Vector3 debrisPosition(const Debris& p,float t){
    Vector3 center=starCenter(t); Vector3 radial=Vector3Normalize(Vector3Scale(center,-1));
    float stretch=smooth(2.2f,4.25f,t); float along=Vector3DotProduct(p.seed,radial);
    Vector3 transverse=Vector3Subtract(p.seed,Vector3Scale(radial,along));
    Vector3 intact=Vector3Add(center,Vector3Add(Vector3Scale(radial,along*(1+4.6f*stretch)),Vector3Scale(transverse,1-.56f*stretch)));
    float u=std::max(0.0f,t-p.release), r0=3.25f+.95f*p.seed.x+.28f*p.lane;
    float theta0=2.72f+.34f*p.seed.z;
    Vector3 target=intact;
    if(p.fate==0){
        float r=std::max(.62f,r0*std::exp(-.24f*u)); float a=theta0+1.15f*u+.19f*u*u;
        target={r*std::cos(a),p.seed.y*.46f*std::exp(-.20f*u),r*std::sin(a)};
    }else if(p.fate==1){
        float settle=std::exp(-.075f*u), base=2.0f+(r0-1.4f)*settle;
        float r=base*(1+.18f*std::sin(p.phase+u*.72f)*settle); float a=theta0+(.73f+.12f*p.heat)*u+.038f*u*u;
        target={r*std::cos(a),p.seed.y*.58f*settle+.08f*std::sin(p.phase+u*1.7f),r*std::sin(a)};
    }else{
        float r=r0+u*(.72f+.32f*p.heat); float a=theta0+.78f*(1-std::exp(-.52f*u))+.07f*p.lane;
        target={r*std::cos(a),p.seed.y*.45f+.12f*u*p.lane,r*std::sin(a)};
    }
    return mix3(intact,target,smooth(p.release,p.release+.82f,t));
}

float debrisAlpha(const Debris& p,float t){
    if(t<p.release) return 1; if(p.fate!=0) return 1;
    Vector3 v=debrisPosition(p,t); return static_cast<float>(tidal::horizonFade(Vector3Length(v),kHorizon));
}

void updateCamera(Camera3D* c,OrbitCamera* o){
    studio::pan(c,o->yaw,o->pitch,o->distance);
    if(IsMouseButtonDown(MOUSE_BUTTON_LEFT)&&!studio::panGesture()){
        Vector2 d=GetMouseDelta(); o->yaw-=d.x*.0043f; o->pitch=std::clamp(o->pitch+d.y*.0043f,-1.38f,1.38f);
    }
    o->distance=std::clamp(o->distance*std::exp(-GetMouseWheelMove()*.085f),6.5f,38.0f);
    float cp=std::cos(o->pitch); c->position=Vector3Add(c->target,{o->distance*cp*std::cos(o->yaw),o->distance*std::sin(o->pitch),o->distance*cp*std::sin(o->yaw)});
}

void drawRing(float radius,float y,Color color,float width=1){
    constexpr int n=180; for(int i=0;i<n;i++){
        float a=2*PI*i/n,b=2*PI*(i+1)/n; DrawLine3D({radius*std::cos(a),y,radius*std::sin(a)},{radius*std::cos(b),y,radius*std::sin(b)},color);
        if(width>1) DrawLine3D({(radius+.018f)*std::cos(a),y,(radius+.018f)*std::sin(a)},{(radius+.018f)*std::cos(b),y,(radius+.018f)*std::sin(b)},color);
    }
}

Vector3 streamPoint(float s,float t,float offset){
    float reveal=smooth(2.0f,7.0f,t), outer=std::max(3.2f,Vector3Length(starCenter(std::min(t,4.0f))));
    outer+=std::max(0.0f,t-5.0f)*.22f;
    float outerAngle=2.83f+offset*.14f;
    float wraps=.30f+1.48f*reveal;
    float a=outerAngle+s*wraps*2*PI;
    float r=outer*(1-s)+1.06f*s+.30f*std::sin(s*PI)*std::sin(t*.7f+offset);
    return {r*std::cos(a),offset*.10f*(1-s)+.05f*std::sin(a*2+t),r*std::sin(a)};
}

void drawStreamRibbons(float t){
    float visible=smooth(2.15f,3.7f,t); if(visible<.01f)return;
    constexpr int segments=110;
    for(int layer=0;layer<34;layer++){
        float f=(layer-16.5f)/16.5f, layerOffset=f*.48f;
        Color c=mixColor({255,73,22,255},{255,245,194,255},1-std::fabs(f));
        float alpha=visible*(.20f+.42f*(1-std::fabs(f)));
        Vector3 previous=streamPoint(0,t,layerOffset);
        for(int i=0;i<segments;i++){
            float s=(i+1)/float(segments); Vector3 next=streamPoint(s,t,layerOffset);
            float envelope=.25f+.75f*std::sin(PI*std::pow(s,.72f));
            next.y+=f*.24f*envelope+.018f*std::sin(i*.9f+layer*1.7f);
            DrawLine3D(previous,next,Fade(c,alpha*envelope)); previous=next;
            if(layer%5==0&&i%4==0) DrawSphere(next,.025f+.045f*envelope,Fade(c,.10f*visible));
        }
    }
}

void drawBlackHole(float t,const std::vector<DiskDust>& disk){
    for(const auto& p:disk){
        Vector3 v=diskPosition(p,t); Color c=mixColor({255,91,39,255},{255,234,176,255},p.heat);
        DrawSphere(v,p.size*1.55f,Fade(c,.035f)); DrawSphere(v,p.size,Fade(c,.42f));
    }
    for(int i=0;i<8;i++) drawRing(kPhotonRing+i*.014f,(i-3.5f)*.006f,Fade({255,178,92,255},.18f+i*.035f),2);
    DrawSphere({0,0,0},1.78f,Fade({76,114,255,255},.025f));
    DrawSphere({0,0,0},kHorizon,BLACK);
    DrawSphereWires({0,0,0},kPhotonRing,24,24,Fade({255,216,151,255},.14f));
}

void drawTrails(const std::vector<Debris>& debris,float t){
    for(size_t i=0;i<debris.size();i+=11){
        const Debris& p=debris[i]; if(t<p.release+.15f) continue;
        float start=std::max(p.release,t-1.65f); Vector3 prev=debrisPosition(p,start);
        for(int s=1;s<=8;s++){
            float q=start+(t-start)*s/8.0f; Vector3 now=debrisPosition(p,q);
            float a=debrisAlpha(p,q)*(s/8.0f)*.26f; Color c=p.fate==2?Color{255,164,77,255}:Color{255,205,103,255};
            DrawLine3D(prev,now,Fade(c,a)); prev=now;
        }
    }
}

void drawStarAndDebris(const std::vector<Debris>& debris,float t){
    drawTrails(debris,t);
    for(const Debris& p:debris){
        Vector3 v=debrisPosition(p,t); float a=debrisAlpha(p,t); if(a<.015f) continue;
        float hot=clamp01(.2f+p.heat*.65f+smooth(3.0f,7.0f,t)*.25f);
        Color c=mixColor({255,119,39,255},{255,250,214,255},hot);
        float s=p.size*(1+.55f*smooth(3.0f,5.0f,t)); DrawSphere(v,s*3.0f,Fade(c,.075f*a)); DrawSphere(v,s,Fade(c,.96f*a));
    }
    float intact=1-smooth(2.7f,4.5f,t); if(intact>.01f){
        Vector3 c=starCenter(t); float pulse=1+.025f*std::sin(t*5.2f);
        Vector3 towardHole=Vector3Normalize(Vector3Scale(c,-1));
        rlDisableDepthTest();
        DrawSphere(c,1.42f*pulse,Fade({255,102,28,255},.045f*intact));
        DrawSphere(c,1.13f*pulse,Fade({255,193,77,255},.14f*intact));
        DrawSphere(c,.91f*pulse,Fade({255,249,204,255},.62f+.32f*intact));
        DrawSphere(c,.68f*pulse,{255,255,238,255});
        DrawCylinderEx(Vector3Add(c,Vector3Scale(towardHole,.34f)),Vector3Add(c,Vector3Scale(towardHole,1.72f)),.54f,.035f,24,{255,246,197,245});
        DrawSphere(Vector3Add(c,Vector3Scale(towardHole,1.68f)),.08f,{255,255,226,255});
        rlEnableDepthTest();
    }
}

void drawHud(float t,float speed,bool paused,int accreted){
    studio::title("Tidal Disruption Event",studio::Style::Observatory);
    studio::readout(TextFormat("%s   |   t = %.1f h   |   debris captured: %d",tidal::phaseName(t),t*3.2f,accreted));
    studio::note("A solar-type star crosses the tidal radius of a supermassive black hole");
    Rectangle box={GetScreenWidth()-295.0f,88,267,151}; studio::panel(box,{12,20,34,228});
    studio::text("LIVE EVENT",box.x+18,box.y+15,13,{231,190,128,255});
    studio::text(TextFormat("Playback  %.2fx%s",speed,paused?"  PAUSED":""),box.x+18,box.y+41,17,{230,235,241,255});
    studio::text("Gold  bound debris",box.x+18,box.y+73,14,{255,211,128,255});
    studio::text("Orange  escaping stream",box.x+18,box.y+96,14,{255,145,74,255});
    studio::text("Dark center  event horizon",box.x+18,box.y+119,14,{162,179,198,255});
    studio::help("Drag: orbit | Wheel: zoom | Shift+drag: pan | Space: pause | R: replay | -/+: speed | 1/2/3: views"); studio::fps();
}

void drawStarScreenGlow(const Camera3D& camera,float t,float cameraDistance){
    float intact=1-smooth(2.7f,4.5f,t); if(intact<=.01f)return;
    Vector2 p=GetWorldToScreen(starCenter(t),camera);
    Vector2 hole=GetWorldToScreen({0,0,0},camera),direction=Vector2Normalize(Vector2Subtract(hole,p));
    float radius=(54.0f+32.0f*intact)*14.8f/std::max(6.5f,cameraDistance);
    DrawCircleGradient(int(p.x),int(p.y),radius*1.75f,Fade({255,119,30,255},.24f*intact),BLANK);
    Vector2 side={-direction.y*radius*.48f,direction.x*radius*.48f};
    Vector2 base=Vector2Add(p,Vector2Scale(direction,radius*.18f));
    Vector2 tip=Vector2Add(p,Vector2Scale(direction,radius*3.1f));
    DrawTriangle(Vector2Add(base,side),Vector2Subtract(base,side),tip,Fade({255,235,174,255},.92f));
    DrawCircleGradient(int(p.x),int(p.y),radius,Fade({255,251,210,255},.96f),Fade({255,147,42,255},.10f));
    DrawCircleV(p,radius*.48f,{255,255,238,255});
}

void preset(int key,OrbitCamera* o,Camera3D* c){
    if(key==KEY_ONE){o->yaw=.76f;o->pitch=-.48f;o->distance=14.8f;c->target={-1.0f,0,0};}
    if(key==KEY_TWO){o->yaw=1.57f;o->pitch=-1.12f;o->distance=17;c->target={0,0,0};}
    if(key==KEY_THREE){o->yaw=3.05f;o->pitch=-.08f;o->distance=13.5f;c->target={-.4f,0,0};}
}

} // namespace

int main(int argc,char** argv){
    float startTime=0; std::string capture; int captureFrames=2;
    for(int i=1;i<argc;i++){
        std::string a=argv[i];
        if(a=="--time"&&i+1<argc) startTime=std::clamp(std::strtof(argv[++i],nullptr),0.0f,kCycle-.01f);
        else if(a=="--capture"&&i+1<argc) capture=argv[++i];
        else if(a=="--frames"&&i+1<argc) captureFrames=std::clamp(std::atoi(argv[++i]),1,1200);
        else { std::fprintf(stderr,"Unknown or incomplete option: %s\n",a.c_str()); return 2; }
    }
    SetConfigFlags(FLAG_WINDOW_RESIZABLE|FLAG_MSAA_4X_HINT);
    if(std::getenv("COMPPHYSICS_SMOKE_FRAMES")) SetConfigFlags(FLAG_WINDOW_HIDDEN);
    InitWindow(kWidth,kHeight,"Tidal Disruption Event - C++ (raylib)"); SetWindowMinSize(1000,650); SetTargetFPS(120);
    Camera3D camera{{0,0,0},{-1.1f,0,0},{0,1,0},46,CAMERA_PERSPECTIVE}; OrbitCamera orbit; preset(KEY_ONE,&orbit,&camera);
    auto debris=makeDebris(); auto disk=makeDisk(); auto sky=makeSky();
    float simTime=startTime,speed=1; bool paused=false; int frame=0;
    while(!WindowShouldClose()){
        if(IsKeyPressed(KEY_SPACE)||IsKeyPressed(KEY_P)) paused=!paused;
        if(IsKeyPressed(KEY_R)){simTime=0;paused=false;}
        if(IsKeyPressed(KEY_MINUS)||IsKeyPressed(KEY_KP_SUBTRACT)) speed=std::max(.25f,speed-.25f);
        if(IsKeyPressed(KEY_EQUAL)||IsKeyPressed(KEY_KP_ADD)) speed=std::min(4.0f,speed+.25f);
        for(int key:{KEY_ONE,KEY_TWO,KEY_THREE}) if(IsKeyPressed(key)) preset(key,&orbit,&camera);
        updateCamera(&camera,&orbit);
        if(!paused&&capture.empty()){simTime+=std::min(GetFrameTime(),.05f)*speed;if(simTime>=kCycle)simTime=0;}
        int accreted=0; for(const auto& p:debris) if(p.fate==0&&debrisAlpha(p,simTime)<.08f) accreted++;
        BeginDrawing(); ClearBackground({2,4,11,255}); BeginMode3D(camera);
        for(const auto& s:sky){DrawSphere(s.p,s.size*2.5f,Fade(s.color,.035f*s.alpha));DrawSphere(s.p,s.size,Fade(s.color,s.alpha));}
        drawStreamRibbons(simTime); drawBlackHole(simTime,disk); drawStarAndDebris(debris,simTime); EndMode3D();
        drawStarScreenGlow(camera,simTime,orbit.distance);
        DrawRectangleGradientV(0,0,GetScreenWidth(),155,{2,4,11,215},{2,4,11,0}); drawHud(simTime,speed,paused,accreted); EndDrawing();
        if(!capture.empty()&&++frame>=captureFrames){std::filesystem::create_directories(std::filesystem::path(capture).parent_path());TakeScreenshot(capture.c_str());break;}
        if(studio::smokeFrame(__FILE__)) break;
    }
    studio::unload(); CloseWindow(); return 0;
}
