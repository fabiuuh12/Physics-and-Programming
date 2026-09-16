#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include "../common/studio.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
const Color bg{19,35,40,255}, paper{240,241,231,255}, dark{35,56,51,255};
const Color muted{103,121,117,255}, teal{39,117,102,255}, metal{100,126,135,255};
const Color fresh{87,218,228,255}, hot{247,134,84,255}, warm{239,201,117,255};
bool airflowArrows=true;
const char* names[]={"Turbo", "Supercharger", "Engine"};
struct EngineState {
    int mode=1;
    float rpm=3000, throttle=0.7f, spool=0;
    bool intercooler=true;
    float target() const { return std::clamp((rpm-1000)/4500*throttle,0.0f,1.0f); }
    void step(float dt) { spool+=(target()-spool)*(1-std::exp(-dt/(target()>spool ? 1.1f : 0.45f))); }
    float ratio() const { return mode==0 ? 1 : 1+(mode==1 ? 1.1f*spool : 0.8f*rpm/7000*throttle); }
    float compressedT() const { return 298.15f*(1+(std::pow(ratio(),2.0f/7)-1)/0.72f); }
    float temp() const { return 298.15f+(compressedT()-298.15f)*(intercooler && mode!=0 ? 0.3f : 1); }
    float pressure() const { return 101.325f*ratio()*(0.28f+0.69f*throttle); }
    float flow() const { return pressure()*1000/(287.05f*temp())*0.002f*rpm/120*0.9f; }
    float work() const { return flow()*1005*(compressedT()-298.15f)/1000; }
};
using Path=std::vector<Vector3>;
Vector3 along(const Path& p,float t) {
    float length=0;
    for(size_t i=1;i<p.size();++i)length+=Vector3Distance(p[i-1],p[i]);
    float d=std::clamp(t,0.0f,1.0f)*length;
    for(size_t i=1;i<p.size();++i){float l=Vector3Distance(p[i-1],p[i]);if(d<=l && l>0)return Vector3Lerp(p[i-1],p[i],d/l);d-=l;}
    return p.back();
}
Path curve(Vector3 a,Vector3 b,Vector3 c,Vector3 d) {
    Path p;
    for(int i=0;i<=60;++i){float t=i/60.0f,u=1-t;p.push_back(Vector3Add(Vector3Add(Vector3Scale(a,u*u*u),Vector3Scale(b,3*u*u*t)),Vector3Add(Vector3Scale(c,3*u*t*t),Vector3Scale(d,t*t*t))));}
    return p;
}
void append(Path& a,const Path& b){a.insert(a.end(),b.begin()+1,b.end());}
void vertex(Vector3 p,Color c){rlColor4ub(c.r,c.g,c.b,c.a);rlVertex3f(p.x,p.y,p.z);}
// Open tube surfaces reveal the flowing interior, like a physical cutaway.
void shell(const Path& p,float radius,Color c,bool cutaway) {
    // Parallel transport keeps adjacent rings joined through bends.
    std::vector<Vector3> ringU(p.size()),ringV(p.size());
    Vector3 lastU{0,0,1};
    for(size_t i=0;i<p.size();++i){
        Vector3 tangent=Vector3Normalize(Vector3Subtract(p[std::min(i+1,p.size()-1)],p[i?i-1:0]));
        Vector3 u=Vector3Subtract(lastU,Vector3Scale(tangent,Vector3DotProduct(lastU,tangent)));
        if(Vector3Length(u)<0.01f)u=Vector3CrossProduct(tangent,{0,1,0});
        ringU[i]=Vector3Normalize(u);ringV[i]=Vector3CrossProduct(tangent,ringU[i]);lastU=ringU[i];
    }
    rlDisableBackfaceCulling();rlBegin(RL_TRIANGLES);
    int sides=24, visible=cutaway ? 15 : sides;
    auto point=[&](size_t i,float a){return Vector3Add(p[i],Vector3Add(Vector3Scale(ringU[i],radius*std::cos(a)),Vector3Scale(ringV[i],radius*std::sin(a))));};
    for(size_t i=1;i<p.size();++i){
        for(int j=0;j<visible;++j){float a=2*PI*j/sides+0.55f,b=2*PI*(j+1)/sides+0.55f;
            Color shade=ColorBrightness(c,0.16f*std::sin(a)-0.08f);
            Vector3 aa=point(i-1,a),bb=point(i,a),cc=point(i,b),dd=point(i-1,b);
            vertex(aa,shade);vertex(bb,shade);vertex(cc,shade);vertex(aa,shade);vertex(cc,shade);vertex(dd,shade);
        }
    }
    rlEnd();rlEnableBackfaceCulling();
}
Color blend(Color a,Color b,float t){return {static_cast<unsigned char>(a.r+(b.r-a.r)*t),static_cast<unsigned char>(a.g+(b.g-a.g)*t),static_cast<unsigned char>(a.b+(b.b-a.b)*t),255};}
void flowArrow(Vector3 a,Vector3 b,Color color){
    Vector3 direction=Vector3Subtract(b,a);float length=Vector3Length(direction);if(length<0.001f)return;
    Vector3 neck=Vector3Lerp(a,b,0.57f);
    DrawCylinderEx(a,neck,0.022f,0.022f,6,color);
    DrawCylinderEx(neck,b,0.075f,0,8,color);
}
void stream(const Path& p,float radius,float phase,Color inlet,Color outlet,bool ribbons,bool particles,float strength=1) {
    if(strength<=0)return;
    for(int lane=0;lane<7;++lane){
        float az=lane*2*PI/7;
        auto point=[&](float t){Vector3 center=along(p,t), tangent=Vector3Normalize(Vector3Subtract(along(p,std::min(t+0.005f,1.0f)),along(p,std::max(t-0.005f,0.0f))));
            Vector3 ref=std::abs(tangent.y)>0.9f ? Vector3{1,0,0}:Vector3{0,1,0};Vector3 u=Vector3Normalize(Vector3CrossProduct(tangent,ref)),v=Vector3CrossProduct(tangent,u);
            float a=az+0.3f*std::sin(t*12-phase*2),r=radius*(0.5f+0.12f*std::sin(t*20+az-phase*3));
            return Vector3Add(center,Vector3Add(Vector3Scale(u,r*std::cos(a)),Vector3Scale(v,r*std::sin(a))));};
        if(ribbons){
            rlDisableBackfaceCulling();rlBegin(RL_TRIANGLES);
            for(int i=0;i<80;++i){float t=i/80.0f,tt=(i+1)/80.0f;Vector3 a=point(t),b=point(tt),w{0,0.017f,0.017f};
                Color c=Fade(blend(inlet,outlet,t),strength*(0.35f+0.25f*std::sin(t*34-phase*5+az)));
                vertex(Vector3Subtract(a,w),c);vertex(Vector3Add(a,w),c);vertex(Vector3Add(b,w),c);
                vertex(Vector3Subtract(a,w),c);vertex(Vector3Add(b,w),c);vertex(Vector3Subtract(b,w),c);
            }
    rlEnd();rlEnableBackfaceCulling();
        }
        if(airflowArrows && (lane==0 || lane==3))for(int i=0;i<10;++i){
            float t=0.02f+0.94f*std::fmod(i/10.0f+phase*0.18f,1.0f);
            flowArrow(point(t),point(std::min(t+0.035f,0.995f)),Fade(blend(inlet,outlet,t),strength));
        }
        if(particles)for(int i=0;i<18;++i){float t=std::fmod(i/18.0f+lane*0.013f+phase*0.18f,1.0f);Vector3 a=point(t),b=point(std::min(t+0.016f,1.0f));DrawLine3D(a,b,Fade(blend(inlet,outlet,t),strength));DrawSphereEx(a,0.025f,3,4,Fade(blend(inlet,outlet,t),strength));}
    }
}
void impeller(Vector3 center,float angle,Color c){
    DrawCylinderEx({center.x,center.y,center.z-0.12f},{center.x,center.y,center.z},0.87f,0.87f,48,metal);
    DrawCylinderEx(center,{center.x,center.y,center.z+0.45f},0.24f,0.08f,24,c);
    rlDisableBackfaceCulling();rlBegin(RL_TRIANGLES);
    for(int blade=0;blade<12;++blade)for(int i=0;i<12;++i){
        float r=0.2f+i*0.055f,rr=r+0.055f,a=angle+blade*PI/6+r*0.65f,b=a+0.055f*0.65f;
        Vector3 p{center.x+r*std::cos(a),center.y+r*std::sin(a),center.z},q{center.x+rr*std::cos(b),center.y+rr*std::sin(b),center.z};
        Vector3 pp=p,qq=q;pp.z+=0.34f*(1-r);qq.z+=0.34f*(1-rr);
        Color shade=ColorBrightness(c,0.2f*std::sin(a));vertex(p,shade);vertex(q,shade);vertex(qq,shade);vertex(p,shade);vertex(qq,shade);vertex(pp,shade);
    }
    rlEnd();rlEnableBackfaceCulling();
}
Path volute(float z){Path p;for(int i=0;i<=90;++i){float a=-PI/2+i/90.0f*2*PI,r=1.08f+0.28f*i/90;p.push_back({r*std::cos(a),2+r*std::sin(a),z});}append(p,curve(p.back(),{1,0.64f,z},{2,0.64f,z},{3.6f,0.64f,z}));return p;}
Path compressorPath(float z){
    Path p=curve({0,2,z+2.2f},{0,2,z+1.1f},{0,2,z+0.5f},{0.18f,2,z+0.24f});
    append(p,curve(p.back(),{0.4f,2,z+0.2f},{0.6f,1.2f,z},{0,0.92f,z}));
    append(p,volute(z));return p;
}
void compressor(float z,float rotor,bool cutaway){
    shell(volute(z),0.32f,metal,cutaway);
    if(!cutaway)shell({{0,2,z},{0,2,z+0.35f}},0.94f,metal,false);
    impeller({0,2,z},rotor,Color{174,200,207,255});
    shell({{0,2,z+0.6f},{0,2,z+2.2f}},0.46f,metal,cutaway);
}
void turbo(float phase,float rotor,bool cutaway,bool ribbons,bool particles){
    compressor(1.15f,rotor,cutaway);
    DrawCylinderEx({0,2,-1.45f},{0,2,1.2f},0.12f,0.12f,20,Color{189,181,149,255});
    shell(volute(-1.45f),0.32f,Color{119,100,85,255},cutaway);
    impeller({0,2,-1.45f},rotor,Color{202,150,110,255});
    shell({{0,2,-1.7f},{0,2,-3.2f}},0.46f,metal,cutaway);
    Path exhaust=volute(-1.45f);std::reverse(exhaust.begin(),exhaust.end());
    append(exhaust,curve(exhaust.back(),{0,1.1f,-1.45f},{0,2,-1.45f},{0,2,-1.8f}));
    append(exhaust,curve(exhaust.back(),{0,2,-2},{0,2,-2.6f},{0,2,-3.2f}));
    stream(compressorPath(1.15f),0.29f,phase,fresh,warm,ribbons,particles);
    stream(exhaust,0.29f,phase*1.1f,hot,warm,ribbons,particles);
}
// Schematic 3/5-lobe male/female pair, not conjugate manufacturing profiles.
// Opposite helix hands and inverse lobe-count speeds preserve phase along the axes.
float screwAngle(bool female,float drive,float z){
    return female ? -drive*3.0f/5-0.42f*(z+2) : drive+0.70f*(z+2);
}
float screwRadius(bool female,float relative){
    return female ? 0.77f-0.18f*std::pow(0.5f+0.5f*std::cos(5*relative),2.0f)
                  : 0.59f+0.27f*std::cos(3*relative);
}
Vector3 screwPoint(bool female,float drive,float a,float z){
    float radius=screwRadius(female,a-screwAngle(female,drive,z));
    return {(female?0.84f:-0.84f)+radius*std::cos(a),2.05f+radius*std::sin(a),z};
}
void screwRotor(bool female,float drive){
    Color silver=female?Color{147,168,179,255}:Color{193,205,211,255};
    rlDisableBackfaceCulling();rlBegin(RL_TRIANGLES);
    for(int k=0;k<48;++k)for(int i=0;i<96;++i){
        float z=-2+4*k/48.0f,zz=z+4.0f/48,a=i*2*PI/96,b=(i+1)*2*PI/96;
        Vector3 p=screwPoint(female,drive,a,z),q=screwPoint(female,drive,b,z),r=screwPoint(female,drive,b,zz),s=screwPoint(female,drive,a,zz);
        Vector3 normal=Vector3Normalize(Vector3CrossProduct(Vector3Subtract(q,p),Vector3Subtract(s,p)));
        float light=Vector3DotProduct(normal,Vector3Normalize({-0.4f,0.8f,0.5f}));
        Color shade=ColorBrightness(silver,light*0.26f-0.08f);
        vertex(p,shade);vertex(q,shade);vertex(r,shade);vertex(p,shade);vertex(r,shade);vertex(s,shade);
        if(k==0){vertex({female?0.84f:-0.84f,2.05f,z},silver);vertex(q,silver);vertex(p,silver);}
        if(k==47){vertex({female?0.84f:-0.84f,2.05f,zz},silver);vertex(s,silver);vertex(r,silver);}
    }
    rlEnd();rlEnableBackfaceCulling();
    float x=female?0.84f:-0.84f;
    DrawCylinderEx({x,2.05f,-2.3f},{x,2.05f,2.65f},0.13f,0.13f,20,metal);
}
Path screwRoute(bool female,float drive,float lane){
    Path p;
    for(int i=0;i<=80;++i){
        float t=i/80.0f,z=-2.75f+4.55f*t;
        float a=screwAngle(female,drive,std::clamp(z,-2.0f,2.0f))+lane;
        float radius=screwRadius(female,lane)+0.14f;
        p.push_back({(female?0.84f:-0.84f)+radius*std::cos(a),2.05f+radius*std::sin(a),z});
    }
    append(p,curve(p.back(),{p.back().x,1.3f,1.95f},{0,0.6f,1.6f},{0,0.45f,1.6f}));return p;
}
void timingGear(float x,float radius,int teeth,float angle){
    DrawCylinderEx({x,2.05f,2.28f},{x,2.05f,2.48f},radius*0.89f,radius*0.89f,40,metal);
    for(int i=0;i<teeth;++i){float a=angle+i*2*PI/teeth;Vector3 p{x+radius*0.87f*std::cos(a),2.05f+radius*0.87f*std::sin(a),2.39f};Vector3 q{x+radius*std::cos(a),2.05f+radius*std::sin(a),2.39f};DrawCylinderEx(p,q,0.055f,0.055f,6,Color{158,178,186,255});}
}
void supercharger(float phase,float rotor,bool cutaway,bool ribbons,bool particles){
    // Low mounting flange, rear plate, and opened roof echo the supplied reference.
    for(float side:{-1.0f,1.0f}){
        DrawCube({side*1.15f,1.02f,0},1.55f,0.18f,4.55f,metal);
        DrawCube({side*1.68f,2.02f,-2.17f},0.3f,2.15f,0.16f,metal);
    }
    DrawCube({0,1.02f,-0.62f},0.75f,0.18f,3.3f,metal);
    DrawCube({0,3.02f,-2.17f},3.65f,0.15f,0.16f,metal);
    DrawCube({-1.8f,1.55f,0},0.15f,0.85f,4.2f,metal);
    DrawCube({1.8f,1.55f,0},0.15f,0.85f,4.2f,metal);
    if(!cutaway){DrawCube({0,3.04f,0},3.7f,0.15f,4.3f,metal);DrawCube({1.8f,2.55f,0},0.15f,1.05f,4.2f,metal);DrawCube({-1.8f,2.55f,0},0.15f,1.05f,4.2f,metal);DrawCube({0,2.1f,2.15f},3.65f,2.05f,0.13f,metal);}
    for(float side:{-1.0f,1.0f})for(int i=0;i<5;++i){float z=-1.9f+i*0.94f;DrawCube({side*1.96f,1.02f,z},0.36f,0.2f,0.38f,metal);DrawCylinderEx({side*1.96f,1.13f,z},{side*1.96f,1.23f,z},0.10f,0.10f,6,dark);}
    screwRotor(false,rotor);screwRotor(true,rotor);
    timingGear(-0.84f,0.63f,15,rotor);timingGear(0.84f,1.05f,25,-rotor*3/5);
    // Long input snout with a visible shaft and grooved pulley.
    shell({{-0.84f,2.05f,2.6f},{-0.84f,2.05f,4.25f}},0.3f,metal,cutaway);
    DrawCylinderEx({-0.84f,2.05f,2.5f},{-0.84f,2.05f,4.6f},0.13f,0.13f,20,paper);
    DrawCylinderEx({-0.84f,2.05f,4.25f},{-0.84f,2.05f,4.6f},0.69f,0.69f,48,dark);
    for(int i=0;i<5;++i)DrawCylinderWiresEx({-0.84f,2.05f,4.28f+i*0.06f},{-0.84f,2.05f,4.29f+i*0.06f},0.70f,0.70f,48,metal);
    for(int i=0;i<4;++i){float a=rotor+i*PI/2;DrawLine3D({-0.84f,2.05f,4.61f},{-0.84f+0.56f*std::cos(a),2.05f+0.56f*std::sin(a),4.61f},metal);}
    for(bool female:{false,true})for(int lane=0;lane<2;++lane){
        float offset=(female?0.0f:PI/3)+lane*2*PI/(female?5:3);
        Path route=screwRoute(female,rotor,offset);
        stream(route,0.075f,phase,fresh,warm,ribbons,particles);
        // Shrinking colored markers illustrate decreasing trapped-pocket volume.
        if(particles)for(int i=0;i<5;++i){float t=std::fmod(i/5.0f+phase*0.18f,1.0f);Vector3 p=along(route,t);float size=0.12f*(1-0.55f*t);DrawSphereEx(p,size,6,8,Fade(blend(fresh,warm,t),0.5f));}
    }
}
float pistonY(float cycle){return 1.05f+0.52f*std::cos(cycle);}
void engine(float phase,float cycle,bool cutaway,bool ribbons,bool particles){
    float y=pistonY(cycle);int stroke=int(cycle/PI);
    Path intake=curve({-3.5f,3.5f,0},{-2,3.5f,0},{-0.6f,3.6f,0},{-0.4f,2.85f,0});
    Path exhaust=curve({0.4f,2.85f,0},{0.6f,3.6f,0},{2,3.5f,0},{3.5f,3.5f,0});
    shell(intake,0.24f,metal,cutaway);shell(exhaust,0.24f,metal,cutaway);
    // Back half of cylinder stays solid; front opens for the gas and piston.
    rlDisableBackfaceCulling();rlBegin(RL_TRIANGLES);
    for(int i=0;i<(cutaway?32:64);++i){float a=PI+i*2*PI/64,b=a+2*PI/64;Vector3 p{0.86f*std::cos(a),0.45f,0.86f*std::sin(a)},q{0.86f*std::cos(b),0.45f,0.86f*std::sin(b)},r=q,s=p;r.y=s.y=2.85f;Color c=ColorBrightness(metal,0.15f*std::sin(a));vertex(p,c);vertex(q,c);vertex(r,c);vertex(p,c);vertex(r,c);vertex(s,c);}
    rlEnd();rlEnableBackfaceCulling();
    DrawCylinderEx({0,y-0.22f,0},{0,y,0},0.8f,0.8f,48,Color{179,195,201,255});
    for(int i=0;i<2;++i)DrawCylinderWiresEx({0,y-0.07f-i*0.09f,0},{0,y-0.065f-i*0.09f,0},0.81f,0.81f,40,dark);
    Vector3 crank{0.32f*std::sin(cycle),-0.15f+0.32f*std::cos(cycle),0};
    DrawCylinderEx(crank,{0,y-0.2f,0},0.09f,0.09f,12,metal);DrawCylinderEx({0,-0.15f,-0.8f},{0,-0.15f,0.8f},0.15f,0.15f,16,metal);
    DrawCylinderEx({0,-0.15f,0.5f},{crank.x,crank.y,0.5f},0.1f,0.1f,12,metal);
    for(int side=0;side<2;++side){bool open=stroke==(side?3:0);float x=side?0.4f:-0.4f,yy=open?2.63f:2.85f;
        DrawCylinderEx({x,yy,0},{x,3.3f,0},0.035f,0.035f,10,metal);DrawCylinderEx({x,yy,0},{x,yy+0.06f,0},0.18f,0.18f,20,open?(side?hot:fresh):metal);}
    if(stroke==0){append(intake,curve(intake.back(),{-0.4f,2.6f,0},{-0.5f,y+0.4f,0},{0,y+0.18f,0.2f}));stream(intake,0.17f,phase*2,fresh,fresh,ribbons,particles);}
    if(stroke==3){Path p=curve({0,y+0.2f,0.1f},{0.4f,y+0.5f,0},{0.4f,2.7f,0},exhaust.front());append(p,exhaust);stream(p,0.17f,phase*2,hot,hot,ribbons,particles);}
    Color gas=stroke==0?fresh:stroke==1?warm:stroke==2?Color{255,191,91,255}:hot;
    if(particles)for(int i=0;i<180;++i){float f=(i+0.5f)/180,a=i*2.39996f+phase*0.7f,r=0.65f*std::sqrt(std::fmod(i*0.618f,1.0f));Vector3 p{r*std::cos(a),y+0.08f+f*(2.75f-y-0.08f),r*std::sin(a)};DrawSphereEx(p,0.025f,3,4,Fade(gas,0.65f));}
    if(ribbons)for(int i=0;i<6;++i){Path p;for(int j=0;j<40;++j){float t=j/39.0f,a=t*5+i*PI/3+phase*0.3f;p.push_back({0.5f*std::cos(a),y+0.1f+t*(2.7f-y-0.1f),0.5f*std::sin(a)});}for(size_t j=1;j<p.size();++j)DrawLine3D(p[j-1],p[j],Fade(gas,0.35f));}
}
void slider(const char* title,float y,float& value,float lo,float hi,const char* unit,bool cameraDragging){
    studio::text(title,38,y,14,muted);studio::text(TextFormat("%.0f %s",value,unit),158,y,14,dark);
    if(!cameraDragging && IsMouseButtonDown(MOUSE_BUTTON_LEFT)&&studio::over({32,y+20,226,27}))value=lo+(hi-lo)*std::clamp((GetMouseX()-38)/214.0f,0.0f,1.0f);
    DrawRectangleRounded({38,y+31,214,5},1,8,Color{204,213,199,255});DrawRectangleRounded({38,y+31,214*(value-lo)/(hi-lo),5},1,8,teal);DrawCircleV({38+214*(value-lo)/(hi-lo),y+33},6,teal);
}
void label(Camera3D camera,Vector3 at,const char* text,Color color){Vector2 p=GetWorldToScreen(at,camera);float w=studio::textWidth(text,14);if(p.x-w/2<286 || p.y<110 || p.y>GetScreenHeight()-170)return;DrawRectangleRounded({p.x-w/2-8,p.y-4,w+16,25},0.2f,6,Fade(bg,0.88f));studio::text(text,p.x-w/2,p.y,14,color);}
int selfTest() {
    int failures=0;
    auto check=[&](bool ok,const char* s){ if(!ok){std::printf("FAIL: %s\n",s); ++failures;} };
    EngineState m; m.mode=0; m.throttle=1;
    check(m.pressure()<101.325f,"NA manifold remains below ambient");
    check(std::abs(m.temp()-298.15f)<0.001f,"NA temperature");
    float base=m.flow(); m.rpm*=2;
    check(std::abs(m.flow()/base-2)<0.0001f,"four-stroke flow scales with RPM");
    m.mode=1; m.spool=1; m.intercooler=false;
    float warm=m.temp(), flow=m.flow(); m.intercooler=true;
    check(m.temp()<warm && m.flow()>flow,"cooling increases charge density");
    m.spool=0; m.step(0.1f); check(m.spool>0 && m.spool<m.target(),"turbo spool is gradual");
    for(int mode=0;mode<3;++mode) for(int r=800;r<=7000;r+=100) for(int t=0;t<=100;t+=5) {
        m.mode=mode; m.rpm=float(r); m.throttle=t/100.0f;
        for(int i=0;i<100;++i) m.step(0.1f);
        check(std::isfinite(m.flow()) && m.flow()>0 && m.temp()>=298.15f && m.ratio()>=1,"finite physical state");
    }
    EngineState a,b; for(int i=0;i<100;++i)a.step(0.01f); b.step(1);
    check(std::abs(a.spool-b.spool)<1e-5,"spool timestep independence");
    Path p{{0,0,0},{2,0,0},{2,1,0}};
    check(Vector3Distance(along(p,0.5f),{1.5f,0,0})<1e-5,"tracers use path arc length");
    for(float z : {0.5f,1.15f}) {
        Path route=compressorPath(z);
        check(Vector3Distance(route.front(),{0,2,z+2.2f})<1e-5,"compressor inlet at axial opening");
        check(Vector3Distance(route.back(),{3.6f,0.64f,z})<1e-5,"compressor outlet meets housing");
        for(size_t i=1;i<route.size();++i)
            check(std::isfinite(route[i].x) && Vector3Distance(route[i],route[i-1])<0.25f,"continuous finite flow route");
    }
    for(bool female:{false,true}){
        Path route=screwRoute(female,0,0);
        check(std::abs(route.front().z+2.75f)<1e-5,"screw inlet at rear");
        check(Vector3Distance(route.back(),{0,0.45f,1.6f})<1e-5,"screw discharge port");
        for(size_t i=1;i<81;++i)check(route[i].z>route[i-1].z,"screw transport is axial");
        for(int i=0;i<96;++i){Vector3 p=screwPoint(female,0,i*2*PI/96,0);check(std::isfinite(p.x)&&std::isfinite(p.y),"finite helical mesh");}
    }
    check(std::abs(3*screwAngle(false,1,0)+5*screwAngle(true,1,0))<1e-5,"3/5 gearing and helix phasing");
    check(pistonY(0)>pistonY(PI),"intake piston descends");
    check(pistonY(2*PI)>pistonY(PI),"compression piston rises");
    check(std::abs(pistonY(0)-pistonY(4*PI))<1e-5,"four-stroke cycle closes");
    std::printf("Engine airflow checks: %s\n",failures ? "FAILED" : "PASS"); return failures ? 1 : 0;
}
}
int main(int argc,char** argv){
    bool smoke=false;int tab=0;
    for(int i=1;i<argc;++i){if(!std::strcmp(argv[i],"--self-test"))return selfTest();if(!std::strcmp(argv[i],"--smoke-test"))smoke=true;if(!std::strcmp(argv[i],"--na")||!std::strcmp(argv[i],"--engine"))tab=2;if(!std::strcmp(argv[i],"--supercharger"))tab=1;}
    SetConfigFlags(FLAG_MSAA_4X_HINT|FLAG_WINDOW_RESIZABLE);InitWindow(1360,900,"Airflow | Component Lab");SetWindowMinSize(1050,740);SetTargetFPS(60);
    EngineState state;state.intercooler=false;
    bool paused=false,cutaway=true,ribbons=true,particles=true,help=false;
    float phase=0,rotor=0,cycle=0,timeScale=1,yaw=1.12f,pitch=0.48f,distance=15;
    int dragButton=-1; bool dragPan=false;
    Camera3D camera{{0,0,0},{-0.9f,1.7f,0},{0,1,0},43,CAMERA_PERSPECTIVE};int frames=0;
    auto home=[&](){camera.target={-0.9f,1.7f,0};yaw=tab==2?1.45f:1.12f;pitch=tab==2?0.26f:0.48f;distance=tab==1?18:15;if(tab==1){camera.target={-0.9f,1.7f,0.8f};yaw=0.82f;pitch=0.65f;}};home();
    while(!WindowShouldClose()){
        int width=GetScreenWidth(),height=GetScreenHeight();float dt=std::min(GetFrameTime(),0.05f);
        Rectangle sidebar{22,22,246,600},transport{width/2.0f-170,height-70.0f,340,48};
        Rectangle pauseButton{transport.x+8,transport.y+6,95,36},resetButton{transport.x+110,transport.y+6,95,36},helpButton{transport.x+212,transport.y+6,120,36};
        Rectangle cutButton{38,353,214,36},ribbonButton{38,401,103,36},particleButton{149,401,103,36};
        int previous=tab;
        for(int i=0;i<3;++i)if(IsKeyPressed(KEY_ONE+i)||studio::clicked({294.0f+i*168,24,156,42}))tab=i;
        if(tab!=previous){home();phase=rotor=cycle=0;state.spool=0;}
        if(IsKeyPressed(KEY_SPACE)||studio::clicked(pauseButton))paused=!paused;
        if(IsKeyPressed(KEY_C)||studio::clicked(cutButton))cutaway=!cutaway;
        if(IsKeyPressed(KEY_V)||studio::clicked(ribbonButton))ribbons=!ribbons;
        if(IsKeyPressed(KEY_P)||studio::clicked(particleButton))particles=!particles;
        if(IsKeyPressed(KEY_G)||studio::clicked({38,445,214,30}))airflowArrows=!airflowArrows;
        if(IsKeyPressed(KEY_H)||studio::clicked(helpButton))help=!help;
        if(IsKeyPressed(KEY_R)||studio::clicked(resetButton)){phase=rotor=cycle=0;state=EngineState{};state.intercooler=false;paused=false;timeScale=1;}
        if(IsKeyPressed(KEY_MINUS))timeScale=std::max(0.125f,timeScale/2);
        if(IsKeyPressed(KEY_EQUAL))timeScale=std::min(2.0f,timeScale*2);
        if(IsKeyPressed(KEY_F))home();if(IsKeyPressed(KEY_T))pitch=1.5f;
        // Capture the gesture where it begins, not wherever the pointer moves.
        bool onUI=studio::over(sidebar)||studio::over(transport)||
                  studio::over({294,24,492,42})||
                  (help&&studio::over({width-374.0f,160,352,250}));
        if(dragButton>=0 && (!IsMouseButtonDown(dragButton)||!IsWindowFocused()))dragButton=-1;
        bool shift=IsKeyDown(KEY_LEFT_SHIFT)||IsKeyDown(KEY_RIGHT_SHIFT);
        if(dragButton<0 && !onUI && IsWindowFocused()) {
            for(int button : {MOUSE_BUTTON_LEFT,MOUSE_BUTTON_RIGHT,MOUSE_BUTTON_MIDDLE}) {
                if(IsMouseButtonPressed(button)) {
                    dragButton=button;dragPan=button==MOUSE_BUTTON_MIDDLE||shift;break;
                }
            }
        }
        if(dragButton>=0) {
            Vector2 delta=GetMouseDelta();
            if(dragPan) {
                float scale=2*distance*std::tan(camera.fovy*PI/360)/std::max(1,height);
                Vector3 rightAxis{std::sin(yaw),0,-std::cos(yaw)};
                Vector3 upAxis{-std::sin(pitch)*std::cos(yaw),std::cos(pitch),-std::sin(pitch)*std::sin(yaw)};
                camera.target=Vector3Add(camera.target,Vector3Scale(Vector3Add(Vector3Scale(rightAxis,-delta.x),Vector3Scale(upAxis,delta.y)),scale));
            } else {
                yaw-=delta.x*0.005f;pitch=std::clamp(pitch+delta.y*0.005f,-1.4f,1.5f);
            }
        }
        if(!onUI)distance=std::clamp(distance*std::exp(-GetMouseWheelMove()*0.08f),5.0f,32.0f);
        yaw+=dt*(float(IsKeyDown(KEY_LEFT))-float(IsKeyDown(KEY_RIGHT)));
        pitch=std::clamp(pitch+dt*(float(IsKeyDown(KEY_UP))-float(IsKeyDown(KEY_DOWN))),-1.4f,1.5f);
        float move=dt*(IsKeyDown(KEY_LEFT_SHIFT)?8:3),forward=float(IsKeyDown(KEY_W))-IsKeyDown(KEY_S),right=float(IsKeyDown(KEY_D))-IsKeyDown(KEY_A);
        camera.target.x+=move*(-forward*std::cos(yaw)+right*std::sin(yaw));camera.target.z+=move*(-forward*std::sin(yaw)-right*std::cos(yaw));camera.target.y+=move*(float(IsKeyDown(KEY_E))-IsKeyDown(KEY_Q));
        camera.position={camera.target.x+distance*std::cos(pitch)*std::cos(yaw),camera.target.y+distance*std::sin(pitch),camera.target.z+distance*std::cos(pitch)*std::sin(yaw)};
        state.mode=tab==0?1:tab==1?2:0;
        if(!paused){state.step(dt*timeScale);phase+=dt*timeScale*(0.35f+state.throttle);rotor=std::fmod(rotor+dt*timeScale*(tab==0?1+state.spool*9:state.rpm/650),10*PI);cycle=std::fmod(cycle+dt*timeScale*state.rpm/60*2*PI/60,4*PI);}
        BeginDrawing();ClearBackground(bg);BeginMode3D(camera);
        DrawCube({0,-0.68f,0},9,0.12f,8,Color{25,45,51,255});
        for(int i=-4;i<=4;++i){DrawLine3D({float(i),-0.61f,-4},{float(i),-0.61f,4},Color{37,60,65,255});DrawLine3D({-4,-0.61f,float(i)},{4,-0.61f,float(i)},Color{37,60,65,255});}
        if(tab==0)turbo(phase,rotor,cutaway,ribbons,particles);else if(tab==1)supercharger(phase,rotor,cutaway,ribbons,particles);else engine(phase,cycle,cutaway,ribbons,particles);
        EndMode3D();
        if(tab==0){label(camera,{0,2.6f,3.35f},"AIR IN",fresh);label(camera,{3.6f,1.15f,1.15f},"COMPRESSED AIR OUT",warm);label(camera,{3.6f,1.15f,-1.45f},"EXHAUST IN",hot);label(camera,{0,2.6f,-3.2f},"EXHAUST OUT",hot);label(camera,{0,2.6f,-0.2f},"SHARED SHAFT",paper);}
        else if(tab==1){label(camera,{0,3.35f,-2.8f},"AIR IN / REAR",fresh);label(camera,{0,0.25f,1.6f},"COMPRESSED AIR OUT",warm);label(camera,{-0.84f,2.95f,4.5f},"DRIVE PULLEY",paper);}

        else{label(camera,{-2.6f,4,0},"INTAKE AIR",fresh);label(camera,{2.6f,4,0},"EXHAUST GAS",hot);}
        studio::panel(sidebar,paper);studio::text("THE AIRFLOW LAB",38,39,13,teal);studio::text("Follow the air.",38,62,29,dark);studio::text("One component. Look inside.",38,106,14,muted);DrawLine(38,143,252,143,Color{204,213,199,255});
        slider(tab==2?"Engine speed":"Drive speed",170,state.rpm,800,7000,"rpm",dragButton>=0);float throttle=state.throttle*100;slider(tab==2?"Throttle":"Drive load",238,throttle,0,100,"%",dragButton>=0);state.throttle=throttle/100;
        studio::text("LOOK INSIDE",38,324,12,muted);studio::button(cutButton,cutaway?"Cutaway: open":"Cutaway: closed",cutaway,teal);studio::button(ribbonButton,"Ribbons",ribbons,teal);studio::button(particleButton,"Particles",particles,teal);
        studio::button({38,445,214,30},"Flow arrows [G]",airflowArrows,teal);
        studio::text("FLOW KEY",38,486,12,muted);DrawCircle(44,518,4,fresh);studio::text("Fresh intake air",57,510,14,dark);DrawCircle(44,544,4,warm);studio::text("Compressed / warmer",57,536,14,dark);DrawCircle(44,570,4,hot);studio::text("Hot exhaust gas",57,562,14,dark);studio::text("- / +   Change animation speed",38,588,12,muted);
        for(int i=0;i<3;++i)studio::button({294.0f+i*168,24,156,42},TextFormat("%d  %s",i+1,names[i]),tab==i,teal);
        studio::text(TextFormat("%s  /  %.3gx playback",paused?"Paused":"Live",timeScale),294,83,16,paper);
        float bottom=height-158.0f;
        const char* descriptions[]={"Exhaust spins the turbine. The shared shaft turns the air compressor.","Twin screw: helical rotors move air lengthwise; shrinking pockets illustrate internal compression.","A single-cylinder cutaway: air enters, compresses, expands, then leaves as exhaust."};
        studio::fit(descriptions[tab],{294,bottom,float(width-320),24},17,paper);
        studio::fit("Illustrative flow paths / no CFD. Rotor speeds are slowed; cylinder motion is slowed 60x.",{294,bottom+28,float(width-320),20},13,Color{163,189,181,255});
        if(tab==2){const char* strokes[]={"01  INTAKE","02  COMPRESSION","03  POWER","04  EXHAUST"};studio::text(strokes[int(cycle/PI)],294,120,20,int(cycle/PI)==0?fresh:warm);}
        else if(tab==0)studio::text(TextFormat("Outlet estimate  %.0f kPa abs  /  %.0f C",101.325f*state.ratio(),state.compressedT()-273.15f),294,120,16,warm);
        if(tab==1)studio::text("TWIN SCREW / 3-LOBE MALE + 5-GROOVE FEMALE",294,120,16,fresh);
        studio::fit("Drag: orbit   /   Shift + drag: pan   /   Scroll: zoom   /   WASD + QE: move   /   F: home",{294,bottom+52,float(width-320),20},13,paper);
        studio::panel(transport,paper);studio::button(pauseButton,paused?"Play":"Pause",!paused,teal);studio::button(resetButton,"Reset",false,teal);studio::button(helpButton,help?"Hide help":"Controls",help,teal);
        if(help){studio::panel({width-374.0f,160,352,250},paper);const char* lines[]={"Move through the experiment","Left / right drag or arrows   Orbit","Wheel   Zoom","Middle / Shift + left drag   Pan","W A S D / Q E   Move focus","F   Home view    T   Top view","Space   Pause    R   Reset tab","C   Cutaway   V / P / G   Airflow","1 / 2 / 3   Component tabs"};for(int i=0;i<9;++i)studio::text(lines[i],width-356,179+i*24,i?14:19,dark);}
        EndDrawing();
        if(smoke&&++frames>=12){Image capture=LoadImageFromScreen();bool ok=ExportImage(capture,TextFormat("/tmp/engine_airflow_tab_%d.png",tab));UnloadImage(capture);studio::unload();CloseWindow();return ok?0:1;}
        if(studio::smokeFrame(__FILE__))break;
    }
    studio::unload();CloseWindow();return 0;
}
