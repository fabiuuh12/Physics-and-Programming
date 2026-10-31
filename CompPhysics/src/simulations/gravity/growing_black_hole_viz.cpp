#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include "../../common/black_hole_growth.h"
#include "../../common/studio.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kWidth=1600,kHeight=960,kDiskCount=7200,kJetCount=1900,kSkyCount=900;
struct CameraRig {float yaw=.68f,pitch=-.32f,distance=15.5f;};
struct DiskParticle {float radius,angle,height,speed,size,heat,lane;};
struct JetParticle {float phase,side,radius,speed,size,twist;};
struct SkyParticle {Vector3 position;float size,brightness;Color color;};

float rr(std::mt19937& rng,float a,float b){return std::uniform_real_distribution<float>(a,b)(rng);}
float sat(float v){return std::clamp(v,0.0f,1.0f);}
float smooth(float a,float b,float x){x=sat((x-a)/(b-a));return x*x*(3-2*x);}
Color mix(Color a,Color b,float t){t=sat(t);return{(unsigned char)(a.r+(b.r-a.r)*t),(unsigned char)(a.g+(b.g-a.g)*t),(unsigned char)(a.b+(b.b-a.b)*t),(unsigned char)(a.a+(b.a-a.a)*t)};}

std::vector<DiskParticle> makeDisk(){
    std::mt19937 rng(8181);std::vector<DiskParticle> p;p.reserve(kDiskCount);
    for(int i=0;i<kDiskCount;i++){
        float u=rr(rng,0,1),r=1.48f+std::pow(u,.72f)*6.6f;
        p.push_back({r,rr(rng,0,2*PI),rr(rng,-1,1)*(.025f+.055f*r/8),.72f/std::pow(r,1.45f),rr(rng,.018f,.075f),1-r/8.3f,rr(rng,-1,1)});
    }return p;
}

std::vector<JetParticle> makeJets(){
    std::mt19937 rng(3939);std::vector<JetParticle> p;p.reserve(kJetCount);
    for(int i=0;i<kJetCount;i++)p.push_back({rr(rng,0,1),i%2?1.0f:-1.0f,rr(rng,0,1),rr(rng,.72f,1.35f),rr(rng,.025f,.09f),rr(rng,0,2*PI)});
    return p;
}

std::vector<SkyParticle> makeSky(){
    std::mt19937 rng(1919);std::vector<SkyParticle> p;p.reserve(kSkyCount);
    for(int i=0;i<kSkyCount;i++){
        float a=rr(rng,0,2*PI),z=rr(rng,-.9f,.9f),r=rr(rng,30,46),q=std::sqrt(1-z*z),temp=rr(rng,0,1);
        p.push_back({{r*q*std::cos(a),r*z,r*q*std::sin(a)},rr(rng,.018f,.055f),rr(rng,.2f,1),mix({132,178,255,255},{255,231,189,255},temp)});
    }return p;
}

Texture2D makeGlowTexture(){
    Image image=GenImageColor(64,64,BLANK);Color* px=(Color*)image.data;
    for(int y=0;y<64;y++)for(int x=0;x<64;x++){
        float dx=(x-31.5f)/31.5f,dy=(y-31.5f)/31.5f,r=std::sqrt(dx*dx+dy*dy);
        float a=sat(1-r);a=a*a*(3-2*a);px[y*64+x]={255,255,255,(unsigned char)(255*a)};
    }
    Texture2D texture=LoadTextureFromImage(image);UnloadImage(image);SetTextureFilter(texture,TEXTURE_FILTER_BILINEAR);return texture;
}

void billboard(Texture2D texture,Vector3 center,float size,Color color,Vector3 right,Vector3 up){
    static const float x[9]={1,.7071f,0,-.7071f,-1,-.7071f,0,.7071f,1};
    static const float y[9]={0,.7071f,1,.7071f,0,-.7071f,-1,-.7071f,0};
    for(int i=0;i<8;i++){
        Vector3 a=Vector3Add(center,Vector3Add(Vector3Scale(right,x[i]*size),Vector3Scale(up,y[i]*size)));
        Vector3 b=Vector3Add(center,Vector3Add(Vector3Scale(right,x[i+1]*size),Vector3Scale(up,y[i+1]*size)));
        rlColor4ub(color.r,color.g,color.b,color.a);rlVertex3f(center.x,center.y,center.z);
        rlColor4ub(color.r,color.g,color.b,0);rlVertex3f(a.x,a.y,a.z);rlVertex3f(b.x,b.y,b.z);
    }(void)texture;
}

void beginBillboards(Texture2D texture){rlSetTexture(0);rlBegin(RL_TRIANGLES);(void)texture;}
void endBillboards(){rlEnd();rlSetTexture(0);}

void cameraBasis(const Camera3D& camera,Vector3* right,Vector3* up){
    Vector3 f=Vector3Normalize(Vector3Subtract(camera.target,camera.position));*right=Vector3Normalize(Vector3CrossProduct(f,camera.up));*up=Vector3Normalize(Vector3CrossProduct(*right,f));
}

Vector3 diskPosition(const DiskParticle& p,float t,float mass){
    float contraction=.48f*mass*(1-p.radius/8.2f),r=p.radius-contraction;
    float a=p.angle+t*p.speed*(1+.48f*mass)+.10f*std::sin(p.lane*4+t*.2f);
    float thickness=p.height*(1-.72f*mass)+.018f*std::sin(a*5+p.lane*8);
    Vector3 v={r*std::cos(a),thickness,r*std::sin(a)};return Vector3RotateByAxisAngle(v,{1,0,0},.10f);
}

void drawSky(const std::vector<SkyParticle>& sky,Texture2D glow,Vector3 right,Vector3 up){
    rlDisableDepthMask();beginBillboards(glow);for(const auto& s:sky)billboard(glow,s.position,s.size,Fade(s.color,s.brightness),right,up);endBillboards();rlEnableDepthMask();
}

void drawDisk(const std::vector<DiskParticle>& disk,Texture2D glow,const Camera3D& camera,float t,float mass){
    int count=(int)(disk.size()*(.035f+.965f*blackhole::diskFraction(mass)));Vector3 right,up;cameraBasis(camera,&right,&up);
    BeginBlendMode(BLEND_ADDITIVE);rlDisableDepthMask();beginBillboards(glow);
    for(int i=0;i<count;i++){
        const auto& p=disk[i];Vector3 v=diskPosition(p,t,mass);float heat=sat(p.heat*.55f+mass*.72f);
        Color c=heat>.7f?mix({255,133,40,255},{218,238,255,255},(heat-.7f)/.3f):mix({170,49,25,255},{255,178,61,255},heat/.7f);
        float alpha=.22f+.58f*mass;billboard(glow,v,p.size*(1+.65f*mass),Fade(c,alpha),right,up);
    }endBillboards();rlEnableDepthMask();EndBlendMode();
}

void drawJets(const std::vector<JetParticle>& jets,Texture2D glow,const Camera3D& camera,float t,float power){
    if(power<.002f)return;Vector3 right,up;cameraBasis(camera,&right,&up);int count=(int)(jets.size()*(.15f+.85f*power));
    BeginBlendMode(BLEND_ADDITIVE);rlDisableDepthMask();beginBillboards(glow);
    for(int i=0;i<count;i++){
        const auto& p=jets[i];float q=std::fmod(p.phase+t*p.speed*.18f,1.0f),y=p.side*(1.15f+q*(7.5f+7.5f*power));
        float width=(.08f+.62f*q)*(.35f+.65f*power),a=p.twist+t*2.4f+p.side*q*13;
        Vector3 v={std::cos(a)*width*p.radius,y,std::sin(a)*width*p.radius};
        Color c=mix({105,173,255,255},{242,247,255,255},1-q);float fade=std::sin(PI*q)*power;
        billboard(glow,v,p.size*(.75f+power),Fade(c,.72f*fade),right,up);
    }endBillboards();rlEnableDepthMask();EndBlendMode();
    for(int side:{-1,1})for(int i=0;i<4;i++){
        float length=(3.5f+i*2.2f)*power;DrawCylinderEx({0,side*1.0f,0},{0,side*(1.0f+length),0},.12f+.06f*i,.035f+.02f*i,18,Fade({104,172,255,255},.025f*power));
    }
}

void drawFeedingClouds(Texture2D glow,const Camera3D& camera,float t,float mass,bool feeding){
    if(!feeding)return;Vector3 right,up;cameraBasis(camera,&right,&up);BeginBlendMode(BLEND_ADDITIVE);rlDisableDepthMask();beginBillboards(glow);
    for(int i=0;i<150;i++){
        float q=std::fmod(i*.618034f+t*.24f,1.0f),a=2*PI*(i*.381966f)+t*.12f,r=11.2f-3.0f*q;
        Vector3 v={r*std::cos(a),.45f*std::sin(i*1.7f)*(1-q),r*std::sin(a)};
        billboard(glow,v,.055f+.06f*q,Fade({255,119,42,255},.20f+.55f*q),right,up);
    }endBillboards();rlEnableDepthMask();EndBlendMode();(void)mass;
}

void drawBlackHoleScreen(const Camera3D& camera,float mass){
    Vector2 c=GetWorldToScreen({0,0,0},camera),edge=GetWorldToScreen({(float)blackhole::horizonRadius(mass),0,0},camera);
    float r=std::max(22.0f,Vector2Distance(c,edge));
    DrawCircleGradient((int)c.x,(int)c.y,r*2.15f,Fade({69,105,255,255},.06f),BLANK);
    DrawRing(c,r*1.30f,r*1.43f,0,360,180,Fade({255,170,69,255},.16f+.22f*mass));
    DrawRing(c,r*1.18f,r*1.29f,0,360,180,Fade({255,241,192,255},.48f+.38f*mass));
    DrawCircleV(c,r,{0,0,1,255});
    DrawRing(c,r*.97f,r*1.035f,0,360,180,{3,4,8,255});
}

void updateCamera(Camera3D* c,CameraRig* r){
    studio::pan(c,r->yaw,r->pitch,r->distance);if(IsMouseButtonDown(MOUSE_BUTTON_LEFT)&&!studio::panGesture()){
        Vector2 d=GetMouseDelta();r->yaw-=d.x*.0042f;r->pitch=std::clamp(r->pitch+d.y*.0042f,-1.42f,1.42f);
    }r->distance=std::clamp(r->distance*std::exp(-GetMouseWheelMove()*.08f),5.0f,32.0f);
    float cp=std::cos(r->pitch);c->position=Vector3Add(c->target,{r->distance*cp*std::cos(r->yaw),r->distance*std::sin(r->pitch),r->distance*cp*std::sin(r->yaw)});
}

void preset(int key,CameraRig* r,Camera3D* c){
    if(key==KEY_ONE){*r={.68f,-.32f,15.5f};c->target={0,0,0};}
    if(key==KEY_TWO){*r={1.56f,-1.24f,15.0f};c->target={0,0,0};}
    if(key==KEY_THREE){*r={1.56f,-.04f,12.0f};c->target={0,1.8f,0};}
}

void drawHud(float mass,float speed,bool paused,bool feeding,bool blazarView,float fps){
    float disk=blackhole::diskFraction(mass),jet=blackhole::jetPower(mass);
    studio::title("Growing Black Hole",studio::Style::Observatory);
    const char* activity=blazarView?"BLAZAR VIEW — JET ALIGNED":blackhole::stateName(mass);
    studio::readout(TextFormat("%s   |   mass reservoir %.0f%%   |   disk %.0f%%   |   jet power %.0f%%",activity,mass*100,disk*100,jet*100));
    studio::note("Feed matter into the gravity well and watch the accretion state evolve");
    Rectangle panel={GetScreenWidth()-312.0f,86,284,196};studio::panel(panel,{11,19,32,236});
    studio::text("ACCRETION CONTROL",panel.x+18,panel.y+16,13,{231,190,128,255});
    Rectangle feed={panel.x+18,panel.y+48,panel.width-36,48};bool over=studio::over(feed);Color fc=feeding?Color{219,119,48,255}:(over?Color{63,77,94,255}:Color{37,50,66,255});
    DrawRectangleRounded(feed,.18f,8,fc);studio::text(feeding?"FEEDING MATTER...":"HOLD TO FEED",feed.x+27,feed.y+14,17,{248,242,226,255});
    studio::text(TextFormat("Horizon radius  %.2f",blackhole::horizonRadius(mass)),panel.x+18,panel.y+113,15,{195,207,220,255});
    studio::text(TextFormat("Disk temperature  %.0f%%",blackhole::temperature(mass)*100),panel.x+18,panel.y+138,15,{255,181,91,255});
    const char* jetText=blazarView?"Jet aimed toward observer":(jet>.05f?"Polar magnetic jets launched":"Jets waiting for high accretion");
    studio::text(jetText,panel.x+18,panel.y+163,15,jet>.05f?Color{139,193,255,255}:Color{133,151,169,255});
    studio::help("Hold F or button: feed | Drag: orbit | Wheel: zoom | 1/2/3: views | Space: pause | R: reset | -/+: speed");
    studio::fps();(void)speed;(void)paused;(void)fps;
}

} // namespace

int main(int argc,char** argv){
    float startMass=.07f;std::string capture;int captureFrames=3;
    for(int i=1;i<argc;i++){
        std::string a=argv[i];if(a=="--mass"&&i+1<argc)startMass=std::clamp(std::strtof(argv[++i],nullptr),0.0f,1.0f);
        else if(a=="--capture"&&i+1<argc)capture=argv[++i];else if(a=="--frames"&&i+1<argc)captureFrames=std::clamp(std::atoi(argv[++i]),1,1200);
        else{std::fprintf(stderr,"Unknown or incomplete option: %s\n",a.c_str());return 2;}
    }
    SetConfigFlags(FLAG_WINDOW_RESIZABLE|FLAG_MSAA_4X_HINT);if(std::getenv("COMPPHYSICS_SMOKE_FRAMES"))SetConfigFlags(FLAG_WINDOW_HIDDEN);
    InitWindow(kWidth,kHeight,"Growing Black Hole - C++ (raylib)");SetWindowMinSize(1000,650);SetTargetFPS(120);
    Texture2D glow=makeGlowTexture();auto disk=makeDisk();auto jets=makeJets();auto sky=makeSky();
    Camera3D camera{{0,0,0},{0,0,0},{0,1,0},47,CAMERA_PERSPECTIVE};CameraRig rig;preset(KEY_ONE,&rig,&camera);
    float mass=startMass,time=0,speed=1;bool paused=false;int frame=0;
    while(!WindowShouldClose()){
        Rectangle feedButton={GetScreenWidth()-294.0f,134,248,48};bool feeding=IsKeyDown(KEY_F)||(studio::over(feedButton)&&IsMouseButtonDown(MOUSE_BUTTON_LEFT));
        if(IsKeyPressed(KEY_SPACE)||IsKeyPressed(KEY_P))paused=!paused;if(IsKeyPressed(KEY_R)){mass=.07f;time=0;paused=false;}
        if(IsKeyPressed(KEY_MINUS)||IsKeyPressed(KEY_KP_SUBTRACT))speed=std::max(.25f,speed-.25f);if(IsKeyPressed(KEY_EQUAL)||IsKeyPressed(KEY_KP_ADD))speed=std::min(4.0f,speed+.25f);
        for(int key:{KEY_ONE,KEY_TWO,KEY_THREE})if(IsKeyPressed(key))preset(key,&rig,&camera);updateCamera(&camera,&rig);
        float dt=std::min(GetFrameTime(),.05f)*speed;if(!paused&&capture.empty()){time+=dt;if(feeding)mass=(float)blackhole::feed(mass,.105,dt);}
        Vector3 right,up;cameraBasis(camera,&right,&up);
        BeginDrawing();ClearBackground({2,4,10,255});BeginMode3D(camera);
        drawSky(sky,glow,right,up);drawFeedingClouds(glow,camera,time,mass,feeding);drawDisk(disk,glow,camera,time,mass);drawJets(jets,glow,camera,time,(float)blackhole::jetPower(mass));EndMode3D();
        bool blazarView=blackhole::jetPower(mass)>.05&&std::fabs(rig.pitch)>1.05f;
        drawBlackHoleScreen(camera,mass);DrawRectangleGradientV(0,0,GetScreenWidth(),150,{2,4,10,220},{2,4,10,0});drawHud(mass,speed,paused,feeding,blazarView,(float)GetFPS());EndDrawing();
        if(!capture.empty()&&++frame>=captureFrames){std::filesystem::create_directories(std::filesystem::path(capture).parent_path());TakeScreenshot(capture.c_str());break;}
        if(studio::smokeFrame(__FILE__))break;
    }
    UnloadTexture(glow);studio::unload();CloseWindow();return 0;
}
