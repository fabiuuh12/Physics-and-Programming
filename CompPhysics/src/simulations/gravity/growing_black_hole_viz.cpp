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
        p.push_back({r,rr(rng,0,2*PI),rr(rng,-1,1)*(.004f+.012f*r/8),.72f/std::pow(r,1.45f),rr(rng,.009f,.043f),1-r/8.3f,rr(rng,-1,1)});
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

float visualInnerRadius(float mass,float spin){
    float normalized=sat(((float)blackhole::iscoRadius(spin)-1.24f)/(9.0f-1.24f));
    return 1.34f+.08f*mass+.68f*normalized;
}

Vector3 diskPosition(const DiskParticle& p,float t,float mass,float spin){
    float contraction=.48f*mass*(1-p.radius/8.2f),r=p.radius-contraction;
    float frameDrag=spin*t*.34f/(r*r+.35f);
    float a=p.angle+t*p.speed*(1+.48f*mass)+frameDrag+.10f*std::sin(p.lane*4+t*.2f);
    float thickness=p.height*(1-.72f*mass)+.018f*std::sin(a*5+p.lane*8);
    Vector3 v={r*std::cos(a),thickness,r*std::sin(a)};return Vector3RotateByAxisAngle(v,{1,0,0},.10f);
}

float diskTemperature(float radius,float inner,float outer){
    if(radius<=inner||radius>=outer)return 0;float x=radius/inner;
    return sat(std::pow(x,-.75f)*std::pow(std::max(0.0f,1-std::sqrt(1/x)),.25f)*2.35f);
}

float dopplerFactor(Vector3 position,const Camera3D& camera,float mass){
    Vector3 tangent=Vector3Normalize({-position.z,0,position.x});
    Vector3 toCamera=Vector3Normalize(Vector3Subtract(camera.position,position));
    return std::clamp(1.0f+.72f*mass*Vector3DotProduct(tangent,toCamera),.38f,1.72f);
}

Color diskSpectrum(float temperature,float doppler,float alpha){
    Color thermal=temperature>.72f?mix({255,174,68,255},{220,238,255,255},(temperature-.72f)/.28f):mix({116,24,14,255},{255,142,39,255},temperature/.72f);
    if(doppler>1)thermal=mix(thermal,{164,215,255,255},(doppler-1)/.72f);else thermal=mix(thermal,{255,55,24,255},(1-doppler)/.62f);
    float boost=std::clamp(std::pow(doppler,3.0f),.20f,2.6f);return Fade(thermal,sat(alpha*boost));
}

void diskVertex(Vector3 p,Color c){rlColor4ub(c.r,c.g,c.b,c.a);rlVertex3f(p.x,p.y,p.z);}

void drawContinuousDisk(const Camera3D& camera,float t,float mass,float spin){
    float amount=(float)blackhole::diskFraction(mass);if(amount<.01f)return;
    constexpr int angular=180,radial=40;float inner=visualInnerRadius(mass,spin),outer=2.4f+5.75f*amount;
    BeginBlendMode(BLEND_ADDITIVE);rlDisableDepthMask();rlDisableBackfaceCulling();rlSetTexture(0);rlBegin(RL_TRIANGLES);
    for(int j=0;j<radial;j++)for(int i=0;i<angular;i++){
        float r0=inner+(outer-inner)*j/radial,r1=inner+(outer-inner)*(j+1)/radial;
        float a0=2*PI*i/angular,a1=2*PI*(i+1)/angular;
        Vector3 p00={r0*std::cos(a0),.003f*std::sin(a0*7+t),r0*std::sin(a0)},p01={r0*std::cos(a1),.003f*std::sin(a1*7+t),r0*std::sin(a1)};
        Vector3 p10={r1*std::cos(a0),.003f*std::sin(a0*7+t),r1*std::sin(a0)},p11={r1*std::cos(a1),.003f*std::sin(a1*7+t),r1*std::sin(a1)};
        float drag0=spin*t*.34f/(r0*r0+.35f),drag1=spin*t*.34f/(r1*r1+.35f);
        float t00=.82f+.18f*std::sin((a0+drag0)*5-r0*3.4f+t*(.8f/std::pow(r0,1.3f))),t01=.82f+.18f*std::sin((a1+drag0)*5-r0*3.4f+t*(.8f/std::pow(r0,1.3f)));
        float t10=.82f+.18f*std::sin((a0+drag1)*5-r1*3.4f+t*(.8f/std::pow(r1,1.3f))),t11=.82f+.18f*std::sin((a1+drag1)*5-r1*3.4f+t*(.8f/std::pow(r1,1.3f)));
        Color c00=diskSpectrum(diskTemperature(r0,inner,outer),dopplerFactor(p00,camera,mass),(.025f+.13f*mass)*t00);
        Color c01=diskSpectrum(diskTemperature(r0,inner,outer),dopplerFactor(p01,camera,mass),(.025f+.13f*mass)*t01);
        Color c10=diskSpectrum(diskTemperature(r1,inner,outer),dopplerFactor(p10,camera,mass),(.018f+.10f*mass)*t10);
        Color c11=diskSpectrum(diskTemperature(r1,inner,outer),dopplerFactor(p11,camera,mass),(.018f+.10f*mass)*t11);
        diskVertex(p00,c00);diskVertex(p10,c10);diskVertex(p11,c11);diskVertex(p00,c00);diskVertex(p11,c11);diskVertex(p01,c01);
    }
    rlEnd();rlEnableBackfaceCulling();rlEnableDepthMask();EndBlendMode();
}

void drawSky(const std::vector<SkyParticle>& sky,Texture2D glow,Vector3 right,Vector3 up){
    rlDisableDepthMask();beginBillboards(glow);for(const auto& s:sky)billboard(glow,s.position,s.size,Fade(s.color,s.brightness),right,up);endBillboards();rlEnableDepthMask();
}

void drawDisk(const std::vector<DiskParticle>& disk,Texture2D glow,const Camera3D& camera,float t,float mass,float spin){
    int count=(int)(disk.size()*(.035f+.965f*blackhole::diskFraction(mass)));Vector3 right,up;cameraBasis(camera,&right,&up);
    BeginBlendMode(BLEND_ADDITIVE);rlDisableDepthMask();beginBillboards(glow);
    for(int i=0;i<count;i++){
        const auto& p=disk[i];Vector3 v=diskPosition(p,t,mass,spin);float inner=visualInnerRadius(mass,spin),outer=2.4f+5.75f*blackhole::diskFraction(mass);
        float heat=diskTemperature(Vector3Length(v),inner,outer),doppler=dopplerFactor(v,camera,mass);
        Color c=diskSpectrum(heat,doppler,.25f+.48f*mass);billboard(glow,v,p.size*(.72f+.55f*mass),c,right,up);
    }endBillboards();rlEnableDepthMask();EndBlendMode();
}

void drawJets(const std::vector<JetParticle>& jets,Texture2D glow,const Camera3D& camera,float t,float power){
    if(power<.002f)return;Vector3 right,up;cameraBasis(camera,&right,&up);int count=(int)(jets.size()*(.15f+.85f*power));
    Vector3 toCamera=Vector3Normalize(camera.position);
    BeginBlendMode(BLEND_ADDITIVE);rlDisableDepthMask();beginBillboards(glow);
    for(int i=0;i<count;i++){
        const auto& p=jets[i];float q=std::fmod(p.phase+t*p.speed*.18f,1.0f),y=p.side*(1.15f+q*(7.5f+7.5f*power));
        float width=(.07f+1.02f*std::pow(q,.82f))*(.30f+.70f*power),a=p.twist+t*2.4f+p.side*q*13;
        Vector3 v={std::cos(a)*width*p.radius,y,std::sin(a)*width*p.radius};
        float beam=std::clamp(1.0f+1.05f*p.side*toCamera.y,.28f,2.0f);
        Color c=mix({91,137,255,255},{242,249,255,255},1-q);float fade=std::sin(PI*q)*power*beam;
        billboard(glow,v,p.size*(.75f+power),Fade(c,.72f*fade),right,up);
    }endBillboards();rlEnableDepthMask();EndBlendMode();
    // Magnetically braided core, expanding sheath, and recollimation shocks.
    BeginBlendMode(BLEND_ADDITIVE);rlDisableDepthMask();rlSetTexture(0);rlBegin(RL_LINES);
    for(int side:{-1,1}){
        float beam=std::clamp(1.0f+1.05f*side*toCamera.y,.28f,2.0f),length=4.0f+10.5f*power;
        for(int strand=0;strand<5;strand++){
            Vector3 previous={0,side*1.02f,0};
            for(int i=1;i<=150;i++){
                float q=i/150.0f,radius=(.035f+.62f*std::pow(q,.86f))*power;
                float angle=strand*2*PI/5+side*q*(13.0f+4.0f*power)-t*2.5f*side;
                Vector3 next={std::cos(angle)*radius,side*(1.02f+q*length),std::sin(angle)*radius};
                Color c=mix({74,119,255,255},{225,246,255,255},1-q);
                c=Fade(c,sat((.16f+.34f*(1-q))*power*beam));
                rlColor4ub(c.r,c.g,c.b,c.a);rlVertex3f(previous.x,previous.y,previous.z);rlVertex3f(next.x,next.y,next.z);previous=next;
            }
        }
        for(int shock=1;shock<=5;shock++){
            float q=shock/5.5f,radius=(.12f+.72f*q)*power*(1+.18f*std::sin(t*4-shock));
            for(int i=0;i<56;i++){
                float a0=2*PI*i/56,a1=2*PI*(i+1)/56;float y=side*(1.02f+q*length);
                Color c=Fade({190,224,255,255},sat(.13f*power*beam*(1-q*.45f)));rlColor4ub(c.r,c.g,c.b,c.a);
                rlVertex3f(std::cos(a0)*radius,y,std::sin(a0)*radius);rlVertex3f(std::cos(a1)*radius,y,std::sin(a1)*radius);
            }
        }
    }
    rlEnd();rlEnableDepthMask();EndBlendMode();
}

void drawFeedingClouds(Texture2D glow,const Camera3D& camera,float t,float mass,bool feeding){
    if(!feeding)return;Vector3 right,up;cameraBasis(camera,&right,&up);BeginBlendMode(BLEND_ADDITIVE);rlDisableDepthMask();beginBillboards(glow);
    for(int i=0;i<150;i++){
        float q=std::fmod(i*.618034f+t*.24f,1.0f),a=2*PI*(i*.381966f)+t*.12f,r=11.2f-3.0f*q;
        Vector3 v={r*std::cos(a),.45f*std::sin(i*1.7f)*(1-q),r*std::sin(a)};
        billboard(glow,v,.055f+.06f*q,Fade({255,119,42,255},.20f+.55f*q),right,up);
    }endBillboards();rlEnableDepthMask();EndBlendMode();(void)mass;
}

const char* kLensFragment=R"GLSL(
#version 330
in vec2 fragTexCoord;
in vec4 fragColor;
uniform sampler2D texture0;
uniform vec2 resolution;
uniform vec2 lensCenter;
uniform float einsteinRadius;
uniform float shadowRadius;
uniform float photonRadius;
uniform float lensStrength;
uniform float luminosity;
uniform float ignition;
uniform float spin;
uniform float proximity;
uniform float radialVelocity;
out vec4 finalColor;
void main(){
    vec2 pixel=fragTexCoord*resolution;
    vec2 d=pixel-lensCenter;
    float r2=max(dot(d,d),4.0);
    float r=sqrt(r2);
    vec2 shadowD=d-vec2(spin*shadowRadius*.16,0.0);
    float shadowR=length(shadowD);
    vec2 source=pixel-lensStrength*einsteinRadius*einsteinRadius*d/r2;
    source+=spin*vec2(-d.y,d.x)*(shadowRadius*shadowRadius/r2)*.22;
    source+=d*radialVelocity*proximity*.032;
    vec2 uv=source/resolution;
    vec3 base=texture(texture0,fragTexCoord).rgb;
    vec3 warped=texture(texture0,clamp(uv,vec2(0.001),vec2(0.999))).rgb;
    vec2 radial=normalize(d+vec2(.001))/resolution;
    float streak=abs(radialVelocity)*proximity*8.0;
    warped+=texture(texture0,clamp(uv+radial*streak,vec2(.001),vec2(.999))).rgb*.34;
    warped+=texture(texture0,clamp(uv-radial*streak,vec2(.001),vec2(.999))).rgb*.34;
    warped/=1.68;
    float influence=1.0-smoothstep(einsteinRadius*1.65,einsteinRadius*3.25,r);
    vec3 color=mix(base,warped,influence);
    float critical=exp(-pow((r-einsteinRadius)/max(1.3,einsteinRadius*.028),2.0));
    color+=critical*vec3(0.035,0.032,0.028);
    float photon=exp(-pow((shadowR-photonRadius)/max(1.0,photonRadius*.012),2.0));
    float subring=exp(-pow((shadowR-photonRadius*1.055)/max(1.0,photonRadius*.022),2.0));
    color+=photon*vec3(1.20,0.91,0.61)+subring*vec3(.16,.12,.09);
    vec2 texel=1.0/resolution;
    float spread=2.0+7.0*luminosity;
    vec3 bloom=vec3(0.0);
    bloom+=max(texture(texture0,fragTexCoord+vec2( spread,0.0)*texel).rgb-vec3(.48),vec3(0.0));
    bloom+=max(texture(texture0,fragTexCoord+vec2(-spread,0.0)*texel).rgb-vec3(.48),vec3(0.0));
    bloom+=max(texture(texture0,fragTexCoord+vec2(0.0, spread)*texel).rgb-vec3(.48),vec3(0.0));
    bloom+=max(texture(texture0,fragTexCoord+vec2(0.0,-spread)*texel).rgb-vec3(.48),vec3(0.0));
    bloom+=max(texture(texture0,fragTexCoord+vec2( spread, spread)*texel).rgb-vec3(.56),vec3(0.0));
    bloom+=max(texture(texture0,fragTexCoord+vec2(-spread,-spread)*texel).rgb-vec3(.56),vec3(0.0));
    color+=bloom*(.16+.30*luminosity)+ignition*photon*vec3(2.2,1.7,1.1);
    float peripheral=smoothstep(einsteinRadius*.72,einsteinRadius*2.7,r);
    color*=mix(vec3(1.0),vec3(1.20,.72,.54),proximity*peripheral*.72);
    color*=mix(vec3(1.0),vec3(.70,.90,1.32),proximity*(1.0-peripheral)*.52);
    float motionShift=radialVelocity*proximity;
    color*=mix(vec3(1.0),motionShift>=0.0?vec3(.72,.91,1.38):vec3(1.35,.82,.63),abs(motionShift)*.42);
    float shadow=smoothstep(shadowRadius*.955,shadowRadius*1.025,shadowR);
    color=mix(vec3(.0015,.0025,.005),color,shadow);
    finalColor=vec4(color,1.0)*fragColor;
})GLSL";

struct LensPass {Shader shader{};int resolution=-1,center=-1,einstein=-1,shadow=-1,photon=-1,strength=-1,luminosity=-1,ignition=-1,spin=-1,proximity=-1,radialVelocity=-1;};

LensPass makeLensPass(){
    LensPass p;p.shader=LoadShaderFromMemory(nullptr,kLensFragment);p.resolution=GetShaderLocation(p.shader,"resolution");p.center=GetShaderLocation(p.shader,"lensCenter");
    p.einstein=GetShaderLocation(p.shader,"einsteinRadius");p.shadow=GetShaderLocation(p.shader,"shadowRadius");p.photon=GetShaderLocation(p.shader,"photonRadius");p.strength=GetShaderLocation(p.shader,"lensStrength");
    p.luminosity=GetShaderLocation(p.shader,"luminosity");p.ignition=GetShaderLocation(p.shader,"ignition");p.spin=GetShaderLocation(p.shader,"spin");p.proximity=GetShaderLocation(p.shader,"proximity");p.radialVelocity=GetShaderLocation(p.shader,"radialVelocity");return p;
}

void drawLensedScene(RenderTexture2D scene,LensPass& lens,const Camera3D& camera,float mass,float spin,float ignition,float radialVelocity){
    Vector3 forward=Vector3Normalize(Vector3Subtract(camera.target,camera.position));
    Vector3 screenRight=Vector3Normalize(Vector3CrossProduct(forward,camera.up));
    Vector2 center=GetWorldToScreen({0,0,0},camera),edge=GetWorldToScreen(Vector3Scale(screenRight,(float)blackhole::horizonRadius(mass)),camera);
    float sx=scene.texture.width/(float)GetScreenWidth(),sy=scene.texture.height/(float)GetScreenHeight(),pixelScale=std::sqrt(sx*sy);
    float shadow=std::max(18.0f,Vector2Distance(center,edge))*pixelScale,einstein=shadow*(2.08f+.12f*mass),photon=shadow*1.055f;
    Vector2 resolution={(float)scene.texture.width,(float)scene.texture.height},textureCenter={center.x*sx,(GetScreenHeight()-center.y)*sy};float strength=.46f+.08f*mass;
    SetShaderValue(lens.shader,lens.resolution,&resolution,SHADER_UNIFORM_VEC2);SetShaderValue(lens.shader,lens.center,&textureCenter,SHADER_UNIFORM_VEC2);
    SetShaderValue(lens.shader,lens.einstein,&einstein,SHADER_UNIFORM_FLOAT);SetShaderValue(lens.shader,lens.shadow,&shadow,SHADER_UNIFORM_FLOAT);
    SetShaderValue(lens.shader,lens.photon,&photon,SHADER_UNIFORM_FLOAT);SetShaderValue(lens.shader,lens.strength,&strength,SHADER_UNIFORM_FLOAT);
    float luminosity=(float)blackhole::luminosity(mass);SetShaderValue(lens.shader,lens.luminosity,&luminosity,SHADER_UNIFORM_FLOAT);SetShaderValue(lens.shader,lens.ignition,&ignition,SHADER_UNIFORM_FLOAT);
    float observerDistance=Vector3Distance(camera.position,{0,0,0})/(float)blackhole::horizonRadius(mass);
    float proximity=1-smooth(1.75f,5.5f,observerDistance);SetShaderValue(lens.shader,lens.spin,&spin,SHADER_UNIFORM_FLOAT);SetShaderValue(lens.shader,lens.proximity,&proximity,SHADER_UNIFORM_FLOAT);SetShaderValue(lens.shader,lens.radialVelocity,&radialVelocity,SHADER_UNIFORM_FLOAT);
    BeginShaderMode(lens.shader);DrawTexturePro(scene.texture,{0,0,(float)scene.texture.width,-(float)scene.texture.height},{0,0,(float)GetScreenWidth(),(float)GetScreenHeight()},{0,0},0,WHITE);EndShaderMode();
}

void updateCamera(Camera3D* c,CameraRig* r,float minimumDistance){
    studio::pan(c,r->yaw,r->pitch,r->distance);if(IsMouseButtonDown(MOUSE_BUTTON_LEFT)&&!studio::panGesture()){
        Vector2 d=GetMouseDelta();r->yaw-=d.x*.0042f;r->pitch=std::remainder(r->pitch+d.y*.0042f,2*PI);
    }r->distance=std::clamp(r->distance*std::exp(-GetMouseWheelMove()*.08f),minimumDistance,32.0f);
    float cp=std::cos(r->pitch);c->position=Vector3Add(c->target,{r->distance*cp*std::cos(r->yaw),r->distance*std::sin(r->pitch),r->distance*cp*std::sin(r->yaw)});
    c->up={-std::sin(r->pitch)*std::cos(r->yaw),std::cos(r->pitch),-std::sin(r->pitch)*std::sin(r->yaw)};
}

void preset(int key,CameraRig* r,Camera3D* c){
    if(key==KEY_ONE){*r={.68f,-.32f,15.5f};c->target={0,0,0};}
    if(key==KEY_TWO){*r={1.56f,-1.24f,15.0f};c->target={0,0,0};}
    if(key==KEY_THREE){*r={1.56f,-.04f,12.0f};c->target={0,1.8f,0};}
}

float effectiveJetPower(float mass,float spin){return sat((float)blackhole::jetPower(mass)*(.65f+.65f*(float)blackhole::spinJetMultiplier(spin)));}

Rectangle controlPanel(){return{GetScreenWidth()-352.0f,64,324,392};}
Rectangle feedControl(){Rectangle p=controlPanel();return{p.x+18,p.y+48,137,46};}
Rectangle rewindControl(){Rectangle p=controlPanel();return{p.x+163,p.y+48,143,46};}
Rectangle spinDownControl(){Rectangle p=controlPanel();return{p.x+240,p.y+313,28,28};}
Rectangle spinUpControl(){Rectangle p=controlPanel();return{p.x+278,p.y+313,28,28};}

void metricBar(Rectangle panel,float y,const char* label,float value,Color color){
    value=sat(value);studio::text(label,panel.x+18,y,13,{163,179,195,255});
    studio::text(TextFormat("%3.0f%%",value*100),panel.x+266,y,13,{224,231,237,255});
    Rectangle track={panel.x+18,y+21,panel.width-36,7};DrawRectangleRounded(track,1,8,{29,42,56,255});
    if(value>.002f)DrawRectangleRounded({track.x,track.y,track.width*value,track.height},1,8,color);
    DrawCircleV({track.x+track.width*value,track.y+track.height*.5f},4,Fade(color,.85f));
}

void drawHud(float mass,float spin,float observerDistance,float radialVelocity,float speed,bool paused,bool feeding,bool rewinding,bool blazarView,float renderScale,float fps){
    float disk=blackhole::diskFraction(mass),jet=effectiveJetPower(mass,spin);
    const char* activity=blazarView?"BLAZAR VIEW - JET ALIGNED":blackhole::stateName(mass);
    studio::text("OBSERVATORY  /  GENERAL RELATIVITY",26,18,11,{205,166,109,255});
    studio::text("Growing Black Hole",26,39,28,{239,241,240,255});DrawLine(26,78,146,78,{205,166,109,255});
    float badgeWidth=studio::textWidth(activity,14)+28;Rectangle badge={26,93,badgeWidth,32};
    DrawRectangleRounded(badge,.22f,8,{18,29,43,232});DrawRectangleRoundedLinesEx(badge,.22f,8,1,Fade({231,190,128,255},.45f));
    DrawCircle(42,109,3,{231,190,128,255});studio::text(activity,53,101,14,{235,211,170,255});
    studio::text(paused?"SIMULATION PAUSED":"LIVE ACCRETION MODEL",26,135,11,paused?Color{255,183,99,255}:Color{126,185,218,255});
    Rectangle observer={26,158,270,78};DrawRectangleRounded(observer,.08f,8,{9,17,28,218});DrawRectangleRoundedLinesEx(observer,.08f,8,1,{48,67,86,175});
    studio::text("OBSERVER JOURNEY",observer.x+14,observer.y+11,10,{126,185,218,255});
    studio::text(blackhole::observerRegime(observerDistance),observer.x+14,observer.y+29,15,{226,232,236,255});
    const char* motion=radialVelocity>.08f?"INBOUND / BLUE SHIFT":(radialVelocity<-.08f?"OUTBOUND / RED SHIFT":"ORBIT HOLD");
    Color motionColor=radialVelocity>.08f?Color{130,190,255,255}:(radialVelocity<-.08f?Color{255,142,105,255}:Color{146,160,175,255});
    studio::text(TextFormat("dTau/dt %.3f",blackhole::properTimeRate(observerDistance)),observer.x+14,observer.y+53,11,{157,173,189,255});
    studio::text(motion,observer.x+104,observer.y+53,11,motionColor);

    Rectangle panel=controlPanel();
    DrawRectangleRounded({panel.x+2,panel.y+5,panel.width,panel.height},.055f,10,{0,0,0,65});
    DrawRectangleRounded(panel,.055f,10,{9,17,28,239});DrawRectangleRoundedLinesEx(panel,.055f,10,1,{55,73,91,190});
    studio::text("SYSTEM EVOLUTION",panel.x+18,panel.y+16,12,{231,190,128,255});
    Rectangle feed=feedControl(),rewind=rewindControl();
    Color fc=feeding?Color{219,119,48,255}:(studio::over(feed)?Color{63,77,94,255}:Color{37,50,66,255});
    Color rc=rewinding?Color{67,125,174,255}:(studio::over(rewind)?Color{63,77,94,255}:Color{37,50,66,255});
    DrawRectangleRounded(feed,.18f,8,fc);DrawRectangleRounded(rewind,.18f,8,rc);
    studio::text(feeding?"FEEDING...":"FEED  +",feed.x+29,feed.y+13,15,{248,242,226,255});
    studio::text(rewinding?"REWINDING...":"REWIND  -",rewind.x+24,rewind.y+13,15,{226,239,249,255});
    metricBar(panel,panel.y+108,"MASS RESERVOIR",mass,{222,126,54,255});
    metricBar(panel,panel.y+153,"ACCRETION DISK",disk,{236,178,79,255});
    metricBar(panel,panel.y+198,"RADIATIVE OUTPUT",(float)blackhole::luminosity(mass),{225,232,245,255});
    metricBar(panel,panel.y+243,"JET POWER",jet,{105,169,245,255});
    studio::text(TextFormat("KERR SPIN  %+.2f",spin),panel.x+18,panel.y+315,13,{183,202,236,255});
    Rectangle spinDown=spinDownControl(),spinUp=spinUpControl();
    studio::button(spinDown,"-",false,{82,119,167,255});studio::button(spinUp,"+",false,{82,119,167,255});
    Rectangle spinTrack={panel.x+18,panel.y+349,210,5};DrawRectangleRounded(spinTrack,1,8,{34,48,64,255});
    float spin01=(spin+1)*.5f;DrawLineEx({spinTrack.x+spinTrack.width*.5f,spinTrack.y-4},{spinTrack.x+spinTrack.width*.5f,spinTrack.y+9},1,{94,111,128,255});
    DrawCircleV({spinTrack.x+spinTrack.width*spin01,spinTrack.y+2.5f},6,{166,196,236,255});
    studio::text(TextFormat("ISCO %.2f GM/c^2",blackhole::iscoRadius(spin)),panel.x+18,panel.y+367,12,{153,169,185,255});
    studio::text(TextFormat("OBSERVER %.1f R_H",observerDistance),panel.x+170,panel.y+367,12,{153,169,185,255});
    studio::help(TextFormat("F/D: feed/rewind | Z/X: Kerr spin | Drag: unlimited orbit | Wheel: zoom | Q: quality %.0f%% | Space: pause",renderScale*100));
    studio::fps();(void)speed;(void)paused;(void)fps;
}

} // namespace

int main(int argc,char** argv){
    float startMass=.07f,startSpin=.72f,startDistance=-1,startRadialVelocity=0;std::string capture;int captureFrames=3;
    for(int i=1;i<argc;i++){
        std::string a=argv[i];if(a=="--mass"&&i+1<argc)startMass=std::clamp(std::strtof(argv[++i],nullptr),0.0f,1.0f);
        else if(a=="--spin"&&i+1<argc)startSpin=std::clamp(std::strtof(argv[++i],nullptr),-.998f,.998f);
        else if(a=="--distance"&&i+1<argc)startDistance=std::clamp(std::strtof(argv[++i],nullptr),1.5f,32.0f);
        else if(a=="--radial-velocity"&&i+1<argc)startRadialVelocity=std::clamp(std::strtof(argv[++i],nullptr),-1.0f,1.0f);
        else if(a=="--capture"&&i+1<argc)capture=argv[++i];else if(a=="--frames"&&i+1<argc)captureFrames=std::clamp(std::atoi(argv[++i]),1,1200);
        else{std::fprintf(stderr,"Unknown or incomplete option: %s\n",a.c_str());return 2;}
    }
    SetConfigFlags(FLAG_WINDOW_RESIZABLE|FLAG_MSAA_4X_HINT);if(std::getenv("COMPPHYSICS_SMOKE_FRAMES"))SetConfigFlags(FLAG_WINDOW_HIDDEN);
    InitWindow(kWidth,kHeight,"Growing Black Hole - C++ (raylib)");SetWindowMinSize(1000,650);SetTargetFPS(120);
    Texture2D glow=makeGlowTexture();auto disk=makeDisk();auto jets=makeJets();auto sky=makeSky();
    float renderScale=1.5f;RenderTexture2D scene=LoadRenderTexture((int)(GetScreenWidth()*renderScale),(int)(GetScreenHeight()*renderScale));LensPass lens=makeLensPass();
    Camera3D camera{{0,0,0},{0,0,0},{0,1,0},47,CAMERA_PERSPECTIVE};CameraRig rig;preset(KEY_ONE,&rig,&camera);
    if(startDistance>0)rig.distance=startDistance;
    float mass=startMass,spin=startSpin,time=0,speed=1,ignition=0,observerVelocity=startRadialVelocity,previousObserverDistance=-1;bool paused=false;int frame=0;
    while(!WindowShouldClose()){
        Rectangle feedButton=feedControl(),rewindButton=rewindControl();
        Rectangle spinDownButton=spinDownControl(),spinUpButton=spinUpControl();
        bool feeding=IsKeyDown(KEY_F)||(studio::over(feedButton)&&IsMouseButtonDown(MOUSE_BUTTON_LEFT));
        bool rewinding=IsKeyDown(KEY_D)||(studio::over(rewindButton)&&IsMouseButtonDown(MOUSE_BUTTON_LEFT));
        if(IsKeyPressed(KEY_SPACE)||IsKeyPressed(KEY_P))paused=!paused;if(IsKeyPressed(KEY_R)){mass=.07f;spin=.72f;time=0;ignition=0;observerVelocity=0;previousObserverDistance=-1;paused=false;}
        if(IsKeyPressed(KEY_MINUS)||IsKeyPressed(KEY_KP_SUBTRACT))speed=std::max(.25f,speed-.25f);if(IsKeyPressed(KEY_EQUAL)||IsKeyPressed(KEY_KP_ADD))speed=std::min(4.0f,speed+.25f);
        bool qualityChanged=false;if(IsKeyPressed(KEY_Q)){renderScale=renderScale>1.1f?1.0f:1.5f;qualityChanged=true;}
        float controlDt=std::min(GetFrameTime(),.05f);bool spinDown=IsKeyDown(KEY_Z)||(studio::over(spinDownButton)&&IsMouseButtonDown(MOUSE_BUTTON_LEFT));
        bool spinUp=IsKeyDown(KEY_X)||(studio::over(spinUpButton)&&IsMouseButtonDown(MOUSE_BUTTON_LEFT));
        if(spinDown&&!spinUp)spin=std::max(-.998f,spin-controlDt*.55f);if(spinUp&&!spinDown)spin=std::min(.998f,spin+controlDt*.55f);
        for(int key:{KEY_ONE,KEY_TWO,KEY_THREE})if(IsKeyPressed(key))preset(key,&rig,&camera);
        updateCamera(&camera,&rig,(float)blackhole::horizonRadius(mass)*1.72f);
        float observerDistance=Vector3Distance(camera.position,{0,0,0})/(float)blackhole::horizonRadius(mass);
        if(previousObserverDistance>0&&controlDt>.0001f){
            float target=std::clamp((previousObserverDistance-observerDistance)/controlDt*.16f,-1.0f,1.0f);
            float follow=1-std::exp(-controlDt/.11f);observerVelocity+=(target-observerVelocity)*follow;
        }previousObserverDistance=observerDistance;
        float dt=std::min(GetFrameTime(),.05f)*speed;if(!paused&&capture.empty()){
            float coordinateRate=.32f+.68f*(float)blackhole::properTimeRate(observerDistance);
            time+=dt*coordinateRate;float previous=mass;if(feeding&&!rewinding)mass=(float)blackhole::feed(mass,.105,dt);if(rewinding&&!feeding)mass=(float)blackhole::rewind(mass,.12,dt);
            if(previous<.68f&&mass>=.68f)ignition=1;ignition=std::max(0.0f,ignition-dt*.42f);
            if(mass<.68f)ignition=0;
        }
        if(IsWindowResized()||qualityChanged){UnloadRenderTexture(scene);scene=LoadRenderTexture((int)(GetScreenWidth()*renderScale),(int)(GetScreenHeight()*renderScale));}
        Vector3 right,up;cameraBasis(camera,&right,&up);
        BeginTextureMode(scene);ClearBackground({2,4,10,255});BeginMode3D(camera);
        float jetPower=effectiveJetPower(mass,spin);
        drawSky(sky,glow,right,up);drawFeedingClouds(glow,camera,time,mass,feeding);drawContinuousDisk(camera,time,mass,spin);drawDisk(disk,glow,camera,time,mass,spin);drawJets(jets,glow,camera,time,jetPower);EndMode3D();EndTextureMode();
        BeginDrawing();ClearBackground({2,4,10,255});drawLensedScene(scene,lens,camera,mass,spin,ignition,observerVelocity);
        bool blazarView=jetPower>.05f&&std::fabs(std::sin(rig.pitch))>.86f;
        DrawRectangleGradientV(0,0,GetScreenWidth(),150,{2,4,10,220},{2,4,10,0});drawHud(mass,spin,observerDistance,observerVelocity,speed,paused,feeding,rewinding,blazarView,renderScale,(float)GetFPS());EndDrawing();
        if(!capture.empty()&&++frame>=captureFrames){std::filesystem::create_directories(std::filesystem::path(capture).parent_path());TakeScreenshot(capture.c_str());break;}
        if(studio::smokeFrame(__FILE__))break;
    }
    UnloadShader(lens.shader);UnloadRenderTexture(scene);UnloadTexture(glow);studio::unload();CloseWindow();return 0;
}
