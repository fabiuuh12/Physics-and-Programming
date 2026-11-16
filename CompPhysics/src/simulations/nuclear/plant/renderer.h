#pragma once
#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include "../../../common/studio.h"
#include "model.h"
#include <vector>
namespace plant {
constexpr Color orange{255,151,76,255}, blue{73,171,243,255}, mint{94,218,189,255}, gold{255,211,100,255};
struct Part {int mesh;Matrix transform;Color color;bool open;};
struct Route {
    std::vector<Vector3> points;
    Color color;float width,length=0;
    std::vector<float> cumulative;
    Route(std::initializer_list<Vector3> p,Color c,float r):points(p),color(c),width(r) {
        cumulative.push_back(0);
        for(size_t i=1;i<points.size();++i) {length+=Vector3Distance(points[i-1],points[i]);cumulative.push_back(length);}
    }
    Vector3 at(float fraction) const {
        float d=(fraction-std::floor(fraction))*length;
        for(size_t i=1;i<points.size();++i) if(d<=cumulative[i])
            return Vector3Lerp(points[i-1],points[i],(d-cumulative[i-1])/(cumulative[i]-cumulative[i-1]));
        return points.back();
    }
};
inline Matrix transform(Vector3 position,Vector3 scale) {return MatrixMultiply(MatrixScale(scale.x,scale.y,scale.z),MatrixTranslate(position.x,position.y,position.z));}
inline Matrix tubeTransform(Vector3 a,Vector3 b,float radius) {
    Vector3 d=Vector3Subtract(b,a);float length=Vector3Length(d);Vector3 n=Vector3Scale(d,1/std::max(length,.001f));
    Vector3 axis=Vector3CrossProduct({0,1,0},n);
    if(Vector3Length(axis)<.001f) axis={1,0,0};
    return MatrixMultiply(MatrixMultiply(MatrixScale(radius,length,radius),MatrixRotate(axis,std::acos(std::clamp(n.y,-1.0f,1.0f)))),MatrixTranslate(a.x,a.y,a.z));
}
class Renderer {
    Mesh meshes[5]{};Material material{};Shader shader{};Texture2D glow{};
    int tintLoc,eyeLoc,openLoc;
    std::vector<Part> fixed;
    void draw(int mesh,Matrix matrix,Color color,bool open=false) {
        float tint[]={color.r/255.0f,color.g/255.0f,color.b/255.0f};float cut=open?1.0f:0;
        SetShaderValue(shader,tintLoc,tint,SHADER_UNIFORM_VEC3);SetShaderValue(shader,openLoc,&cut,SHADER_UNIFORM_FLOAT);
        DrawMesh(meshes[mesh],material,matrix);
    }
    void add(int mesh,Matrix matrix,Color c,bool open=false) {fixed.push_back({mesh,matrix,c,open});}
    void box(Vector3 p,Vector3 s,Color c) {add(0,transform(p,s),c);}
    void pipe(Vector3 a,Vector3 b,float r,Color c,bool open=false) {add(open?3:1,tubeTransform(a,b,r),c,open);}
    void vessel(Vector3 p,float radius,float height) {
        pipe(p,Vector3Add(p,{0,height,0}),radius,{141,170,185,255},true);
        add(2,transform(p,{radius,radius*.38f,radius}),{115,144,164,255},true);
        add(2,transform(Vector3Add(p,{0,height,0}),{radius,radius*.38f,radius}),{160,184,195,255},true);
        for(float y : {0.0f,height*.5f,height}) pipe(Vector3Add(p,{0,y-.055f,0}),Vector3Add(p,{0,y+.055f,0}),radius*1.05f,{69,95,118,255},true);
    }
public:
    std::vector<Route> routes;
    bool ready=false;
    Renderer() {
        const char* vs=R"(#version 330
in vec3 vertexPosition;
in vec3 vertexNormal;
uniform mat4 mvp;
uniform mat4 matModel;
out vec3 pos;
out vec3 normal;
void main(){pos=vec3(matModel*vec4(vertexPosition,1));normal=normalize(transpose(inverse(mat3(matModel)))*vertexNormal);gl_Position=mvp*vec4(vertexPosition,1);}
)";
        const char* fs=R"(#version 330
in vec3 pos;
in vec3 normal;
uniform vec3 tint;
uniform vec3 eye;
uniform float opening;
out vec4 finalColor;
void main(){vec3 n=normalize(normal),v=normalize(eye-pos);
if(opening>.5 && dot(n,v)>.12)discard;
float diffuse=max(dot(n,normalize(vec3(-.3,.9,.6))),0.0);
float spec=pow(max(dot(reflect(-normalize(vec3(-.3,.9,.6)),n),v),0.0),40.0);
vec3 c=tint*(.36+.64*diffuse)+vec3(spec*.28);
finalColor=vec4(c,1);}
)";
        shader=LoadShaderFromMemory(vs,fs);ready=shader.id && shader.id!=rlGetShaderIdDefault();
        if(!ready) return;
        tintLoc=GetShaderLocation(shader,"tint");eyeLoc=GetShaderLocation(shader,"eye");openLoc=GetShaderLocation(shader,"opening");
        material=LoadMaterialDefault();material.shader=shader;
        meshes[0]=GenMeshCube(1,1,1);meshes[1]=GenMeshCylinder(1,1,16);meshes[2]=GenMeshSphere(1,16,24);
        // Uncapped cylinder shell, so the cutaway does not leave opaque end disks.
        Mesh& m=meshes[3];m.vertexCount=32*6;m.triangleCount=m.vertexCount/3;
        m.vertices=(float*)MemAlloc(m.vertexCount*3*sizeof(float));m.normals=(float*)MemAlloc(m.vertexCount*3*sizeof(float));
        int k=0;for(int i=0;i<32;++i) {
            float a=2*PI*i/32,b=2*PI*(i+1)/32;
            const float angles[]={a,b,a,a,b,b},ys[]={0,0,1,1,0,1};
            for(int j=0;j<6;++j) {m.vertices[k]=m.normals[k]=std::cos(angles[j]);++k;m.vertices[k]=ys[j];m.normals[k++]=0;m.vertices[k]=m.normals[k]=std::sin(angles[j]);++k;}
        }
        UploadMesh(&m,false);
        Image image=GenImageColor(48,48,BLANK);auto* pixels=(Color*)image.data;
        for(int y=0;y<48;++y)for(int x=0;x<48;++x) {float r=(std::pow((x-23.5f)/24,2)+std::pow((y-23.5f)/24,2));pixels[y*48+x]={255,255,255,(unsigned char)(255*std::pow(std::max(0.0f,1-r),3))};}
        glow=LoadTextureFromImage(image);UnloadImage(image);
        // Smooth, open-ended hyperboloid tower, cached as one mesh.
        Mesh& tower=meshes[4];tower.vertexCount=24*48*6;tower.triangleCount=tower.vertexCount/3;
        tower.vertices=(float*)MemAlloc(tower.vertexCount*3*sizeof(float));
        tower.normals=(float*)MemAlloc(tower.vertexCount*3*sizeof(float));
        int out=0;
        for(int y=0;y<24;++y)for(int a=0;a<48;++a) {
            int dy[]={0,0,1,1,0,1},da[]={0,1,0,0,1,1};
            for(int j=0;j<6;++j) {
                float u=(y+dy[j])/24.0f,angle=(a+da[j])*2*PI/48;
                float radius=1.15f+.8f*std::pow((u-.65f)/.65f,2);
                Vector3 n=Vector3Normalize({std::cos(angle),-1.6f*(u-.65f)/(.65f*.65f*5),std::sin(angle)});
                tower.vertices[out]=radius*std::cos(angle);tower.normals[out++]=n.x;
                tower.vertices[out]=.35f+u*5;tower.normals[out++]=n.y;
                tower.vertices[out]=-6+radius*std::sin(angle);tower.normals[out++]=n.z;
            }
        }
        UploadMesh(&tower,false);
        build();
    }
    void build() {
        Color steel{106,139,158,255},concrete{133,153,163,255};
        box({.5f,-.3f,0},{25,.6f,18},{30,46,61,255});
        // Raised equipment islands and inset floor markings.
        for(float x : {-7.0f,-3.0f,2.0f,6.0f,9.0f}) box({x,.06f,0},{2.8f,.12f,3.5f},{48,67,81,255});
        vessel({-7,.6f,0},1.05f,3.5f);vessel({-3,.5f,0},.95f,4.5f);
        // Containment: an open architectural frame, with a rear concrete wall.
        box({-6,1.9f,-2.5f},{7,.0f+3.8f,.18f},concrete);
        for(float x : {-9.5f,-2.5f}) box({x,2,-2.3f},{.20f,4,.3f},concrete);
        for(int i=0;i<24;++i) {
            float a=PI*i/24,b=PI*(i+1)/24;
            pipe({-6+3.5f*std::cos(a),3.8f+2*std::sin(a),-2.5f},{-6+3.5f*std::cos(b),3.8f+2*std::sin(b),-2.5f},.10f,concrete);
        }
        // Fuel assemblies, steam-generator U tubes, and separate secondary water.
        for(int x=-2;x<=2;++x)for(int z=-2;z<=2;++z) if(x*x+z*z<7)
            pipe({-7+x*.25f,1.1f,z*.25f},{-7+x*.25f,3,z*.25f},.075f,{204,143,76,255});
        for(int i=0;i<7;++i) {
            float x=-3+(i-3)*.16f;
            pipe({x,1.0f,-.38f},{x,3.8f,-.38f},.045f,orange);
            pipe({x,3.8f,-.38f},{x,3.8f,.38f},.045f,orange);
            pipe({x,3.8f,.38f},{x,1,.38f},.045f,blue);
        }
        vessel({-5,.5f,-1.7f},.43f,2.5f); // Pressurizer connected to the hot leg.
        pipe({-5,1,-1.7f},{-5,3.8f,-1.7f},.10f,steel);pipe({-5,3.8f,-1.7f},{-5,3.8f,0},.10f,steel);
        // Primary loop: never connects to the turbine.
        routes.emplace_back(std::initializer_list<Vector3>{{-7,3.6f,0},{-7,3.8f,1},{-3,3.8f,1},{-3,3.8f,0}},orange,.14f);
        routes.emplace_back(std::initializer_list<Vector3>{{-3,3.8f,0},{-3,1,0},{-3,1,2},{-5,1,2},{-7,1,2},{-7,1,0},{-7,3.6f,0}},blue,.14f);
        // Secondary loop: steam, exhaust, condensate, feedwater.
        routes.emplace_back(std::initializer_list<Vector3>{{-3,4.8f,0},{-3,5.4f,0},{.3f,5.4f,0},{.3f,3.2f,0},{1,3.2f,0}},gold,.17f);
        routes.emplace_back(std::initializer_list<Vector3>{{3.5f,3.2f,0},{3.5f,2,0},{3.5f,2,2.8f},{2,1.5f,2.8f}}, Color{194,215,225,255},.25f);
        routes.emplace_back(std::initializer_list<Vector3>{{2,.8f,2.8f},{2,.5f,4},{-3,.5f,4},{-3,1,4},{-3,1,0}},blue,.13f);
        // Third circuit: condenser cooling water and cooling-tower heat rejection.
        routes.emplace_back(std::initializer_list<Vector3>{{2,.8f,3.2f},{5,.8f,3.2f},{5,.8f,-6},{0,.8f,-6}},mint,.15f);
        routes.emplace_back(std::initializer_list<Vector3>{{0,.4f,-6},{-1.8f,.4f,-6},{-1.8f,.4f,3.4f},{2,.8f,3.4f}},blue,.15f);
        for(const auto& r:routes)for(size_t i=1;i<r.points.size();++i) {
            pipe(r.points[i-1],r.points[i],r.width,{110,151,171,255},true);
            add(2,transform(r.points[i],{r.width,r.width,r.width}),{110,151,171,255},true);
        }
        // Coolant and feedwater pumps.
        for(Vector3 p : {Vector3{-5,.85f,2},Vector3{-1,.5f,4},Vector3{5,.6f,-4}}) {
            add(2,transform(p,{.38f,.38f,.38f}),mint);
            pipe(Vector3Add(p,{0,0,0}),Vector3Add(p,{0,.65f,0}),.20f,steel);
        }
        // Turbine bearing supports and shaft. The rotor is drawn separately.
        for(float x : {.7f,3.7f,5.0f,6.8f}) {box({x,1.5f,0},{.28f,3,.65f},steel);}
        pipe({.4f,3.2f,0},{7,3.2f,0},.12f,{200,216,224,255});
        // Generator copper stator coils surround the shaft.
        for(int j=0;j<12;++j) {
            float a=j*2*PI/12;Vector3 p{5,3.2f+.65f*std::sin(a),.65f*std::cos(a)};
            pipe(p,Vector3Add(p,{1.7f,0,0}),.11f,{194,117,66,255});
        }
        pipe({5,3.2f,0},{6.7f,3.2f,0},.83f,steel,true);
        // Condenser and its physically separate cooling-water tube bundle.
        box({2,.6f,2.8f},{3,.3f,1.8f},steel);
        for(int i=0;i<9;++i) pipe({.6f,1+(i%3)*.18f,2.4f+(i/3)*.28f},{3.4f,1+(i%3)*.18f,2.4f+(i/3)*.28f},.055f,mint);
        // Hyperboloid cooling tower, open at the top.
        add(4,MatrixIdentity(),concrete);
        for(int i=0;i<12;++i) {float a=i*PI/6;box({1.5f*std::cos(a),.3f,-6+1.5f*std::sin(a)},{.1f,.6f,.1f},steel);}
        // Generator cables lead to the transformer; pulses follow this exact path.
        pipe({6.8f,3.2f,0},{9,2.5f,0},.045f,{170,131,70,255});
        // Transformer, insulators, and outgoing transmission line.
        box({9,1.0f,0},{1.4f,1.8f,1.2f},steel);
        for(int i=0;i<7;++i) box({8.3f+i*.2f,1,0},{.055f,1.5f,1.45f},concrete);
        for(float z : {-.4f,0.0f,.4f}) {pipe({9,1.8f,z},{9,2.5f,z},.10f,{204,177,113,255});pipe({10,.1f,-3+z},{10,6,-3+z},.045f,steel);}
        box({10,5.8f,-3},{.2f,.2f,3},steel);
        for(float z : {-4.0f,-3.0f,-2.0f}) pipe({9,2.5f,0},{10,5.8f,z},.025f,gold);
        // Physical operator console, always controllable with mouse or hands.
        box({-1,.55f,6},{6,1.1f,1.2f},{44,64,82,255});
        box({-1,1.5f,5.7f},{6,2,.18f},{62,85,104,255});
        pipe({-3,1.0f,6},{-3,2.6f,6},.055f,steel);
        pipe({-1,1.65f,5.8f},{-1,1.65f,6.05f},.45f,steel);
    }
    void render(Camera3D camera,const Model& model,bool cutaway,bool flows) {
        float eye[]={camera.position.x,camera.position.y,camera.position.z};SetShaderValue(shader,eyeLoc,eye,SHADER_UNIFORM_VEC3);
        rlDisableBackfaceCulling();
        for(const auto& p:fixed) draw(p.mesh,p.transform,p.color,p.open&&cutaway);
        // Animated rods physically enter the fuel lattice.
        for(int x=-1;x<=1;++x) for(int z=-1;z<=1;++z) {
            float bottom=3.1f-1.9f*model.rods;
            draw(1,tubeTransform({-7+x*.32f,bottom,z*.32f},{-7+x*.32f,bottom+2.3f,z*.32f},.045f),{72,89,102,255});
        }
        // Turbine disks and staggered blades rotate about the same generator shaft.
        for(int i=0;i<7;++i) {
            float x=1+i*.37f,r=.36f+i*.07f;
            draw(1,tubeTransform({x-.025f,3.2f,0},{x+.025f,3.2f,0},r*.55f),{160,184,197,255});
            for(int j=0;j<12;++j) {
                float a=model.angle+j*2*PI/12+i*.2f;
                Vector3 a0{x,3.2f+r*.4f*std::sin(a),r*.4f*std::cos(a)};
                Vector3 b{x+.06f,3.2f+r*std::sin(a+.15f),r*std::cos(a+.15f)};
                draw(1,tubeTransform(a0,b,.035f),{198,216,224,255});
            }
        }
        draw(0,MatrixMultiply(MatrixRotateX(model.angle),transform({5.8f,3.2f,0},{1.5f,.5f,.25f})),orange);
        // Real control hardware: sliding rod lever, rotary load dial, shutdown button.
        float lever=1.0f+1.6f*(1-model.rods);
        draw(0,transform({-3,lever,6.15f},{.55f,.18f,.30f}),orange);
        float a=(-135+270*model.load)*DEG2RAD;
        draw(1,tubeTransform({-1,1.65f,6.10f},{-1+std::sin(a)*.34f,1.65f+std::cos(a)*.34f,6.10f},.035f),gold);
        draw(1,tubeTransform({1,1.5f,5.9f},{1,1.5f,6.25f},.30f),model.scrammed?Color{124,43,46,255}:Color{242,81,76,255});
        rlEnableBackfaceCulling();
        // Billboards batch into one texture; no per-particle sphere meshes.
        BeginBlendMode(BLEND_ADDITIVE);rlDisableDepthMask();
        if(flows) for(size_t i=0;i<routes.size();++i) {
            const auto& r=routes[i];float rate=i<2?.12f:i<5?.07f+.10f*model.steam:.10f;
            for(int j=0;j<36;++j) DrawBillboard(camera,glow,r.at(j/36.0f+model.time*rate),.16f,r.color);
        }
        for(int i=0;i<36;++i) {
            float a=i*2.4f+model.time*.7f,r=.5f*std::abs(std::sin(i*7.3f));
            DrawBillboard(camera,glow,{-7+r*std::cos(a),1.2f+std::fmod(i*.071f+model.time*.2f,1.6f),r*std::sin(a)},.16f,Fade(orange,model.heat));
        }
        // Visible plume is water vapor from the third cooling circuit.
        for(int i=0;i<60;++i) {
            float u=std::fmod(i/60.0f+model.time*.08f,1.0f),a=i*2.4f;
            DrawBillboard(camera,glow,{std::cos(a)*u*1.6f+u,5.4f+u*4,-6+std::sin(a)*u},.7f+u*1.8f,Fade({188,215,231,255},(1-u)*.13f*model.steam));
        }
        if(flows)for(int j=0;j<18;++j) {
            float u=std::fmod(j/18.0f+model.time*.25f,1.0f);
            Vector3 p=u<.35f?Vector3Lerp({6.8f,3.2f,0},{9,2.5f,0},u/.35f):Vector3Lerp({9,2.5f,0},{10,5.8f,-3},(u-.35f)/.65f);
            DrawBillboard(camera,glow,p,.2f,Fade(gold,model.electric()/1000));
        }
        rlEnableDepthMask();EndBlendMode();
    }
    ~Renderer() {
        if(!ready)return;
        for(auto& mesh:meshes) UnloadMesh(mesh);
        UnloadTexture(glow);
        material.shader={};UnloadMaterial(material);UnloadShader(shader);
    }
};
}
