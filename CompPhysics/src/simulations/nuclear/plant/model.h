#pragma once
#include <algorithm>
#include <cmath>
namespace plant {
// Illustrative lumped energy model: controls are normalized, powers are MW.
struct Model {
    float rods=.28f,load=.85f,heat=.72f,steam=.72f,rotor=.61f;
    float angle=0,time=0;
    bool scrammed=false;
    void shutdown() {scrammed=true;rods=1;}
    void reset() {*this=Model{};}
    void step(float dt) {
        dt=std::clamp(dt,0.0f,.05f);
        if(scrammed) rods=1;
        float target=std::pow(1-std::clamp(rods,0.0f,1.0f),1.35f);
        auto follow=[dt](float v,float target,float tau) {return target+(v-target)*std::exp(-dt/tau);};
        heat=follow(heat,target,3.5f); steam=follow(steam,heat,2.5f);
        rotor=follow(rotor,steam*std::clamp(load,0.0f,1.0f),2);
        angle=std::fmod(angle+dt*rotor*15,6.2831853f);time+=dt;
    }
    float thermal() const {return 3000*steam;}
    float gross() const {return 1080*std::min(steam,rotor);}
    float auxiliary() const {return std::min(gross(),thermal()*.025f);}
    float electric() const {return gross()-auxiliary();}
    float rejected() const {return thermal()-electric();}
};
// Capture is acquired only on a new pinch. Tracking loss releases it and
// requires an open hand before reacquisition, preventing a surprise re-grab.
struct Grab {
    int held=-1; bool wasDown=true;
    bool update(bool tracked,bool down,int hover) {
        if(!tracked) {held=-1;wasDown=true;return false;}
        bool pressed=down&&!wasDown;
        if(!down) held=-1;
        if(pressed) held=hover;
        wasDown=down;return pressed;
    }
};
}
