#pragma once
#include <algorithm>
#include <array>
#include <cmath>

// Rendering-independent numerical kernels, shared with the regression tests.
namespace physics {
constexpr double pi=3.14159265358979323846;
struct Clock {
    double remainder=0;
    template<class Step> void advance(double frameTime, Step step) {
        // Fixed integration/sample cadence, with a bounded catch-up after stalls.
        constexpr double dt=1.0/240.0;
        remainder+=std::clamp(frameTime,0.0,0.1);
        while (remainder+1e-12>=dt) { step(dt); remainder-=dt; }
        remainder=std::max(0.0,remainder);
    }
    void reset() { remainder=0; }
};

using Flight=std::array<double,4>; // x, y, vx, vy
inline Flight projectileDerivative(Flight s, double drag) {
    double speed=std::hypot(s[2],s[3]);
    return {s[2],s[3],-drag*speed*s[2],-9.81-drag*speed*s[3]};
}
inline Flight add(Flight a, Flight b, double scale) {
    for (int i=0;i<4;++i) a[i]+=scale*b[i];
    return a;
}
inline Flight projectileStep(Flight s, double drag, double dt) {
    Flight a=projectileDerivative(s,drag);
    Flight b=projectileDerivative(add(s,a,dt/2),drag);
    Flight c=projectileDerivative(add(s,b,dt/2),drag);
    Flight d=projectileDerivative(add(s,c,dt),drag);
    for (int i=0;i<4;++i) s[i]+=dt*(a[i]+2*b[i]+2*c[i]+d[i])/6;
    return s;
}
inline void oscillatorStep(double& x,double& v,double time,double dt,
                           double mass,double spring,double damping,double drive,double omega) {
    auto acceleration=[&](double xx,double vv,double t) {
        return (drive*std::cos(omega*t)-spring*xx-damping*vv)/mass;
    };
    double ax=v, av=acceleration(x,v,time);
    double bx=v+dt*av/2, bv=acceleration(x+dt*ax/2,v+dt*av/2,time+dt/2);
    double cx=v+dt*bv/2, cv=acceleration(x+dt*bx/2,v+dt*bv/2,time+dt/2);
    double dx=v+dt*cv, dv=acceleration(x+dt*cx,v+dt*cv,time+dt);
    x+=dt*(ax+2*bx+2*cx+dx)/6;
    v+=dt*(av+2*bv+2*cv+dv)/6;
}
inline double circleOverlap(double separation,double radiusA,double radiusB) {
    double d=std::abs(separation), a=radiusA,b=radiusB;
    if (d>=a+b) return 0;
    if (d<=std::abs(a-b)) return pi*std::min(a*a,b*b);
    double aa=std::acos(std::clamp((d*d+a*a-b*b)/(2*d*a),-1.0,1.0));
    double bb=std::acos(std::clamp((d*d+b*b-a*a)/(2*d*b),-1.0,1.0));
    double lens=std::max(0.0,(-d+a+b)*(d+a-b)*(d-a+b)*(d+a+b));
    return a*a*aa+b*b*bb-0.5*std::sqrt(lens);
}
inline double gaussianDensity(double x,double center,double sigma) {
    double u=(x-center)/sigma;
    return std::exp(-0.5*u*u)/(std::sqrt(2*pi)*sigma);
}
inline double eccentricAnomaly(double mean,double eccentricity) {
    double value=mean;
    for (int i=0;i<12;++i) value-=(value-eccentricity*std::sin(value)-mean)/(1-eccentricity*std::cos(value));
    return value;
}
inline double packetSigma(double initial,double time) {
    // hbar = mass = 1. Momentum width remains 1/(2*initial).
    return std::sqrt(initial*initial+time*time/(4*initial*initial));
}
inline double binaryEntropy(int a,int b) {
    if (!a || !b) return 0;
    double p=double(a)/(a+b);
    return -p*std::log2(p)-(1-p)*std::log2(1-p);
}
inline double reflect(double& position,double& velocity,double lo,double hi,double dt) {
    // Exact free flight inside a one-dimensional reflecting box. The loop
    // accounts for every collision, including multiple impacts in one step.
    double impulse=0;
    position=std::clamp(position,lo,hi);
    while (dt>0 && velocity!=0) {
        double wall=velocity>0 ? hi : lo;
        double arrival=(wall-position)/velocity;
        if (arrival>dt) { position+=velocity*dt; break; }
        position=wall; dt-=std::max(0.0,arrival);
        impulse+=2*std::abs(velocity);
        velocity=-velocity;
    }
    return impulse;
}
} // namespace physics
