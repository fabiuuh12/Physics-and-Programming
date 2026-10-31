#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

// G=c=1. Exterior Schwarzschild geometry and an ultrastatic Ellis wormhole.
// Embedding height is extrinsic; it is not a time coordinate or a force.
namespace spacetime {
constexpr double pi=3.14159265358979323846;
using State=std::array<double,4>;
inline State add(State a,const State& b,double h){for(int i=0;i<4;i++)a[i]+=h*b[i];return a;}
template<class Derivative> State rk4(State s,double h,Derivative derivative){
    State a=derivative(s),b=derivative(add(s,a,h/2)),c=derivative(add(s,b,h/2)),d=derivative(add(s,c,h));
    for(int i=0;i<4;i++)s[i]+=h*(a[i]+2*b[i]+2*c[i]+d[i])/6;
    return s;
}
inline double height(double r,double rs,double outer=16){
    return 2*(std::sqrt(rs*std::max(0.0,r-rs))-std::sqrt(rs*std::max(0.0,outer-rs)));
}
inline double staticRate(double r,double rs,double surface=0){
    if(r<=rs||r<surface)return std::numeric_limits<double>::quiet_NaN();
    return std::sqrt(1-rs/r);
}
// T = v-r, where v is ingoing Eddington-Finkelstein time. Radial incoming
// light has dr/dT=-1; outgoing light has the following coordinate speed.
inline double outgoingSlope(double r,double rs){return (r-rs)/(r+rs);}
inline State photonDerivative(State s,double rs,double angularMomentum){
    double r2=s[0]*s[0]+s[1]*s[1];
    double coefficient=-1.5*rs*angularMomentum*angularMomentum/std::pow(std::max(r2,1e-16),2.5);
    return {s[2],s[3],coefficient*s[0],coefficient*s[1]};
}
inline double photonEnergySquared(State s,double rs){
    double r=std::hypot(s[0],s[1]),L=s[0]*s[3]-s[1]*s[2];
    return s[2]*s[2]+s[3]*s[3]-rs*L*L/(r*r*r);
}
struct Point{double x,z,branch=1;};
struct Path{std::vector<Point> points;bool captured=false;bool crossed=false;};
inline Path schwarzschildPath(double impact,double rs,double surface,double step=.015){
    State s{-14,impact,1,0};
    double L=s[0]*s[3]-s[1]*s[2];
    double stop=std::max(surface,rs*1.002);
    Path path;
    for(int i=0;i<6500;++i){
        double r=std::hypot(s[0],s[1]);
        if(r<=stop){path.captured=true;break;}
        if(r>22&&i>10)break;
        if(i%2==0)path.points.push_back({s[0],s[1],1});
        double h=std::min(step,std::max(.0005,r*.012));
        s=rk4(s,h,[&](State p){return photonDerivative(p,rs,L);});
    }
    return path;
}
// Ellis metric ds²=-dt²+dl²+(l²+b²)dOmega². Here s=(l,phi,dl/dt,unused).
inline State ellisDerivative(State s,double b,double L){
    double r2=s[0]*s[0]+b*b;
    return {s[2],L/r2,L*L*s[0]/(r2*r2),0};
}
inline double ellisHeight(double l,double b){return b*std::asinh(l/b);}
inline Path ellisPath(double impact,double b,double step=.015){
    double r=std::hypot(14.,impact),L=-impact;
    State s{std::sqrt(r*r-b*b),std::atan2(impact,-14.),-std::sqrt(1-L*L/(r*r)),0};
    Path path;
    for(int i=0;i<6500;++i){
        double rr=std::sqrt(s[0]*s[0]+b*b);
        if(rr>22&&i>10)break;
        if(s[0]<0)path.crossed=true;
        if(i%2==0)path.points.push_back({rr*std::cos(s[1]),rr*std::sin(s[1]),s[0]>=0?1.:-1.});
        s=rk4(s,step,[&](State p){return ellisDerivative(p,b,L);});
    }
    return path;
}
} // namespace spacetime
