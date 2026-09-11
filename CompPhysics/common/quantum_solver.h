#pragma once
#include <cmath>
#include <complex>
#include <vector>
#include "physics_models.h"

namespace physics {
// 1D time-dependent Schrodinger equation, hbar=m=1, Dirichlet endpoints.
// Crank-Nicolson is unitary for this real, time-independent potential.
class WavePacket {
public:
    static constexpr int count=512;
    static constexpr double left=-12, dx=24.0/(count-1), dt=1.0/480;
    using Complex=std::complex<double>;
    std::vector<Complex> psi=std::vector<Complex>(count);
    double barrier=1.15, width=1, time=0;
    double x(int i) const { return left+i*dx; }
    double potential(int i) const { return std::abs(x(i))<=width/2 ? barrier : 0; }
    void reset(double energy,double height) {
        barrier=height; time=0;
        double k=std::sqrt(2*energy);
        for (int i=0;i<count;++i) {
            double offset=x(i)+7;
            psi[i]=std::exp(-offset*offset/(4*0.8*0.8))*std::exp(Complex(0,k*x(i)));
        }
        psi.front()=psi.back()=0;
        double scale=1/std::sqrt(norm());
        for (auto& p:psi) p*=scale;
        prepare();
    }
    double norm() const {
        double sum=0; for (auto p:psi) sum+=std::norm(p)*dx; return sum;
    }
    std::array<double,3> regions() const {
        std::array<double,3> sums{};
        for (int i=0;i<count;++i) sums[x(i)<-width/2 ? 0 : (x(i)>width/2 ? 2 : 1)]+=std::norm(psi[i])*dx;
        return sums;
    }
    void step() {
        Complex off(0,-dt/(4*dx*dx));
        for (int i=1;i<count-1;++i) {
            Complex diagonal(1,dt/2*(1/(dx*dx)+potential(i)));
            Complex rhs=std::conj(diagonal)*psi[i]-off*(psi[i-1]+psi[i+1]);
            rhsScratch[i]=(rhs-off*rhsScratch[i-1])*inversePivot[i];
        }
        for (int i=count-2;i>=1;--i) psi[i]=rhsScratch[i]-upper[i]*psi[i+1];
        time+=dt;
    }
private:
    std::vector<Complex> upper=std::vector<Complex>(count);
    std::vector<Complex> inversePivot=std::vector<Complex>(count);
    std::vector<Complex> rhsScratch=std::vector<Complex>(count);
    void prepare() {
        Complex off(0,-dt/(4*dx*dx));
        for (int i=1;i<count-1;++i) {
            Complex diagonal(1,dt/2*(1/(dx*dx)+potential(i)));
            inversePivot[i]=1.0/(diagonal-off*upper[i-1]);
            upper[i]=off*inversePivot[i];
        }
    }
};
} // namespace physics
