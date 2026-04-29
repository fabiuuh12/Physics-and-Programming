#pragma once

#include <array>
#include <string>

namespace cnav {

using State = std::array<double, 4>;

struct RiskWeights {
    double w_r = 0.30;
    double w_v = 0.20;
    double w_g = 0.20;
    double w_m = 0.20;
    double w_l = 0.10;
    double r0 = 0.01;
    double v0 = 0.01;
    double epsilon = 1.0e-4;
    double risk_max = 8.0;
};

State cr3bp_rhs(const State &state, double mu);
State rk4_step(const State &state, double dt, double mu);
double navigation_risk(double sigma_r, double sigma_v, double geometry, double missed, double lighting, const RiskWeights &weights);
double clipped_risk(double risk, const RiskWeights &weights);
void write_demo_csv(const std::string &path, const State &initial, int steps, double dt, double mu);

}  // namespace cnav
