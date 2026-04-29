#include "cislunar_nav.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace cnav {

namespace {

State add_scaled(const State &a, const State &b, double scale) {
    return {a[0] + scale * b[0], a[1] + scale * b[1], a[2] + scale * b[2], a[3] + scale * b[3]};
}

}  // namespace

State cr3bp_rhs(const State &state, double mu) {
    const double x = state[0];
    const double y = state[1];
    const double vx = state[2];
    const double vy = state[3];
    const double r1 = std::sqrt((x + mu) * (x + mu) + y * y);
    const double r2 = std::sqrt((x - 1.0 + mu) * (x - 1.0 + mu) + y * y);
    const double ax = 2.0 * vy + x - (1.0 - mu) * (x + mu) / (r1 * r1 * r1) -
                      mu * (x - 1.0 + mu) / (r2 * r2 * r2);
    const double ay = -2.0 * vx + y - (1.0 - mu) * y / (r1 * r1 * r1) - mu * y / (r2 * r2 * r2);
    return {vx, vy, ax, ay};
}

State rk4_step(const State &state, double dt, double mu) {
    const State k1 = cr3bp_rhs(state, mu);
    const State k2 = cr3bp_rhs(add_scaled(state, k1, 0.5 * dt), mu);
    const State k3 = cr3bp_rhs(add_scaled(state, k2, 0.5 * dt), mu);
    const State k4 = cr3bp_rhs(add_scaled(state, k3, dt), mu);
    return {
        state[0] + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
        state[1] + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
        state[2] + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]),
        state[3] + (dt / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]),
    };
}

double navigation_risk(double sigma_r, double sigma_v, double geometry, double missed, double lighting, const RiskWeights &weights) {
    const double safe_geometry = std::max(0.0, geometry);
    const double risk = weights.w_r * sigma_r / weights.r0 + weights.w_v * sigma_v / weights.v0 +
                        weights.w_g / (safe_geometry + weights.epsilon) + weights.w_m * missed + weights.w_l * lighting;
    return std::max(0.0, risk);
}

double clipped_risk(double risk, const RiskWeights &weights) {
    return std::clamp(risk, 0.0, weights.risk_max);
}

void write_demo_csv(const std::string &path, const State &initial, int steps, double dt, double mu) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Could not write CSV: " + path);
    }
    output << "step,x,y,vx,vy\n";
    State state = initial;
    output << 0 << ',' << state[0] << ',' << state[1] << ',' << state[2] << ',' << state[3] << '\n';
    for (int step = 1; step <= steps; ++step) {
        state = rk4_step(state, dt, mu);
        output << step << ',' << state[0] << ',' << state[1] << ',' << state[2] << ',' << state[3] << '\n';
    }
}

}  // namespace cnav
