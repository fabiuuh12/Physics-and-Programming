#pragma once

#include <string>
#include <vector>

namespace asf {

struct Config {
    std::string experiment_name = "asf_cpp";
    int seed = 0;
    int nx = 64;
    int ny = 64;
    double dx = 1.0;
    double dy = 1.0;
    double G = 1.0;
    double mu0 = 1.0;
    double rho0 = 1.0;
    double omega0 = 1.0;
    double temperature0 = 1.0;
    double length0 = 1.0;
    double lambda_A = 0.0;
    double chi_B = 0.0;
    double chi_omega = 0.0;
    double chi_T = 0.0;
    double pressure_floor = 1.0e-6;
    double rho_floor = 1.0e-6;
    int iterations = 500;
    double tolerance = 1.0e-8;
    int smooth_passes = 0;
};

struct Grid {
    int nx = 0;
    int ny = 0;
    std::vector<double> values;

    Grid() = default;
    Grid(int nx_value, int ny_value, double fill = 0.0);
    double &operator()(int x, int y);
    double operator()(int x, int y) const;
};

struct FieldState {
    Grid rho;
    Grid pressure;
    Grid bx;
    Grid by;
    Grid vx;
    Grid vy;
    Grid temperature;
    Grid structure;
};

Config load_config(const std::string &path);
FieldState build_controlled_fields(const Config &config, bool structured);
Grid compute_structure_field(const FieldState &state, const Config &config);
Grid smooth_box(const Grid &field, int passes);
Grid laplacian(const Grid &field, double dx, double dy);
std::pair<Grid, Grid> gradient(const Grid &field, double dx, double dy);
std::pair<Grid, std::vector<double>> poisson_jacobi(const Grid &source, const Config &config);
double poisson_residual(const Grid &phi, const Grid &source, double dx, double dy);
Grid effective_potential(const Grid &phi_n, const Grid &structure, const Config &config);
std::pair<Grid, Grid> acceleration(const Grid &phi, double dx, double dy);
void write_csv(const std::string &path, const FieldState &state, const Grid &phi_n, const Grid &phi_eff);

}  // namespace asf
