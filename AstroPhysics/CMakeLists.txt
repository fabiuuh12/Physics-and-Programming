cmake_minimum_required(VERSION 3.20)
project(astrophysics_cpp_viz LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(raylib CONFIG REQUIRED)

add_executable(4th_dimension_viz_cpp "4th_dimension_viz.cpp")
target_link_libraries(4th_dimension_viz_cpp PRIVATE raylib)

add_executable(angular_momentum_viz_cpp "angular_momentum_viz.cpp")
target_link_libraries(angular_momentum_viz_cpp PRIVATE raylib)

add_executable(aerodynamics_viz_cpp "aerodynamics_viz.cpp")
target_link_libraries(aerodynamics_viz_cpp PRIVATE raylib)

add_executable(atom_viz_cpp "atom_viz.cpp")
target_link_libraries(atom_viz_cpp PRIVATE raylib)

add_executable(atomic_bomb_viz_cpp "atomic_bomb_viz.cpp")
target_link_libraries(atomic_bomb_viz_cpp PRIVATE raylib)

add_executable(blackhole_viz_cpp "blackhole_viz.cpp")
target_link_libraries(blackhole_viz_cpp PRIVATE raylib)

add_executable(black_hole_accretion_beaming_viz_cpp "black_hole_accretion_beaming_viz.cpp")
target_link_libraries(black_hole_accretion_beaming_viz_cpp PRIVATE raylib)

add_executable(collision_bh_viz_cpp "collision_bh_viz.cpp")
target_link_libraries(collision_bh_viz_cpp PRIVATE raylib)

add_executable(cosmic_expansion_sandbox_viz_cpp "cosmic_expansion_sandbox_viz.cpp")
target_link_libraries(cosmic_expansion_sandbox_viz_cpp PRIVATE raylib)

add_executable(damped_forced_oscillator_viz_cpp "damped_forced_oscillator_viz.cpp")
target_link_libraries(damped_forced_oscillator_viz_cpp PRIVATE raylib)

add_executable(dark_matter_viz_cpp "dark_matter_viz.cpp")
target_link_libraries(dark_matter_viz_cpp PRIVATE raylib)

add_executable(doppler_eff_viz_cpp "doppler_eff_viz.cpp")
target_link_libraries(doppler_eff_viz_cpp PRIVATE raylib)

add_executable(double_pendulum_chaos_viz_cpp "double_pendulum_chaos_viz.cpp")
target_link_libraries(double_pendulum_chaos_viz_cpp PRIVATE raylib)

add_executable(double_slit_viz_cpp "double_slit_viz.cpp")
target_link_libraries(double_slit_viz_cpp PRIVATE raylib)

add_executable(electric_charges_interaction_cpp "electric charges interaction.cpp")
target_link_libraries(electric_charges_interaction_cpp PRIVATE raylib)

add_executable(electric_field_cpp "electric field.cpp")
target_link_libraries(electric_field_cpp PRIVATE raylib)

add_executable(enthropy_viz_cpp "enthropy_viz.cpp")
target_link_libraries(enthropy_viz_cpp PRIVATE raylib)

add_executable(exoplanet_transit_lab_viz_cpp "exoplanet_transit_lab_viz.cpp")
target_link_libraries(exoplanet_transit_lab_viz_cpp PRIVATE raylib)

add_executable(fission_fusion_viz_cpp "fission_fusion_viz.cpp")
target_link_libraries(fission_fusion_viz_cpp PRIVATE raylib)

add_executable(fluid_mechanics_channel_viz_cpp "fluid_mechanics_channel_viz.cpp")
target_link_libraries(fluid_mechanics_channel_viz_cpp PRIVATE raylib)

add_executable(fluid_vortex_viz_cpp "fluid_vortex_viz.cpp")
target_link_libraries(fluid_vortex_viz_cpp PRIVATE raylib)

add_executable(gravitational_lensing_viz_cpp "gravitational_lensing_viz.cpp")
target_link_libraries(gravitational_lensing_viz_cpp PRIVATE raylib)

add_executable(gravitational_lensing_playground_viz_cpp "gravitational_lensing_playground_viz.cpp")
target_link_libraries(gravitational_lensing_playground_viz_cpp PRIVATE raylib)

add_executable(gravity_lagrange_viz_cpp "gravity_lagrange_viz.cpp")
target_link_libraries(gravity_lagrange_viz_cpp PRIVATE raylib)

add_executable(higgs_field_viz_cpp "higgs_field_viz.cpp")
target_link_libraries(higgs_field_viz_cpp PRIVATE raylib)

add_executable(higgs_particle_viz_cpp "higgs_particle_viz.cpp")
target_link_libraries(higgs_particle_viz_cpp PRIVATE raylib)

add_executable(hawking_particles_escape_viz_cpp "hawking_particles_escape_viz.cpp")
target_link_libraries(hawking_particles_escape_viz_cpp PRIVATE raylib)

add_executable(hawkin_s_rad_viz_cpp "hawkin's_rad_viz.cpp")
target_link_libraries(hawkin_s_rad_viz_cpp PRIVATE raylib)

add_executable(hydrogen_bomb_viz_cpp "hydrogen_bomb_viz.cpp")
target_link_libraries(hydrogen_bomb_viz_cpp PRIVATE raylib)

add_executable(hr_diagram_evolution_viz_cpp "hr_diagram_evolution_viz.cpp")
target_link_libraries(hr_diagram_evolution_viz_cpp PRIVATE raylib)

add_executable(interferometer_gw_viz_cpp "interferometer_gw_viz.cpp")
target_link_libraries(interferometer_gw_viz_cpp PRIVATE raylib)

add_executable(maxwell_equations_viz_cpp "maxwell_equations_viz.cpp")
target_link_libraries(maxwell_equations_viz_cpp PRIVATE raylib)

add_executable(maxwell_wave_viz_cpp "maxwell_wave_viz.cpp")
target_link_libraries(maxwell_wave_viz_cpp PRIVATE raylib)

add_executable(newton_laws_viz_cpp "newton_laws_viz.cpp")
target_link_libraries(newton_laws_viz_cpp PRIVATE raylib)

add_executable(particle_acc_viz_cpp "particle_acc_viz.cpp")
target_link_libraries(particle_acc_viz_cpp PRIVATE raylib)

add_executable(particle_entang_viz_cpp "particle_entang_viz.cpp")
target_link_libraries(particle_entang_viz_cpp PRIVATE raylib)

add_executable(pulsar_viz_cpp "pulsar_viz.cpp")
target_link_libraries(pulsar_viz_cpp PRIVATE raylib)

add_executable(pulsar_timing_gw_viz_cpp "pulsar_timing_gw_viz.cpp")
target_link_libraries(pulsar_timing_gw_viz_cpp PRIVATE raylib)

add_executable(projectile_drag_viz_cpp "projectile_drag_viz.cpp")
target_link_libraries(projectile_drag_viz_cpp PRIVATE raylib)

add_executable(quantum_cpp "Quantum.cpp")
target_link_libraries(quantum_cpp PRIVATE raylib)

add_executable(quantum_particle_viz_cpp "quantum_particle_viz.cpp")
target_link_libraries(quantum_particle_viz_cpp PRIVATE raylib)

add_executable(quantum_search_cpp "quantum_search.cpp")
target_link_libraries(quantum_search_cpp PRIVATE raylib)

add_executable(quantum_slit_wave_viz_cpp "quantum_slit_wave_viz.cpp")
target_link_libraries(quantum_slit_wave_viz_cpp PRIVATE raylib)

add_executable(quantum_tunneling_viz_cpp "quantum_tunneling_viz.cpp")
target_link_libraries(quantum_tunneling_viz_cpp PRIVATE raylib)

add_executable(quantum_wave_viz_cpp "quantum_wave_viz.cpp")
target_link_libraries(quantum_wave_viz_cpp PRIVATE raylib)

add_executable(relativistic_time_dilation_viz_cpp "relativistic_time_dilation_viz.cpp")
target_link_libraries(relativistic_time_dilation_viz_cpp PRIVATE raylib)

add_executable(schrodinger_cat_viz_cpp "schrodinger_cat_viz.cpp")
target_link_libraries(schrodinger_cat_viz_cpp PRIVATE raylib)

add_executable(simple_harmonic_oscillator_viz_cpp "simple_harmonic_oscillator_viz.cpp")
target_link_libraries(simple_harmonic_oscillator_viz_cpp PRIVATE raylib)

add_executable(galaxy_rotation_dark_matter_viz_cpp "galaxy_rotation_dark_matter_viz.cpp")
target_link_libraries(galaxy_rotation_dark_matter_viz_cpp PRIVATE raylib)

add_executable(solar_system_spacetime_viz_cpp "solar_system_spacetime_viz.cpp")
target_link_libraries(solar_system_spacetime_viz_cpp PRIVATE raylib)

add_executable(thermodynamics_laws_viz_cpp "thermodynamics_laws_viz.cpp")
target_link_libraries(thermodynamics_laws_viz_cpp PRIVATE raylib)

add_executable(two_body_orbit_viz_cpp "two_body_orbit_viz.cpp")
target_link_libraries(two_body_orbit_viz_cpp PRIVATE raylib)

add_executable(sun_planet_spacetime_viz_cpp "sun_planet_spacetime_viz.cpp")
target_link_libraries(sun_planet_spacetime_viz_cpp PRIVATE raylib)

add_executable(uncertainty_wavepacket_viz_cpp "uncertainty_wavepacket_viz.cpp")
target_link_libraries(uncertainty_wavepacket_viz_cpp PRIVATE raylib)

add_executable(wormhole_viz_cpp "wormhole_viz.cpp")
target_link_libraries(wormhole_viz_cpp PRIVATE raylib)
