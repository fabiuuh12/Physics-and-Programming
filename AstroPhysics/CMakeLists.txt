cmake_minimum_required(VERSION 3.20)
project(astrophysics_cpp_viz LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(raylib CONFIG REQUIRED)

add_executable(4th_dimension_viz_cpp "dimensions/4th_dimension_viz.cpp")
target_link_libraries(4th_dimension_viz_cpp PRIVATE raylib)

add_executable(angular_momentum_viz_cpp "mechanics/angular_momentum_viz.cpp")
target_link_libraries(angular_momentum_viz_cpp PRIVATE raylib)

add_executable(aerodynamics_viz_cpp "mechanics/aerodynamics_viz.cpp")
target_link_libraries(aerodynamics_viz_cpp PRIVATE raylib)

add_executable(hand_biomechanics_viz_cpp "mechanics/hand_biomechanics_viz.cpp")
target_link_libraries(hand_biomechanics_viz_cpp PRIVATE raylib)
if (WIN32)
    target_link_libraries(hand_biomechanics_viz_cpp PRIVATE ws2_32)
endif()

add_executable(atom_viz_cpp "quantum/atom_viz.cpp")
target_link_libraries(atom_viz_cpp PRIVATE raylib)

add_executable(atomic_bomb_viz_cpp "nuclear/atomic_bomb_viz.cpp")
target_link_libraries(atomic_bomb_viz_cpp PRIVATE raylib)

add_executable(blackhole_viz_cpp "gravity/blackhole_viz.cpp")
target_link_libraries(blackhole_viz_cpp PRIVATE raylib)

add_executable(blackhole_realism_viz_cpp "gravity/blackhole_realism_viz.cpp")
target_link_libraries(blackhole_realism_viz_cpp PRIVATE raylib)

add_executable(black_hole_accretion_beaming_viz_cpp "gravity/black_hole_accretion_beaming_viz.cpp")
target_link_libraries(black_hole_accretion_beaming_viz_cpp PRIVATE raylib)

add_executable(quasar_core_viz_cpp "gravity/quasar_core_viz.cpp")
target_link_libraries(quasar_core_viz_cpp PRIVATE raylib)

add_executable(black_hole_particle_field_viz_cpp "gravity/black_hole_particle_field_viz.cpp")
target_link_libraries(black_hole_particle_field_viz_cpp PRIVATE raylib)

add_executable(dual_black_white_hole_viz_cpp "gravity/dual_black_white_hole_viz.cpp")
target_link_libraries(dual_black_white_hole_viz_cpp PRIVATE raylib)

add_executable(collision_bh_viz_cpp "gravity/collision_bh_viz.cpp")
target_link_libraries(collision_bh_viz_cpp PRIVATE raylib)

add_executable(cosmic_expansion_sandbox_viz_cpp "astronomy/cosmic_expansion_sandbox_viz.cpp")
target_link_libraries(cosmic_expansion_sandbox_viz_cpp PRIVATE raylib)

add_executable(damped_forced_oscillator_viz_cpp "mechanics/damped_forced_oscillator_viz.cpp")
target_link_libraries(damped_forced_oscillator_viz_cpp PRIVATE raylib)

add_executable(dark_matter_viz_cpp "astronomy/dark_matter_viz.cpp")
target_link_libraries(dark_matter_viz_cpp PRIVATE raylib)

add_executable(neutron_star_merger_kilonova_viz_cpp "astronomy/neutron_star_merger_kilonova_viz.cpp")
target_link_libraries(neutron_star_merger_kilonova_viz_cpp PRIVATE raylib)

add_executable(doppler_eff_viz_cpp "relativity/doppler_eff_viz.cpp")
target_link_libraries(doppler_eff_viz_cpp PRIVATE raylib)

add_executable(double_pendulum_chaos_viz_cpp "mechanics/double_pendulum_chaos_viz.cpp")
target_link_libraries(double_pendulum_chaos_viz_cpp PRIVATE raylib)

add_executable(double_slit_viz_cpp "quantum/double_slit_viz.cpp")
target_link_libraries(double_slit_viz_cpp PRIVATE raylib)

add_executable(electric_charges_interaction_cpp "electromagnetism/electric charges interaction.cpp")
target_link_libraries(electric_charges_interaction_cpp PRIVATE raylib)

add_executable(electric_field_cpp "electromagnetism/electric field.cpp")
target_link_libraries(electric_field_cpp PRIVATE raylib)

add_executable(magnetosphere_solar_wind_viz_cpp "electromagnetism/magnetosphere_solar_wind_viz.cpp")
target_link_libraries(magnetosphere_solar_wind_viz_cpp PRIVATE raylib)

add_executable(enthropy_viz_cpp "thermodynamics/enthropy_viz.cpp")
target_link_libraries(enthropy_viz_cpp PRIVATE raylib)

add_executable(exoplanet_transit_lab_viz_cpp "astronomy/exoplanet_transit_lab_viz.cpp")
target_link_libraries(exoplanet_transit_lab_viz_cpp PRIVATE raylib)

add_executable(orbital_construction_hand_lab_viz_cpp "astronomy/orbital_construction_hand_lab_viz.cpp")
target_link_libraries(orbital_construction_hand_lab_viz_cpp PRIVATE raylib)
if (WIN32)
    target_link_libraries(orbital_construction_hand_lab_viz_cpp PRIVATE ws2_32)
endif()

add_executable(fission_fusion_viz_cpp "nuclear/fission_fusion_viz.cpp")
target_link_libraries(fission_fusion_viz_cpp PRIVATE raylib)

add_executable(fluid_mechanics_channel_viz_cpp "fluids/fluid_mechanics_channel_viz.cpp")
target_link_libraries(fluid_mechanics_channel_viz_cpp PRIVATE raylib)

add_executable(fluid_vortex_viz_cpp "fluids/fluid_vortex_viz.cpp")
target_link_libraries(fluid_vortex_viz_cpp PRIVATE raylib)

add_executable(gravitational_lensing_viz_cpp "gravity/gravitational_lensing_viz.cpp")
target_link_libraries(gravitational_lensing_viz_cpp PRIVATE raylib)

add_executable(gravitational_lensing_playground_viz_cpp "gravity/gravitational_lensing_playground_viz.cpp")
target_link_libraries(gravitational_lensing_playground_viz_cpp PRIVATE raylib)

add_executable(gravity_lagrange_viz_cpp "gravity/gravity_lagrange_viz.cpp")
target_link_libraries(gravity_lagrange_viz_cpp PRIVATE raylib)

add_executable(higgs_field_viz_cpp "particle_physics/higgs_field_viz.cpp")
target_link_libraries(higgs_field_viz_cpp PRIVATE raylib)

add_executable(higgs_particle_viz_cpp "particle_physics/higgs_particle_viz.cpp")
target_link_libraries(higgs_particle_viz_cpp PRIVATE raylib)

add_executable(hawking_particles_escape_viz_cpp "gravity/hawking_particles_escape_viz.cpp")
target_link_libraries(hawking_particles_escape_viz_cpp PRIVATE raylib)

add_executable(hawkin_s_rad_viz_cpp "gravity/hawkin's_rad_viz.cpp")
target_link_libraries(hawkin_s_rad_viz_cpp PRIVATE raylib)

add_executable(hydrogen_bomb_viz_cpp "nuclear/hydrogen_bomb_viz.cpp")
target_link_libraries(hydrogen_bomb_viz_cpp PRIVATE raylib)

add_executable(hr_diagram_evolution_viz_cpp "astronomy/hr_diagram_evolution_viz.cpp")
target_link_libraries(hr_diagram_evolution_viz_cpp PRIVATE raylib)

add_executable(interferometer_gw_viz_cpp "gravity/interferometer_gw_viz.cpp")
target_link_libraries(interferometer_gw_viz_cpp PRIVATE raylib)

add_executable(maxwell_equations_viz_cpp "electromagnetism/maxwell_equations_viz.cpp")
target_link_libraries(maxwell_equations_viz_cpp PRIVATE raylib)

add_executable(maxwell_wave_viz_cpp "electromagnetism/maxwell_wave_viz.cpp")
target_link_libraries(maxwell_wave_viz_cpp PRIVATE raylib)

add_executable(newton_laws_viz_cpp "mechanics/newton_laws_viz.cpp")
target_link_libraries(newton_laws_viz_cpp PRIVATE raylib)

add_executable(particle_acc_viz_cpp "particle_physics/particle_acc_viz.cpp")
target_link_libraries(particle_acc_viz_cpp PRIVATE raylib)

add_executable(tokamak_confinement_viz_cpp "plasma/tokamak_confinement_viz.cpp")
target_link_libraries(tokamak_confinement_viz_cpp PRIVATE raylib)

add_executable(particle_entang_viz_cpp "particle_physics/particle_entang_viz.cpp")
target_link_libraries(particle_entang_viz_cpp PRIVATE raylib)

add_executable(pulsar_viz_cpp "astronomy/pulsar_viz.cpp")
target_link_libraries(pulsar_viz_cpp PRIVATE raylib)

add_executable(pulsar_timing_gw_viz_cpp "astronomy/pulsar_timing_gw_viz.cpp")
target_link_libraries(pulsar_timing_gw_viz_cpp PRIVATE raylib)

add_executable(projectile_drag_viz_cpp "mechanics/projectile_drag_viz.cpp")
target_link_libraries(projectile_drag_viz_cpp PRIVATE raylib)

add_executable(quantum_cpp "quantum/Quantum.cpp")
target_link_libraries(quantum_cpp PRIVATE raylib)

add_executable(quantum_particle_viz_cpp "quantum/quantum_particle_viz.cpp")
target_link_libraries(quantum_particle_viz_cpp PRIVATE raylib)

add_executable(quantum_search_cpp "quantum/quantum_search.cpp")
target_link_libraries(quantum_search_cpp PRIVATE raylib)

add_executable(quantum_slit_wave_viz_cpp "quantum/quantum_slit_wave_viz.cpp")
target_link_libraries(quantum_slit_wave_viz_cpp PRIVATE raylib)

add_executable(quantum_tunneling_viz_cpp "quantum/quantum_tunneling_viz.cpp")
target_link_libraries(quantum_tunneling_viz_cpp PRIVATE raylib)

add_executable(quantum_wave_viz_cpp "quantum/quantum_wave_viz.cpp")
target_link_libraries(quantum_wave_viz_cpp PRIVATE raylib)

add_executable(relativistic_time_dilation_viz_cpp "relativity/relativistic_time_dilation_viz.cpp")
target_link_libraries(relativistic_time_dilation_viz_cpp PRIVATE raylib)

add_executable(penrose_diagram_3d_viz_cpp "relativity/penrose_diagram_3d_viz.cpp")
target_link_libraries(penrose_diagram_3d_viz_cpp PRIVATE raylib)

add_executable(schrodinger_cat_viz_cpp "quantum/schrodinger_cat_viz.cpp")
target_link_libraries(schrodinger_cat_viz_cpp PRIVATE raylib)

add_executable(simple_harmonic_oscillator_viz_cpp "mechanics/simple_harmonic_oscillator_viz.cpp")
target_link_libraries(simple_harmonic_oscillator_viz_cpp PRIVATE raylib)

add_executable(galaxy_rotation_dark_matter_viz_cpp "astronomy/galaxy_rotation_dark_matter_viz.cpp")
target_link_libraries(galaxy_rotation_dark_matter_viz_cpp PRIVATE raylib)

add_executable(galaxy_merger_nbody_viz_cpp "astronomy/galaxy_merger_nbody_viz.cpp")
target_link_libraries(galaxy_merger_nbody_viz_cpp PRIVATE raylib)

add_executable(solar_system_spacetime_viz_cpp "gravity/solar_system_spacetime_viz.cpp")
target_link_libraries(solar_system_spacetime_viz_cpp PRIVATE raylib)

add_executable(solar_system_solar_wind_viz_cpp "astronomy/solar_system_solar_wind_viz.cpp")
target_link_libraries(solar_system_solar_wind_viz_cpp PRIVATE raylib)

add_executable(thermodynamics_laws_viz_cpp "thermodynamics/thermodynamics_laws_viz.cpp")
target_link_libraries(thermodynamics_laws_viz_cpp PRIVATE raylib)

add_executable(two_body_orbit_viz_cpp "mechanics/two_body_orbit_viz.cpp")
target_link_libraries(two_body_orbit_viz_cpp PRIVATE raylib)

add_executable(sun_planet_spacetime_viz_cpp "gravity/sun_planet_spacetime_viz.cpp")
target_link_libraries(sun_planet_spacetime_viz_cpp PRIVATE raylib)

add_executable(uncertainty_wavepacket_viz_cpp "quantum/uncertainty_wavepacket_viz.cpp")
target_link_libraries(uncertainty_wavepacket_viz_cpp PRIVATE raylib)

add_executable(wormhole_viz_cpp "gravity/wormhole_viz.cpp")
target_link_libraries(wormhole_viz_cpp PRIVATE raylib)

add_executable(wormhole_gateway_viz_cpp "gravity/wormhole_gateway_viz.cpp")
target_link_libraries(wormhole_gateway_viz_cpp PRIVATE raylib)

add_executable(wormhole_hand_lab_viz_cpp "gravity/wormhole_hand_lab_viz.cpp")
target_link_libraries(wormhole_hand_lab_viz_cpp PRIVATE raylib)
if (WIN32)
    target_link_libraries(wormhole_hand_lab_viz_cpp PRIVATE ws2_32)
endif()

add_executable(defensive_sys_3d_cpp "DefensiveSys/defensive_sys_3d.cpp")
target_link_libraries(defensive_sys_3d_cpp PRIVATE raylib)
if (WIN32)
    target_link_libraries(defensive_sys_3d_cpp PRIVATE ws2_32)
endif()

add_executable(vision_two_hands_scene_cpp "vision/two_hands_scene.cpp")
target_link_libraries(vision_two_hands_scene_cpp PRIVATE raylib)
if (WIN32)
    target_link_libraries(vision_two_hands_scene_cpp PRIVATE ws2_32)
endif()

add_executable(gravitational_microlensing_viz_cpp "gravity/gravitational_microlensing_viz.cpp")
target_link_libraries(gravitational_microlensing_viz_cpp PRIVATE raylib)

add_executable(pulsar_beam_timing_viz_cpp "astronomy/pulsar_beam_timing_viz.cpp")
target_link_libraries(pulsar_beam_timing_viz_cpp PRIVATE raylib)

add_executable(supernova_remnant_expansion_viz_cpp "astronomy/supernova_remnant_expansion_viz.cpp")
target_link_libraries(supernova_remnant_expansion_viz_cpp PRIVATE raylib)
