cmake_minimum_required(VERSION 3.20)
project(computational_physics_cpp_viz LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions(COMPPHYSICS_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

find_package(raylib CONFIG REQUIRED)

add_executable(engine_airflow_viz_cpp "src/simulations/fluids/engine_airflow_viz.cpp")
target_link_libraries(engine_airflow_viz_cpp PRIVATE raylib)

add_executable(shallow_water_sandbox_viz_cpp "src/simulations/fluids/shallow_water_sandbox_viz.cpp")
target_link_libraries(shallow_water_sandbox_viz_cpp PRIVATE raylib)

add_executable(4th_dimension_viz_cpp "src/simulations/dimensions/4th_dimension_viz.cpp")
target_link_libraries(4th_dimension_viz_cpp PRIVATE raylib)

add_executable(angular_momentum_viz_cpp "src/simulations/mechanics/angular_momentum_viz.cpp")
target_link_libraries(angular_momentum_viz_cpp PRIVATE raylib)

add_executable(aerodynamics_viz_cpp "src/simulations/mechanics/aerodynamics_viz.cpp")
target_link_libraries(aerodynamics_viz_cpp PRIVATE raylib)

add_executable(artemis_voyager_missions_viz_cpp "src/simulations/astronomy/artemis_voyager_missions_viz.cpp")
target_link_libraries(artemis_voyager_missions_viz_cpp PRIVATE raylib)

add_executable(hand_biomechanics_viz_cpp "src/simulations/mechanics/hand_biomechanics_viz.cpp")
target_link_libraries(hand_biomechanics_viz_cpp PRIVATE raylib)
if (WIN32)
    target_link_libraries(hand_biomechanics_viz_cpp PRIVATE ws2_32)
endif()

add_executable(hand_tesla_coil_viz_cpp "src/simulations/mechanics/hand_tesla_coil_viz.cpp")
target_link_libraries(hand_tesla_coil_viz_cpp PRIVATE raylib)
if (WIN32)
    target_link_libraries(hand_tesla_coil_viz_cpp PRIVATE ws2_32)
endif()

add_executable(atom_viz_cpp "src/simulations/quantum/atom_viz.cpp")
target_link_libraries(atom_viz_cpp PRIVATE raylib)

add_executable(hydrogen_orbital_explorer_viz_cpp "src/simulations/quantum/hydrogen_orbital_explorer_viz.cpp")
target_link_libraries(hydrogen_orbital_explorer_viz_cpp PRIVATE raylib)

add_executable(atomic_bomb_viz_cpp "src/simulations/nuclear/atomic_bomb_viz.cpp")
target_link_libraries(atomic_bomb_viz_cpp PRIVATE raylib)

add_executable(blackhole_viz_cpp "src/simulations/gravity/blackhole_viz.cpp")
target_link_libraries(blackhole_viz_cpp PRIVATE raylib)

add_executable(blackhole_realism_viz_cpp "src/simulations/gravity/blackhole_realism_viz.cpp")
target_link_libraries(blackhole_realism_viz_cpp PRIVATE raylib)

add_executable(black_hole_accretion_beaming_viz_cpp "src/simulations/gravity/black_hole_accretion_beaming_viz.cpp")
target_link_libraries(black_hole_accretion_beaming_viz_cpp PRIVATE raylib)

add_executable(quasar_core_viz_cpp "src/simulations/gravity/quasar_core_viz.cpp")
target_link_libraries(quasar_core_viz_cpp PRIVATE raylib)

add_executable(black_hole_particle_field_viz_cpp "src/simulations/gravity/black_hole_particle_field_viz.cpp")
target_link_libraries(black_hole_particle_field_viz_cpp PRIVATE raylib)

add_executable(dual_black_white_hole_viz_cpp "src/simulations/gravity/dual_black_white_hole_viz.cpp")
target_link_libraries(dual_black_white_hole_viz_cpp PRIVATE raylib)

add_executable(collision_bh_viz_cpp "src/simulations/gravity/collision_bh_viz.cpp")
target_link_libraries(collision_bh_viz_cpp PRIVATE raylib)

add_executable(cosmic_expansion_sandbox_viz_cpp "src/simulations/astronomy/cosmic_expansion_sandbox_viz.cpp")
target_link_libraries(cosmic_expansion_sandbox_viz_cpp PRIVATE raylib)

add_executable(procedural_rose_viz_cpp "src/simulations/mathematics/procedural_rose_viz.cpp")
target_link_libraries(procedural_rose_viz_cpp PRIVATE raylib)

add_executable(damped_forced_oscillator_viz_cpp "src/simulations/mechanics/damped_forced_oscillator_viz.cpp")
target_link_libraries(damped_forced_oscillator_viz_cpp PRIVATE raylib)

add_executable(dark_matter_viz_cpp "src/simulations/astronomy/dark_matter_viz.cpp")
target_link_libraries(dark_matter_viz_cpp PRIVATE raylib)

add_executable(neutron_star_merger_kilonova_viz_cpp "src/simulations/astronomy/neutron_star_merger_kilonova_viz.cpp")
target_link_libraries(neutron_star_merger_kilonova_viz_cpp PRIVATE raylib)

add_executable(doppler_eff_viz_cpp "src/simulations/relativity/doppler_eff_viz.cpp")
target_link_libraries(doppler_eff_viz_cpp PRIVATE raylib)

add_executable(double_pendulum_chaos_viz_cpp "src/simulations/mechanics/double_pendulum_chaos_viz.cpp")
target_link_libraries(double_pendulum_chaos_viz_cpp PRIVATE raylib)

add_executable(electric_charges_interaction_cpp "src/simulations/electromagnetism/electric charges interaction.cpp")
target_link_libraries(electric_charges_interaction_cpp PRIVATE raylib)

add_executable(electric_field_cpp "src/simulations/electromagnetism/electric field.cpp")
target_link_libraries(electric_field_cpp PRIVATE raylib)

add_executable(circuit_em_energy_flow_viz_cpp "src/simulations/electromagnetism/circuit_em_energy_flow_viz.cpp")
target_link_libraries(circuit_em_energy_flow_viz_cpp PRIVATE raylib)
if (WIN32)
    target_link_libraries(circuit_em_energy_flow_viz_cpp PRIVATE ws2_32)
endif()

add_executable(em_helical_poynting_viz_cpp "src/simulations/electromagnetism/em_helical_poynting_viz.cpp")
target_link_libraries(em_helical_poynting_viz_cpp PRIVATE raylib)

add_executable(magnetosphere_solar_wind_viz_cpp "src/simulations/electromagnetism/magnetosphere_solar_wind_viz.cpp")
target_link_libraries(magnetosphere_solar_wind_viz_cpp PRIVATE raylib)

add_executable(earth_weather_globe_viz_cpp "src/simulations/meteorology/earth_weather_globe_viz.cpp")
target_link_libraries(earth_weather_globe_viz_cpp PRIVATE raylib)

add_executable(enthropy_viz_cpp "src/simulations/thermodynamics/enthropy_viz.cpp")
target_link_libraries(enthropy_viz_cpp PRIVATE raylib)

add_executable(exoplanet_transit_lab_viz_cpp "src/simulations/astronomy/exoplanet_transit_lab_viz.cpp")
target_link_libraries(exoplanet_transit_lab_viz_cpp PRIVATE raylib)

add_executable(orbital_construction_hand_lab_viz_cpp "src/simulations/astronomy/orbital_construction_hand_lab_viz.cpp")
target_link_libraries(orbital_construction_hand_lab_viz_cpp PRIVATE raylib)
if (WIN32)
    target_link_libraries(orbital_construction_hand_lab_viz_cpp PRIVATE ws2_32)
endif()

add_executable(fission_fusion_viz_cpp "src/simulations/nuclear/fission_fusion_viz.cpp")
target_link_libraries(fission_fusion_viz_cpp PRIVATE raylib)

add_executable(feynman_diagram_simulator_cpp "src/simulations/particle_physics/feynman_diagram_simulator.cpp")
target_link_libraries(feynman_diagram_simulator_cpp PRIVATE raylib)

add_executable(fluid_mechanics_channel_viz_cpp "src/simulations/fluids/fluid_mechanics_channel_viz.cpp")
target_link_libraries(fluid_mechanics_channel_viz_cpp PRIVATE raylib)

add_executable(fluid_vortex_viz_cpp "src/simulations/fluids/fluid_vortex_viz.cpp")
target_link_libraries(fluid_vortex_viz_cpp PRIVATE raylib)

add_executable(helical_wave_lab_viz_cpp "src/simulations/waves/helical_wave_lab_viz.cpp")
target_link_libraries(helical_wave_lab_viz_cpp PRIVATE raylib)

add_executable(gravitational_lensing_viz_cpp "src/simulations/gravity/gravitational_lensing_viz.cpp")
target_link_libraries(gravitational_lensing_viz_cpp PRIVATE raylib)

add_executable(gravitational_lensing_animation_viz_cpp "src/simulations/gravity/gravitational_lensing_animation_viz.cpp")
target_link_libraries(gravitational_lensing_animation_viz_cpp PRIVATE raylib)

add_executable(gravitational_lensing_playground_viz_cpp "src/simulations/gravity/gravitational_lensing_playground_viz.cpp")
target_link_libraries(gravitational_lensing_playground_viz_cpp PRIVATE raylib)

add_executable(gravity_well_grid_viz_cpp "src/simulations/gravity/gravity_well_grid_viz.cpp")
target_link_libraries(gravity_well_grid_viz_cpp PRIVATE raylib)

add_executable(gravity_lagrange_viz_cpp "src/simulations/gravity/gravity_lagrange_viz.cpp")
target_link_libraries(gravity_lagrange_viz_cpp PRIVATE raylib)

add_executable(higgs_field_viz_cpp "src/simulations/particle_physics/higgs_field_viz.cpp")
target_link_libraries(higgs_field_viz_cpp PRIVATE raylib)

add_executable(higgs_particle_viz_cpp "src/simulations/particle_physics/higgs_particle_viz.cpp")
target_link_libraries(higgs_particle_viz_cpp PRIVATE raylib)

add_executable(hawking_particles_escape_viz_cpp "src/simulations/gravity/hawking_particles_escape_viz.cpp")
target_link_libraries(hawking_particles_escape_viz_cpp PRIVATE raylib)

add_executable(hawkin_s_rad_viz_cpp "src/simulations/gravity/hawkin's_rad_viz.cpp")
target_link_libraries(hawkin_s_rad_viz_cpp PRIVATE raylib)

add_executable(hydrogen_bomb_viz_cpp "src/simulations/nuclear/hydrogen_bomb_viz.cpp")
target_link_libraries(hydrogen_bomb_viz_cpp PRIVATE raylib)

add_executable(nuclear_power_plant_viz_cpp "src/simulations/nuclear/nuclear_power_plant_viz.cpp")
target_link_libraries(nuclear_power_plant_viz_cpp PRIVATE raylib)

add_executable(hr_diagram_evolution_viz_cpp "src/simulations/astronomy/hr_diagram_evolution_viz.cpp")
target_link_libraries(hr_diagram_evolution_viz_cpp PRIVATE raylib)

add_executable(hohmann_transfer_viz_cpp "src/simulations/orbital_mechanics/hohmann_transfer_viz.cpp")
target_link_libraries(hohmann_transfer_viz_cpp PRIVATE raylib)

add_executable(interferometer_gw_viz_cpp "src/simulations/gravity/interferometer_gw_viz.cpp")
target_link_libraries(interferometer_gw_viz_cpp PRIVATE raylib)

add_executable(launch_window_porkchop_viz_cpp "src/simulations/orbital_mechanics/launch_window_porkchop_viz.cpp")
target_link_libraries(launch_window_porkchop_viz_cpp PRIVATE raylib)

add_executable(solar_system_orbit_planner_viz_cpp "src/simulations/orbital_mechanics/solar_system_orbit_planner_viz.cpp")
target_link_libraries(solar_system_orbit_planner_viz_cpp PRIVATE raylib)

add_executable(three_body_problem_viz_cpp "src/simulations/gravity/three_body_problem_viz.cpp")
target_link_libraries(three_body_problem_viz_cpp PRIVATE raylib)

add_executable(maxwell_equations_viz_cpp "src/simulations/electromagnetism/maxwell_equations_viz.cpp")
target_link_libraries(maxwell_equations_viz_cpp PRIVATE raylib)

add_executable(maxwell_wave_viz_cpp "src/simulations/electromagnetism/maxwell_wave_viz.cpp")
target_link_libraries(maxwell_wave_viz_cpp PRIVATE raylib)

add_executable(em_wave_field_lab_viz_cpp "src/simulations/electromagnetism/em_wave_field_lab_viz.cpp")
target_link_libraries(em_wave_field_lab_viz_cpp PRIVATE raylib)

add_executable(newton_laws_viz_cpp "src/simulations/mechanics/newton_laws_viz.cpp")
target_link_libraries(newton_laws_viz_cpp PRIVATE raylib)

add_executable(particle_acc_viz_cpp "src/simulations/particle_physics/particle_acc_viz.cpp")
target_link_libraries(particle_acc_viz_cpp PRIVATE raylib)

add_executable(tokamak_confinement_viz_cpp "src/simulations/plasma/tokamak_confinement_viz.cpp")
target_link_libraries(tokamak_confinement_viz_cpp PRIVATE raylib)

add_executable(particle_entang_viz_cpp "src/simulations/particle_physics/particle_entang_viz.cpp")
target_link_libraries(particle_entang_viz_cpp PRIVATE raylib)

add_executable(pulsar_viz_cpp "src/simulations/astronomy/pulsar_viz.cpp")
target_link_libraries(pulsar_viz_cpp PRIVATE raylib)

add_executable(pulsar_timing_gw_viz_cpp "src/simulations/astronomy/pulsar_timing_gw_viz.cpp")
target_link_libraries(pulsar_timing_gw_viz_cpp PRIVATE raylib)

add_executable(projectile_drag_viz_cpp "src/simulations/mechanics/projectile_drag_viz.cpp")
target_link_libraries(projectile_drag_viz_cpp PRIVATE raylib)

add_executable(quantum_cpp "src/simulations/quantum/Quantum.cpp")
target_link_libraries(quantum_cpp PRIVATE raylib)

add_executable(quantum_particle_viz_cpp "src/simulations/quantum/quantum_particle_viz.cpp")
target_link_libraries(quantum_particle_viz_cpp PRIVATE raylib)

add_executable(field_excitation_viz_cpp "src/simulations/quantum/field_excitation_viz.cpp")
target_link_libraries(field_excitation_viz_cpp PRIVATE raylib)

add_executable(probability_field_wave_merge_viz_cpp "src/simulations/quantum/probability_field_wave_merge_viz.cpp")
target_link_libraries(probability_field_wave_merge_viz_cpp PRIVATE raylib)

add_executable(quantum_search_cpp "src/simulations/quantum/quantum_search.cpp")
target_link_libraries(quantum_search_cpp PRIVATE raylib)

add_executable(quantum_slit_wave_viz_cpp "src/simulations/quantum/quantum_slit_wave_viz.cpp")
target_link_libraries(quantum_slit_wave_viz_cpp PRIVATE raylib)

add_executable(quantum_tunneling_viz_cpp "src/simulations/quantum/quantum_tunneling_viz.cpp")
target_link_libraries(quantum_tunneling_viz_cpp PRIVATE raylib)

add_executable(quantum_wave_viz_cpp "src/simulations/quantum/quantum_wave_viz.cpp")
target_link_libraries(quantum_wave_viz_cpp PRIVATE raylib)

add_executable(relativistic_time_dilation_viz_cpp "src/simulations/relativity/relativistic_time_dilation_viz.cpp")
target_link_libraries(relativistic_time_dilation_viz_cpp PRIVATE raylib)

add_executable(penrose_diagram_3d_viz_cpp "src/simulations/relativity/penrose_diagram_3d_viz.cpp")
target_link_libraries(penrose_diagram_3d_viz_cpp PRIVATE raylib)

add_executable(schrodinger_cat_viz_cpp "src/simulations/quantum/schrodinger_cat_viz.cpp")
target_link_libraries(schrodinger_cat_viz_cpp PRIVATE raylib)

add_executable(simple_harmonic_oscillator_viz_cpp "src/simulations/mechanics/simple_harmonic_oscillator_viz.cpp")
target_link_libraries(simple_harmonic_oscillator_viz_cpp PRIVATE raylib)

add_executable(galaxy_rotation_dark_matter_viz_cpp "src/simulations/astronomy/galaxy_rotation_dark_matter_viz.cpp")
target_link_libraries(galaxy_rotation_dark_matter_viz_cpp PRIVATE raylib)

add_executable(galaxy_merger_nbody_viz_cpp "src/simulations/astronomy/galaxy_merger_nbody_viz.cpp")
target_link_libraries(galaxy_merger_nbody_viz_cpp PRIVATE raylib)

add_executable(solar_system_spacetime_viz_cpp "src/simulations/gravity/solar_system_spacetime_viz.cpp")
target_link_libraries(solar_system_spacetime_viz_cpp PRIVATE raylib)

add_executable(solar_system_solar_wind_viz_cpp "src/simulations/astronomy/solar_system_solar_wind_viz.cpp")
target_link_libraries(solar_system_solar_wind_viz_cpp PRIVATE raylib)

add_executable(thermodynamics_laws_viz_cpp "src/simulations/thermodynamics/thermodynamics_laws_viz.cpp")
target_link_libraries(thermodynamics_laws_viz_cpp PRIVATE raylib)

add_executable(two_body_orbit_viz_cpp "src/simulations/mechanics/two_body_orbit_viz.cpp")
target_link_libraries(two_body_orbit_viz_cpp PRIVATE raylib)

add_executable(sun_planet_spacetime_viz_cpp "src/simulations/gravity/sun_planet_spacetime_viz.cpp")
target_link_libraries(sun_planet_spacetime_viz_cpp PRIVATE raylib)

add_executable(uncertainty_wavepacket_viz_cpp "src/simulations/quantum/uncertainty_wavepacket_viz.cpp")
target_link_libraries(uncertainty_wavepacket_viz_cpp PRIVATE raylib)

add_executable(wormhole_viz_cpp "src/simulations/gravity/wormhole_viz.cpp")
target_link_libraries(wormhole_viz_cpp PRIVATE raylib)

add_executable(wormhole_gateway_viz_cpp "src/simulations/gravity/wormhole_gateway_viz.cpp")
target_link_libraries(wormhole_gateway_viz_cpp PRIVATE raylib)

add_executable(wormhole_hand_lab_viz_cpp "src/simulations/gravity/wormhole_hand_lab_viz.cpp")
target_link_libraries(wormhole_hand_lab_viz_cpp PRIVATE raylib)
if (WIN32)
    target_link_libraries(wormhole_hand_lab_viz_cpp PRIVATE ws2_32)
endif()

add_executable(defensive_sys_3d_cpp "src/simulations/defensive_systems/defensive_sys_3d.cpp")
target_link_libraries(defensive_sys_3d_cpp PRIVATE raylib)
if (WIN32)
    target_link_libraries(defensive_sys_3d_cpp PRIVATE ws2_32)
endif()

add_executable(vision_two_hands_scene_cpp "src/vision/two_hands_scene.cpp")
target_link_libraries(vision_two_hands_scene_cpp PRIVATE raylib)
if (WIN32)
    target_link_libraries(vision_two_hands_scene_cpp PRIVATE ws2_32)
endif()

add_executable(gravitational_microlensing_viz_cpp "src/simulations/gravity/gravitational_microlensing_viz.cpp")
target_link_libraries(gravitational_microlensing_viz_cpp PRIVATE raylib)

add_executable(pulsar_beam_timing_viz_cpp "src/simulations/astronomy/pulsar_beam_timing_viz.cpp")
target_link_libraries(pulsar_beam_timing_viz_cpp PRIVATE raylib)

add_executable(supernova_remnant_expansion_viz_cpp "src/simulations/astronomy/supernova_remnant_expansion_viz.cpp")
target_link_libraries(supernova_remnant_expansion_viz_cpp PRIVATE raylib)

add_executable(planet_magnetosphere_compare_viz_cpp "src/simulations/astronomy/planet_magnetosphere_compare_viz.cpp")
target_link_libraries(planet_magnetosphere_compare_viz_cpp PRIVATE raylib)

add_executable(observable_universe_scale_viz_cpp "src/simulations/astronomy/observable_universe_scale_viz.cpp")
target_link_libraries(observable_universe_scale_viz_cpp PRIVATE raylib)
