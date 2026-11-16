# Nuclear power plant

An orbitable C++ / raylib cutaway of a pressurized-water plant. Fuel assemblies,
control rods, pressurizer, pumps, steam-generator tubes, turbine blades, copper
stator windings, condenser, cooling tower, transformer and transmission line are
visible. The turbine and generator share a shaft. Three distinct water circuits
carry animated tracers; primary reactor water never flows to the turbine.

Run the plant with hand tracking (Python only handles the webcam):

```bash
python3 CompPhysics/src/simulations/nuclear/nuclear_plant_hands.py
```

Install `CompPhysics/config/requirements.txt` in your Python environment for
OpenCV, MediaPipe and NumPy. Allow webcam access when your OS requests it. The
launcher builds the C++ executable, opens the existing mirrored hand-tracker
preview and stops its child processes when the plant closes. On macOS/Linux it
uses the project's native build script and vendored raylib; Windows uses CMake.
If tracking dependencies or the camera are unavailable, mouse controls still work.

To run without a camera:

```bash
python3 CompPhysics/src/simulations/nuclear/nuclear_plant_hands.py --mouse-only
# Or on macOS/Linux:
./CompPhysics/scripts/run nuclear_power_plant_viz
```

On the physical console at the front of the model:

- Move an open hand over the orange rod lever, pinch, and slide vertically.
  Down inserts the rods and reduces heat production.
- Pinch the load dial and move around its face to change the generator load.
- Pinch the red **SCRAM** button to insert all rods. Press **R** to restart.
- Release the pinch to let go. Either hand works; each control has one owner.
  Losing tracking releases the control, and an open hand is required to grab again.
- The mouse operates the same controls. Drag elsewhere to orbit; scroll to zoom.

The hand cursor uses mirrored webcam coordinates. Keep the preview mirrored (its
**M** key changes mirroring), move your hand to match a cursor with a console
control, open your fingers, then pinch. The tracker sends the existing 133-value
landmark packet over localhost UDP port 50515. Other demos using this port should
be closed; the plant reports a port conflict and retains mouse controls.

**Space/P** pauses the model; **C** toggles cutaway; **L** toggles plant labels;
**F** toggles flow tracers; **Up/Down** moves the rods; **Left/Right** changes load;
**X** shuts down; **R** restarts. The five tabs explain the energy conversion stages.

The C++ source is split between the application (`nuclear_power_plant_viz.cpp`),
energy model (`plant/model.h`), renderer (`plant/renderer.h`) and interaction
(`plant/controls.h`). Python launches the processes and performs hand tracking;
it does not render or simulate the plant.

This is an educational energy-flow model, not an engineering reactor simulator.
Rod insertion changes an authored heat target, followed by thermal and rotor lag.
The display uses a nominal 3000 MW thermal plant, 36% maximum gross conversion and
2.5% auxiliary load. Net electricity plus rejected heat equals transferred thermal
power. Shutdown heat falls with a simplified lag; decay-heat kinetics, pressure,
reactivity feedback, protection systems and grid synchronization are not modeled.
Rotor speed and tracer motion are slowed for visibility. Cooling-tower plumes
represent water vapor from the separate cooling circuit, not reactor exhaust.

The circuit layout follows the [NRC PWR overview](https://www.nrc.gov/reactors/power/pwrs)
and [electricity-generation explanation](https://www.nrc.gov/education-regulatory-research/the-student-corner/science-101/how-does-a-nuclear-power-plant-make-electricity).

For repeatable render checks, `COMPPHYSICS_SMOKE_FRAMES=8` captures a preview in
`CompPhysics/output/reports/`; `COMPPHYSICS_PLANT_RODS=1` sets inserted rods during
smoke runs. Normal interactive sessions do not capture images or close automatically.

`python3 CompPhysics/tests/plant_hands_smoke.py` replays two simultaneous hands
against the real console, then checks a pinch on the shutdown button. It needs a
graphics desktop but does not use a webcam. Model tests also cover energy balance,
shutdown, pinch capture, release and tracking-loss recovery.
