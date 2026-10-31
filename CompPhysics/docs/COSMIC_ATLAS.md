# Cosmic Atlas — formation and evolution

A standalone raylib/OpenGL field guide with four journeys, twelve selectable
tracks and sixty annotated stages. Everything is procedural: no downloaded image
assets, camera, network connection or extra Python runtime is needed to run it.

```bash
./CompPhysics/scripts/run cosmic_evolution_viz
# Or use the repository menu:
python3 explore.py run cosmic_evolution_viz
```

## Explore

- **Planets:** rocky world, gas giant and ice giant. Follow disk grains, growing
  building blocks, accretion and mature surfaces. The rocky track includes a
  glowing magma surface, differentiated terrain, clouds and an atmospheric rim.
- **Stars:** a 0.2-solar-mass red dwarf, a Sun-like star and a 25-solar-mass star.
  Compare vastly different lifetimes, temperatures and possible remnants.
- **Black holes:** massive-star collapse, an accreting intermediate-mass seed
  and an active galactic nucleus. Watch a hot disk and stylized photon ring;
  compare the selected mass with its non-spinning horizon radius.
- **Galaxies:** a spiral, elliptical and irregular system. Volumetric light,
  clumpy star-forming regions, a bulge and spiral arms illustrate different
  assembly histories rather than a mandatory sequence between galaxy types.

The window resizes with a consistent layout. Drag inside the scene to orbit;
scroll there to zoom. Clicking a journey or type starts its timeline again.
The five stage labels jump directly to the start of their stages. Drag the
progress bar to scrub backward or forward; scrubbing pauses playback.

| Control | Action |
| --- | --- |
| `1`–`4` | Select journey |
| `Tab` | Cycle object type |
| `Space` | Play/pause; replay when finished |
| `Left` / `Right` | Scrub by one percent |
| `R` | Restart time, keeping type and camera |
| `C` | Restore camera |
| `A` | Toggle automatic camera orbit |
| `L` | Toggle scene labels |
| Buttons `0.25x`, `1x`, `4x` | Playback speed |

A journey takes 100 seconds at 1x. It stops at the end so the final state can be
inspected. The camera and surface animation pause with the timeline; manual
camera movement remains available.

## Time, physics and visual interpretation

Each journey has its own elapsed-year clock, starting at the beginning of the
selected scenario. The five stages each receive twenty percent of the slider;
years interpolate linearly **within** a stage, but not across the whole slider.
This lets a seconds-long collapse and a millions-of-years progenitor phase both
be explored. The age label is rounded, so short collapse intervals can share the
same displayed million-year age. It is not the age of the universe.

These are curated educational tracks, not stellar evolution, hydrodynamics,
planetary accretion or cosmological N-body solvers. The formation dates, final
masses and temperatures are representative presets, not predictions fitted to
an observed object. Planet temperatures refer to the mature world; star
surface temperatures refer to the main sequence. Neither readout reports an
instantaneous temperature throughout formation.

- A low-mass red dwarf does not undergo the Sun's red giant evolution. Its
  trillion-year future is theoretical; the universe is too young to have
  observed the endpoint of an isolated star with this mass.
- The massive-star preset chooses a black-hole endpoint. Actual core-collapse
  outcomes depend on composition, mass loss and other properties. Not every
  massive star leaves a black hole or produces a visible supernova.
- The stellar black-hole track displays a **chosen final remnant** mass even
  before collapse. The other two tracks begin with existing seeds. Their
  illustrative growth is `M(t) = M_seed exp(0.55 t / 45 million years)`;
  the factor 0.55 is a chosen effective duty factor. This does not simulate gas
  supply or feedback. The quiet-nucleus image dims without losing hole mass.
- The non-spinning horizon uses `r_s = 2GM/c² = 2.95325 km × M/M_sun`.
  Ring images and disk brightening are artistic approximations, not general
  relativistic ray tracing. Black holes do not emit light from their horizons.
- Galaxy morphology is a selected pathway, not a classification based on mass
  alone. The dark-matter halo is described but is not luminous in the view.
- Size, brightness, orbit speed, surface color and gas density are independently
  rescaled for legibility. A larger drawn object is not a calibrated ruler.
  Clouds and surfaces use seeded procedural noise, and orbital solids follow
  prescribed paths rather than integrated collisions. Ocean formation is a
  conditional example, not an inevitable result for rocky planets.

The GLSL shader reconstructs perspective camera rays, intersects shaded spheres
and disk layers, and integrates optically thin emissive cloud and galaxy volumes.
The scene renders at 1000 × 530 before composition into a 1440 × 900 layout.
The shader is loaded from `resources/shaders/cosmic_evolution.fs`; changing that
file takes effect on the next launch without rebuilding the executable.

## Checks and reproducible renders

```bash
python3 CompPhysics/scripts/check.py
python3 CompPhysics/tests/graphics_smoke.py --filter cosmic_evolution
python3 CompPhysics/tests/cosmic_evolution_graphics.py
```

The numerical suite checks all twelve age clocks, endpoints, stage boundaries,
stellar lifetime ordering and black-hole mass/radius relationships. The desktop
matrix renders all sixty stages and checks shader compilation, nonblank scenes
and a visible change between each track's initial and final states. Captures and
logs go under `CompPhysics/output/reports/` and are not source artifacts.

For a deterministic, hidden-window PNG (chapter 0–3, variant 0–2):

```bash
CompPhysics/output/native/cosmic_evolution_viz_cpp \
  --chapter 3 --variant 0 --progress 0.85 \
  --capture CompPhysics/output/reports/cosmic_spiral.png
```

Rendering requires a desktop OpenGL 3.3-capable context. The ordinary numerical
suite does not need a graphics context. A shader failure exits with an error.

## Scientific reading

The qualitative stages follow these NASA overviews; the numerical presets and
procedural imagery are project-authored illustrative choices.

- [Stars and their lifetimes](https://science.nasa.gov/universe/stars/)
- [Webb and star formation](https://science.nasa.gov/mission/webb/science-overview/science-explainers/webbs-star-formation-discoveries/)
- [Planetary systems and accretion](https://science.nasa.gov/universe/stars/planetary-system/)
- [Black-hole types and formation pathways](https://science.nasa.gov/universe/black-holes/types/)
- [Galaxies](https://science.nasa.gov/universe/galaxies/)
- [Galaxy types](https://science.nasa.gov/universe/galaxies/types/)
