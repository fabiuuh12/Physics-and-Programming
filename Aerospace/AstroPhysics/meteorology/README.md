# Meteorology Visualizations

Interactive C++/raylib demos for Earth weather and atmospheric dynamics.

## Demos

| Target | Source | Focus |
| --- | --- | --- |
| `earth_weather_globe_viz_cpp` | `earth_weather_globe_viz.cpp` | Rotating Earth globe with synthetic pressure, temperature, cloud bands, jet streams, wind particles, and cyclone systems |

## Controls

- Mouse drag: orbit camera
- Mouse wheel: zoom
- `1`: toggle temperature layer
- `2`: toggle pressure layer
- `3`: toggle wind particles
- `+/-`: change simulation speed
- `Space`: pause or resume
- `R`: reset

The current model is synthetic and offline. A future version can ingest live NOAA, Open-Meteo, or GRIB/NetCDF weather data and map it onto the same globe.
