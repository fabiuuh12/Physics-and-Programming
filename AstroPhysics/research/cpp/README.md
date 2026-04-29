# ASF C++ Prototype

This C++ prototype mirrors the Python v1 model while avoiding external
dependencies. It is intended for numerical checks and later Raylib integration.

Compile the static-field runner:

```bash
c++ -std=c++17 -O2 -I AstroPhysics/research/cpp/include \
  AstroPhysics/research/cpp/src/asf_sim.cpp \
  AstroPhysics/research/cpp/src/run_static_field.cpp \
  -o /tmp/asf_static
```

Run:

```bash
/tmp/asf_static AstroPhysics/research/experiments/baseline_static_field/config.json \
  AstroPhysics/research/results/cpp_static.csv
```

Compile tests:

```bash
c++ -std=c++17 -O2 -I AstroPhysics/research/cpp/include \
  AstroPhysics/research/cpp/src/asf_sim.cpp \
  AstroPhysics/research/cpp/tests/asf_numeric_tests.cpp \
  -o /tmp/asf_tests
```

Run:

```bash
/tmp/asf_tests
```
