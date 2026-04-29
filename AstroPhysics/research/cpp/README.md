# C++ Cislunar Navigation Core

The C++ code mirrors the core CR3BP propagation and navigation-risk equation
without external dependencies.

Compile tests:

```bash
c++ -std=c++17 -O2 -I AstroPhysics/research/cpp/include \
  AstroPhysics/research/cpp/src/cislunar_nav.cpp \
  AstroPhysics/research/cpp/tests/cislunar_numeric_tests.cpp \
  -o /private/tmp/cislunar_cpp_tests
```

Run:

```bash
/private/tmp/cislunar_cpp_tests
```

Compile the demo trajectory writer:

```bash
c++ -std=c++17 -O2 -I AstroPhysics/research/cpp/include \
  AstroPhysics/research/cpp/src/cislunar_nav.cpp \
  AstroPhysics/research/cpp/src/run_demo.cpp \
  -o /private/tmp/cislunar_demo
```
