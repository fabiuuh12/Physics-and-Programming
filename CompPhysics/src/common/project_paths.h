#pragma once
#include <cstdlib>
#include <filesystem>

namespace project_paths {
inline std::filesystem::path root() {
    if (const char* configured = std::getenv("COMPPHYSICS_ROOT")) return configured;
#ifdef COMPPHYSICS_ROOT
    return COMPPHYSICS_ROOT;
#else
    return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path();
#endif
}
}
