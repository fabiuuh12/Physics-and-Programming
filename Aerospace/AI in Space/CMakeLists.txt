cmake_minimum_required(VERSION 3.16)

project(ai_in_space LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(space_dynamics
    cpp/src/orbit.cpp
)

target_include_directories(space_dynamics
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

add_executable(rendezvous_cpp
    cpp/apps/rendezvous_cpp.cpp
)

target_link_libraries(rendezvous_cpp
    PRIVATE
        space_dynamics
)
