cmake_minimum_required(VERSION 3.16)
project(alice_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(alice
  src/main.cpp
  src/string_utils.cpp
  src/config.cpp
  src/intent.cpp
  src/memory_store.cpp
  src/executor.cpp
  src/llm_client.cpp
  src/brain.cpp
)

target_include_directories(alice PRIVATE include)

if(APPLE)
  target_compile_definitions(alice PRIVATE ALICE_PLATFORM_MACOS=1)
endif()
