cmake_minimum_required(VERSION 3.16)

if(APPLE)
  project(alice_cpp LANGUAGES CXX OBJCXX)
else()
  project(alice_cpp LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ALICE_SOURCES
  src/main.cpp
  src/string_utils.cpp
  src/config.cpp
  src/intent.cpp
  src/memory_store.cpp
  src/executor.cpp
  src/llm_client.cpp
  src/brain.cpp
)

if(APPLE)
  list(APPEND ALICE_SOURCES src/ui_macos.mm src/voice_listener_macos.mm src/face_tracker_macos.mm)
else()
  list(APPEND ALICE_SOURCES src/ui_stub.cpp src/voice_listener_stub.cpp src/face_tracker_stub.cpp)
endif()

add_executable(alice ${ALICE_SOURCES})

target_include_directories(alice PRIVATE include)

if(APPLE)
  set_source_files_properties(src/ui_macos.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
  set_source_files_properties(src/voice_listener_macos.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
  set_source_files_properties(src/face_tracker_macos.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
  target_link_options(alice PRIVATE "-Wl,-sectcreate,__TEXT,__info_plist,${CMAKE_CURRENT_SOURCE_DIR}/resources/AliceInfo.plist")
  target_link_libraries(alice
    "-framework AppKit"
    "-framework SceneKit"
    "-framework AVFoundation"
    "-framework Speech"
    "-framework Vision"
    "-framework CoreMedia"
    "-framework CoreVideo"
  )
endif()
