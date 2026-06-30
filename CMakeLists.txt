cmake_minimum_required(VERSION 3.10)
project(rkllm_server CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Path to the rknn-llm runtime (from airockchip/rknn-llm).
#
# Set via cmake -DRKLLM_RUNTIME_DIR=... or export RKLLM_RUNTIME_DIR=...
# The directory must contain:
#   <RKLLM_RUNTIME_DIR>/include/rkllm.h
#   <RKLLM_RUNTIME_DIR>/aarch64/librkllmrt.so
#
# Tested against airockchip/rknn-llm release-v1.2.1.
if(NOT DEFINED RKLLM_RUNTIME_DIR)
    if(DEFINED ENV{RKLLM_RUNTIME_DIR})
        set(RKLLM_RUNTIME_DIR "$ENV{RKLLM_RUNTIME_DIR}")
    else()
        set(RKLLM_RUNTIME_DIR "$ENV{HOME}/rknn-llm/rkllm-runtime/Linux/librkllm_api")
    endif()
endif()

message(STATUS "rkllm runtime dir: ${RKLLM_RUNTIME_DIR}")

if(NOT EXISTS "${RKLLM_RUNTIME_DIR}/include/rkllm.h")
    message(FATAL_ERROR
        "rkllm.h not found at ${RKLLM_RUNTIME_DIR}/include/rkllm.h\n"
        "Get the runtime from https://github.com/airockchip/rknn-llm\n"
        "Then point to it: cmake -DRKLLM_RUNTIME_DIR=/path/to/librkllm_api ..")
endif()

include_directories(${RKLLM_RUNTIME_DIR}/include)
link_directories(${RKLLM_RUNTIME_DIR}/aarch64)

find_package(Threads REQUIRED)

add_executable(rkllm_server src/main.cpp)

target_include_directories(rkllm_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_link_libraries(rkllm_server PRIVATE
    rkllmrt
    Threads::Threads
)

target_compile_options(rkllm_server PRIVATE
    -O2
    -Wall
    -Wno-unused-parameter
)

# Bake an rpath so the binary finds librkllmrt.so without LD_LIBRARY_PATH gymnastics.
set_target_properties(rkllm_server PROPERTIES
    INSTALL_RPATH "${RKLLM_RUNTIME_DIR}/aarch64"
    BUILD_WITH_INSTALL_RPATH TRUE
)

install(TARGETS rkllm_server RUNTIME DESTINATION bin)
