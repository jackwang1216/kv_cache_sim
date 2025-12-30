cmake_minimum_required(VERSION 3.14)
project(kv_sum LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_executable(
    kv_sum
    src/sim_main.cpp
    src/simulator.cpp
    src/io_config.cpp
    src/io_trace.cpp
    src/io_output.cpp
)

target_include_directories(kv_sim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(kv_sim PRIVATE -Wall -Wextra -Wpedantic)