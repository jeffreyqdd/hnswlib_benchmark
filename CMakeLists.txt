cmake_minimum_required(VERSION 3.0...3.26)

project(hnsw-benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find_library(HNSWLIB hnswlib PATHS /usr/local/lib REQUIRED)
include_directories(/usr/local/include)
include_directories(.)

add_executable(bench_st src/bench_st.cpp)

# add_executable(MY_EXEC_ALT src/main_alt.cpp)