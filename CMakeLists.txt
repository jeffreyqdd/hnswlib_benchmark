cmake_minimum_required(VERSION 3.0...3.26)
project(hnsw-benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FAISS_ENABLE_GPU OFF)       # Disable GPU support
set(BUILD_TESTING OFF)          # Optionally disable Faiss unit tests to speed up the build
set(FAISS_ENABLE_PYTHON OFF) 

set(faiss_SOURCE_DIR "${CMAKE_SOURCE_DIR}/dependencies/faiss")
set(hnswlib_SOURCE_DIR "${CMAKE_SOURCE_DIR}/dependencies/hnswlib")

add_subdirectory(${faiss_SOURCE_DIR} ${faiss_BINARY_DIR})
add_subdirectory(${hnswlib_SOURCE_DIR} ${hnswlib_BINARY_DIR})


include_directories("."j)
include_directories("./dependencies/faiss")
include_directories("./dependencies/hnswlib")


add_executable(bench_st_sq src/bench_st_single_query.cpp)
target_link_libraries(bench_st_sq PRIVATE hnswlib faiss)