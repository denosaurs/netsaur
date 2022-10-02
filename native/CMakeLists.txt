cmake_minimum_required(VERSION 3.0.0)
project(netsaur VERSION 0.1.0)

set (CMAKE_BUILD_TYPE Release)

set(
  SOURCES
    src/matrix.c
    src/util.c
    src/activation.c
    src/cost.c
    src/network.c
    src/layers/common.c
    src/layers/dense.c
)

add_executable(netsaur_test src/main.c ${SOURCES})
add_library(netsaur SHARED ${SOURCES})

target_link_libraries(netsaur m)
target_link_libraries(netsaur_test m)

target_link_libraries(netsaur gomp)
target_link_libraries(netsaur_test gomp)

set (COMPILE_OPTIONS -Ofast -Wno-unused-result -march=native -fopenmp)

target_compile_options(netsaur PRIVATE ${COMPILE_OPTIONS})
target_compile_options(netsaur_test PRIVATE ${COMPILE_OPTIONS})

target_include_directories(netsaur_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(netsaur PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
