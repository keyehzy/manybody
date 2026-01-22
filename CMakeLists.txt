cmake_minimum_required(VERSION 3.16)
project(manybody LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Armadillo REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenBLAS REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED IMPORTED_TARGET fftw3)
pkg_check_modules(ARPACK REQUIRED IMPORTED_TARGET arpack)
pkg_check_modules(SuperLU REQUIRED IMPORTED_TARGET superlu)

find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER   ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

option(MANYBODY_BUILD_TESTS "Build the test suite" OFF)

add_subdirectory(src)
add_subdirectory(examples)

if (MANYBODY_BUILD_TESTS)
    add_subdirectory(third-party)
    enable_testing()
    add_subdirectory(tests)
endif()
