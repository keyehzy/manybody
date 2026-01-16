cmake_minimum_required(VERSION 3.16)
project(manybody LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Armadillo REQUIRED)
find_package(OpenMP REQUIRED)

add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(tests)
