cmake_minimum_required(VERSION 3.16)

# Disable RapidCheck's bundled Catch2 (ext/catch submodule not present)
set(RC_ENABLE_CATCH OFF CACHE BOOL "" FORCE)

add_subdirectory(rapidcheck)

# Manually add RapidCheck's Catch2 integration header library
add_subdirectory(rapidcheck/extras/catch)
