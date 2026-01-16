file(GLOB_RECURSE SOURCES "*.cpp")
list(FILTER SOURCES EXCLUDE REGEX "main\\.cpp$")

add_library(manybody STATIC ${SOURCES})
target_include_directories(manybody PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third-party)
target_link_libraries(manybody PUBLIC
    ${ARMADILLO_LIBRARIES}
    OpenMP::OpenMP_CXX)
target_include_directories(manybody PUBLIC ${ARMADILLO_INCLUDE_DIRS})
