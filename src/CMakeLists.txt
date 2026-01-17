file(GLOB_RECURSE SOURCES "*.cpp")
list(FILTER SOURCES EXCLUDE REGEX "main\\.cpp$")

add_library(manybody STATIC ${SOURCES})

target_include_directories(manybody PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third-party)

target_compile_definitions(manybody PUBLIC
    ARMA_DONT_USE_WRAPPER
    ARMA_USE_BLAS
    ARMA_USE_LAPACK
    ARMA_USE_ARPACK
    ARMA_USE_SUPERLU
    ARMA_USE_SUPERLU_HEADERS
    ARMA_SUPERLU_INCLUDE_DIR=${SuperLU_INCLUDE_DIRS}/
    ARMA_USE_FFTW3
    ARMA_USE_OPENMP
    ARMA_OPENMP_THREADS=1
    OPENBLAS_NUM_THREADS=1)

if(TARGET Armadillo::Armadillo)
    target_link_libraries(manybody PUBLIC Armadillo::Armadillo)
else()
    target_include_directories(manybody PUBLIC ${ARMADILLO_INCLUDE_DIRS})
    target_link_libraries(manybody PUBLIC ${ARMADILLO_LIBRARIES})
endif()

if(TARGET OpenBLAS::OpenBLAS)
    target_link_libraries(manybody PUBLIC OpenBLAS::OpenBLAS)
else()
    target_include_directories(manybody PUBLIC ${OpenBLAS_INCLUDE_DIRS})
    target_link_libraries(manybody PUBLIC ${OpenBLAS_LIBRARIES})
endif()

target_link_libraries(manybody PUBLIC
     PkgConfig::FFTW3
     PkgConfig::ARPACK
     PkgConfig::SuperLU
     OpenMP::OpenMP_CXX)
