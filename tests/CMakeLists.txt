add_executable(tests main.cpp)
target_link_libraries(tests PRIVATE manybody)
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/tests)

enable_testing()
add_test(NAME tests COMMAND tests)
