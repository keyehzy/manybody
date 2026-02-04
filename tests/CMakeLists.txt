file(GLOB_RECURSE tests_sources CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_executable(tests ${tests_sources})
target_link_libraries(tests PRIVATE manybody rapidcheck_catch)
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/tests)

add_test(NAME tests COMMAND tests)
