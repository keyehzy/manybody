#pragma once

#include <exception>
#include <iostream>
#include <string>
#include <vector>

struct TestCase {
  const char* name;
  void (*fn)();
};

inline std::vector<TestCase>& test_registry() {
  static std::vector<TestCase> tests;
  return tests;
}

inline int& test_failures() {
  static int failures = 0;
  return failures;
}

struct TestRegistrar {
  TestRegistrar(const char* name, void (*fn)()) { test_registry().push_back({name, fn}); }
};

#define TEST(name)                                    \
  static void name();                                 \
  static TestRegistrar name##_registrar(#name, name); \
  static void name()

#define EXPECT_TRUE(cond)                                                                      \
  do {                                                                                         \
    if (!(cond)) {                                                                             \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " expected true for " #cond "\n"; \
      test_failures()++;                                                                       \
    }                                                                                          \
  } while (0)

#define EXPECT_EQ(lhs, rhs)                                                                      \
  do {                                                                                           \
    const auto& lhs_val = (lhs);                                                                 \
    const auto& rhs_val = (rhs);                                                                 \
    if (!(lhs_val == rhs_val)) {                                                                 \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " expected " #lhs " == " #rhs "\n"; \
      test_failures()++;                                                                         \
    }                                                                                            \
  } while (0)

inline int run_all_tests() {
  constexpr const char* kGreen = "\033[32m";
  constexpr const char* kRed = "\033[31m";
  constexpr const char* kReset = "\033[0m";
  int failed_tests = 0;
  for (const auto& test : test_registry()) {
    const int failures_before = test_failures();
    try {
      test.fn();
    } catch (const std::exception& ex) {
      std::cerr << kRed << "FAIL " << kReset << test.name << " threw: " << ex.what() << "\n";
      test_failures()++;
    } catch (...) {
      std::cerr << kRed << "FAIL " << kReset << test.name << " threw unknown exception\n";
      test_failures()++;
    }

    if (test_failures() > failures_before) {
      failed_tests++;
    } else {
      std::cerr << kGreen << "PASSED " << kReset << test.name << "\n";
    }
  }

  const int total = static_cast<int>(test_registry().size());
  if (failed_tests == 0) {
    std::cout << "OK " << total << " tests\n";
    return 0;
  }

  std::cout << "FAILED " << failed_tests << " of " << total << " tests\n";
  return 1;
}
