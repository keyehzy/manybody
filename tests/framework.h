#pragma once

#include "catch.hpp"

#define TEST(name) TEST_CASE(#name)
#define EXPECT_TRUE(cond) CHECK(cond)
#define EXPECT_EQ(lhs, rhs) CHECK((lhs) == (rhs))
