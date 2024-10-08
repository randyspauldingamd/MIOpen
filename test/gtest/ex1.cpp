#include <gtest/gtest.h>
// TODO TRJS delete file
struct paramType { std::string value; };

class MyFixture : public testing::TestWithParam<paramType> {};
class FixtureA : public MyFixture {};
class FixtureB : public MyFixture {};

TEST_P(FixtureA, TestNameA0) { auto& myParam = GetParam(); EXPECT_GT(myParam.value.size(), 0ULL); }
TEST_P(FixtureB, TestNameA0) { auto& myParam = GetParam(); EXPECT_GT(myParam.value.size(), 0ULL); }

INSTANTIATE_TEST_SUITE_P(PIN0, FixtureA, testing::Values(paramType{"v00"}, paramType{"v01"}));
INSTANTIATE_TEST_SUITE_P(PIN1, FixtureA, testing::Values(paramType{"v10"}, paramType{"v11"}, paramType{"v12"}));
INSTANTIATE_TEST_SUITE_P(PIN2, FixtureB, testing::Values(paramType{"v00"}, paramType{"v11"}));

