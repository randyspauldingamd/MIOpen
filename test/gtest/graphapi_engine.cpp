/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <miopen/graphapi/rng.hpp>

#include <tuple>

#include <gtest/gtest.h>

#include "graphapi_gtest_common.hpp"

namespace {

using miopen::graphapi::ValidatedValue;
using miopen::graphapi::ValidatedVector;
using DescriptorTuple = std::tuple<bool,
                                   miopenRngDistribution_t,
                                   double,
                                   ValidatedValue<double>,
                                   double,
                                   double,
                                   ValidatedValue<double>>;

// using miopen::graphapi::Rng;
// using miopen::graphapi::RngBuilder;

} // namespace

class GraphApiEngineBuilder : public testing::TestWithParam<DescriptorTuple>
{
protected:
    bool mAttrsValid;
    // miopenRngDistribution_t distribution;
    // double mNormalMean;
    // ValidatedValue<double> mNormalStdev;
    // double mUniformMin;
    // double mUniformMax;
    // ValidatedValue<double> mBernoulliProb;

    void SetUp() override
    {
        std::tie(mAttrsValid,
                 distribution,
                 mNormalMean,
                 mNormalStdev,
                 mUniformMin,
                 mUniformMax,
                 mBernoulliProb) = GetParam();
    }
};

TEST_P(GraphApiEngineBuilder, ValidateAttributes)
{
    if(mAttrsValid && mNormalStdev.valid && mBernoulliProb.valid)
    {
        EXPECT_NO_THROW({
            EngineBuilder()
                .setDistribution(distribution)
                .setNormalMean(mNormalMean)
                .setNormalStdev(mNormalStdev.value)
                .setUniformMin(mUniformMin)
                .setUniformMax(mUniformMax)
                .setBernoulliProb(mBernoulliProb.value)
                .build();
        }) << "Builder failed on valid attributes";
    }
    else
    {
        EXPECT_ANY_THROW({
            EngineBuilder()
                .setDistribution(distribution)
                .setNormalMean(mNormalMean)
                .setNormalStdev(mNormalStdev.value)
                .setUniformMin(mUniformMin)
                .setUniformMax(mUniformMax)
                .setBernoulliProb(mBernoulliProb.value)
                .build();
        }) << "Buider failed to detect invalid attributes";
    }
    if(mNormalStdev.valid)
    {
        EXPECT_NO_THROW({ EngineBuilder().setNormalStdev(mNormalStdev.value); })
            << "EngineBuilder::setNormalStdev(double) failed on valid attribute";
    }
    else
    {
        EXPECT_ANY_THROW({ EngineBuilder().setNormalStdev(mNormalStdev.value); })
            << "EngineBuilder::setNormalStdev(double) failed on invalid attribute";
    }
    if(mBernoulliProb.valid)
    {
        EXPECT_NO_THROW({ EngineBuilder().setBernoulliProb(mBernoulliProb.value); })
            << "EngineBuilder::setBernoulliProb(double) failed on valid attribute";
    }
    else
    {
        EXPECT_ANY_THROW({ EngineBuilder().setBernoulliProb(mBernoulliProb.value); })
            << "EngineBuilder::setBernoulliProb(double) failed on invalid attribute";
    }
}

namespace {

using miopen::graphapi::GTestDescriptorAttribute;
using miopen::graphapi::GTestDescriptorSingleValueAttribute;
using miopen::graphapi::GTestGraphApiExecute;

} // namespace

class GraphApiEngine : public testing::TestWithParam<DescriptorTuple>
{
private:
    // Pointers to these are used in mExecute object below
    // GTestDescriptorSingleValueAttribute<miopenRngDistribution_t, char> mDistribution;
    // GTestDescriptorSingleValueAttribute<double, char> mNormalMean;

protected:
    GTestGraphApiExecute<GTestDescriptorAttribute*> mExecute;

    void SetUp() override
    {
        auto [valid, distribution, normalMean, normalStdev, uniformMin, uniformMax, bernoulliProb] =
            GetParam();

    }
};

TEST_P(GraphApiEngine, CFunctions) { mExecute(); }

static auto validAttributesNormal =
    testing::Combine(testing::Values(true),
                     testing::Values(MIOPEN_RNG_DISTRIBUTION_NORMAL),
                     testing::Values(0.0),
                     testing::Values(ValidatedValue<double>{true, 0.5}),
                     testing::Values(0.0, 1.0),
                     testing::Values(0.0, 1.0),
                     testing::Values(ValidatedValue<double>{true, 0.5},
                                     ValidatedValue<double>{false, -0.5},
                                     ValidatedValue<double>{false, 1.5}));

static auto validAttributesUniform = testing::Combine(
    testing::Values(true),
    testing::Values(MIOPEN_RNG_DISTRIBUTION_UNIFORM),
    testing::Values(0.0),
    testing::Values(ValidatedValue<double>{true, 0.5}, ValidatedValue<double>{false, -0.5}),
    testing::Values(0.0),
    testing::Values(1.0),
    testing::Values(ValidatedValue<double>{true, 0.5},
                    ValidatedValue<double>{false, -0.5},
                    ValidatedValue<double>{false, 1.5}));

static auto validAttributesBernoulli = testing::Combine(
    testing::Values(true),
    testing::Values(MIOPEN_RNG_DISTRIBUTION_BERNOULLI),
    testing::Values(0.0),
    testing::Values(ValidatedValue<double>{true, 0.5}, ValidatedValue<double>{false, -0.5}),
    testing::Values(0.0, 1.0),
    testing::Values(0.0, 1.0),
    testing::Values(ValidatedValue<double>{true, 0.5}));

static auto invalidAttributesNormal =
    testing::Combine(testing::Values(false),
                     testing::Values(MIOPEN_RNG_DISTRIBUTION_NORMAL),
                     testing::Values(0.0),
                     testing::Values(ValidatedValue<double>{false, -0.5}),
                     testing::Values(0.0, 1.0),
                     testing::Values(0.0, 1.0),
                     testing::Values(ValidatedValue<double>{true, 0.5},
                                     ValidatedValue<double>{false, -0.5},
                                     ValidatedValue<double>{false, 1.5}));

static auto invalidAttributesUniform = testing::Combine(
    testing::Values(false),
    testing::Values(MIOPEN_RNG_DISTRIBUTION_UNIFORM),
    testing::Values(0.0),
    testing::Values(ValidatedValue<double>{true, 0.5}, ValidatedValue<double>{false, -0.5}),
    testing::Values(1.0),
    testing::Values(0.0),
    testing::Values(ValidatedValue<double>{true, 0.5},
                    ValidatedValue<double>{false, -0.5},
                    ValidatedValue<double>{false, 1.5}));

static auto invalidAttributesBernoulli = testing::Combine(
    testing::Values(false),
    testing::Values(MIOPEN_RNG_DISTRIBUTION_BERNOULLI),
    testing::Values(0.0),
    testing::Values(ValidatedValue<double>{true, 0.5}, ValidatedValue<double>{false, -0.5}),
    testing::Values(0.0, 1.0),
    testing::Values(0.0, 1.0),
    testing::Values(ValidatedValue<double>{false, -0.5}, ValidatedValue<double>{false, 1.5}));

INSTANTIATE_TEST_SUITE_P(ValidAttributesNormal, GraphApiRngBuilder, validAttributesNormal);
INSTANTIATE_TEST_SUITE_P(ValidAttributesUniform, GraphApiRngBuilder, validAttributesUniform);
INSTANTIATE_TEST_SUITE_P(ValidAttributesBernoulli, GraphApiRngBuilder, validAttributesBernoulli);

INSTANTIATE_TEST_SUITE_P(InvalidAttributesNormal, GraphApiRngBuilder, invalidAttributesNormal);
INSTANTIATE_TEST_SUITE_P(InvalidAttributesUniform, GraphApiRngBuilder, invalidAttributesUniform);
INSTANTIATE_TEST_SUITE_P(InvalidAttributesBernoulli,
                         GraphApiRngBuilder,
                         invalidAttributesBernoulli);

INSTANTIATE_TEST_SUITE_P(ValidAttributesNormal, GraphApiRng, validAttributesNormal);
INSTANTIATE_TEST_SUITE_P(ValidAttributesUniform, GraphApiRng, validAttributesUniform);
INSTANTIATE_TEST_SUITE_P(ValidAttributesBernoulli, GraphApiRng, validAttributesBernoulli);

INSTANTIATE_TEST_SUITE_P(InvalidAttributesNormal, GraphApiRng, invalidAttributesNormal);
INSTANTIATE_TEST_SUITE_P(InvalidAttributesUniform, GraphApiRng, invalidAttributesUniform);
INSTANTIATE_TEST_SUITE_P(InvalidAttributesBernoulli, GraphApiRng, invalidAttributesBernoulli);
