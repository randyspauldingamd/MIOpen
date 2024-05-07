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
#include "graphapi_opgraph_common.hpp"
#include <miopen/graphapi/engine.hpp>
#include <miopen/graphapi/opgraph.hpp>

using namespace miopen::graphapi;
using namespace graphapi_opgraph_tests;

namespace {

// using miopen::graphapi::ValidatedValue;
// using miopen::graphapi::ValidatedVector;
using DescriptorTuple = std::tuple<
                            bool,
                            int64_t,
                            int32_t
                        >;

// using miopen::graphapi::Engine;
// using miopen::graphapi::EngineBuilder;

} // namespace

class GraphApiEngineBuilder : public testing::TestWithParam<DescriptorTuple>
{
protected:
    bool mAttrsValid;
    int64_t mGlobalIndex;
    int32_t mSmCount;

    void SetUp() override
    {
        std::tie(mAttrsValid,
                 mGlobalIndex,
                 mSmCount) = GetParam();

    }
};

TEST_P(GraphApiEngineBuilder, ValidateAttributes)
{
    auto dg1 = makeDiamondGraph();
    EXPECT_EQ(dg1->graph().getEngines().size(), 4)
        << "Tests will fail because makeDiamondGraph has been modified.";

    if(mAttrsValid)
    {
        EXPECT_NO_THROW({
            EngineBuilder()
                .setOpGraph(&dg1->graph())
                .setGlobalIndex(mGlobalIndex)
                .setSmCount(mSmCount)
                .build()
            ;
        }) << "Engine Builder failed on valid attributes";
    }
    else
    {
        EXPECT_ANY_THROW({
            EngineBuilder()
                .setOpGraph(&dg1->graph())
                .setGlobalIndex(mGlobalIndex)
                .setSmCount(mSmCount)
                .build()
            ;
        }) << "Engine Builder failed to detect invalid attributes";
    }
}

// namespace {

// using miopen::graphapi::GTestDescriptorAttribute;
// using miopen::graphapi::GTestDescriptorSingleValueAttribute;
// using miopen::graphapi::GTestGraphApiExecute;

// } // namespace

class GraphApiEngine : public testing::TestWithParam<DescriptorTuple>
{
private:
    // Pointers to these are used in mExecute object below
    // TODO: Need guidance: we should mock Solution, but don't know how because the header uses #pragma once.
    // GTestDescriptorSingleValueAttribute<Solution, char> mSolution; TEMPCODE: ignore Solution
    GTestDescriptorSingleValueAttribute<BackendOperationGraphDescriptor, char> mOpGraphDescr;
    GTestDescriptorVectorAttribute<BackendEngineDescriptor, char> mOps;
    GTestDescriptorSingleValueAttribute<int64_t, char> mGlobalIndex;
    GTestDescriptorSingleValueAttribute<int32_t, char> mSmCount;

protected:
    GTestGraphApiExecute<GTestDescriptorAttribute*> mExecute;

    void SetUp() override
    {
        auto [valid, globalIndex, smCount] =
            GetParam();

        auto dg1 = graphapi_opgraph_tests::makeDiamondGraph();
        EXPECT_EQ(dg1->graph().getEngines().size(), 4)
            << "Tests will fail because makeDiamondGraph has been modified.";

        BackendEngineDescriptor opDescr;
        // RJS note: I couldn't get this part to work yet due to various deleted ctors etc.
        // mOps = {
        //     true,
        //     "MIOPEN_ATTR_OPERATIONGRAPH_OPS",
        //     MIOPEN_ATTR_OPERATIONGRAPH_OPS,
        //     MIOPEN_TYPE_BACKEND_DESCRIPTOR,
        //     MIOPEN_TYPE_CHAR,
        //     5,
        //     {opDescr, opDescr, opDescr, opDescr}}
        // ;

        // auto [isCorrect,
        //         textName,
        //         name,
        //         type,
        //         count,
        //         data,
        //         invalidType,
        //         invalidTypeData,
        //         invalidCount,
        //         invalidCountData,
        //         readBuffer] = mOps.getTestCase();

        BackendOperationGraphDescriptor opGraphDescr;
        // opGraphDescr.setAttribute(
        //     MIOPEN_ATTR_OPERATIONGRAPH_OPS,
        //     MIOPEN_TYPE_BACKEND_DESCRIPTOR,
        //     4,
        //     data
        // );

        mGlobalIndex = {
            true,
            "MIOPEN_ATTR_ENGINE_GLOBAL_INDEX",
            MIOPEN_ATTR_ENGINE_GLOBAL_INDEX,
            MIOPEN_TYPE_INT64,
            MIOPEN_TYPE_CHAR,
            2,
            globalIndex
        };

        mSmCount = {
            true,
            "MIOPEN_ATTR_ENGINE_SM_COUNT_TARGET",
            MIOPEN_ATTR_ENGINE_SM_COUNT_TARGET,
            MIOPEN_TYPE_INT32,
            MIOPEN_TYPE_CHAR,
            2,
            smCount
        };

        mExecute.descriptor.textName   = "MIOPEN_BACKEND_ENGINE_DESCRIPTOR";
        mExecute.descriptor.type       = MIOPEN_BACKEND_ENGINE_DESCRIPTOR;
        mExecute.descriptor.attrsValid = valid;

        mExecute.descriptor.attributes = {&mOpGraphDescr,
                                          &mGlobalIndex,
                                          &mSmCount};
    }
};

TEST_P(GraphApiEngine, CFunctions) { mExecute(); }

static auto validAttributes =
    testing::Combine(
        testing::Values(true),
        testing::Values(0, 1, 2, 3),
        testing::Values(1, 2, 4, 8, 16)
    );

static auto invalidEngine =
    testing::Combine(
        testing::Values(false),
        testing::Values(-1, 4),
        testing::Values(1, 2, 4, 8, 16)
    );

static auto invalidSmCount =
    testing::Combine(
        testing::Values(false),
        testing::Values(0, 1, 2, 3),
        testing::Values(-1, 0)
    );

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GraphApiEngineBuilder);
// INSTANTIATE_TEST_SUITE_P(ValidAttributes, GraphApiEngineBuilder, validAttributes);

// INSTANTIATE_TEST_SUITE_P(InvalidEngine, GraphApiEngineBuilder, invalidEngine);

// INSTANTIATE_TEST_SUITE_P(InvalidSmCount, GraphApiEngineBuilder, invalidSmCount);

INSTANTIATE_TEST_SUITE_P(ValidAttributes, GraphApiEngine, validAttributes);

INSTANTIATE_TEST_SUITE_P(InvalidEngine, GraphApiEngine, invalidEngine);

INSTANTIATE_TEST_SUITE_P(InvalidSmCount, GraphApiEngine, invalidSmCount);

