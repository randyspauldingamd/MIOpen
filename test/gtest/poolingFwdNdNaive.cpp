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

#ifndef POOLING_GTEST_BUILD

#include <gtest/gtest.h>
#include <miopen/env.hpp>
#include "get_handle.hpp"
#include "test_env.hpp"

#include "pooling2d.hpp"

#include "tensor_holder.hpp"
#include "miopen/tensor_layout.hpp"

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLAGS_ARGS)

namespace env = miopen::env;

namespace {

template <typename T>
struct layout_data
{
    static std::vector<int> get_layout_lengths(int n, int c, std::vector<int>& dims)
    {
        auto ret = std::vector<int>{n, c};
        ret.insert(ret.end(), dims.cbegin(), dims.cend());

        return ret;
    }

    static std::vector<int>
    get_strides(std::vector<int>& lens, int dims, miopenTensorLayout_t tensor_layout)
    {
        std::vector<int> strides;
        std::string layout_default = miopen::tensor_layout_get_default(dims + 2);
        std::string layout_string  = miopen::TensorDescriptor::GetLayoutStr(tensor_layout);

        miopen::tensor_layout_to_strides(lens, layout_default, layout_string, strides);

        constexpr int min_stride_multiplier = 1;
        constexpr int max_stride_multiplier = 4;

        auto c = prng::gen_A_to_B(min_stride_multiplier, max_stride_multiplier);
        for(auto& v : strides)
        {
            // cppcheck-suppress useStlAlgorithm
            v = v * c;
        }

        return strides;
    }

    static miopenTensorDescriptor_t init_tensor_descriptor(miopenDataType_t type,
                                                           const std::vector<int>& lens,
                                                           const std::vector<int>& strides)
    {
        miopenTensorDescriptor_t desc;

        EXPECT_TRUE(miopenCreateTensorDescriptor(&desc) == miopenStatusSuccess);
        EXPECT_TRUE(
            miopenSetTensorDescriptor(desc, type, lens.size(), lens.data(), strides.data()) ==
            miopenStatusSuccess);

        return desc;
    }

    layout_data(int _n, std::vector<int> _dims, int _c, miopenTensorLayout_t _tensor_layout)
    {
        auto lens    = get_layout_lengths(_n, _c, _dims);
        auto strides = get_strides(lens, _dims.size(), _tensor_layout);
        descriptor   = miopen::TensorDescriptor{miopen_type<T>{}, lens, strides};
        host         = tensor<T>{lens, strides}.generate(gen_value<T>);
    }

    ~layout_data() {}

    void read_gpu_data(miopen::Handle& handle, const miopen::Allocator::ManageDataPtr& ddata)
    {
        check      = tensor<T>{descriptor.GetLengths(), descriptor.GetStrides()};
        check.data = handle.Read<T>(ddata, check.data.size());
    }

    tensor<T> check{};
    tensor<T> host;
    miopen::TensorDescriptor descriptor;
};

}

class PoolingFwd : public testing::TestWithParam<std::vector<std::string>> {};
class PoolingFwdFloat : public PoolingFwd {};
class PoolingFwdHalf : public PoolingFwd {};

void Run2dDriver(miopenDataType_t prec);

namespace {

static bool SkipTest(void) { return env::disabled(MIOPEN_TEST_ALL); }

void GetArgs(const std::string& param, std::vector<std::string>& tokens)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

bool IsTestSupportedForDevice(const miopen::Handle& handle) { return true; }

std::vector<std::string> GetTestCases(const std::string precision)
{
    const auto& flag_arg = env::value(MIOPEN_TEST_FLAGS_ARGS);

    const std::vector<std::string> test_cases = {
        // clang-format off
    {"test_pooling2d " + precision + " --all --dataset 1 --limit 0 " + flag_arg}
        // clang-format on
    };

    return test_cases;
}
} // namespace pooling_tests
// using namespace pooling_tests;

TEST_P(PoolingFwdFloat, NNT)    // NDNaiveTranspose
{
    const auto& handle = get_handle();
    if(!IsTestSupportedForDevice(handle))   std::cout << "WOULD SKIP BECAUSE NOT SUPPORTED!" << std::endl;
    if(SkipTest())                          std::cout << "WOULD SKIP BECAUSE SKIPTEST!" << std::endl;
    // if(!IsTestRunWith("--float"))           std::cout << "WOULD SKIP BECAUSE NOT FLOAT!" << std::endl;
        // Run2dDriver(miopenFloat);   return; // TEMPCODE RJS
    //  && IsTestRunWith("--float")
    if(IsTestSupportedForDevice(handle) && !SkipTest())
    {
        Run2dDriver(miopenFloat);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(PoolingFwdHalf, NNT)
{
    const auto& handle = get_handle();
    if(!IsTestSupportedForDevice(handle))   std::cout << "WOULD SKIP BECAUSE NOT SUPPORTED!" << std::endl;
    if(SkipTest())                          std::cout << "WOULD SKIP BECAUSE SKIPTEST!" << std::endl;
    // if(!IsTestRunWith("--half"))           std::cout << "WOULD SKIP BECAUSE NOT HALF!" << std::endl;

    if(IsTestSupportedForDevice(handle) && !SkipTest()) //  && IsTestRunWith("--half") TEMPCODE RJS
    {
        Run2dDriver(miopenHalf);
    }
    else
    {
        GTEST_SKIP();
    }
};

void Run2dDriver(miopenDataType_t prec)
{
    auto cases = GetTestCases("--float");
       std::cerr << " Cases: " << cases.size() << std::endl;
    for(const auto& test_value : cases)
    {
        std::cerr << "      : " << test_value << std::endl;    // TEMPCODE RJS
    }
 
    std::vector<std::string> params;
    switch(prec)
    {
    case miopenFloat: params = PoolingFwdFloat_NNT_Test::GetParam(); break;
    case miopenHalf: params = PoolingFwdHalf_NNT_Test::GetParam(); break;
    case miopenBFloat16:
    case miopenInt8:
    case miopenInt32:
    case miopenDouble:
    case miopenFloat8:
    case miopenBFloat8:
    case miopenInt64:
        FAIL()
            << "miopenBFloat16, miopenInt8, miopenInt32, miopenDouble, miopenFloat8, miopenBFloat8, miopenInt64 "
               "data type not supported by "
               "poolingFwdNdNaive test";

    default: params = PoolingFwdFloat_NNT_Test::GetParam();
    }

    std::cerr << "Params: " << params.size() << std::endl;
    for(const auto& test_value : params)
    {
        std::cerr << "      : " << test_value << std::endl;    // TEMPCODE RJS
    }
    for(const auto& test_value : params)
    {
        std::cerr << "Testing: " << test_value << std::endl;    // TEMPCODE RJS
        std::vector<std::string> tokens;
        GetArgs(test_value, tokens);
        std::vector<const char*> ptrs;

        std::transform(tokens.begin(), tokens.end(), std::back_inserter(ptrs), [](const auto& str) {
            return str.data();
        });

        testing::internal::CaptureStderr();
        test_drive<pooling2d_driver>(ptrs.size(), ptrs.data());
        auto capture = testing::internal::GetCapturedStderr();
        std::cout << capture;
    }
}

INSTANTIATE_TEST_SUITE_P(Float, PoolingFwdFloat, testing::Values(GetTestCases("--float")));

INSTANTIATE_TEST_SUITE_P(Half, PoolingFwdHalf, testing::Values(GetTestCases("--half")));

#endif