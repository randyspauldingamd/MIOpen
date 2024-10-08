/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#pragma once

#include <miopen/miopen.h>
#include <miopen/names.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

#include <string>

namespace miopen {

struct ProblemDescriptionLayoutBase : ProblemDescriptionBase
{
    ProblemDescriptionLayoutBase()                              = default;
    ProblemDescriptionLayoutBase(const ProblemDescriptionLayoutBase&) = default;
    ProblemDescriptionLayoutBase(const TensorDescriptor& in_, // x for Forward, y for Backward*
                                const TensorDescriptor& out_ // y for Forward, x for Backward*
                       )
    : ProblemDescriptionBase(),
      in(in_),
      out(out_),
          in_layout(ComputeInLayout()),
          out_layout(ComputeOutLayout())
    {}
    virtual ~ProblemDescriptionLayoutBase()                     = default;

    ProblemDescriptionLayoutBase& operator=(const ProblemDescriptionLayoutBase&) = default;

    [[nodiscard]] virtual NetworkConfig MakeNetworkConfig() const = 0;

protected:
    TensorDescriptor in;
    TensorDescriptor out;
    std::string in_layout;
    std::string out_layout;

    std::string ComputeInLayout() const
    {
        return in.GetLayout(in.GetLayout_str());
    }

    std::string ComputeOutLayout() const
    {
        return out.GetLayout(out.GetLayout_str());
    }
};

struct ProblemDescriptionWeightsBase : ProblemDescriptionLayoutBase
{
    ProblemDescriptionWeightsBase()                              = default;
    ProblemDescriptionWeightsBase(const ProblemDescriptionWeightsBase&) = default;
    ProblemDescriptionWeightsBase(const TensorDescriptor& in_, // x for Forward, y for Backward*
                       const TensorDescriptor& weights_,
                       const TensorDescriptor& out_ // y for Forward, x for Backward*
                       )
        : ProblemDescriptionLayoutBase(in_, out_),
          weights(weights_),
          weights_layout(ComputeWeightsLayout())
    {}
    virtual ~ProblemDescriptionWeightsBase()                     = default;

    ProblemDescriptionWeightsBase& operator=(const ProblemDescriptionWeightsBase&) = default;

    [[nodiscard]] virtual NetworkConfig MakeNetworkConfig() const = 0;

protected:
    TensorDescriptor weights;
    std::string weights_layout;

    std::string ComputeWeightsLayout() const
    {
        return weights.GetLayout(weights.GetLayout_str());
    }
};

} // namespace miopen
