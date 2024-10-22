/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#include <miopen/pooling/solvers.hpp>

#include <miopen/pooling/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/pooling.hpp>
#include <miopen/kernel_build_params.hpp>

#include "ck/library/tensor_operation_instance/gpu/pool3d_fwd.hpp"

namespace miopen {

namespace solver {

namespace pooling {

namespace {

// convert 2D problems to 3D
constexpr ck::index_t InOutRank  = 5;
constexpr ck::index_t WindowRank = 3;

using F8   = ck::half_t;     // TRJS does CK support float8?
using F16  = ck::half_t;
using F32  = float;
using F64  = double;
using BF16 = ushort;

struct CKArgsPoolingFwd
{
    template <typename T>
    static void MakeCkVec(std::vector<ck::index_t>& ck, std::vector<T> in)
    {
        ck.resize(in.size());
        std::copy(in.begin(), in.end(), ck.begin());
    }

    CKArgsPoolingFwd(const miopen::pooling::ProblemDescription& problem)
    {
        MakeCkVec(in_length, problem.GetXDesc().GetLengths());
        MakeCkVec(out_length, problem.GetYDesc().GetLengths());
        MakeCkVec(window_spatial_lengths, problem.GetPooling().GetLengths());
        MakeCkVec(window_strides, problem.GetPooling().GetStrides());
        MakeCkVec(input_pads, problem.GetPooling().GetPads());
        MakeCkVec(in_tensor_stride, problem.GetXDesc().GetStrides());
        MakeCkVec(out_tensor_stride, problem.GetYDesc().GetStrides());

        if(in_length.size() < 5)
        {
            TransformPool2dparamToPool3d(in_length,
                                        window_spatial_lengths,
                                        out_length,
                                        in_tensor_stride,
                                        out_tensor_stride,
                                        window_strides,
                                        input_pads);
        }
        std::cout << "args:" << std::endl;  // TRJS
        std::cout << "  input_lengths   :"; for(auto v : in_length) std::cout << std::setw(10) << v; std::cout << std::endl;
        std::cout << "  window_lengths  :"; for(auto v : window_spatial_lengths) std::cout << std::setw(10) << v; std::cout << std::endl;
        std::cout << "  output_lengths  :"; for(auto v : out_length) std::cout << std::setw(10) << v; std::cout << std::endl;
        std::cout << "  input_stride    :"; for(auto v : in_tensor_stride) std::cout << std::setw(10) << v; std::cout << std::endl;
        std::cout << "  output_stride   :"; for(auto v : out_tensor_stride) std::cout << std::setw(10) << v; std::cout << std::endl;
        std::cout << "  window_strides  :"; for(auto v : window_strides) std::cout << std::setw(10) << v; std::cout << std::endl;
        std::cout << "  input_pads      :"; for(auto v : input_pads) std::cout << std::setw(10) << v; std::cout << std::endl;
    }

    void TransformPool2dparamToPool3d(std::vector<ck::index_t>& input_lengths,
                                    std::vector<ck::index_t>& window_lengths,
                                    std::vector<ck::index_t>& output_lengths,
                                    std::vector<ck::index_t>& input_stride,
                                    std::vector<ck::index_t>& output_stride,
                                    std::vector<ck::index_t>& window_strides_,
                                    std::vector<ck::index_t>& input_pads_)
    {
        // NCHW to NCDHW
        input_lengths.insert(input_lengths.begin() + 2, 1);
        output_lengths.insert(output_lengths.begin() + 2, 1);
        input_stride.insert(input_stride.begin() + 2, 0);
        output_stride.insert(output_stride.begin() + 2, 0);

        // YX to ZYX
        window_lengths.insert(window_lengths.begin(), 1);
        window_strides_.insert(window_strides_.begin(), 0);
        input_pads_.insert(input_pads_.begin(), 0);
    }

    // Pool API only support the order of NCDHW
    std::vector<ck::index_t> in_length;
    std::vector<ck::index_t> window_spatial_lengths;
    std::vector<ck::index_t> out_length;
    std::vector<ck::index_t> window_strides;
    std::vector<ck::index_t> window_dilations{1, 1, 1};
    std::vector<ck::index_t> input_pads;
    std::vector<ck::index_t> pooling_dims{2, 3, 4};

    // tensor layout = NDHWC
    std::vector<ck::index_t> in_tensor_stride;
    std::vector<ck::index_t> out_tensor_stride;
};

} // namespace

bool PoolingForwardCkNd::IsApplicable(const ExecutionContext& context,
                                    const miopen::pooling::ProblemDescription& problem) const
{
    // TRJS: does CK do miopenPoolingAverage or miopenPoolingAverageInclusive?
    // TRJS: which workspace index mask mode does CK use?
    // TRJS: does CK produce NCDHW output? (doubt it)
    // TRJS: which types does CK support?
    // TRJS: which index types does CK support?
    const auto x_type = problem.GetXDesc().GetType();
    const auto y_type = problem.GetYDesc().GetType();
    std::vector<miopenDataType_t> types {miopenHalf, miopenFloat, miopenInt8, miopenBFloat16, miopenFloat8};

    const auto idx_type = problem.GetPooling().GetIndexType();
    std::vector<miopenIndexType_t> idx_types {miopenIndexUint32}; // TRJS

    const auto mode = problem.GetPooling().GetMode();
    std::vector<miopenPoolingMode_t> modes {miopenPoolingMax, miopenPoolingAverage/* , miopenPoolingAverageInclusive */};

    const auto x_layout = problem.GetXDesc().GetLayout_str();
    const auto y_layout = problem.GetYDesc().GetLayout_str();
    std::vector<std::string> layouts {"NHWC", "NDHWC"};

    std::cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^      Ck:" <<
        (int)(x_type == y_type) <<
        (int)(x_layout == y_layout) <<
        (int)(std::find(layouts.cbegin(), layouts.cend(), x_layout) != layouts.end()) <<
        (int)(std::find(types.cbegin(), types.cend(), x_type) != types.cend()) <<
        (int)(std::find(idx_types.cbegin(), idx_types.cend(), idx_type) != idx_types.cend()) <<
        (int)(std::find(modes.cbegin(), modes.cend(), mode) != modes.cend()) <<
        (int)(problem.GetPooling().GetMode() == miopenPoolingMax                             //
//             && problem.GetPooling().GetWorkspaceIndexMode() == miopenPoolingWorkspaceIndexMask //
             || problem.SaveIndex() == false) <<
        std::endl;

    return problem.GetDirection() == miopen::pooling::Direction::Forward                      //
           && (x_type == y_type)                                                              //
           && (x_layout == y_layout)                                                          //
           && (std::find(layouts.cbegin(), layouts.cend(), x_layout) != layouts.end())        //
           && (std::find(types.cbegin(), types.cend(), x_type) != types.cend())               //
           && (std::find(idx_types.cbegin(), idx_types.cend(), idx_type) != idx_types.cend()) //
           && (std::find(modes.cbegin(), modes.cend(), mode) != modes.cend())                 //
           && (problem.GetPooling().GetMode() == miopenPoolingMax                             //
//             && problem.GetPooling().GetWorkspaceIndexMode() == miopenPoolingWorkspaceIndexMask //
             || problem.SaveIndex() == false);
}

namespace {

struct CKPoolingRunner
{
    CKPoolingRunner(const Handle& handle_,
                    const ExecutionContext& context_,
                    const miopen::pooling::ProblemDescription& problem_,
                    const AnyInvokeParams& primitive_parameters_) :
                        handle(handle_),
                        context(context_),
                        problem(problem_),
                        primitive_parameters(primitive_parameters_)
                    { }

    void operator()()
    {
        switch(problem.GetXDesc().GetType())
        {
            case miopenHalf: Run<F16>(); break;
            case miopenFloat: Run<F32>(); break;
            case miopenInt8: Run<int8_t>(); break;
            case miopenBFloat16: Run<BF16>(); break;
            case miopenFloat8: Run<F8>(); break;
            default: MIOPEN_THROW("CK pooling does not support types miopenInt32, miopenInt64, miopenBFloat8, miopenDouble");
        }
    }

    template<typename InDataType>
    void Run()
    {
        switch(problem.GetPooling().GetIndexType())
        {
            case miopenIndexUint8: Run<InDataType, int8_t>(); break;
            case miopenIndexUint16: Run<InDataType, int16_t>(); break;
            case miopenIndexUint32: Run<InDataType, int32_t>(); break;
            case miopenIndexUint64: Run<InDataType, int64_t>(); break;
            default: MIOPEN_THROW("Unsupported index type for CK pooling");
        }
    }

    template<typename InDataType,
            typename IndexDataType>
    void Run()
    {
        switch(problem.GetPooling().GetMode())
        {
            case miopenPoolingAverage:
                RunCKPoolingSolution<InDataType, InDataType, IndexDataType, ck::ReduceTensorOp::AVG, false>(
                    handle, primitive_parameters, problem);
                break;
            case miopenPoolingMax:
                problem.SaveIndex() ?    
                    RunCKPoolingSolution<InDataType, InDataType, IndexDataType, ck::ReduceTensorOp::MAX, true>(
                        handle, primitive_parameters, problem) :
                    RunCKPoolingSolution<InDataType, InDataType, IndexDataType, ck::ReduceTensorOp::MAX, false>(
                        handle, primitive_parameters, problem);
                break;
            default: MIOPEN_THROW("CK pooling does not support miopenPoolingAverageInclusive");
        }
    }

    template <typename InDataType,
            typename OutDataType,
            typename IndexDataType,
            ck::ReduceTensorOp ReduceOpId,
            bool OutputIndex>
    static void RunCKPoolingSolution(const Handle& handle,
                            const AnyInvokeParams& primitive_parameters,
                            const miopen::pooling::ProblemDescription& problem)
    {
        const auto& args = CKArgsPoolingFwd{problem};

        using DeviceOp = ck::tensor_operation::device::DevicePoolFwd<InOutRank,
                                                                    WindowRank,
                                                                    InDataType,
                                                                    OutDataType,
                                                                    IndexDataType,
                                                                    ck::tensor_layout::convolution::NDHWC,
                                                                    ck::tensor_layout::convolution::NDHWC,
                                                                    ReduceOpId,
                                                                    OutputIndex>;

        // get device op instances
        const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
            DeviceOp>::GetInstances();

        std::cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^     found " << op_ptrs.size() << " instances" << std::endl;     // TRJS
        const auto& params = primitive_parameters.CastTo<miopen::pooling::FwdInvokeParams>();

        for(int i = 0; i < op_ptrs.size(); ++i)
        {
            auto& op_ptr      = op_ptrs[i];
            auto argument_ptr = op_ptr->MakeArgumentPointer(
                params.x,
                params.y,
                params.workspace,
                args.in_length,
                args.window_spatial_lengths,
                args.out_length,
                args.in_tensor_stride,
                args.out_tensor_stride,
                args.out_tensor_stride,
                args.window_strides,
                args.window_dilations,
                args.input_pads,
                args.input_pads,
                args.pooling_dims);

            auto invoker_ptr = op_ptr->MakeInvokerPointer();
            auto enable_profiling = handle.IsProfilingEnabled();
            enable_profiling = true;    // TRJS enable_profiling

            std::string op_name = op_ptr->GetTypeString();

            if(op_ptr->IsSupportedArgument(argument_ptr.get()))
            {
                std::cout << "Running Ck kernel '" << op_name << "' (index " << i << ")" << std::endl;   // TRJS
                float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{handle.GetStream(), enable_profiling});
                if(enable_profiling)
                {
                    std::cout << "-- execution took " << std::setw(10) << ave_time << " ms" << std::endl;
                }
                // TRJS: anything else to do for success?
                return;
            }
        }

        // TODO no failure is typical
    }

    const Handle& handle;
    const ExecutionContext& context;
    const miopen::pooling::ProblemDescription& problem;
    const AnyInvokeParams& primitive_parameters;
};
}

ConvSolution PoolingForwardCkNd::GetSolution(
    [[maybe_unused]] const ExecutionContext& context,
    [[maybe_unused]] const miopen::pooling::ProblemDescription& problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    ConvSolution result;
    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        std::ignore = kernels;
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
// TRJS: rotate window dims?
// TRJS: does CK support asymmetrical windows at all?
            CKPoolingRunner(handle,
                context,
                problem,
                primitive_parameters)();
        };
    };
    return result;
#else
    return {};
#endif
}

std::size_t
PoolingForwardCkNd::GetWorkspaceSize(const ExecutionContext&,
                                   const miopen::pooling::ProblemDescription& problem) const
{
    if(problem.GetPooling().GetMode() != miopenPoolingMax)
        return 0;
    return problem.GetYDesc().GetElementSize() * get_data_size(problem.GetPooling().GetIndexType());
}

} // namespace pooling

} // namespace solver

} // namespace miopen
