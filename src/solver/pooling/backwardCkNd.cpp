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

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_max_pool_bwd.hpp"
#include "ck/utility/reduction_enums.hpp"   // TRJS build only--prob not needed

#include "ck/library/tensor_operation_instance/gpu/avg_pool3d_bwd.hpp"
#include "ck/library/tensor_operation_instance/gpu/max_pool_bwd.hpp"

namespace miopen {

namespace solver {

namespace pooling {

namespace {

// convert 2D problems to 3D
constexpr ck::index_t InOutRank  = 5;
constexpr ck::index_t WindowRank = 3;

using DOutLayout = ck::tensor_layout::convolution::NDHWC;
using DInLayout  = ck::tensor_layout::convolution::NDHWC;

using F8   = ck::half_t;     // TRJS does CK support float8?
using F16  = ck::half_t;
using F32  = float;
using F64  = double;
using BF16 = ushort;

struct CKArgsPoolingBwd
{
    template <typename T>
    static void MakeCkVec(std::vector<ck::index_t>& ck, std::vector<T> in)
    {
        ck.resize(in.size());
        std::copy(in.begin(), in.end(), ck.begin());
    }

    CKArgsPoolingBwd(const miopen::pooling::ProblemDescription& problem)
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

bool PoolingBackwardCkNd::IsApplicable(const ExecutionContext&,
                                     const miopen::pooling::ProblemDescription& problem) const
{
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
        (int)!(problem.GetPooling().GetMode() != miopenPoolingMax && problem.SaveIndex() == true) <<
        std::endl;

    return problem.GetDirection() == miopen::pooling::Direction::Backward                      //
           && (x_type == y_type)                                                              //
           && (x_layout == y_layout)                                                          //
           && (std::find(layouts.cbegin(), layouts.cend(), x_layout) != layouts.end())        //
           && (std::find(types.cbegin(), types.cend(), x_type) != types.cend())               //
           && (std::find(idx_types.cbegin(), idx_types.cend(), idx_type) != idx_types.cend()) //
           && (std::find(modes.cbegin(), modes.cend(), mode) != modes.cend())                 //
           && !(problem.GetPooling().GetMode() != miopenPoolingMax                            //
//             && problem.GetPooling().GetWorkspaceIndexMode() == miopenPoolingWorkspaceIndexMask //
             && problem.SaveIndex() == true);
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

    template <typename DInDataType,
            typename DOutDataType,
            typename IndexDataType,
            ck::ReduceTensorOp ReduceOpId,
            bool OutputIndex>
    static void RunCKMaxPoolingBwdSolution(const Handle& handle,
                            const AnyInvokeParams& primitive_parameters,
                            const miopen::pooling::ProblemDescription& problem)
    {
        const auto& args = CKArgsPoolingBwd{problem};

        using MaxPoolBwdDeviceOp =
            ck::tensor_operation::device::DeviceMaxPoolBwd<DOutDataType, IndexDataType, DInDataType>;

        // get device op instances
        const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
            MaxPoolBwdDeviceOp>::GetInstances();

        std::cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^     found " << op_ptrs.size() << " instances" << std::endl;     // TRJS
        const auto& params = primitive_parameters.CastTo<miopen::pooling::BwdInvokeParams>();

        for(int i = 0; i < op_ptrs.size(); ++i)
        {
            auto& op_ptr      = op_ptrs[i];
            auto argument_ptr = op_ptr->MakeArgumentPointer(
                params.dy,
                params.workspace,
                params.dx,
                args.out_length,
                args.in_length,
                args.window_spatial_lengths,
                args.window_strides,
                args.window_dilations);

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

    template <typename DInDataType,
            typename DOutDataType,
            typename IndexDataType,
            ck::ReduceTensorOp ReduceOpId,
            bool OutputIndex>
    static void RunCKAvgPoolingBwdSolution(const Handle& handle,
                            const AnyInvokeParams& primitive_parameters,
                            const miopen::pooling::ProblemDescription& problem)
    {
        const auto& args = CKArgsPoolingBwd{problem};

        using DeviceOp = ck::tensor_operation::device::
            DeviceAvgPoolBwd<WindowRank, DOutDataType, DInDataType, DOutLayout, DInLayout>;

        // get device op instances
        const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
            DeviceOp>::GetInstances();

        std::cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^     found " << op_ptrs.size() << " instances" << std::endl;     // TRJS
        const auto& params = primitive_parameters.CastTo<miopen::pooling::BwdInvokeParams>();

        for(int i = 0; i < op_ptrs.size(); ++i)
        {
            auto& op_ptr      = op_ptrs[i];
            auto argument_ptr = op_ptr->MakeArgumentPointer(
                params.dy,
                params.dx,
                args.out_length,
                args.in_length,
                args.out_tensor_stride,
                args.in_tensor_stride,
                args.window_spatial_lengths,
                args.window_strides,
                args.window_dilations,
                args.input_pads,
                args.input_pads);

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

ConvSolution
PoolingBackwardCkNd::GetSolution(const ExecutionContext&,
                               const miopen::pooling::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

//     auto kernel        = KernelInfo{};
//     kernel.kernel_file = "MIOpenPoolingBwdND.cl";
//     kernel.kernel_name = "mloPoolingND";
// // TODO: backwardCkNd kernel
//     if(problem.GetPooling().GetMode() == miopenPoolingMax)
//     {
//         kernel.kernel_name += "MaxBwd";
//     }
//     else if(problem.GetPooling().GetMode() == miopenPoolingAverage ||
//             problem.GetPooling().GetMode() == miopenPoolingAverageInclusive)
//     {
//         kernel.kernel_name += "AveBwd";
//     }

//     const auto& bot = problem.GetXDesc();
//     const auto& top = problem.GetYDesc();

//     std::size_t batch_sz, n_inputs, in_height, in_width;
//     std::tie(batch_sz, n_inputs, in_height, in_width) = miopen::tien<4>(bot.GetLengths(), 1);

//     const int pooling_method = (problem.GetPooling().GetMode() == miopenPoolingMax)
//                                    ? MLO_POOLING_OP_MAX
//                                    : ((problem.GetPooling().GetMode() == miopenPoolingAverage)
//                                           ? MLO_POOLING_OP_AVE
//                                           : MLO_POOLING_OP_AVE_INCLUSIVE);

//     int pix_w_per_work = 1;
//     int pix_h_per_work = 4;
//     int pix_d_per_work = 2;

//     int batch = top.GetLengths()[0];
//     int chal  = top.GetLengths()[1];

//     const bool is2d = (bot.GetNumDims() == 4);

//     int bot_d = is2d ? 1 : *(bot.GetLengths().rbegin() + 2);
//     int bot_h = *(bot.GetLengths().rbegin() + 1);
//     int bot_w = *(bot.GetLengths().rbegin());

//     int pix_blk_w = std::max((bot_w + pix_w_per_work - 1) / pix_w_per_work, 1);
//     int pix_blk_h = std::max((bot_h + pix_h_per_work - 1) / pix_h_per_work, 1);
//     int pix_blk_d = std::max((bot_d + pix_d_per_work - 1) / pix_d_per_work, 1);

//     int max_activ_workitem = 65536;
//     int total_work         = batch * chal * pix_blk_w * pix_blk_h * pix_blk_d;
//     int activ_work         = std::min(total_work, max_activ_workitem);

// #if WORKAROUND_ISSUE_MIFIN_80
//     const std::size_t wavesize = 64;
// #else
//     const std::size_t wavesize = context.GetStream().GetWavefrontWidth();
// #endif
//     size_t grp_num = (activ_work + wavesize - 1) / wavesize;

//     auto strides = problem.GetPooling().strides;
//     auto lens    = problem.GetPooling().lens;
//     auto pads    = problem.GetPooling().pads;

//     if(is2d)
//     {
//         strides.push_back(strides[1]);
//         strides[1] = strides[0];
//         lens.push_back(lens[1]);
//         lens[1] = lens[0];
//         lens[0] = 1;
//         pads.push_back(pads[1]);
//         pads[1] = pads[0];
//         pads[0] = 0;
//     }

//     bool territory_overlap = false;
//     for(std::size_t i = 0; i < strides.size(); i++)
//         territory_overlap |= (strides[i] < lens[i]);

//     const auto build_params =
//         KernelBuildParameters{
//             {"MLO_POOLING_OP_ID", pooling_method},
//             {"MAX_ACTIV_WORKITEM", max_activ_workitem},
//             {"MLO_POOLING_GROUP_SZ0", wavesize},
//             {"MLO_POOLING_GROUP_SZ1", 1},
//             {"MLO_POOLING_GROUP_SZ2", 1},
//             {"PIX_W_PER_WORK", pix_w_per_work},
//             {"PIX_H_PER_WORK", pix_h_per_work},
//             {"PIX_D_PER_WORK", pix_d_per_work},
//             {"KERNEL_SZ_D", lens[0]},
//             {"KERNEL_SZ_H", lens[1]},
//             {"KERNEL_SZ_W", lens[2]},
//             {"STRIDE_D", strides[0]},
//             {"STRIDE_H", strides[1]},
//             {"STRIDE_W", strides[2]},
//             {"TERRITORY_OVERLAP", static_cast<int>(territory_overlap)},
//             {"MLO_POOLING_INDEX_TYPE",
//              get_pooling_index_type_name(problem.GetPooling().GetIndexType())},
//             {"MLO_POOLING_INDEX_MAX",
//              get_pooling_index_type_max_name(problem.GetPooling().GetIndexType())},
//         }
//         << GetDataTypeKBP(problem.GetDYDesc().GetType());

//     kernel.comp_options = build_params.GenerateFor(kbp::OpenCL{});

//     kernel.l_wk = {wavesize, 1, 1};
//     kernel.g_wk = {wavesize * grp_num, 1, 1};

//     result.construction_params.push_back(kernel);

//     const auto top_d = is2d ? 1 : *(top.GetLengths().rbegin() + 2);
//     const auto top_h = *(top.GetLengths().rbegin() + 1);
//     const auto top_w = *(top.GetLengths().rbegin());

//     auto unpackStrides = [is2d](const auto& strides) {
//         return std::make_tuple(strides[0], // N stride
//                                strides[1], // C stride
//                                strides[2], // D stride. Same as H_stride in 3D converted from 2D.
//                                is2d        //
//                                    ? strides[2] // 2D H stride
//                                    : strides[3] // 3D H stride
//         );
//     };

//     std::size_t bot_n_stride, bot_c_stride, bot_d_stride, bot_h_stride;
//     std::size_t top_n_stride, top_c_stride, top_d_stride, top_h_stride;
//     std::tie(bot_n_stride, bot_c_stride, bot_d_stride, bot_h_stride) =
//         unpackStrides(bot.GetStrides());
//     std::tie(top_n_stride, top_c_stride, top_d_stride, top_h_stride) =
//         unpackStrides(top.GetStrides());

//     result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
//         return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
//             decltype(auto) kernel = handle_.Run(kernels.front());
//             decltype(auto) params = raw_params.CastTo<miopen::pooling::BwdInvokeParams>();

//             if(params.pooling.GetMode() == miopenPoolingMax)
//             {
//                 kernel(params.dy,
//                        params.dx,
//                        params.workspace,
//                        static_cast<unsigned>(pads[0]),
//                        static_cast<unsigned>(pads[1]),
//                        static_cast<unsigned>(pads[2]),
//                        static_cast<unsigned>(batch),
//                        static_cast<unsigned>(chal),
//                        static_cast<unsigned>(bot_d),
//                        static_cast<unsigned>(bot_h),
//                        static_cast<unsigned>(bot_w),
//                        static_cast<unsigned>(top_d),
//                        static_cast<unsigned>(top_h),
//                        static_cast<unsigned>(top_w),
//                        static_cast<unsigned>(bot_n_stride),
//                        static_cast<unsigned>(bot_c_stride),
//                        static_cast<unsigned>(bot_d_stride),
//                        static_cast<unsigned>(bot_h_stride),
//                        static_cast<unsigned>(top_n_stride),
//                        static_cast<unsigned>(top_c_stride),
//                        static_cast<unsigned>(top_d_stride),
//                        static_cast<unsigned>(top_h_stride),
//                        static_cast<unsigned>(total_work));
//             }
//             else
//             {
//                 kernel(params.dy,
//                        params.dx,
//                        static_cast<unsigned>(pads[0]),
//                        static_cast<unsigned>(pads[1]),
//                        static_cast<unsigned>(pads[2]),
//                        static_cast<unsigned>(batch),
//                        static_cast<unsigned>(chal),
//                        static_cast<unsigned>(bot_d),
//                        static_cast<unsigned>(bot_h),
//                        static_cast<unsigned>(bot_w),
//                        static_cast<unsigned>(top_d),
//                        static_cast<unsigned>(top_h),
//                        static_cast<unsigned>(top_w),
//                        static_cast<unsigned>(bot_n_stride),
//                        static_cast<unsigned>(bot_c_stride),
//                        static_cast<unsigned>(bot_d_stride),
//                        static_cast<unsigned>(bot_h_stride),
//                        static_cast<unsigned>(top_n_stride),
//                        static_cast<unsigned>(top_c_stride),
//                        static_cast<unsigned>(top_d_stride),
//                        static_cast<unsigned>(top_h_stride),
//                        static_cast<unsigned>(total_work));
//             }
//         };
//     };

    return result;
}

std::size_t
PoolingBackwardCkNd::GetWorkspaceSize(const ExecutionContext&,
                                    const miopen::pooling::ProblemDescription& problem) const
{
    if(problem.GetPooling().GetMode() != miopenPoolingMax)
        return 0;
    return problem.GetYDesc().GetElementSize() * get_data_size(problem.GetPooling().GetIndexType());
}

} // namespace pooling

} // namespace solver

} // namespace miopen
