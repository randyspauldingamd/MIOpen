/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/pooling.hpp>
#include <miopen/pooling/invoke_params.hpp>
#include <miopen/pooling/solvers.hpp>

#include <miopen/pooling/poolingNdNhwcArgs.hpp>

#define WORKAROUND_ISSUE_MIFIN_80 1 // https://github.com/ROCm/MIFin/issues/80

namespace miopen {

namespace solver {

namespace pooling {

namespace {

#if !MIOPEN_NDEBUG && !WORKAROUND_ISSUE_MIFIN_80
template <typename T>
bool IsPower2(T v)
{
    return (v != 0) && ((v & (v - 1)) == 0);
}
#endif

template <typename T>
T RoundUpNearestPower2Positive(T v) = delete;

inline uint32_t RoundUpNearestPower2Positive(uint32_t v)
{
    assert(v > 0);
    --v;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return std::max(++v, 1U); // Shut clang-tidy.
}

} // namespace

bool PoolingForwardNDNhwcNaive::IsApplicable(const ExecutionContext&,
                                       const miopen::pooling::ProblemDescription& problem) const
{
    auto x_type = problem.GetXDesc().GetType();
    auto y_type = problem.GetYDesc().GetType();
    std::vector<miopenDataType_t> types {miopenFloat, miopenHalf, miopenInt8, miopenFloat8}; // , miopenBFloat16

    auto mode = problem.GetPooling().GetMode();
    std::vector<miopenPoolingMode_t> modes {miopenPoolingMax, miopenPoolingAverage, miopenPoolingAverageInclusive};

    auto x_layout = problem.GetXDesc().GetLayout_str();
    auto y_layout = problem.GetYDesc().GetLayout_str();
    std::vector<std::string> layouts {"NHWC", "NDHWC"};

    bool app = (problem.GetDirection() == miopen::pooling::Direction::Forward)          //
        && (x_type == y_type)                                                       //
        && (x_layout == y_layout)                                                   //
        && (std::find(types.cbegin(), types.cend(), x_type) != types.cend())        //
        && (std::find(modes.cbegin(), modes.cend(), mode) != modes.cend())          //)
        && (std::find(layouts.cbegin(), layouts.cend(), x_layout) != layouts.end());

    // TODO RJS check grid size

    std::cout << "%%%%%%%%%% PoolingForwardNDNhwcNaive::IsApplicable: " << app << " " <<  problem.GetXDesc().GetLayout_str() << "->" << problem.GetXDesc().GetLayout(x_layout)
     << "  " << problem.GetYDesc().GetLayout_str() << "->" << problem.GetYDesc().GetLayout(y_layout)
       << "  "  << (problem.GetDirection() == miopen::pooling::Direction::Forward)
        << (x_type == y_type)
        << (x_layout == y_layout) << (std::find(types.cbegin(), types.cend(), x_type) != types.cend())
        << (std::find(modes.cbegin(), modes.cend(), mode) != modes.cend()) << (std::find(layouts.cbegin(), layouts.cend(), x_layout) != layouts.end()) << std::endl;

    return app;
}

#include <iomanip>  // TEMPCODE RJS
namespace {
    template<typename T>
    void printVec(std::string name, std::vector<T> vec)
    {
         return;
      std::cout << "Vector Printing: " << std::setw(20) << name << "[" << vec.size() << "]: ";
        for(auto i : vec)    std::cout << std::setw(8) << i;
        std::cout << std::endl;
    }
}

ConvSolution
PoolingForwardNDNhwcNaive::GetSolution(const ExecutionContext& context,
                                 const miopen::pooling::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};
    poolingNdNhwcArgs args; 

    auto input_dtype  = miopen::GetDataType(problem.GetXDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetYDesc().GetType());

    const auto bot  = problem.GetXDesc();
    const auto top  = problem.GetYDesc();
    const bool is2d = (bot.GetNumDims() == 4);
    const bool is_transpose = problem.GetXDesc().GetLayout_str()[1] != 'C';
    if(!is_transpose)
    {
        MIOPEN_THROW("Tried to run NHWC solver on NCHW data");
    }

    // To compact code:
    const auto& pooling = problem.GetPooling();
    const auto& lengths = pooling.GetLengths();
    const auto& strides = pooling.GetStrides();
    const auto& pads    = pooling.GetPads();

    // This also deduces 3D (DHW) parameters from 2D (HW) descriptor.
    uint32_t idx = 0;
    args.filter_d        = is2d ? 1 : lengths[idx++];
    args.filter_h        = lengths[idx++];
    args.filter_w        = lengths[idx++];

    idx = 0;
    args.filter_d_stride = is2d ? (strides[0]) : strides[idx++];
    args.filter_h_stride = strides[idx++];
    args.filter_w_stride = strides[idx++];

    idx = 0;
    args.filter_d_pad    = is2d ? 0 : pads[idx++];
    args.filter_h_pad    = pads[idx++];
    args.filter_w_pad    = pads[idx++];

    // TODO RJS move pooling_method to shared code
    const int pooling_method = (pooling.GetMode() == miopenPoolingMax) ? MLO_POOLING_OP_MAX
                               : (pooling.GetMode() == miopenPoolingAverage)
                                   ? MLO_POOLING_OP_AVE
                                   : MLO_POOLING_OP_AVE_INCLUSIVE;

    const auto save_index = problem.SaveIndex();
    const auto index_mode = pooling.GetWorkspaceIndexMode();
    const auto index_type = pooling.GetIndexType();

    /// \anchor multiply_dims_overflow_assumption
    ///
    /// Preventing overflow during dimension-related computations:
    /// Let's assume that multiplication of three dims always fits into 32 bits (unsigned).
    /// Then let's use size_t when we need to multiply more than three dims.
    /// For example, in NCDHW layout, the N and C strides are results of multiplication
    /// of >= 3 dims, so we have to use size_t for storing them.
    ///
    /// We need to pay special attention to muls of D stride with some other dims.
    /// The D stride is a result of 2 muls. Therefore (d_stride * dim) does
    /// not require widening to size_t prior mul, but (d_stride * dim * dim)
    /// requires it because the total number of muls is 4.

    // TEMPCODE RJS
    printVec("======================================================================", std::vector<int>{});
    printVec("bot lengths", bot.GetLengths());
    printVec("bot strides", bot.GetStrides());
    printVec("top lengths", top.GetLengths());
    printVec("top strides", top.GetStrides());
    printVec("pool lengths", lengths);
    printVec("pool strides", strides);
    printVec("pool pads", pads);
    printVec("======================================================================", std::vector<int>{});

    const auto spatial_dim = is2d ? 2U : 3U;

    std::tie(args.all_n, args.all_c, args.bot_d, args.bot_h, args.bot_w) = miopen::GetNCDHW(spatial_dim, bot.GetLengths());

    std::tie(args.bot_n_stride, args.bot_c_stride, args.bot_d_stride, args.bot_h_stride, args.bot_w_stride) =
        miopen::GetNCDHW(spatial_dim, bot.GetStrides());

    std::tie(std::ignore, std::ignore, args.top_d, args.top_h, args.top_w) =
        miopen::GetNCDHW(spatial_dim, top.GetLengths());

    std::tie(args.top_n_stride, args.top_c_stride, args.top_d_stride, args.top_h_stride,args. top_w_stride) =
        miopen::GetNCDHW(spatial_dim, top.GetStrides());

    // Mask data is always NCDHW layout
    args.mask_w_stride = 1;
    args.mask_h_stride = args.mask_w_stride * args.top_w;
    args.mask_d_stride = args.mask_h_stride * args.top_h;
    args.mask_c_stride   = static_cast<BIGONE>(args.mask_d_stride) * args.top_d;
    args.mask_n_stride   = args.mask_c_stride * args.all_c;

    /// About optimal grid size:
    /// top D, H, and W are mapped directly onto grid dimensions, except in very small problems
    /// when they are combined into workgroup items in an attempt to improve overlapping and coalescense.
    /// N seems to be generally small, so we'll multiply it into the 'D' dimension.
    ///
    /// \anchor naive_pooling_max_grid_size
    /// * Assumption: Max grid size is >= 2^32-1 (4G-1) i.e. std::max<uint32_t>.
    ///   However, assume the product of two dimensions is always <= 2^30.
    ///   Currently this limitation is valid for both ROCm HIP and OCL runtimes.
    ///
    /// Selecting the optimal workgroup size is an interesting problem.
    /// We'll first map N * D to blockIdx.x. H and W are canonically mapped into
    /// blockIdx.y and z, respectively. C, being the fastest index, is mapped
    /// into threadIdx.x up to the maximum items. For larger C, the remainder are
    /// mapped into blockIdx.z.
    ///
    /// For small C, we favor more waves over more blocks. W/H are mapped into threadIdx.z/y,
    /// in that order, fractionally in powers of 2 if possible, up to a maximum
    /// of 256 workitems. Finally, any remaining W/H are then mapped onto blockIdx.z/y.
    ///
    /// The workgroup size does not have the restrictions imposed by synchronization between
    /// workitems because the kernel does not require synchronization.

    std::ignore = context;
    constexpr uint32_t MAX_THREADS       = 512;
    constexpr uint32_t LARGE_C_MAX_ITEMS = MAX_THREADS;
    constexpr uint32_t SMALL_C_TGT_ITEMS = 256;

    auto nd_ = args.all_n * args.top_d;
    auto h_  = args.top_h;
    auto w_  = args.top_w;
    auto c_  = args.all_c;

    // These are hip-style indexes (not OCL)
    uint32_t l1 = 1U;
    uint32_t l2 = 1U;

    if(c_ > LARGE_C_MAX_ITEMS)
    {
        auto c2 = (c_ + LARGE_C_MAX_ITEMS - 1) / LARGE_C_MAX_ITEMS;
        c_ = LARGE_C_MAX_ITEMS;
        w_ *= c2;
    }
    else if(c_ <= SMALL_C_TGT_ITEMS / 2)    // Small C, remap H and W to increase occupancy
    {
        if(c_ * w_ < SMALL_C_TGT_ITEMS)
        {
            std::swap(l2, w_);              // full w mapped to threads
        }

        while(w_ > 2 && ((c_ * l2) < SMALL_C_TGT_ITEMS))
        {
            w_ = (w_ + 1) / 2;              // partial w mapped to threads (rounddown-safe)
            l2 *= 2;
        }

        if(c_ * l2 * h_ < SMALL_C_TGT_ITEMS)
        {
            std::swap(l1, h_);              // full h mapped to threads
        }

        while(h_ > 2 && ((c_ * l1 * l2) < SMALL_C_TGT_ITEMS))
        {
            h_ = (h_ + 1 ) / 2;             // partial h mapped to threads (rounddown-safe)
            l1 *= 2;
        }
    }

    const auto g0 = nd_;
    const auto g1 = h_;
    const auto g2 = w_;
    const auto l0 = c_;

    {
        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenPoolingForwardNDNhwcNaive.cpp";
        kernel.kernel_name = "mloPoolingForwardNDNhwcNaive";

        auto build_params = KernelBuildParameters{
            {"MLO_POOLING_OP_ID", pooling_method}, // We need this at compile time in order to
                                                   // engage mixed precision only when necessary.
            {"MLO_POOLING_INDEX_TYPE", get_pooling_index_type_name(index_type)},
            {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
            {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype}
        };

        build_params << GetDataTypeKBP(bot.GetType());
        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        // [Informative] The total number of kernels required to cover the whole
        // forward pooling problem space is 3*4*2*2*2 = 96. The solver is dynamic.
        // * 3: the number of supported operations
        // * 4: the number of supported index types
        // * 2: the number of supported data types
        // * 2: layout (NCHW vs NHWC)
        // * 2: 2D and 3D kernels (optimization)

        // KernelInfo uses OCL-style indexes
        kernel.l_wk.clear();
        kernel.l_wk.push_back(l0);
        kernel.l_wk.push_back(l1);
        kernel.l_wk.push_back(l2);
        kernel.g_wk.clear();
        kernel.g_wk.push_back(g0 * l0);
        kernel.g_wk.push_back(g1 * l1);
        kernel.g_wk.push_back(g2 * l2);

        // TEMPCODE RJS
std::cout << "Kernel dims: g[" << kernel.g_wk.size() << "] " << kernel.g_wk[0] << " " << kernel.g_wk[1] << " " << kernel.g_wk[2]
<< " | l[" << kernel.l_wk.size() << "] " << kernel.l_wk[0] << " " << kernel.l_wk[1] << " " << kernel.l_wk[2] << std::endl;
        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::pooling::FwdInvokeParams>();

            kernel(
                params.x,
                params.y,
                params.junk,   // TEMPCODE RJS
                params.workspace,
                save_index,
                index_mode,
                args.filter_d, args.filter_h, args.filter_w,
                args.filter_d_stride, args.filter_h_stride, args.filter_w_stride,
                args.filter_d_pad, args.filter_h_pad, args.filter_w_pad,
                args.all_n,
                args.all_c,
                args.bot_d, args.bot_h, args.bot_w,
                args.bot_n_stride, args.bot_c_stride, args.bot_d_stride, args.bot_h_stride, args.bot_w_stride,
                args.top_d, args.top_h, args.top_w,
                args.top_n_stride, args.top_c_stride, args.top_d_stride, args.top_h_stride, args.top_w_stride,
                args.mask_n_stride, args.mask_c_stride, args.mask_d_stride, args.mask_h_stride, args.mask_w_stride
            );
        };
    };

    return result;
}

std::size_t
PoolingForwardNDNhwcNaive::GetWorkspaceSize(const ExecutionContext&,
                                      const miopen::pooling::ProblemDescription& problem) const
{
    if(problem.GetPooling().GetMode() != miopenPoolingMax || !problem.SaveIndex())
        return 0;
    return problem.GetYDesc().GetElementSize() * get_data_size(problem.GetPooling().GetIndexType());
}

} // namespace pooling

} // namespace solver

} // namespace miopen
