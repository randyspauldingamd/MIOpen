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

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "pooling_functions.h"

#include <algorithm>

#if(MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE) || (MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE_INCLUSIVE)
#define AVERAGE_OPS 1
#else
#define AVERAGE_OPS 0
#endif

// Let's use extended-precision accumulator only in FP16 pooling and only for averaging.
// For all other ops and datatypes, use native accumulator, i.e. treate FLOAT_ACCUM as FLOAT.
#if !(AVERAGE_OPS && MIOPEN_USE_FP16)
#define MIOPEN_USE_NATIVE_DATATYPE_ACCUM 1
#endif
#include "float_types.h"

#ifndef MLO_POOLING_IS2D_KERNEL
#error "MLO_POOLING_IS2D_KERNEL must be defined"
#endif

#if AVERAGE_OPS
#define ARG_UNUSED_FOR_AVERAGE __attribute__((__unused__))
#else
#define ARG_UNUSED_FOR_AVERAGE
#endif

#if MLO_POOLING_IS2D_KERNEL
#define ARG_UNUSED_FOR_2D __attribute__((__unused__))
#else
#define ARG_UNUSED_FOR_2D
#endif

// Out N, D, H are encoded into the block indices x, y, z
// Requires all lens, strides, pads to be in DHW[NC] order. The code is
// cleaner and more performant this way.
// No 2D-only optimization.
template <typename TI, typename TO>
__device__ void poolingFwdNDNhwcNaive(const TI* in_data,
                                    TO* out_data,
                                    ARG_UNUSED_FOR_AVERAGE index_t* mask_ptr,
                                    ARG_UNUSED_FOR_AVERAGE int save_index,
                                    ARG_UNUSED_FOR_AVERAGE int index_mode,
                                    std::vector<uint32_t> filter_lens,
                                    std::vector<uint32_t> filter_strides,
                                    std::vector<uint32_t> filter_pads,
                                    uint32_t all_n,
                                    uint32_t all_c,
                                    std::vector<uint32_t> lens,
                                    std::vector<size_t> strides,
                                    std::vector<uint32_t> out_lens,
                                    std::vector<size_t> out_strides,
                                    ARG_UNUSED_FOR_AVERAGE std::vector<size_t> mask_strides)
{
    constexpr uint32_t D_IDX = 0;
    constexpr uint32_t H_IDX = 1;
    constexpr uint32_t W_IDX = 2;
    constexpr uint32_t N_IDX = 3;
    constexpr uint32_t C_IDX = 4;

    const uint32_t b = blockIdx.x;  // out N
    if(!(b < all_n))
        return;

    const uint32_t k = blockIdx.y;  // out D
    if(!(k < out_lens[D_IDX]))
        return;

    const uint32_t j = blockIdx.z;  // out H
    if(!(j < out_lens[H_IDX]))
        return;

    for(uint32_t i = 0; i < out_lens[W_IDX]; ++i)  // out W
    {
        for(uint32_t o = 0; o < all_c ++o)  // out C
        {
            const auto int_dstart   = static_cast<int64_t>(k * filter_strides[D_IDX]) - static_cast<int64_t>(filter_pads[D_IDX]);
            const auto int_hstart   = static_cast<int>(j * filter_strides[H_IDX]) - static_cast<int>(filter_pads[H_IDX]);
            const auto int_wstart        = static_cast<int>(i * filter_strides[W_IDX]) - static_cast<int>(filter_pads[W_IDX]);
            const auto dend           = static_cast<size_t>(min(int_dstart + static_cast<int64_t>(filter_lens[D_IDX]), static_cast<int64_t>(out_lens[D_IDX])));
            const auto hend             = static_cast<uint32_t>(min(int_hstart + static_cast<int>(filter_lens[H_IDX]), static_cast<int>(out_lens[H_IDX])));
            const auto wend             = static_cast<uint32_t>(min(int_wstart + static_cast<int>(filter_lens[W_IDX]), static_cast<int>(out_lens[W_IDX])));
            const auto dstart         = static_cast<size_t>(max(int_dstart, 0));
            const auto hstart         = static_cast<uint32_t>(max(int_hstart, 0));
            const auto wstart           = static_cast<uint32_t>(max(int_wstart, 0));

#if MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE
        uint32_t pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
        pool_size       = (pool_size == 0) ? 1 : pool_size;
#elif MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE_INCLUSIVE
        const uint32_t pool_size = filter_lens[D_IDX] * filter_lens[H_IDX] * filter_lens[W_IDX];
#endif

#if AVERAGE_OPS
        _FLOAT_ACCUM res = (_FLOAT_ACCUM)(0);
#else // MAX
            _FLOAT_ACCUM res     = (_FLOAT_ACCUM)(-MAX_VAL_ACCUM);
            bool found           = false; // May remain false if bot contains only NaNs/-INFs.
            uint32_t d_save          = 0;
            uint32_t h_save          = 0;
            uint32_t w_save          = 0;
#endif
        for(size_t d = dstart; d < dend; ++d)
        {
            for(uint32_t h = hstart; h < hend; ++h)
            {
                for(uint32_t w = wstart; w < wend; ++w)
                {
                    const size_t in_index = b * strides[N_IDX] +           //
                                             o * strides[C_IDX] +           //
                                             d * strides[D_IDX] + //
                                             static_cast<size_t>(h * strides[H_IDX]) + //
                                             static_cast<size_t>(w * strides[W_IDX]);
#if AVERAGE_OPS
                    res += in_data[in_index];
#else // MAX
                        if(static_cast<_FLOAT_ACCUM>(bot_ptr[bot_index] > res))
                        {
                            res = in_data[in_index];
                            if(save_index)
                            {
                                found  = true;
                                d_save = d;
                                h_save = h;
                                w_save = w;
                            }
                        }
#endif
                }
            }
        }

#if AVERAGE_OPS
        res *= CVT_FP32_2ACCUM(1.f) / static_cast<_FLOAT_ACCUM>(pool_size);
#else // MAX
            if(save_index)
            {
                index_t res_index = 0;

                /// Preventing overflow during computation of res_index:
                /// If Index is shorter than uint, then let's perform computation in 32-bit
                /// domain and then convert to narrower Index. That would reduce the probability of
                /// overflow. If Index is wider then 32 bits, then it seems like it is better to
                /// convert to Index type before multiplication. However this is not actually
                /// necessary, see \ref multiply_dims_overflow_assumption. Let's always compute in
                /// 32 bits and then convert.

                if(found)
                {
                    if(index_mode == 1)
                        res_index = (index_t)(d_save * lens[H_IDX] * lens[W_IDX] //
                                              + h_save * lens[W_IDX]       //
                                              + w_save);
                    else
                        res_index = (index_t)(                                                    //
                            ((d_save - k * filter_strides[D_IDX] + filter_pads[D_IDX]) * filter_lens[W_IDX] * filter_lens[H_IDX]) //
                            + ((h_save - j * filter_strides[H_IDX] + filter_pads[H_IDX]) * filter_lens[W_IDX])          //
                            + (w_save - i * filter_strides[W_IDX] + filter_pads[W_IDX])                       //
                        );
                }

                const size_t mask_index = b * mask_strides[N_IDX]             //
                                          + o * mask_strides[C_IDX]           //
                                          + (size_t)(k * mask_strides[D_IDX]) //
                                          + (size_t)(j * mask_strides[H_IDX]) //
                                          + (size_t)(i * mask_strides[W_IDX]);
                mask_ptr[mask_index] = res_index;
            }
#endif
        const size_t out_index = out_strides[N_IDX]             //
                                 + o * out_strides[C_IDX]           //
                                 + (size_t)(k * out_strides[D_IDX]) //
                                 + (size_t)(j * out_strides[H_IDX]) //
                                 + (size_t)(i * out_strides]W_IDX]);

        out_data[out_index] = (_FLOAT)res;
    }
}
}

extern "C" __global__ void mloPoolingForwardNDNhwcNaive(const INPUT_TYPE* __restrict__ in_data,
                                     OUTPUT_TYPE* out_data,
                                     ARG_UNUSED_FOR_AVERAGE index_t* mask_ptr,
                                     ARG_UNUSED_FOR_AVERAGE int save_index,
                                     ARG_UNUSED_FOR_AVERAGE int index_mode,
                                     std::vector<uint32_t> filter_lens,
                                     std::vector<uint32_t> filter_strides,
                                     std::vector<uint32_t> filter_pads,
                                     uint32_t all_n,
                                     uint32_t all_c,
                                     std::vector<uint32_t> lens,
                                     std::vector<size_t> strides,
                                     std::vector<uint32_t> out_lens,
                                     std::vector<size_t> out_strides,
                                     ARG_UNUSED_FOR_AVERAGE std::vector<size_t> mask_strides)
{
    poolingFwdNDNhwcNaive<INPUT_TYPE, OUTPUT_TYPE>(
        in_data,
        out_data,
        mask_ptr,
        save_index,
        index_mode,
        filter_lens,
        filter_strides,
        filter_pads,
        all_n,
        all_c,
        lens,
        strides,
        out_lens,
        out_strides,
        mask_strides
    );
}
