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
#include <hip/hip_runtime.h>
#endif

#include "pooling_functions.h"
#include "poolingNdNhwcArgs.hpp"

// TODO: add ability to decode network string into pooling descriptor or similar for targeted debugging

#if(MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE) || (MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE_INCLUSIVE)
#define AVERAGE_OPS 1
#else
#define AVERAGE_OPS 0
#endif

// Let's use extended-precision accumulator only in FP16 pooling and only for averaging.
// For all other ops and datatypes, use native accumulator, i.e. treate FLOAT_ACCUM as FLOAT.
#if !(AVERAGE_OPS && MIOPEN_USE_FP16)
#define MIOPEN_USE_NATIVE_DATATYPE_ACCUM 1
// #else
// #define MIOPEN_USE_NATIVE_DATATYPE_ACCUM 0
#endif

#include "float_types.h"
#include "miopen_cstdint.hpp"

// This implementation is extremely memory-bound, so float type is used for all calculations
#define _FLOAT          float
#define _FLOAT_ACCUM    float

#if MIOPEN_USE_INT8 == 1
    #if !AVERAGE_OPS
        #ifndef FLT_MAX
        #define MAX_VAL 127 /* max value */
        #else
        #define MAX_VAL FLT_MAX
        #endif
    #endif
#endif
#if MIOPEN_USE_BFP16
    #define NATIVE_CAST(_x)     (_FLOAT)bfloat16_to_float(_x)
    #define NATIVE_UNCAST(_x)   (_FLOAT)float_to_bfloat16(_x)
#else
    #define NATIVE_CAST(_x)     (_FLOAT)(_x)
    #define NATIVE_UNCAST(_x)   (_FLOAT)(_x)
#endif

#if AVERAGE_OPS
#define ARG_UNUSED_FOR_AVERAGE __attribute__((__unused__))
#else
#define ARG_UNUSED_FOR_AVERAGE
#endif

// Out N, D, H are encoded into the block indices x, y, z
// No 2D-only optimization.
template <typename TI>
__device__ void poolingForwardNDNhwcNaive(const TI* __restrict__ bot_ptr,
                                    TI* __restrict__ top_ptr,
                                    ARG_UNUSED_FOR_AVERAGE index_t* __restrict__ mask_ptr,
                                    ARG_UNUSED_FOR_AVERAGE int save_index,
                                    ARG_UNUSED_FOR_AVERAGE int index_mode,
                                    poolingNdNhwcArgs args
)
{
    // naming: caps=count, lowercase=index, <canonical>_<modified>
    const uint32_t nd = blockIdx.x;
    const uint32_t h_ = blockIdx.y;
    const uint32_t w_c = blockIdx.z;
    const uint32_t w_ = w_c % args.top_w;                   // CAN w=fast index

    const uint32_t C_WH = blockDim.x;
    const uint32_t _H = blockDim.y;
    const uint32_t _W = blockDim.z;

    const uint32_t c  = threadIdx.x;
    const uint32_t _h = threadIdx.y;
    const uint32_t _w = threadIdx.z;

    const uint32_t nn = nd / args.top_d;                    // n=slow index
    const uint32_t cc = (w_c / args.top_w) * C_WH + c;      // c=slow index (lg-C)
    const uint32_t td = nd % args.top_d;                    // top d=fast index
    const uint32_t th = h_ * _H + _h;                       // top h: blockIdx is slow (sm-C)
    const uint32_t tw = w_ * _W + _w;                       // top w: blockIdx is slow (sm-C)

    if(nn >= args.all_n) return;
    if(td >= args.top_d) return;
    if(th >= args.top_h) return;
    if(tw >= args.top_w) return;
    if(cc >= args.all_c) return;

    const auto int_dstart   = static_cast<int64_t>(td * args.filter_d_stride) - static_cast<int64_t>(args.filter_d_pad);
    /* const */ auto dend         = static_cast<size_t>(min(int_dstart + static_cast<int64_t>(args.filter_d), static_cast<int64_t>(args.bot_d)));
    const auto dstart       = static_cast<size_t>(max(int_dstart, 0));

    const auto int_hstart   = static_cast<int>(th * args.filter_h_stride) - static_cast<int>(args.filter_h_pad);
    /* const */ auto hend             = static_cast<uint32_t>(min(int_hstart + static_cast<int>(args.filter_h), static_cast<int>(args.bot_h)));
    const auto hstart         = static_cast<uint32_t>(max(int_hstart, 0));

    const auto int_wstart        = static_cast<int>(tw * args.filter_w_stride) - static_cast<int>(args.filter_w_pad);
    /* const */ auto wend             = static_cast<uint32_t>(min(int_wstart + static_cast<int>(args.filter_w), static_cast<int>(args.bot_w)));
    const auto wstart           = static_cast<uint32_t>(max(int_wstart, 0));

    size_t top_index = 
            nn * args.top_n_stride +            //
            cc * args.top_c_stride +            //
            td * args.top_d_stride +            //
            th * args.top_h_stride +            //
            tw * args.top_w_stride;

#if MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE
    uint32_t pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
    pool_size       = (pool_size == 0) ? 1 : pool_size;
#elif MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE_INCLUSIVE
    const uint32_t pool_size = args.filter_d * args.filter_h * args.filter_w;
#endif

#if AVERAGE_OPS
    _FLOAT_ACCUM res = (_FLOAT_ACCUM)(0);
#else // MAX
    _FLOAT_ACCUM res     = (_FLOAT_ACCUM)NATIVE_CAST(-MAX_VAL_ACCUM);
    bool found           = false; // May remain false if bot contains only NaNs/-INFs.
    uint32_t d_save          = 0;
    uint32_t h_save          = 0;
    uint32_t w_save          = 0;
    uint32_t saved_index     = 0;
#endif

    size_t bot_ncd = static_cast<size_t>(nn * args.bot_n_stride + cc * args.bot_c_stride + dstart * args.bot_d_stride);
    for(size_t bd = dstart; bd < dend; ++bd)
    {
        size_t bot_ncdh = bot_ncd + hstart * args.bot_h_stride;
        for(uint32_t bh = hstart; bh < hend; ++bh)
        {
            size_t bot_index = bot_ncdh + wstart * args.bot_w_stride;
            for(uint32_t bw = wstart; bw < wend; ++bw)
            {
#if AVERAGE_OPS
                res += static_cast<_FLOAT_ACCUM>(NATIVE_CAST(bot_ptr[bot_index]));
#else // MAX
                auto val = static_cast<_FLOAT_ACCUM>(NATIVE_CAST(bot_ptr[bot_index]));
                if(val > res)
                {
                    res = val;
                    if(save_index)
                    {
                        found  = true;
                        d_save = bd;
                        h_save = bh;
                        w_save = bw;
                        saved_index = bot_index;
                    }
                }
#endif
                bot_index += args.bot_w_stride;
            }
            bot_ncdh += args.bot_h_stride;
        }
        bot_ncd += args.bot_d_stride;
    }

#if AVERAGE_OPS
    res /= static_cast<_FLOAT_ACCUM>(pool_size);
#else // MAX
    if(save_index)
    {
        index_t res_index = saved_index;

        /// Preventing overflow during computation of res_index:
        /// If Index is shorter than uint, then let's perform computation in 32-bit
        /// domain and then convert to narrower Index. That would reduce the probability of
        /// overflow. If Index is wider then 32 bits, then it seems like it is better to
        /// convert to Index type before multiplication. However this is not actually
        /// necessary, see \ref multiply_dims_overflow_assumption. Let's always compute in
        /// 32 bits and then convert.

        if(found)
        {
            if(index_mode == 0)
                res_index = (index_t)(                                                    //
                    ((d_save - td * args.filter_d_stride + args.filter_d_pad) * args.filter_h * args.filter_w) + //
                    ((h_save - th * args.filter_h_stride + args.filter_h_pad) * args.filter_w) +         //
                    (w_save - tw * args.filter_w_stride + args.filter_w_pad)                       //
                );
        }

        const size_t mask_index = nn * args.mask_n_stride               //
                                    + cc * args.mask_c_stride           //
                                    + (size_t)(td * args.mask_d_stride) //
                                    + (size_t)(th * args.mask_h_stride) //
                                    + (size_t)(tw * args.mask_w_stride);
        mask_ptr[mask_index] = res_index;
    }
#endif

    top_ptr[top_index] = NATIVE_UNCAST(res);
}

extern "C" __global__ void mloPoolingForwardNDNhwcNaive(
                                    const INPUT_TYPE* __restrict__ bot_ptr,
                                    INPUT_TYPE* __restrict__ top_ptr,
                                    index_t* __restrict__ mask_ptr,
                                    int save_index,
                                    int index_mode,
                                    poolingNdNhwcArgs args
)
{
    poolingForwardNDNhwcNaive<INPUT_TYPE>(
        bot_ptr,
        top_ptr,
        mask_ptr,
        save_index,
        index_mode,
        args
    );
}
