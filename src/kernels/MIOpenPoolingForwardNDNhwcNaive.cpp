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

//#define TEMPCODE RJS
#ifdef TEMPCODE
#define MIOPEN_USE_NATIVE_DATATYPE_ACCUM 0

#define MLO_POOLING_OP_ID MLO_POOLING_OP_AVE

#define MLO_POOLING_INDEX_TYPE int
#define MLO_POOLING_IS2D_KERNEL 0
#define INPUT_TYPE _FLOAT
#define OUTPUT_TYPE _FLOAT
// #define TI INPUT_TYPE
// #define TO OUTPUT_TYPE
#define CVT_FP32_2ACCUM(x) (x)
#endif

#define _FLOAT float
#define _FLOAT_ACCUM _FLOAT

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#ifdef TEMPCODE
#include "float_types.h"
#endif
#include "pooling_functions.h"
#include "poolingNdNhwcArgs.hpp"

#if(MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE) || (MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE_INCLUSIVE)
#define AVERAGE_OPS 1
#else
#define AVERAGE_OPS 0
#endif

// Let's use extended-precision accumulator only in FP16 pooling and only for averaging.
// For all other ops and datatypes, use native accumulator, i.e. treate FLOAT_ACCUM as FLOAT.
#ifndef TEMPCODE
#if !(AVERAGE_OPS && MIOPEN_USE_FP16)
#define MIOPEN_USE_NATIVE_DATATYPE_ACCUM 1
// #else
// #define MIOPEN_USE_NATIVE_DATATYPE_ACCUM 0
#endif

#include "float_types.h"
#endif // TEMPCODE

#if AVERAGE_OPS
#define ARG_UNUSED_FOR_AVERAGE __attribute__((__unused__))
#else
#define ARG_UNUSED_FOR_AVERAGE
#endif

// Out N, D, H are encoded into the block indices x, y, z
// No 2D-only optimization.
template <typename TI, typename TO>
__device__ void poolingForwardNDNhwcNaive(const TI* __restrict__ bot_ptr,
                                    TO* __restrict__ top_ptr,
                                    TO* __restrict__ junk_ptr,  // TEMPCODE RJS
                                    ARG_UNUSED_FOR_AVERAGE index_t* __restrict__ mask_ptr,
                                    ARG_UNUSED_FOR_AVERAGE int save_index,
                                    ARG_UNUSED_FOR_AVERAGE int index_mode,
                                    poolingNdNhwcArgs args
                                    // UU uint32_t filter_d,
                                    // UU uint32_t filter_h,
                                    // UU uint32_t filter_w,
                                    // UU uint32_t filter_d_stride,
                                    // UU uint32_t filter_h_stride,
                                    // UU uint32_t filter_w_stride,
                                    // UU uint32_t filter_d_pad,
                                    // UU uint32_t filter_h_pad,
                                    // UU uint32_t filter_w_pad,
                                    // uint32_t all_n,
                                    // UU uint32_t all_c, // TEMPCODE RJS
                                    // UU uint32_t bot_d,
                                    // UU uint32_t bot_h,
                                    // UU uint32_t bot_w,
                                    // UU BIGONE bot_n_stride,
                                    // UU uint32_t bot_c_stride,
                                    // UU BIGONE bot_d_stride,
                                    // UU uint32_t bot_h_stride,
                                    // UU uint32_t bot_w_stride,
                                    // uint32_t top_d,
                                    // uint32_t top_h,
                                    // uint32_t top_w,
                                    // BIGONE top_n_stride,
                                    // uint32_t top_c_stride,
                                    // BIGONE top_d_stride,
                                    // uint32_t top_h_stride,
                                    // uint32_t top_w_stride,
                                    // UU ARG_UNUSED_FOR_AVERAGE BIGONE mask_n_stride,
                                    // UU ARG_UNUSED_FOR_AVERAGE BIGONE mask_c_stride,
                                    // UU ARG_UNUSED_FOR_AVERAGE uint32_t mask_d_stride,
                                    // UU ARG_UNUSED_FOR_AVERAGE uint32_t mask_h_stride,
                                    // UU ARG_UNUSED_FOR_AVERAGE uint32_t mask_w_stride
)
{
    const uint32_t nn = blockIdx.x / args.top_d;                          // N=slow index
    const uint32_t td = blockIdx.x % args.top_d;                          // top D=fast index
    const uint32_t th = blockIdx.y;  // top H
    const uint32_t tw = blockIdx.z % args.all_c;  // top W=fast index
    const auto c_base = (blockIdx.z / args.all_c) * blockDim.x;
    if(blockDim.x > args.all_c)
    {
        // // TODO: h, w, or both may be encoded into threadIdx
        // if(top_h > 1 && blockDim.y == 1)    
    }

    auto log_ptr = junk_ptr;
    if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 &&  threadIdx.y == 0 &&  threadIdx.z == 0)
    {
        for(int i = 0; i < 320; ++i)
        {
            junk_ptr[i] = (_FLOAT)1.11111;
        }
        int idx = 0;
        log_ptr[idx++] = gridDim.x;     // ND
        log_ptr[idx++] = gridDim.y;     // H
        log_ptr[idx++] = gridDim.z;     // W (*C overflow)
        log_ptr[idx++] = -9;

        log_ptr[idx++] = blockDim.x;    // C
        log_ptr[idx++] = blockDim.y;    // small-C H
        log_ptr[idx++] = blockDim.z;    // small-C W
        log_ptr[idx++] = -8;

        log_ptr[idx++] = args.filter_d;
        log_ptr[idx++] = args.filter_h;
        log_ptr[idx++] = args.filter_w;
        log_ptr[idx++] = -7;

        log_ptr[idx++] = args.filter_d_stride;
        log_ptr[idx++] = args.filter_h_stride;
        log_ptr[idx++] = args.filter_w_stride;
        log_ptr[idx++] = -6;

        log_ptr[idx++] = args.filter_d_pad;
        log_ptr[idx++] = args.filter_h_pad;
        log_ptr[idx++] = args.filter_w_pad;
        log_ptr[idx++] = -5;

        log_ptr[idx++] = args.all_n;
        log_ptr[idx++] = args.all_c;
        log_ptr[idx++] = args.bot_n_stride;
        log_ptr[idx++] = args.bot_c_stride;

        log_ptr[idx++] = args.top_n_stride;
        log_ptr[idx++] = args.top_c_stride;
        #if AVERAGE_OPS
        log_ptr[idx++] = -4;
        log_ptr[idx++] = -4;
        #else
        log_ptr[idx++] = args.mask_n_stride;
        log_ptr[idx++] = args.mask_c_stride;
        #endif

        log_ptr[idx++] = args.bot_d;
        log_ptr[idx++] = args.bot_h;
        log_ptr[idx++] = args.bot_w;
        log_ptr[idx++] = -3;

        log_ptr[idx++] = args.bot_d_stride;
        log_ptr[idx++] = args.bot_h_stride;
        log_ptr[idx++] = args.bot_w_stride;
        log_ptr[idx++] = -2;

        log_ptr[idx++] = args.top_d;
        log_ptr[idx++] = args.top_h;
        log_ptr[idx++] = args.top_w;
        log_ptr[idx++] = -1;
    
        log_ptr[idx++] = args.top_d_stride;
        log_ptr[idx++] = args.top_h_stride;
        log_ptr[idx++] = args.top_w_stride;
        log_ptr[idx++] = -9;

        #if AVERAGE_OPS
        log_ptr[idx++] = -8;
        log_ptr[idx++] = -8;
        log_ptr[idx++] = -8;
        #else
        log_ptr[idx++] = args.mask_d_stride;
        log_ptr[idx++] = args.mask_h_stride;
        log_ptr[idx++] = args.mask_w_stride;
        #endif
        log_ptr[idx++] = -7;
        while(idx < 64) log_ptr[idx++] = (_FLOAT)0;
    }

    // if(nn >= args.all_n)
    //     return;

    // if(td >= args.top_d)
    //     return;

    // if(th >= args.top_h)
    //     return;

    // if(tw >= args.top_w)
    //     return;

if(true) {  // TEMPCODE RJS
    const auto int_dstart   = static_cast<int64_t>(td * args.filter_d_stride) - static_cast<int64_t>(args.filter_d_pad);
    const auto dend           = static_cast<size_t>(min(int_dstart + static_cast<int64_t>(args.filter_d), static_cast<int64_t>(args.bot_d)));
    const auto dstart         = static_cast<size_t>(max(int_dstart, 0));

    const auto int_hstart   = static_cast<int>(th * args.filter_h_stride) - static_cast<int>(args.filter_h_pad);
    const auto hend             = static_cast<uint32_t>(min(int_hstart + static_cast<int>(args.filter_h), static_cast<int>(args.bot_h)));
    const auto hstart         = static_cast<uint32_t>(max(int_hstart, 0));

    const auto int_wstart        = static_cast<int>(tw * args.filter_w_stride) - static_cast<int>(args.filter_w_pad);
    const auto wend             = static_cast<uint32_t>(min(int_wstart + static_cast<int>(args.filter_w), static_cast<int>(args.bot_w)));
    const auto wstart           = static_cast<uint32_t>(max(int_wstart, 0));

    uint32_t cc = c_base + threadIdx.x;
    // if(cc > args.all_c) return;

    size_t top_index = 
            nn * args.top_n_stride             // TEMPCODE RJS
            + cc * args.top_c_stride           //
            + (size_t)(td * args.top_d_stride) //
            + (size_t)(th * args.top_h_stride) //
            + (size_t)(tw * args.top_w_stride);
if(true) {
        top_ptr[top_index] = (TO)-1.11111;
        junk_ptr[64 + top_index] = top_index;
}

#if MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE
        uint32_t pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
        pool_size       = (pool_size == 0) ? 1 : pool_size;
#elif MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE_INCLUSIVE
        const uint32_t pool_size = args.filter_d * args.filter_h * args.filter_w;
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
        for(size_t bd = dstart; bd < dend; ++bd)
        {
            for(uint32_t bh = hstart; bh < hend; ++bh)
            {
                for(uint32_t bw = wstart; bw < wend; ++bw)
                {
                    const size_t bot_index = nn * args.bot_n_stride +           //
                                            cc * args.bot_c_stride +           //
                                            bd * args.bot_d_stride + //
                                            static_cast<size_t>(bh * args.bot_h_stride) + //
                                            static_cast<size_t>(bw * args.bot_w_stride);
#if AVERAGE_OPS
                    res += static_cast<_FLOAT_ACCUM>(bot_ptr[bot_index]);
#else // MAX
                    if(static_cast<_FLOAT_ACCUM>(bot_ptr[bot_index]) > res)
                    {
                        res = bot_ptr[bot_index];
                        if(save_index)
                        {
                            found  = true;
                            d_save = bd;
                            h_save = bh;
                            w_save = bw;
                        }
                    }
#endif
                }
            }
        }

#if AVERAGE_OPS
        res *= CVT_FP32_2ACCUM(1.f) / static_cast<_FLOAT_ACCUM>(pool_size);
#else // MAX
res *= 1.0; // TEMPCODE RJS fix UNUSED
        if(save_index)
        {
            index_t res_index = 0;

            // / Preventing overflow during computation of res_index:
            // / If Index is shorter than uint, then let's perform computation in 32-bit
            // / domain and then convert to narrower Index. That would reduce the probability of
            // / overflow. If Index is wider then 32 bits, then it seems like it is better to
            // / convert to Index type before multiplication. However this is not actually
            // / necessary, see \ref multiply_dims_overflow_assumption. Let's always compute in
            // / 32 bits and then convert.

            if(found)
            {
                if(index_mode == 1)
                    res_index = (index_t)(d_save * args.bot_h * args.bot_w //
                                            + h_save * args.bot_w       //
                                            + w_save);
                else
                    res_index = (index_t)(                                                    //
                        ((d_save - td * args.filter_d_stride + args.filter_d_pad) * args.filter_h * args.filter_w) //
                        + ((h_save - th * args.filter_h_stride + args.filter_h_pad) * args.filter_w)          //
                        + (w_save - tw * args.filter_w_stride + args.filter_w_pad)                       //
                    );
            }

            const size_t mask_index = nn * args.mask_n_stride             //
                                        + cc * args.mask_c_stride           //
                                        + (size_t)(td * args.mask_d_stride) //
                                        + (size_t)(tw * args.mask_h_stride) //
                                        + (size_t)(th * args.mask_w_stride);
            mask_ptr[mask_index] = res_index;
        }
#endif
        // top_index = nn * args.top_n_stride             //
        //                         + cc * args.top_c_stride           //
        //                         + (size_t)(td * args.top_d_stride) //
        //                         + (size_t)(th * args.top_h_stride) //
        //                         + (size_t)(tw * args.top_w_stride);

        top_ptr[top_index] = (_FLOAT)res;    // TEMPCODE RJS
        top_ptr[top_index] = (_FLOAT)1.11111;    // TEMPCODE RJS

        cc += blockDim.x;
} // TEMPCODE
}

extern "C" __global__ void mloPoolingForwardNDNhwcNaive(
                                    const INPUT_TYPE* __restrict__ bot_ptr,
                                    OUTPUT_TYPE* __restrict__ top_ptr,
                                    OUTPUT_TYPE* __restrict__ junk_ptr,    // TEMPCODE RJS
                                    index_t* __restrict__ mask_ptr,
                                    int save_index,
                                    int index_mode,
poolingNdNhwcArgs args
)
{
    poolingForwardNDNhwcNaive<INPUT_TYPE, OUTPUT_TYPE>(
        bot_ptr,
        top_ptr,
        junk_ptr,
        mask_ptr,
        save_index,
        index_mode,
        args
        // args.filter_d, args.filter_h, args.filter_w,
        // args.filter_d_stride, args.filter_h_stride, args.filter_w_stride,
        // args.filter_d_pad, args.filter_h_pad, args.filter_w_pad,
        // args.all_n,
        // args.all_c,
        // args.bot_d, args.bot_h, args.bot_w,
        // args.bot_n_stride, args.bot_c_stride, args.bot_d_stride, args.bot_h_stride, args.bot_w_stride,
        // args.top_d, args.top_h, args.top_w,
        // args.top_n_stride, args.top_c_stride, args.top_d_stride, args.top_h_stride, args.top_w_stride,
        // args.mask_n_stride, args.mask_c_stride, args.mask_d_stride, args.mask_h_stride, args.mask_w_stride
    );
}
