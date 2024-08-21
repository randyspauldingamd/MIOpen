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

#define doUU 0
#if doUU
#define UU  __attribute__((__unused__))
#else
#define UU
#endif

#define doUU1 0
#if doUU1
#define UU1  __attribute__((__unused__))
#else
#define UU1
#endif

// Out N, D, H are encoded into the block indices x, y, z
// No 2D-only optimization.
template <typename TI, typename TO>
__device__ void poolingForwardNDNhwcNaive(UU1 const TI* __restrict__ bot_ptr,
                                    TO* __restrict__ top_ptr,
                                    UU1 TO* __restrict__ junk_ptr,  // TEMPCODE RJS
                                    UU1 ARG_UNUSED_FOR_AVERAGE index_t* __restrict__ mask_ptr,
                                    UU1 ARG_UNUSED_FOR_AVERAGE int save_index,
                                    UU1 ARG_UNUSED_FOR_AVERAGE int index_mode,
                                    UU uint32_t filter_d,
                                    UU uint32_t filter_h,
                                    UU uint32_t filter_w,
                                    UU uint32_t filter_d_stride,
                                    UU uint32_t filter_h_stride,
                                    UU uint32_t filter_w_stride,
                                    UU uint32_t filter_d_pad,
                                    UU uint32_t filter_h_pad,
                                    UU uint32_t filter_w_pad,
                                    uint32_t all_n,
                                    UU uint32_t all_c, // TEMPCODE RJS
                                    UU uint32_t bot_d,
                                    UU uint32_t bot_h,
                                    UU uint32_t bot_w,
                                    UU size_t bot_n_stride,
                                    UU uint32_t bot_c_stride,
                                    UU size_t bot_d_stride,
                                    UU uint32_t bot_h_stride,
                                    UU uint32_t bot_w_stride,
                                    uint32_t top_d,
                                    uint32_t top_h,
                                    uint32_t top_w,
                                    size_t top_n_stride,
                                    uint32_t top_c_stride,
                                    size_t top_d_stride,
                                    uint32_t top_h_stride,
                                    uint32_t top_w_stride,
                                    UU ARG_UNUSED_FOR_AVERAGE size_t mask_n_stride,
                                    UU ARG_UNUSED_FOR_AVERAGE uint32_t mask_c_stride,
                                    UU ARG_UNUSED_FOR_AVERAGE uint32_t mask_d_stride,
                                    UU ARG_UNUSED_FOR_AVERAGE uint32_t mask_h_stride,
                                    UU ARG_UNUSED_FOR_AVERAGE size_t mask_w_stride)
{

    auto log_ptr = junk_ptr;
    if(nn == 0 && td == 0 && th == 0 && tw == 0)
    {
        int idx = 0;
        log_ptr[idx++] = gridDim.x;
        log_ptr[idx++] = gridDim.y;
        log_ptr[idx++] = gridDim.z;
        log_ptr[idx++] = -9;

        log_ptr[idx++] = blockDim.x;
        log_ptr[idx++] = blockDim.y;
        log_ptr[idx++] = blockDim.z;
        log_ptr[idx++] = -8;

        log_ptr[idx++] = filter_d;
        log_ptr[idx++] = filter_h;
        log_ptr[idx++] = filter_w;
        log_ptr[idx++] = -7;

        log_ptr[idx++] = filter_d_stride;
        log_ptr[idx++] = filter_h_stride;
        log_ptr[idx++] = filter_w_stride;
        log_ptr[idx++] = -6;

        log_ptr[idx++] = filter_d_pad;
        log_ptr[idx++] = filter_h_pad;
        log_ptr[idx++] = filter_w_pad;
        log_ptr[idx++] = -5;

        log_ptr[idx++] = all_n;
        log_ptr[idx++] = all_c;
        log_ptr[idx++] = bot_n_stride;
        log_ptr[idx++] = bot_c_stride;

        log_ptr[idx++] = top_n_stride;
        log_ptr[idx++] = top_c_stride;
        #if AVERAGE_OPS
        log_ptr[idx++] = -4;
        log_ptr[idx++] = -4;
        #else
        log_ptr[idx++] = mask_n_stride;
        log_ptr[idx++] = mask_c_stride;
        #endif

        log_ptr[idx++] = bot_d;
        log_ptr[idx++] = bot_h;
        log_ptr[idx++] = bot_w;
        log_ptr[idx++] = -3;

        log_ptr[idx++] = bot_d_stride;
        log_ptr[idx++] = bot_h_stride;
        log_ptr[idx++] = bot_w_stride;
        log_ptr[idx++] = -2;

        log_ptr[idx++] = top_d;
        log_ptr[idx++] = top_h;
        log_ptr[idx++] = top_w;
        log_ptr[idx++] = -1;
    
        log_ptr[idx++] = top_d_stride;
        log_ptr[idx++] = top_h_stride;
        log_ptr[idx++] = top_w_stride;
        log_ptr[idx++] = -9;

        #if AVERAGE_OPS
        log_ptr[idx++] = -8;
        log_ptr[idx++] = -8;
        log_ptr[idx++] = -8;
        #else
        log_ptr[idx++] = mask_d_stride;
        log_ptr[idx++] = mask_h_stride;
        log_ptr[idx++] = mask_w_stride;
        #endif
        log_ptr[idx++] = -7;
    }
    const uint32_t nn = blockIdx.x / top_d;                          // N=slow index
    if(nn >= all_n)
        return;

    const uint32_t td = blockIdx.x % top_d;                          // top D=fast index
    if(td >= top_d)
        return;

    const uint32_t th = blockIdx.y;  // top H
    const uint32_t j = (gridDim.y == 1) ? threadIdx.y : blockIdx.y;  // top H
    if(th >= top_h)
        return;

    const uint32_t tw = blockIdx.z % top_w;  // top W=fast index
    if(tw >= top_w)
        return;
if(true) {
    uint32_t cc = 0;
            size_t top_index = 
                    nn * top_n_stride             // TEMPCODE RJS
                    + cc * top_c_stride           //
                    + (size_t)(td * top_d_stride) //
                    + (size_t)(th * top_h_stride) //
                    + (size_t)(tw * top_w_stride);

        junk_ptr[top_index] = top_index;
}
if(true) {  // TEMPCODE RJS
    const auto int_dstart   = static_cast<int64_t>(td * filter_d_stride) - static_cast<int64_t>(filter_d_pad);
    const auto dend           = static_cast<size_t>(min(int_dstart + static_cast<int64_t>(filter_d), static_cast<int64_t>(bot_d)));
    const auto dstart         = static_cast<size_t>(max(int_dstart, 0));

    const auto int_hstart   = static_cast<int>(th * filter_h_stride) - static_cast<int>(filter_h_pad);
    const auto hend             = static_cast<uint32_t>(min(int_hstart + static_cast<int>(filter_h), static_cast<int>(bot_h)));
    const auto hstart         = static_cast<uint32_t>(max(int_hstart, 0));

    const auto int_wstart        = static_cast<int>(tw * filter_w_stride) - static_cast<int>(filter_w_pad);
    const auto wend             = static_cast<uint32_t>(min(int_wstart + static_cast<int>(filter_w), static_cast<int>(bot_w)));
    const auto wstart           = static_cast<uint32_t>(max(int_wstart, 0));
    // const auto c_base = (blockDim.x == all_c) ? 0 : (blockIdx.z / top_w) * blockDim.x;

    for(uint32_t cc = 0; cc < 1; ++cc)  // top C loop
    {
        if(cc >= all_c)   return;

#if MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE
        uint32_t pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
        pool_size       = (pool_size == 0) ? 1 : pool_size;
#elif MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE_INCLUSIVE
        const uint32_t pool_size = filter_d * filter_h * filter_w;
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
                    const size_t bot_index = nn * bot_n_stride +           //
                                            cc * bot_c_stride +           //
                                            bd * bot_d_stride + //
                                            static_cast<size_t>(bh * bot_h_stride) + //
                                            static_cast<size_t>(bw * bot_w_stride);
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
                    res_index = (index_t)(d_save * bot_h * bot_w //
                                            + h_save * bot_w       //
                                            + w_save);
                else
                    res_index = (index_t)(                                                    //
                        ((d_save - td * filter_d_stride + filter_d_pad) * filter_h * filter_w) //
                        + ((h_save - th * filter_h_stride + filter_h_pad) * filter_w)          //
                        + (w_save - tw * filter_w_stride + filter_w_pad)                       //
                    );
            }

            const size_t mask_index = nn * mask_n_stride             //
                                        + cc * mask_c_stride           //
                                        + (size_t)(td * mask_d_stride) //
                                        + (size_t)(tw * mask_h_stride) //
                                        + (size_t)(th * mask_w_stride);
            mask_ptr[mask_index] = res_index;
        }
#endif
        size_t top_index = nn * top_n_stride             //
                                + cc * top_c_stride           //
                                + (size_t)(td * top_d_stride) //
                                + (size_t)(th * top_h_stride) //
                                + (size_t)(tw * top_w_stride);

        top_ptr[top_index] = (_FLOAT)res;    // TEMPCODE RJS
    }
} // TEMPCODE
}

extern "C" __global__ void mloPoolingForwardNDNhwcNaive(
                                    const INPUT_TYPE* __restrict__ bot_ptr,
                                    OUTPUT_TYPE* __restrict__ top_ptr,
                                    OUTPUT_TYPE* __restrict__ junk_ptr,    // TEMPCODE RJS
                                    index_t* __restrict__ mask_ptr,
                                    int save_index,
                                    int index_mode,
                                    uint32_t filter_d, uint32_t filter_h, uint32_t filter_w,
                                    uint32_t filter_d_stride, uint32_t filter_h_stride, uint32_t filter_w_stride,
                                    uint32_t filter_d_pad, uint32_t filter_h_pad, uint32_t filter_w_pad,
                                    uint32_t all_n,
                                    uint32_t all_c,
                                    uint32_t bot_d, uint32_t bot_h, uint32_t bot_w,
                                    size_t bot_n_stride, uint32_t bot_c_stride, size_t bot_d_stride, uint32_t bot_h_stride, uint32_t bot_w_stride,
                                    uint32_t top_d, uint32_t top_h, uint32_t top_w,
                                    size_t top_n_stride, uint32_t top_c_stride, size_t top_d_stride, uint32_t top_h_stride, uint32_t top_w_stride,
                                    size_t mask_n_stride, size_t mask_c_stride, uint32_t mask_d_stride, uint32_t mask_h_stride, uint32_t mask_w_stride)
{
    poolingForwardNDNhwcNaive<INPUT_TYPE, OUTPUT_TYPE>(
        bot_ptr,
        top_ptr,
        junk_ptr,
        mask_ptr,
        save_index,
        index_mode,
        filter_d, filter_h, filter_w,
        filter_d_stride, filter_h_stride, filter_w_stride,
        filter_d_pad, filter_h_pad, filter_w_pad,
        all_n,
        all_c,
        bot_d, bot_h, bot_w,
        bot_n_stride, bot_c_stride, bot_d_stride, bot_h_stride, bot_w_stride,
        top_d, top_h, top_w,
        top_n_stride, top_c_stride, top_d_stride, top_h_stride, top_w_stride,
        mask_n_stride, mask_c_stride, mask_d_stride, mask_h_stride, mask_w_stride
    );
}
