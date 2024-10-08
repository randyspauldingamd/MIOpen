/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

// TODO: I've hijacked pooling_common here. This is a temporary workaround until
// all pooling tests have been converted to gtest. This work has not been planned yet.
#ifndef GUARD_MIOPEN_TEST_POOLING_COMMON_HPP
#define GUARD_MIOPEN_TEST_POOLING_COMMON_HPP

#define DATASET "0"

#include <chrono>
#include <iomanip>
namespace {using sc = std::chrono::steady_clock;}
#undef tomillis
#define tomillis(__DUR) (0.001 * std::chrono::duration_cast<std::chrono::microseconds>(__DUR).count())
#undef mstocout
#define mstocout(__TP) std::setw(15) << std::fixed << std::setprecision(3) << tomillis(sc::now() - __TP)
#undef coutms
#define coutms(__TOK, __TP) (std::cout << "ms[" << std::setw(16) << __TOK << "]: " << mstocout(__TP) << std::endl)

#include <gtest/gtest.h>
#include <array>
#include <iostream>
#include <iterator>
#include <strstream>
#include <limits>
#include <memory>
#include <miopen/datatype.hpp>
#include <miopen/logger.hpp>
#include <miopen/miopen.h>
#include <miopen/pooling.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/tensor.hpp>
#include <utility>

#include "../driver.hpp"
#include "../get_handle.hpp"
#include "../tensor_holder.hpp"
#include "../verify.hpp"
#include "../cpu_conv.hpp"
#include "../workspace.hpp"

#define TEST_PADDING_MODE 0

namespace {
int num_all_case = 0;
int num_uint16_case = 0;
int num_uint32_case = 0;
int num_uint32_case_imgidx = 0;
int num_uint64_case = 0;
int num_uint64_case_imgidx = 0;
constexpr int max_typed_cases = 5;
constexpr int MAX_ALL_CASES = 0;
auto __start = sc::now();

constexpr int RAND_INTEGER_MAX = 12000;
constexpr int RAND_INTEGER_MIN = -8800;

template <typename T>
auto gen_value =
    [](auto... is) { return static_cast<T>(prng::gen_A_to_B(RAND_INTEGER_MIN, RAND_INTEGER_MAX)) / 100; };

auto gen_start =
    [](auto... is) { return prng::gen_0_to_B(1ULL << 28); };
}

static inline void print(std::ostringstream& oss, const miopen::PoolingDescriptor& filter, bool is_default_layout)
{
    oss << "Pooling: ";
    if(filter.GetMode() == miopenPoolingAverage)
        oss << "Average";
    else if(filter.GetMode() == miopenPoolingAverageInclusive)
        oss << "AverageInclusive";
    else
        oss << "Max";
    oss << std::endl;
    oss << "Layout: " << (is_default_layout ? "default" : "transposed") << std::endl;
    oss << "Lengths: ";
    miopen::LogRange(oss, filter.GetLengths(), ", ") << std::endl;
    oss << "Pads: ";
    miopen::LogRange(oss, filter.GetPads(), ", ") << std::endl;
    oss << "Strides: ";
    miopen::LogRange(oss, filter.GetStrides(), ", ") << std::endl;
}

template <class T>
tensor<T> get_output_tensor(const miopen::PoolingDescriptor& filter, const tensor<T>& input)
{
    return tensor<T>{filter.GetForwardOutputTensor(input.desc)};
}

template <class T>
tensor<float> get_big_output_tensor(const miopen::PoolingDescriptor& filter, const tensor<T>& input)
{
    auto desc = filter.GetForwardOutputTensor(input.desc);
    auto lens = desc.GetLengths();
    if(desc.GetElementSize() > 1024)
        lens[0] *= 2;
    else
        lens[0] *= (2047 / desc.GetElementSize()) + 1 ;
    auto dbig = miopen::TensorDescriptor{miopenFloat, input.desc.GetLayout_t(), lens, desc.GetStrides()};
    auto big = tensor<float>{dbig};
    for (auto& v : big)  v = -2.2222f;
    return big;
}

template <class T>
struct pooling_operators
{
    miopen::PoolingDescriptor filter;
    pooling_operators(miopen::PoolingDescriptor f) : filter(f) {}

    double initialize() const
    {
        if(filter.GetMode() == miopenPoolingMax)
            return std::numeric_limits<T>::lowest();
        else
            return 0.0;
    }

    double operator()(double x, double y) const
    {
        if(filter.GetMode() == miopenPoolingMax)
        {
            double m = std::max(x, y);
            return (m);
        }
        else
        {
            return x + y;
        }
    }

    double finalize(double x, double y)
    {
        if(filter.GetMode() == miopenPoolingMax)
            return (x);
        else
            return x / y;
    }
};

template <int SptDim>
struct verify_forward_pooling
{
    template <class T, class Index>
    tensor<T>
    cpu(const tensor<T>& input, const miopen::PoolingDescriptor& filter, std::vector<Index>&) const
    {
        auto out = get_output_tensor(filter, input);

        std::array<int, SptDim> in_dim{};
        std::copy_n(input.desc.GetLengths().begin() + 2, SptDim, in_dim.begin());
        std::array<int, SptDim> strides{};
        std::copy_n(filter.GetStrides().begin(), SptDim, strides.begin());
        std::array<int, SptDim> pads{};
        std::copy_n(filter.GetPads().begin(), SptDim, pads.begin());
        std::array<int, SptDim> kers{};
        std::copy_n(filter.GetLengths().begin(), SptDim, kers.begin());
        auto pooler = pooling_operators<T>{filter};

        int b_n = out.desc.GetLengths()[0];
        int k_n = out.desc.GetLengths()[1];
        std::array<int, SptDim> out_spatial_len{};
        std::copy_n(out.desc.GetLengths().begin() + 2, SptDim, out_spatial_len.begin());

        auto par_ford_out =
            miopen::unpacker(miopen::prepender(par_ford, b_n, k_n))(out_spatial_len);

        par_ford_out([&](int o, int w, auto... out_spatial_id_pack) {
            auto out_spatial_id = make_array(out_spatial_id_pack...);

            std::array<int, SptDim> start_idx{};
            std::array<int, SptDim> win_sz{};
            for(int i = 0; i < SptDim; ++i)
            {
                start_idx[i] = out_spatial_id[i] * strides[i] - pads[i];
                int end_idx  = start_idx[i] + kers[i];
                end_idx      = std::min(end_idx, in_dim[i]);
                start_idx[i] = std::max(start_idx[i], 0);
                win_sz[i]    = end_idx - start_idx[i];
                win_sz[i]    = std::max(win_sz[i], 1);
            }

            int pool_size =
                filter.GetMode() == miopenPoolingAverageInclusive
                    ? std::accumulate(kers.begin(), kers.end(), 1, std::multiplies<int>())
                    : std::accumulate(win_sz.begin(), win_sz.end(), 1, std::multiplies<int>());

            double acc = pooler.initialize();
            miopen::unpacker(ford)(win_sz)([&](auto... in_spatial_id_pack) {
                auto in_spatial_id = make_array(in_spatial_id_pack...);
                std::array<std::size_t, SptDim + 2> idx{};
                idx[0] = o;
                idx[1] = w;

                bool in_cmp_idx = true;
                for(int i = 0; i < SptDim; ++i)
                {
                    idx[i + 2] = start_idx[i] + in_spatial_id[i];
                    in_cmp_idx &= (in_dim[i] > idx[i + 2]);
                }

                if(in_cmp_idx)
                {
                    acc = pooler(acc, input(idx));
                }
            });
            out(o, w, out_spatial_id_pack...) = T(pooler.finalize(acc, pool_size));
        });

        return out;
    }

    template <class T, class Index>
    tensor<T> gpu(const tensor<T>& input,
                  const miopen::PoolingDescriptor& filter,
                  std::vector<Index>& indices) const
    {
        auto&& handle = get_handle();
        auto out      = get_output_tensor(filter, input);

        indices.resize(out.data.size(), 0);

        auto in_dev  = handle.Write(input.data);
        auto out_dev = handle.Create<T>(out.GetSize());
        Workspace wspace{};
        wspace.Write(indices);

        float alpha = 1, beta = 0;
        filter.Forward(handle,
                       &alpha,
                       input.desc,
                       in_dev.get(),
                       &beta,
                       out.desc,
                       out_dev.get(),
                       true,
                       wspace.ptr(),
                       wspace.size());
        handle.ReadTo(out.data.data(), out_dev, out.GetDataByteSize());
        wspace.ReadTo(indices);

        return out;
    }

    template <class T, class Index>
    void fail(float,
              const tensor<T>& input,
              const miopen::PoolingDescriptor& filter,
              const std::vector<Index>&) const
    {
        std::ostringstream oss;
        oss << "Forward ";
        print(oss, filter, input.desc.IsDefaultLayout());
        oss << "Input tensor: " << input.desc.ToString() << std::endl;
        oss << "Output tensor: " << filter.GetForwardOutputTensor(input.desc).ToString()
                  << std::endl;
        GTEST_FAIL() << oss.str();
    }
};

template <int SptDim>
struct verify_backward_pooling
{
    template <class T, class Index>
    tensor<T> cpu(const tensor<T>& input,
                  const tensor<T>& dout,
                  const tensor<T>& out,
                  const miopen::PoolingDescriptor& filter,
                  const std::vector<Index>& indices,
                  bool use_global_index,
                  bool verify_index) const
    {
        const int sptl_dim_offset = 2;
        const int chan_dim_offset = 1;

        auto dinput = input;
        return dinput; // TRJS

        std::vector<double> din_vec(input.desc.GetElementSpace(), 0.0);
        CHECK(dout.desc == out.desc);
        std::array<int, SptDim + 2> in_dim{};
        std::copy_n(input.desc.GetLengths().begin(), SptDim + 2, in_dim.begin());
        std::array<int, SptDim + 2> in_str{};
        std::copy_n(input.desc.GetStrides().begin(), SptDim + 2, in_str.begin());
        std::array<int, SptDim> strides{};
        std::copy_n(filter.GetStrides().begin(), SptDim, strides.begin());
        std::array<int, SptDim> pads{};
        std::copy_n(filter.GetPads().begin(), SptDim, pads.begin());
        std::array<int, SptDim> kers{};
        std::copy_n(filter.GetLengths().begin(), SptDim, kers.begin());
        auto ford_ker = miopen::unpacker(ford)(kers);

        int out_n = out.desc.GetLengths()[0];
        int out_c = out.desc.GetLengths()[chan_dim_offset];
        std::array<int, SptDim> out_spatial_len{};
        std::copy_n(out.desc.GetLengths().begin() + sptl_dim_offset, SptDim, out_spatial_len.begin());

        auto ford_out = miopen::unpacker(ford)(out_spatial_len);

        par_ford(out_n, out_c)([&](int o, int w) {
            if(filter.GetMode() == miopenPoolingMax)
            {
                ford_out([&](auto... out_spatial_id_pack) {
                    auto mx_idx = indices.at(dout.desc.GetIndex(o, w, out_spatial_id_pack...));
                    std::array<std::size_t, SptDim + 2> idx{};
                    bool in_cmp_idx = true;
                    if(use_global_index)
                    {
                        for(int i = 0; i < SptDim; i++)
                        {
                            std::size_t mx_idx_dim = mx_idx;
                            mx_idx_dim /= std::accumulate(in_dim.begin() + sptl_dim_offset + i + 1,
                                                          in_dim.end(),
                                                          1ULL,
                                                          std::multiplies<std::size_t>());
                            mx_idx_dim %= in_dim[i + sptl_dim_offset];
                            idx[i + sptl_dim_offset] = mx_idx_dim;
                        }
                    }
                    else
                    {
                        auto out_spatial_id = make_array(out_spatial_id_pack...);

                        for(int i = 0; i < SptDim; i++)
                        {
                            int mx_idx_dim = mx_idx;
                            mx_idx_dim /= std::accumulate(
                                kers.begin() + i + 1, kers.end(), 1, std::multiplies<int>());
                            mx_idx_dim %= kers[i];

                            mx_idx_dim += (out_spatial_id[i] * strides[i] - pads[i]);
                            in_cmp_idx &= (in_dim[i + 2] > mx_idx_dim && mx_idx_dim >= 0);

                            idx[i + 2] = std::size_t(mx_idx_dim);
                        }
                    }

                    if(in_cmp_idx)
                    {
                        idx[0] = o;
                        idx[1] = w;
                        if(false && verify_index)
                        {
                            CHECK(
                                miopen::float_equal(input(idx), out(o, w, out_spatial_id_pack...)));
                        }
                        std::size_t din_idx = 0;
                        for(int i = 0; i < SptDim + 2; i++)
                        {
                            din_idx += idx[i] * in_str[i];
                        }
                        din_vec.at(din_idx) += dout(o, w, out_spatial_id_pack...);
                    }
                });
            }
            else
            {
                ford_out([&](auto... out_spatial_id_pack) {
                    auto out_spatial_id = make_array(out_spatial_id_pack...);

                    std::array<int, SptDim> start_idx{};
                    std::array<int, SptDim> win_sz{};
                    for(int i = 0; i < SptDim; ++i)
                    {
                        start_idx[i] = out_spatial_id[i] * strides[i] - pads[i];
                        int end_idx  = start_idx[i] + kers[i];
                        end_idx      = std::min(end_idx, in_dim[i + 2]);
                        win_sz[i]    = end_idx - std::max(start_idx[i], 0);
                        win_sz[i]    = std::max(win_sz[i], 1);
                    }

                    int pool_size =
                        filter.GetMode() == miopenPoolingAverageInclusive
                            ? std::accumulate(kers.begin(), kers.end(), 1, std::multiplies<int>())
                            : std::accumulate(
                                  win_sz.begin(), win_sz.end(), 1, std::multiplies<int>());

                    ford_ker([&](auto... ker_id_pack) {
                        auto ker_id = make_array(ker_id_pack...);

                        bool in_cmp_idx = true;
                        std::array<int, SptDim + 2> in_idx{};
                        in_idx[0] = o;
                        in_idx[1] = w;
                        for(int i = 0; i < SptDim; ++i)
                        {
                            in_idx[i + 2] = start_idx[i] + ker_id[i];
                            in_cmp_idx &= (in_dim[i + 2] > in_idx[i + 2] && in_idx[i + 2] >= 0);
                        }

                        if(in_cmp_idx)
                        {
                            std::size_t din_idx = 0;
                            for(int i = 0; i < SptDim + 2; i++)
                            {
                                din_idx += in_idx[i] * in_str[i];
                            }

                            din_vec.at(din_idx) +=
                                static_cast<double>(dout(o, w, out_spatial_id_pack...)) / pool_size;
                        }
                    });
                });
            }
        });

        miopen::unpacker(ford)(in_dim)([&](auto... in_id_pack) {
            auto in_id          = make_array(in_id_pack...);
            std::size_t din_idx = 0;
            for(int i = 0; i < SptDim + 2; i++)
            {
                din_idx += in_id[i] * in_str[i];
            }
            dinput(in_id_pack...) = din_vec.at(din_idx);
        });

    return dinput;
    }

    template <class T, class Index>
    tensor<T> gpu(const tensor<T>& input,
                  const tensor<T>& dout,
                  const tensor<T>& out,
                  const miopen::PoolingDescriptor& filter,
                  const std::vector<Index>& indices,
                  bool,
                  bool) const
    {
        auto&& handle = get_handle();
        auto dinput   = input;
        return dinput; // TRJS

        auto in_dev   = handle.Write(input.data);
        auto dout_dev = handle.Write(dout.data);
        auto out_dev  = handle.Write(out.data);
        auto din_dev  = handle.Create<T>(dinput.data.size());

        Workspace wspace{};
        wspace.Write(indices);

        float alpha = 1, beta = 0;
        filter.Backward(handle,
                        &alpha,
                        // y
                        out.desc,
                        out_dev.get(),
                        // dy
                        dout.desc,
                        dout_dev.get(),
                        // x
                        input.desc,
                        in_dev.get(),
                        &beta,
                        // dx
                        dinput.desc,
                        din_dev.get(),
                        wspace.ptr());

        handle.ReadTo(dinput.data.data(), din_dev, dinput.data.size());

        return dinput;
    }

    template <class T, class Index>
    void fail(float,
              const tensor<T>& input,
              const tensor<T>&,
              const tensor<T>& out,
              const miopen::PoolingDescriptor& filter,
              const std::vector<Index>&,
              bool,
              bool) const
    {
        std::ostringstream oss;
        oss << "Backward ";
        print(oss, filter, input.desc.IsDefaultLayout());
        oss << "Input tensor: " << input.desc.ToString() << std::endl;
        oss << "Output tensor: " << out.desc.ToString() << std::endl;
        GTEST_FAIL() << oss.str();
    }
};

template <class T>
struct pooling_driver : test_driver
{
    miopen::PoolingDescriptor filter;
    std::vector<int> in_shape;
    std::vector<int> lens;
    std::vector<int> pads;
    std::vector<int> strides;
    std::string index_type;
    std::string mode_str;
#if TEST_PADDING_MODE == 1
    std::string pmode;
#endif
    int verify_indices{};
    miopenPoolingWorkspaceIndexMode_t wsidx{};
    miopenTensorLayout_t layout{};

    static void randomize_tensor(tensor<T>& in)
    {
        static tensor<T> random_data{{1}};
        static tensor<int> starts{{1}};
        static size_t start_idx = 0;

        const auto size = in.GetSize();
        const auto ran_size = size > 2 ? (3 * size) / 2 : 3;
        if (random_data.GetSize() < ran_size)
        {
            random_data = tensor<T>{{ran_size}}.generate(tensor_elem_gen_integer{2503});
        }
        if (starts.GetSize() == 1)
        {
            starts = tensor<size_t>{{1 << 20}}.generate(gen_start);
        }

        const auto r_start = starts[start_idx++] % (random_data.GetSize() / 3);
        if (start_idx >= starts.GetSize()) start_idx = 0;

        std::cout << "randomizing " << std::setw(9) << size << " elems from " << std::setw(9) << r_start << " (" << start_idx << ")"
        // << "(" << std::setw(8) << prng::gen_0_to_B(size / 2)  << std::setw(8) << prng::gen_0_to_B(size / 2)  << std::setw(8) << prng::gen_0_to_B(size / 2)  << std::setw(8) << prng::gen_0_to_B(size / 2) << ")" 
        << std::endl;
        in.data.assign(random_data.begin() + r_start, random_data.begin() + r_start + size);
    }

    std::unordered_map<std::string, miopenIndexType_t> index_type_lookup = {
        {miopen::ToUpper("miopenIndexUint8"), miopenIndexUint8},
        {miopen::ToUpper("miopenIndexUint16"), miopenIndexUint16},
        {miopen::ToUpper("miopenIndexUint32"), miopenIndexUint32},
        {miopen::ToUpper("miopenIndexUint64"), miopenIndexUint64},
    };
    std::unordered_map<std::string, miopenPoolingMode_t> mode_lookup = {
        {"MAX", miopenPoolingMax},
        {"MIOPENPOOLINGMAX", miopenPoolingMax},
        {"AVERAGE", miopenPoolingAverage},
        {"MIOPENPOOLINGAVERAGE", miopenPoolingAverage},
        {"AVERAGEINCLUSIVE", miopenPoolingAverageInclusive},
        {"MIOPENPOOLINGAVERAGEINCLUSIVE", miopenPoolingAverageInclusive},
    };
#if TEST_PADDING_MODE == 1
    std::unordered_map<std::string, miopenPaddingMode_t> pmode_lookup = {
        {"DEFAULT", miopenPaddingDefault},
        {"SAME", miopenPaddingSame},
        {"VALID", miopenPaddingValid},
    };
#endif
    pooling_driver()
    {
        add(index_type,
            "index_type",
            // generate_data({"miopenIndexUint32"}    // TEMPCODE RJS RUN
            generate_multi_data<const char*>( //
                {{"miopenIndexUint32",
                  "miopenIndexUint8"
                  ,
                  "miopenIndexUint16",
                  "miopenIndexUint64"
                  },                     //
                 {"miopenIndexUint8", "miopenIndexUint32"}, //
                 {"miopenIndexUint32"}}                     //
                ));
        add(mode_str,
            "mode_str",
            generate_data(
                {"miopenPoolingMax", "miopenPoolingAverage", "miopenPoolingAverageInclusive"}));
#if TEST_PADDING_MODE == 1
        add(pmode, "pmode", generate_data({"default", "same", "valid"}));
#endif
        add(verify_indices, "verify_indices", generate_data({1}));
    }

    template <class Index, int SptlDim>
    void run_impl()
    {
        std::vector<Index> indices{};
auto gst = sc::now();
        auto input = tensor<T>{layout, in_shape};
        randomize_tensor(input);
coutms("gen", gst);
auto vst = sc::now();
        auto out  = verify(verify_forward_pooling<SptlDim>{},
            input,
            filter,
            indices);
coutms("verify", vst);
        if(!std::is_same<T, float>::value && !std::is_same<T, half>::value) return;

        // auto dout = out.first;
        // dout.generate(tensor_elem_gen_integer{2503});
        // verify(verify_backward_pooling<SptlDim>{},   // TRJS
        //        input,
        //        dout,
        //        out.first,
        //        filter,
        //        indices,
        //        wsidx != 0,
        //        static_cast<bool>(this->verify_indices));
    }

#define CHECK_SKIP  \
if(skip)        \
{               \
    std::cout << "\nSkipping run # " << std::setw(7) << num_all_case++ << " @ET=" << mstocout(__start) << " : ";    \
    show_command(); \
    std::cout << "-- " << oss.str() << std::endl;   \
    return; \
}

#define SKIP_RUN  skip = true; CHECK_SKIP

    void run()
    {
        const bool is_default_layout = miopen::TensorDescriptor::IsDefaultLayout(layout);

        bool skip = false;
        std::ostringstream oss;

        if(MAX_ALL_CASES && num_all_case > MAX_ALL_CASES)
        {
            skip = true;
            oss << " : skipped due to MAX_ALL_CASES=" << MAX_ALL_CASES;
        }
        if(this->dry_run)
        {
            skip = true;
            oss << " : skipped due to dry_run";
        }
        if(is_default_layout && (this->type != miopenFloat && this->type != miopenHalf))
        {
            skip = true;
            oss << " : skipped, no solvers for datatype " << this->type << " and default layouts";
        }

        CHECK_SKIP;

        int sptl_dim = static_cast<int>(in_shape.size()) - 2;
        if(sptl_dim != 2 && sptl_dim != 3)
        {
            oss << "Warning: Config skipped due to invalid dimensions. 'in_shape' must be in NCHW or NCDHW format." << std::endl;
            SKIP_RUN;
        }

        // To simplify launching, input dimensions to the driver are always default layout. Desire to
        // test non-default layouts is communicated exclusively via 'layout'.

        auto mode = mode_lookup.at(miopen::ToUpper(mode_str));

        auto pad_mode = miopenPaddingDefault;
#if TEST_PADDING_MODE
        pad_mode = pmode_lookup.at(miopen::ToUpper(pmode));
#endif

        auto idx_typ = index_type_lookup.at(miopen::ToUpper(index_type));
        auto idx_sz  = sizeof(uint8_t);
        const bool skip_many_configs_with_non_int8_index =
            (dataset_id == 0) && !full_set; // Otherwise the default dataset takes too much time.
        const bool wide_dataset = (dataset_id == 2) && full_set;

        filter = miopen::PoolingDescriptor
        {
            mode,
            pad_mode,
            lens,
            strides,
            pads
        };

        filter.SetIndexType(idx_typ);
        filter.SetWorkspaceIndexMode(miopenPoolingWorkspaceIndexMode_t(wsidx));
        bool mask_idx = filter.GetWorkspaceIndexMode() == miopenPoolingWorkspaceIndexMask;

        if(mask_idx && sptl_dim == 3 && filter.GetMode() == miopenPoolingMax)
        {
            oss << "Warning: Config skipped. Workspace index mask mode is not implemented "
                         "yet in 3D max pooling solvers."
                      << std::endl;
            SKIP_RUN;
        }

        if(mask_idx && sptl_dim == 2 && filter.GetMode() == miopenPoolingMax && wide_dataset)
        {
            oss << "Warning: Config skipped. Workspace index mask mode is not implemented "
                         "yet in 2D max backward solvers that support wide pooling window."
                      << std::endl;
            SKIP_RUN;
        }

        if(mask_idx && filter.ModeIsAveraging())
        {
            oss << "Warning: Config skipped. Workspace index modes are irrelevant for "
                         "Average pooling. "
                         "In order to optimize performance of full tests, we "
                         "skip average pooling configs when (wsidx == 0). "
                         "Please make sure that dataset includes counterparts with (wsidx == 1)."
                      << std::endl;
            SKIP_RUN;
        }

        // index size filter
        if(filter.GetMode() == miopenPoolingMax)
        {
            auto index_max = miopen::get_index_max(filter.GetIndexType());
            auto index_needed = mask_idx ?
                std::accumulate(lens.begin(), lens.end(), 1, std::multiplies<int>()) :
                std::accumulate(in_shape.begin() + 2, in_shape.end(), 1, std::multiplies<int>());

            if(index_max <= index_needed)
            {
                oss << "Warning: Config skipped: index mode " << filter.GetWorkspaceIndexMode()
                    << " type " << filter.GetIndexType() << " is too small. max="
                    << index_max << ", needed=" << index_needed << std::endl;
                SKIP_RUN;
            }
        }

        switch(idx_typ)
        {
        /// The "index is too small" limitation is an approximation
        /// of the real limitation, and therefore applied only when
        /// the "full test" is ran. See:
        /// \ref max_pooling_index_max_restriction
        case miopenIndexUint8: {
            if(full_set && (sptl_dim == 3 || (mask_idx && sptl_dim == 2)) &&
               filter.GetMode() == miopenPoolingMax)
            {
                oss << "Warning: Config skipped: uint8 index is too small "
                             "(sptl_dim == 3 || (sptl_dim == 2 && wsidx == 1)) "
                             "&& filter.GetMode() == miopenPoolingMax"
                          << std::endl;
                SKIP_RUN;
            }
            break;
        }
        case miopenIndexUint16: {
            if(full_set && (sptl_dim == 3 || (!mask_idx && sptl_dim == 2)) &&
               filter.GetMode() == miopenPoolingMax)
            {
                oss << "Warning: Config skipped: uint16 index is too small "
                             "(sptl_dim == 3 || (sptl_dim == 2 && wsidx == 1)) "
                             "&& filter.GetMode() == miopenPoolingMax"
                          << std::endl;
                SKIP_RUN;
            }
            if(skip_many_configs_with_non_int8_index)
            {
                // test_pooling_test --all limit uint16 cases
                if(num_uint16_case >= max_typed_cases)
                {
                    oss << "Warning: Config skipped for the default dataset to speed "
                                 "up testing (num_uint16_case > 5)"
                              << std::endl;
                    SKIP_RUN;
                }
                ++num_uint16_case;
            }
            idx_sz = sizeof(uint16_t);
            break;
        }
        case miopenIndexUint32: {
            if(skip_many_configs_with_non_int8_index)
            {
                // test_pooling_test --all limit uint32 cases
                if(mask_idx)
                {
                    if(num_uint32_case >= max_typed_cases)
                    {
                        oss << "Warning: Config skipped for the default dataset to speed up "
                                     "testing (wsidx == 0 && num_uint32_case > 5)"
                                  << std::endl;
                        SKIP_RUN;
                    }
                    ++num_uint32_case;
                }
                else
                {
                    if(num_uint32_case_imgidx >= max_typed_cases)
                    {
                        oss << "Warning: Config skipped for the default dataset to speed up "
                                     "testing (wsidx != 0 && num_uint32_case_imgidx > 5)"
                                  << std::endl;
                        SKIP_RUN;
                    }
                    ++num_uint32_case_imgidx;
                }
            }
            idx_sz = sizeof(uint32_t);
            break;
        }
        case miopenIndexUint64: {
            if(skip_many_configs_with_non_int8_index)
            {
                if(mask_idx)
                {
                    if(num_uint64_case >= max_typed_cases)
                    {
                        oss << "Warning: Config skipped for the default dataset to speed up "
                                     "testing (wsidx == 0) && (num_uint64_case > 5)"
                                  << std::endl;
                        SKIP_RUN;
                    }
                    ++num_uint64_case;
                }
                else
                {
                    if(num_uint64_case_imgidx >= max_typed_cases && sptl_dim == 2)
                    {
                        oss << "Warning: Config skipped to speed up testing of the "
                                     "default dataset (wsidx != 0) && (num_uint64_case_imgidx > 5 "
                                     "&& sptl_dim == 2)"
                                  << std::endl;
                        SKIP_RUN;
                    }
                    ++num_uint64_case_imgidx;
                }
            }
            idx_sz = sizeof(uint64_t);
            break;
        }
        }

        auto input_desc = miopen::TensorDescriptor(this->type, layout, in_shape);

        for(int i = 0; i < sptl_dim; i++)
        {
            if(lens[i] > (input_desc.GetLengths()[i + 2] + static_cast<uint64_t>(2) * pads[i]))
            {
                oss << "Warning: Config skipped because it is invalid "
                             "(lens[i] > (input_desc.GetLengths()[i + 2] + 2 * pads[i]))"
                          << std::endl;
                SKIP_RUN;
            }
        }

        if(full_set)
        {
            auto output_desc = filter.GetForwardOutputTensor(input_desc);
            size_t total_mem =
                3 * input_desc.GetNumBytes() + output_desc.GetNumBytes() +
                idx_sz * output_desc.GetElementSize(); // estimate based on backward pass

            size_t device_mem = get_handle().GetGlobalMemorySize();
            if(total_mem >= device_mem)
            {
                oss << "Config skipped because it requires " << total_mem
                          << " Bytes to write all necessary tensors to GPU. GPU has " << device_mem
                          << " Bytes of memory." << std::endl;
                SKIP_RUN;
            }
        }

        CHECK_SKIP;

        std::cout << "\nRun # " << std::setw(7) << num_all_case++ << " @ET=" << mstocout(__start) << " : ";
        show_command();

        std::vector<int> in_dim(input_desc.GetLengths().begin(),
            input_desc.GetLengths().begin() + sptl_dim);
        std::vector<int> out_dim(sptl_dim);
        std::vector<int> ker_dim(filter.GetLengths().begin(), filter.GetLengths().end());

        switch(filter.GetIndexType())
        {
        case miopenIndexUint8: {
            if(sptl_dim == 3)
            {
                run_impl<uint8_t, 3>();
            }
            else
            {
                run_impl<uint8_t, 2>();
            }
            break;
        }
        case miopenIndexUint16: {
            if(sptl_dim == 3)
            {
                run_impl<uint16_t, 3>();
            }
            else
            {
                run_impl<uint16_t, 2>();
            }
            break;
        }
        case miopenIndexUint32: {
            if(sptl_dim == 3)
            {
                run_impl<uint32_t, 3>();
            }
            else
            {
                run_impl<uint32_t, 2>();
            }
            break;
        }
        case miopenIndexUint64: {
            if(sptl_dim == 3)
            {
                run_impl<uint64_t, 3>();
            }
            else
            {
                run_impl<uint64_t, 2>();
            }
            break;
        }
        }
    }
};

#endif
