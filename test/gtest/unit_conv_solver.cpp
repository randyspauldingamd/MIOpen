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

#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/generic_search.hpp>

#include "unit_conv_solver.hpp"

#include "get_handle.hpp"
#include "conv_common.hpp"
#include "conv_tensor_gen.hpp"
#include "tensor_holder.hpp"

#include "../workspace.hpp"

MIOPEN_LIB_ENV_VAR(MIOPEN_DEBUG_ENABLE_DEPRECATED_SOLVERS)

namespace miopen {
namespace unit_tests {

namespace {

class DeprecatedSolversScopedEnabler
{
public:
    DeprecatedSolversScopedEnabler() noexcept {}
    DeprecatedSolversScopedEnabler(const DeprecatedSolversScopedEnabler&) = delete;
    DeprecatedSolversScopedEnabler(DeprecatedSolversScopedEnabler&&)      = delete;
    DeprecatedSolversScopedEnabler& operator=(const DeprecatedSolversScopedEnabler&) = delete;
    DeprecatedSolversScopedEnabler& operator=(DeprecatedSolversScopedEnabler&&) = delete;

    ~DeprecatedSolversScopedEnabler()
    {
        if(changed)
        {
            if(prev)
                lib_env::update(MIOPEN_DEBUG_ENABLE_DEPRECATED_SOLVERS, false);
            else
                lib_env::clear(MIOPEN_DEBUG_ENABLE_DEPRECATED_SOLVERS);
        }
    }

    void Enable()
    {
        if(MIOPEN_DEBUG_ENABLE_DEPRECATED_SOLVERS)
            prev = lib_env::value<bool>(MIOPEN_DEBUG_ENABLE_DEPRECATED_SOLVERS);
        if(prev != true)
        {
            lib_env::update(MIOPEN_DEBUG_ENABLE_DEPRECATED_SOLVERS, true);
            changed = true;
        }
    }

private:
    std::optional<bool> prev;
    bool changed = false;
};

class ConvAttrFp16AltScopedSetter
{
public:
    ConvAttrFp16AltScopedSetter() noexcept {}
    ConvAttrFp16AltScopedSetter(const ConvAttrFp16AltScopedSetter&) = delete;
    ConvAttrFp16AltScopedSetter(ConvAttrFp16AltScopedSetter&&)      = delete;
    ConvAttrFp16AltScopedSetter& operator=(const ConvAttrFp16AltScopedSetter&) = delete;
    ConvAttrFp16AltScopedSetter& operator=(ConvAttrFp16AltScopedSetter&&) = delete;

    ~ConvAttrFp16AltScopedSetter()
    {
        if(changed)
        {
            if(prev)
                lib_env::update(wa::MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL, prev.value());
            else
                lib_env::clear(wa::MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL);
        }
    }

    void SetValue(uint64_t value)
    {
        if(wa::MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL)
            prev = lib_env::value<uint64_t>(wa::MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL);
        if(value == prev)
            return;
        lib_env::update(wa::MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL, value);
        changed = true;
    }

private:
    std::optional<uint64_t> prev;
    bool changed = false;
};

bool IsDeviceSupported(Gpu supported_devs, Gpu dev)
{
    if((supported_devs & dev) != Gpu::None)
        return true;
    return false;
}

} // namespace

//************************************************************************************
// ConvTestCase
//************************************************************************************

ConvTestCase::ConvTestCase() : x(miopenHalf, {}), w(miopenHalf, {}), conv({}, {}, {}){};

ConvTestCase::ConvTestCase(std::vector<size_t>&& x_,
                           std::vector<size_t>&& w_,
                           std::vector<int>&& pad_,
                           std::vector<int>&& stride_,
                           std::vector<int>&& dilation_,
                           miopenDataType_t type_)
    : ConvTestCase(std::move(x_),
                   std::move(w_),
                   std::move(pad_),
                   std::move(stride_),
                   std::move(dilation_),
                   type_,
                   type_,
                   type_)
{
}

ConvTestCase::ConvTestCase(std::vector<size_t>&& x_,
                           std::vector<size_t>&& w_,
                           std::vector<int>&& pad_,
                           std::vector<int>&& stride_,
                           std::vector<int>&& dilation_,
                           miopenDataType_t type_x_,
                           miopenDataType_t type_w_,
                           miopenDataType_t type_y_)
    : ConvTestCase(
          TensorDescriptorParams{type_x_, std::move(x_)},
          TensorDescriptorParams{type_w_, std::move(w_)},
          type_y_,
          ConvolutionDescriptorParams{std::move(pad_), std::move(stride_), std::move(dilation_)})
{
}

ConvTestCase::ConvTestCase(TensorDescriptorParams&& x_,
                           TensorDescriptorParams&& w_,
                           miopenDataType_t type_y_,
                           ConvolutionDescriptorParams&& conv_)
    : x(std::move(x_)), w(std::move(w_)), type_y(type_y_), conv(std::move(conv_))
{
    const auto num_spatial_dims = conv.GetNumSpatialDims();
    const auto num_tensor_dims  = num_spatial_dims + 2;
    const auto group_count      = conv.GetGroupCount();

    if(x.GetNumDims() != num_tensor_dims || w.GetNumDims() != num_tensor_dims ||
       x.GetLens()[1] != w.GetLens()[1] * group_count)
    {
        throw std::runtime_error("wrong test case format");
    }
}

miopen::TensorDescriptor ConvTestCase::GetXTensorDescriptor() const
{
    return x.GetTensorDescriptor();
}

miopen::TensorDescriptor ConvTestCase::GetWTensorDescriptor() const
{
    return w.GetTensorDescriptor();
}

miopenDataType_t ConvTestCase::GetXDataType() const { return x.GetDataType(); }

miopenDataType_t ConvTestCase::GetWDataType() const { return w.GetDataType(); }

miopenDataType_t ConvTestCase::GetYDataType() const { return type_y; }

miopen::ConvolutionDescriptor ConvTestCase::GetConv() const
{
    return conv.GetConvolutionDescriptor();
}

miopen::conv::ProblemDescription
ConvTestCase::GetProblemDescription(miopen::conv::Direction direction) const
{
    const auto x_desc    = GetXTensorDescriptor();
    const auto w_desc    = GetWTensorDescriptor();
    const auto conv_desc = GetConv();
    const auto y_desc    = conv_desc.GetForwardOutputTensor(x_desc, w_desc, GetYDataType());

    switch(direction)
    {
    case miopen::conv::Direction::Forward:
        return miopen::conv::ProblemDescription(x_desc, w_desc, y_desc, conv_desc, direction);
    case miopen::conv::Direction::BackwardData:
    case miopen::conv::Direction::BackwardWeights:
        return miopen::conv::ProblemDescription(y_desc, w_desc, x_desc, conv_desc, direction);
    default: throw std::runtime_error("unknown direction");
    }
}

std::ostream& operator<<(std::ostream& os, const ConvTestCase& tc)
{
    os << "(";
    os << "x:(" << tc.x << "), ";
    os << "w:(" << tc.w << "), ";
    os << "type_y:" << tc.type_y << "), ";
    os << "conv:(" << tc.conv << ")";
    os << ")";
    return os;
}

//************************************************************************************
// Unit test for convolution solver
//************************************************************************************
uint64_t Tolerances::GetKey(Gpu gpu, miopenDataType_t type)
{
    static_assert(sizeof(gpu) <= sizeof(uint64_t) / 2);
    static_assert(sizeof(type) <= sizeof(uint64_t) / 2);

    return (static_cast<uint64_t>(gpu) << 32) | static_cast<uint64_t>(type);
}

void Tolerances::Set(Gpu gpu, miopenDataType_t type, float value)
{
    values[GetKey(gpu, type)] = value;
}

float Tolerances::Get(Gpu gpu, miopenDataType_t type) const
{
    const auto& v = values.find(GetKey(gpu, type));
    if(v == values.cend())
        return 1.0f; // default value
    return v->second;
}

std::ostream& operator<<(std::ostream& os, const Tolerances& t)
{
    os << "(";
    for(const auto [key, value] : t.values)
        os << std::hex << "0x" << key << std::dec << ":" << value << ",";
    os << ")";
    return os;
}

UnitTestConvSolverParams::UnitTestConvSolverParams() : UnitTestConvSolverParams(Gpu::None) {}

UnitTestConvSolverParams::UnitTestConvSolverParams(Gpu supported_devs_)
    : supported_devs(supported_devs_),
      use_cpu_ref(false),
      enable_deprecated_solvers(false),
      tunable(false),
      check_xnack_disabled(false)
{
}

void UnitTestConvSolverParams::UseCpuRef() { use_cpu_ref = true; }

void UnitTestConvSolverParams::EnableDeprecatedSolvers() { enable_deprecated_solvers = true; }

void UnitTestConvSolverParams::Tunable(std::size_t iterations_max_)
{
    tunable               = true;
    tuning_iterations_max = iterations_max_;
}

void UnitTestConvSolverParams::CheckXnackDisabled() { check_xnack_disabled = true; }

void UnitTestConvSolverParams::SetConvAttrFp16Alt(uint64_t value) { conv_attr_fp16_alt = value; }

void UnitTestConvSolverParams::SetTolerance(Gpu gpu, miopenDataType_t type, float value)
{
    tolerances.Set(gpu, type, value);
}

std::ostream& operator<<(std::ostream& os, const UnitTestConvSolverParams& p)
{
    os << "(";
    os << "Devs:" << std::hex << "0x"
       << static_cast<std::underlying_type_t<decltype(p.supported_devs)>>(p.supported_devs)
       << std::dec;
    if(p.use_cpu_ref)
        os << ", CpuRef:" << p.use_cpu_ref;
    if(p.enable_deprecated_solvers)
        os << ", EnDerpSolver:" << p.enable_deprecated_solvers;
    if(p.tunable)
        os << ", IterMax:" << p.tuning_iterations_max;
    if(p.check_xnack_disabled)
        os << ", CheckXnackOff:" << p.check_xnack_disabled;
    if(p.conv_attr_fp16_alt)
        os << ", AttrFp16Alt:" << p.conv_attr_fp16_alt.value();
    os << ", Tolerances:" << p.tolerances;
    os << ")";
    return os;
}

namespace {

miopen::solver::ConvSolution FindSolution(const miopen::solver::conv::ConvSolverInterface& solv,
                                          const UnitTestConvSolverParams& params,
                                          const miopen::ExecutionContext& ctx,
                                          const miopen::conv::ProblemDescription& problem,
                                          const AnyInvokeParams& invoke_ctx)
{
    if(params.tunable)
    {
        using IterationLimiter = miopen::solver::debug::TuningIterationScopedLimiter;
        IterationLimiter tuning_limit{params.tuning_iterations_max};
        const auto& tunable_solv =
            dynamic_cast<const miopen::solver::conv::ConvSolverInterfaceTunable&>(solv);
        return tunable_solv.FindSolutionSimple(ctx, problem, invoke_ctx);
    }
    else
    {
        const auto& non_tunable_solv =
            dynamic_cast<const miopen::solver::conv::ConvSolverInterfaceNonTunable&>(solv);
        return non_tunable_solv.GetSolution(ctx, problem);
    }
}

template <typename T>
double GetThreshold(miopenConvAlgorithm_t algo,
                    miopen::conv::Direction direction,
                    const Tolerances& tolerances)
{
    double tolerance = tolerances.Get(GetDevGpuType(), miopen_type<T>{});
    double threshold = std::numeric_limits<T>::epsilon() * tolerance;
    return threshold;
}

// TRJS
template <typename T>
void CompareData(std::vector<T>& prev_data, const std::vector<T>& data, std::string preamble)
{
    size_t diffs = 0;
    int first_diff = -1;
    for(int i = 0; i < data.size(); ++i)
    {
        if(fabs(1 - data[i] / prev_data[i]) > 1E-7) {
            ++diffs;
            if(first_diff == -1) first_diff = i;
        }
    }

    std::stringstream diff_str;
    if(diffs > 0) {
        diff_str << std::endl << "first diff = " << first_diff;
    first_diff = ((first_diff + 15) / 16) * 16;
    } else {
        diff_str << std::endl << "no diffs in dataset.";
    }
    diff_str << " Printing from " << first_diff << std::endl;
    for(int i = first_diff; i < first_diff + 16; ++i) diff_str << std::setw(16) << std::setprecision(9) << prev_data[i];
    diff_str << std::endl;
    for(int i = first_diff; i < first_diff + 16; ++i) diff_str << std::setw(16) << std::setprecision(9) << data[i];
    for(int i = first_diff; i < first_diff + 16; ++i) { auto prev = prev_data[i]; if(prev == 0.0) prev = data[i]; diff_str << std::setw(16) << std::setprecision(9) << 1 - data[i] / prev; }

    std::ofstream compare_out(preamble + "_compare.txt");
    compare_out << preamble << " : " << diffs << " diffs out of " << data.size() << " (" << (100.0 * diffs / data.size()) << "%)" << diff_str.str().c_str() << std::endl;
}

template <typename T, typename Tref>
void VerifyData(const std::vector<T>& data,
                const std::vector<Tref>& ref_data,
                miopenConvAlgorithm_t algo,
                miopen::conv::Direction direction,
                const Tolerances& tolerances)
{
    ASSERT_FALSE(miopen::range_zero(ref_data)) << "Reference data is all zeros";
    if constexpr(!std::is_integral_v<T>)
    {
        ASSERT_LT(miopen::find_idx(ref_data, miopen::not_finite), 0)
            << "Non finite number found in the reference data";
    }

    ASSERT_FALSE(miopen::range_zero(data)) << "Gpu data is all zeros";
    if constexpr(!std::is_integral_v<T>)
    {
        ASSERT_LT(miopen::find_idx(data, miopen::not_finite), 0)
            << "Non finite number found in the Gpu data";
    }

    ASSERT_EQ(miopen::range_distance(ref_data), miopen::range_distance(data));

    if constexpr(std::is_integral_v<T>)
    {
        const auto error = miopen::max_diff_v2(ref_data, data);
        static_assert(std::is_integral_v<decltype(error)>);
        ASSERT_EQ(error, 0) << "Error beyond tolerance";
    }
    else
    {
        // TRJS
        const auto datasize = data.size() * sizeof(T);
        const auto refdatasize = ref_data.size() * sizeof(Tref);
        if (std::filesystem::exists("data.bin")) {
            std::vector<T> prev_data;
            prev_data.resize(datasize);
            std::ifstream data_ifs("data.bin", std::ios::binary);
            data_ifs.read((char*)prev_data.data(), datasize);
            CompareData<T>(prev_data, data, "data");

            std::vector<Tref> prev_ref_data;
            prev_ref_data.resize(refdatasize);
            std::ifstream ref_data_ifs("ref_data.bin", std::ios::binary);
            ref_data_ifs.read((char*)prev_ref_data.data(), refdatasize);
            CompareData<Tref>(prev_ref_data, ref_data, "ref_data");
        }

        std::ofstream data_out("data.bin", std::ios::binary);
        data_out.write((const char*)data.data(), datasize);

        std::ofstream ref_data_out("ref_data.bin", std::ios::binary);
        ref_data_out.write((const char*)ref_data.data(), refdatasize);

        const auto error       = miopen::rms_range(ref_data, data);
        const double threshold = GetThreshold<T>(algo, direction, tolerances);
        ASSERT_LT(error, threshold) << "Error beyond tolerance in " << __FILE__;
        // std::cout << "error: " << error << " threshold: " << threshold << std::endl;
    }
}

//**********************************
// Fwd
//**********************************
template <typename Tin, typename Twei, typename Tout, typename Tref>
void RunSolverFwd(const miopen::solver::conv::ConvSolverInterface& solv,
                  const UnitTestConvSolverParams& params,
                  const ConvTestCase& conv_config,
                  miopenConvAlgorithm_t algo)
{
    //**********************************
    // Prepare
    //**********************************

    auto input   = tensor<Tin>{conv_config.GetXTensorDescriptor()};
    auto weights = tensor<Twei>{conv_config.GetWTensorDescriptor()};

    if(weights.desc.GetLayoutEnum() == miopenTensorCHWNc4 ||
       weights.desc.GetLayoutEnum() == miopenTensorCHWNc8)
    {
        throw std::runtime_error("GenConvData do not support CHWNc filter layout");
    }

    const auto conv_desc = conv_config.GetConv();

    const auto output_desc =
        conv_desc.GetForwardOutputTensor(input.desc, weights.desc, miopen_type<Tout>{});

    auto output = tensor<Tout>{output_desc};

    input.generate(GenConvData<Tin, Tout>{weights.desc.GetLengths()});

    // TRJS
    {
    std::ofstream input_out("input.txt");
    Tin in_max = std::numeric_limits<Tin>::min();
    Tin in_min = std::numeric_limits<Tin>::max();
    for(auto v : input) {
        float fv = fabs(v);
        if(fv > in_max) in_max = fv;
        if(fv < in_min) in_min = fv;
    }
    input_out << "input max mag: " << in_max << "  min mag: " << in_min << "  scale: " << in_max / in_min << std::endl;
    for(auto v : input) input_out << v << "  ";
    input_out << std::endl;
    }

    weights.generate(GenConvData<Twei, Tout>{weights.desc.GetLengths()});
    // TRJS
    {
    std::ofstream weights_out("weights.txt");
    Twei wei_max = std::numeric_limits<Twei>::min();
    Twei wei_min = std::numeric_limits<Twei>::max();
    for(auto v : weights) {
        float fv = fabs(v);
        if(fv > wei_max) wei_max = fv;
        if(fv < wei_min) wei_min = fv;
    }
    weights_out << "weights max mag: " << wei_max << "  min mag: " << wei_min << "  scale: " << wei_max / wei_min << std::endl;
    for(auto v : weights) weights_out << v << "  ";
    weights_out << std::endl;
    }
    std::fill(output.begin(), output.end(), Tout());

    auto&& handle = get_handle();
    auto in_dev   = handle.Write(input.data);
    auto wei_dev  = handle.Write(weights.data);
    auto out_dev  = handle.Write(output.data);

    //**********************************
    // Run solver
    //**********************************

    const auto tensors = miopen::ConvFwdTensors{
        input.desc, in_dev.get(), weights.desc, wei_dev.get(), output.desc, out_dev.get()};

    const auto problem = miopen::conv::ProblemDescription(
        input.desc, weights.desc, output.desc, conv_desc, miopen::conv::Direction::Forward);
    const auto ctx = [&] {
        auto tmp = miopen::ExecutionContext{&handle};
        problem.SetupFloats(tmp);
        return tmp;
    }();

    if(!solv.IsApplicable(ctx, problem))
    {
        // Do not put GTEST_SKIP here.
        // The usage of non-applicable config should be considered as a bug in the test.
        GTEST_FAIL();
    }

    Workspace wspace;
    if(solv.MayNeedWorkspace())
    {
        const auto cur_sol_ws = solv.GetWorkspaceSize(ctx, problem);
        wspace.resize(cur_sol_ws);
    }

    const auto invoke_params = miopen::conv::DataInvokeParams{
        tensors, wspace.ptr(), wspace.size(), conv_desc.attribute.gfx90aFp16alt.GetFwd()};

    const auto sol = FindSolution(solv, params, ctx, problem, invoke_params);
    ASSERT_TRUE(sol.Succeeded());
    ASSERT_TRUE(sol.invoker_factory);
    const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
    (invoker)(handle, invoke_params);
    handle.Finish();

    //**********************************
    // Verify
    //**********************************

    auto ref_out = tensor<Tref>{output.desc};
    if(params.use_cpu_ref)
    {
        cpu_convolution_forward(conv_desc.GetSpatialDimension(),
                                input,
                                weights,
                                ref_out,
                                conv_desc.GetConvPads(),
                                conv_desc.GetConvStrides(),
                                conv_desc.GetConvDilations(),
                                conv_desc.GetGroupCount());
    }
    else
    {
        ref_out = ref_conv_fwd(input, weights, ref_out, conv_desc);
    }

    output.data = handle.Read<Tout>(out_dev, output.data.size());

    VerifyData(
        output.data, ref_out.data, algo, miopen::conv::Direction::Forward, params.tolerances);
}

template <typename T, typename Tref>
void RunSolverFwd(const miopen::solver::conv::ConvSolverInterface& solv,
                  const UnitTestConvSolverParams& params,
                  const ConvTestCase& conv_config,
                  miopenConvAlgorithm_t algo)
{
    RunSolverFwd<T, T, T, Tref>(solv, params, conv_config, algo);
}

//**********************************
// Bwd
//**********************************
template <typename Tin, typename Twei, typename Tout, typename Tref>
void RunSolverBwd(const miopen::solver::conv::ConvSolverInterface& solv,
                  const UnitTestConvSolverParams& params,
                  const ConvTestCase& conv_config,
                  miopenConvAlgorithm_t algo)
{
    //**********************************
    // Prepare
    //**********************************

    auto input   = tensor<Tin>{conv_config.GetXTensorDescriptor()};
    auto weights = tensor<Twei>{conv_config.GetWTensorDescriptor()};

    if(weights.desc.GetLayoutEnum() == miopenTensorCHWNc4 ||
       weights.desc.GetLayoutEnum() == miopenTensorCHWNc8)
    {
        throw std::runtime_error("GenConvData do not support CHWNc filter layout");
    }

    const auto conv_desc = conv_config.GetConv();

    const auto output_desc =
        conv_desc.GetForwardOutputTensor(input.desc, weights.desc, miopen_type<Tout>{});

    auto output = tensor<Tout>{output_desc};

    output.generate(GenConvData<Tout, Tin>{weights.desc.GetLengths()});
    // TRJS
    {
    std::ofstream out_out("out.txt");
    Tin out_max = std::numeric_limits<Tout>::min();
    Tin out_min = std::numeric_limits<Tout>::max();
    for(auto v : output) {
        float fv = fabs(v);
        if(fv > out_max) out_max = fv;
        if(fv < out_min) out_min = fv;
    }
    out_out << "output max mag: " << out_max << "  min mag: " << out_min << "  scale: " << out_max / out_min << std::endl;
    for(auto v : output) out_out << (float)v << "  ";
    out_out << std::endl;
    }

    weights.generate(GenConvData<Twei, Tin>{weights.desc.GetLengths()});
    // TRJS
    {
    std::ofstream weights_out("weights.txt");
    Twei wei_max = std::numeric_limits<Twei>::min();
    Twei wei_min = std::numeric_limits<Twei>::max();
    for(auto v : weights) {
        float fv = fabs(v);
        if(fv > wei_max) wei_max = fv;
        if(fv < wei_min) wei_min = fv;
    }
    weights_out << "weights max mag: " << wei_max << "  min mag: " << wei_min << "  scale: " << wei_max / wei_min << std::endl;
    for(auto v : weights) weights_out << v << "  ";
    weights_out << std::endl;
    }

    std::fill(input.begin(), input.end(), Tin());

    auto&& handle = get_handle();
    auto in_dev   = handle.Write(input.data);
    auto wei_dev  = handle.Write(weights.data);
    auto out_dev  = handle.Write(output.data);

    //**********************************
    // Run solver
    //**********************************

    const auto tensors = miopen::ConvBwdTensors{
        output.desc, out_dev.get(), weights.desc, wei_dev.get(), input.desc, in_dev.get()};

    const auto problem = miopen::conv::ProblemDescription(
        output.desc, weights.desc, input.desc, conv_desc, miopen::conv::Direction::BackwardData);
    const auto ctx = [&] {
        auto tmp = miopen::ExecutionContext{&handle};
        problem.SetupFloats(tmp);
        return tmp;
    }();

    if(!solv.IsApplicable(ctx, problem))
    {
        // Do not put GTEST_SKIP here.
        // The usage of non-applicable config should be considered as a bug in the test.
        GTEST_FAIL();
    }

    Workspace wspace;
    if(solv.MayNeedWorkspace())
    {
        const auto cur_sol_ws = solv.GetWorkspaceSize(ctx, problem);
        wspace.resize(cur_sol_ws);
    }

    const auto invoke_params = miopen::conv::DataInvokeParams{
        tensors, wspace.ptr(), wspace.size(), conv_desc.attribute.gfx90aFp16alt.GetBwd()};

    const auto sol = FindSolution(solv, params, ctx, problem, invoke_params);
    ASSERT_TRUE(sol.Succeeded());
    ASSERT_TRUE(sol.invoker_factory);
    const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
    (invoker)(handle, invoke_params);
    handle.Finish();

    //**********************************
    // Verify
    //**********************************

    auto ref_in = tensor<Tref>{input.desc};
    if(params.use_cpu_ref)
    {
        cpu_convolution_backward_data(conv_desc.GetSpatialDimension(),
                                      ref_in,
                                      weights,
                                      output,
                                      conv_desc.GetConvPads(),
                                      conv_desc.GetConvStrides(),
                                      conv_desc.GetConvDilations(),
                                      conv_desc.GetGroupCount());
    }
    else
    {
        ref_in = ref_conv_bwd(ref_in, weights, output, conv_desc);
    }

    input.data = handle.Read<Tin>(in_dev, input.data.size());

    VerifyData(
        input.data, ref_in.data, algo, miopen::conv::Direction::BackwardData, params.tolerances);
}

template <typename T, typename Tref>
void RunSolverBwd(const miopen::solver::conv::ConvSolverInterface& solv,
                  const UnitTestConvSolverParams& params,
                  const ConvTestCase& conv_config,
                  miopenConvAlgorithm_t algo)
{
    RunSolverBwd<T, T, T, Tref>(solv, params, conv_config, algo);
}

//**********************************
// Wrw
//**********************************
template <typename Tin, typename Twei, typename Tout, typename Tref>
void RunSolverWrw(const miopen::solver::conv::ConvSolverInterface& solv,
                  const UnitTestConvSolverParams& params,
                  const ConvTestCase& conv_config,
                  miopenConvAlgorithm_t algo)
{
    //**********************************
    // Prepare
    //**********************************

    auto input   = tensor<Tin>{conv_config.GetXTensorDescriptor()};
    auto weights = tensor<Twei>{conv_config.GetWTensorDescriptor()};

    const auto conv_desc = conv_config.GetConv();

    const auto output_desc =
        conv_desc.GetForwardOutputTensor(input.desc, weights.desc, miopen_type<Tout>{});

    if(output_desc.GetLayoutEnum() == miopenTensorCHWNc4 ||
       output_desc.GetLayoutEnum() == miopenTensorCHWNc8)
    {
        throw std::runtime_error("GenConvData do not support CHWNc filter layout");
    }

    auto output = tensor<Tout>{output_desc};

    input.generate(GenConvData<Tin, Twei>{output_desc.GetLengths()});

    // TRJS
    float range_factor = 0.1 * env::value(MIOPEN_RANGE_FACTORX10);
    if(range_factor == 0.0) range_factor = 1.0;

    {
    std::ofstream input_out("input.txt");
    Tin in_max = std::numeric_limits<Tin>::min();
    Tin in_min = std::numeric_limits<Tin>::max();
    for(auto v : input) {
        float fv = fabs(v);
        if(fv > in_max) in_max = fv;
        if(fv < in_min) in_min = fv;
    }
    input_out << "input max mag: " << in_max << "  min mag: " << in_min << "  scale: " << in_max / in_min << 
        "  eps: " << range_factor * in_max * static_cast<float>(std::numeric_limits<Tin>::epsilon())
         << std::endl;
    for(auto v : input) {
        input_out << v << "  ";
    }
    input_out << std::endl;
    }

    output.generate(GenConvData<Tout, Twei>{output_desc.GetLengths()});
    // TRJS
    {
    std::ofstream out_out("output.txt");
    Tout out_max = std::numeric_limits<Tout>::min();
    Tout out_min = std::numeric_limits<Tout>::max();
    for(auto v : output) {
        float fv = fabs(v);
        if(fv > out_max) out_max = fv;
        if(fv < out_min) out_min = fv;
    }
    out_out << "output max mag: " << out_max << "  min mag: " << out_min << "  scale: " << out_max / out_min << 
        "  eps: " << range_factor * out_max * static_cast<float>(std::numeric_limits<Tout>::epsilon())
    << std::endl;
    for(auto v : output) out_out << v << "  ";
    out_out << std::endl;
    }

    std::fill(weights.begin(), weights.end(), Twei());

    auto&& handle = get_handle();
    auto in_dev   = handle.Write(input.data);
    auto wei_dev  = handle.Write(weights.data);
    auto out_dev  = handle.Write(output.data);

    //**********************************
    // Run solver
    //**********************************

    const auto tensors = miopen::ConvWrwTensors{
        output.desc, out_dev.get(), input.desc, in_dev.get(), weights.desc, wei_dev.get()};

    const auto problem = miopen::conv::ProblemDescription(
        output.desc, weights.desc, input.desc, conv_desc, miopen::conv::Direction::BackwardWeights);
    const auto ctx = [&] {
        auto tmp = miopen::ExecutionContext{&handle};
        problem.SetupFloats(tmp);
        return tmp;
    }();

    if(!solv.IsApplicable(ctx, problem))
    {
        // Do not put GTEST_SKIP here.
        // The usage of non-applicable config should be considered as a bug in the test.
        GTEST_FAIL();
    }

    Workspace wspace;
    if(solv.MayNeedWorkspace())
    {
        const auto cur_sol_ws = solv.GetWorkspaceSize(ctx, problem);
        wspace.resize(cur_sol_ws);
    }

    const auto invoke_params = miopen::conv::WrWInvokeParams{
        tensors, wspace.ptr(), wspace.size(), conv_desc.attribute.gfx90aFp16alt.GetWrW()};

    const auto sol = FindSolution(solv, params, ctx, problem, invoke_params);
    ASSERT_TRUE(sol.Succeeded());
    ASSERT_TRUE(sol.invoker_factory);
    const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
    (invoker)(handle, invoke_params);
    handle.Finish();

    //**********************************
    // Verify
    //**********************************

    auto ref_weights = tensor<Tref>{weights.desc};
    if(params.use_cpu_ref)
    {
        cpu_convolution_backward_weight(conv_desc.GetSpatialDimension(),
                                        input,
                                        ref_weights,
                                        output,
                                        conv_desc.GetConvPads(),
                                        conv_desc.GetConvStrides(),
                                        conv_desc.GetConvDilations(),
                                        conv_desc.GetGroupCount());
    }
    else
    {
        ref_weights = ref_conv_wrw(input, ref_weights, output, conv_desc);
    }

    weights.data = handle.Read<Twei>(wei_dev, weights.data.size());

    VerifyData(weights.data,
               ref_weights.data,
               algo,
               miopen::conv::Direction::BackwardWeights,
               params.tolerances);
}

template <typename T, typename Tref>
void RunSolverWrw(const miopen::solver::conv::ConvSolverInterface& solv,
                  const UnitTestConvSolverParams& params,
                  const ConvTestCase& conv_config,
                  miopenConvAlgorithm_t algo)
{
    RunSolverWrw<T, T, T, Tref>(solv, params, conv_config, algo);
}

template <typename T, typename Tref>
void RunSolver(const miopen::solver::conv::ConvSolverInterface& solver,
               const UnitTestConvSolverParams& params,
               miopen::conv::Direction direction,
               const ConvTestCase& conv_config,
               miopenConvAlgorithm_t algo)
{
    // clang-format off
    switch(direction)
    {
    case miopen::conv::Direction::Forward:
        RunSolverFwd<T, Tref>(solver, params, conv_config, algo);
        return;
    case miopen::conv::Direction::BackwardData:
        RunSolverBwd<T, Tref>(solver, params, conv_config, algo);
        return;
    case miopen::conv::Direction::BackwardWeights:
        RunSolverWrw<T, Tref>(solver, params, conv_config, algo);
        return;
    default:
        throw std::runtime_error("unknown direction");
    }
    // clang-format on
}

void RunSolver(const miopen::solver::conv::ConvSolverInterface& solver,
               const UnitTestConvSolverParams& params,
               miopen::conv::Direction direction,
               const ConvTestCase& conv_config,
               miopenConvAlgorithm_t algo)
{
    if(conv_config.GetXDataType() == conv_config.GetWDataType() &&
       conv_config.GetWDataType() == conv_config.GetYDataType())
    {
        // clang-format off
        switch(conv_config.GetXDataType())
        {
        case miopenHalf:
            RunSolver<half_float::half, half_float::half>(solver, params, direction, conv_config, algo);
            return;
        case miopenFloat:
            RunSolver<float, float>(solver, params, direction, conv_config, algo);
            return;
        case miopenBFloat16:
            RunSolver<bfloat16, bfloat16>(solver, params, direction, conv_config, algo);
            return;
        case miopenInt8:
            RunSolver<int8_t, int8_t>(solver, params, direction, conv_config, algo);
            return;
        default:
            throw std::runtime_error("handling of this data type is not yet implemented");
        }
        // clang-format on
    }
    else if(direction == miopen::conv::Direction::Forward &&
            conv_config.GetXDataType() == miopenInt8 && conv_config.GetWDataType() == miopenInt8 &&
            conv_config.GetYDataType() == miopenInt32)
    {
        RunSolverFwd<int8_t, int8_t, int32_t, int32_t>(solver, params, conv_config, algo);
        return;
    }

    throw std::runtime_error("handling of mixed data types is not yet implemented");
}

} // namespace

void UnitTestConvSolverBase::SetUpImpl(const UnitTestConvSolverParams& params)
{
    if(!IsTestSupportedByDevice(params.supported_devs))
    {
        GTEST_SKIP();
    }
    else if(params.check_xnack_disabled && get_handle_xnack())
    {
        GTEST_SKIP();
    }
}

void UnitTestConvSolverBase::RunTestImpl(const miopen::solver::conv::ConvSolverInterface& solver,
                                         const UnitTestConvSolverParams& params,
                                         miopen::conv::Direction direction,
                                         const ConvTestCase& conv_config,
                                         miopenConvAlgorithm_t algo)
{
    DeprecatedSolversScopedEnabler deprecated_solv_enabler;
    if(params.enable_deprecated_solvers)
    {
        deprecated_solv_enabler.Enable();
    }

    ConvAttrFp16AltScopedSetter conv_attr_fp16_alt_setter;
    if(params.conv_attr_fp16_alt)
        conv_attr_fp16_alt_setter.SetValue(params.conv_attr_fp16_alt.value());

    RunSolver(solver, params, direction, conv_config, algo);
}

//************************************************************************************
// This test is designed to detect the expansion of the solver's device applicability
//************************************************************************************

void UnitTestConvSolverDevApplicabilityBase::RunTestImpl(
    const miopen::solver::conv::ConvSolverInterface& solver,
    const UnitTestConvSolverParams& params,
    miopen::conv::Direction direction,
    const ConvTestCase& conv_config)
{
    DeprecatedSolversScopedEnabler deprecated_solv_enabler;
    if(params.enable_deprecated_solvers)
    {
        deprecated_solv_enabler.Enable();
    }

    ConvAttrFp16AltScopedSetter conv_attr_fp16_alt_setter;
    if(params.conv_attr_fp16_alt)
        conv_attr_fp16_alt_setter.SetValue(params.conv_attr_fp16_alt.value());

    const auto problem = conv_config.GetProblemDescription(direction);

    const auto all_known_devs = GetAllKnownDevices();
    for(const auto& [dev, dev_descr] : all_known_devs)
    {
        const auto supported = IsDeviceSupported(params.supported_devs, dev);
        // std::cout << "Test " << dev_descr << " (supported: " << supported << ")" << std::endl;

        auto handle    = MockHandle{dev_descr, params.check_xnack_disabled};
        const auto ctx = [&] {
            auto tmp = miopen::ExecutionContext{&handle};
            problem.SetupFloats(tmp);
            return tmp;
        }();

        const auto is_applicable = solver.IsApplicable(ctx, problem);
        // std::cout << "IsApplicable: " << is_applicable << std::endl;
        if(is_applicable != supported)
        {
            GTEST_FAIL() << dev_descr << " is" << (is_applicable ? "" : " not")
                         << " applicable for " << solver.SolverDbId() << " but "
                         << (supported ? "" : "not ") << "marked as supported";
        }
    }
}

} // namespace unit_tests
} // namespace miopen
