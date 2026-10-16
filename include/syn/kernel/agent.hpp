#pragma once

// Writing an agent program once for C++ and WGSL.
//
//   SYN_AGENT(name, statements...)
//
// defines `name` as a C++ function template (float and double) and as WGSL source (the same
// tokens, stringified). The statements are written in the common subset of the two languages:
//   let x = ...;            an immutable value          (C++: const auto)
//   var x = ...;            a variable                  (C++: auto)
//   ARR(name, type, n);     a zero-initialised array    (WGSL: var name : array<type, n>;)
//   f32(..), u32(..), i32(..)  conversions; float literals are always written f32(0.5)
//   if / else if / else, for (var k = 0u; k < n; k = k + 1u), while, break, continue, return;
//   select(f, t, c) instead of c ? t : f;  min, max, clamp, abs, exp, log, sqrt, pow, floor,
//   cos, sin, tanh, atan, acos;  the space (syn/space/space.hpp): V5, v5, dot5, chance, shift, ...
// and the agent interface:
//   A, JOB, NA       the agent, its job, the number of agents (u32)
//   P(i), J(i)       a model constant, a value of the job's row (f32)
//   U(k)             the k-th uniform in (0, 1) of this agent under the job's seed
//   stat(i, v)       adds v to statistic i;   quant(q, v) keeps value q (in [0, 1]) for quantiles
// Comments are allowed (the preprocessor removes them before stringifying).

#include <array>
#include <cmath>
#include <cstdint>

#include "syn/kernel/kernel.hpp"
#include "syn/kernel/random.hpp"
#include "syn/space/space.hpp"

namespace syn::kernel::lang {

using u32 = std::uint32_t;
using i32 = std::int32_t;

template <class F>
inline F select(F f, F t, bool c) {
  return c ? t : f;
}
template <class F>
inline F clamp(F x, F lo, F hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}
template <class F>
inline F min(F a, F b) {
  return a < b ? a : b;
}
template <class F>
inline F max(F a, F b) {
  return a > b ? a : b;
}
template <class F>
inline F abs(F x) {
  return x < F(0) ? -x : x;
}
using std::acos;
using std::atan;
using std::cos;
using std::exp;
using std::floor;
using std::log;
using std::pow;
using std::sin;
using std::sqrt;
using std::tanh;

}  // namespace syn::kernel::lang

// The C++ side of an agent. Inside the body, f32 is the engine's float type.
#define SYN_AGENT(NAME, ...)                                                                           \
  namespace syn_agents {                                                                               \
  template <class f32>                                                                                 \
  void NAME(::syn::kernel::Ctx<f32>& ctx_) {                                                           \
    using namespace ::syn::kernel::lang;                                                               \
    using namespace ::syn::space;                                                                      \
    using V5 [[maybe_unused]] = ::syn::space::V5<f32>;                                                 \
    [[maybe_unused]] const u32 A = ctx_.agent;                                                         \
    [[maybe_unused]] const u32 JOB = ctx_.job;                                                         \
    [[maybe_unused]] const u32 NA = ctx_.plan->agents;                                                 \
    [[maybe_unused]] auto P = [&](u32 i) { return f32(ctx_.params[i]); };                              \
    [[maybe_unused]] auto J = [&](u32 i) { return f32(ctx_.row[i]); };                                 \
    [[maybe_unused]] auto U = [&](u32 k) { return ::syn::kernel::uniform<f32>(ctx_.seed, ctx_.agent, k); }; \
    [[maybe_unused]] auto stat = [&](u32 i, f32 v) { ctx_.stats[i] = ctx_.stats[i] + v; };             \
    [[maybe_unused]] auto quant = [&](u32 q, f32 v) { ctx_.quants[q] = v; };                           \
    [[maybe_unused]] auto pole = []() { return ::syn::space::pole<f32>(); };                           \
    [[maybe_unused]] auto antipode = []() { return ::syn::space::antipode<f32>(); };                   \
    [[maybe_unused]] auto equator = [](int ch) { return ::syn::space::equator<f32>(ch); };              \
    __VA_ARGS__                                                                                        \
  }                                                                                                    \
  inline const char* NAME##_source() { return #__VA_ARGS__; }                                          \
  }

#define let const auto
#define var auto
#define ARR(NAME, TYPE, N) std::array<TYPE, N> NAME{}
