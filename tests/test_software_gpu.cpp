// The software GPU: WGSL semantics, the execution model (workgroups, lanes, barriers, divergence)
// and dispatches split across workers.
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "syn/gpu/software_gpu.hpp"
#include "test_framework.hpp"

using namespace syn::gpu;

namespace {

// Runs `body` in a one-invocation shader; results go to OUT (u32 words) via OUT[i] = ...
std::vector<std::uint32_t> one(const std::string& decls, const std::string& body, std::size_t words = 64) {
  const std::string src = "@group(0) @binding(0) var<storage, read_write> OUT : array<u32>;\n" + decls +
                          "\n@compute @workgroup_size(1)\nfn main() {\n" + body + "\n}\n";
  const auto shader = Shader::compile(src);
  Buffer out = Buffer::zeros(words);
  dispatch(*shader, {&out}, 1, 1, 1);
  return out.words;
}

bool throwsOnCompile(const std::string& src) {
  try {
    Shader::compile(src);
  } catch (const std::runtime_error&) {
    return true;
  }
  return false;
}

}  // namespace

TEST(software_gpu_integer_semantics) {
  const auto o = one("", R"(
    let big = 4000000000u;
    OUT[0] = big + big;                          // u32 wraps
    let m = -2147483647i - 1i;
    OUT[1] = u32(m - 1i);                        // i32 wraps
    var s = 33u;
    OUT[3] = 1u << s;
    OUT[4] = u32(-8i >> 1u);                     // arithmetic shift for i32
    OUT[5] = 0xF0000000u >> 4u;                  // logical shift for u32
    var z = 0u;
    OUT[6] = 7u / z;                             // integer division by zero gives the dividend
    OUT[7] = 7u % z;                             //   and the remainder 0
    OUT[8] = u32(-7i / 2i);                      // truncation towards zero
    OUT[9] = u32(-7i % 2i);
    OUT[10] = ~5u;
    OUT[11] = (6u & 3u) | (8u ^ 12u);)");
  CHECK(o[0] == static_cast<std::uint32_t>(4000000000ull * 2ull));
  CHECK(o[1] == 0x7fffffffu);
  CHECK(o[3] == 2u);
  CHECK(o[4] == static_cast<std::uint32_t>(-4));
  CHECK(o[5] == 0x0F000000u);
  CHECK(o[6] == 7u);
  CHECK(o[7] == 0u);
  CHECK(o[8] == static_cast<std::uint32_t>(-3));
  CHECK(o[9] == static_cast<std::uint32_t>(-1));
  CHECK(o[10] == ~5u);
  CHECK(o[11] == (2u | 4u));
}

TEST(software_gpu_expressions) {
  const std::string src = R"wgsl(
@group(0) @binding(0) var<storage, read_write> F : array<f32>;
@group(0) @binding(1) var<storage, read_write> U : array<u32>;
struct S { a : vec4<f32>, b : f32, };
const K : u32 = 3u;
fn twice(v : vec3<f32>) -> vec3<f32> { return v * 2.0; }
fn early(x : f32) -> f32 {
  if (x > 0.0) { return 1.0; }
  return -1.0;
}
@compute @workgroup_size(1)
fn main() {
  var x = 3.7;
  U[0] = u32(x);
  U[1] = u32(-x);
  var big = 1e20;
  U[2] = u32(big);
  U[3] = u32(i32(-x));
  F[0] = f32(7u) / 2.0;
  F[1] = f32(true);
  U[4] = u32(bool(0.0)) + 2u * u32(bool(3u));
  F[2] = -7.5 % 2.0;
  F[3] = clamp(x, 0.0, 1.0);
  F[4] = select(1.0, 2.0, x > 3.0);
  F[5] = round(2.5);
  F[6] = sqrt(16.0) + exp(0.0) + log(1.0) + abs(-2.0) + floor(1.9) + fract(1.25);
  let v = vec3<f32>(1.0, 2.0, 3.0);
  let w = twice(v).zyx;
  F[7] = w.x; F[8] = w.y; F[9] = w.z;
  F[10] = dot(v, v);
  F[11] = length(vec2<f32>(3.0, 4.0));
  let n = normalize(vec3<f32>(0.0, 3.0, 4.0));
  F[12] = n.y;
  var s = S(vec4<f32>(1.0, 2.0, 3.0, 4.0), 5.0);
  var i = 2u;
  s.a[i] = 30.0;                                    // dynamic component of a member
  F[13] = s.a.z + s.b;
  var arr : array<S, 3>;
  arr[1] = s;
  arr[i].b = 9.0;
  F[14] = arr[1].a.z + arr[2].b + arr[0].b;
  let k = array<f32, 3>(0.1, 0.5, 0.9);
  F[15] = k[i];
  F[16] = k[i + 7u];                                // out-of-range indices are clamped
  F[17] = early(2.0) + early(-2.0) * 10.0;
  var t = 0.0;
  t += 2.0; t *= 3.0; t -= 1.0;
  F[18] = t;
  var c = 0u;
  for (var a = 0u; a < 10u; a = a + 1u) {
    if (a == 2u) { continue; }
    if (a == 6u) { break; }
    c = c + a;
  }
  var w2 = 0u;
  while (w2 < K * 2u) { w2++; }
  U[5] = c;
  U[6] = w2;
  U[7] = u32(true && (false || x > 1.0));
  U[8] = u32(min(3i, -2i));
  U[9] = max(3u, 9u);
  F[19] = mix(1.0, 3.0, 0.25);
}
)wgsl";
  const auto shader = Shader::compile(src);
  Buffer F = Buffer::zeros(32), U = Buffer::zeros(16);
  dispatch(*shader, {&F, &U}, 1, 1, 1);
  const auto fv = F.floats();
  const auto& u = U.words;
  CHECK(u[0] == 3u);
  CHECK(u[1] == 0u);
  CHECK(u[2] == 4294967040u);
  CHECK(u[3] == static_cast<std::uint32_t>(-3));
  CHECK(fv[0] == 3.5f);
  CHECK(fv[1] == 1.0f);
  CHECK(u[4] == 2u);
  CHECK(fv[2] == -1.5f);
  CHECK(fv[3] == 1.0f);
  CHECK(fv[4] == 2.0f);
  CHECK(fv[5] == 2.0f);
  CHECK_NEAR(fv[6], 4.0 + 1.0 + 0.0 + 2.0 + 1.0 + 0.25, 1e-6);
  CHECK(fv[7] == 6.0f && fv[8] == 4.0f && fv[9] == 2.0f);
  CHECK(fv[10] == 14.0f);
  CHECK(fv[11] == 5.0f);
  CHECK_NEAR(fv[12], 0.6, 1e-7);
  CHECK(fv[13] == 35.0f);
  CHECK(fv[14] == 30.0f + 9.0f + 0.0f);
  CHECK(fv[15] == 0.9f);
  CHECK(fv[16] == 0.9f);
  CHECK(fv[17] == 1.0f - 10.0f);
  CHECK(fv[18] == 5.0f);
  CHECK(u[5] == 0u + 1u + 3u + 4u + 5u);
  CHECK(u[6] == 6u);
  CHECK(u[7] == 1u);
  CHECK(u[8] == static_cast<std::uint32_t>(-2));
  CHECK(u[9] == 9u);
  CHECK(fv[19] == 1.5f);
}

TEST(software_gpu_workgroups_barriers_and_divergence) {
  // A tree reduction in workgroup memory (barriers inside a loop), a divergent loop whose trip
  // count differs per invocation, and builtins: every invocation of every workgroup.
  const std::string src = R"wgsl(
@group(0) @binding(0) var<storage, read> IN : array<f32>;
@group(0) @binding(1) var<storage, read_write> OUT : array<f32>;
@group(0) @binding(2) var<storage, read_write> TRIPS : array<u32>;
var<workgroup> RED : array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid : vec3<u32>, @builtin(workgroup_id) wid : vec3<u32>,
        @builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  var trips = 0u;
  var k = 0u;
  while (k < lid.x % 7u) {                    // divergent: 0..6 iterations
    trips = trips + 1u;
    if (trips == 5u) { break; }
    k = k + 1u;
  }
  TRIPS[gid.x + 1000u * wid.y] = trips + 100u * nwg.x;
  RED[lid.x] = IN[gid.x];
  workgroupBarrier();
  for (var w = 32u; w > 0u; w = w >> 1u) {
    if (lid.x < w) { RED[lid.x] = RED[lid.x] + RED[lid.x + w]; }
    workgroupBarrier();
  }
  if (lid.x == 0u) { OUT[wid.x + 10u * wid.y] = RED[0]; }
}
)wgsl";
  const auto shader = Shader::compile(src);
  CHECK(shader->hasBarriers);
  std::vector<float> in(256);
  for (std::size_t i = 0; i < in.size(); ++i) in[i] = static_cast<float>(i % 13);
  Buffer IN = Buffer::ofFloats(in), OUT = Buffer::zeros(32), TRIPS = Buffer::zeros(4096);
  dispatch(*shader, {&IN, &OUT, &TRIPS}, 4, 2, 1);
  const auto out = OUT.floats();
  for (std::uint32_t y = 0; y < 2; ++y)
    for (std::uint32_t g = 0; g < 4; ++g) {
      float sum = 0;
      for (std::uint32_t i = 0; i < 64; ++i) sum += in[g * 64 + i];
      CHECK(out[g + 10 * y] == sum);
    }
  for (std::uint32_t y = 0; y < 2; ++y)
    for (std::uint32_t i = 0; i < 256; ++i) CHECK(TRIPS.words[i + 1000 * y] == std::min(i % 64 % 7, 5u) + 400u);
}

TEST(software_gpu_atomics_and_buffers) {
  const std::string src = R"wgsl(
@group(0) @binding(0) var<storage, read> IN : array<u32>;
@group(0) @binding(1) var<storage, read_write> OUT : array<u32>;
var<workgroup> HIST : array<atomic<u32>, 8>;
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid : vec3<u32>) {
  if (lid.x < 8u) { atomicStore(&HIST[lid.x], 0u); }
  workgroupBarrier();
  atomicAdd(&HIST[IN[lid.x] % 8u], 1u);
  workgroupBarrier();
  if (lid.x < 8u) { OUT[lid.x] = atomicLoad(&HIST[lid.x]); }
  if (lid.x == 0u) {
    OUT[100] = IN[100000u];                     // out of bounds: reads 0
    OUT[100000u] = 7u;                          // out of bounds: dropped
  }
}
)wgsl";
  const auto shader = Shader::compile(src);
  std::vector<std::uint32_t> in(128);
  for (std::uint32_t i = 0; i < 128; ++i) in[i] = i * 3u;
  Buffer IN = Buffer::ofWords(in), OUT = Buffer::zeros(128);
  OUT.words[100] = 99u;
  dispatch(*shader, {&IN, &OUT}, 1, 1, 1);
  for (std::uint32_t b = 0; b < 8; ++b) CHECK(OUT.words[b] == 16u);
  CHECK(OUT.words[100] == 0u);
  CHECK(OUT.words.size() == 128);
}

TEST(software_gpu_split_dispatch_equals_one_dispatch) {
  const std::string src = R"wgsl(
@group(0) @binding(0) var<storage, read_write> OUT : array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  var x = f32(gid.x + 7u * gid.y);
  for (var k = 0u; k < gid.x % 5u; k = k + 1u) { x = sin(x) * 3.0 + 1.0; }
  OUT[gid.x + 1000u * gid.y] = x;
}
)wgsl";
  const auto shader = Shader::compile(src);
  Buffer whole = Buffer::zeros(4000);
  dispatch(*shader, {&whole}, 10, 3, 1);
  Buffer split = Buffer::zeros(4000, true);
  for (std::size_t first : {20u, 0u, 7u}) dispatch(*shader, {&split}, 10, 3, 1, first, first == 0u ? 7u : (first == 7u ? 20u : 30u));
  CHECK(split.words == whole.words);
  std::size_t written = 0;
  for (auto w : split.written) written += w;
  CHECK(written == 3u * 640u);
}

TEST(software_gpu_reports_errors) {
  CHECK(throwsOnCompile("@compute @workgroup_size(1) fn main() { let x = y; }"));
  CHECK(throwsOnCompile("@compute @workgroup_size(1) fn main() { let x = 1u + 1.5; }"));
  CHECK(throwsOnCompile("fn f() -> f32 { return f(); } @compute @workgroup_size(1) fn main() { let x = f(); }"));
  CHECK(throwsOnCompile("@compute @workgroup_size(1) fn main() { let a = 1u; a = 2u; }"));
  CHECK(throwsOnCompile("fn notAnEntry() {}"));
  bool threw = false;
  try {
    const auto s = Shader::compile("@group(0) @binding(0) var<storage, read_write> O : array<u32>;\n@compute @workgroup_size(1) fn main() { O[0] = 1u; }");
    dispatch(*s, {}, 1, 1, 1);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CHECK(threw);
}
