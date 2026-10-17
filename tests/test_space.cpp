// The space exists once, in WGSL. These tests run its functions on the software GPU (32-bit
// floats, as on a GPU) and check them against the geometric product of Cl(4, 1) (syn::Multivector,
// 64-bit) and against the closed forms of the readings.
#include <cctype>
#include <cmath>
#include <string>
#include <vector>

#include "syn/core/multivector.hpp"
#include "syn/gpu/software_gpu.hpp"
#include "syn/kernel/kernel.hpp"
#include "syn/space/space.hpp"
#include "test_framework.hpp"

using namespace syn;

namespace {

// Runs WGSL statements with the space in scope; put(i, X) and putf(i, v) write results.
std::vector<float> wgsl(const std::string& body, std::size_t words = 256) {
  const std::string src = space::spaceWgsl() + R"wgsl(
@group(0) @binding(0) var<storage, read_write> OUT : array<f32>;
fn put(i : u32, p : V5) {
  OUT[i * 5u] = p.x.x; OUT[i * 5u + 1u] = p.x.y; OUT[i * 5u + 2u] = p.x.z; OUT[i * 5u + 3u] = p.x.w; OUT[i * 5u + 4u] = p.e;
}
fn putf(i : u32, v : f32) { OUT[i] = v; }
@compute @workgroup_size(1)
fn main() {
)wgsl" + body + "\n}\n";
  const auto shader = gpu::Shader::compile(src);
  gpu::Buffer out = gpu::Buffer::zeros(words);
  gpu::dispatch(*shader, {&out}, 1, 1, 1);
  return out.floats();
}

struct P5 {
  double x[4], e;
};
P5 at(const std::vector<float>& v, std::size_t i) { return {{v[i * 5], v[i * 5 + 1], v[i * 5 + 2], v[i * 5 + 3]}, v[i * 5 + 4]}; }
Multivector mv(const P5& v) { return Multivector::vector(4, 1, {v.x[0], v.x[1], v.x[2], v.x[3], v.e}); }
double maxDiff(const Multivector& a, const P5& b) {
  const auto x = a.vectorPart();
  double d = std::fabs(x[4] - b.e);
  for (int i = 0; i < 4; ++i) d = std::max(d, std::fabs(x[i] - b.x[i]));
  return d;
}
Multivector basis(int i) { return Multivector::basis(4, 1, i); }

}  // namespace

TEST(space_inner_product_matches_the_algebra) {
  const auto v = wgsl(R"(
    let p = atLogit(0.7, kA);
    let q = atLogit(-1.3, kB);
    put(0u, p); put(1u, q);
    putf(10u, dot5(p, q)); putf(11u, dot5(p, p));)");
  const P5 p = at(v, 0), q = at(v, 1);
  CHECK_NEAR(v[10], scalarProduct(mv(p), mv(q)), 1e-6);
  CHECK_NEAR(v[11], 0.0, 1e-6);  // null
  CHECK_NEAR(scalarProduct(mv(p), mv(p)), 0.0, 1e-6);
}

TEST(space_chance_is_the_logistic_of_the_logit) {
  const double xs[6] = {-6.0, -2.0, -0.3, 0.0, 0.9, 4.0};
  std::string body;
  for (int i = 0; i < 6; ++i) {
    const std::string k = std::to_string(i) + "u";
    body += "{ let p = atLogit(" + std::to_string(xs[i]) + ", kA); put(" + k + ", p); putf(30u + " + k + ", chance(p)); putf(40u + " + k +
            ", logit(p)); }\n";
  }
  body += "putf(50u, chance(atChance(0.3, kC)));";
  const auto v = wgsl(body);
  for (int i = 0; i < 6; ++i) {
    const double x = xs[i];
    CHECK_NEAR(v[30 + i], 1.0 / (1.0 + std::exp(-x)), 2e-6);
    CHECK_NEAR(v[40 + i], x, 2e-4 * (1.0 + std::fabs(x)));
    // chance = 1 + ½·X·L with L = N + e
    CHECK_NEAR(v[30 + i], 1.0 + 0.5 * scalarProduct(mv(at(v, i)), basis(0) + basis(4)), 2e-6);
  }
  CHECK_NEAR(v[50], 0.3, 1e-6);
}

TEST(space_shift_is_the_dilation_versor) {
  // D = exp(−d/4 · N e) = cosh(d/4) − sinh(d/4)·N e acting by the sandwich D X D̃.
  const double ds[3] = {-2.0, 0.4, 1.7};
  std::string body = "let p = atLogit(0.6, kB); put(0u, p);\n";
  for (int i = 0; i < 3; ++i)
    body += "put(" + std::to_string(i + 1) + "u, shift(p, " + std::to_string(ds[i]) + ")); putf(" + std::to_string(30 + i) + "u, logit(shift(p, " +
            std::to_string(ds[i]) + ")));\n";
  body += "putf(40u, logit(shift(shift(atLogit(-0.5, kA), 0.8), 1.1)));";
  const auto v = wgsl(body);
  const P5 p = at(v, 0);
  for (int i = 0; i < 3; ++i) {
    const double d = ds[i];
    const Multivector Ne = basis(0) * basis(4);
    const Multivector D = Multivector::scalar(4, 1, std::cosh(d / 4)) - Ne * std::sinh(d / 4);
    Multivector y = sandwich(D, mv(p)).grade(1);
    y = y / y.vectorPart()[4];  // normalise the weight
    CHECK(maxDiff(y, at(v, i + 1)) < 2e-6);
    CHECK_NEAR(v[30 + i], 0.6 + d, 1e-4);
  }
  // Dilations about the pole compose by adding logits.
  CHECK_NEAR(v[40], -0.5 + 1.9, 1e-4);
}

TEST(space_rotate_is_a_rotor) {
  const double a = 0.9;
  const auto v = wgsl(R"(
    let p = atLogit(0.4, kA);
    put(0u, p); put(1u, rotate(p, kA, kB, 0.9));
    putf(20u, chance(p)); putf(21u, chance(rotate(p, kA, kB, 0.9)));)");
  const Multivector R = Multivector::scalar(4, 1, std::cos(a / 2)) - basis(1) * basis(2) * std::sin(a / 2);
  CHECK(maxDiff(sandwich(R, mv(at(v, 0))).grade(1), at(v, 1)) < 2e-6);
  CHECK_NEAR(v[21], v[20], 1e-6);
}

TEST(space_towards_is_the_geodesic_rotor) {
  const auto v = wgsl(R"(
    let p = atLogit(1.2, kA);
    let q = atLogit(-0.8, kC);
    let mid = towards(p, q, 0.5);
    put(0u, p); put(1u, q); put(2u, mid);
    putf(20u, dot5(mid, mid)); putf(21u, similarity(p, mid)); putf(22u, similarity(mid, q));
    putf(23u, similarity(p, towards(p, q, 1.0))); putf(24u, similarity(p, q));)");
  CHECK_NEAR(v[20], 0.0, 1e-6);
  CHECK_NEAR(v[21], v[22], 1e-6);
  CHECK_NEAR(v[23], v[24], 1e-6);
  // The rotor of the plane p∧q: R = cos(tθ/2) − sin(tθ/2)·B̂ with B̂ = x̂∧ŷ/|x̂∧ŷ|.
  const Multivector x = mv(at(v, 0)) - basis(4), y = mv(at(v, 1)) - basis(4);
  const Multivector B = x ^ y;
  const double s2 = B.normSquared(), theta = std::acos(scalarProduct(x, y)), t = 0.5;
  const Multivector R = Multivector::scalar(4, 1, std::cos(t * theta / 2)) - B / std::sqrt(s2) * std::sin(t * theta / 2);
  CHECK(maxDiff(sandwich(R, mv(at(v, 0))).grade(1), at(v, 2)) < 2e-6);
}

TEST(space_mixtures_and_caps) {
  const auto v = wgsl(R"(
    let p = atChance(0.3, kB);
    putf(0u, chance(raise(p, 0.5)));
    putf(1u, chance(lower(p, 0.5)));
    putf(2u, chance(nudge(p, 0.2)));
    putf(3u, chance(nudge(p, -0.2)));
    putf(4u, f32(channelOf(raise(p, 0.5))));
    putf(5u, f32(inside(atChance(0.71, kA), chanceCap(0.7))));
    putf(6u, f32(inside(atChance(0.69, kA), chanceCap(0.7))));
    let c = cap(atLogit(0.0, kA), 0.5);
    putf(7u, f32(inside(towards(atLogit(0.0, kA), atLogit(0.0, kB), 0.3), c)));
    putf(8u, f32(inside(atLogit(0.0, kB), c)));)");
  CHECK_NEAR(v[0], 0.65, 1e-6);
  CHECK_NEAR(v[1], 0.15, 1e-6);
  CHECK_NEAR(v[2], 0.5, 1e-6);
  CHECK_NEAR(v[3], 0.1, 1e-6);
  CHECK(v[4] == 2.0f);  // kB
  CHECK(v[5] == 1.0f);
  CHECK(v[6] == 0.0f);
  CHECK(v[7] == 1.0f);
  CHECK(v[8] == 0.0f);
}

TEST(space_model_checks_on_the_software_gpu) {
  Value spec = Value::object();
  spec.set("space", Value::object().set("samples", 5000));
  const Value a = kernel::run("space", spec);
  CHECK(a.at("checks").num("chanceIsLogistic", 1) < 1e-5);
  CHECK(a.at("checks").num("dilationAddsLogits", 1) < 1e-4);
  CHECK(a.at("checks").num("mixtureIsLinear", 1) < 1e-5);
  CHECK(a.at("checks").num("pointsAreNull", 1) < 1e-5);
  CHECK(a.at("checks").num("rotorKeepsChance", 1) < 1e-5);
  // Random points: logits with mean 0 and sd 1.
  CHECK(std::fabs(a.at("random").num("meanLogit", 9)) < 0.05);
  CHECK(std::fabs(a.at("random").num("sdLogit", 9) - 1.0) < 0.05);
}

TEST(kernel_slices_across_workers_give_the_same_result) {
  Value spec = Value::object();
  spec.set("sim", Value::object().set("learners", 300).set("seed", 5));
  for (const char* name : {"classroom", "memory", "rubric"}) {
    const Value whole = kernel::run(name, spec);
    std::vector<Value> parts;
    for (int w = 2; w >= 0; --w) parts.push_back(kernel::simulateSlices(name, spec, static_cast<std::size_t>(w), 3));
    CHECK(kernel::analyseSlices(name, spec, parts).json() == whole.json());
  }
}

TEST(kernel_wgsl_is_assembled_and_compiles_for_every_model) {
  const std::string src = kernel::agentKernel(kernel::model("space"));
  CHECK(src.find("fn agent()") != std::string::npos);
  CHECK(src.find("fn shift(") != std::string::npos);
  CHECK(src.find("@compute") != std::string::npos);
  for (const auto* m : kernel::models()) {
    const auto shader = gpu::Shader::compile(kernel::agentKernel(*m));
    CHECK(shader->workgroupSize[0] == kernel::kWorkgroup);
    CHECK(shader->bindings.size() == 5);
  }
  CHECK(gpu::Shader::compile(kernel::reduceKernel())->bindings.size() == 3);
  CHECK(gpu::Shader::compile(kernel::quantileKernel())->workgroupSize[0] == kernel::kQuantileWorkgroup);
}

TEST(kernel_agents_use_no_wgsl_reserved_words) {
  // Reserved words of WGSL: the browsers' WGSL compilers reject them as identifiers (the
  // software GPU's compiler does not check them).
  static const char* reserved[] = {
      "NULL", "Self", "abstract", "active", "alignas", "alignof", "as", "asm", "asm_fragment", "async", "attribute", "await",
      "become", "binding_array", "cast", "catch", "class", "co_await", "co_return", "co_yield", "coherent", "column_major",
      "common", "compile", "compile_fragment", "concept", "const_cast", "consteval", "constexpr", "constinit", "crate",
      "debugger", "decltype", "delete", "demote", "demote_to_helper", "do", "dynamic_cast", "enum", "explicit", "export",
      "extends", "extern", "external", "fallthrough", "filter", "final", "finally", "friend", "from", "fxgroup", "get", "goto",
      "groupshared", "highp", "impl", "implements", "import", "inline", "instanceof", "interface", "layout", "lowp", "macro",
      "macro_rules", "match", "mediump", "meta", "mod", "module", "move", "mut", "mutable", "namespace", "new", "nil",
      "noexcept", "noinline", "nointerpolation", "non_coherent", "noncoherent", "noperspective", "null", "nullptr", "of",
      "operator", "package", "packoffset", "partition", "pass", "patch", "pixelfragment", "precise", "precision", "premerge",
      "priv", "protected", "pub", "public", "readonly", "ref", "regardless", "register", "reinterpret_cast", "require",
      "resource", "restrict", "self", "set", "shared", "sizeof", "smooth", "snorm", "static", "static_assert", "static_cast",
      "std", "subroutine", "super", "target", "template", "this", "thread_local", "throw", "trait", "try", "type", "typedef",
      "typeid", "typename", "typeof", "union", "unless", "unorm", "unsafe", "unsized", "use", "using", "varying", "virtual",
      "volatile", "wgsl", "where", "with", "writeonly", "yield", "alias", "case", "const", "const_assert", "continuing",
      "default", "diagnostic", "discard", "enable", "loop", "override", "requires", "struct", "switch", "fn"};
  for (const auto* m : kernel::models()) {
    const std::string src = m->agentSource;
    std::string word;
    for (std::size_t i = 0; i <= src.size(); ++i) {
      if (i + 1 < src.size() && src[i] == '/' && src[i + 1] == '/') {  // comments may say anything
        while (i < src.size() && src[i] != '\n') ++i;
        word.clear();
        continue;
      }
      const char ch = i < src.size() ? src[i] : ' ';
      if (std::isalnum(static_cast<unsigned char>(ch)) || ch == '_') {
        word += ch;
        continue;
      }
      for (const char* r : reserved)
        if (word == r) syntest::fail(m->name + ": the agent uses the WGSL reserved word '" + word + "'", __FILE__, __LINE__);
      word.clear();
    }
  }
}
