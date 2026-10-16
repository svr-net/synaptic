#include <cctype>
#include <cmath>
#include <string>

#include "syn/core/multivector.hpp"
#include "syn/kernel/kernel.hpp"
#include "syn/space/space.hpp"
#include "test_framework.hpp"

using namespace syn;
namespace sp = syn::space;

namespace {
Multivector mv(const sp::V5<double>& v) { return Multivector::vector(4, 1, {v.x[0], v.x[1], v.x[2], v.x[3], v.e}); }
double maxDiff(const Multivector& a, const sp::V5<double>& b) {
  const auto x = a.vectorPart();
  double d = std::fabs(x[4] - b.e);
  for (int i = 0; i < 4; ++i) d = std::max(d, std::fabs(x[i] - b.x[i]));
  return d;
}
Multivector basis(int i) { return Multivector::basis(4, 1, i); }
}  // namespace

TEST(space_inner_product_matches_the_algebra) {
  const auto p = sp::atLogit(0.7, sp::kA), q = sp::atLogit(-1.3, sp::kB);
  CHECK_NEAR(sp::dot5(p, q), scalarProduct(mv(p), mv(q)), 1e-12);
  CHECK_NEAR(sp::dot5(p, p), 0.0, 1e-12);  // null
  CHECK_NEAR(scalarProduct(mv(p), mv(p)), 0.0, 1e-12);
}

TEST(space_chance_is_the_logistic_of_the_logit) {
  for (double x : {-6.0, -2.0, -0.3, 0.0, 0.9, 4.0}) {
    const auto p = sp::atLogit(x, sp::kA);
    CHECK_NEAR(sp::chance(p), 1.0 / (1.0 + std::exp(-x)), 1e-12);
    CHECK_NEAR(sp::logit(p), x, 1e-9);
    // chance = 1 + ½·X·L with L = N + e
    CHECK_NEAR(sp::chance(p), 1.0 + 0.5 * scalarProduct(mv(p), basis(0) + basis(4)), 1e-12);
  }
  CHECK_NEAR(sp::chance(sp::atChance(0.3, sp::kC)), 0.3, 1e-12);
}

TEST(space_shift_is_the_dilation_versor) {
  // D = exp(−d/4 · N e) = cosh(d/4) − sinh(d/4)·N e acting by the sandwich D X D̃.
  for (double d : {-2.0, 0.4, 1.7}) {
    const auto p = sp::atLogit(0.6, sp::kB);
    const Multivector Ne = basis(0) * basis(4);
    const Multivector D = Multivector::scalar(4, 1, std::cosh(d / 4)) - Ne * std::sinh(d / 4);
    Multivector y = sandwich(D, mv(p)).grade(1);
    y = y / y.vectorPart()[4];  // normalise the weight
    CHECK(maxDiff(y, sp::shift(p, d)) < 1e-12);
    CHECK_NEAR(sp::logit(sp::shift(p, d)), 0.6 + d, 1e-9);
  }
  // Dilations about the pole compose by adding logits.
  const auto p = sp::atLogit(-0.5, sp::kA);
  CHECK_NEAR(sp::logit(sp::shift(sp::shift(p, 0.8), 1.1)), -0.5 + 1.9, 1e-9);
}

TEST(space_rotate_is_a_rotor) {
  const double a = 0.9;
  const auto p = sp::atLogit(0.4, sp::kA);
  const Multivector R = Multivector::scalar(4, 1, std::cos(a / 2)) - basis(1) * basis(2) * std::sin(a / 2);
  CHECK(maxDiff(sandwich(R, mv(p)).grade(1), sp::rotate(p, sp::kA, sp::kB, a)) < 1e-12);
  CHECK_NEAR(sp::chance(sp::rotate(p, sp::kA, sp::kB, a)), sp::chance(p), 1e-12);
}

TEST(space_towards_is_the_geodesic_rotor) {
  const auto p = sp::atLogit(1.2, sp::kA), q = sp::atLogit(-0.8, sp::kC);
  const auto mid = sp::towards(p, q, 0.5);
  CHECK_NEAR(sp::dot5(mid, mid), 0.0, 1e-12);
  CHECK_NEAR(sp::similarity(p, mid), sp::similarity(mid, q), 1e-12);
  CHECK(sp::similarity(p, sp::towards(p, q, 1.0)) - sp::similarity(p, q) < 1e-12);
  // The rotor of the plane p∧q: R = cos(tθ/2) − sin(tθ/2)·B̂ with B̂ = x̂∧ŷ/|x̂∧ŷ|.
  const Multivector x = mv(p) - basis(4), y = mv(q) - basis(4);
  const Multivector B = x ^ y;
  const double s2 = B.normSquared(), theta = std::acos(scalarProduct(x, y)), t = 0.5;
  const Multivector R = Multivector::scalar(4, 1, std::cos(t * theta / 2)) - B / std::sqrt(s2) * std::sin(t * theta / 2);
  CHECK(maxDiff(sandwich(R, mv(p)).grade(1), mid) < 1e-12);
}

TEST(space_mixtures_and_caps) {
  const auto p = sp::atChance(0.3, sp::kB);
  CHECK_NEAR(sp::chance(sp::raise(p, 0.5)), 0.65, 1e-12);
  CHECK_NEAR(sp::chance(sp::lower(p, 0.5)), 0.15, 1e-12);
  CHECK_NEAR(sp::chance(sp::nudge(p, 0.2)), 0.5, 1e-12);
  CHECK_NEAR(sp::chance(sp::nudge(p, -0.2)), 0.1, 1e-12);
  CHECK(sp::channelOf(sp::raise(p, 0.5)) == sp::kB);
  CHECK(sp::inside(sp::atChance(0.71, sp::kA), sp::chanceCap(0.7)));
  CHECK(!sp::inside(sp::atChance(0.69, sp::kA), sp::chanceCap(0.7)));
  const auto c = sp::cap(sp::atLogit(0.0, sp::kA), 0.5);
  CHECK(sp::inside(sp::towards(sp::atLogit(0.0, sp::kA), sp::atLogit(0.0, sp::kB), 0.3), c));
  CHECK(!sp::inside(sp::atLogit(0.0, sp::kB), c));
}

TEST(kernel_engines_agree_on_the_space_model) {
  Value spec = Value::object();
  spec.set("space", Value::object().set("samples", 5000));
  const Value a = kernel::run("space", spec, kernel::Precision::F64);
  const Value b = kernel::run("space", spec, kernel::Precision::F32);
  const auto& ca = a.at("chance").asNumbers();
  const auto& cb = b.at("chance").asNumbers();
  for (std::size_t i = 0; i < ca.size(); ++i) CHECK_NEAR(ca[i], cb[i], 1e-5);
  CHECK(a.at("checks").num("chanceIsLogistic", 1) < 1e-12);
  CHECK(a.at("checks").num("dilationAddsLogits", 1) < 1e-9);
  CHECK(a.at("checks").num("mixtureIsLinear", 1) < 1e-12);
  CHECK(a.at("checks").num("pointsAreNull", 1) < 1e-12);
  CHECK(a.at("checks").num("rotorKeepsChance", 1) < 1e-12);
  CHECK(b.at("checks").num("chanceIsLogistic", 1) < 1e-5);
  // Random points: logits with mean 0 and sd 1.
  CHECK(std::fabs(a.at("random").num("meanLogit", 9)) < 0.05);
  CHECK(std::fabs(a.at("random").num("sdLogit", 9) - 1.0) < 0.05);
  CHECK_NEAR(a.at("random").num("meanLogit", 9), b.at("random").num("meanLogit", 7), 1e-4);
  // The same tiles split across workers give the same result.
  std::vector<Value> parts;
  for (int w = 0; w < 3; ++w) parts.push_back(kernel::runSlices("space", spec, w, 3));
  const Value c = kernel::analyseSlices("space", spec, parts);
  CHECK_NEAR(c.at("random").num("meanLogit", 9), b.at("random").num("meanLogit", 7), 1e-9);
}

TEST(kernel_wgsl_is_assembled) {
  const std::string src = kernel::agentKernel(kernel::model("space"));
  CHECK(src.find("fn agent()") != std::string::npos);
  CHECK(src.find("fn shift(") != std::string::npos);
  CHECK(src.find("ARR(") == std::string::npos);
  CHECK(src.find("@compute") != std::string::npos);
}

TEST(kernel_agents_use_no_wgsl_reserved_words) {
  // Reserved words and keywords of WGSL that are ordinary identifiers in C++.
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
