// Two devices' results of one model, number by number (see compareResults in syn/kernel/kernel.hpp).
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "syn/kernel/kernel.hpp"

namespace syn::kernel {

namespace {

struct Tolerance {
  double flips = 0, relative = 2e-3;
};

// The models whose agents take threshold decisions (an argmax among near-ties may flip one agent's path) and the
// ratios of small variances.
Tolerance toleranceOf(const std::string& name) {
  static const std::map<std::string, Tolerance> kAllow = {
      {"equilibration", {0.02, 2e-3}}, {"fgaAlgebra", {0.01, 2e-3}}, {"sensitivity", {0, 0.03}}, {"readingStudent", {0, 0.05}}};
  const auto it = kAllow.find(name);
  return it == kAllow.end() ? Tolerance{} : it->second;
}

// Timings and device names; and the picks a summary makes among values it reports (ranked, top, best...): the values
// are compared, while among near-ties rounding may pick another.
bool skipped(const std::string& k) {
  static const std::set<std::string> kSkipped = {"engineMs", "adapter", "ranked", "top", "best", "bestSingle", "bestTutoring",
                                                 "bestInstrument", "bestZone", "bestLasting"};
  return kSkipped.count(k) > 0;
}

void leaves(const Value& v, const std::string& path, std::vector<std::pair<std::string, double>>& out) {
  if (v.isNumber()) {
    out.emplace_back(path, v.asNumber());
  } else if (v.isNumbers()) {
    const auto& n = v.asNumbers();
    for (std::size_t i = 0; i < n.size(); ++i) out.emplace_back(path + "[" + std::to_string(i) + "]", n[i]);
  } else if (v.isArray()) {
    const auto& a = v.asArray();
    for (std::size_t i = 0; i < a.size(); ++i) leaves(a[i], path + "[" + std::to_string(i) + "]", out);
  } else if (v.isObject()) {
    for (const auto& [k, x] : v.asObject())
      if (!skipped(k)) leaves(x, path + "." + k, out);
  }
}

}  // namespace

Value compareResults(const std::string& name, const Value& a, const Value& b) {
  std::vector<std::pair<std::string, double>> x, y;
  leaves(a, "", x);
  leaves(b, "", y);
  if (x.size() != y.size())
    return Value::object().set("error", std::to_string(x.size()) + " values against " + std::to_string(y.size()));
  const Tolerance t = toleranceOf(name);
  Value worst = Value::object().set("values", static_cast<double>(y.size())).set("excess", 0.0);
  double most = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double u = x[i].second, v = y[i].second;
    double excess;
    if (!std::isfinite(u) || !std::isfinite(v)) excess = std::isfinite(u) == std::isfinite(v) ? 0.0 : INFINITY;
    else excess = std::fabs(u - v) / (2e-3 + t.flips + t.relative * std::fabs(v));
    if (excess > most) {
      most = excess;
      worst.set("excess", std::isfinite(excess) ? excess : 1e300).set("path", x[i].first).set("x", u).set("y", v);
    }
  }
  return worst;
}

}  // namespace syn::kernel
