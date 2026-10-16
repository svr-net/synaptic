// Tour of the library: every model, run as its fused kernel on the CPU (64-bit floats), with the
// headline numbers of its result. `learning_theories_demo <model> [--json]` prints one model's
// whole result as JSON.

#include <cstdio>
#include <cstring>
#include <string>

#include "syn/core/value.hpp"
#include "syn/kernel/kernel.hpp"

using namespace syn;

namespace {

// The scalars at the top of a result (or one level down), as "name value" pairs.
void headline(const std::string& prefix, const Value& v, int depth, int& shown) {
  if (!v.isObject()) return;
  for (const auto& [key, field] : v.asObject()) {
    if (shown >= 6) return;
    if (key == "engineMs" || key == "plans" || key == "jobs") continue;
    if (field.isNumber()) {
      std::printf("    %-40s %10.4f\n", (prefix + key).c_str(), field.asNumber());
      ++shown;
    } else if (depth == 0 && field.isObject()) {
      headline(prefix + key + ".", field, depth + 1, shown);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  Value spec = Value::object();
  Value sim = Value::object();
  sim.set("learners", 1000).set("seed", 7);
  spec.set("sim", sim);
  if (argc > 1) {
    const Value r = kernel::run(argv[1], spec);
    std::printf("%s\n", r.json().c_str());
    return 0;
  }
  for (const kernel::Model* m : kernel::models()) {
    std::printf("== %s: %s\n", m->name.c_str(), m->title.c_str());
    int shown = 0;
    headline("", kernel::run(m->name, spec), 0, shown);
  }
  return 0;
}
