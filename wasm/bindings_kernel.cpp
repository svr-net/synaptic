// Generic Embind entry points: every model of the library through the same five calls.
//
//   models()              [{ name, title }]
//   kernelSources({model}) { agent, reduce, quantile }   WGSL sources
//   runModel({model, ...}) the result, on WebAssembly (64-bit floats)
//   modelPlans({model, ...})  the GPU plans: buffers, dispatch sizes, buffer sizes
//   modelAnalyse({model, ..., outputs: [{means, quantiles}]})   the result from GPU read-backs
//   modelSlices({model, ..., part, parts})                     CPU kernel tiles (32-bit floats)
//   modelAnalyseSlices({model, ..., parts: [slices of every worker]})
// Specifications and results are syn::Value trees converted to and from plain JavaScript.
// Errors are returned as { error } rather than thrown across the boundary.

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <string>
#include <vector>

#include "syn/core/value.hpp"
#include "syn/kernel/kernel.hpp"

using emscripten::val;
using syn::Value;

namespace {

Value fromJs(const val& v);

Value fromObject(const val& v) {
  Value out = Value::object();
  const val keys = val::global("Object").call<val>("keys", v);
  const int n = keys["length"].as<int>();
  for (int i = 0; i < n; ++i) {
    const std::string k = keys[i].as<std::string>();
    out.set(k, fromJs(v[k]));
  }
  return out;
}

Value fromArray(const val& v) {
  const int n = v["length"].as<int>();
  bool numeric = true;
  for (int i = 0; i < n && numeric; ++i) numeric = v[i].isNumber();
  if (numeric) return Value(emscripten::convertJSArrayToNumberVector<double>(v));
  Value out = Value::array();
  for (int i = 0; i < n; ++i) out.push(fromJs(v[i]));
  return out;
}

Value fromJs(const val& v) {
  if (v.isUndefined() || v.isNull()) return Value();
  if (v.isNumber()) return Value(v.as<double>());
  if (v.isTrue()) return Value(true);
  if (v.isFalse()) return Value(false);
  if (v.isString()) return Value(v.as<std::string>());
  if (val::global("ArrayBuffer").call<bool>("isView", v)) return Value(emscripten::convertJSArrayToNumberVector<double>(v));
  if (val::global("Array").call<bool>("isArray", v)) return fromArray(v);
  return fromObject(v);
}

val toJs(const Value& v) {
  if (v.isNull()) return val::null();
  if (v.isBool()) return val(v.asBool());
  if (v.isNumber()) return val(v.asNumber());
  if (v.isString()) return val(v.asString());
  if (v.isNumbers()) {
    const auto& n = v.asNumbers();
    return val(emscripten::typed_memory_view(n.size(), n.data())).call<val>("slice");
  }
  if (v.isArray()) {
    val out = val::array();
    for (const auto& x : v.asArray()) out.call<void>("push", toJs(x));
    return out;
  }
  val out = val::object();
  for (const auto& [k, x] : v.asObject()) out.set(k, toJs(x));
  return out;
}

template <class T>
val typed(const std::vector<T>& v, const char* type) {
  val view(emscripten::typed_memory_view(v.size(), v.data()));
  return val::global(type).new_(view);
}

val errorOf(const std::string& message) {
  val e = val::object();
  e.set("error", message);
  return e;
}

std::string modelName(const Value& spec) {
  const std::string name = spec.str("model", "");
  if (name.empty()) throw std::invalid_argument("the specification names no model");
  return name;
}

val models() {
  val out = val::array();
  for (const auto* m : syn::kernel::models()) {
    val x = val::object();
    x.set("name", m->name);
    x.set("title", m->title);
    out.call<void>("push", x);
  }
  return out;
}

val kernelSources(val spec) {
  try {
    const auto& m = syn::kernel::model(modelName(fromJs(spec)));
    val out = val::object();
    out.set("agent", syn::kernel::agentKernel(m));
    out.set("reduce", syn::kernel::reduceKernel());
    out.set("quantile", syn::kernel::quantileKernel());
    return out;
  } catch (const std::exception& e) {
    return errorOf(e.what());
  }
}

val runModel(val spec) {
  try {
    const Value s = fromJs(spec);
    return toJs(syn::kernel::run(modelName(s), s, syn::kernel::Precision::F64));
  } catch (const std::exception& e) {
    return errorOf(e.what());
  }
}

val planToJs(const syn::kernel::Plan& p) {
  val x = val::object();
  x.set("header", typed(p.header(), "Uint32Array"));
  x.set("constants", typed(p.constants(), "Float32Array"));
  x.set("seeds", typed(p.seeds, "Uint32Array"));
  val dispatch = val::object();
  dispatch.set("agents", p.workgroups());
  dispatch.set("jobs", p.jobs);
  dispatch.set("reduce", static_cast<std::uint32_t>((p.sumFloats() + 63) / 64));
  dispatch.set("quantiles", p.quantiles);
  x.set("dispatch", dispatch);
  val bytes = val::object();
  bytes.set("scratch", static_cast<double>(p.scratchFloats() * 4));
  bytes.set("partials", static_cast<double>(p.partialFloats() * 4));
  bytes.set("sums", static_cast<double>(p.sumFloats() * 4));
  bytes.set("quantiles", static_cast<double>(p.quantileFloats() * 4));
  bytes.set("keep", static_cast<double>(p.keepAgents > 0 && p.quantiles > 0 ? p.scratchFloats() * 4 : 0));
  x.set("bytes", bytes);
  x.set("agents", p.agents);
  x.set("jobs", p.jobs);
  return x;
}

val modelPlans(val spec) {
  try {
    const Value s = fromJs(spec);
    const auto& m = syn::kernel::model(modelName(s));
    val out = val::array();
    for (const auto& p : m.compile(s)) out.call<void>("push", planToJs(p));
    return out;
  } catch (const std::exception& e) {
    return errorOf(e.what());
  }
}

val modelAnalyse(val spec) {
  try {
    const Value s = fromJs(spec);
    std::vector<syn::kernel::Output> outputs;
    std::vector<std::vector<double>> scratch;
    for (const auto& o : s.at("outputs").asArray()) {
      syn::kernel::Output out;
      out.means = o.nums("means", {});
      out.quantiles = o.nums("quantiles", {});
      outputs.push_back(std::move(out));
      scratch.push_back(o.nums("scratch", {}));
    }
    return toJs(syn::kernel::analyseOutputs(modelName(s), s, std::move(outputs), scratch));
  } catch (const std::exception& e) {
    return errorOf(e.what());
  }
}

val modelSlices(val spec) {
  try {
    const Value s = fromJs(spec);
    return toJs(syn::kernel::runSlices(modelName(s), s, static_cast<std::size_t>(s.num("part", 0)), static_cast<std::size_t>(s.num("parts", 1))));
  } catch (const std::exception& e) {
    return errorOf(e.what());
  }
}

val modelAnalyseSlices(val spec) {
  try {
    const Value s = fromJs(spec);
    std::vector<Value> parts;
    for (const auto& p : s.at("parts").asArray()) parts.push_back(p);
    return toJs(syn::kernel::analyseSlices(modelName(s), s, parts));
  } catch (const std::exception& e) {
    return errorOf(e.what());
  }
}

}  // namespace

EMSCRIPTEN_BINDINGS(syn_kernel) {
  emscripten::function("models", &models);
  emscripten::function("kernelSources", &kernelSources);
  emscripten::function("runModel", &runModel);
  emscripten::function("modelPlans", &modelPlans);
  emscripten::function("modelAnalyse", &modelAnalyse);
  emscripten::function("modelSlices", &modelSlices);
  emscripten::function("modelAnalyseSlices", &modelAnalyseSlices);
}
