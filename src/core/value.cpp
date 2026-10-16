#include "syn/core/value.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace syn {

namespace {
const Value& nullValue() {
  static const Value v;
  return v;
}
const Value& emptyObject() {
  static const Value v = Value::object();
  return v;
}
}  // namespace

Value::Value(const std::vector<std::string>& strings) : v_(Array{}) {
  for (const auto& s : strings) std::get<Array>(v_).emplace_back(s);
}

bool Value::asBool() const {
  if (isBool()) return std::get<bool>(v_);
  if (isNumber()) return std::get<double>(v_) != 0.0;
  throw std::invalid_argument("Value: not a boolean");
}
double Value::asNumber() const {
  if (isNumber()) return std::get<double>(v_);
  if (isBool()) return std::get<bool>(v_) ? 1.0 : 0.0;
  throw std::invalid_argument("Value: not a number");
}
const std::string& Value::asString() const {
  if (!isString()) throw std::invalid_argument("Value: not a string");
  return std::get<std::string>(v_);
}
const Value::Numbers& Value::asNumbers() const {
  if (!isNumbers()) throw std::invalid_argument("Value: not a numeric array");
  return std::get<Numbers>(v_);
}
const Value::Array& Value::asArray() const {
  if (!isArray()) throw std::invalid_argument("Value: not an array");
  return std::get<Array>(v_);
}
const Value::Object& Value::asObject() const {
  if (!isObject()) throw std::invalid_argument("Value: not an object");
  return std::get<Object>(v_);
}
Value::Numbers& Value::numbers() {
  if (isNull()) v_ = Numbers{};
  if (!isNumbers()) throw std::invalid_argument("Value: not a numeric array");
  return std::get<Numbers>(v_);
}
Value::Array& Value::items() {
  if (isNull()) v_ = Array{};
  if (!isArray()) throw std::invalid_argument("Value: not an array");
  return std::get<Array>(v_);
}
Value::Object& Value::fields() {
  if (isNull()) v_ = Object{};
  if (!isObject()) throw std::invalid_argument("Value: not an object");
  return std::get<Object>(v_);
}

std::size_t Value::size() const {
  if (isArray()) return std::get<Array>(v_).size();
  if (isNumbers()) return std::get<Numbers>(v_).size();
  if (isObject()) return std::get<Object>(v_).size();
  return 0;
}

Value Value::operator[](std::size_t i) const {
  if (isArray()) return std::get<Array>(v_).at(i);
  if (isNumbers()) return Value(std::get<Numbers>(v_).at(i));
  throw std::invalid_argument("Value: not an array");
}

bool Value::has(const std::string& key) const {
  return isObject() && std::get<Object>(v_).count(key) > 0 && !std::get<Object>(v_).at(key).isNull();
}

const Value& Value::at(const std::string& key) const {
  if (!isObject()) return nullValue();
  const auto& o = std::get<Object>(v_);
  const auto it = o.find(key);
  return it == o.end() ? nullValue() : it->second;
}

Value& Value::set(const std::string& key, Value v) {
  fields()[key] = std::move(v);
  return *this;
}

Value& Value::push(Value v) {
  items().push_back(std::move(v));
  return *this;
}

double Value::num(const std::string& key, double fallback) const {
  const Value& v = at(key);
  return v.isNumber() || v.isBool() ? v.asNumber() : fallback;
}
int Value::integer(const std::string& key, int fallback) const {
  const Value& v = at(key);
  return v.isNumber() ? static_cast<int>(std::lround(v.asNumber())) : fallback;
}
bool Value::flag(const std::string& key, bool fallback) const {
  const Value& v = at(key);
  return v.isBool() || v.isNumber() ? v.asBool() : fallback;
}
std::string Value::str(const std::string& key, const std::string& fallback) const {
  const Value& v = at(key);
  return v.isString() ? v.asString() : fallback;
}
Value::Numbers Value::nums(const std::string& key, const Numbers& fallback) const {
  const Value& v = at(key);
  if (v.isNumbers()) return v.asNumbers();
  if (v.isArray()) {
    Numbers out;
    for (const auto& x : v.asArray()) out.push_back(x.asNumber());
    return out;
  }
  return fallback;
}
const Value& Value::sub(const std::string& key) const {
  const Value& v = at(key);
  return v.isObject() ? v : emptyObject();
}

namespace {
void quote(std::string& out, const std::string& s) {
  out += '"';
  for (char c : s) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (static_cast<unsigned char>(c) < 0x20) {
      char buf[8];
      std::snprintf(buf, sizeof buf, "\\u%04x", c);
      out += buf;
    } else {
      out += c;
    }
  }
  out += '"';
}
void number(std::string& out, double d) {
  if (!std::isfinite(d)) {
    out += "null";
    return;
  }
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.17g", d);
  out += buf;
}
void write(std::string& out, const Value& v) {
  if (v.isNull()) out += "null";
  else if (v.isBool()) out += v.asBool() ? "true" : "false";
  else if (v.isNumber()) number(out, v.asNumber());
  else if (v.isString()) quote(out, v.asString());
  else if (v.isNumbers()) {
    out += '[';
    bool first = true;
    for (double d : v.asNumbers()) {
      if (!first) out += ',';
      first = false;
      number(out, d);
    }
    out += ']';
  } else if (v.isArray()) {
    out += '[';
    bool first = true;
    for (const auto& x : v.asArray()) {
      if (!first) out += ',';
      first = false;
      write(out, x);
    }
    out += ']';
  } else {
    out += '{';
    bool first = true;
    for (const auto& [k, x] : v.asObject()) {
      if (!first) out += ',';
      first = false;
      quote(out, k);
      out += ':';
      write(out, x);
    }
    out += '}';
  }
}
}  // namespace

std::string Value::json() const {
  std::string out;
  write(out, *this);
  return out;
}

}  // namespace syn
