#pragma once

// A JSON-like value: the one format for every model's specification and result.
//
// Every model of the library reads its settings from a Value and returns its result as a Value,
// so the WebAssembly bindings, the engines and the tests treat all models alike. Numeric arrays
// are stored as std::vector<double> (Float64Array in JavaScript).

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace syn {

class Value {
 public:
  using Object = std::map<std::string, Value>;
  using Array = std::vector<Value>;
  using Numbers = std::vector<double>;

  Value() = default;
  Value(std::nullptr_t) {}
  Value(bool b) : v_(b) {}
  Value(double d) : v_(d) {}
  Value(int i) : v_(static_cast<double>(i)) {}
  Value(unsigned i) : v_(static_cast<double>(i)) {}
  Value(long i) : v_(static_cast<double>(i)) {}
  Value(unsigned long i) : v_(static_cast<double>(i)) {}
  Value(long long i) : v_(static_cast<double>(i)) {}
  Value(unsigned long long i) : v_(static_cast<double>(i)) {}
  Value(const char* s) : v_(std::string(s)) {}
  Value(std::string s) : v_(std::move(s)) {}
  Value(Numbers n) : v_(std::move(n)) {}
  Value(Array a) : v_(std::move(a)) {}
  Value(Object o) : v_(std::move(o)) {}
  Value(const std::vector<std::string>& strings);

  static Value object() { return Value(Object{}); }
  static Value array() { return Value(Array{}); }

  bool isNull() const { return v_.index() == 0; }
  bool isBool() const { return v_.index() == 1; }
  bool isNumber() const { return v_.index() == 2; }
  bool isString() const { return v_.index() == 3; }
  bool isNumbers() const { return v_.index() == 4; }
  bool isArray() const { return v_.index() == 5; }
  bool isObject() const { return v_.index() == 6; }

  bool asBool() const;
  double asNumber() const;
  const std::string& asString() const;
  const Numbers& asNumbers() const;
  const Array& asArray() const;
  const Object& asObject() const;
  Numbers& numbers();
  Array& items();
  Object& fields();

  /** Number of elements of an array, numeric array or object (0 otherwise). */
  std::size_t size() const;
  /** Element i of an array or numeric array. */
  Value operator[](std::size_t i) const;

  bool has(const std::string& key) const;
  /** Field of an object; a null Value if absent (or if this is not an object). */
  const Value& at(const std::string& key) const;
  /** Sets a field (this becomes an object if it was null). Returns *this for chaining. */
  Value& set(const std::string& key, Value v);
  /** Appends to an array (this becomes an array if it was null). */
  Value& push(Value v);

  // Lookups with defaults: the field if present and of the right type, else the fallback.
  double num(const std::string& key, double fallback) const;
  int integer(const std::string& key, int fallback) const;
  bool flag(const std::string& key, bool fallback) const;
  std::string str(const std::string& key, const std::string& fallback) const;
  Numbers nums(const std::string& key, const Numbers& fallback) const;
  /** A sub-object (an empty object if absent). */
  const Value& sub(const std::string& key) const;

  /** Compact JSON (numbers with 17 significant digits). */
  std::string json() const;

 private:
  std::variant<std::nullptr_t, bool, double, std::string, Numbers, Array, Object> v_;
};

}  // namespace syn
