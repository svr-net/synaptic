#pragma once

// Minimal self-registering test framework (no external dependencies).

#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace syntest {

struct TestCase {
  std::string name;
  std::function<void()> body;
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> tests;
  return tests;
}

struct Registrar {
  Registrar(const char* name, std::function<void()> body) { registry().push_back({name, std::move(body)}); }
};

struct Failure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

inline void fail(const std::string& message, const char* file, int line) {
  std::ostringstream os;
  os << file << ":" << line << ": " << message;
  throw Failure(os.str());
}

}  // namespace syntest

#define SYN_CONCAT_INNER(a, b) a##b
#define SYN_CONCAT(a, b) SYN_CONCAT_INNER(a, b)

#define TEST(name)                                                                             \
  static void SYN_CONCAT(test_, name)();                                                       \
  static ::syntest::Registrar SYN_CONCAT(registrar_, name)(#name, &SYN_CONCAT(test_, name));   \
  static void SYN_CONCAT(test_, name)()

#define CHECK(condition)                                                     \
  do {                                                                       \
    if (!(condition)) ::syntest::fail("CHECK(" #condition ") failed", __FILE__, __LINE__); \
  } while (0)

#define CHECK_NEAR(actual, expected, tolerance)                                                   \
  do {                                                                                            \
    const double syn_a = (actual);                                                                \
    const double syn_e = (expected);                                                              \
    if (!(std::fabs(syn_a - syn_e) <= (tolerance))) {                                             \
      std::ostringstream syn_os;                                                                  \
      syn_os.precision(12);                                                                       \
      syn_os << "CHECK_NEAR(" #actual ", " #expected ") failed: " << syn_a << " vs " << syn_e      \
             << " (tolerance " << (tolerance) << ")";                                             \
      ::syntest::fail(syn_os.str(), __FILE__, __LINE__);                                          \
    }                                                                                             \
  } while (0)

#define CHECK_THROWS(statement)                                                       \
  do {                                                                                \
    bool syn_threw = false;                                                           \
    try {                                                                             \
      statement;                                                                      \
    } catch (...) {                                                                   \
      syn_threw = true;                                                               \
    }                                                                                 \
    if (!syn_threw) ::syntest::fail("expected exception: " #statement, __FILE__, __LINE__); \
  } while (0)
