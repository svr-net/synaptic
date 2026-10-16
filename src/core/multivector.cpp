#include "syn/core/multivector.hpp"

#include <algorithm>
#include <bitset>
#include <cmath>
#include <stdexcept>

namespace syn {

namespace {
// Blade-product signs for the inline algebras, computed once per signature: [p][q][a * 32 + b].
// Built once, before first use, for every signature with p + q ≤ 5 (thread-safe static init).
const double* signTable(int p, int q) {
  static const auto tables = [] {
    std::array<std::array<std::vector<double>, 6>, 6> t;
    for (int pp = 0; pp <= 5; ++pp)
      for (int qq = 0; pp + qq <= 5; ++qq) {
        const std::uint32_t n = std::uint32_t{1} << (pp + qq);
        t[pp][qq].resize(std::size_t{n} * n);
        for (std::uint32_t a = 0; a < n; ++a)
          for (std::uint32_t b = 0; b < n; ++b) t[pp][qq][a * n + b] = Multivector::bladeSign(a, b, pp);
      }
    return t;
  }();
  return tables[p][q].data();
}

void requireSame(const Multivector& a, const Multivector& b) {
  if (!a.sameAlgebra(b)) throw std::invalid_argument("Multivector: operands belong to different algebras");
}
}  // namespace

Multivector::Multivector(int p, int q) : p_(p), q_(q), size_(0) {
  if (p < 0 || q < 0 || p + q > 10) throw std::invalid_argument("Multivector: need p, q >= 0 and p + q <= 10");
  size_ = std::size_t{1} << (p + q);
  if (size_ > small_.size()) large_.assign(size_, 0.0);
}

std::uint32_t Multivector::check(std::uint32_t blade) const {
  if (blade >= size_) throw std::out_of_range("Multivector: blade index out of range");
  return blade;
}

Multivector Multivector::scalar(int p, int q, double s) {
  Multivector m(p, q);
  m.data()[0] = s;
  return m;
}

Multivector Multivector::vector(int p, int q, const std::vector<double>& v) {
  Multivector m(p, q);
  if (static_cast<int>(v.size()) != p + q) throw std::invalid_argument("Multivector::vector: wrong number of components");
  for (int i = 0; i < p + q; ++i) m.data()[std::size_t{1} << i] = v[i];
  return m;
}

Multivector Multivector::basis(int p, int q, int i) {
  Multivector m(p, q);
  if (i < 0 || i >= p + q) throw std::invalid_argument("Multivector::basis: index out of range");
  m.data()[std::size_t{1} << i] = 1.0;
  return m;
}

std::vector<double> Multivector::vectorPart() const {
  std::vector<double> v(dimension());
  for (int i = 0; i < dimension(); ++i) v[i] = data()[std::size_t{1} << i];
  return v;
}

int Multivector::gradeOf(std::uint32_t blade) { return static_cast<int>(std::bitset<32>(blade).count()); }

double Multivector::bladeSign(std::uint32_t a, std::uint32_t b, int p) {
  // Reordering: each vector of b passes every vector of a with a higher index.
  int swaps = 0;
  for (std::uint32_t x = a >> 1; x; x >>= 1) swaps += gradeOf(x & b);
  double s = (swaps & 1) ? -1.0 : 1.0;
  // Metric: vectors common to both blades contract to their square (−1 beyond index p).
  const std::uint32_t common = a & b;
  const std::uint32_t negative = common >> p;
  if (gradeOf(negative) & 1) s = -s;
  return s;
}

Multivector Multivector::operator+(const Multivector& o) const {
  requireSame(*this, o);
  Multivector r = *this;
  for (std::size_t i = 0; i < size_; ++i) r.data()[i] += o.data()[i];
  return r;
}

Multivector Multivector::operator-(const Multivector& o) const {
  requireSame(*this, o);
  Multivector r = *this;
  for (std::size_t i = 0; i < size_; ++i) r.data()[i] -= o.data()[i];
  return r;
}

Multivector Multivector::operator-() const { return *this * -1.0; }

Multivector Multivector::operator*(double s) const {
  Multivector r = *this;
  for (std::size_t i = 0; i < size_; ++i) r.data()[i] *= s;
  return r;
}

Multivector Multivector::operator/(double s) const {
  if (s == 0.0) throw std::invalid_argument("Multivector: division by zero");
  return *this * (1.0 / s);
}

Multivector Multivector::operator*(const Multivector& o) const {
  requireSame(*this, o);
  Multivector r(p_, q_);
  const double* x = data();
  const double* y = o.data();
  double* z = r.data();
  const double* sign = size_ <= 32 ? signTable(p_, q_) : nullptr;
  auto sgn = [&](std::uint32_t a, std::uint32_t b) { return sign ? sign[a * size_ + b] : bladeSign(a, b, p_); };
  for (std::uint32_t a = 0; a < size_; ++a) {
    if (x[a] == 0.0) continue;
    for (std::uint32_t b = 0; b < size_; ++b) {
      if (y[b] == 0.0) continue;
      z[a ^ b] += sgn(a, b) * x[a] * y[b];
    }
  }
  return r;
}

Multivector Multivector::operator^(const Multivector& o) const {
  requireSame(*this, o);
  Multivector r(p_, q_);
  const double* x = data();
  const double* y = o.data();
  double* z = r.data();
  const double* sign = size_ <= 32 ? signTable(p_, q_) : nullptr;
  auto sgn = [&](std::uint32_t a, std::uint32_t b) { return sign ? sign[a * size_ + b] : bladeSign(a, b, p_); };
  for (std::uint32_t a = 0; a < size_; ++a) {
    if (x[a] == 0.0) continue;
    for (std::uint32_t b = 0; b < size_; ++b) {
      if (y[b] == 0.0 || (a & b)) continue;
      z[a | b] += sgn(a, b) * x[a] * y[b];
    }
  }
  return r;
}

Multivector Multivector::operator|(const Multivector& o) const {
  requireSame(*this, o);
  Multivector r(p_, q_);
  const double* x = data();
  const double* y = o.data();
  double* z = r.data();
  const double* sign = size_ <= 32 ? signTable(p_, q_) : nullptr;
  auto sgn = [&](std::uint32_t a, std::uint32_t b) { return sign ? sign[a * size_ + b] : bladeSign(a, b, p_); };
  for (std::uint32_t a = 0; a < size_; ++a) {
    if (x[a] == 0.0) continue;
    for (std::uint32_t b = 0; b < size_; ++b) {
      if (y[b] == 0.0 || (a & b) != a) continue;  // left contraction: a ⊆ b
      z[a ^ b] += sgn(a, b) * x[a] * y[b];
    }
  }
  return r;
}

Multivector Multivector::reverse() const {
  Multivector r = *this;
  for (std::uint32_t a = 0; a < size_; ++a) {
    const int k = gradeOf(a);
    if ((k * (k - 1) / 2) & 1) r.data()[a] = -r.data()[a];
  }
  return r;
}

Multivector Multivector::involute() const {
  Multivector r = *this;
  for (std::uint32_t a = 0; a < size_; ++a)
    if (gradeOf(a) & 1) r.data()[a] = -r.data()[a];
  return r;
}

Multivector Multivector::grade(int k) const {
  Multivector r(p_, q_);
  for (std::uint32_t a = 0; a < size_; ++a)
    if (gradeOf(a) == k) r.data()[a] = data()[a];
  return r;
}

double Multivector::normSquared() const { return scalarProduct(*this, reverse()); }

double Multivector::maxAbs() const {
  double m = 0.0;
  for (std::size_t i = 0; i < size_; ++i) m = std::max(m, std::fabs(data()[i]));
  return m;
}

double scalarProduct(const Multivector& a, const Multivector& b) {
  if (!a.sameAlgebra(b)) throw std::invalid_argument("scalarProduct: operands belong to different algebras");
  // ⟨a b⟩₀ only pairs each blade with itself.
  double s = 0.0;
  for (std::uint32_t i = 0; i < a.size(); ++i) {
    const double x = a[i], y = b[i];
    if (x != 0.0 && y != 0.0) s += Multivector::bladeSign(i, i, a.p()) * x * y;
  }
  return s;
}

Multivector sandwich(const Multivector& v, const Multivector& x) { return v * x * v.reverse(); }

}  // namespace syn
