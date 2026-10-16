#include "syn/core/elliptic_cga.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace syn {

EllipticCGA::EllipticCGA(int n) : n_(n) {
  if (n < 1 || n + 2 > 10) throw std::invalid_argument("EllipticCGA: need 1 <= n <= 8");
}

Multivector EllipticCGA::e() const { return Multivector::basis(n_ + 1, 1, eIndex()); }

std::vector<double> EllipticCGA::unit(const std::vector<double>& v) const {
  if (static_cast<int>(v.size()) != n_ + 1) throw std::invalid_argument("EllipticCGA: direction must have n + 1 components");
  double s = 0.0;
  for (double x : v) s += x * x;
  if (!(s > 0.0)) throw std::invalid_argument("EllipticCGA: zero direction");
  std::vector<double> u(v);
  const double inv = 1.0 / std::sqrt(s);
  for (double& x : u) x *= inv;
  return u;
}

Multivector EllipticCGA::point(const std::vector<double>& direction) const {
  std::vector<double> c = unit(direction);
  c.push_back(1.0);  // + e
  return Multivector::vector(n_ + 1, 1, c);
}

Multivector EllipticCGA::pointAt(const std::vector<double>& from, const std::vector<double>& towards, double phi) const {
  const Multivector a = point(from), b = point(towards);
  const Multivector plane = (a - e()) ^ (b - e());  // Euclidean bivector of the two directions
  const double s2 = plane.normSquared();
  if (s2 < 1e-24) throw std::invalid_argument("EllipticCGA::pointAt: directions must not be parallel");
  const Multivector rotor = Multivector::scalar(n_ + 1, 1, std::cos(phi / 2)) - plane / std::sqrt(s2) * std::sin(phi / 2);
  return move(rotor, a);
}

double EllipticCGA::weight(const Multivector& x) const { return -dot(x, e()); }

Multivector EllipticCGA::normalise(const Multivector& x) const {
  const double w = weight(x);
  if (std::fabs(w) < 1e-300) throw std::invalid_argument("EllipticCGA::normalise: zero weight");
  return x / w;
}

std::vector<double> EllipticCGA::direction(const Multivector& x) const {
  const Multivector p = normalise(x);
  std::vector<double> v = p.vectorPart();
  v.pop_back();
  return v;
}

bool EllipticCGA::isPoint(const Multivector& x, double tol) const {
  if ((x - x.grade(1)).maxAbs() > tol) return false;
  return std::fabs(dot(x, x)) <= tol * std::max(1.0, weight(x) * weight(x));
}

double EllipticCGA::dot(const Multivector& a, const Multivector& b) const {
  // Inner product of two vectors: Σ e_i² a_i b_i over the n + 2 basis vectors (e² = −1 last).
  if (!a.sameAlgebra(b) || a.p() != n_ + 1 || a.q() != 1) throw std::invalid_argument("EllipticCGA::dot: wrong algebra");
  const double* x = a.coefficients();
  const double* y = b.coefficients();
  double s = 0.0;
  for (int i = 0; i <= n_; ++i) s += x[std::size_t{1} << i] * y[std::size_t{1} << i];
  return s - x[std::size_t{1} << eIndex()] * y[std::size_t{1} << eIndex()];
}

double EllipticCGA::angle(const Multivector& x, const Multivector& y) const {
  const Multivector a = normalise(x), b = normalise(y);
  const double cosTheta = 1.0 + dot(a, b);                               // X·Y = cos θ − 1
  const double sinTheta = std::sqrt(std::max(0.0, ((a - e()) ^ (b - e())).normSquared()));  // |x ∧ y|
  return std::atan2(sinTheta, cosTheta);
}

Multivector EllipticCGA::cap(const Multivector& centre, double rho) const {
  if (!(rho > 0.0 && rho < M_PI)) throw std::invalid_argument("EllipticCGA::cap: radius must lie in (0, π)");
  const Multivector c = normalise(centre);
  // c + e is the centre point; c + cos ρ e = (c + e) − (1 − cos ρ) e.
  return c - e() * (1.0 - std::cos(rho));
}

double EllipticCGA::capSide(const Multivector& x, const Multivector& s) const { return dot(x, s); }

Multivector EllipticCGA::rotorTowards(const Multivector& a, const Multivector& b, double t) const {
  const Multivector ea = normalise(a) - e(), eb = normalise(b) - e();  // Euclidean parts
  const Multivector plane = ea ^ eb;
  const double s2 = plane.normSquared();  // ⟨B B̃⟩₀ = sin²θ for a Euclidean bivector
  const double theta = angle(a, b);
  if (s2 < 1e-24 || theta < 1e-12) return Multivector::scalar(n_ + 1, 1, 1.0);
  const Multivector unitB = plane / std::sqrt(s2);
  return Multivector::scalar(n_ + 1, 1, std::cos(t * theta / 2)) - unitB * std::sin(t * theta / 2);
}

Multivector EllipticCGA::dilation(const std::vector<double>& pole, double beta) const {
  std::vector<double> nvec = unit(pole);
  nvec.push_back(0.0);
  const Multivector ne = Multivector::vector(n_ + 1, 1, nvec) * e();  // (n e)² = +1
  return Multivector::scalar(n_ + 1, 1, std::cosh(beta / 2)) + ne * std::sinh(beta / 2);
}

Multivector EllipticCGA::move(const Multivector& versor, const Multivector& x) const {
  return normalise(sandwich(versor, x).grade(1));
}

Multivector EllipticCGA::centre(const std::vector<Multivector>& points) const {
  if (points.empty()) throw std::invalid_argument("EllipticCGA::centre: no points");
  Multivector sum(n_ + 1, 1);
  for (const auto& p : points) sum = sum + normalise(p);
  std::vector<double> v = sum.vectorPart();
  v.pop_back();
  return point(v);
}

Multivector EllipticCGA::liftStereographic(const std::vector<double>& u, double scale) const {
  if (static_cast<int>(u.size()) != n_) throw std::invalid_argument("liftStereographic: need n coordinates");
  if (!(scale > 0.0)) throw std::invalid_argument("liftStereographic: scale must be positive");
  double r2 = 0.0;
  for (double x : u) r2 += (x / scale) * (x / scale);
  std::vector<double> y;
  for (double x : u) y.push_back(2.0 * (x / scale) / (1.0 + r2));
  y.push_back((1.0 - r2) / (1.0 + r2));  // u = 0 ↦ the pole +e_{n+1}
  return point(y);
}

std::vector<double> EllipticCGA::projectStereographic(const Multivector& x, double scale) const {
  const std::vector<double> y = direction(x);
  const double denom = 1.0 + y[n_];
  if (denom < 1e-12) throw std::invalid_argument("projectStereographic: the projection pole has no image");
  std::vector<double> u;
  for (int i = 0; i < n_; ++i) u.push_back(scale * y[i] / denom);
  return u;
}

}  // namespace syn
