#pragma once

// A small Clifford (geometric) algebra Cl(p, q) with dense coefficients over basis blades.
//
// Basis vectors e_1..e_p square to +1 and e_{p+1}..e_{p+q} square to −1. A basis blade is a
// bit mask of the vectors it contains (bit i ↔ e_{i+1}), and a multivector stores one
// coefficient per blade. Products iterate only over non-zero coefficients, so sparse objects
// (vectors, rotors, boosts) are cheap even in larger algebras. Algebras of dimension ≤ 5 keep
// their coefficients inline (no heap allocation), which covers every model in the library.

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace syn {

class Multivector {
 public:
  /** The zero multivector of Cl(p, q); p + q ≤ 10. */
  Multivector(int p, int q);

  static Multivector scalar(int p, int q, double s);
  /** Grade-1 element Σ v_i e_{i+1}; v.size() must be p + q. */
  static Multivector vector(int p, int q, const std::vector<double>& v);
  /** The basis vector e_{i+1} (0-based index i). */
  static Multivector basis(int p, int q, int i);

  int p() const { return p_; }
  int q() const { return q_; }
  int dimension() const { return p_ + q_; }
  std::size_t size() const { return size_; }

  double& operator[](std::uint32_t blade) { return data()[check(blade)]; }
  double operator[](std::uint32_t blade) const { return data()[check(blade)]; }
  double scalarPart() const { return data()[0]; }
  /** All coefficients, indexed by blade (size() of them). */
  const double* coefficients() const { return data(); }
  /** Coefficients of the grade-1 part, in basis order. */
  std::vector<double> vectorPart() const;

  Multivector operator+(const Multivector& o) const;
  Multivector operator-(const Multivector& o) const;
  Multivector operator-() const;
  Multivector operator*(double s) const;
  Multivector operator/(double s) const;
  /** Geometric product. */
  Multivector operator*(const Multivector& o) const;
  /** Outer (wedge) product. */
  Multivector operator^(const Multivector& o) const;
  /** Left contraction a ⌋ b. */
  Multivector operator|(const Multivector& o) const;

  /** Reverse: reverses the order of the vectors in every blade. */
  Multivector reverse() const;
  /** Grade involution: negates odd grades. */
  Multivector involute() const;
  Multivector grade(int k) const;
  /** ⟨a ã⟩₀: squared norm (may be negative in a mixed signature). */
  double normSquared() const;
  /** Largest absolute coefficient. */
  double maxAbs() const;
  bool sameAlgebra(const Multivector& o) const { return p_ == o.p_ && q_ == o.q_; }

  /** Sign (±1) of the product of two basis blades, including the metric; 0 never occurs here. */
  static double bladeSign(std::uint32_t a, std::uint32_t b, int p);
  static int gradeOf(std::uint32_t blade);

 private:
  static constexpr int kInlineDimension = 5;
  std::uint32_t check(std::uint32_t blade) const;
  double* data() { return size_ <= small_.size() ? small_.data() : large_.data(); }
  const double* data() const { return size_ <= small_.size() ? small_.data() : large_.data(); }

  int p_, q_;
  std::size_t size_;
  std::array<double, std::size_t{1} << kInlineDimension> small_{};
  std::vector<double> large_;
};

/** Scalar product ⟨a b⟩₀; for two vectors this is their inner product. */
double scalarProduct(const Multivector& a, const Multivector& b);

/** Sandwich V X Ṽ (versor action). */
Multivector sandwich(const Multivector& v, const Multivector& x);

}  // namespace syn
