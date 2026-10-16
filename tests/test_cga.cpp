#include <cmath>

#include "syn/core/elliptic_cga.hpp"
#include "syn/core/math.hpp"
#include "syn/core/multivector.hpp"
#include "test_framework.hpp"

using namespace syn;

TEST(clifford_algebra_axioms) {
  // Cl(2, 1): e1² = e2² = 1, e3² = −1, distinct vectors anticommute.
  const auto e1 = Multivector::basis(2, 1, 0), e2 = Multivector::basis(2, 1, 1), e3 = Multivector::basis(2, 1, 2);
  CHECK_NEAR((e1 * e1).scalarPart(), 1.0, 1e-15);
  CHECK_NEAR((e3 * e3).scalarPart(), -1.0, 1e-15);
  CHECK_NEAR(((e1 * e2) + (e2 * e1)).maxAbs(), 0.0, 1e-15);
  CHECK_NEAR(((e1 * e2) - (e1 ^ e2)).maxAbs(), 0.0, 1e-15);
  // Associativity and the vector identity ab = a·b + a∧b.
  const auto a = Multivector::vector(2, 1, {0.3, -1.2, 0.7}), b = Multivector::vector(2, 1, {1.1, 0.4, -0.5}),
             c = Multivector::vector(2, 1, {-0.2, 0.9, 1.3});
  CHECK_NEAR((((a * b) * c) - (a * (b * c))).maxAbs(), 0.0, 1e-12);
  CHECK_NEAR(((a * b) - (Multivector::scalar(2, 1, scalarProduct(a, b)) + (a ^ b))).maxAbs(), 0.0, 1e-12);
  CHECK_NEAR(scalarProduct(a, b), 0.3 * 1.1 - 1.2 * 0.4 - 0.7 * -0.5, 1e-12);
  CHECK_NEAR(((a * b).reverse() - (b * a)).maxAbs(), 0.0, 1e-12);
  CHECK_NEAR((a | (b ^ c)).grade(1).maxAbs() > 0 ? 1.0 : 0.0, 1.0, 0.0);
  CHECK_THROWS(Multivector(9, 2));
  CHECK_THROWS(a * Multivector::basis(3, 0, 0));
}

TEST(elliptic_points_are_null_and_measure_angles) {
  EllipticCGA g(2);
  const auto x = g.point({1, 0, 0}), y = g.point({0, 1, 0}), z = g.point({0, 0, 3});
  CHECK(g.isPoint(x) && g.isPoint(z));
  CHECK_NEAR(g.dot(x, x), 0.0, 1e-15);
  CHECK_NEAR(g.dot(x, y), std::cos(M_PI / 2) - 1.0, 1e-15);
  CHECK_NEAR(g.angle(x, y), M_PI / 2, 1e-12);
  CHECK_NEAR(g.angle(x, g.point({-1, 0, 0})), M_PI, 1e-7);
  CHECK_NEAR(g.weight(x * 2.5), 2.5, 1e-15);
  CHECK_NEAR(g.angle(x * 2.5, y), M_PI / 2, 1e-12);  // homogeneous: weight does not matter
  for (double phi : {0.1, 1.0, 2.5}) CHECK_NEAR(g.angle(x, g.pointAt({1, 0, 0}, {0, 1, 0}, phi)), phi, 1e-12);
}

TEST(caps_test_membership_by_inner_product) {
  EllipticCGA g(2);
  const auto c = g.point({0, 0, 1});
  const auto s = g.cap(c, 0.8);
  CHECK_NEAR(g.dot(s, s), std::sin(0.8) * std::sin(0.8), 1e-12);  // a real sphere
  for (double phi : {0.1, 0.5, 0.79, 0.81, 1.5, 3.0}) {
    const auto x = g.pointAt({0, 0, 1}, {1, 0, 0}, phi);
    CHECK_NEAR(g.capSide(x, s), std::cos(phi) - std::cos(0.8), 1e-12);
    CHECK(g.inside(x, s) == (phi <= 0.8));
  }
}

TEST(rotors_are_isometries) {
  EllipticCGA g(3);
  const auto a = g.point({1, 0.2, -0.3, 0.5}), b = g.point({-0.4, 1, 0.7, 0.1}), c = g.point({0.3, -0.6, 1, 0.2});
  const double theta = g.angle(a, b);
  for (double t : {0.0, 0.25, 0.5, 1.0}) {
    const auto r = g.rotorTowards(a, b, t);
    CHECK_NEAR((r * r.reverse()).scalarPart(), 1.0, 1e-12);
    const auto at = g.move(r, a);
    CHECK(g.isPoint(at));
    CHECK_NEAR(g.angle(a, at), t * theta, 1e-9);
    CHECK_NEAR(g.angle(at, b), (1 - t) * theta, 1e-9);
    // Angles between moved points are preserved.
    CHECK_NEAR(g.angle(g.move(r, b), g.move(r, c)), g.angle(b, c), 1e-9);
  }
}

TEST(dilations_rescale_the_half_angle_tangent) {
  EllipticCGA g(1);
  const std::vector<double> pole = {1, 0};
  for (double beta : {-1.3, -0.2, 0.0, 0.7}) {
    const auto d = g.dilation(pole, beta);
    CHECK_NEAR((d * d.reverse()).scalarPart(), 1.0, 1e-12);
    for (double phi : {0.2, 1.0, 2.0, 3.0}) {
      const auto x = g.pointAt(pole, {0, 1}, phi);
      const auto y = g.move(d, x);
      CHECK(g.isPoint(y));
      CHECK_NEAR(std::tan(g.angle(g.point(pole), y) / 2), std::exp(beta) * std::tan(phi / 2), 1e-9);
    }
    // ±pole are fixed points.
    CHECK_NEAR(g.angle(g.move(d, g.point(pole)), g.point(pole)), 0.0, 1e-7);
  }
}

TEST(stereographic_lift_round_trip_and_centre) {
  EllipticCGA g(3);
  const std::vector<double> u = {0.4, -1.2, 2.0};
  const auto x = g.liftStereographic(u, 2.0);
  CHECK(g.isPoint(x));
  const auto back = g.projectStereographic(x, 2.0);
  for (int i = 0; i < 3; ++i) CHECK_NEAR(back[i], u[i], 1e-12);
  CHECK_NEAR(g.angle(g.liftStereographic({0, 0, 0}), g.point({0, 0, 0, 1})), 0.0, 1e-12);
  // The centre of points spread symmetrically around a pole is the pole.
  std::vector<Multivector> ring;
  for (int k = 0; k < 6; ++k) ring.push_back(g.pointAt({0, 0, 0, 1}, {std::cos(k * M_PI / 3), std::sin(k * M_PI / 3), 0, 0}, 0.4));
  CHECK_NEAR(g.angle(g.centre(ring), g.point({0, 0, 0, 1})), 0.0, 1e-7);
}
