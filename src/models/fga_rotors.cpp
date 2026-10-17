// Learning coercions as rotors (functional geometric algebra, Pustejovsky 2026, §6.2, §6.4, §9.3).
//
// Regular polysemy — chicken, lamb, turkey and salmon as animals and as food — is a rotor family
// in the paper: one rotation plane shared by the alternation, an angle that differs a little from
// word to word. §9.3 proposes the experiment run here: learn the contextual modulation once as a
// rotor (grade-preserving, norm-preserving, invertible) and once as a learned linear projection
// (the Q/K/V of a transformer), from a few examples, and compare how they generalise to the words
// of the family they never saw. An agent is a learner with its own world: a hidden plane of
// V_E = ⟨A, B, C⟩, eight nouns whose animal senses are points of the sphere of V_E, and food
// senses rotated in that plane by the family's angle plus a word-specific deviation. The rotor
// learner moves a rotor of Cl(4, 1) by gradient steps in all ten coordinate planes of the space —
// six elliptic, four hyperbolic (those containing e) — each step the exact rotor of its plane; the
// projection learner moves a 5 × 5 matrix from the identity. Jobs: one to six training nouns.
#include <algorithm>
#include <string>

#include "syn/models/models.hpp"
#include "syn/space/algebra.hpp"

// The agent program, in WGSL.
constexpr const char* rotors_agent = R"wgsl(
  let train = u32(J(0u));
  let nouns = u32(Q(0u));
  let epochs = u32(Q(1u));
  let rate = Q(2u);
  let rateW = Q(3u);
  let theta = Q(4u);
  let spread = Q(5u);
  let rot = 0u;  let x = 1u;  let y = 3u;  let t0 = 4u;  let t1 = 5u;  let bk = 6u;  let by = 7u;
  let yb = 8u;  let rk = 9u;  let nr = 10u;  let animal = 11u;  let food = 19u;  let bs = 27u;
  let u = 28u;  let v = 29u;  let rs = 30u;  let diff = 31u;
  // The hidden plane: u ∧ v for orthonormal u, v of V_E.
  let z = 2.0 * U(0u) - 1.0;
  let phi = 6.283185307 * U(1u);
  let r0 = sqrt(max(0.0, 1.0 - z * z));
  let uu = vec3<f32>(r0 * cos(phi), r0 * sin(phi), z);
  var helper = vec3<f32>(1.0, 0.0, 0.0);
  if (abs(uu.x) > 0.8) { helper = vec3<f32>(0.0, 1.0, 0.0); }
  let w0 = normalize(vec3<f32>(uu.y * helper.z - uu.z * helper.y, uu.z * helper.x - uu.x * helper.z, uu.x * helper.y - uu.y * helper.x));
  let w1 = vec3<f32>(uu.y * w0.z - uu.z * w0.y, uu.z * w0.x - uu.x * w0.z, uu.x * w0.y - uu.y * w0.x);
  let psi = 6.283185307 * U(2u);
  let vv = normalize(cos(psi) * w0 + sin(psi) * w1);
  mvVector(u, v5(0.0, uu.x, uu.y, uu.z, 0.0));
  mvVector(v, v5(0.0, vv.x, vv.y, vv.z, 0.0));
  mvMul(bs, u, v, 3u);  // the unit bivector of the plane (squares to −1)
  // The nouns: animal senses on the sphere, food senses = R_i x R̃_i, R_i = cos(φ_i/2) − sin(φ_i/2) B.
  for (var k = 0u; k < nouns; k = k + 1u) {
    let zk = 2.0 * U(10u + 3u * k) - 1.0;
    let pk = 6.283185307 * U(11u + 3u * k);
    let rk0 = sqrt(max(0.0, 1.0 - zk * zk));
    mvVector(animal + k, v5(0.0, rk0 * cos(pk), rk0 * sin(pk), zk, 0.0));
    let angle = theta + spread * (2.0 * U(12u + 3u * k) - 1.0);
    mvCopy(rs, bs);
    mvScale(rs, -sin(0.5 * angle));
    mvSet(rs, 0u, cos(0.5 * angle));
    mvSandwich(food + k, rs, animal + k, t0, t1);
  }
  // The learners: a rotor from 1, a projection from the identity.
  mvZero(rot);
  mvSet(rot, 0u, 1.0);
  var W : array<f32, 25>;
  for (var i = 0u; i < 25u; i = i + 1u) { W[i] = select(0.0, 1.0, i % 6u == 0u); }
  let slot = array<u32, 5>(1u, 2u, 4u, 8u, 16u);  // the blades of N, A, B, C, e
  for (var ep = 0u; ep < epochs; ep = ep + 1u) {
    // The rotor: the gradient of Σ|R x R̃ − y|² along each coordinate plane, dy/dθ = −½(B y − y B).
    var grad : array<f32, 10>;
    var gW : array<f32, 25>;
    for (var i = 0u; i < 10u; i = i + 1u) { grad[i] = 0.0; }  // explicitly: declared in a loop
    for (var i = 0u; i < 25u; i = i + 1u) { gW[i] = 0.0; }
    for (var k = 0u; k < train; k = k + 1u) {
      mvSandwich(y, rot, animal + k, t0, t1);
      mvCopy(diff, y);
      mvAdd(diff, food + k, -1.0);
      var plane = 0u;
      for (var i = 0u; i < 5u; i = i + 1u) {
        for (var j = i + 1u; j < 5u; j = j + 1u) {
          mvZero(bk);
          mvSet(bk, (1u << i) | (1u << j), 1.0);
          mvMul(by, bk, y, 0u);
          mvMul(yb, y, bk, 0u);
          mvAdd(by, yb, -1.0);
          grad[plane] = grad[plane] - mvDot(diff, by);  // 2 · diff · (−½)(B y − y B)
          plane = plane + 1u;
        }
      }
      // The projection: y = W x, gradient 2 (W x − y) xᵀ.
      for (var a = 0u; a < 5u; a = a + 1u) {
        var wx = 0.0;
        for (var b = 0u; b < 5u; b = b + 1u) { wx = wx + W[a * 5u + b] * mvGet(animal + k, slot[b]); }
        let d = wx - mvGet(food + k, slot[a]);
        for (var b = 0u; b < 5u; b = b + 1u) { gW[a * 5u + b] = gW[a * 5u + b] + 2.0 * d * mvGet(animal + k, slot[b]); }
      }
    }
    var plane = 0u;
    for (var i = 0u; i < 5u; i = i + 1u) {
      for (var j = i + 1u; j < 5u; j = j + 1u) {
        mvPlaneRotor(rk, i, j, -rate * grad[plane] / f32(train));
        mvMul(nr, rk, rot, 0u);
        mvCopy(rot, nr);
        plane = plane + 1u;
      }
    }
    for (var i = 0u; i < 25u; i = i + 1u) { W[i] = W[i] - rateW * gW[i] / f32(train); }
    mvNormaliseRotor(rot, t0);
  }
  // Evaluation on the nouns read (train) and the rest of the family (test).
  var errR = array<f32, 2>(0.0, 0.0);
  var errW = array<f32, 2>(0.0, 0.0);
  var normR = 0.0;
  var normW = 0.0;
  var invR = 0.0;
  var invW = 0.0;
  // The projection's inverse, by Gauss–Jordan elimination.
  var M : array<f32, 50>;
  for (var a = 0u; a < 5u; a = a + 1u) {
    for (var b = 0u; b < 5u; b = b + 1u) {
      M[a * 10u + b] = W[a * 5u + b];
      M[a * 10u + 5u + b] = select(0.0, 1.0, a == b);
    }
  }
  var singular = false;
  for (var c = 0u; c < 5u; c = c + 1u) {
    var piv = c;
    for (var a = c + 1u; a < 5u; a = a + 1u) { if (abs(M[a * 10u + c]) > abs(M[piv * 10u + c])) { piv = a; } }
    if (abs(M[piv * 10u + c]) < 1e-6) { singular = true; continue; }
    for (var b = 0u; b < 10u; b = b + 1u) {
      let tmp = M[c * 10u + b];
      M[c * 10u + b] = M[piv * 10u + b];
      M[piv * 10u + b] = tmp;
    }
    let d = M[c * 10u + c];
    for (var b = 0u; b < 10u; b = b + 1u) { M[c * 10u + b] = M[c * 10u + b] / d; }
    for (var a = 0u; a < 5u; a = a + 1u) {
      if (a == c) { continue; }
      let f = M[a * 10u + c];
      for (var b = 0u; b < 10u; b = b + 1u) { M[a * 10u + b] = M[a * 10u + b] - f * M[c * 10u + b]; }
    }
  }
  let tests = nouns - train;
  for (var k = 0u; k < nouns; k = k + 1u) {
    let part = select(1u, 0u, k < train);
    mvSandwich(y, rot, animal + k, t0, t1);
    errR[part] = errR[part] + mvDistance(y, food + k);
    var e2 = 0.0;
    var n2 = 0.0;
    for (var a = 0u; a < 5u; a = a + 1u) {
      var wx = 0.0;
      for (var b = 0u; b < 5u; b = b + 1u) { wx = wx + W[a * 5u + b] * mvGet(animal + k, slot[b]); }
      e2 = e2 + (wx - mvGet(food + k, slot[a])) * (wx - mvGet(food + k, slot[a]));
      n2 = n2 + wx * wx;
    }
    errW[part] = errW[part] + sqrt(e2);
    if (part == 1u) {
      normR = normR + abs(mvSize(y) - 1.0);
      normW = normW + abs(sqrt(n2) - 1.0);
      // Back from food to animal: R̃ y R, and the projection's inverse.
      mvReverse(nr, rot);
      mvSandwich(x, nr, food + k, t0, t1);
      invR = invR + mvDistance(x, animal + k);
      var ie = 0.0;
      for (var a = 0u; a < 5u; a = a + 1u) {
        var back = 0.0;
        for (var b = 0u; b < 5u; b = b + 1u) { back = back + M[a * 10u + 5u + b] * mvGet(food + k, slot[b]); }
        ie = ie + (back - mvGet(animal + k, slot[a])) * (back - mvGet(animal + k, slot[a]));
      }
      invW = invW + select(sqrt(ie), 2.0, singular);
    }
  }
  stat(0u, errR[1] / f32(tests));
  stat(1u, errR[0] / f32(train));
  stat(2u, errW[1] / f32(tests));
  stat(3u, errW[0] / f32(train));
  stat(4u, normR / f32(tests));
  stat(5u, normW / f32(tests));
  stat(6u, invR / f32(tests));
  stat(7u, invW / f32(tests));
  // How much of the learned rotor is hyperbolic (planes with e) and how far it turned.
  var hyp = 0.0;
  var ell = 0.0;
  for (var i = 0u; i < 32u; i = i + 1u) {
    if (gradeOf(i) != 2u) { continue; }
    let c = mvGet(rot, i) * mvGet(rot, i);
    if ((i & 16u) != 0u) { hyp = hyp + c; } else { ell = ell + c; }
  }
  stat(8u, hyp / max(hyp + ell, 1e-12));
  // The angle of the learned rotor against the family's: |2 acos ⟨R⟩₀| − θ.
  stat(9u, 2.0 * acos(clamp(abs(mvGet(rot, 0u)), 0.0, 1.0)));
  // Alignment of the learned plane with the hidden one: |⟨R⟩₂ · B| / |⟨R⟩₂|.
  var dotp = 0.0;
  for (var i = 0u; i < 32u; i = i + 1u) { if (gradeOf(i) == 2u) { dotp = dotp + mvGet(rot, i) * mvGet(bs, i); } }
  stat(10u, abs(dotp) / max(sqrt(hyp + ell), 1e-9));
)wgsl";

namespace syn::models {

const kernel::Model* fgaRotorsModel() {
  static const kernel::Model model = [] {
    kernel::Model m;
    m.name = "fgaRotors";
    m.title = "Learning coercions as rotors of Cl(4, 1) against learned projections: regular polysemy from a few words";
    m.maxStats = 11;
    m.agentSource = rotors_agent;
    m.helpersWgsl = space::algebraWgsl(32);
    m.compile = [](const Value& spec) {
      const Value& c = spec.sub("fgaRotors");
      const int nouns = 8;
      const std::vector<double> q = {double(nouns), double(std::clamp(c.integer("epochs", 40), 1, 400)), c.num("rate", 0.3), c.num("projectionRate", 0.3),
                                     c.num("angle", 1.2), c.num("spread", 0.15)};
      std::vector<std::vector<double>> rows;
      for (double n : {1.0, 2.0, 3.0, 4.0, 6.0}) rows.push_back({n});
      return std::vector<Plan>{makePlan("fgaRotors", static_cast<std::uint32_t>(c.integer("learners", 64)), 11, space::withSignTable(q), rows, seedOf(spec))};
    };
    m.summarise = [](const Value&, const std::vector<Plan>& plans, const std::vector<Output>& out) {
      const Plan& p = plans[0];
      const Output& o = out[0];
      Value rows = Value::array();
      for (std::uint32_t j = 0; j < p.jobs; ++j)
        rows.push(Value::object()
                      .set("train", static_cast<double>(p.jobRows[j * p.jobStride]))
                      .set("rotorTest", o.mean(p, j, 0))
                      .set("rotorTrain", o.mean(p, j, 1))
                      .set("projectionTest", o.mean(p, j, 2))
                      .set("projectionTrain", o.mean(p, j, 3))
                      .set("rotorNorm", o.mean(p, j, 4))
                      .set("projectionNorm", o.mean(p, j, 5))
                      .set("rotorInverse", o.mean(p, j, 6))
                      .set("projectionInverse", o.mean(p, j, 7))
                      .set("hyperbolic", o.mean(p, j, 8))
                      .set("angle", o.mean(p, j, 9))
                      .set("plane", o.mean(p, j, 10)));
      return Value::object().set("runs", rows).set("nouns", 8.0).set("angle", static_cast<double>(p.params[space::kSignTable + 4]))
          .set("spread", static_cast<double>(p.params[space::kSignTable + 5]));
    };
    return m;
  }();
  return &model;
}

}  // namespace syn::models
