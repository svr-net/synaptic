// The algebra of the space, in WGSL: multivectors of Cl(4, 1) (see syn/space/algebra.hpp). The algebra itself is
// svr-net/compute's (compute_algebra.hpp); here the space reads its signs from the table at the start of a model's
// constants and gives the model its own constants as Q(i).
#include "syn/space/algebra.hpp"

#include "../core/compute_algebra.hpp"

namespace syn::space {

std::vector<double> signTable() { return compute::algebra::signTable(); }

std::vector<double> withSignTable(const std::vector<double>& constants) {
  std::vector<double> p = signTable();
  p.insert(p.end(), constants.begin(), constants.end());
  return p;
}

std::string algebraWgsl(unsigned slots) {
  return R"wgsl(
fn Q(i : u32) -> f32 { return P(1024u + i); }
fn bladeSign(a : u32, b : u32) -> f32 { return P(a * 32u + b); }
)wgsl" + compute::algebra::algebraWgsl(slots) + R"wgsl(
// A vector of the space as a multivector.
fn mvVector(d : u32, p : V5) { mvVector5(d, p.x.x, p.x.y, p.x.z, p.x.w, p.e); }
)wgsl";
}

}  // namespace syn::space
