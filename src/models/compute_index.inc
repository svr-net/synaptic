// The growing index's WGSL helpers, carried verbatim from svr-net/compute@f4f07623eb771fac98b1525b1d3d8730a48f982a
// (include/compute/index_cga.hpp, kIndexHelpersWgsl). Refresh with tools/sync_compute.sh; do not edit by hand.
R"WGSL(
// compute/index_cga.hpp: an index of points of S³ in Cl(4,1), grown by appending.
const ixSlabs : u32 = 16u;
const ixCells : u32 = 4096u;
const ixCoincideTol : f32 = 1e-6;
// The inverse stereographic projection of u onto S³ (N the pole): x of the point x + e.
fn ixLift(u : vec3<f32>) -> vec4<f32> {
  let r2 = u.x * u.x + u.y * u.y + u.z * u.z;
  let k = 1.0 / (1.0 + r2);
  return vec4<f32>((1.0 - r2) * k, 2.0 * u.x * k, 2.0 * u.y * k, 2.0 * u.z * k);
}
// X·S for the sphere S = e_a − t e_N + t e of axis a and step t, the image of the plane u_a = t:
// x_a − t x_N − t, the sign of u_a − t.
fn ixSide(x : vec4<f32>, e : f32, a : u32, t : f32) -> f32 {
  return x[a + 1u] - t * x.x - t * e;
}
// The cell of S³ the point lies in: along each axis, the count of spheres it is on the positive side of.
fn ixCell(x : vec4<f32>, e : f32) -> u32 {
  var c = 0u;
  var scale = 1u;
  for (var a = 0u; a < 3u; a = a + 1u) {
    var slab = 0u;
    for (var j = 0u; j < ixSlabs - 1u; j = j + 1u) {
      let t = -2.0 + 0.25 * f32(j + 1u);
      if (ixSide(x, e, a, t) >= 0.0) { slab = slab + 1u; }
    }
    c = c + slab * scale;
    scale = scale * ixSlabs;
  }
  return c;
}
// Whether two points x + e, y + f are one.
fn ixCoincide(x : vec4<f32>, e : f32, y : vec4<f32>, f : f32) -> bool {
  let d = abs(x - y);
  return max(max(d.x, d.y), max(d.z, d.w)) <= ixCoincideTol && abs(e - f) <= ixCoincideTol;
}
)WGSL"
