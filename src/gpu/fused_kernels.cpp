// WGSL compute kernels of the fused classroom pipeline (see fused_classroom.hpp).
//
// 1. classroomFusedKernel: one invocation per learner and job. Samples the learner, runs the
//    course session by session (needs, affect, motivation, ZPD in elliptic CGA, encoding and
//    review slots, response to the outcome), evaluates the test score on every reporting day,
//    writes its statistics, and the workgroup sums them into one partial per (job, workgroup).
// 2. reducePartialsKernel: sums the workgroup partials into per-job means.
// 3. scoreQuantilesKernel: one workgroup per (day, job) histograms the learners' scores and reads
//    off the 10%, 50% and 90% quantiles.
//
// Buffer layouts are produced by gpu::compile(); constants match fused_classroom.hpp and
// runFusedReference() mirrors this code on the CPU, draw for draw. Every kernel stays within the
// limits of WebGPU's compatibility mode (4 storage buffers per stage, 128 invocations per
// workgroup, 4 KiB of workgroup memory), so OpenGL ES-backed adapters can run them too.
#include "syn/gpu/fused_classroom.hpp"

namespace syn::gpu {

static_assert(kWorkgroupSize == 64 && kMaxSessions == 16 && kDayStats == 4 && kSessionStats == 10 && kJobStats == 5 &&
                  kParamCount == 40 && kJobStride == 16 && kHistogramBins == 1024 && kQuantileWorkgroupSize == 128,
              "the WGSL sources below hard-code these constants");

namespace {
const char* kCommon = R"wgsl(
struct Header {
  nL : u32, nJobs : u32, nS : u32, nD : u32,
  stride : u32, profiles : u32, numWG : u32, offSession : u32,
  offJob : u32, endDay : u32, testDay : u32, offJobs : u32,
};
)wgsl";
}  // namespace

const std::string& classroomFusedKernel() {
  static const std::string source = std::string(kCommon) + R"wgsl(
const WG : u32 = 64u;
const MAXS : u32 = 16u;
const NPARAM : u32 = 40u;
const DRAWS_PER_SESSION : u32 = 24u;

@group(0) @binding(0) var<uniform> H : Header;
@group(0) @binding(1) var<storage, read> C : array<f32>;              // params[40] | days[nD] | jobs[nJobs][16]
@group(0) @binding(2) var<storage, read> SEEDS : array<u32>;          // [nJobs]
@group(0) @binding(3) var<storage, read_write> SCRATCH : array<f32>;  // [job][learner][stride]
@group(0) @binding(4) var<storage, read_write> PARTIALS : array<f32>; // [job][workgroup][stride]

fn pcg(v : u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

// Counter-based uniform in (0, 1): draw k of learner l under seed s.
fn draw(s : u32, l : u32, k : u32) -> f32 {
  let h = pcg(s ^ pcg((l * 2654435769u) ^ pcg(k)));
  return (f32(h >> 8u) + 0.5) / 16777216.0;
}

// Box–Muller pair from draws k and k + 1.
fn gauss2(s : u32, l : u32, k : u32) -> vec2<f32> {
  let u1 = draw(s, l, k);
  let u2 = draw(s, l, k + 1u);
  let a = 6.2831853 * u2;
  return sqrt(-2.0 * log(u1)) * vec2<f32>(cos(a), sin(a));
}

fn ncdf(x : f32) -> f32 {
  let z = abs(x) * 0.70710678;
  let t = 1.0 / (1.0 + 0.3275911 * z);
  let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  let erf = 1.0 - poly * exp(-z * z);
  return select(0.5 * (1.0 - erf), 0.5 * (1.0 + erf), x >= 0.0);
}

fn sigma(x : f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
fn c01(x : f32) -> f32 { return clamp(x, 0.0, 1.0); }
fn P(i : u32) -> f32 { return C[i]; }

// ---- pure elliptic CGA on the circle of tasks: Cl(2, 1), vectors (n, m, e), e² = −1 ----
fn cgaDot(a : vec3<f32>, b : vec3<f32>) -> f32 { return a.x * b.x + a.y * b.y - a.z * b.z; }
// The task of relative difficulty Δ: the null vector at angle φ = 2·atan(e^{Δ/2}) from the learner.
fn taskPoint(delta : f32) -> vec3<f32> {
  let phi = 2.0 * atan(exp(0.5 * delta));
  return vec3<f32>(cos(phi), sin(phi), 1.0);
}
// The dilation versor D = exp(β/2 · n e) acting by D X D̃: a hyperbolic rotation of the (n, e)
// components; the result is renormalised to a point (weight −X·e = its e-component).
fn dilate(x : vec3<f32>, beta : f32) -> vec3<f32> {
  let ch = cosh(beta);
  let sh = sinh(beta);
  let y = vec3<f32>(x.x * ch - x.z * sh, x.y, x.z * ch - x.x * sh);
  return y / y.z;
}
const LEARNER : vec3<f32> = vec3<f32>(1.0, 0.0, 1.0);
fn success(t : vec3<f32>) -> f32 { return 1.0 + 0.5 * cgaDot(t, LEARNER); }
// The cap of tasks the learner succeeds on with probability ≥ p: L − 2(1 − p)·e.
fn insideCap(t : vec3<f32>, p : f32) -> bool { return cgaDot(t, vec3<f32>(1.0, 0.0, 1.0 - 2.0 * (1.0 - p))) >= 0.0; }

fn retrievability(t : f32, stability : f32, last : f32) -> f32 {
  if (t <= last) { return 1.0; }
  return inverseSqrt(1.0 + (19.0 / 81.0) * (t - last) / stability);
}

var<workgroup> liveFlags : array<u32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid : vec3<u32>, @builtin(workgroup_id) wid : vec3<u32>) {
  let job = wid.y;
  let l = wid.x * WG + lid.x;
  let stride = H.stride;
  let base = (job * H.nL + l) * stride;
  let live = l < H.nL;
  liveFlags[lid.x] = select(0u, 1u, live);
  if (live) {
    let seed = SEEDS[job];
    let J = H.offJobs + job * 16u;
    let reinforcement = C[J + 0u]; let direction = C[J + 1u]; let empathy = C[J + 2u]; let choice = C[J + 3u];
    let relevance = C[J + 4u]; let transparency = C[J + 5u]; let prior = C[J + 6u]; let retrieval = C[J + 7u];
    let spacing = C[J + 8u]; let interleaving = C[J + 9u]; let feedback = C[J + 10u]; let problemFirst = C[J + 11u];
    let scaffolding = C[J + 12u]; let dialogue = C[J + 13u]; let inclusivity = C[J + 14u]; let growthFeedback = C[J + 15u];

    // The learner.
    let z01 = gauss2(seed, l, 0u); let z23 = gauss2(seed, l, 2u); let z45 = gauss2(seed, l, 4u); let z67 = gauss2(seed, l, 6u);
    let age = max(4.0, P(0u) + P(1u) * z01.x);
    let capital = c01(P(12u) + P(13u) * z01.y);
    var theta = P(2u) + P(3u) * z23.x + P(14u) * (capital - P(12u));
    var anxiety = c01(P(4u) + P(5u) * z23.y);
    var growth = c01(P(6u) + P(7u) * z45.x);
    var efficacy = c01(P(8u) + P(9u) * z45.y);
    var intrinsic = c01(P(10u) + P(11u) * z67.x);
    let physiological = c01(0.85 + 0.1 * z67.y);
    let start = theta;
    let adult = c01((age - 12.0) / 10.0);
    let formal = ncdf((age - P(35u)) / P(34u));
    let abstractFit = 1.0 - P(18u) * (1.0 - formal) * (1.0 - 0.5 * scaffolding);
    let access = 1.0 - 0.5 * (1.0 - capital) * (1.0 - inclusivity);
    var habit = 0.0;
    var discrimination = 0.0;
    var persistence = 0.0;
    var engaged = true;
    var learned : array<f32, 16>;
    var stab : array<f32, 16>;
    var last : array<f32, 16>;
    var units = 0u;
    var nextDay = 0u;
    var endScore = 0.0;
    var delayed = 0.0;

    for (var s = 0u; s <= H.nS; s = s + 1u) {
      let t = select(P(15u) * f32(s), 1e30, s == H.nS);
      // Report every day before this session (after the last session: every remaining day).
      loop {
        if (nextDay >= H.nD) { break; }
        let day = C[NPARAM + nextDay];
        if (!(day < t - 1e-4)) { break; }
        var rec = 0.0;
        for (var k = 0u; k < units; k = k + 1u) { rec = rec + learned[k] * retrievability(day, stab[k], last[k]); }
        rec = rec / f32(H.nS);
        let skill = sigma(theta - P(20u));
        let pc = 1.0 / 3.0 + 2.0 / 3.0 * (1.0 - exp(-(0.6 + discrimination)));
        let score = rec * pc * (0.5 + 0.5 * skill);
        let o = base + nextDay * 4u;
        SCRATCH[o] = score;
        SCRATCH[o + 1u] = score * score;
        SCRATCH[o + 2u] = rec;
        SCRATCH[o + 3u] = select(0.0, 1.0, score < P(19u));
        if (nextDay == H.endDay) { endScore = score; }
        if (nextDay == H.testDay) { delayed = score; }
        nextDay = nextDay + 1u;
      }
      if (s == H.nS) { break; }

      let k0 = 16u + s * DRAWS_PER_SESSION;
      var u : array<f32, 15>;
      for (var j = 0u; j < 15u; j = j + 1u) { u[j] = draw(seed, l, k0 + j); }
      let noise = gauss2(seed, l, k0 + 16u).x;

      // 1. Needs (Maslow): climate lifts safety and belonging; readiness = gates through belonging.
      let safety0 = c01(0.45 + 0.25 * capital - 0.2 * anxiety);
      let belonging0 = c01(0.35 + 0.35 * capital);
      let safety = safety0 + c01(0.8 * empathy) * (1.0 - safety0);
      let belonging = belonging0 + c01(0.5 * inclusivity + 0.3 * dialogue + 0.3 * empathy) * (1.0 - belonging0);
      let readiness = sigma((physiological - P(29u)) / P(30u)) * sigma((safety - P(29u)) / P(30u)) * sigma((belonging - P(29u)) / P(30u));

      // 2. Affect.
      anxiety = c01(anxiety + 0.08 * problemFirst * (1.0 - growth) * (1.0 - empathy));
      anxiety = anxiety * (1.0 - 0.25 * empathy);
      let clarity = 1.0 / (1.0 + 0.5 * anxiety);

      // 3. Motivation.
      intrinsic = max(0.0, intrinsic - 0.01 * reinforcement * intrinsic);
      let value = clamp(0.2 + 0.35 * intrinsic + relevance * (0.15 + 0.2 * adult) + transparency * (0.05 + 0.15 * adult) +
                        choice * (0.05 + 0.15 * adult) + reinforcement * (0.05 + 0.3 * (1.0 - adult)) - 0.1 * adult * direction, 0.0, 1.2);
      let motivation = value * efficacy;
      if (engaged) { habit = habit + 0.1 * reinforcement * (1.0 - habit); }
      let effort = select(0.1, c01(0.35 + motivation + 0.15 * habit + persistence), engaged);

      // 4. Challenge and support: the ZPD in elliptic CGA.
      let level = P(16u) + P(17u) * f32(s);
      let b = (1.0 - scaffolding) * level + scaffolding * (theta + 0.75);
      let support = 1.5 * scaffolding + 0.4 * direction + 0.8 * dialogue * sigma(-theta + 0.5);
      let task = taskPoint(b - theta);
      let helped = dilate(task, -0.5 * support);
      let pAlone = success(task);
      let pAssisted = success(helped);
      let inZpd = !insideCap(task, P(31u)) && insideCap(helped, P(32u));

      // 5–7. Learning.
      let struggle = 1.0 + problemFirst * (0.5 * (growth + empathy) - 0.35);
      let drive = effort * clarity * (0.6 + 0.4 * readiness) * access * abstractFit;
      theta = theta + 0.9 * drive * pAssisted * (1.0 - pAlone) * (1.0 + 0.3 * dialogue) * (1.0 + 0.2 * feedback) * struggle *
              max(0.0, 1.0 + 0.15 * noise);

      // 6. Encoding and the review slots.
      learned[s] = c01(2.2 * drive * (0.5 + 0.5 * pAssisted) * (0.75 + 0.25 * direction) * (1.0 + 0.1 * transparency) * (1.0 + 0.15 * feedback));
      stab[s] = P(21u) * (1.0 + 0.6 * prior + P(28u) * problemFirst + 0.4 * relevance + 0.3 * dialogue);
      last[s] = P(15u) * f32(s);
      units = s + 1u;
      for (var k = 0u; k < 3u; k = k + 1u) {
        let uTarget = u[3u + 4u * k]; let uMode = u[4u + 4u * k]; let uRecall = u[5u + 4u * k]; let uFeedback = u[6u + 4u * k];
        var unit = s;
        if (s > 0u && uTarget < spacing) { unit = min(s - 1u, u32(uTarget / spacing * f32(s))); }
        let at = P(15u) * f32(s) + 0.01 * f32(k + 1u);
        let r = retrievability(at, stab[unit], last[unit]);
        var ok = true;
        if (uMode < retrieval) {
          if (uRecall < r) {
            stab[unit] = stab[unit] * (1.0 + P(24u) + P(25u) * (1.0 - r));
            last[unit] = at;
          } else {
            ok = false;
            if (uFeedback < feedback) {
              stab[unit] = max(P(27u), P(26u) * stab[unit]) * (1.0 + P(22u) + P(23u) * (1.0 - r));
              last[unit] = at;
              learned[unit] = c01(learned[unit] + 0.15 * (1.0 - learned[unit]));
            }
          }
        } else {
          stab[unit] = stab[unit] * (1.0 + P(22u) + P(23u) * (1.0 - r));
          last[unit] = at;
        }
        if (effort > 0.3) { learned[unit] = c01(learned[unit] + 0.03 * (1.0 - learned[unit])); }
      }
      discrimination = discrimination + 0.03 + 0.25 * interleaving;

      // 8. Response to the outcome.
      let succeeded = u[0] < pAssisted;
      efficacy = c01(efficacy + (0.1 + 0.1 * feedback) * (select(0.0, 1.0, succeeded) - efficacy));
      if (!succeeded) {
        persistence = clamp(persistence + 0.06 * growth - 0.08 * (1.0 - growth), -0.3, 0.2);
        efficacy = c01(efficacy - 0.04 * (1.0 - growth) * (1.0 - growthFeedback));
        anxiety = c01(anxiety + 0.05 * (1.0 - growth) * (1.0 - empathy));
      } else {
        anxiety = c01(anxiety - 0.03);
      }
      growth = c01(growth + 0.05 * growthFeedback * (1.0 - growth));
      if (engaged && !succeeded && effort < 0.45 && growth < 0.45 && u[1] < 0.25 * (1.0 - empathy)) {
        engaged = false;
      } else if (!engaged && u[2] < 0.3 * empathy + 0.2 * relevance) {
        engaged = true;
      }

      if (H.profiles != 0u) {
        let o = base + H.offSession + s * 10u;
        SCRATCH[o] = theta - start;
        SCRATCH[o + 1u] = effort;
        SCRATCH[o + 2u] = motivation;
        SCRATCH[o + 3u] = value;
        SCRATCH[o + 4u] = efficacy;
        SCRATCH[o + 5u] = anxiety;
        SCRATCH[o + 6u] = readiness;
        SCRATCH[o + 7u] = growth;
        SCRATCH[o + 8u] = select(0.0, 1.0, engaged);
        SCRATCH[o + 9u] = select(0.0, 1.0, inZpd);
      }
    }
    let high = select(0.0, 1.0, capital > P(33u));
    let o = base + H.offJob;
    SCRATCH[o] = endScore;
    SCRATCH[o + 1u] = delayed;
    SCRATCH[o + 2u] = theta - start;
    SCRATCH[o + 3u] = delayed * high;
    SCRATCH[o + 4u] = high;
  }

  // Workgroup reduction: each invocation sums some statistics over the workgroup's learners.
  storageBarrier();
  workgroupBarrier();
  let first = wid.x * WG;
  let dst = (job * H.numWG + wid.x) * stride;
  for (var j = lid.x; j < stride; j = j + WG) {
    var acc = 0.0;
    for (var m = 0u; m < WG; m = m + 1u) {
      if (liveFlags[m] != 0u) { acc = acc + SCRATCH[(job * H.nL + first + m) * stride + j]; }
    }
    PARTIALS[dst + j] = acc;
  }
}
)wgsl";
  return source;
}

const std::string& reducePartialsKernel() {
  static const std::string source = std::string(kCommon) + R"wgsl(
@group(0) @binding(0) var<uniform> H : Header;
@group(0) @binding(1) var<storage, read> PARTIALS : array<f32>;   // [job][workgroup][stride]
@group(0) @binding(2) var<storage, read_write> SUMS : array<f32>; // [job][stride], means

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= H.nJobs * H.stride) { return; }
  let job = i / H.stride;
  let j = i % H.stride;
  var acc = 0.0;
  for (var w = 0u; w < H.numWG; w = w + 1u) { acc = acc + PARTIALS[(job * H.numWG + w) * H.stride + j]; }
  SUMS[i] = acc / f32(H.nL);
}
)wgsl";
  return source;
}

const std::string& scoreQuantilesKernel() {
  static const std::string source = std::string(kCommon) + R"wgsl(
const BINS : u32 = 1024u;
const QWG : u32 = 128u;

@group(0) @binding(0) var<uniform> H : Header;
@group(0) @binding(1) var<storage, read> SCRATCH : array<f32>;     // [job][learner][stride]
@group(0) @binding(2) var<storage, read_write> QUANT : array<f32>; // [job][day][q10, q50, q90]

var<workgroup> hist : array<atomic<u32>, 1024>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid : vec3<u32>, @builtin(workgroup_id) wid : vec3<u32>) {
  let day = wid.x;
  let job = wid.y;
  for (var b = lid.x; b < BINS; b = b + QWG) { atomicStore(&hist[b], 0u); }
  workgroupBarrier();
  for (var l = lid.x; l < H.nL; l = l + QWG) {
    let score = clamp(SCRATCH[(job * H.nL + l) * H.stride + day * 4u], 0.0, 1.0);
    atomicAdd(&hist[min(BINS - 1u, u32(score * f32(BINS)))], 1u);
  }
  workgroupBarrier();
  if (lid.x == 0u) {
    // Type-7 quantile rank floor(q·(n − 1)), located in the cumulative histogram (bin centre).
    var q = array<f32, 3>(0.1, 0.5, 0.9);
    for (var i = 0u; i < 3u; i = i + 1u) {
      let rank = u32(floor(q[i] * f32(H.nL - 1u)));
      var cum = 0u;
      var bin = 0u;
      for (var b = 0u; b < BINS; b = b + 1u) {
        cum = cum + atomicLoad(&hist[b]);
        if (cum > rank) { bin = b; break; }
      }
      QUANT[(job * H.nD + day) * 3u + i] = (f32(bin) + 0.5) / f32(BINS);
    }
  }
}
)wgsl";
  return source;
}

}  // namespace syn::gpu
