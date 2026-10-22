#!/usr/bin/env bash
# Refresh what synaptic carries from svr-net/compute, verbatim, with the commit it came from (compute is the home of
# the fused kernels and of the lock-free parallel work):
#   src/models/compute_index.inc   the index kernel's WGSL helpers (include/compute/index_cga.hpp, kIndexHelpersWgsl)
#   src/core/compute_parallel.hpp  lanes, Once and Lazy -- no lock, no blocking wait (include/compute/parallel.hpp)
#
#   tools/sync_compute.sh [COMPUTE_CHECKOUT]     (default: ../compute)
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
compute="${1:-$here/../compute}"
header="$compute/include/compute/index_cga.hpp"
commit="$(git -C "$compute" rev-parse HEAD)"
out="$here/src/models/compute_index.inc"
{
  echo "// The growing index's WGSL helpers, carried verbatim from svr-net/compute@$commit"
  echo "// (include/compute/index_cga.hpp, kIndexHelpersWgsl). Refresh with tools/sync_compute.sh; do not edit by hand."
  echo 'R"WGSL('
  awk '/kIndexHelpersWgsl = R"WGSL\(/{on=1; next} on && /^\)WGSL";/{exit} on' "$header"
  echo ')WGSL"'
} > "$out"
echo "wrote $out from compute@$commit"
parallel="$here/src/core/compute_parallel.hpp"
{
  echo "// Carried verbatim from svr-net/compute@$commit (include/compute/parallel.hpp)."
  echo "// Refresh with tools/sync_compute.sh; do not edit by hand."
  cat "$compute/include/compute/parallel.hpp"
} > "$parallel"
echo "wrote $parallel from compute@$commit"
