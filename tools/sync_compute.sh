#!/usr/bin/env bash
# Refresh src/models/compute_index.inc, the index kernel's WGSL helpers carried from svr-net/compute
# (include/compute/index_cga.hpp, kIndexHelpersWgsl). compute is the home of the fused kernels; the
# helpers are carried verbatim, with the commit they came from.
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
