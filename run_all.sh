#!/usr/bin/env bash
# Reproduce every number, figure, table, appendix, and the interactive supplement, in order.
# Run from the repository root after installing (see README). Takes about 5 minutes on a recent
# laptop; the first step downloads the embedding model.
set -euo pipefail
export PYTHONPATH="src:scripts"
steps=(
  embed_titles                # regenerate the vectors and check them against data/embeddings
  screen                      # flag each title by its neighbors; resampling check
  subject_groups              # Table 2, Figures 2 and 3, grouping stability
  label_groups                # rebuild the labeling evidence and replay the archived labels
  table3_cutoff_sweep         # Table 3
  figure4_cutoff_surface      # Figure 4
  figure1_distance_demo       # Figure 1
  appendix_a_similarity       # Appendix A
  appendix_b_closely_related  # Appendix B
  framing_check               # wording check (robustness)
  group_screen_comparison     # group-level screen, for the supplement's comparison
  supplement_page             # docs/index.html (interactive supplement)
)
for s in "${steps[@]}"; do
  echo "== ${s}"
  python "scripts/${s}.py"
done
