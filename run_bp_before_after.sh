#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_bp_before_after.sh [comparison_date] [csv_path] [output_dir] [gap_days]
#
# Defaults:
#   comparison_date: 2026-01-06
#   csv_path:        bp.csv
#   output_dir:      reports
#   gap_days:        2

COMP_DATE="${1:-2026-01-06}"
CSV_PATH="${2:-bp.csv}"
OUTDIR="${3:-reports}"
GAP_DAYS="${4:-2}"

mkdir -p "$OUTDIR"

OUTPDF="$OUTDIR/bp_before_after_${COMP_DATE}_gap${GAP_DAYS}_compare.pdf"

python3 bp_report.py "$CSV_PATH" \
  -o "$OUTPDF" \
  --title "Blood Pressure — Before/After ${COMP_DATE}" \
  --before-after \
  --comparison-date "$COMP_DATE" \
  --gap-outliers \
  --gap-days "$GAP_DAYS" \
  --compare

echo "Wrote:"
echo "  $OUTPDF"
