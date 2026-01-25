#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_bp_before_after.sh [comparison_date] [csv_path] [output_dir]
#
# Defaults:
#   comparison_date: 2026-01-06
#   csv_path:        bp.csv
#   output_dir:      reports

COMP_DATE="${1:-2026-01-06}"
CSV_PATH="${2:-bp.csv}"
OUTDIR="${3:-reports}"

mkdir -p "$OUTDIR"

# Report 1: all rows strictly BEFORE comparison date (DateTime < COMP_DATE at 00:00)
python3 bp_report.py "$CSV_PATH" \
  -o "$OUTDIR/bp_before_${COMP_DATE}_gap2_compare.pdf" \
  --title "Blood Pressure — Before ${COMP_DATE}" \
  --end-date "$COMP_DATE" \
  --gap-outliers \
  --gap-days 2 \
  --compare

# Report 2: all rows ON/AFTER comparison date (DateTime >= COMP_DATE at 00:00)
python3 bp_report.py "$CSV_PATH" \
  -o "$OUTDIR/bp_after_${COMP_DATE}_gap2_compare.pdf" \
  --title "Blood Pressure — After ${COMP_DATE}" \
  --start-date "$COMP_DATE" \
  --gap-outliers \
  --gap-days 2 \
  --compare

echo "Wrote:"
echo "  $OUTDIR/bp_before_${COMP_DATE}_gap2_compare.pdf"
echo "  $OUTDIR/bp_after_${COMP_DATE}_gap2_compare.pdf"
