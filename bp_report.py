#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

# --- Optional SciPy normality tests (recommended) ---
try:
    from scipy import stats as scipy_stats  # type: ignore
except Exception:  # pragma: no cover
    scipy_stats = None


@dataclass(frozen=True)
class SeriesStats:
    mean: float
    median: float
    std: float
    n: int
    skew: float
    kurt_excess: float
    shapiro_p: Optional[float] = None
    dagostino_p: Optional[float] = None
    jarque_bera_p: Optional[float] = None


def compute_stats(x: np.ndarray) -> SeriesStats:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)

    mean = float(np.mean(x)) if n else float("nan")
    median = float(np.median(x)) if n else float("nan")
    std = float(np.std(x, ddof=1)) if n > 1 else float("nan")

    # Shape hints (population moments for stability at small n)
    if n >= 3 and np.isfinite(std) and std > 0:
        m3 = np.mean((x - mean) ** 3)
        m4 = np.mean((x - mean) ** 4)
        denom = np.std(x, ddof=0)
        skew = float(m3 / (denom**3)) if denom > 0 else float("nan")
        kurt_excess = float(m4 / (denom**4) - 3.0) if denom > 0 else float("nan")
    else:
        skew = float("nan")
        kurt_excess = float("nan")

    shapiro_p = None
    dagostino_p = None
    jb_p = None

    if scipy_stats is not None and n >= 3:
        try:
            shapiro_p = float(scipy_stats.shapiro(x).pvalue)
        except Exception:
            shapiro_p = None

        if n >= 8:
            try:
                dagostino_p = float(scipy_stats.normaltest(x).pvalue)
            except Exception:
                dagostino_p = None

        if n >= 5:
            try:
                jb_p = float(scipy_stats.jarque_bera(x).pvalue)
            except Exception:
                jb_p = None

    return SeriesStats(
        mean=mean,
        median=median,
        std=std,
        n=n,
        skew=skew,
        kurt_excess=kurt_excess,
        shapiro_p=shapiro_p,
        dagostino_p=dagostino_p,
        jarque_bera_p=jb_p,
    )


def normality_summary(s: SeriesStats) -> str:
    parts = []
    if s.shapiro_p is not None:
        parts.append(f"Shapiro p={s.shapiro_p:.3g} ({'~normal' if s.shapiro_p >= 0.05 else 'non-normal'})")
    if s.dagostino_p is not None:
        parts.append(f"D’Agostino p={s.dagostino_p:.3g} ({'~normal' if s.dagostino_p >= 0.05 else 'non-normal'})")
    if s.jarque_bera_p is not None:
        parts.append(f"JB p={s.jarque_bera_p:.3g} ({'~normal' if s.jarque_bera_p >= 0.05 else 'non-normal'})")

    shape_bits = []
    if np.isfinite(s.skew):
        shape_bits.append(f"skew={s.skew:.2f}")
    if np.isfinite(s.kurt_excess):
        shape_bits.append(f"excess kurt={s.kurt_excess:.2f}")
    if shape_bits:
        parts.append(", ".join(shape_bits))

    if not parts:
        return "Normality: SciPy not available; reporting skew/kurtosis only."
    return "Normality: " + " | ".join(parts)


def read_bp_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, skipinitialspace=True)

    # Normalize column names coming from different BP apps/export formats.
    # Example export headers:
    #   Systolic (mmHg), Diastolic (mmHg), Pulse (bpm)
    def norm_col(c: str) -> str:
        c = str(c).strip().lower()
        c = c.replace("\ufeff", "")  # BOM if present
        c = " ".join(c.split())
        return c

    colmap = {}
    for c in list(df.columns):
        nc = norm_col(c)
        if nc == "date":
            colmap[c] = "Date"
        elif nc == "time":
            colmap[c] = "Time"
        elif nc in {"systolic", "systolic(mmhg)", "systolic (mmhg)", "sys", "sbp"}:
            colmap[c] = "Systolic"
        elif nc in {"diastolic", "diastolic(mmhg)", "diastolic (mmhg)", "dia", "dbp"}:
            colmap[c] = "Diastolic"
        elif nc in {"pulse", "pulse(bpm)", "pulse (bpm)", "hr", "heart rate", "heart rate (bpm)"}:
            colmap[c] = "Pulse"
        elif nc in {"notes", "note", "comment", "comments"}:
            colmap[c] = "Notes"

    if colmap:
        df = df.rename(columns=colmap)

    # Minimal required columns to proceed
    required = ["Date", "Time", "Systolic", "Diastolic"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV missing required columns: {missing}. Found: {list(df.columns)}"
        )

    # Optional columns
    if "Pulse" not in df.columns:
        df["Pulse"] = np.nan
    if "Notes" not in df.columns:
        df["Notes"] = ""

    dt = pd.to_datetime(
        df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip(),
        errors="coerce",
    )
    df = df.copy()
    df["DateTime"] = dt

    df["Systolic"] = pd.to_numeric(df["Systolic"], errors="coerce")
    df["Diastolic"] = pd.to_numeric(df["Diastolic"], errors="coerce")
    df["Pulse"] = pd.to_numeric(df["Pulse"], errors="coerce")

    df = df.dropna(subset=["DateTime", "Systolic", "Diastolic"]).sort_values("DateTime")
    df["IsGapOutlier"] = False
    return df


def mark_gap_outliers(df: pd.DataFrame, gap_days: int) -> pd.DataFrame:
    """
    Mark the first measurement after a gap (in calendar time) as an outlier.

    Rule:
      If (current DateTime - previous DateTime) > gap_days, mark current row as outlier.

    Notes:
      - Uses elapsed time between successive measurements, not missing-calendar-days counting.
      - Applies to the measurement as a whole (both systolic and diastolic).
    """
    if gap_days <= 0:
        # No gap concept; keep all False
        out = df.copy()
        out["IsGapOutlier"] = False
        return out

    out = df.sort_values("DateTime").copy()
    delta = out["DateTime"].diff()
    # strictly greater than gap_days
    out["IsGapOutlier"] = delta > pd.Timedelta(days=gap_days)
    out["IsGapOutlier"] = out["IsGapOutlier"].fillna(False)
    return out


def filter_sitting_readings(df: pd.DataFrame, window_minutes: int = 15) -> tuple[pd.DataFrame, int]:
    """
    Collapse multiple readings taken within a single "sitting".

    Rule:
      - Sort by DateTime ascending.
      - Consecutive readings with a time gap <= window_minutes are considered part of the same sitting.
      - Keep ONLY the most recent reading in each sitting (i.e., the last row in that cluster).

    Returns:
      (filtered_df, num_dropped)
    """
    if df.empty:
        return df.copy(), 0
    if window_minutes <= 0:
        return df.copy(), 0

    d = df.sort_values("DateTime").copy()
    dt = pd.to_datetime(d["DateTime"])
    diffs_min = dt.diff().dt.total_seconds().div(60.0)

    # Start a new group when the gap is greater than the sitting window.
    group_id = (diffs_min.isna() | (diffs_min > window_minutes)).cumsum()

    filtered = d.groupby(group_id, as_index=False).tail(1).copy()
    dropped = int(len(d) - len(filtered))
    return filtered, dropped


def prompt_series_start_date(df: pd.DataFrame) -> Tuple[pd.Timestamp, bool, str]:
    """
    Prompt user for the first date of the analysis series in ISO format (YYYY-MM-DD).

    - Default is the date (not time) of the earliest DateTime in the CSV.
    - Returns (start_timestamp, user_provided, default_date_str).
    """
    default_ts = pd.to_datetime(df["DateTime"].min())
    default_date_str = default_ts.date().isoformat()

    raw = input(f"Enter first date of series [YYYY-MM-DD] (default {default_date_str}): ").strip()
    if raw == "":
        return pd.Timestamp(default_date_str), False, default_date_str

    try:
        ts = pd.Timestamp(raw)
    except Exception:
        raise SystemExit("Invalid date format. Use YYYY-MM-DD.")

    return pd.Timestamp(ts.date().isoformat()), True, default_date_str


def add_value_labels(ax, x, y, color: str, y_offset: float):
    for xi, yi in zip(x, y):
        if not np.isfinite(yi):
            continue
        ax.annotate(
            f"{int(round(yi))}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, y_offset),
            ha="center",
            va="bottom" if y_offset >= 0 else "top",
            fontsize=8,
            color=color,
            alpha=0.95,
        )


def _fmt(v: float) -> str:
    return "—" if not np.isfinite(v) else f"{v:.2f}"


def _make_figure_and_axes() -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[4.8, 1.8], hspace=0.15)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    return fig, ax, ax2


def plot_single_report_page(
    df: pd.DataFrame,
    title: str,
    subtitle: str,
    show_normality: bool,
    highlight_gap_outliers: bool,
    start_date_note: str = "",
    comparison_note: str = "",
) -> plt.Figure:
    # Data
    x_all = df["DateTime"].to_numpy()
    sys_all = df["Systolic"].to_numpy(dtype=float)
    dia_all = df["Diastolic"].to_numpy(dtype=float)

    sys_stats = compute_stats(sys_all)
    dia_stats = compute_stats(dia_all)

    fig, ax, ax2 = _make_figure_and_axes()

    # Sigma bands (series-colored)
    if np.isfinite(sys_stats.std):
        ax.axhspan(sys_stats.mean - 2 * sys_stats.std, sys_stats.mean + 2 * sys_stats.std, color="red", alpha=0.08)
        ax.axhspan(sys_stats.mean - 1 * sys_stats.std, sys_stats.mean + 1 * sys_stats.std, color="red", alpha=0.12)
    if np.isfinite(dia_stats.std):
        ax.axhspan(dia_stats.mean - 2 * dia_stats.std, dia_stats.mean + 2 * dia_stats.std, color="green", alpha=0.08)
        ax.axhspan(dia_stats.mean - 1 * dia_stats.std, dia_stats.mean + 1 * dia_stats.std, color="green", alpha=0.12)

    # Mean/median lines (series-colored)
    ax.axhline(sys_stats.mean, color="red", linestyle="-", linewidth=1.2)
    ax.axhline(sys_stats.median, color="red", linestyle="--", linewidth=1.2)
    ax.axhline(dia_stats.mean, color="green", linestyle="-", linewidth=1.2)
    ax.axhline(dia_stats.median, color="green", linestyle="--", linewidth=1.2)

    # Plot non-outliers as circles, outliers (optional) as x markers
    if highlight_gap_outliers and "IsGapOutlier" in df.columns:
        m_out = df["IsGapOutlier"].to_numpy(dtype=bool)
    else:
        m_out = np.zeros(len(df), dtype=bool)

    m_in = ~m_out

    # Lines (use all points for continuity)
    ax.plot(x_all, sys_all, color="red", linewidth=1.0, alpha=0.6)
    ax.plot(x_all, dia_all, color="green", linewidth=1.0, alpha=0.6)

    # Points
    ax.plot(x_all[m_in], sys_all[m_in], color="red", marker="o", linewidth=0, markersize=5, label="Systolic")
    ax.plot(x_all[m_in], dia_all[m_in], color="green", marker="o", linewidth=0, markersize=5, label="Diastolic")

    if highlight_gap_outliers and np.any(m_out):
        ax.plot(x_all[m_out], sys_all[m_out], color="red", marker="x", linewidth=0, markersize=8, label="Systolic (gap outlier)")
        ax.plot(x_all[m_out], dia_all[m_out], color="green", marker="x", linewidth=0, markersize=8, label="Diastolic (gap outlier)")

    # Value labels
    add_value_labels(ax, x_all, sys_all, color="red", y_offset=8)
    add_value_labels(ax, x_all, dia_all, color="green", y_offset=-10)

    # Axis formatting
    ax.set_title(f"{title}\n{subtitle}" if subtitle else title)
    ax.set_xlabel("Date")
    ax.set_ylabel("mmHg")

    locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.autofmt_xdate(rotation=0)

    ax.grid(True, which="major", axis="both", alpha=0.25)
    ax.legend(loc="upper left")

    # Stats block
    gap_outlier_count = int(df["IsGapOutlier"].sum()) if "IsGapOutlier" in df.columns else 0

    lines = []
    lines.append("Summary statistics")
    lines.append("")

    if start_date_note:
        lines.append(start_date_note)
    if comparison_note:
        lines.append(comparison_note)
    if sitting_note:
        lines.append(sitting_note)
        lines.append("")
    if highlight_gap_outliers:
        lines.append(f"Gap-outlier rule: first reading after a gap is flagged. Gap outliers on this page: {gap_outlier_count}")
        lines.append("")

    lines.append(
        f"Systolic (n={sys_stats.n}):   mean {_fmt(sys_stats.mean)}   median {_fmt(sys_stats.median)}   std {_fmt(sys_stats.std)}"
    )
    lines.append(
        f"Diastolic (n={dia_stats.n}):  mean {_fmt(dia_stats.mean)}   median {_fmt(dia_stats.median)}   std {_fmt(dia_stats.std)}"
    )

    if show_normality:
        lines.append("")
        lines.append(normality_summary(sys_stats))
        lines.append(normality_summary(dia_stats))
        if scipy_stats is None:
            lines.append("")
            lines.append("Note: SciPy not installed; formal normality tests unavailable.")

    ax2.text(0.01, 0.95, "\n".join(lines), va="top", ha="left", fontsize=11)

    return fig


def write_report_pdf(
    segments: list[tuple[str, pd.DataFrame]],
    out_pdf: Path,
    title: str,
    show_normality: bool,
    gap_outliers: bool,
    gap_days: int,
    compare: bool,
    start_date_note: str = "",
    comparison_note: str = "",
    sitting_note: str = "",
):
    """
    Write a PDF report. `segments` is a list of (segment_label, df) pairs.
    If multiple segments are provided, each segment becomes one (or two) pages,
    depending on `compare` + `gap_outliers`.
    """
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_pdf) as pdf:
        for segment_label, df in segments:
            if df.empty:
                # Skip empty segments (e.g., no readings before/after a split date)
                continue

            if gap_outliers:
                df_marked = mark_gap_outliers(df, gap_days=gap_days)
            else:
                df_marked = df.copy()
                df_marked["IsGapOutlier"] = False

            # Build a per-page subtitle prefix
            prefix = segment_label.strip()
            if prefix:
                prefix = prefix + " — "

            if compare and gap_outliers:
                # Page A: include outliers (flagged)
                fig1 = plot_single_report_page(
                    df=df_marked,
                    title=title,
                    subtitle=prefix + f"Included gap-outliers (flagged as 'x'); gap_days={gap_days}",
                    show_normality=show_normality,
                    highlight_gap_outliers=True,
                    start_date_note=start_date_note,
                    comparison_note=comparison_note,
                )
                pdf.savefig(fig1, bbox_inches="tight")
                plt.close(fig1)

                # Page B: exclude outliers
                df_filtered = df_marked.loc[~df_marked["IsGapOutlier"]].copy()
                fig2 = plot_single_report_page(
                    df=df_filtered,
                    title=title,
                    subtitle=prefix + f"Excluded gap-outliers; gap_days={gap_days}",
                    show_normality=show_normality,
                    highlight_gap_outliers=False,
                    start_date_note=start_date_note,
                    comparison_note=comparison_note,
                )
                pdf.savefig(fig2, bbox_inches="tight")
                plt.close(fig2)
            else:
                subtitle = prefix
                highlight = False
                if gap_outliers:
                    subtitle = prefix + f"Gap-outliers flagged as 'x'; gap_days={gap_days}"
                    highlight = True

                fig = plot_single_report_page(
                    df=df_marked,
                    title=title,
                    subtitle=subtitle.strip(" —"),
                    show_normality=show_normality,
                    highlight_gap_outliers=highlight,
                    start_date_note=start_date_note,
                    comparison_note=comparison_note,
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a BP time series PDF from a CSV export.")
    ap.add_argument("csv", type=Path, help="Input CSV with columns: Date, Time, Systolic, Diastolic, Pulse, Notes")
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("bp_report.pdf"),
        help="Output PDF path (default: bp_report.pdf in current directory)",
    )
    ap.add_argument("--title", type=str, default="Blood Pressure Time Series", help="Report title")

    # Date-window controls (for automation / non-interactive usage)
    ap.add_argument("--start-date", type=str, default=None, help="Start date (ISO YYYY-MM-DD). Overrides interactive prompt.")
    ap.add_argument("--prompt-start-date", action="store_true", help="Prompt for start date interactively (ISO YYYY-MM-DD). Default behavior is non-interactive.")
    ap.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date boundary (ISO YYYY-MM-DD). Keeps rows with DateTime < end_date.",
    )

    # Before/after split
    ap.add_argument(
        "--before-after",
        action="store_true",
        help="Create a before/after report split by --comparison-date (before: < date, after: >= date).",
    )
    ap.add_argument(
        "--comparison-date",
        type=str,
        default="2026-01-06",
        help="Comparison date (ISO YYYY-MM-DD) used for --before-after. Default: 2026-01-06",
    )

    ap.add_argument(
        "--normality",
        action="store_true",
        help="Include normality diagnostics (Shapiro, D’Agostino, Jarque–Bera, skew/kurtosis).",
    )

    ap.add_argument(
        "--gap-outliers",
        action="store_true",
        help="Flag the first BP measurement after a gap as an outlier (based on --gap-days).",
    )
    ap.add_argument(
        "--gap-days",
        type=int,
        default=3,
        help="Gap threshold in days. If time between consecutive measurements is > gap_days, the next reading is flagged.",
    )

    ap.add_argument(
        "--sitting-window-minutes",
        type=int,
        default=15,
        help="Collapse multiple readings within a sitting window (minutes), keeping only the most recent reading (default: 15).",
    )
    ap.add_argument(
        "--no-sitting-filter",
        action="store_true",
        help="Disable sitting-window collapsing (keep all readings).",
    )
    ap.add_argument(
        "--compare",
        action="store_true",
        help="When used with --gap-outliers, write a 2-page PDF comparing included vs excluded gap-outliers.",
    )

    args = ap.parse_args()

    df = read_bp_csv(args.csv)
    if df.empty:
        raise SystemExit("No valid rows after parsing Date/Time and numeric BP values.")


    # ---- Start-date selection ----
    default_ts = pd.to_datetime(df["DateTime"].min())
    default_start = default_ts.date().isoformat()

    start_date_note = ""
    user_provided_start = False

    if args.start_date:
        try:
            start_date = pd.Timestamp(args.start_date).normalize()
        except Exception:
            raise SystemExit("Invalid --start-date. Use YYYY-MM-DD.")
        user_provided_start = True
        start_date_note = f"Series start date (CLI): {start_date.date().isoformat()}"
    elif args.prompt_start_date and sys.stdin.isatty():
        start_date, user_provided_start, default_start = prompt_series_start_date(df)
        if user_provided_start:
            start_date_note = (
                f"Series start date (user): {start_date.date().isoformat()} (default was {default_start})"
            )
    else:
        # Non-interactive default: earliest date in the CSV
        start_date = pd.Timestamp(default_start).normalize()

    df = df[df["DateTime"] >= start_date].copy()
    if df.empty:
        raise SystemExit("No rows remain after applying start-date filter.")

    if user_provided_start:
        start_date_note = f"Series start date: {start_date.date().isoformat()} (default was {default_start})"

    # ---- Optional end-date boundary (exclusive) ----
    if args.end_date:
        try:
            end_date = pd.Timestamp(args.end_date).normalize()
        except Exception:
            raise SystemExit("Invalid --end-date. Use YYYY-MM-DD.")
        df = df[df["DateTime"] < end_date].copy()
        if df.empty:
            raise SystemExit("No rows remain after applying end-date filter.")

    # ---- Segmentation: all vs before/after ----
    segments: list[tuple[str, pd.DataFrame]] = []
    comparison_note = ""

    if args.before_after:
        try:
            comp_date = pd.Timestamp(args.comparison_date).normalize()
        except Exception:
            raise SystemExit("Invalid --comparison-date. Use YYYY-MM-DD.")

        comparison_note = f"Comparison date: {comp_date.date().isoformat()} (before: < date; after: ≥ date)"

        df_before = df[df["DateTime"] < comp_date].copy()
        df_after = df[df["DateTime"] >= comp_date].copy()
        segments = [
            (f"Before {comp_date.date().isoformat()}", df_before),
            (f"After {comp_date.date().isoformat()}", df_after),
        ]
    else:
        segments = [("", df)]

    write_report_pdf(
        segments=segments,
        out_pdf=args.out,
        title=args.title,
        show_normality=args.normality,
        gap_outliers=args.gap_outliers,
        gap_days=args.gap_days,
        compare=args.compare,
        start_date_note=start_date_note,
        comparison_note=comparison_note,
        sitting_note=sitting_note,
    )
    out_path = args.out.resolve()
    print(f"✅ Wrote {out_path}")

    # Auto-open PDF on macOS
    import subprocess
    subprocess.run(["open", str(out_path)], check=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
