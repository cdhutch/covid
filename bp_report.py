#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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


def _safe_float(x) -> float:
    return float(x) if x is not None and not (isinstance(x, float) and math.isnan(x)) else float("nan")


def compute_stats(x: np.ndarray) -> SeriesStats:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)

    mean = float(np.mean(x)) if n else float("nan")
    median = float(np.median(x)) if n else float("nan")
    std = float(np.std(x, ddof=1)) if n > 1 else float("nan")

    # sample skewness / excess kurtosis (Fisher) via formulas
    if n >= 3 and np.isfinite(std) and std > 0:
        m3 = np.mean((x - mean) ** 3)
        m4 = np.mean((x - mean) ** 4)
        skew = float(m3 / (np.std(x, ddof=0) ** 3))  # population version for stability
        kurt_excess = float(m4 / (np.std(x, ddof=0) ** 4) - 3.0)
    else:
        skew = float("nan")
        kurt_excess = float("nan")

    shapiro_p = None
    dagostino_p = None
    jb_p = None

    if scipy_stats is not None and n >= 3:
        # Shapiro-Wilk works best for n <= 5000; SciPy handles larger but interpretation varies
        try:
            shapiro_p = float(scipy_stats.shapiro(x).pvalue)
        except Exception:
            shapiro_p = None

        # D’Agostino’s K^2 needs n >= 8
        if n >= 8:
            try:
                dagostino_p = float(scipy_stats.normaltest(x).pvalue)
            except Exception:
                dagostino_p = None

        # Jarque-Bera (skew/kurt based), usually needs n >= ~20 to be informative
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
    """
    Interpret p-values in a lightweight way:
    - p >= 0.05: no strong evidence against normality
    - p < 0.05: evidence against normality
    """
    parts = []
    if s.shapiro_p is not None:
        parts.append(f"Shapiro p={s.shapiro_p:.3g} ({'~normal' if s.shapiro_p >= 0.05 else 'non-normal'})")
    if s.dagostino_p is not None:
        parts.append(f"D’Agostino p={s.dagostino_p:.3g} ({'~normal' if s.dagostino_p >= 0.05 else 'non-normal'})")
    if s.jarque_bera_p is not None:
        parts.append(f"JB p={s.jarque_bera_p:.3g} ({'~normal' if s.jarque_bera_p >= 0.05 else 'non-normal'})")

    # Always include shape hints
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
    df = pd.read_csv(csv_path)

    required = ["Date", "Time", "Systolic", "Diastolic", "Pulse", "Notes"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Found: {list(df.columns)}")

    # Parse combined datetime
    # Accept common formats; pandas will infer.
    dt = pd.to_datetime(df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip(), errors="coerce")
    df = df.copy()
    df["DateTime"] = dt

    # Coerce numeric
    df["Systolic"] = pd.to_numeric(df["Systolic"], errors="coerce")
    df["Diastolic"] = pd.to_numeric(df["Diastolic"], errors="coerce")
    df["Pulse"] = pd.to_numeric(df["Pulse"], errors="coerce")

    df = df.dropna(subset=["DateTime", "Systolic", "Diastolic"]).sort_values("DateTime")
    return df


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


# def plot_report(df: pd.DataFrame, out_pdf: Path, title: str):
def plot_report(df: pd.DataFrame, out_pdf: Path, title: str, show_normality: bool):
    x = df["DateTime"].to_numpy()
    sys = df["Systolic"].to_numpy(dtype=float)
    dia = df["Diastolic"].to_numpy(dtype=float)

    sys_stats = compute_stats(sys)
    dia_stats = compute_stats(dia)

    # Figure with two vertical regions: plot + text/table
    fig = plt.figure(figsize=(11, 8.5))  # landscape-ish
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[4.8, 1.8], hspace=0.15)

    ax = fig.add_subplot(gs[0])

    # --- Bands and reference lines (mean/median) ---
    # Systolic bands
    if np.isfinite(sys_stats.std):
        # ax.axhspan(sys_stats.mean - 2 * sys_stats.std, sys_stats.mean + 2 * sys_stats.std, alpha=0.08)
        # ax.axhspan(sys_stats.mean - 1 * sys_stats.std, sys_stats.mean + 1 * sys_stats.std, alpha=0.12)
        ax.axhspan(
            sys_stats.mean - 2 * sys_stats.std,
            sys_stats.mean + 2 * sys_stats.std,
            color="red",
            alpha=0.08,
        )
        ax.axhspan(
            sys_stats.mean - 1 * sys_stats.std,
            sys_stats.mean + 1 * sys_stats.std,
            color="red",
            alpha=0.12,
        )
    ax.axhline(sys_stats.mean, color="red", linestyle="-", linewidth=1.2)
    ax.axhline(sys_stats.median, color="red", linestyle="--", linewidth=1.2)

    # Diastolic bands
    if np.isfinite(dia_stats.std):
        # ax.axhspan(dia_stats.mean - 2 * dia_stats.std, dia_stats.mean + 2 * dia_stats.std, alpha=0.08)
        # ax.axhspan(dia_stats.mean - 1 * dia_stats.std, dia_stats.mean + 1 * dia_stats.std, alpha=0.12)
        ax.axhspan(
            dia_stats.mean - 2 * dia_stats.std,
            dia_stats.mean + 2 * dia_stats.std,
            color="green",
            alpha=0.08,
        )
        ax.axhspan(
            dia_stats.mean - 1 * dia_stats.std,
            dia_stats.mean + 1 * dia_stats.std,
            color="green",
            alpha=0.12,
        )


    ax.axhline(dia_stats.mean, color="green", linestyle="-", linewidth=1.2)
    ax.axhline(dia_stats.median, color="green", linestyle="--", linewidth=1.2)

    # --- Time series ---
    # ax.plot(x, sys, marker="o", linewidth=1.0, label="Systolic")
    # ax.plot(x, dia, marker="o", linewidth=1.0, label="Diastolic")

    ax.plot(
        x,
        sys,
        color="red",
        marker="o",
        linewidth=1.2,
        markersize=5,
        label="Systolic",
    )

    ax.plot(
        x,
        dia,
        color="green",
        marker="o",
        linewidth=1.2,
        markersize=5,
        label="Diastolic",
    )

    # Value labels (offset to reduce overlap)
    add_value_labels(ax, x, sys, color="red", y_offset=8)
    add_value_labels(ax, x, dia, color="green", y_offset=-10)

    # Axis formatting
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("mmHg")

    # Date locator/formatter: show dates cleanly even with many points
    locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.autofmt_xdate(rotation=0)

    ax.grid(True, which="major", axis="both", alpha=0.25)
    ax.legend(loc="upper left")

    # --- Stats block below ---
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")

    def fmt(v: float) -> str:
        return "—" if not np.isfinite(v) else f"{v:.2f}"

    # Table-like text (simple, robust in PDF)
    # lines = []
    # lines.append("Summary statistics")
    # lines.append("")
    # lines.append(f"Systolic (n={sys_stats.n}):   mean {fmt(sys_stats.mean)}   median {fmt(sys_stats.median)}   std {fmt(sys_stats.std)}")
    # lines.append(f"Diastolic (n={dia_stats.n}):  mean {fmt(dia_stats.mean)}   median {fmt(dia_stats.median)}   std {fmt(dia_stats.std)}")
    # lines.append("")
    # lines.append(normality_summary(sys_stats))
    # lines.append(normality_summary(dia_stats))
    # if scipy_stats is None:
    #     lines.append("")
    #     lines.append("Note: Install SciPy to enable formal normality tests: pip install scipy")

    lines = []
    lines.append("Summary statistics")
    lines.append("")
    lines.append(
        f"Systolic (n={sys_stats.n}):   mean {fmt(sys_stats.mean)}   "
        f"median {fmt(sys_stats.median)}   std {fmt(sys_stats.std)}"
    )
    lines.append(
        f"Diastolic (n={dia_stats.n}):  mean {fmt(dia_stats.mean)}   "
        f"median {fmt(dia_stats.median)}   std {fmt(dia_stats.std)}"
    )

    if show_normality:
        lines.append("")
        lines.append(normality_summary(sys_stats))
        lines.append(normality_summary(dia_stats))

        if scipy_stats is None:
            lines.append("")
            lines.append("Note: SciPy not installed; formal normality tests unavailable.")


    ax2.text(0.01, 0.95, "\n".join(lines), va="top", ha="left", fontsize=11)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    # ap = argparse.ArgumentParser(description="Generate a BP time series PDF from a CSV export.")
    # ap.add_argument("csv", type=Path, help="Input CSV with columns: Date, Time, Systolic, Diastolic, Pulse, Notes")
    # ap.add_argument(
    #     "-o",
    #     "--out",
    #     type=Path,
    #     default=Path("bp_report.pdf"),
    #     help="Output PDF path (default: bp_report.pdf in current directory)",
    # )
    # ap.add_argument("--title", type=str, default="Blood Pressure Time Series", help="Report title")
    # args = ap.parse_args()

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
    ap.add_argument(
        "--normality",
        action="store_true",
        help="Include normality diagnostics (Shapiro, D’Agostino, Jarque–Bera, skew/kurtosis)",
    )
    args = ap.parse_args()

    df = read_bp_csv(args.csv)
    if df.empty:
        raise SystemExit("No valid rows after parsing Date/Time and numeric BP values.")

    # plot_report(df=df, out_pdf=args.out, title=args.title)
    plot_report(
        df=df,
        out_pdf=args.out,
        title=args.title,
        show_normality=args.normality,
    )
    print(f"✅ Wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())