# covid

Utilities for working with COVID-related datasets and personal health time-series analysis.

This repository contains:

- `covid.py` – original COVID data analysis script  
- `bp_report.py` – blood pressure (BP) time-series reporting and statistical analysis tool  

---

# 1. bp_report.py — Blood Pressure Time-Series Reporting

`bp_report.py` generates a publication-quality PDF report from a CSV export of blood pressure measurements.

---

## 1.1 Core Features

The report includes:

- Time-series plot  
  - **Systolic (red)**  
  - **Diastolic (green)**  
- Datapoint value labels  
- Mean and median lines  
- ±1σ and ±2σ shaded bands  
- Summary statistics  
- Optional normality diagnostics  
- Optional gap-outlier handling  
- Optional before/after comparison  
- Automatic opening of generated PDF (macOS)  

---

## 1.2 Input CSV Format

Supported headers:

### Preferred

```
Date,Time,Systolic,Diastolic,Pulse,Notes
```

### App Export (auto-normalized)

```
Date,Time,Systolic (mmHg),Diastolic (mmHg),Pulse (bpm),Notes
```

Required columns:

- Date  
- Time  
- Systolic  
- Diastolic  

Pulse and Notes are optional.

---

## 1.3 Sitting Filter (NEW)

Per physician recommendation, multiple readings taken in the same sitting should discard the first reading.

### Default Behavior

The script:

- Treats readings within **15 minutes** as the same sitting  
- Keeps only the **most recent reading** within that window  
- Discards earlier readings in that window  

If only one reading exists within a 15-minute window, it is kept.

### Control Flags

Adjust window length:

```bash
--sitting-window-minutes 15
```

Disable sitting filter entirely:

```bash
--no-sitting-filter
```

The report annotates when readings are removed.

---

## 1.4 Gap-Outlier Detection

```bash
--gap-outliers
```

Flags the first reading after a gap exceeding:

```bash
--gap-days 2
```

Used with:

```bash
--compare
```

Produces include vs exclude comparison pages.

---

## 1.5 Normality Diagnostics

```bash
--normality
```

Adds:

- Shapiro-Wilk test  
- D’Agostino K²  
- Jarque–Bera  
- Skewness  
- Excess kurtosis  

Requires `scipy`.

---

## 1.6 Before / After Comparison

```bash
--before-after
--comparison-date YYYY-MM-DD
```

Default comparison date:

```
2026-01-06
```

Boundary rule:

- Before: DateTime < comparison-date 00:00  
- After:  DateTime ≥ comparison-date 00:00  

Can be combined with:

- Gap-outliers  
- Compare  
- Normality  

Produces multi-page PDF.

---

## 1.7 Date Filtering

Non-interactive:

```bash
--start-date YYYY-MM-DD
--end-date YYYY-MM-DD
```

Interactive (optional):

```bash
--prompt-start-date
```

---

## 1.8 Example Workflows

### Simple report

```bash
python3 bp_report.py bp.csv -o bp.pdf
```

### Full analysis

```bash
python3 bp_report.py bp.csv -o bp_full.pdf \
  --before-after --comparison-date 2026-01-06 \
  --gap-outliers --gap-days 2 \
  --compare --normality
```

---

## 1.9 Runner Script

`run_bp_before_after.sh`

Default behavior:

- Comparison date: 2026-01-06  
- Gap window: 2 days  
- Sitting window: 15 minutes  
- Produces single multi-page PDF  

Run:

```bash
./run_bp_before_after.sh
```

---

## 1.10 Dependencies

```bash
pip install pandas numpy matplotlib scipy
```

`scipy` required only for `--normality`.

---

## 1.11 Privacy

Local health data (`bp.csv`) and generated PDFs are ignored via `.gitignore`.

---

# 2. covid.py

Original COVID data utility script (unchanged).

---

# Repository Layout

```
covid/
├── covid.py
├── bp_report.py
├── run_bp_before_after.sh
├── README.md
└── bp.csv        (local only, ignored)
```

---

# License

Personal use / internal tooling.
