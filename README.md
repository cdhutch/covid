# covid

Utilities for working with COVID-related datasets and personal health time-series analysis.

This repository currently contains two primary tools:

- `covid.py` – original COVID data analysis script
- `bp_report.py` – blood pressure (BP) time-series reporting and statistics generator

---

# 1. covid.py

> Original script (unchanged)

`covid.py` contains utilities for processing COVID datasets and generating derived outputs.

Typical usage:

```bash
python3 covid.py <input-file> [options]
```

(Refer to inline comments in `covid.py` for specific options and data formats.)

---

# 2. bp_report.py — Blood Pressure Time-Series Report Generator

`bp_report.py` generates a publication-quality **PDF report** from a CSV export of blood pressure measurements.

The report includes:

- Time-series plot  
  - Systolic (red)  
  - Diastolic (green)  
- Datapoint value labels  
- Mean and median lines  
- ±1σ and ±2σ shaded bands  
- Summary statistics  
- Optional normality diagnostics  
- Optional gap-outlier handling  
- Optional before/after date comparisons  

---

## 2.1 Input CSV Format

The script accepts either of the following header styles:

### Preferred

```
Date,Time,Systolic,Diastolic,Pulse,Notes
```

### App Export (auto-normalized)

```
Date,Time,Systolic (mmHg),Diastolic (mmHg),Pulse (bpm),Notes
```

Notes:
- `Pulse` and `Notes` are optional
- `Date`, `Time`, `Systolic`, and `Diastolic` are required

Example:

```
"Jan 04, 2026",07:01,123,86,58,
```

---

## 2.2 Basic Usage

```bash
python3 bp_report.py bp.csv -o bp_report.pdf
```

Generates a single-page PDF containing:

- Plot
- Mean / median / standard deviation

---

## 2.3 Optional Features (Flags)

### Normality diagnostics

```bash
--normality
```

Adds:
- Shapiro-Wilk p-value
- D’Agostino K² p-value
- Jarque–Bera p-value
- Skewness
- Excess kurtosis

---

### Gap-outlier detection

```bash
--gap-outliers
```

Flags the **first reading after a long gap** as an outlier.

Default gap length:

```bash
--gap-days 2
```

Rule:
If time difference between consecutive readings > gap-days → later point flagged.

---

### Compare including vs excluding gap outliers

```bash
--gap-outliers --compare
```

Produces a **two-page PDF**:

1. Including gap outliers  
2. Excluding gap outliers  

---

### Start-date filtering

```bash
--start-date YYYY-MM-DD
```

Only include rows where:

```
DateTime >= start-date
```

---

### End-date filtering

```bash
--end-date YYYY-MM-DD
```

Only include rows where:

```
DateTime < end-date
```

---

### Interactive start-date prompt

If `--start-date` not supplied:

```
Enter first date of series [YYYY-MM-DD] (default <earliest-date>):
```

If user types a date, it is annotated in the PDF.

---

### Before / After comparison

```bash
--before-after
```

Splits dataset at a comparison date.

Default:

```bash
--comparison-date 2026-01-06
```

Boundary:

```
Before: DateTime < comparison-date 00:00
After : DateTime >= comparison-date 00:00
```

Combine with gap-outlier comparison:

```bash
--before-after --gap-outliers --compare
```

Produces four pages.

---

## 2.4 Common Workflows

### Simple

```bash
python3 bp_report.py bp.csv -o bp.pdf
```

### With normality

```bash
python3 bp_report.py bp.csv -o bp.pdf --normality
```

### Gap compare

```bash
python3 bp_report.py bp.csv -o bp.pdf --gap-outliers --compare
```

### Before/After

```bash
python3 bp_report.py bp.csv -o bp_before_after.pdf --before-after --comparison-date 2026-01-06
```

### Full

```bash
python3 bp_report.py bp.csv -o bp_full.pdf \
  --before-after --comparison-date 2026-01-06 \
  --gap-outliers --gap-days 2 --compare --normality
```

---

## 2.5 Python Dependencies

```bash
pip install pandas numpy matplotlib scipy
```

(`scipy` optional unless using `--normality`)

---

## 2.6 Privacy Note

`bp.csv` contains personal health information.

Consider:

```bash
echo "bp.csv" >> .gitignore
```

---

# 3. Repository Layout

```
covid/
├── covid.py
├── bp_report.py
├── bp.csv        (optional / local)
└── README.md
```

---

# 4. License

Personal use / internal tooling.
