# Wind Turbine Anomaly Detection
**DSMLC × Enbridge Competition 2026 — CARE to Compare Dataset**

## The Problem

Wind turbines operate in harsh environments where undetected faults can cause costly downtime and catastrophic failures. **Early anomaly detection** gives maintenance teams time to act before a failure occurs — reducing repair costs and maximising turbine uptime.

## Dataset

Data comes from the **CARE to Compare** SCADA dataset, hosted on Zenodo:

> https://zenodo.org/records/15846963

Download and extract `CARE_To_Compare.zip` into the project before running anything. The dataset contains three wind farms (A, B, C) across 36 turbines, 95 labelled events (44 anomaly / 51 normal), and up to 957 sensors per turbine at 10-minute resolution — roughly 89 years of total data.


## Our Approach

We built a multi-model ensemble combining three complementary detectors:

| Model | Role | CARE Axis |
|---|---|---|
| **Isolation Forest** | Flags global outliers across the feature space | Reliability |
| **Autoencoder (bottleneck MLP)** | Detects subtle co-sensor deviations from learned normal | Coverage |
| **CUSUM** | Accumulates evidence of sustained upward drift over time | Earliness |

Models are trained exclusively on the `train` split. An ensemble score (CUSUM 40% · AE 35% · IF 25%) is computed and threshold crossings flag **Tier 1** (CUSUM early-warning) and **Tier 2** (ensemble confirmed) alerts.


## Results

| Farm | Coverage | Accuracy (AUC) | Reliability | Median Lead |
|---|---|---|---|---|
| Wind Farm A | 100% | 0.674 | 95% | 149h |
| Wind Farm B | 100% | 0.440 | 95% | 152h |
| Wind Farm C | 100% | 0.774 | 95% | 107h |

100% of anomaly events were detected across all farms at a 5% false alarm rate. The earliest single detection was **Farm C Event 67 — flagged 376 hours (over 15 days) before failure.**


## Running the Pipeline

Perform the steps below **in order**. Each script writes its outputs to `outputs/` for the next step to consume.

### Step 1 — Extract Data
Unzip the dataset into data/raw (see structure below) so the following exists:
```
CARE_To_Compare/
├── Wind Farm A/
├── Wind Farm B/
└── Wind Farm C/
```

### Step 2 — EDA
```bash
python src/thermal-eda.py
```
Generates thermal sensor distribution and time-series plots into `outputs/figures/`.

### Step 3 — Feature Engineering
```bash
python src/feature_engineering.py
```
Computes rolling statistics, lag features, and cross-sensor residuals. Outputs engineered feature CSVs per event.

### Step 4 — Ensemble (Train + Score)
```bash
python src/ensemble.py
```
Trains Isolation Forest, Autoencoder, and CUSUM per farm on normal training data. Scores all events and writes per-event score CSVs to `outputs/scores/` and feature importance to `outputs/features/`.

### Step 5 — Evaluate
```bash
python src/evaluate.py
```
Computes CARE score components (Coverage, Accuracy, Reliability, Earliness) and writes `outputs/evaluation/care_scores.csv` and `outputs/scores/detection_summary.csv`.

### Step 6 — Visualise Scores
```bash
python src/visualize_scores.py
```
Generates static score plot PNGs per event into `outputs/figures/scores/`.

### Step 7 — Dashboard
```bash
streamlit run src/dashboard.py
```
Launches the interactive dashboard for exploring results.


## Project Structure

```
src/
├── thermal-eda.py          # Sequence 1 — EDA
├── feature_engineering.py  # Sequence 2 — Feature engineering
├── ensemble.py             # Sequence 3 — Model training & scoring
├── evaluate.py             # Sequence 4 — CARE score evaluation
├── visualize_scores.py     # Sequence 5 — Static score plots
└── dashboard.py            # Sequence 6 — Interactive dashboard
outputs/
├── scores/                 # Per-event anomaly score CSVs + detection summary
├── features/               # Feature importance CSVs
├── evaluation/             # CARE score results
└── figures/                # Static plots
data/ (not saved due to file size)
├── raw/                    # `CARE_To_Compare.zip` will go here
├── processed/              # Processed data for the sequence will be put here
```
