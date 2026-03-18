"""
Thermal & Electrical System Anomaly EDA
Step 1: Sensor identification, event filtering, and baseline visualizations.
"""

import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

# - Paths ----------------------------------
ROOT       = Path(__file__).resolve().parent.parent
DATA_RAW   = ROOT / "data" / "raw"
DATA_PROC  = ROOT / "data" / "processed"
FIGS       = ROOT / "outputs" / "figures"
ZIP_PATH   = DATA_RAW / "care-to-compare.zip"
EXTRACTED  = DATA_RAW / "CARE_To_Compare"

DATA_PROC.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# - Keywords --------------------------------─
SENSOR_KW = [
    "temperature", "temp", "current", "voltage", "frequency",
    "phase", "power", "winding", "bearing", "generator",
    "transformer", "gearbox oil",
]
EVENT_KW = [
    "transformer", "generator bearing", "hydraulic",
    "gearbox", "overheating", "winding", "stator",
]

FARMS = ["Wind Farm A", "Wind Farm B", "Wind Farm C"]


# --------------------------------------─
# 1a. Extract & load feature descriptions
# --------------------------------------─

def extract_zip():
    if EXTRACTED.exists():
        print(f"[1a] Already extracted -> {EXTRACTED}")
        return
    print(f"[1a] Extracting {ZIP_PATH} ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_RAW)
    print(f"[1a] Done -> {EXTRACTED}")


def load_feature_descriptions() -> dict[str, pd.DataFrame]:
    """Return {farm_name: filtered_df} for thermal/electrical sensors."""
    result = {}
    for farm in FARMS:
        path = EXTRACTED / farm / "feature_description.csv"
        if not path.exists():
            print(f"  [WARN] missing {path}")
            continue
        df = pd.read_csv(path, sep=";", encoding="latin-1")
        # normalise column names
        df.columns = df.columns.str.strip().str.lower()
        # detect description column
        desc_col = next(
            (c for c in df.columns if "desc" in c), df.columns[1] if len(df.columns) > 1 else df.columns[0]
        )
        mask = df[desc_col].str.lower().str.contains("|".join(SENSOR_KW), na=False)
        result[farm] = df[mask].copy()
        print(f"  [1a] {farm}: {mask.sum()} thermal/electrical sensors found (of {len(df)} total)")
    return result


def print_sensor_table(feat_dfs: dict[str, pd.DataFrame]):
    for farm, df in feat_dfs.items():
        print(f"\n{'='*60}")
        print(f"  {farm} — Thermal / Electrical Sensors")
        print(f"{'='*60}")
        print(df.to_string(index=False))


def save_sensor_list(feat_dfs: dict[str, pd.DataFrame]):
    rows = []
    for farm, df in feat_dfs.items():
        tmp = df.copy()
        tmp.insert(0, "farm", farm)
        rows.append(tmp)
    combined = pd.concat(rows, ignore_index=True)
    out = DATA_PROC / "thermal_sensor_list.csv"
    combined.to_csv(out, index=False)
    print(f"\n[1a] Sensor list saved -> {out}")
    return combined


# --------------------------------------─
# 1b. Load event info & filter relevant anomaly events
# --------------------------------------─

def load_event_info() -> dict[str, pd.DataFrame]:
    result = {}
    for farm in FARMS:
        path = EXTRACTED / farm / "event_info.csv"
        if not path.exists():
            print(f"  [WARN] missing {path}")
            continue
        df = pd.read_csv(path, sep=";")
        df.columns = df.columns.str.strip().str.lower()
        result[farm] = df
    return result


def filter_thermal_events(event_dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    result = {}
    for farm, df in event_dfs.items():
        # find description column
        desc_col = next((c for c in df.columns if "desc" in c), None)
        label_col = next((c for c in df.columns if "label" in c), None)
        if desc_col is None:
            print(f"  [WARN] {farm}: no description column found in {list(df.columns)}")
            result[farm] = df[df[label_col] == "anomaly"].copy() if label_col else df.copy()
            continue

        anomaly_mask = df[label_col].str.lower() == "anomaly" if label_col else pd.Series(True, index=df.index)
        thermal_mask = df[desc_col].str.lower().str.contains("|".join(EVENT_KW), na=False)
        filtered = df[anomaly_mask & thermal_mask].copy()

        if len(filtered) == 0:
            # fallback: all anomaly events
            print(f"  [1b] {farm}: no keyword matches — using all anomaly events as fallback")
            filtered = df[anomaly_mask].copy()

        result[farm] = filtered
        print(f"  [1b] {farm}: {len(filtered)} thermal/electrical anomaly events")
        if desc_col in filtered.columns:
            for _, row in filtered.iterrows():
                print(f"        event_id={row.get('event_id', '?')}  {row[desc_col]}")
    return result


def save_event_shortlist(thermal_events: dict[str, pd.DataFrame]):
    rows = []
    for farm, df in thermal_events.items():
        tmp = df.copy()
        tmp.insert(0, "farm", farm)
        rows.append(tmp)
    combined = pd.concat(rows, ignore_index=True)
    out = DATA_PROC / "thermal_event_shortlist.csv"
    combined.to_csv(out, index=False)
    print(f"[1b] Event shortlist saved -> {out}")
    return combined


# --------------------------------------─
# 1c. Load target datasets
# --------------------------------------─

def get_event_ids(event_df: pd.DataFrame) -> list:
    id_col = next((c for c in event_df.columns if c in ("event_id", "id", "dataset_id")), event_df.columns[0])
    return event_df[id_col].tolist()


def load_datasets_for_farm(farm: str, event_ids: list, n_normal: int = 4) -> tuple[list, list]:
    """Return (anomaly_dfs, normal_dfs)."""
    datasets_dir = EXTRACTED / farm / "datasets"
    if not datasets_dir.exists():
        print(f"  [WARN] {farm}: datasets dir not found at {datasets_dir}")
        return [], []

    all_csvs = sorted(datasets_dir.glob("*.csv"))
    print(f"  [1c] {farm}: {len(all_csvs)} dataset CSVs found")

    anomaly_dfs, normal_dfs = [], []

    for csv in all_csvs:
        # derive event id from filename (e.g. "68.csv" or "event_68.csv")
        stem = csv.stem.replace("event_", "").strip()
        try:
            fid = int(stem)
        except ValueError:
            fid = stem

        try:
            df = pd.read_csv(csv, sep=";", parse_dates=["time_stamp"], encoding="latin-1")
        except Exception as e:
            print(f"    [WARN] could not read {csv.name}: {e}")
            continue

        if fid in event_ids:
            anomaly_dfs.append((fid, df))
        elif len(normal_dfs) < n_normal:
            # check if this is a normal event by checking event_label if available
            normal_dfs.append((fid, df))

    print(f"  [1c] {farm}: loaded {len(anomaly_dfs)} anomaly + {len(normal_dfs)} normal datasets")
    return anomaly_dfs, normal_dfs


# --------------------------------------─
# Helpers
# --------------------------------------─

def get_thermal_sensor_cols(df: pd.DataFrame, feat_df: pd.DataFrame | None = None) -> list[str]:
    """Return sensor columns relevant to thermal/electrical signals."""
    all_sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

    if feat_df is not None and len(feat_df) > 0:
        # use the filtered feature list
        name_col = next((c for c in feat_df.columns if "name" in c or "sensor" in c.lower()), feat_df.columns[0])
        known = set(feat_df[name_col].str.strip().tolist())
        # match _avg columns
        matched = [c for c in all_sensor_cols if any(c.startswith(k) for k in known)]
        if matched:
            # prefer _avg columns
            avg_cols = [c for c in matched if c.endswith("_avg")]
            return avg_cols if avg_cols else matched

    # fallback: use thermally-named sensors from CLAUDE.md knowledge
    thermal_ids = [6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 21, 23, 24, 25, 26]
    preferred = [f"sensor_{i}_avg" for i in thermal_ids if f"sensor_{i}_avg" in df.columns]
    if preferred:
        return preferred
    # last resort: all _avg columns (up to 12)
    avg_cols = [c for c in all_sensor_cols if c.endswith("_avg")]
    return avg_cols[:12]


def get_prediction_window(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where train_test == 'test' (prediction window)."""
    if "train_test" in df.columns:
        mask = df["train_test"].str.lower().str.strip().isin(["test", "prediction", "1", 1])
        return df[mask]
    return df


# --------------------------------------─
# 1d. Visualizations
# --------------------------------------─

def plot_time_series(farm: str, anomaly_dfs: list, normal_dfs: list, feat_df: pd.DataFrame | None = None):
    """Overlay anomaly vs. normal time-series for key thermal sensors."""
    if not anomaly_dfs:
        print(f"  [1d] {farm}: no anomaly datasets to plot")
        return

    # pick first anomaly dataset
    event_id, adf = anomaly_dfs[0]
    pred_window = get_prediction_window(adf)
    sensor_cols = get_thermal_sensor_cols(adf, feat_df)

    if not sensor_cols:
        print(f"  [1d] {farm}: no thermal sensor columns found")
        return

    # use up to 6 sensors
    sensors_to_plot = sensor_cols[:6]
    n = len(sensors_to_plot)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=False)
    if n == 1:
        axes = [axes]

    fig.suptitle(f"{farm} — Thermal Sensor Time Series\nAnomaly event {event_id} (prediction window)", fontsize=13)

    for ax, col in zip(axes, sensors_to_plot):
        # plot anomaly
        if "time_stamp" in pred_window.columns:
            ax.plot(pred_window["time_stamp"], pred_window[col], color="tomato", linewidth=0.8,
                    label=f"Anomaly (event {event_id})", alpha=0.85)
        else:
            ax.plot(pred_window[col].values, color="tomato", linewidth=0.8,
                    label=f"Anomaly (event {event_id})", alpha=0.85)

        # overlay normal datasets (up to 2)
        colors = ["steelblue", "seagreen"]
        for (nid, ndf), color in zip(normal_dfs[:2], colors):
            npw = get_prediction_window(ndf)
            if col not in npw.columns:
                continue
            if "time_stamp" in npw.columns:
                ax.plot(npw["time_stamp"], npw[col], color=color, linewidth=0.5,
                        label=f"Normal (event {nid})", alpha=0.5)
            else:
                ax.plot(npw[col].values, color=color, linewidth=0.5,
                        label=f"Normal (event {nid})", alpha=0.5)

        ax.set_ylabel(col, fontsize=8)
        ax.legend(fontsize=7, loc="upper left")
        ax.tick_params(axis="x", labelsize=7)
        if "time_stamp" in pred_window.columns:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            fig.autofmt_xdate()

    plt.tight_layout()
    out = FIGS / f"{farm.replace(' ', '_')}_thermal_timeseries.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  [1d] Saved -> {out}")


def plot_distributions(farm: str, anomaly_dfs: list, normal_dfs: list, feat_df: pd.DataFrame | None = None):
    """KDE/boxplot per sensor: anomaly vs. normal (prediction window)."""
    if not anomaly_dfs:
        return

    # aggregate prediction-window data
    anom_frames = [get_prediction_window(df) for _, df in anomaly_dfs]
    norm_frames  = [get_prediction_window(df) for _, df in normal_dfs]

    anom_all = pd.concat(anom_frames, ignore_index=True) if anom_frames else pd.DataFrame()
    norm_all  = pd.concat(norm_frames, ignore_index=True) if norm_frames else pd.DataFrame()

    if anom_all.empty:
        print(f"  [1d] {farm}: empty anomaly prediction window data")
        return

    sensor_cols = get_thermal_sensor_cols(anom_all, feat_df)[:8]
    if not sensor_cols:
        return

    n = len(sensor_cols)
    ncols = 2
    nrows = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten() if n > 1 else [axes]

    fig.suptitle(f"{farm} — Thermal Sensor Distributions\n(Anomaly vs. Normal, prediction window)", fontsize=12)

    for ax, col in zip(axes, sensor_cols):
        if col not in anom_all.columns:
            ax.set_visible(False)
            continue
        a_vals = anom_all[col].dropna()
        n_vals = norm_all[col].dropna() if not norm_all.empty and col in norm_all.columns else pd.Series(dtype=float)

        data, labels, palette = [], [], []
        if len(a_vals) > 0:
            data.append(a_vals)
            labels.append("Anomaly")
            palette.append("tomato")
        if len(n_vals) > 0:
            data.append(n_vals)
            labels.append("Normal")
            palette.append("steelblue")

        plot_df = pd.concat(
            [pd.DataFrame({"value": v, "type": lbl}) for v, lbl in zip(data, labels)],
            ignore_index=True,
        )
        try:
            sns.kdeplot(data=plot_df, x="value", hue="type", ax=ax,
                        palette={"Anomaly": "tomato", "Normal": "steelblue"},
                        fill=True, alpha=0.4, linewidth=1.2, common_norm=False)
        except Exception:
            # fallback to histogram if KDE fails
            if len(a_vals) > 0:
                ax.hist(a_vals, bins=40, color="tomato", alpha=0.5, label="Anomaly", density=True)
            if len(n_vals) > 0:
                ax.hist(n_vals, bins=40, color="steelblue", alpha=0.5, label="Normal", density=True)
            ax.legend(fontsize=7)

        ax.set_title(col, fontsize=9)
        ax.set_xlabel("Value", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)

    # hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    out = FIGS / f"{farm.replace(' ', '_')}_thermal_distributions.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  [1d] Saved -> {out}")


def plot_correlation_heatmap(farm: str, anomaly_dfs: list, feat_df: pd.DataFrame | None = None):
    """Correlation heatmap among thermal sensors during anomaly prediction window."""
    if not anomaly_dfs:
        return

    anom_frames = [get_prediction_window(df) for _, df in anomaly_dfs]
    anom_all = pd.concat(anom_frames, ignore_index=True)

    sensor_cols = get_thermal_sensor_cols(anom_all, feat_df)[:12]
    if len(sensor_cols) < 2:
        return

    corr = anom_all[sensor_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, ax=ax, annot=True, fmt=".2f", cmap="RdYlBu_r",
        vmin=-1, vmax=1, linewidths=0.4, annot_kws={"size": 7},
    )
    ax.set_title(f"{farm} — Thermal Sensor Correlation\n(Anomaly prediction window)", fontsize=12)
    plt.tight_layout()
    out = FIGS / f"{farm.replace(' ', '_')}_thermal_correlation.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  [1d] Saved -> {out}")


# --------------------------------------─
# Main
# --------------------------------------─

def main():
    print("\n" + "="*70)
    print("  THERMAL & ELECTRICAL EDA — Wind Turbine Anomaly Detection")
    print("="*70)

    # - 1a ---------------------------------
    print("\n[STEP 1a] Extracting data & loading feature descriptions")
    extract_zip()
    feat_dfs = load_feature_descriptions()
    print_sensor_table(feat_dfs)
    sensor_list = save_sensor_list(feat_dfs)

    # - 1b ---------------------------------
    print("\n[STEP 1b] Loading event info & filtering thermal anomaly events")
    event_dfs = load_event_info()
    thermal_events = filter_thermal_events(event_dfs)
    save_event_shortlist(thermal_events)

    # - 1c + 1d ------------------------------─
    print("\n[STEP 1c] Loading datasets & [STEP 1d] Generating visualizations")

    for farm in FARMS:
        if farm not in thermal_events or farm not in event_dfs:
            print(f"  [SKIP] {farm}: missing event data")
            continue

        print(f"\n  - {farm} -")
        farm_thermal_events = thermal_events[farm]
        farm_all_events     = event_dfs[farm]

        anomaly_ids = get_event_ids(farm_thermal_events)
        label_col   = next((c for c in farm_all_events.columns if "label" in c), None)
        if label_col:
            normal_df   = farm_all_events[farm_all_events[label_col].str.lower() == "normal"]
            normal_ids  = get_event_ids(normal_df)
        else:
            normal_ids = []

        # load CSVs
        anomaly_dfs, _       = load_datasets_for_farm(farm, anomaly_ids, n_normal=0)
        _, normal_dfs        = load_datasets_for_farm(farm, anomaly_ids, n_normal=5)

        # for normal, explicitly load known normal ids
        if normal_ids:
            normal_dfs_explicit = []
            datasets_dir = EXTRACTED / farm / "datasets"
            for csv in sorted((datasets_dir).glob("*.csv")):
                stem = csv.stem.replace("event_", "").strip()
                try:
                    fid = int(stem)
                except ValueError:
                    fid = stem
                if fid in normal_ids and len(normal_dfs_explicit) < 4:
                    try:
                        df = pd.read_csv(csv, sep=";", parse_dates=["time_stamp"], encoding="latin-1")
                        normal_dfs_explicit.append((fid, df))
                    except Exception as e:
                        print(f"    [WARN] {csv.name}: {e}")
            if normal_dfs_explicit:
                normal_dfs = normal_dfs_explicit

        feat_df = feat_dfs.get(farm)

        plot_time_series(farm, anomaly_dfs, normal_dfs, feat_df)
        plot_distributions(farm, anomaly_dfs, normal_dfs, feat_df)
        plot_correlation_heatmap(farm, anomaly_dfs, feat_df)

    # - Summary -------------------------------
    print("\n" + "="*70)
    print("  EDA complete.")
    print(f"  Figures   -> {FIGS}")
    print(f"  Processed -> {DATA_PROC}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
