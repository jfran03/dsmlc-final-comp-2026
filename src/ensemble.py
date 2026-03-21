"""
Ensemble orchestrator: trains CUSUM + Autoencoder + Isolation Forest
on normal events and scores all events in the thermal shortlist.

Weights  (tuned for CARE score):
  CUSUM  0.40  — drives Earliness
  AE     0.35  — drives Coverage (multivariate joint anomalies)
  IF     0.25  — drives Reliability (low false alarms)

Two-tier alert system
  Tier 1 (early warning) : CUSUM normalised score > 60th pct of normal scores
  Tier 2 (confirmed)     : ensemble score > 95th pct, sustained ≥ 3 consecutive steps

Outputs
  outputs/scores/{Farm_X}/{event_id}_scores.csv   — per-timestep scores
  outputs/scores/detection_summary.csv             — detection timestamps per event
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from models.cusum import CUSUMDetector
from models.autoencoder import DenseAutoencoder
from models.isolation_forest import IFDetector

DATA_PROC    = ROOT / "data" / "processed"
FEATURES_DIR = ROOT / "outputs" / "features"
SCORES_DIR   = ROOT / "outputs" / "scores"

FARMS = ["Wind Farm A", "Wind Farm B", "Wind Farm C"]

# Ensemble weights
W_CUSUM = 0.40
W_AE    = 0.35
W_IF    = 0.25

# Minimum consecutive steps above threshold for a confirmed alert
MIN_DURATION = 3

# Percentile thresholds (calibrated on normal events' test-window scores)
TIER1_PCT = 80    # Tier 1: early warning (CUSUM only)
TIER2_PCT = 95    # Tier 2: confirmed anomaly (ensemble)


# ---------- Helpers ----------

def _normalise(scores: np.ndarray, ref_scores: np.ndarray) -> np.ndarray:
    """Min-max normalise scores using ref_scores percentiles as [0,1] range."""
    lo  = float(np.nanpercentile(ref_scores, 1))
    hi  = float(np.nanpercentile(ref_scores, 99))
    if hi - lo < 1e-9:
        return np.zeros_like(scores, dtype=float)
    return np.clip((scores - lo) / (hi - lo), 0, None)


def _get_train_test_mask(df: pd.DataFrame):
    """Return (train_mask, test_mask) boolean arrays."""
    if "train_test" not in df.columns:
        n = len(df)
        return np.ones(n, dtype=bool), np.zeros(n, dtype=bool)
    tt = df["train_test"].str.lower().str.strip()
    train_mask = tt.isin(["train", "0"]).values
    test_mask  = tt.isin(["test",  "1", "prediction"]).values
    return train_mask, test_mask


def _normal_train_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return training rows with normal status."""
    train_mask, _ = _get_train_test_mask(df)
    if "status_type_id" in df.columns:
        normal_mask = df["status_type_id"].isin({0, 2}).values
    else:
        normal_mask = np.ones(len(df), dtype=bool)
    return df[train_mask & normal_mask]


def _first_sustained_alert(scores: np.ndarray, threshold: float,
                            min_dur: int = MIN_DURATION) -> int | None:
    """
    Return the index of the first timestep in a run of ≥ min_dur
    consecutive values above threshold.  None if no such run exists.
    """
    count = 0
    for i, s in enumerate(scores):
        if s > threshold:
            count += 1
            if count >= min_dur:
                return i - min_dur + 1   # start of the run
        else:
            count = 0
    return None


def _feature_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns, excluding metadata."""
    meta = {"time_stamp", "asset_id", "id", "train_test", "status_type_id"}
    return [c for c in df.columns
            if c not in meta and pd.api.types.is_numeric_dtype(df[c])]


# ---------- Per-farm model training ----------

def train_models_for_farm(farm: str, event_infos: dict
                           ) -> tuple[CUSUMDetector, DenseAutoencoder, IFDetector] | None:
    """
    Train CUSUM, AE, and IF on normal events' training windows for a farm.
    Returns the three fitted model objects, or None if no normal data available.
    """
    farm_feat_dir = FEATURES_DIR / farm.replace(" ", "_")
    if not farm_feat_dir.exists():
        print(f"  [WARN] {farm}: feature dir not found — skipping")
        return None

    event_info = event_infos.get(farm, pd.DataFrame())
    lbl_col    = next((c for c in event_info.columns if "label" in c), None)
    id_col     = next((c for c in event_info.columns
                       if c in ("event_id", "id", "dataset_id")), None)

    if lbl_col and id_col:
        normal_ids = set(
            event_info.loc[event_info[lbl_col].str.lower() == "normal", id_col]
        )
    else:
        normal_ids = set()   # no label info — fall back to all available

    # Collect normal training rows
    normal_frames = []
    csv_files     = sorted(farm_feat_dir.glob("*.csv"))

    for csv_path in csv_files:
        stem = csv_path.stem
        try:
            eid = int(stem)
        except ValueError:
            eid = stem
        # If we have labels, restrict to known normal events; otherwise use all
        if normal_ids and eid not in normal_ids:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"    [WARN] cannot read {csv_path.name}: {exc}")
            continue
        normal_rows = _normal_train_rows(df)
        if len(normal_rows) > 0:
            normal_frames.append(normal_rows)

    if not normal_frames:
        print(f"  [WARN] {farm}: no normal training rows found — skipping farm")
        return None

    X_normal = pd.concat(normal_frames, ignore_index=True)
    feat_cols = _feature_cols(X_normal)
    X_normal  = X_normal[feat_cols]
    print(f"  [{farm}] Normal training pool: {len(X_normal):,} rows, "
          f"{len(feat_cols)} features")

    print(f"  [{farm}] Fitting CUSUM ...")
    cusum = CUSUMDetector(k_factor=0.5, h_factor=4.0).fit(X_normal)

    print(f"  [{farm}] Fitting Autoencoder ...")
    ae = DenseAutoencoder(max_iter=300).fit(X_normal)

    print(f"  [{farm}] Fitting Isolation Forest ...")
    iso = IFDetector(contamination=0.05).fit(X_normal)

    return cusum, ae, iso


# ---------- Reference score calibration ----------

def calibrate_thresholds(farm: str, cusum: CUSUMDetector,
                          ae: DenseAutoencoder, iso: IFDetector,
                          event_infos: dict) -> dict:
    """
    Score normal events' test windows to calibrate Tier 1 / Tier 2 thresholds.
    Returns {tier1_cusum, tier2_ensemble, cusum_ref, ae_ref, if_ref}.
    """
    farm_feat_dir = FEATURES_DIR / farm.replace(" ", "_")
    event_info    = event_infos.get(farm, pd.DataFrame())
    lbl_col       = next((c for c in event_info.columns if "label" in c), None)
    id_col        = next((c for c in event_info.columns
                          if c in ("event_id", "id", "dataset_id")), None)

    if lbl_col and id_col:
        normal_ids = set(
            event_info.loc[event_info[lbl_col].str.lower() == "normal", id_col]
        )
    else:
        normal_ids = set()

    cusum_ref_scores, ae_ref_scores, if_ref_scores = [], [], []

    for csv_path in sorted(farm_feat_dir.glob("*.csv")):
        stem = csv_path.stem
        try:
            eid = int(stem)
        except ValueError:
            eid = stem
        if normal_ids and eid not in normal_ids:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        _, test_mask = _get_train_test_mask(df)
        df_test = df[test_mask]
        if df_test.empty:
            continue
        feat_cols = _feature_cols(df_test)
        X = df_test[feat_cols]

        cusum_ref_scores.extend(cusum.score(X).tolist())
        ae_ref_scores.extend(ae.score(X).tolist())
        if_ref_scores.extend(iso.score(X).tolist())

    cusum_ref = np.array(cusum_ref_scores) if cusum_ref_scores else np.array([0.0])
    ae_ref    = np.array(ae_ref_scores)    if ae_ref_scores    else np.array([0.0])
    if_ref    = np.array(if_ref_scores)    if if_ref_scores    else np.array([0.0])

    # Tier 1: CUSUM Tier 1 (raw score percentile on normal test windows)
    tier1_cusum = float(np.nanpercentile(cusum_ref, TIER1_PCT))
    # Tier 2: ensemble threshold (on normalised ensemble scores on normal test windows)
    ens_ref_norm = (
        W_CUSUM * _normalise(cusum_ref, cusum_ref) +
        W_AE    * _normalise(ae_ref,    ae_ref)    +
        W_IF    * _normalise(if_ref,    if_ref)
    )
    tier2_ens = float(np.nanpercentile(ens_ref_norm, TIER2_PCT))

    return {
        "tier1_cusum":     tier1_cusum,
        "tier2_ensemble":  tier2_ens,
        "cusum_ref":       cusum_ref,
        "ae_ref":          ae_ref,
        "if_ref":          if_ref,
    }


# ---------- Scoring anomaly events ----------

def score_event(event_id, farm: str,
                cusum: CUSUMDetector, ae: DenseAutoencoder, iso: IFDetector,
                thresholds: dict, event_info_row: pd.Series | None,
                out_dir: Path) -> dict | None:
    """
    Score a single event's test window. Saves per-timestep scores CSV.
    Returns detection metadata dict.
    """
    feat_path = FEATURES_DIR / farm.replace(" ", "_") / f"{event_id}.csv"
    if not feat_path.exists():
        print(f"    [WARN] Feature file not found: {feat_path}")
        return None

    try:
        df = pd.read_csv(feat_path, parse_dates=["time_stamp"])
    except Exception as exc:
        print(f"    [WARN] Cannot read {feat_path.name}: {exc}")
        return None

    _, test_mask = _get_train_test_mask(df)
    df_test = df[test_mask].reset_index(drop=True)

    if df_test.empty:
        print(f"    [WARN] event {event_id}: empty test window")
        return None

    feat_cols = _feature_cols(df_test)
    X_test    = df_test[feat_cols]

    # Raw scores
    cusum_raw = cusum.score(X_test)
    ae_raw    = ae.score(X_test)
    if_raw    = iso.score(X_test)

    # Normalise relative to normal reference distributions
    cusum_n = _normalise(cusum_raw, thresholds["cusum_ref"])
    ae_n    = _normalise(ae_raw,    thresholds["ae_ref"])
    if_n    = _normalise(if_raw,    thresholds["if_ref"])

    ensemble = W_CUSUM * cusum_n + W_AE * ae_n + W_IF * if_n

    # --- Detection timestamps ---
    timestamps = df_test.get("time_stamp", pd.Series(range(len(df_test))))

    # Tier 1: first CUSUM raw score > tier1 threshold (earliest signal)
    tier1_idx = _first_sustained_alert(cusum_raw, thresholds["tier1_cusum"], min_dur=3)
    tier1_ts  = timestamps.iloc[tier1_idx] if tier1_idx is not None else None

    # Tier 2: ensemble sustained ≥ MIN_DURATION steps above threshold
    tier2_idx = _first_sustained_alert(ensemble, thresholds["tier2_ensemble"], MIN_DURATION)
    tier2_ts  = timestamps.iloc[tier2_idx] if tier2_idx is not None else None

    # Build output DataFrame
    scores_df = pd.DataFrame({
        "time_stamp":   timestamps.values,
        "cusum_raw":    cusum_raw,
        "ae_raw":       ae_raw,
        "if_raw":       if_raw,
        "cusum_norm":   cusum_n,
        "ae_norm":      ae_n,
        "if_norm":      if_n,
        "ensemble":     ensemble,
    })

    out_path = out_dir / f"{event_id}_scores.csv"
    scores_df.to_csv(out_path, index=False)

    # Per-feature AE reconstruction error (only for tracked anomaly events)
    if event_info_row is not None:
        try:
            recon_df = ae.per_feature_error(X_test)
            recon_df.insert(0, "time_stamp", timestamps.values)
            recon_df.to_csv(out_dir / f"{event_id}_ae_recon.csv", index=False)
        except Exception as exc:
            print(f"    [WARN] AE reconstruction save failed for {event_id}: {exc}")

    # Earliness calculation
    event_start = None
    if event_info_row is not None:
        start_col = next(
            (c for c in event_info_row.index if "event_start" in c and "id" not in c),
            None,
        )
        if start_col:
            try:
                event_start = pd.to_datetime(event_info_row[start_col])
            except Exception:
                pass

    def _lead_hours(detection_ts, event_start):
        if detection_ts is None or event_start is None:
            return None
        try:
            delta = pd.to_datetime(event_start) - pd.to_datetime(detection_ts)
            return round(delta.total_seconds() / 3600, 1)
        except Exception:
            return None

    result = {
        "farm":              farm,
        "event_id":          event_id,
        "n_test_rows":       len(df_test),
        "event_start":       str(event_start) if event_start else None,
        "tier1_detection":   str(tier1_ts) if tier1_ts is not None else None,
        "tier2_detection":   str(tier2_ts) if tier2_ts is not None else None,
        "tier1_lead_hours":  _lead_hours(tier1_ts, event_start),
        "tier2_lead_hours":  _lead_hours(tier2_ts, event_start),
        "max_cusum_norm":    float(np.nanmax(cusum_n)),
        "max_ensemble":      float(np.nanmax(ensemble)),
        "tier2_threshold":   thresholds["tier2_ensemble"],
        "scores_path":       str(out_path),
    }

    flag = "DETECTED" if tier2_ts is not None else "MISSED"
    early_str = (f"{result['tier1_lead_hours']:.0f}h early"
                 if result["tier1_lead_hours"] is not None else "n/a")
    print(f"    event {event_id}: {flag} | Tier1 lead={early_str} "
          f"| max_ens={result['max_ensemble']:.3f}")
    return result


# ---------- Feature importance ----------

def _base_sensor(col: str) -> str:
    """Strip engineered suffixes to recover the raw sensor column name."""
    col = re.sub(r'_(res|perkw|ewm\d+|rm\d+|rs\d+|d\d+)$', '', col)
    return col


def compute_feature_importance(farm: str, iso: IFDetector, ae: DenseAutoencoder,
                                event_infos: dict, sensor_list_df: pd.DataFrame | None,
                                out_dir: Path) -> None:
    """
    Compute IF permutation importance + mean AE error on anomaly test windows,
    group by base sensor, merge descriptions, and save CSV + bar chart.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    farm_feat_dir = FEATURES_DIR / farm.replace(" ", "_")
    event_info    = event_infos.get(farm, pd.DataFrame())
    lbl_col = next((c for c in event_info.columns if "label" in c), None)
    id_col  = next((c for c in event_info.columns
                    if c in ("event_id", "id", "dataset_id")), None)

    if lbl_col and id_col:
        anomaly_ids = set(
            event_info.loc[event_info[lbl_col].str.lower() == "anomaly", id_col]
        )
        normal_ids = set(
            event_info.loc[event_info[lbl_col].str.lower() == "normal", id_col]
        )
    else:
        anomaly_ids, normal_ids = set(), set()

    # --- Collect normal test rows for IF permutation importance ---
    normal_frames = []
    for csv_path in sorted(farm_feat_dir.glob("*.csv")):
        stem = csv_path.stem
        try:
            eid = int(stem)
        except ValueError:
            eid = stem
        if normal_ids and eid not in normal_ids:
            continue
        try:
            df = pd.read_csv(csv_path)
            _, test_mask = _get_train_test_mask(df)
            df_test = df[test_mask]
            if not df_test.empty:
                normal_frames.append(df_test[_feature_cols(df_test)])
        except Exception:
            pass

    if not normal_frames:
        print(f"  [WARN] {farm}: no normal test data for feature importance")
        return

    X_normal_test = pd.concat(normal_frames, ignore_index=True)

    # --- IF permutation importance ---
    print(f"  [{farm}] Computing IF feature importance ...")
    try:
        if_imp = iso.feature_importance(X_normal_test, n_repeats=3)
    except Exception as exc:
        print(f"  [WARN] IF importance failed: {exc}")
        if_imp = pd.Series(dtype=float)

    # --- Mean AE reconstruction error on anomaly test windows ---
    ae_errors = []
    for csv_path in sorted(farm_feat_dir.glob("*.csv")):
        stem = csv_path.stem
        try:
            eid = int(stem)
        except ValueError:
            eid = stem
        if anomaly_ids and eid not in anomaly_ids:
            continue
        try:
            df = pd.read_csv(csv_path)
            _, test_mask = _get_train_test_mask(df)
            df_test = df[test_mask]
            if not df_test.empty:
                err = ae.per_feature_error(df_test[_feature_cols(df_test)])
                ae_errors.append(err.mean())
        except Exception:
            pass

    ae_imp = pd.concat(ae_errors).groupby(level=0).mean() if ae_errors else pd.Series(dtype=float)

    # --- Group by base sensor, take max importance ---
    def _group_by_base(imp: pd.Series) -> pd.Series:
        base = imp.index.map(_base_sensor)
        return imp.groupby(base).max().sort_values(ascending=False)

    if_grouped = _group_by_base(if_imp) if not if_imp.empty else pd.Series(dtype=float)
    ae_grouped = _group_by_base(ae_imp) if not ae_imp.empty else pd.Series(dtype=float)

    # --- Merge with sensor descriptions ---
    desc_map = {}
    if sensor_list_df is not None:
        farm_sensors = sensor_list_df[sensor_list_df["farm"] == farm]
        for _, row in farm_sensors.iterrows():
            name = str(row["sensor_name"]).strip()
            for sfx in ["_avg", "_average", "_mean", ""]:
                desc_map[f"{name}{sfx}"] = str(row.get("description", "")).strip()

    all_sensors = list(dict.fromkeys(
        list(if_grouped.index) + list(ae_grouped.index)
    ))
    rows = []
    for sensor in all_sensors:
        rows.append({
            "sensor":       sensor,
            "description":  desc_map.get(sensor, ""),
            "if_importance": round(if_grouped.get(sensor, float("nan")), 6),
            "ae_mean_error": round(ae_grouped.get(sensor, float("nan")), 6),
        })
    imp_df = pd.DataFrame(rows)
    imp_df = imp_df.sort_values("if_importance", ascending=False).reset_index(drop=True)

    csv_path = out_dir / f"{farm.replace(' ', '_')}_feature_importance.csv"
    imp_df.to_csv(csv_path, index=False)
    print(f"  [{farm}] Feature importance -> {csv_path}")

    # --- Bar chart: top 20 sensors by IF importance ---
    top = imp_df.head(20).copy()
    if top.empty:
        return
    labels = [
        f"{r['sensor']}" + (f"\n({r['description'][:30]})" if r['description'] else "")
        for _, r in top.iterrows()
    ]
    fig, ax = plt.subplots(figsize=(10, max(6, len(top) * 0.4)))
    bars = ax.barh(range(len(top)), top["if_importance"].values, color="#4a90d9", alpha=0.8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("IF Permutation Importance (mean score increase on shuffle)", fontsize=9)
    ax.set_title(f"{farm} — Top Sensor Feature Importance (Isolation Forest)", fontsize=11)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    fig_path = out_dir / f"{farm.replace(' ', '_')}_feature_importance.png"
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [{farm}] Feature importance chart -> {fig_path}")


# ---------- Main ----------

def main():
    SCORES_DIR.mkdir(parents=True, exist_ok=True)

    # Load event info
    event_infos = {}
    for farm in FARMS:
        path = ROOT / "data" / "raw" / "CARE_To_Compare" / farm / "event_info.csv"
        if path.exists():
            df = pd.read_csv(path, sep=";")
            df.columns = df.columns.str.strip().str.lower()
            event_infos[farm] = df

    # Load sensor descriptions for feature importance labelling
    sensor_list_path = ROOT / "data" / "processed" / "thermal_sensor_list.csv"
    sensor_list_df = pd.read_csv(sensor_list_path) if sensor_list_path.exists() else None

    # Load thermal event shortlist (anomaly targets)
    shortlist_path = DATA_PROC / "thermal_event_shortlist.csv"
    shortlist = pd.read_csv(shortlist_path) if shortlist_path.exists() else pd.DataFrame()

    all_results = []

    for farm in FARMS:
        farm_feat_dir = FEATURES_DIR / farm.replace(" ", "_")
        if not farm_feat_dir.exists():
            print(f"\n[SKIP] {farm}: no feature directory")
            continue

        print(f"\n{'='*60}")
        print(f"  {farm}")
        print(f"{'='*60}")

        print(f"\n[1] Training models ...")
        models = train_models_for_farm(farm, event_infos)
        if models is None:
            continue
        cusum, ae, iso = models

        print(f"\n[2] Calibrating thresholds ...")
        thresholds = calibrate_thresholds(farm, cusum, ae, iso, event_infos)
        print(f"     Tier1 CUSUM threshold : {thresholds['tier1_cusum']:.4f}")
        print(f"     Tier2 ensemble thresh : {thresholds['tier2_ensemble']:.4f}")

        out_dir = SCORES_DIR / farm.replace(" ", "_")
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[2b] Computing feature importance ...")
        feat_out_dir = ROOT / "outputs" / "features"
        feat_out_dir.mkdir(parents=True, exist_ok=True)
        compute_feature_importance(farm, iso, ae, event_infos, sensor_list_df, feat_out_dir)

        # Score thermal anomaly events
        if not shortlist.empty:
            farm_events = shortlist[shortlist["farm"] == farm]
        else:
            farm_events = pd.DataFrame()

        print(f"\n[3] Scoring {len(farm_events)} thermal anomaly events ...")
        event_info_df = event_infos.get(farm, pd.DataFrame())

        for _, ev_row in farm_events.iterrows():
            event_id = ev_row["event_id"]
            # Look up event_info row for timestamps
            id_col = next((c for c in event_info_df.columns
                           if c in ("event_id", "id")), None)
            ev_info_row = None
            if id_col and not event_info_df.empty:
                match = event_info_df[event_info_df[id_col] == event_id]
                if not match.empty:
                    ev_info_row = match.iloc[0]

            result = score_event(event_id, farm, cusum, ae, iso,
                                 thresholds, ev_info_row, out_dir)
            if result:
                all_results.append(result)

        # Score normal events so evaluate.py can compute AUC and FAR
        lbl_col = next((c for c in event_info_df.columns if "label" in c), None)
        id_col  = next((c for c in event_info_df.columns
                        if c in ("event_id", "id")), None)
        if lbl_col and id_col:
            normal_ids = set(
                event_info_df.loc[
                    event_info_df[lbl_col].str.lower() == "normal", id_col
                ]
            )
        else:
            normal_ids = set()

        print(f"\n[4] Scoring {len(normal_ids)} normal events for FAR/AUC ...")
        for event_id in sorted(normal_ids):
            score_event(event_id, farm, cusum, ae, iso, thresholds, None, out_dir)

    # Save detection summary
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = SCORES_DIR / "detection_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'='*60}")
        print(f"  Detection summary -> {summary_path}")
        print(f"  {summary_df['tier2_detection'].notna().sum()} / {len(summary_df)} "
              f"events confirmed (Tier 2)")
        tier1_leads = summary_df["tier1_lead_hours"].dropna()
        if len(tier1_leads) > 0:
            print(f"  Tier 1 lead time   : median {tier1_leads.median():.0f}h, "
                  f"max {tier1_leads.max():.0f}h")
        print(f"{'='*60}\n")
    else:
        print("\n[WARN] No results produced.")


if __name__ == "__main__":
    main()
