"""
CARE score evaluation.

Computes the four CARE axes from detection_summary.csv and score files:
  Coverage   — fraction of anomaly events where Tier 2 fires
  Accuracy   — AUC separating anomaly vs. normal test-window ensemble scores
  Reliability — false alarm rate on normal events
  Earliness  — median / mean Tier 1 lead time before event_start (hours)

Outputs: outputs/evaluation/care_scores.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DATA_PROC    = ROOT / "data" / "processed"
FEATURES_DIR = ROOT / "outputs" / "features"
SCORES_DIR   = ROOT / "outputs" / "scores"
EVAL_DIR     = ROOT / "outputs" / "evaluation"

FARMS = ["Wind Farm A", "Wind Farm B", "Wind Farm C"]


def load_detection_summary() -> pd.DataFrame:
    path = SCORES_DIR / "detection_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run ensemble.py first — {path} not found")
    return pd.read_csv(path)


def load_event_infos() -> dict[str, pd.DataFrame]:
    result = {}
    for farm in FARMS:
        path = (ROOT / "data" / "raw" / "CARE_To_Compare"
                / farm / "event_info.csv")
        if path.exists():
            df = pd.read_csv(path, sep=";")
            df.columns = df.columns.str.strip().str.lower()
            result[farm] = df
    return result


def _get_train_test_mask(df: pd.DataFrame):
    if "train_test" not in df.columns:
        return np.zeros(len(df), dtype=bool), np.ones(len(df), dtype=bool)
    tt = df["train_test"].str.lower().str.strip()
    return (tt.isin(["train", "0"]).values,
            tt.isin(["test", "1", "prediction"]).values)


def compute_auc_score(anomaly_scores: np.ndarray, normal_scores: np.ndarray) -> float:
    """Mann-Whitney U AUC (no sklearn dependency on labels)."""
    all_scores = np.concatenate([anomaly_scores, normal_scores])
    labels     = np.concatenate([np.ones(len(anomaly_scores)),
                                  np.zeros(len(normal_scores))])
    order = np.argsort(-all_scores)   # descending
    n_pos, n_neg = len(anomaly_scores), len(normal_scores)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp, fp = 0, 0
    auc = 0.0
    prev_fp = 0
    for i in order:
        if labels[i] == 1:
            tp += 1
        else:
            fp += 1
            auc += tp * (fp - prev_fp)
            prev_fp = fp
    auc /= (n_pos * n_neg)
    return float(auc)


def collect_test_window_scores(farm: str, label: str,
                                event_infos: dict) -> np.ndarray:
    """
    Collect all ensemble scores from test windows of events with the given label.
    """
    farm_feat_dir  = FEATURES_DIR / farm.replace(" ", "_")
    farm_score_dir = SCORES_DIR   / farm.replace(" ", "_")
    event_info     = event_infos.get(farm, pd.DataFrame())

    lbl_col = next((c for c in event_info.columns if "label" in c), None)
    id_col  = next((c for c in event_info.columns
                    if c in ("event_id", "id")), None)

    if lbl_col and id_col:
        target_ids = set(
            event_info.loc[event_info[lbl_col].str.lower() == label, id_col]
        )
    else:
        target_ids = set()

    all_scores = []

    # For anomaly events: load from score files (already scored)
    if label == "anomaly":
        for csv_path in sorted(farm_score_dir.glob("*_scores.csv")):
            eid_str = csv_path.stem.replace("_scores", "")
            try:
                eid = int(eid_str)
            except ValueError:
                eid = eid_str
            if target_ids and eid not in target_ids:
                continue
            try:
                df = pd.read_csv(csv_path)
                if "ensemble" in df.columns:
                    all_scores.extend(df["ensemble"].dropna().tolist())
            except Exception:
                pass

    # For normal events: re-score from feature files using raw ensemble scores
    # (We approximate: load feature file, check if score file exists, else skip)
    elif label == "normal":
        for csv_path in sorted(farm_score_dir.glob("*_scores.csv")):
            eid_str = csv_path.stem.replace("_scores", "")
            try:
                eid = int(eid_str)
            except ValueError:
                eid = eid_str
            if target_ids and eid not in target_ids:
                continue
            # Skip anomaly events
            event_info = event_infos.get(farm, pd.DataFrame())
            lbl_col2 = next((c for c in event_info.columns if "label" in c), None)
            id_col2  = next((c for c in event_info.columns
                             if c in ("event_id", "id")), None)
            if lbl_col2 and id_col2:
                anomaly_ids = set(
                    event_info.loc[event_info[lbl_col2].str.lower() == "anomaly", id_col2]
                )
                if eid in anomaly_ids:
                    continue
            try:
                df = pd.read_csv(csv_path)
                if "ensemble" in df.columns:
                    all_scores.extend(df["ensemble"].dropna().tolist())
            except Exception:
                pass

    return np.array(all_scores, dtype=float)


def compute_false_alarm_rate(farm: str, event_infos: dict,
                              tier2_threshold: float) -> float:
    """
    FAR = fraction of normal events' test-window timesteps above tier2_threshold.
    Uses pre-computed score files for normal events if available.
    """
    farm_score_dir = SCORES_DIR / farm.replace(" ", "_")
    event_info     = event_infos.get(farm, pd.DataFrame())

    lbl_col = next((c for c in event_info.columns if "label" in c), None)
    id_col  = next((c for c in event_info.columns
                    if c in ("event_id", "id")), None)

    if lbl_col and id_col:
        anomaly_ids = set(
            event_info.loc[event_info[lbl_col].str.lower() == "anomaly", id_col]
        )
    else:
        anomaly_ids = set()

    total, alerts = 0, 0
    for csv_path in sorted(farm_score_dir.glob("*_scores.csv")):
        eid_str = csv_path.stem.replace("_scores", "")
        try:
            eid = int(eid_str)
        except ValueError:
            eid = eid_str
        if eid in anomaly_ids:
            continue   # skip anomaly events for FAR
        try:
            df = pd.read_csv(csv_path)
            if "ensemble" in df.columns:
                scores = df["ensemble"].dropna().values
                total  += len(scores)
                alerts += int((scores > tier2_threshold).sum())
        except Exception:
            pass

    return alerts / total if total > 0 else float("nan")


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  CARE Score Evaluation")
    print("=" * 60)

    detection = load_detection_summary()
    event_infos = load_event_infos()

    results = []

    for farm in FARMS:
        farm_rows = detection[detection["farm"] == farm]
        if farm_rows.empty:
            print(f"\n  {farm}: no detection data — skipping")
            continue

        print(f"\n  {farm}")
        print(f"  {'-'*40}")

        # --- Coverage ---
        n_events   = len(farm_rows)
        n_detected = farm_rows["tier2_detection"].notna().sum()
        coverage   = n_detected / n_events if n_events > 0 else float("nan")
        print(f"  Coverage     : {n_detected}/{n_events} = {coverage:.2%}")

        # --- Earliness ---
        # Prefer Tier 1 lead times; fall back to Tier 2 if Tier 1 never fires.
        tier1_leads = farm_rows["tier1_lead_hours"].dropna()
        if len(tier1_leads) > 0:
            lead_hours   = tier1_leads
            earliness_src = "Tier1"
        else:
            lead_hours   = farm_rows["tier2_lead_hours"].dropna()
            earliness_src = "Tier2(fallback)"
        earliness_median = float(lead_hours.median()) if len(lead_hours) > 0 else float("nan")
        earliness_mean   = float(lead_hours.mean())   if len(lead_hours) > 0 else float("nan")
        print(f"  Earliness    : median {earliness_median:.0f}h | "
              f"mean {earliness_mean:.0f}h | max {lead_hours.max():.0f}h [{earliness_src}]"
              if len(lead_hours) > 0 else "  Earliness    : n/a")

        # --- Accuracy (AUC) ---
        anom_scores = collect_test_window_scores(farm, "anomaly", event_infos)
        norm_scores = collect_test_window_scores(farm, "normal",  event_infos)
        auc = compute_auc_score(anom_scores, norm_scores)
        print(f"  Accuracy AUC : {auc:.3f}" if not np.isnan(auc)
              else "  Accuracy AUC : n/a (no normal scores available)")

        # --- Reliability (FAR) ---
        tier2_thr = farm_rows["tier2_threshold"].iloc[0]
        far = compute_false_alarm_rate(farm, event_infos, tier2_thr)
        reliability = 1 - far if not np.isnan(far) else float("nan")
        print(f"  Reliability  : FAR={far:.1%} → score={reliability:.2%}"
              if not np.isnan(far) else "  Reliability  : n/a")

        results.append({
            "farm":               farm,
            "n_anomaly_events":   n_events,
            "coverage":           round(coverage, 4),
            "earliness_median_h": round(earliness_median, 1) if not np.isnan(earliness_median) else None,
            "earliness_mean_h":   round(earliness_mean, 1)   if not np.isnan(earliness_mean)   else None,
            "earliness_source":   earliness_src,
            "accuracy_auc":       round(auc, 4) if not np.isnan(auc) else None,
            "far":                round(far, 4) if not np.isnan(far) else None,
            "reliability":        round(reliability, 4) if not np.isnan(reliability) else None,
        })

    if results:
        out_df = pd.DataFrame(results)
        out_path = EVAL_DIR / "care_scores.csv"
        out_df.to_csv(out_path, index=False)
        print(f"\n  Saved -> {out_path}")
        print("\n" + out_df.to_string(index=False))

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
