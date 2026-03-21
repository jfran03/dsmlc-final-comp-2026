"""
Visualization: anomaly score time-series overlaid with event boundaries.
Generates slide-ready plots for each scored event.

Output: outputs/figures/scores/{farm}/{event_id}_score_plot.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

SCORES_DIR   = ROOT / "outputs" / "scores"
FIGS_DIR     = ROOT / "outputs" / "figures" / "scores"
DATA_PROC    = ROOT / "data" / "processed"
FEATURES_DIR = ROOT / "outputs" / "features"

FARMS = ["Wind Farm A", "Wind Farm B", "Wind Farm C"]

PALETTE = {
    "cusum":    "#e07b39",   # orange — earliest signal
    "ae":       "#4a90d9",   # blue   — multivariate
    "if":       "#5ba85a",   # green  — reliability
    "ensemble": "#222222",   # black  — final score
    "tier1":    "#e07b39",
    "tier2":    "#c0392b",
    "event":    "#c0392b",
}


def _try_parse_dt(val) -> pd.Timestamp | None:
    try:
        result = pd.to_datetime(val)
        return None if pd.isnull(result) else result
    except Exception:
        return None


def plot_event(event_id, farm: str, scores_df: pd.DataFrame,
               detection_row: pd.Series | None, out_dir: Path):
    """
    Four-panel plot:
      Top    : CUSUM normalised score + Tier 1 threshold
      Middle : AE and IF normalised scores
      Bottom : Ensemble score + Tier 2 threshold + event_start line
    """
    has_ts = "time_stamp" in scores_df.columns
    if has_ts:
        x = pd.to_datetime(scores_df["time_stamp"])
    else:
        x = np.arange(len(scores_df))

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"{farm}  —  Event {event_id}  |  Anomaly Score Timeline",
        fontsize=13, fontweight="bold",
    )

    # ---- Panel 1: CUSUM ----
    ax = axes[0]
    ax.plot(x, scores_df["cusum_norm"], color=PALETTE["cusum"],
            linewidth=0.9, alpha=0.85, label="CUSUM (normalised)")
    ax.set_ylabel("CUSUM score", fontsize=9)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="upper left")
    ax.axhline(1.0, color=PALETTE["tier1"], linestyle="--",
               linewidth=0.8, label="Tier 1 boundary (≥1)")
    ax.legend(fontsize=8, loc="upper left")

    # ---- Panel 2: AE + IF ----
    ax = axes[1]
    ax.plot(x, scores_df["ae_norm"], color=PALETTE["ae"],
            linewidth=0.8, alpha=0.8, label="Autoencoder (normalised)")
    ax.plot(x, scores_df["if_norm"], color=PALETTE["if"],
            linewidth=0.8, alpha=0.8, label="Isolation Forest (normalised)")
    ax.set_ylabel("Component scores", fontsize=9)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="upper left")

    # ---- Panel 3: Ensemble ----
    ax = axes[2]
    ax.fill_between(x, scores_df["ensemble"],
                    alpha=0.25, color=PALETTE["ensemble"])
    ax.plot(x, scores_df["ensemble"], color=PALETTE["ensemble"],
            linewidth=1.1, label="Ensemble score")
    ax.set_ylabel("Ensemble score", fontsize=9)
    ax.set_ylim(bottom=0)

    # Tier 2 threshold line
    if detection_row is not None and pd.notna(detection_row.get("tier2_threshold")):
        thr = detection_row["tier2_threshold"]
        ax.axhline(thr, color=PALETTE["tier2"], linestyle="--",
                   linewidth=1.0, label=f"Tier 2 threshold ({thr:.2f})")

    # Event start line (on all panels)
    event_start = (_try_parse_dt(detection_row["event_start"])
                   if detection_row is not None else None)
    tier1_ts    = (_try_parse_dt(detection_row.get("tier1_detection"))
                   if detection_row is not None else None)
    tier2_ts    = (_try_parse_dt(detection_row.get("tier2_detection"))
                   if detection_row is not None else None)

    for a in axes:
        if has_ts and event_start:
            a.axvline(event_start, color=PALETTE["event"], linestyle="-",
                      linewidth=1.5, alpha=0.8)
        if has_ts and tier1_ts:
            a.axvline(tier1_ts, color=PALETTE["tier1"], linestyle=":",
                      linewidth=1.2, alpha=0.9)
        if has_ts and tier2_ts:
            a.axvline(tier2_ts, color=PALETTE["tier2"], linestyle=":",
                      linewidth=1.2, alpha=0.9)

    # Legend entries for vertical lines
    ax = axes[2]
    from matplotlib.lines import Line2D
    legend_extra = []
    if event_start:
        legend_extra.append(
            Line2D([0], [0], color=PALETTE["event"], linewidth=1.5, label="Event start")
        )
    if tier1_ts:
        legend_extra.append(
            Line2D([0], [0], color=PALETTE["tier1"], linestyle=":",
                   linewidth=1.2, label="Tier 1 detection (early warning)")
        )
    if tier2_ts:
        legend_extra.append(
            Line2D([0], [0], color=PALETTE["tier2"], linestyle=":",
                   linewidth=1.2, label="Tier 2 detection (confirmed)")
        )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + legend_extra,
              labels=labels + [l.get_label() for l in legend_extra],
              fontsize=8, loc="upper left")

    # Annotation: lead time
    if (detection_row is not None and
            pd.notna(detection_row.get("tier1_lead_hours"))):
        lead = detection_row["tier1_lead_hours"]
        ax.set_xlabel(
            f"Time   |   Tier 1 lead: {lead:.0f}h before event_start",
            fontsize=9,
        )

    # Date formatting
    if has_ts:
        for a in axes:
            a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate(rotation=30, ha="right")

    for a in axes:
        a.tick_params(labelsize=8)
        a.grid(True, alpha=0.25, linewidth=0.4)

    plt.tight_layout()
    out_path = out_dir / f"{event_id}_score_plot.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_path}")


def _load_sensor_desc(farm: str) -> dict:
    """Return {col: description} for a farm from thermal_sensor_list.csv."""
    path = DATA_PROC / "thermal_sensor_list.csv"
    if not path.exists():
        return {}
    import re
    sl = pd.read_csv(path)
    farm_sl = sl[sl["farm"] == farm]
    desc_map = {}
    for _, row in farm_sl.iterrows():
        name = str(row["sensor_name"]).strip()
        desc = str(row.get("description", "")).strip()
        for sfx in ["_avg", "_average", "_mean", ""]:
            desc_map[f"{name}{sfx}"] = desc
    return desc_map


def _base_sensor(col: str) -> str:
    import re
    return re.sub(r'_(res|perkw|ewm\d+|rm\d+|rs\d+|d\d+)$', '', col)


def plot_ae_reconstruction(event_id, farm: str, recon_df: pd.DataFrame,
                            detection_row: pd.Series | None, out_dir: Path,
                            top_n: int = 8):
    """
    Plot per-sensor AE reconstruction error for the top_n most anomalous sensors.
    Shows reconstruction error over time with event_start marked.
    Answers case question b: how AE reconstruction highlights subtle deviations.
    """
    feat_cols = [c for c in recon_df.columns if c != "time_stamp"]
    if not feat_cols:
        return

    has_ts = "time_stamp" in recon_df.columns
    x = pd.to_datetime(recon_df["time_stamp"]) if has_ts else np.arange(len(recon_df))

    # Group by base sensor, keep the feature with highest mean error per base sensor
    base_map: dict[str, tuple[str, float]] = {}  # base -> (best_col, mean_err)
    for col in feat_cols:
        base = _base_sensor(col)
        mean_err = float(recon_df[col].mean())
        if base not in base_map or mean_err > base_map[base][1]:
            base_map[base] = (col, mean_err)

    # Pick top_n base sensors by mean error
    ranked = sorted(base_map.items(), key=lambda kv: kv[1][1], reverse=True)[:top_n]
    if not ranked:
        return

    desc_map = _load_sensor_desc(farm)
    n = len(ranked)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f"{farm}  —  Event {event_id}  |  AE Reconstruction Error by Sensor",
        fontsize=12, fontweight="bold",
    )

    event_start = (_try_parse_dt(detection_row["event_start"])
                   if detection_row is not None else None)

    cmap = plt.cm.tab10.colors
    for idx, (base, (col, _)) in enumerate(ranked):
        ax = axes[idx]
        ax.fill_between(x, recon_df[col].values, alpha=0.35, color=cmap[idx % 10])
        ax.plot(x, recon_df[col].values, linewidth=0.8, color=cmap[idx % 10])
        label = desc_map.get(base, base)[:45]
        ax.set_ylabel(label, fontsize=7, labelpad=4)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25, linewidth=0.4)
        if has_ts and event_start:
            ax.axvline(event_start, color=PALETTE["event"], linestyle="-",
                       linewidth=1.5, alpha=0.85, label="Event start")
            if idx == 0:
                ax.legend(fontsize=7, loc="upper left")

    axes[-1].set_xlabel("Time", fontsize=9)
    if has_ts:
        for a in axes:
            a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate(rotation=30, ha="right")

    plt.tight_layout()
    out_path = out_dir / f"{event_id}_ae_reconstruction.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_path}")


def main():
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    # Load detection summary for metadata
    summary_path = SCORES_DIR / "detection_summary.csv"
    if not summary_path.exists():
        print(f"[ERROR] Run ensemble.py first — {summary_path} not found")
        return
    detection = pd.read_csv(summary_path)

    print("=" * 60)
    print("  ANOMALY SCORE VISUALIZATION")
    print("=" * 60)

    for farm in FARMS:
        farm_score_dir = SCORES_DIR / farm.replace(" ", "_")
        if not farm_score_dir.exists():
            print(f"\n[SKIP] {farm}: no scores directory")
            continue

        out_dir = FIGS_DIR / farm.replace(" ", "_")
        out_dir.mkdir(parents=True, exist_ok=True)

        farm_detection = detection[detection["farm"] == farm]
        score_files    = sorted(farm_score_dir.glob("*_scores.csv"))

        print(f"\n  {farm}: {len(score_files)} score files")

        for csv_path in score_files:
            eid_str = csv_path.stem.replace("_scores", "")
            try:
                event_id = int(eid_str)
            except ValueError:
                event_id = eid_str

            try:
                scores_df = pd.read_csv(csv_path)
            except Exception as exc:
                print(f"  [WARN] Cannot read {csv_path.name}: {exc}")
                continue

            # Match detection row
            det_rows = farm_detection[farm_detection["event_id"] == event_id]
            det_row  = det_rows.iloc[0] if not det_rows.empty else None

            plot_event(event_id, farm, scores_df, det_row, out_dir)

            # AE reconstruction plot (only if recon file was saved by ensemble.py)
            recon_path = farm_score_dir / f"{eid_str}_ae_recon.csv"
            if recon_path.exists() and det_row is not None:
                try:
                    recon_df = pd.read_csv(recon_path)
                    plot_ae_reconstruction(event_id, farm, recon_df, det_row, out_dir)
                except Exception as exc:
                    print(f"  [WARN] AE recon plot failed for {event_id}: {exc}")

    print("\n" + "=" * 60)
    print(f"  Figures -> {FIGS_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
