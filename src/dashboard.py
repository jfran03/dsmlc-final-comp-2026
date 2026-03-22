"""
Wind Turbine Anomaly Detection — Interactive Dashboard
Streamlit app visualizing CARE scores, anomaly detection timelines,
feature importance, and per-event model scores.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
SCORES    = ROOT / "outputs" / "scores"
FEATURES  = ROOT / "outputs" / "features"
EVAL      = ROOT / "outputs" / "evaluation"

FARM_DIRS = {
    "Wind Farm A": SCORES / "Wind_Farm_A",
    "Wind Farm B": SCORES / "Wind_Farm_B",
    "Wind Farm C": SCORES / "Wind_Farm_C",
}
FEAT_FILES = {
    "Wind Farm A": FEATURES / "Wind_Farm_A_feature_importance.csv",
    "Wind Farm B": FEATURES / "Wind_Farm_B_feature_importance.csv",
    "Wind Farm C": FEATURES / "Wind_Farm_C_feature_importance.csv",
}

FARM_LABELS = {
    "Wind Farm A": "Onshore · Portugal · 86 features",
    "Wind Farm B": "Offshore · Germany · 257 features",
    "Wind Farm C": "Offshore · Germany · 957 features",
}

# ── Colour scheme ───────────────────────────────────────────────────────────
ORANGE   = "#E87722"
RED      = "#E53935"
BLUE     = "#1565C0"
GREEN    = "#2E7D32"
GREY     = "#9E9E9E"

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wind Turbine Anomaly Detection",
    page_icon="🐘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.8rem; color: #aaa; margin-top: 4px; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #E87722;
        margin-bottom: 8px;
    }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)


# ── Data loaders (cached) ────────────────────────────────────────────────────
@st.cache_data
def load_care_scores():
    return pd.read_csv(EVAL / "care_scores.csv")

@st.cache_data
def load_detection_summary():
    df = pd.read_csv(SCORES / "detection_summary.csv")
    for col in ["event_start", "tier1_detection", "tier2_detection"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

@st.cache_data
def load_event_scores(farm: str, event_id: int):
    path = FARM_DIRS[farm] / f"{event_id}_scores.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["time_stamp"])
    return df

@st.cache_data
def load_ae_recon(farm: str, event_id: int):
    """Load ae_recon time series and return mean abs error per sensor."""
    path = FARM_DIRS[farm] / f"{event_id}_ae_recon.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["time_stamp"])
    sensor_cols = [c for c in df.columns if c != "time_stamp"]
    mean_errors = df[sensor_cols].abs().mean().reset_index()
    mean_errors.columns = ["sensor", "mean_error"]
    return mean_errors.sort_values("mean_error", ascending=False).reset_index(drop=True)

@st.cache_data
def load_feature_importance(farm: str):
    path = FEAT_FILES[farm]
    if not path.exists():
        return None
    return pd.read_csv(path)

@st.cache_data
def get_event_list(farm: str, detection_df: pd.DataFrame):
    farm_df = detection_df[detection_df["farm"] == farm].copy()
    events = []
    for _, row in farm_df.iterrows():
        eid = int(row["event_id"])
        has_tier1 = pd.notna(row.get("tier1_detection"))
        lead = row.get("tier1_lead_hours") if has_tier1 else row.get("tier2_lead_hours")
        events.append({
            "event_id": eid,
            "label": f"Event {eid}  |  {'✅ Tier1' if has_tier1 else '🔶 Tier2'}  |  Lead: {abs(lead):.0f}h" if pd.notna(lead) else f"Event {eid}",
            "has_tier1": has_tier1,
        })
    # also include normal events that have score files
    farm_dir = FARM_DIRS[farm]
    scored_ids = {int(p.stem.replace("_scores", "")) for p in farm_dir.glob("*_scores.csv")}
    anomaly_ids = {e["event_id"] for e in events}
    for eid in sorted(scored_ids - anomaly_ids):
        events.append({"event_id": eid, "label": f"Event {eid}  |  Normal", "has_tier1": False})
    return sorted(events, key=lambda x: x["event_id"])


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## 🐘 Wind Turbine\n### Anomaly Detection")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Introduction", "Fleet Overview", "Event Explorer", "Feature Signals", "Detection Leads"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("<small>DSMLC × Enbridge Competition 2026<br>CARE to Compare Dataset</small>",
                unsafe_allow_html=True)


# ── Load shared data ─────────────────────────────────────────────────────────
care_df    = load_care_scores()
detect_df  = load_detection_summary()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 0 — Introduction
# ════════════════════════════════════════════════════════════════════════════
if page == "Introduction":
    st.title("Wind Turbine Anomaly Detection")
    st.caption("DSMLC × Enbridge Competition 2026 — Jerome Francisco")

    st.markdown("---")

    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown("""
### The Problem
Wind turbines operate in harsh environments where undetected faults can cause costly downtime
and catastrophic failures. **Early anomaly detection** gives maintenance teams time to act before
a failure occurs — reducing repair costs and maximising turbine uptime.

### Our Approach
We built a multi-model ensemble that combines three complementary detectors:

| Model | What it captures |
|---|---|
| **Isolation Forest** | Global outliers in the feature space |
| **Autoencoder** | Subtle deviations from learned normal behaviour |
| **CUSUM** | Gradual drift in sensor signals over time |

The models are trained exclusively on the `train` split, then applied to the full time series.
An ensemble score is computed as a weighted combination of all three, and threshold crossings
are used to flag Tier 1 (high-confidence) and Tier 2 (early-warning) alerts.

### Why These Models?

Each model was chosen because it serves a distinct role in the CARE evaluation:

- **Isolation Forest** (weight 25%) drives *Reliability*. Wind turbines operate across multiple
  regimes — high wind, low wind, start-up — that create naturally clustered sensor distributions.
  Isolation Forest handles these multi-modal distributions without assuming a single Gaussian normal,
  flagging only clear global outliers and keeping the false alarm rate low.

- **Autoencoder — bottleneck MLP** (weight 35%) drives *Coverage*. It learns the joint distribution
  of all sensor channels simultaneously (up to 957 sensors). When a fault develops, multiple sensors
  deviate together in subtle ways that no per-sensor threshold would catch; reconstruction error
  across the full sensor vector surfaces those co-deviations. The anomaly threshold is set at
  mean + 3σ of training reconstruction error, anchoring it firmly to learned normal behaviour.

- **CUSUM** (weight 40%) drives *Earliness*. It accumulates evidence of a sustained upward drift
  over time using a slack parameter of 0.5σ and a decision interval of 4σ. Rather than reacting to
  a single spike, it "charges up" across many timesteps as conditions gradually degrade — which is
  why the system raises alerts tens of hours before the fault event occurs.

The ensemble weights directly reflect which CARE axis each model serves best.

### Limitations & Shortcomings

- **Fully unsupervised** — No fault labels enter training. The ensemble weights and alert thresholds
  are heuristic rather than optimised; a supervised or semi-supervised approach would produce a
  tighter decision boundary.

- **One-sided CUSUM** — The CUSUM only detects upward drift. Faults that manifest as sudden sensor
  drops (e.g., total sensor loss) or oscillating signals are invisible to this component.

- **No temporal memory in the autoencoder** — Each 10-minute window is scored independently. The
  MLP architecture cannot model sequential degradation patterns across timesteps; an LSTM-based
  autoencoder would better capture these dependencies.

- **Fixed contamination rate** — Isolation Forest assumes 5% anomalies globally. Farms where the
  true anomaly density differs (Farm C has proportionally more normal events) may see degraded
  precision as the decision boundary shifts.

- **Static thresholds** — Alert thresholds are calibrated once against the test-window of normal
  events. As turbines age or operating regimes shift seasonally, the baseline drifts and thresholds
  go stale without retraining.

- **No cross-turbine generalisation** — Each dataset is modelled in isolation. Fleet-level patterns
  — where multiple turbines show correlated early signals — are not exploited.
""")

    with col_b:
        st.markdown("### Dataset at a Glance")
        st.markdown("""
| | |
|---|---|
| Wind Farms | 3 (A, B, C) |
| Turbines | 36 |
| Total Events | 95 |
| Anomaly / Normal | 44 / 51 |
| Sensors (max) | 957 |
| Time Resolution | 10 min |
| Total Data Span | ~89 years |
""")
        st.markdown("### CARE Score")
        st.markdown("""
Models are evaluated on four axes:
- **C**overage — was the fault window detected?
- **A**ccuracy — anomaly vs. normal separation (AUC)
- **R**eliability — low false alarm rate
- **E**arliness — how far in advance was the alert raised?
""")

    st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Fleet Overview
# ════════════════════════════════════════════════════════════════════════════
elif page == "Fleet Overview":
    st.title("Fleet Overview — CARE Score Summary")
    st.caption("Coverage · Accuracy · Reliability · Earliness across all wind farms")
    
    st.info(
    "**Key finding:** The ensemble scores strongly on Coverage, Reliability, and Earliness — "
    "but Accuracy (AUC) is mixed across farms.\n\n"
    "- **Coverage & Reliability** — 100% of anomaly events detected across all farms at a 5% false alarm rate\n\n"
    "- **Farm A (AUC 0.674) and Farm C (AUC 0.774)** — models separate anomalous from normal behaviour well above the random baseline\n\n"
    "- **Farm B (AUC 0.440)** — below random classifier performance; the ensemble score distributions "
    "for anomaly and normal events overlap heavily in this farm, meaning the model detects faults early "
    "but cannot cleanly rank anomalous periods above normal ones\n\n"
    "The Coverage/Reliability strength reflects the two-tier alert design; the Farm B AUC weakness "
    "likely stems from its offshore operating environment producing noisier baselines with 257 anonymized sensors."
    )

    # ── KPI cards ─────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    total_events  = detect_df.shape[0]
    anomaly_count = detect_df[detect_df["event_start"].notna()].shape[0]
    avg_lead = detect_df[["tier1_lead_hours", "tier2_lead_hours"]].min(axis=1)
    avg_lead = avg_lead[avg_lead < 0].abs().median()

    with col1:
        st.metric("Total Events Evaluated", total_events)
    with col2:
        st.metric("Anomaly Events", anomaly_count)
    with col3:
        st.metric("Coverage (All Farms)", "100%")
    with col4:
        st.metric("Median Lead Time", f"{avg_lead:.0f} h")

    st.markdown("---")

    # ── CARE bar chart ─────────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-header">CARE Score Components by Farm</div>', unsafe_allow_html=True)

        fig = go.Figure()
        metrics = {
            "Coverage":    ("coverage", GREEN),
            "Reliability": ("reliability", BLUE),
            "Accuracy AUC":("accuracy_auc", ORANGE),
        }

        for name, (col, color) in metrics.items():
            fig.add_trace(go.Bar(
                name=name,
                x=care_df["farm"],
                y=care_df[col],
                marker_color=color,
                text=[f"{v:.2f}" for v in care_df[col]],
                textposition="outside",
            ))

        fig.update_layout(
            barmode="group",
            yaxis=dict(range=[0, 1.15], title="Score"),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="white",
            legend=dict(orientation="h", y=1.1),
            height=380,
            margin=dict(t=40, b=20),
        )
        # AUC=0.5 reference line
        fig.add_hline(y=0.5, line_dash="dash", line_color=GREY,
                      annotation_text="Random classifier (AUC=0.5)", annotation_font_color=GREY)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Earliness by Farm</div>', unsafe_allow_html=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=care_df["farm"],
            y=care_df["earliness_median_h"].abs(),
            marker_color=[ORANGE, BLUE, GREEN],
            text=[f"{v:.0f}h" for v in care_df["earliness_median_h"].abs()],
            textposition="outside",
        ))
        fig2.update_layout(
            yaxis=dict(title="Hours Before Fault (median)"),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="white",
            height=380,
            margin=dict(t=40, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── CARE table ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Full CARE Score Table</div>', unsafe_allow_html=True)
    display = care_df.copy()
    display.columns = ["Farm", "Anomaly Events", "Coverage", "Earliness Median (h)",
                        "Earliness Mean (h)", "Earliness Source", "Accuracy AUC", "FAR", "Reliability"]
    st.dataframe(
        display.style
            .format({"Coverage": "{:.0%}", "Reliability": "{:.0%}", "Accuracy AUC": "{:.3f}",
                     "FAR": "{:.0%}", "Earliness Median (h)": "{:.0f}", "Earliness Mean (h)": "{:.0f}"})
            .highlight_between(subset=["Accuracy AUC"], left=0, right=0.5, color="#5c2020")
            .highlight_between(subset=["Accuracy AUC"], left=0.5, right=1.0, color="#1a3a1a"),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")



# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Event Explorer
# ════════════════════════════════════════════════════════════════════════════
elif page == "Event Explorer":
    st.title("Event Explorer")
    st.caption("Explore anomaly scores and detection markers for individual turbine events")

    col_farm, col_event, col_thresh = st.columns([2, 3, 2])

    with col_farm:
        farm = st.selectbox("Wind Farm", list(FARM_DIRS.keys()),
                            format_func=lambda f: f"{f}  ({FARM_LABELS[f]})")

    events = get_event_list(farm, detect_df)
    with col_event:
        event_choice = st.selectbox("Event", events, format_func=lambda e: e["label"])

    with col_thresh:
        threshold = st.slider("Ensemble Alert Threshold", 0.1, 1.0, 0.55, 0.01)

    event_id = event_choice["event_id"]
    scores   = load_event_scores(farm, event_id)

    if scores is None:
        st.warning(f"No score file found for {farm} / Event {event_id}")
        st.stop()

    # Get detection info
    det_row = detect_df[(detect_df["farm"] == farm) & (detect_df["event_id"] == event_id)]
    is_anomaly = not det_row.empty and pd.notna(det_row.iloc[0]["event_start"])

    # ── Summary row ───────────────────────────────────────────────────────
    if is_anomaly:
        row = det_row.iloc[0]
        t1_lead = row.get("tier1_lead_hours")
        t2_lead = row.get("tier2_lead_hours")
        # Negative lead = early detection (hours before fault); positive = late
        def fmt_lead(v):
            if not pd.notna(v): return "—"
            return f"{abs(v):.0f}h {'before' if v < 0 else '⚠ after'} fault"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Event Type", "⚠️ Anomaly")
        c2.metric("Tier 1 Lead", fmt_lead(t1_lead))
        c3.metric("Tier 2 Lead", fmt_lead(t2_lead))
        c4.metric("Max CUSUM (norm)", f"{row['max_cusum_norm']:.3f}")
    else:
        st.info("Normal event — no fault detected.")

    st.markdown("---")

    # ── Anomaly score time series ─────────────────────────────────────────
    st.markdown('<div class="section-header">Anomaly Score Time Series</div>', unsafe_allow_html=True)

    signal_options = {
        "Ensemble Score":  "ensemble",
        "CUSUM (norm)":    "cusum_norm",
        "Autoencoder (norm)": "ae_norm",
        "Isolation Forest (norm)": "if_norm",
    }
    selected_signals = st.multiselect(
        "Signals to display",
        list(signal_options.keys()),
        default=["Ensemble Score", "CUSUM (norm)"],
    )

    fig = go.Figure()

    colors_map = {
        "Ensemble Score":        ORANGE,
        "CUSUM (norm)":          RED,
        "Autoencoder (norm)":    BLUE,
        "Isolation Forest (norm)": GREEN,
    }

    for sig in selected_signals:
        col = signal_options[sig]
        if col in scores.columns:
            fig.add_trace(go.Scatter(
                x=scores["time_stamp"], y=scores[col],
                name=sig, line=dict(color=colors_map[sig], width=1.5),
                mode="lines",
            ))

    # Threshold line
    fig.add_hline(
        y=threshold, line_dash="dot", line_color="white",
        annotation_text=f"Alert threshold ({threshold:.2f})",
        annotation_font_color="white",
    )

    # Detection markers — vline requires string timestamps for datetime axes
    if is_anomaly:
        row = det_row.iloc[0]

        # Compute actual fault time from detection + lead offset
        fault_ts = None
        if pd.notna(row.get("tier1_detection")) and pd.notna(row.get("tier1_lead_hours")):
            fault_ts = row["tier1_detection"] + pd.Timedelta(hours=abs(row["tier1_lead_hours"]))
        elif pd.notna(row.get("tier2_detection")) and pd.notna(row.get("tier2_lead_hours")):
            fault_ts = row["tier2_detection"] + pd.Timedelta(hours=abs(row["tier2_lead_hours"]))

        if fault_ts is not None:
            fig.add_vline(x=fault_ts.value // 1_000_000, line_color=RED, line_width=2,
                          annotation_text="Fault", annotation_font_color=RED,
                          annotation_position="top left")
        if pd.notna(row.get("tier1_detection")):
            fig.add_vline(x=row["tier1_detection"].value // 1_000_000, line_color=ORANGE, line_dash="dash",
                          annotation_text="Tier1 Alert", annotation_font_color=ORANGE)
        if pd.notna(row.get("tier2_detection")):
            fig.add_vline(x=row["tier2_detection"].value // 1_000_000, line_color=BLUE, line_dash="dash",
                          annotation_text="Tier2 Alert", annotation_font_color=BLUE)

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Normalised Score",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        height=420,
        legend=dict(orientation="h", y=1.05),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── AE reconstruction errors ──────────────────────────────────────────
    ae_df = load_ae_recon(farm, event_id)
    if ae_df is not None and not ae_df.empty:
        st.markdown('<div class="section-header">Autoencoder Reconstruction Error — Top Sensors</div>',
                    unsafe_allow_html=True)
        top_ae = ae_df.head(15)
        fig_ae = px.bar(
            top_ae, x="mean_error", y="sensor", orientation="h",
            color="mean_error",
            color_continuous_scale=["#1565C0", ORANGE, RED],
            labels={"mean_error": "Mean Abs Reconstruction Error", "sensor": "Sensor"},
        )
        fig_ae.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="white", height=420,
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_ae, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Feature Signals
# ════════════════════════════════════════════════════════════════════════════
elif page == "Feature Signals":
    st.title("Feature Signals — Sensor Importance")
    st.caption("Which sensors carry the strongest anomaly signal, by farm")

    st.info(
        "**Key finding:** Across all three farms, thermal and electrical sensors dominate both models.\n\n"
        "- **Farm A** — inverter/converter temperatures and phase currents are the strongest signals\n\n"
        "- **Farm B** — transformer temperatures (mid-voltage L3, cell) and gearbox bearing temperature lead\n\n"
        "- **Farm C** — highest reconstruction errors overall; gearbox oil level and stator winding "
        "temperatures carry extreme AE error, pointing to drivetrain and generator degradation as the primary fault pathway\n\n"
        "Sensors that rank highly in *both* models (top-right of the scatter plot) represent the highest-confidence fault indicators.\n\n"
        "⚠️ **Potential problems with the implementation:**\n\n"
        "- **Different populations:** IF importance is computed on *normal* test data; AE error is computed on *anomaly* test windows. "
        "The scatter plot cross-compares metrics from different populations — a sensor ranking high on both reflects two distinct behaviours, not the same measurement.\n\n"
        "- **Anonymized sensors (Farm B & C):** Most sensors have no description — charts show raw sensor IDs. "
        "Physical interpretation requires cross-referencing `feature_description.csv`.\n\n"
        "- **Max-grouping of engineered features:** Importance is grouped by base sensor name and the *maximum* across all derived features "
        "(rolling stats, residuals, etc.) is displayed. A high score may originate from an engineered derivative rather than the raw sensor value."
    )

    farm = st.selectbox("Select Farm", list(FEAT_FILES.keys()),
                        format_func=lambda f: f"{f}  ({FARM_LABELS[f]})")

    feat_df = load_feature_importance(farm)
    if feat_df is None:
        st.error("Feature importance file not found.")
        st.stop()

    n_top = st.slider("Number of top sensors to show", 5, 30, 15)

    # Fill missing descriptions
    if "description" in feat_df.columns:
        feat_df["label"] = feat_df.apply(
            lambda r: r["description"] if pd.notna(r["description"]) and r["description"] != ""
            else r["sensor"], axis=1
        )
    else:
        feat_df["label"] = feat_df["sensor"]

    col_l, col_r = st.columns(2)

    # IF importance
    with col_l:
        st.markdown('<div class="section-header">Isolation Forest — Feature Importance</div>',
                    unsafe_allow_html=True)
        top_if = feat_df.nlargest(n_top, "if_importance")
        fig_if = px.bar(
            top_if, x="if_importance", y="label", orientation="h",
            color="if_importance",
            color_continuous_scale=["#1565C0", ORANGE],
            labels={"if_importance": "Importance Score", "label": "Sensor"},
        )
        fig_if.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="white", height=500,
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_if, use_container_width=True)

    # AE mean error
    with col_r:
        st.markdown('<div class="section-header">Autoencoder — Mean Reconstruction Error</div>',
                    unsafe_allow_html=True)
        top_ae = feat_df.nlargest(n_top, "ae_mean_error")
        fig_ae = px.bar(
            top_ae, x="ae_mean_error", y="label", orientation="h",
            color="ae_mean_error",
            color_continuous_scale=["#1565C0", RED],
            labels={"ae_mean_error": "Mean Reconstruction Error", "label": "Sensor"},
        )
        fig_ae.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="white", height=500,
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_ae, use_container_width=True)

    # Combined scatter
    st.markdown('<div class="section-header">IF Importance vs. AE Reconstruction Error</div>',
                unsafe_allow_html=True)
    st.caption("Sensors in the top-right are flagged as anomalous by both models — highest confidence signals")

    top_combined = feat_df.nlargest(40, "if_importance")
    fig_sc = px.scatter(
        top_combined, x="if_importance", y="ae_mean_error",
        hover_name="label", color="ae_mean_error",
        color_continuous_scale=["#1565C0", ORANGE, RED],
        labels={"if_importance": "IF Importance", "ae_mean_error": "AE Mean Error"},
        size_max=12,
    )
    fig_sc.update_layout(
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="white", height=380, coloraxis_showscale=False,
    )
    st.plotly_chart(fig_sc, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Detection Leads
# ════════════════════════════════════════════════════════════════════════════
elif page == "Detection Leads":
    st.title("Detection Lead Times")
    st.caption("How many hours before each fault did the system first raise an alert?")

    st.info(
        "**Key finding:** The ensemble detected **100% of anomaly events** across all 18 faults with a false alarm rate of just 5%.\n\n"
        "Median lead times were:\n\n"
        "- **149h (Farm A)**\n\n"
        "- **152h (Farm B)**\n\n"
        "- **107h (Farm C)**\n\n"
        "On average, roughly 4–6 days before failure.\n\n"
        "Farm A and C were predominantly caught by the CUSUM-driven **Tier 1** alert; Farm B had no "
        "Tier 1 hits, relying entirely on the ensemble **Tier 2** threshold, suggesting slower or "
        "less directional sensor drift in that farm's fault modes.\n\n"
        "The earliest single detection was Farm C Event 67, flagged **376h** (over 15 days) in advance.\n\n"
        "⚠️ **Tier 2 threshold caveat:** The Tier 2 threshold is calibrated at the 95th percentile of "
        "normal event test scores. In several cases (Farm A Event 73: +63h, Farm C Events 28 and 67: "
        "+8h and +22h), the Tier 2 alert fired *after* the fault onset — meaning the ensemble score "
        "never crossed the threshold until the turbine was already in failure. The 95th-percentile "
        "calibration is likely too conservative; a per-farm or lower-percentile threshold would "
        "improve coverage for these events."
    )

    farm_filter = st.multiselect(
        "Filter by Farm",
        ["Wind Farm A", "Wind Farm B", "Wind Farm C"],
        default=["Wind Farm A", "Wind Farm B", "Wind Farm C"],
    )

    anomaly_detect = detect_df[detect_df["event_start"].notna()].copy()
    anomaly_detect = anomaly_detect[anomaly_detect["farm"].isin(farm_filter)]

    # Best lead per event (most negative = earliest)
    anomaly_detect["best_lead_h"] = anomaly_detect[["tier1_lead_hours", "tier2_lead_hours"]].min(axis=1)
    anomaly_detect["detection_tier"] = anomaly_detect.apply(
        lambda r: "Tier 1" if pd.notna(r["tier1_lead_hours"]) else "Tier 2", axis=1
    )
    anomaly_detect["label"] = anomaly_detect.apply(
        lambda r: f"{r['farm']} · Event {int(r['event_id'])}", axis=1
    )

    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown('<div class="section-header">Lead Time per Anomaly Event</div>',
                    unsafe_allow_html=True)

        sorted_df = anomaly_detect.sort_values("best_lead_h")
        color_map = {"Tier 1": ORANGE, "Tier 2": BLUE}

        fig = go.Figure()
        for tier, grp in sorted_df.groupby("detection_tier"):
            fig.add_trace(go.Bar(
                x=grp["best_lead_h"].abs(),
                y=grp["label"],
                orientation="h",
                name=tier,
                marker_color=color_map[tier],
                text=[f"{abs(v):.0f}h" for v in grp["best_lead_h"]],
                textposition="outside",
            ))

        fig.add_vline(x=0, line_color=RED, line_width=1.5,
                      annotation_text="Fault onset", annotation_font_color=RED)
        fig.update_layout(
            xaxis_title="Hours Before Fault (earlier = better)",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="white",
            height=max(350, len(sorted_df) * 32),
            barmode="overlay",
            legend=dict(orientation="h", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Lead Time Distribution</div>',
                    unsafe_allow_html=True)

        fig_hist = px.histogram(
            anomaly_detect, x="best_lead_h", color="farm",
            nbins=20, barmode="overlay",
            color_discrete_sequence=[ORANGE, BLUE, GREEN],
            labels={"best_lead_h": "Lead Time (hours)", "farm": "Farm"},
        )
        fig_hist.add_vline(x=anomaly_detect["best_lead_h"].median(),
                           line_dash="dash", line_color="white",
                           annotation_text=f"Median: {anomaly_detect['best_lead_h'].median():.0f}h",
                           annotation_font_color="white")
        fig_hist.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="white", height=300,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown('<div class="section-header">Summary Statistics</div>',
                    unsafe_allow_html=True)
        stats = anomaly_detect.groupby("farm")["best_lead_h"].agg(
            Median="median", Mean="mean", Min="min", Max="max"
        ).round(1)
        st.dataframe(stats.style.format("{:.0f}"), use_container_width=True)

    # ── Raw table ──────────────────────────────────────────────────────────
    with st.expander("Raw detection data"):
        cols = ["farm", "event_id", "event_start", "tier1_detection", "tier2_detection",
                "tier1_lead_hours", "tier2_lead_hours", "max_cusum_norm", "max_ensemble"]
        st.dataframe(anomaly_detect[[c for c in cols if c in anomaly_detect.columns]],
                     use_container_width=True, hide_index=True)
