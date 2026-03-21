"""
Feature engineering for thermal & electrical anomaly detection.
Produces per-event feature matrices saved to outputs/features/.

Feature groups computed:
  1. Ambient-detrended residuals  (OLS: temp ~ ambient, fit on normal train rows)
  2. Rolling statistics on residuals  (windows: 6, 12, 24, 72 steps)
  3. Cross-sensor imbalance features  (max - min within sensor groups)
  4. Load-normalised temperatures  (temp / power)
  5. Rate-of-change features  (delta at 1, 6, 144 steps)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression

# ---------- Paths ----------
ROOT         = Path(__file__).resolve().parent.parent
DATA_RAW     = ROOT / "data" / "raw"
DATA_PROC    = ROOT / "data" / "processed"
EXTRACTED    = DATA_RAW / "CARE_To_Compare"
FEATURES_DIR = ROOT / "outputs" / "features"

FARMS          = ["Wind Farm A", "Wind Farm B", "Wind Farm C"]
NORMAL_STATUS  = {0, 2}
ROLLING_WINDOWS = [6, 12, 24, 72]   # steps (10-min each)
ROC_LAGS       = [1, 6, 144]        # 10 min, 1 h, 24 h
EPS            = 1e-3               # avoid division by zero

# Roles for which extended features (rolling, roc) are computed.
# Limits feature explosion on high-dimensional farms.
KEY_ROLES = [
    "ambient",
    "gearbox_bearing", "gearbox_oil",
    "gen_bearing",
    "stator_winding",
    "igbt_temp",
    "transformer_temp",
    "hydraulic_oil",
    "nacelle_temp",
]

# ---------- Semantic role detection ----------
# Each entry: (role_name, list_of_rule_word_groups)
# A sensor matches a role if ALL words in ANY single rule_word_group appear in its description.
ROLE_RULES = [
    ("ambient",          [["ambient"], ["outside temperature"]]),
    ("gearbox_bearing",  [["gearbox", "bearing"]]),
    ("gen_bearing",      [["generator", "bearing"]]),
    ("stator_winding",   [["stator", "winding"]]),
    ("transformer_temp", [["transformer", "temperature"], ["transformer", "temp"],
                          ["temperature", "transformer"]]),
    ("igbt_temp",        [["igbt"]]),
    ("gearbox_oil",      [["gearbox", "oil"]]),
    ("hydraulic_oil",    [["hydraulic", "oil"]]),
    ("nacelle_temp",     [["nacelle"]]),
    ("current_phase",    [["current", "phase"], ["rms current"], ["averaged current"]]),
    ("current",          [["current"]]),
    ("voltage_phase",    [["voltage", "phase"], ["rms voltage"], ["averaged voltage"]]),
    ("voltage",          [["voltage"]]),
    ("active_power",     [["grid power"], ["active power", "grid"]]),
    ("apparent_power",   [["apparent power"]]),
    ("temperature",      [["temperature"], ["temp"]]),   # catch-all: all thermal sensors
]


def _match_roles(desc: str) -> list[str]:
    """Return all role names whose rule matches the description string."""
    desc_lower = desc.lower()
    matched = []
    for role, rule_groups in ROLE_RULES:
        for words in rule_groups:
            if all(w in desc_lower for w in words):
                matched.append(role)
                break
    return matched


def get_sensor_roles(sensor_list_df: pd.DataFrame, farm: str, available_cols: set) -> dict:
    """
    Return {role: [col, ...]} mapping available columns to semantic roles.
    Tries <name>_avg then <name> as column name.
    """
    farm_df = sensor_list_df[sensor_list_df["farm"] == farm]
    roles: dict[str, list] = {}

    for _, row in farm_df.iterrows():
        name = str(row["sensor_name"]).strip()
        desc = str(row.get("description", "")).strip()

        col = None
        for sfx in ["_avg", "_average", "_mean", ""]:
            if f"{name}{sfx}" in available_cols:
                col = f"{name}{sfx}"
                break
        if col is None:
            continue

        for role in _match_roles(desc):
            roles.setdefault(role, [])
            if col not in roles[role]:
                roles[role].append(col)

    return roles


# ---------- Individual feature transformations ----------

def ambient_detrend(df: pd.DataFrame, temp_cols: list, ambient_col: str,
                    normal_train_mask: np.ndarray) -> pd.DataFrame:
    """OLS: temp ~ ambient fitted on normal training rows. Adds <col>_res columns."""
    amb = df[ambient_col].values.reshape(-1, 1)
    for col in temp_cols:
        if col not in df.columns or col == ambient_col:
            continue
        res_col = f"{col}_res"
        valid = normal_train_mask & ~np.isnan(df[col].values) & ~np.isnan(df[ambient_col].values)
        if valid.sum() < 20:
            df[res_col] = df[col]
            continue
        reg = LinearRegression().fit(amb[valid], df[col].values[valid])
        # Predict only on non-NaN rows to avoid sklearn validation warnings
        predicted = np.full(len(df), np.nan)
        non_nan = ~np.isnan(df[ambient_col].values)
        predicted[non_nan] = reg.predict(amb[non_nan]).flatten()
        df[res_col] = df[col].values - predicted
    return df


def add_rolling_features(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Rolling mean, std, and EWMA for each column."""
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col]
        for w in ROLLING_WINDOWS:
            df[f"{col}_rm{w}"] = s.rolling(w, min_periods=1).mean()
            df[f"{col}_rs{w}"] = s.rolling(w, min_periods=1).std()
        df[f"{col}_ewm24"] = s.ewm(span=24, adjust=False, ignore_na=True).mean()
        df[f"{col}_ewm72"] = s.ewm(span=72, adjust=False, ignore_na=True).mean()
    return df


def add_imbalance_features(df: pd.DataFrame, roles: dict) -> pd.DataFrame:
    """Max-min spread within multi-sensor groups."""
    imbalance_roles = [
        "stator_winding", "transformer_temp", "gen_bearing",
        "gearbox_bearing", "current_phase", "voltage_phase", "igbt_temp",
    ]
    for role in imbalance_roles:
        cols = [c for c in roles.get(role, []) if c in df.columns]
        if len(cols) >= 2:
            df[f"imb_{role}"] = df[cols].max(axis=1) - df[cols].min(axis=1)
    return df


def add_load_normalized(df: pd.DataFrame, temp_cols: list,
                        power_col: str | None) -> pd.DataFrame:
    """temp / (power + eps) for each thermal temperature column."""
    if power_col is None or power_col not in df.columns:
        return df
    power = df[power_col].clip(lower=0)
    for col in temp_cols:
        if col in df.columns:
            df[f"{col}_perkw"] = df[col] / (power + EPS)
    return df


def add_rate_of_change(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Differenced features at multiple lags."""
    for col in cols:
        if col not in df.columns:
            continue
        for lag in ROC_LAGS:
            df[f"{col}_d{lag}"] = df[col].diff(lag)
    return df


# ---------- Master feature computation ----------

def compute_features(df: pd.DataFrame, roles: dict) -> pd.DataFrame:
    """
    Apply all feature transformations to df.
    Operates on a copy; returns the feature-enriched DataFrame with
    only metadata + thermal raw + computed feature columns.
    """
    df = df.copy()
    available = set(df.columns)
    original_cols = set(df.columns)

    # --- masks ---
    if "status_type_id" in df.columns:
        normal_mask = df["status_type_id"].isin(NORMAL_STATUS).values
    else:
        normal_mask = np.ones(len(df), dtype=bool)

    if "train_test" in df.columns:
        train_mask = df["train_test"].str.lower().str.strip().isin(
            ["train", "0"]).values
    else:
        train_mask = np.ones(len(df), dtype=bool)

    normal_train_mask = normal_mask & train_mask

    # --- 1. Ambient detrending ---
    temp_cols    = [c for c in roles.get("temperature", []) if c in available]
    ambient_cols = [c for c in roles.get("ambient",     []) if c in available]
    ambient_col  = ambient_cols[0] if ambient_cols else None

    if ambient_col and temp_cols:
        df = ambient_detrend(df, temp_cols, ambient_col, normal_train_mask)

    residual_cols = [f"{c}_res" for c in temp_cols
                     if f"{c}_res" in df.columns]

    # --- 2. Rolling features (on residuals of key roles only) ---
    key_raw  = list(dict.fromkeys(
        c for role in KEY_ROLES for c in roles.get(role, []) if c in available
    ))
    key_res  = [f"{c}_res" for c in key_raw if f"{c}_res" in df.columns]

    df = add_rolling_features(df, key_res)   # residuals preferred
    df = add_rolling_features(df, key_raw)   # also raw temps for context

    # --- 3. Imbalance features ---
    df = add_imbalance_features(df, roles)

    # --- 4. Load-normalised temperatures ---
    power_col = None
    for pk in ("active_power", "apparent_power"):
        cands = [c for c in roles.get(pk, []) if c in available]
        if cands:
            power_col = cands[0]
            break

    thermal_keys = [
        "gearbox_bearing", "gearbox_oil", "gen_bearing",
        "stator_winding", "transformer_temp", "hydraulic_oil",
    ]
    thermal_cols = list(dict.fromkeys(
        c for k in thermal_keys for c in roles.get(k, []) if c in available
    ))
    df = add_load_normalized(df, thermal_cols, power_col)

    # --- 5. Rate of change on key columns + their residuals ---
    roc_targets = list(dict.fromkeys(key_raw + key_res))
    df = add_rate_of_change(df, roc_targets)

    # --- Select output columns ---
    meta_cols = [c for c in
                 ["time_stamp", "asset_id", "id", "train_test", "status_type_id"]
                 if c in df.columns]
    thermal_raw = list(dict.fromkeys(
        c for cols in roles.values() for c in cols if c in df.columns
    ))
    computed = [c for c in df.columns if c not in original_cols]
    keep = list(dict.fromkeys(meta_cols + thermal_raw + computed))
    return df[keep]


# ---------- Loaders ----------

def load_sensor_list() -> pd.DataFrame:
    return pd.read_csv(DATA_PROC / "thermal_sensor_list.csv")


def load_event_info() -> dict[str, pd.DataFrame]:
    result = {}
    for farm in FARMS:
        path = EXTRACTED / farm / "event_info.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, sep=";")
        df.columns = df.columns.str.strip().str.lower()
        result[farm] = df
    return result


def _event_label(event_id, event_info_df: pd.DataFrame) -> str:
    id_col  = next((c for c in event_info_df.columns
                    if c in ("event_id", "id", "dataset_id")), None)
    lbl_col = next((c for c in event_info_df.columns if "label" in c), None)
    if id_col is None or lbl_col is None:
        return "unknown"
    row = event_info_df[event_info_df[id_col] == event_id]
    return str(row[lbl_col].iloc[0]).lower() if not row.empty else "unknown"


# ---------- Main processing loop ----------

def process_all_events(sensor_list_df: pd.DataFrame,
                       event_infos: dict) -> pd.DataFrame:
    """Process every event CSV in every farm; save feature matrices."""
    summary_rows = []

    for farm in FARMS:
        farm_dir = EXTRACTED / farm / "datasets"
        if not farm_dir.exists():
            print(f"[SKIP] {farm}: datasets dir not found")
            continue

        out_dir = FEATURES_DIR / farm.replace(" ", "_")
        out_dir.mkdir(parents=True, exist_ok=True)

        event_info = event_infos.get(farm, pd.DataFrame())
        all_csvs   = sorted(farm_dir.glob("*.csv"))
        print(f"\n[{farm}]  {len(all_csvs)} event CSVs found")

        for csv_path in all_csvs:
            stem = csv_path.stem.replace("event_", "").strip()
            try:
                event_id = int(stem)
            except ValueError:
                event_id = stem

            label = (_event_label(event_id, event_info)
                     if not event_info.empty else "unknown")

            try:
                df = pd.read_csv(csv_path, sep=";",
                                 parse_dates=["time_stamp"],
                                 encoding="latin-1")
            except Exception as exc:
                print(f"  [WARN] Cannot read {csv_path.name}: {exc}")
                continue

            df.columns = df.columns.str.strip()
            available  = set(df.columns)

            roles = get_sensor_roles(sensor_list_df, farm, available)
            if not roles:
                print(f"  [WARN] event {event_id}: no sensor roles — skipping")
                continue

            try:
                df_feat = compute_features(df, roles)
            except Exception as exc:
                print(f"  [WARN] event {event_id}: feature computation failed — {exc}")
                continue

            out_path = out_dir / f"{event_id}.csv"
            df_feat.to_csv(out_path, index=False)

            n_train = (
                df_feat["train_test"].str.lower().str.strip()
                .isin(["train", "0"]).sum()
                if "train_test" in df_feat.columns else 0
            )
            n_test    = len(df_feat) - n_train
            n_feats   = len(df_feat.columns)
            print(f"  event {event_id:3d} ({label:8s}): "
                  f"{n_train:5d} train / {n_test:4d} test rows, "
                  f"{n_feats:4d} cols -> {out_path.name}")

            summary_rows.append({
                "farm":        farm,
                "event_id":    event_id,
                "label":       label,
                "n_train":     n_train,
                "n_test":      n_test,
                "n_features":  n_feats,
                "output_path": str(out_path),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = FEATURES_DIR / "feature_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[Summary] {len(summary_rows)} events -> {summary_path}")
    return summary_df


def main():
    print("=" * 70)
    print("  FEATURE ENGINEERING — Thermal & Electrical Anomaly Detection")
    print("=" * 70)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1] Loading sensor list ...")
    sensor_list_df = load_sensor_list()
    print(f"    {len(sensor_list_df)} sensor definitions ({sensor_list_df['farm'].nunique()} farms)")

    print("\n[2] Loading event info ...")
    event_infos = load_event_info()
    for farm, df in event_infos.items():
        n_anom = (df[next(c for c in df.columns if "label" in c)]
                  .str.lower().eq("anomaly").sum()
                  if any("label" in c for c in df.columns) else "?")
        print(f"    {farm}: {len(df)} events ({n_anom} anomaly)")

    print("\n[3] Processing events ...")
    summary = process_all_events(sensor_list_df, event_infos)

    print("\n" + "=" * 70)
    print(f"  Done. {len(summary)} events processed.")
    print(f"  Output: {FEATURES_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
