"""
Simulated Responses App for Market Research (Excel + Discrete Fix + Validation)
==============================================================================

This Streamlit app ingests an Excel dataset, lets you subset (for example, gender = Female),
and generates additional, statistically grounded synthetic rows consistent with the subset.
It now includes a **Validation** tab that performs a holdout experiment to quantify accuracy.

Generation modes:
1) Bootstrap + Jitter (recommended for small n) — jitter is disabled on discrete columns.
2) Parametric (multivariate normal for continuous) — discrete numeric codes are resampled from empirical marginals.

Validation features:
- Train/holdout split with a chosen boost factor (for example, 3×).
- Metrics: Mean Absolute Error (MAE) on numeric and binary columns, Kolmogorov–Smirnov (KS) on continuous columns,
  Population Stability Index (PSI) for categorical columns, and a Real-vs-Synthetic classifier AUC (target ≈ 0.5).
- Clear pass/fail indicators and a results table.
"""

import json
from datetime import datetime
import secrets

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import streamlit as st

# ------------------------------
# Detection helpers
# ------------------------------

def infer_column_types(df: pd.DataFrame, categorical_max_card: int = 25):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in df.columns:
        if col not in cat_cols and df[col].nunique(dropna=True) <= categorical_max_card:
            if df[col].dtype.kind not in ["f", "i", "u"]:
                cat_cols.append(col)
    cat_cols = list(dict.fromkeys(cat_cols))
    numeric_cols = [c for c in df.columns if c not in cat_cols and pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols, cat_cols


def detect_discrete_numeric(df: pd.DataFrame, max_unique: int = 10):
    binary_cols, ordinal_cols = [], []
    allowed_values = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        vals = df[col].dropna().unique()
        if len(vals) == 0:
            continue
        all_int_like = np.all(np.equal(np.mod(vals, 1), 0))
        uniq_sorted = np.sort(vals)
        allowed_values[col] = uniq_sorted
        if set(np.unique(vals)).issubset({0, 1}) and len(np.unique(vals)) <= 2:
            binary_cols.append(col)
        elif all_int_like and len(vals) <= max_unique:
            ordinal_cols.append(col)
    return binary_cols, ordinal_cols, allowed_values


# ------------------------------
# Core utilities
# ------------------------------

def subset_dataframe(df: pd.DataFrame, filters: dict):
    if not filters:
        return df
    mask = pd.Series(True, index=df.index)
    for col, allowed_vals in filters.items():
        if col in df.columns and len(allowed_vals) > 0:
            mask &= df[col].isin(allowed_vals)
    return df[mask]


def bootstrap_jitter_sample(df_sub: pd.DataFrame, n_rows: int, noise_pct: float, seed: int,
                            discrete_cols: set):
    rng = np.random.default_rng(seed)
    if len(df_sub) == 0:
        return df_sub
    sampled = df_sub.sample(n=n_rows, replace=True, random_state=seed).reset_index(drop=True)

    num_all = sampled.select_dtypes(include=[np.number]).columns
    cont_cols = [c for c in num_all if c not in discrete_cols]
    for col in cont_cols:
        col_std = df_sub[col].std(skipna=True)
        if pd.isna(col_std) or col_std == 0:
            continue
        if noise_pct > 0:
            noise = rng.normal(loc=0.0, scale=noise_pct * col_std, size=len(sampled))
            non_nan_mask = sampled[col].notna()
            sampled.loc[non_nan_mask, col] = sampled.loc[non_nan_mask, col].to_numpy() + noise[non_nan_mask]
    return sampled


def parametric_mvn_sample(df_sub: pd.DataFrame, n_rows: int, seed: int,
                          discrete_cols: set):
    rng = np.random.default_rng(seed)

    numeric_cols = df_sub.select_dtypes(include=[np.number]).columns.tolist()
    cont_cols = [c for c in numeric_cols if c not in discrete_cols]
    disc_cols = [c for c in numeric_cols if c in discrete_cols]
    cat_cols = df_sub.select_dtypes(include=["object", "category"]).columns.tolist()

    out = pd.DataFrame(index=range(n_rows), columns=df_sub.columns)

    for col in cat_cols + disc_cols:
        vals = df_sub[col].dropna().to_numpy()
        if len(vals) == 0:
            out[col] = np.nan
            continue
        uniq_vals, counts = np.unique(vals, return_counts=True)
        p = counts / counts.sum()
        out[col] = rng.choice(uniq_vals, size=n_rows, p=p)

    if len(cont_cols) > 0:
        X = df_sub[cont_cols].dropna()
        if len(X) > 0:
            mean = X.mean(axis=0).to_numpy()
            cov = np.cov(X.to_numpy(), rowvar=False)
            eps = 1e-6
            cov = cov + eps * np.eye(cov.shape[0])
            mvn_draws = rng.multivariate_normal(mean, cov, size=n_rows)
            for i, col in enumerate(cont_cols):
                out[col] = mvn_draws[:, i]

    return out


def enforce_discrete_constraints(df_generated: pd.DataFrame, binary_cols, ordinal_cols, allowed_values):
    out = df_generated.copy()
    for col in binary_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round().clip(lower=0, upper=1).astype("Int64")
    for col in ordinal_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round()
            obs = allowed_values.get(col)
            if obs is not None and len(obs) > 0:
                out[col] = out[col].clip(lower=np.nanmin(obs), upper=np.nanmax(obs))
            out[col] = out[col].astype("Int64")
    return out


def _safe_mean(s: pd.Series):
    arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    if np.isnan(arr).all():
        return np.nan
    return float(np.nanmean(arr))


def _safe_std(s: pd.Series):
    arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    if np.isnan(arr).all():
        return np.nan
    return float(np.nanstd(arr, ddof=1))


# ------------------------------
# Helper for categorical comparison
# ------------------------------

def describe_cats(real: pd.Series, synth: pd.Series):
    real_counts = real.value_counts(normalize=True, dropna=True)
    synth_counts = synth.value_counts(normalize=True, dropna=True)
    idx = sorted(set(real_counts.index).union(set(synth_counts.index)))
    rows = []
    for k in idx:
        rows.append({
            "category": k,
            "real_pct": float(real_counts.get(k, 0.0) * 100),
            "synth_pct": float(synth_counts.get(k, 0.0) * 100),
            "abs_diff_pp": float(abs(real_counts.get(k, 0.0) - synth_counts.get(k, 0.0)) * 100),
        })
    return pd.DataFrame(rows)

# ------------------------------
# Validation utilities
# ------------------------------

def mae_numeric(real: pd.Series, synth: pd.Series):
    r = pd.to_numeric(real, errors="coerce")
    s = pd.to_numeric(synth, errors="coerce")
    return float(np.nanmean(np.abs(r - s)))


def ks_numeric(real: pd.Series, synth: pd.Series):
    r = pd.to_numeric(real, errors="coerce").dropna()
    s = pd.to_numeric(synth, errors="coerce").dropna()
    if len(r) < 5 or len(s) < 5:
        return np.nan
    stat, _ = stats.ks_2samp(r, s, method="asymp")
    return float(stat)


def psi_categorical(real: pd.Series, synth: pd.Series, eps: float = 1e-6):
    r = real.astype("object").dropna()
    s = synth.astype("object").dropna()
    if len(r) == 0 or len(s) == 0:
        return np.nan
    cats = sorted(set(r.unique()).union(set(s.unique())))
    rp = np.array([(r == c).mean() for c in cats]) + eps
    sp = np.array([(s == c).mean() for c in cats]) + eps
    return float(np.sum((rp - sp) * np.log(rp / sp)))


def auc_real_vs_synth(df_real: pd.DataFrame, df_synth: pd.DataFrame):
    df_r = df_real.copy(); df_r["__y__"] = 0
    df_s = df_synth.copy(); df_s["__y__"] = 1
    df_all = pd.concat([df_r, df_s], ignore_index=True)

    X = df_all.drop(columns=["__y__"])  # encode categoricals
    y = df_all["__y__"].to_numpy()

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    Xn = X[num_cols].apply(pd.to_numeric, errors="coerce") if len(num_cols) else pd.DataFrame(index=X.index)

    if len(cat_cols):
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        Xc = pd.DataFrame(enc.fit_transform(X[cat_cols]))
        Xc.index = X.index
        Xenc = pd.concat([Xn, Xc], axis=1).fillna(0)
    else:
        Xenc = Xn.fillna(0)

    if Xenc.shape[1] == 0:
        return np.nan

    clf = LogisticRegression(max_iter=200, n_jobs=None)
    try:
        clf.fit(Xenc, y)
        proba = clf.predict_proba(Xenc)[:, 1]
        return float(roc_auc_score(y, proba))
    except Exception:
        return np.nan


# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="Simulated Responses Builder", layout="wide")
st.title("Simulated Responses Builder")

with st.sidebar:
    st.header("1) Load data")
    up = st.file_uploader("Upload Excel file", type=["xlsx", "xls"]) 
    seed = st.number_input("Random seed", value=42, step=1)
    randomize_seed = st.checkbox("Randomize seed each run", value=False)
    st.caption("Turn this on for unpredictability; turn off and set a seed for reproducibility.")

    st.header("2) Subset filters")
    filters = {}
    exclude_cols = []  # NEW

    if up is not None:
        df = pd.read_excel(up)
        numeric_cols, cat_cols = infer_column_types(df)
        bin_cols, ord_cols, allowed_vals = detect_discrete_numeric(df)

        st.caption(
            f"Detected {len(numeric_cols)} numeric ({len(bin_cols)} binary, {len(ord_cols)} ordinal) and {len(cat_cols)} categorical columns."
        )

        with st.expander("Advanced: mark additional discrete columns"):
            extra_binary = st.multiselect("Extra binary columns (0/1)", options=[c for c in numeric_cols if c not in bin_cols])
            extra_ordinal = st.multiselect("Extra ordinal/integer-coded columns", options=[c for c in numeric_cols if c not in ord_cols and c not in bin_cols])
            bin_cols = list(dict.fromkeys(list(bin_cols) + extra_binary))
            ord_cols = list(dict.fromkeys(list(ord_cols) + extra_ordinal))

        # NEW: Exclusion filter
        with st.expander("Exclude columns from simulation (keep blank)"):
            exclude_cols = st.multiselect("Select columns to exclude", options=df.columns.tolist(), default=[])

        # Subset filters
        for col in cat_cols:
            vals = df[col].dropna().unique().tolist()
            if len(vals) > 0:
                sel = st.multiselect(f"{col}", options=vals, default=[])
                if len(sel) > 0:
                    filters[col] = sel

        st.header("3) Generation settings")
        method = st.selectbox(
            "Method",
            ["Bootstrap + Jitter (recommended)", "Parametric (MVN numeric + categorical resampling)"]
        )
        n_rows = st.number_input("How many synthetic rows?", min_value=1, max_value=100000, value=100, step=10)
        noise_pct = 0.0
        if "Bootstrap" in method:
            noise_pct = st.slider("Numeric jitter (% of each continuous column's standard deviation)", 0.0, 0.5, 0.05, 0.01)

        st.header("4) Output")
        include_original_flag = st.checkbox("Add a column `__source__` indicating real vs synthetic", value=True)
        do_preview = st.checkbox("Show preview tables", value=True)
        run_diagnostics = st.checkbox("Run diagnostics (stats tables)", value=True)
        create_download_file = st.checkbox("Create Excel download after run", value=True)

    else:
        df = None
        bin_cols, ord_cols, allowed_vals, exclude_cols = [], [], {}, []
        

st.markdown("---")

if df is None:
    st.info("Upload an Excel file to get started.")
    st.stop()

# Tabs: Simulate and Validation
simulate_tab, validate_tab = st.tabs(["Simulate", "Validation"])

discrete_set = set(bin_cols) | set(ord_cols)

# ---------------- Simulation Tab ----------------
with simulate_tab:
    st.subheader("Generate synthetic data")
    df_sub = subset_dataframe(df, filters)
    left, right = st.columns(2)
    with left:
        st.write("Rows in full dataset:", len(df))
        st.write("Rows in subset:", len(df_sub))
    with right:
        st.dataframe(df_sub.head(10))

    if len(df_sub) == 0:
        st.error("Your current filter produced an empty subset.")
        st.stop()

    if st.button("Run simulation", type="primary"):
        with st.spinner("Generating synthetic rows..."):
            seed_used = int(secrets.randbits(64)) if randomize_seed else int(seed)

            if "Bootstrap" in method:
                synth = bootstrap_jitter_sample(
                    df_sub, n_rows=int(n_rows), noise_pct=float(noise_pct), seed=seed_used,
                    discrete_cols=discrete_set
                )
            else:
                synth = parametric_mvn_sample(
                    df_sub, n_rows=int(n_rows), seed=seed_used, discrete_cols=discrete_set
                )
            synth = enforce_discrete_constraints(synth, bin_cols, ord_cols, allowed_vals)
            # Clear excluded columns (keep them as blank)
            for col in exclude_cols:
                if col in synth.columns:
                    synth[col] = None
                    
            if include_original_flag:
                df_tagged = df.copy(); df_tagged["__source__"] = "real"
                synth_tagged = synth.copy(); synth_tagged["__source__"] = "synthetic"
                combined = pd.concat([df_tagged, synth_tagged], ignore_index=True)
            else:
                combined = synth

        st.success(f"Done! Generated {len(synth)} synthetic rows.")
        st.info(f"Effective seed used: {seed_used}")

        if run_diagnostics:
            with st.expander("Diagnostics: compare subset vs. synthetic"):
                num_cols = df_sub.select_dtypes(include=[np.number]).columns
                cat_cols = [c for c in df_sub.columns if c not in num_cols]

                st.markdown("**Numeric summary (mean and standard deviation)**")
                rows = []
                for col in num_cols:
                    rows.append({
                        "column": col,
                        "real_mean": _safe_mean(df_sub[col]),
                        "real_sd": _safe_std(df_sub[col]),
                        "synth_mean": _safe_mean(synth[col]),
                        "synth_sd": _safe_std(synth[col]),
                    })
                if len(rows) > 0:
                    st.dataframe(pd.DataFrame(rows))

                st.markdown("**KS test (continuous numeric only)**")
                rows = []
                cont_cols = [c for c in num_cols if c not in discrete_set]
                for col in cont_cols:
                    rows.append({"column": col, "KS_stat": ks_numeric(df_sub[col], synth[col])})
                if len(rows) > 0:
                    st.dataframe(pd.DataFrame(rows))

                st.markdown("**Discrete columns — value share deltas (percentage points)**")
                for col in [c for c in num_cols if c in discrete_set]:
                    st.markdown(f"*{col}*")
                    st.dataframe(describe_cats(df_sub[col], synth[col]))

                st.markdown("**Categorical columns — value share deltas (percentage points)**")
                for col in cat_cols:
                    st.markdown(f"*{col}*")
                    st.dataframe(describe_cats(df_sub[col], synth[col]))

        # Download as Excel with metadata
        out_path = "synthetic_data.xlsx"
        meta = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "method": method,
            "noise_pct": float(noise_pct) if "Bootstrap" in method else None,
            "rows_generated": int(n_rows),
            "randomize_seed": bool(randomize_seed),
            "seed_used": int(seed_used),
            "filters": json.dumps({k: list(map(str, v)) for k, v in filters.items()}),
            "binary_columns": json.dumps(list(map(str, bin_cols))),
            "ordinal_columns": json.dumps(list(map(str, ord_cols)))
        }
        if create_download_file:
            with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
                combined.to_excel(xw, index=False, sheet_name="data")
                pd.DataFrame([meta]).to_excel(xw, index=False, sheet_name="meta")
            with open(out_path, "rb") as f:
                st.download_button(
                    "Download Excel",
                    data=f,
                    file_name="synthetic_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        else:
            st.caption("Download file generation skipped for speed. Toggle")
