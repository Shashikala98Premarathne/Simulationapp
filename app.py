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
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
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

# CTGAN generator (modern SDV API, for SDV >= 1.20)
# ------------------------------
# CTGAN generator (handles nulls safely)
# ------------------------------
# ------------------------------
# CTGAN generator (robust, null-safe)
# ------------------------------
def ctgan_generate(df_sub: pd.DataFrame, n_rows: int, epochs: int = 100, seed: int = 42):
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata

    # --- Step 1: Make a safe copy and convert mixed types ---
    df_clean = df_sub.copy()

    # Convert any object columns that are actually numeric (strings of numbers)
    for col in df_clean.columns:
        if df_clean[col].dtype == "object":
            try:
                df_clean[col] = pd.to_numeric(df_clean[col])
            except Exception:
                pass  # leave as object if conversion fails

    # --- Step 2: Impute missing values robustly ---
    for col in df_clean.columns:
        if df_clean[col].dtype.kind in "biufc":  # numeric columns
            if df_clean[col].isnull().all():
                # If the column is fully NaN, fill with 0
                df_clean[col] = 0
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        else:
            if df_clean[col].isnull().all():
                df_clean[col] = "Unknown"
            else:
                mode_val = df_clean[col].mode()
                df_clean[col] = df_clean[col].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown")

    # --- Step 3: Verify no nulls remain ---
    if df_clean.isnull().any().any():
        df_clean = df_clean.fillna(0)  # force fill last remaining ones

    # --- Step 4: Build metadata ---
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_clean)

    # --- Step 5: Initialize CTGAN ---
    synthesizer = CTGANSynthesizer(
        metadata=metadata,
        enforce_min_max_values=True,
        epochs=epochs,
        verbose=True
    )

      # --- Step 6: Train the model ---
    synthesizer.fit(df_clean)

    # --- Step 7: Generate synthetic samples (without randomize_seed) ---
    np.random.seed(seed)
    synthetic_data = synthesizer.sample(num_rows=n_rows)

    return synthetic_data



    # --- Step 4: Fit the model ---
    synthesizer.fit(df_clean)

    # --- Step 5: Generate synthetic samples ---
    synthetic_data = synthesizer.sample(
        num_rows=n_rows,
        randomize_seed=seed
    )

    return synthetic_data



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
    # Drop metadata columns if they exist
    for drop_col in ["__source__", "__y__"]:
        df_real = df_real.drop(columns=[drop_col], errors="ignore")
        df_synth = df_synth.drop(columns=[drop_col], errors="ignore")

    # Keep only common columns
    common_cols = [c for c in df_real.columns if c in df_synth.columns]
    if len(common_cols) == 0:
        return np.nan

    df_real = df_real[common_cols].copy()
    df_synth = df_synth[common_cols].copy()

    # Convert all column names to strings (prevents sklearn error)
    df_real.columns = df_real.columns.astype(str)
    df_synth.columns = df_synth.columns.astype(str)

    # Convert categorical/object columns to strings to ensure consistent encoding
    for c in common_cols:
        if df_real[c].dtype == "object" or df_synth[c].dtype == "object":
            df_real[c] = df_real[c].astype(str)
            df_synth[c] = df_synth[c].astype(str)

    # Combine and label data
    df_r = df_real.copy(); df_r["__y__"] = 0
    df_s = df_synth.copy(); df_s["__y__"] = 1
    df_all = pd.concat([df_r, df_s], ignore_index=True)

    X = df_all.drop(columns=["__y__"])
    y = df_all["__y__"].to_numpy()

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    Xn = X[num_cols].fillna(0) if len(num_cols) > 0 else pd.DataFrame(index=X.index)

    # Encode categoricals safely
    if len(cat_cols) > 0:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        Xc = pd.DataFrame(enc.fit_transform(X[cat_cols]))
        Xc.index = X.index
        Xenc = pd.concat([Xn, Xc], axis=1)
    else:
        Xenc = Xn

    # Ensure all feature names are strings
    Xenc.columns = Xenc.columns.astype(str)

    # Fit classifier
    clf = LogisticRegression(max_iter=500)
    clf.fit(Xenc, y)
    auc = roc_auc_score(y, clf.predict_proba(Xenc)[:, 1])
    return auc

# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="Simulated Responses Builder", layout="wide")
def set_background_solid(main="#7DD9E651", sidebar="#EEEFF3"):
    st.markdown(f"""
    <style>
      [data-testid="stAppViewContainer"],
      [data-testid="stAppViewContainer"] .main,
      [data-testid="stAppViewContainer"] .block-container {{
        background-color: {main} !important;
      }}
      [data-testid="stSidebar"],
      [data-testid="stSidebar"] > div,
      [data-testid="stSidebar"] .block-container {{
        background-color: {sidebar} !important;
      }}
      header[data-testid="stHeader"] {{ background: transparent; }}
      [data-testid="stDataFrame"],
      [data-testid="stTable"] {{ background-color: transparent !important; }}
    </style>
    """, unsafe_allow_html=True)
set_background_solid()

st.title("📊 Simulated Responses Builder")
st.caption("💡 Tip: For subsets (n < 500), use **Bootstrap + Jitter** for realistic results. "
    "For medium subsets (n = 500–1000), use **Parametric (MVN)** to preserve correlations. "
    "For larger subsets (n > 1000), try **CTGAN** for more variety and deep-pattern synthesis. "
    "Check the *Method* selector in the sidebar.")


with st.sidebar:
    st.header("1) Load data")
    up = st.file_uploader("Upload Excel file", type=["xlsx", "xls"]) 
    seed = st.number_input("Random seed", value=42, step=1)
    randomize_seed = st.checkbox("Randomize seed each run", value=False)
    seed_used = int(secrets.randbits(64)) if randomize_seed else int(seed)
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
            ["Bootstrap + Jitter (recommended)", "Parametric (MVN numeric + categorical resampling)", "CTGAN (Deep Learning — realistic synthetic data)"]
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

    # ✅ All simulation logic stays *inside* the tab
    if st.button("Run simulation", type="primary"):
        with st.spinner("Generating synthetic rows..."):
            seed_used = int(secrets.randbits(64)) if randomize_seed else int(seed)

            if "Bootstrap" in method:
                synth = bootstrap_jitter_sample(
                    df_sub,
                    n_rows=int(n_rows),
                    noise_pct=float(noise_pct),
                    seed=seed_used,
                    discrete_cols=discrete_set
                )
            elif "Parametric" in method:
                synth = parametric_mvn_sample(
                    df_sub,
                    n_rows=int(n_rows),
                    seed=seed_used,
                    discrete_cols=discrete_set
                )
            else:  # CTGAN
                with st.spinner("Training CTGAN model (this may take a minute)..."):
                    synth = ctgan_generate(
                        df_sub,
                        n_rows=int(n_rows),
                        epochs=100,
                        seed=seed_used
                    )

            # Clear excluded columns (keep them blank)
            for col in exclude_cols:
                if col in synth.columns:
                    synth[col] = None

            if include_original_flag:
                df_tagged = df.copy()
                df_tagged["__source__"] = "real"
                synth_tagged = synth.copy()
                synth_tagged["__source__"] = "synthetic"
                combined = pd.concat([df_tagged, synth_tagged], ignore_index=True)
            else:
                combined = synth

        st.success(f"Done! Generated {len(synth)} synthetic rows.")
        st.info(f"Effective seed used: {seed_used}")

        if run_diagnostics:
            with st.expander("Diagnostics: compare subset vs. synthetic"):
                ...
                # (keep diagnostics section as is)

        # ✅ Download Excel section — also stays inside this tab
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
            st.caption("Download file generation skipped for speed. Toggle the checkbox above to enable it.")

# ---------------- Validation Tab ----------------
with validate_tab:
    st.subheader("Validation: Evaluate Synthetic Data Quality")

    if df is None or len(df) == 0:
        st.warning("Please upload your data in the Simulation tab first.")
        st.stop()

    st.markdown("""
    This section tests how well your chosen simulation method reproduces unseen patterns
    using a **train / holdout split**. The holdout data acts as "ground truth" for comparison.
    """)

    val_fraction = st.slider("Holdout fraction", 0.1, 0.5, 0.2, 0.05)
    boost_factor = st.number_input("Synthetic multiplier (rows per real row in train set)", 1, 10, 3)
    run_val = st.button("Run Validation", type="primary")

    if run_val:
        with st.spinner("Running validation..."):
            # Split data
            train_df, holdout_df = train_test_split(df, test_size=val_fraction, random_state=int(seed))
            seed_val = int(secrets.randbits(64)) if randomize_seed else int(seed)

            # Generate synthetic data
            if "Bootstrap" in method:
                synth_val = bootstrap_jitter_sample(
                    train_df, n_rows=len(train_df) * int(boost_factor),
                    noise_pct=float(noise_pct), seed=seed_val, discrete_cols=discrete_set
                )
            elif "Parametric" in method:
                synth_val = parametric_mvn_sample(
                    train_df, n_rows=len(train_df) * int(boost_factor),
                    seed=seed_val, discrete_cols=discrete_set
                )
            else:  # CTGAN
                synth_val = ctgan_generate(
                    train_df, n_rows=len(train_df) * int(boost_factor),
                    seed=seed_val
                )

            synth_val = enforce_discrete_constraints(synth_val, bin_cols, ord_cols, allowed_vals)

            # Compute per-column metrics
            rows = []
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    if col in discrete_set:
                        val = mae_numeric(train_df[col], synth_val[col])
                        metric = "MAE (Discrete)"
                    else:
                        val = ks_numeric(train_df[col], synth_val[col])
                        metric = "KS (Continuous)"
                else:
                    val = psi_categorical(train_df[col], synth_val[col])
                    metric = "PSI (Categorical)"
                rows.append({"Column": col, "Metric": metric, "Value": round(val, 4) if pd.notna(val) else None})

            result_df = pd.DataFrame(rows)

            st.markdown("Validation Metrics")
            st.dataframe(result_df)

            auc_val = auc_real_vs_synth(holdout_df, synth_val)
            st.metric("Classifier AUC (Real vs Synthetic)", f"{auc_val:.3f}")

            if auc_val < 0.55:
                st.success("✅ Excellent: Synthetic data is highly realistic and hard to distinguish from real data.")
            elif auc_val < 0.7:
                st.info("⚠️ Acceptable: Synthetic data is somewhat realistic but could be improved.")
            else:
                st.warning("🚨 Warning: Synthetic data is easily distinguishable. Try adjusting the generation method or parameters.")


            st.success("Validation completed successfully!")
       