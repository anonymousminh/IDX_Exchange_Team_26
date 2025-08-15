from __future__ import annotations
import sys
import os
import json
import importlib.util
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from joblib import load as joblib_load
from sklearn.metrics import r2_score

# ===================== CONFIG =====================
sys.path.append(os.path.abspath("../RF_App"))
sys.path.append(os.path.abspath("../LGBM_App"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    "DATA_CSV": os.path.join(BASE_DIR, "data", "CRMLSSold202507.csv"),
    "TARGET": "ClosePrice",

    "RF_MODEL":   os.path.join(BASE_DIR, "..", "RF_App", "model.pkl"),
    "RF_PREPROC": os.path.join(BASE_DIR, "..", "RF_App", "preprocessor.pkl"),
    "PRICE_BOUNDS": os.path.join(BASE_DIR, "..", "RF_App", "price_bounds.pkl"),

    "LGBM_MODEL":      os.path.join(BASE_DIR, "..", "LGBM_App", "model.pkl"),
    "LGBM_PREPROC_PY": os.path.join(BASE_DIR, "..", "LGBM_App", "preprocessing.py"),
    "LGBM_REF_JSON":   None,

    "OUT_CSV": os.path.join(BASE_DIR, "output", "compare_results.csv"),
}

SLICING = {
    "top_k_counties": 5,
    "min_n_per_slice": 150,
    "county_col": "CountyOrParish",
    "price_col": "ClosePrice",
    "year_col": "YearBuilt",
}

BOUNDS_MODE = "both"
# ==================================================

# ===================== METRICS =====================
def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))

def r2_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return np.nan
# ==================================================

# ===================== HELPERS =====================
def _safe_joblib(path: Optional[str]):
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    return joblib_load(path)

def _load_module(py_path: str):
    if not os.path.exists(py_path):
        raise FileNotFoundError(f"Not found: {py_path}")
    spec = importlib.util.spec_from_file_location("lgbm_preproc_mod", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def _auto_ref_values(df: pd.DataFrame, numeric_cols: List[str], cat_cols: List[str]) -> Dict[str, object]:
    ref: Dict[str, object] = {}
    for c in numeric_cols:
        ref[c] = float(pd.to_numeric(df[c], errors="coerce").median()) if c in df.columns else 0.0
    for c in cat_cols:
        if c in df.columns and df[c].notna().any():
            ref[c] = df[c].mode(dropna=True).iloc[0]
        else:
            ref[c] = "Unknown"
    if "YearBuilt" in df.columns:
        yb = pd.to_numeric(df["YearBuilt"], errors="coerce").median()
        ref["HomeAge"] = float(2025 - yb) if pd.notna(yb) else 30.0
    else:
        ref["HomeAge"] = 30.0
    return ref
# ==================================================

# ===================== SLICES =====================
def _quartile_masks(series: pd.Series, min_n: int):
    s = pd.to_numeric(series, errors="coerce")
    q = s.quantile([0.25, 0.5, 0.75]).to_dict()
    cuts = [
        ("Price_Q1", (-np.inf, q[0.25])),
        ("Price_Q2", (q[0.25], q[0.5])),
        ("Price_Q3", (q[0.5], q[0.75])),
        ("Price_Q4", (q[0.75], np.inf)),
    ]
    masks = {}
    for name, (lo, hi) in cuts:
        m = (s > lo) & (s <= hi)
        if m.sum() >= min_n:
            masks[name] = m
    return masks

def _topk_county_masks(df: pd.DataFrame, county_col: str, k: int, min_n: int):
    masks = {}
    top = df[county_col].astype(str).value_counts(dropna=False).head(k).index.tolist()
    for c in top:
        m = df[county_col].astype(str).eq(str(c))
        if m.sum() >= min_n:
            masks[f"County::{c}"] = m
    other = ~df[county_col].astype(str).isin([str(x) for x in top])
    if other.sum() >= min_n:
        masks["County::Other"] = other
    return masks

def _year_masks(df: pd.DataFrame, year_col: str, min_n: int):
    y = pd.to_numeric(df[year_col], errors="coerce")
    buckets = [
        ("YearBuilt_<1980", (-np.inf, 1980)),
        ("YearBuilt_1980_2009", (1980, 2010)),
        ("YearBuilt_2010plus", (2010, np.inf)),
    ]
    masks = {}
    for name, (lo, hi) in buckets:
        m = (y >= lo) & (y < hi)
        if m.sum() >= min_n:
            masks[name] = m
    return masks
# ==================================================

# ===================== MAIN =====================
def main():
    os.makedirs(os.path.dirname(PATHS["OUT_CSV"]), exist_ok=True)

    df_all = pd.read_csv(PATHS["DATA_CSV"]).dropna(subset=[PATHS["TARGET"]]).copy()
    print(f"[Data] loaded {len(df_all):,} rows")

    df_for_rf = df_all.copy()
    df_for_lgbm = df_all.copy()

    if PATHS.get("PRICE_BOUNDS") and BOUNDS_MODE in {"both", "rf_only"}:
        bounds_obj = _safe_joblib(PATHS["PRICE_BOUNDS"])
        lo, hi = _parse_bounds(bounds_obj)
        tgt = PATHS["TARGET"]

        if BOUNDS_MODE == "both":
            before = len(df_all)
            df_all = df_all[(df_all[tgt] >= lo) & (df_all[tgt] <= hi)].copy()
            df_for_rf = df_all
            df_for_lgbm = df_all
            print(f"[Bounds both] {before:,} -> {len(df_all):,} rows kept (lo={lo:.0f}, hi={hi:.0f})")

        elif BOUNDS_MODE == "rf_only":
            before = len(df_for_rf)
            df_for_rf = df_for_rf[(df_for_rf[tgt] >= lo) & (df_for_rf[tgt] <= hi)].copy()
            print(f"[Bounds rf_only] RF: {before:,} -> {len(df_for_rf):,} kept; "
                  f"LGBM: {len(df_for_lgbm):,} (no bounds)")
    else:
        print("[Bounds] OFF")

    # ----- RF -----
    rf_model = _safe_joblib(PATHS["RF_MODEL"])
    rf_preproc = _safe_joblib(PATHS["RF_PREPROC"])
    df_rf = rf_preproc.transform(df_for_rf)
    rf_feats = [c for c in df_rf.columns if c != PATHS["TARGET"]]

    # model gives log(price), change back
    rf_preds_log = rf_model.predict(df_rf[rf_feats])
    y_pred_rf = pd.Series(np.expm1(rf_preds_log), index=df_rf.index, name="_pred_rf")

    base_rf = df_rf.copy()
    base_rf["_pred_rf"] = y_pred_rf

    # ----- LGBM -----
    mod = _load_module(PATHS["LGBM_PREPROC_PY"])
    preprocess_input = getattr(mod, "preprocess_input")
    NUMERIC_FEATURES = getattr(mod, "NUMERIC_FEATURES")
    CATEGORICAL_FEATURES = getattr(mod, "CATEGORICAL_FEATURES")

    if PATHS["LGBM_REF_JSON"]:
        with open(PATHS["LGBM_REF_JSON"], "r") as f:
            ref_values = json.load(f)
    else:
        ref_values = _auto_ref_values(df_for_lgbm, NUMERIC_FEATURES, CATEGORICAL_FEATURES)

    X_lgbm = preprocess_input(df_for_lgbm, ref_values)
    for c in CATEGORICAL_FEATURES:
        if c in X_lgbm.columns and X_lgbm[c].dtype.name != "category":
            X_lgbm[c] = X_lgbm[c].astype("category")

    lgbm_model = _safe_joblib(PATHS["LGBM_MODEL"])

    # change log(price) back
    lgbm_preds_log = lgbm_model.predict(X_lgbm)
    y_pred_lgbm = pd.Series(np.expm1(lgbm_preds_log), index=df_for_lgbm.index, name="_pred_lgbm")

    base_lgbm = df_for_lgbm.copy()
    base_lgbm["_pred_lgbm"] = y_pred_lgbm

    # ----- Slicing & Metrics -----
    rows = []
    def eval_model(base_df, pred_col, model_name):
        masks = {"ALL": pd.Series(True, index=base_df.index)}
        if SLICING["price_col"] in base_df.columns:
            masks.update(_quartile_masks(base_df[SLICING["price_col"]], SLICING["min_n_per_slice"]))
        if SLICING["county_col"] in base_df.columns:
            masks.update(_topk_county_masks(base_df, SLICING["county_col"], SLICING["top_k_counties"], SLICING["min_n_per_slice"]))
        if SLICING["year_col"] in base_df.columns:
            masks.update(_year_masks(base_df, SLICING["year_col"], SLICING["min_n_per_slice"]))

        for name, m in masks.items():
            idx = base_df.index[m]
            if len(idx) == 0:
                continue
            y = base_df.loc[idx, PATHS["TARGET"]].to_numpy()
            p = base_df.loc[idx, pred_col].to_numpy()
            rows.append({
                "model": model_name,
                "slice": name,
                "n": int(len(idx)),
                "R2": r2_safe(y, p),
                "MAPE": mape(y, p)
            })

    eval_model(base_rf, "_pred_rf", "RF")
    eval_model(base_lgbm, "_pred_lgbm", "LGBM")

    res = pd.DataFrame(rows).sort_values(["model", "slice"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(PATHS["OUT_CSV"]), exist_ok=True)
    res.to_csv(PATHS["OUT_CSV"], index=False)
    print(f"[Saved] {PATHS['OUT_CSV']}")
    print(res)

def _parse_bounds(obj: Any) -> Tuple[float, float]:
    if isinstance(obj, (list, tuple)) and len(obj) == 2:
        return float(obj[0]), float(obj[1])
    if isinstance(obj, dict):
        for k1, k2 in [("low", "high"), ("min", "max")]:
            if k1 in obj and k2 in obj:
                return float(obj[k1]), float(obj[k2])
    raise ValueError("Unrecognized price_bounds format.")

if __name__ == "__main__":
    main()
