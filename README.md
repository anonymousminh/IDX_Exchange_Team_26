# Purpose:
   Side-by-side evaluation of RF vs LGBM on the SAME OOT dataset
   with optional price-bounds filtering. It supports:
     - Different preprocessors for RF and LGBM
     - Fair comparison on the intersection of predicted rows
     - Slices: ALL / Price quartiles / Top-K counties / Year buckets
   Metrics: R2, MAPE only (per your requirement for CA housing)
# ------------------------------------------------------------

# Folder layout (siblings under one parent folder):
   IDX_Exchange_Team_26/
     ├─ RF_App/
     │  ├─ model.pkl
     │  ├─ preprocessor.pkl
     │  └─ price_bounds.pkl
     ├─ LGBM_App/
     │  ├─ lgbm_model.pkl
     │  └─ lgbm_preprocess.py  (contains preprocess_input & feature lists)
     └─ Compare_RF_LGBM/
        ├─ compare_rf_lgbm.py  (this file)
        ├─ data/CRMLSSold202507.csv
        └─ output/compare_results.csv (auto generated)
# ------------------------------------------------------------