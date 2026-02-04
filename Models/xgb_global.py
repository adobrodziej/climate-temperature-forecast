#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Trening 7 globalnych modeli XGBoost (po horyzoncie) + kalibracja hybrydowa (predykcja vs klimatologia).
- Odczytuje features_global.parquet
- One-hot: station_id, region
- Zbiór cech: czasowe (DOY/DOW/month/weekend/Fourier), meta (lat/lon/elev), klimatologia, lagi/role, bieżące obserwacje t_mean
- Oddzielnie dla h=1..7 uczy model z early stopping (val), zapisuje model i metryki
- Baseliney: persistence (t_mean_lag_h) i klimatologia (clim_t_mean)
- Kalibracja: LinearRegression(y ~ pred_raw + clim)
- Artefakty: metrics_val_calibrated.csv, metrics_test_calibrated.csv, metrics_test_baseline.csv,
             metrics_test_per_station.csv, preds_test_calibrated.csv, improvement_vs_baseline_test.csv,
             models/, plots/, params.json, runtime.txt

Globalny XGBoost (po horyzoncie) — wersja pod (Python 3.10, pandas 2.3, XGBoost 3.0.4).
- Early stopping przez callback (XGB ≥2/3) z fallbackiem na parametr (XGB 1.7).
- Opcjonalny filtr: `--restrict_to_meta --meta_csv ...` (trening tylko na stacjach z metadanych — 62).
- Drop kolumn tekstowych / dtype=object (np. station_name_x/y) + filtr typów cech akceptowanych przez XGB.
- Odporność na NaN: trening/metryki/kalibracja na maskach kompletności (bez NaN w labelach i baseline).
- Bbaseline **linreg7** obok persistence i klimatologii (prosta regresja po 7 ostatnich dniach).

Uruchomienie:
python xgb_global.py `
  --features "E:\Final_mgr\Database\Dobowe_klimat\processed\reports\global_xgb\features_global_geoFix.parquet" `
  --outdir   "E:\Final_mgr\Database\Dobowe_klimat\processed\reports\global_xgb\runs\run_geoFix" `
  --meta_csv "E:\Final_mgr\Database\Dobowe_klimat\processed\reports\global_xgb\station_meta_clean.csv" `
  --restrict_to_meta `
  --horizons 1 2 3 4 5 6 7 `
  --learning_rate 0.02 `
  --n_estimators 900 `
  --max_depth 6 `
  --subsample 0.85 `
  --colsample_bytree 0.85 `
  --min_child_weight 12 `
  --gamma 0.2 `
  --reg_alpha 0.2 `
  --reg_lambda 1.0 `
  --early_stopping_rounds 50

"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

# EarlyStopping (XGBoost ≥2)
try:
    from xgboost.callback import EarlyStopping as XGB_EarlyStopping
except Exception:
    XGB_EarlyStopping = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HORIZONS_DEFAULT = [1, 2, 3, 4, 5, 6, 7]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--meta_csv", type=str, default=None, help="CSV z metadanymi stacji (filtr i etykiety)")
    p.add_argument("--restrict_to_meta", action="store_true", help="Filtruj do station_id z meta_csv")

    p.add_argument("--horizons", type=int, nargs="*", default=HORIZONS_DEFAULT)
    p.add_argument("--early_stopping_rounds", type=int, default=50)
    p.add_argument("--n_estimators", type=int, default=2000)
    p.add_argument("--learning_rate", type=float, default=0.03)
    p.add_argument("--max_depth", type=int, default=8)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample_bytree", type=float, default=0.8)
    p.add_argument("--reg_lambda", type=float, default=1.0)
    p.add_argument("--min_child_weight", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--reg_alpha", type=float, default=0.0)

    p.add_argument("--n_jobs", type=int, default=0)
    return p.parse_args()


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude_prefixes = ["y_h", "model_pred_h", "pred_cal_h"]
    exclude_exact = {"date", "split", "station_name"}
    cols = []
    for c in df.columns:
        if any(c.startswith(p) for p in exclude_prefixes):
            continue
        if c in exclude_exact:
            continue
        if c.startswith("station_id_") or c.startswith("region_"):
            cols.append(c)
        elif c in {"station_id", "region"}:
            continue
        else:
            cols.append(c)
    accepted = set(df.select_dtypes(include=["number", "bool", "category"]).columns)
    cols = [c for c in cols if c in accepted]
    return cols


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# === pomoc do etykiet stacji ===

def _cap_first(s: str) -> str:
    return s[:1].upper() + s[1:].lower() if s else s


def _norm_spaces(txt: str) -> str:
    return " ".join(txt.split())


def format_station_label(name: str) -> str:
    """Skracanie nazw stacji do formy czytelnej na wykresach.
    Zasady:
      - kapitalizacja zdaniowa (pierwszy wyraz pełny),
      - kolejne słowa po SPACJACH → inicjały z kropką: "Bukowina T.", "Dolina P. S.",
      - jeśli pierwszy token ma MYŚLNIK: "Solina-J.", dalsze słowa po spacji ignorujemy
        (np. "Warszawa-Obserwatorium II" → "Warszawa-O.").
    """
    if not isinstance(name, str) or not name.strip():
        return ""
    name = _norm_spaces(name.strip())
    tokens = name.split(" ")
    if "-" in tokens[0]:
        parts = [p.strip() for p in tokens[0].split("-") if p.strip()]
        first = _cap_first(parts[0]) if parts else ""
        second_init = (parts[1][:1].upper() + ".") if len(parts) > 1 else ""
        return first + ("-" + second_init if second_init else "")
    first = _cap_first(tokens[0])
    inits = [t[:1].upper() + "." for t in tokens[1:] if t]
    return " ".join([first] + inits)


def select_top_mid_bottom(df: pd.DataFrame, k_each: int = 5) -> pd.DataFrame:
    d = df.sort_values("mae_delta", ascending=False).reset_index(drop=True)
    n = len(d)
    if n <= 15:
        return d
    top = d.head(k_each)
    bot = d.tail(k_each)
    mid_center = n // 2
    half = max(1, k_each // 2)
    start = max(k_each, mid_center - half)
    end = min(n - k_each, start + k_each)
    mid = d.iloc[start:end]
    return pd.concat([top, mid, bot], ignore_index=True)


def plot_improvement_bar(per_station: pd.DataFrame, out_png: Path, title: str, label_col: str = "station_label"):
    fig, ax = plt.subplots(figsize=(10, 5))
    per_station_sorted = per_station.sort_values("mae_delta", ascending=False)
    xlabels = per_station_sorted[label_col].astype(str)
    ax.bar(xlabels, per_station_sorted["mae_delta"].astype(float).values)
    ax.set_xlabel("stacja")
    ax.set_ylabel("MAE baseline − MAE model_cal [°C]")
    ax.set_title(title)
    ax.tick_params(axis='x', labelrotation=90)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def _label_season(dates: pd.Series) -> pd.Series:
    m = pd.to_datetime(dates).dt.month
    s = pd.Series(index=dates.index, dtype="object")
    s[(m == 12) | (m <= 2)] = "DJF"
    s[(m >= 3) & (m <= 5)] = "MAM"
    s[(m >= 6) & (m <= 8)] = "JJA"
    s[(m >= 9) & (m <= 11)] = "SON"
    return s


def _baseline_linreg7_block(df_block: pd.DataFrame, h: int, col_prefix: str = "t_mean_lag_") -> pd.Series:
    lag_cols = [f"{col_prefix}{i}" for i in range(7, 0, -1)]
    if not all(c in df_block.columns for c in lag_cols):
        return df_block.get(f"{col_prefix}{h}", pd.Series(np.nan, index=df_block.index))
    Y = df_block[lag_cols].astype(float).to_numpy()
    x = np.arange(1, 8, dtype=float)
    x_mean = x.mean(); x_c = x - x_mean; denom = np.sum(x_c**2)
    slope = (Y @ x_c) / denom
    y_mean = Y.mean(axis=1)
    intercept = y_mean - slope * x_mean
    return pd.Series(intercept + slope * (7 + h), index=df_block.index)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    (outdir / "models").mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    df = pd.read_parquet(args.features)

    # META: etykiety i ewentualny filtr
    id2name = {}
    meta = None
    if args.meta_csv and os.path.exists(args.meta_csv):
        meta = pd.read_csv(args.meta_csv)
        if "station_id" in meta.columns:
            meta["station_id"] = meta["station_id"].astype(str)
        if "station_name" in meta.columns:
            id2name = dict(zip(meta["station_id"].astype(str), meta["station_name"].astype(str)))

    if args.restrict_to_meta:
        if meta is None:
            raise ValueError("--restrict_to_meta wymaga podania --meta_csv")
        df["station_id"] = df["station_id"].astype(str)
        before = len(df)
        df = df[df["station_id"].isin(set(meta["station_id"]))].copy()
        print(f"[filter] restrict_to_meta: {before} -> {len(df)}; stacje: {df['station_id'].nunique()}")

    # Flaga braku regionu
    if "region" in df.columns:
        df["region_nan"] = df["region"].isna().astype("int8")

    # One-hot dla station_id i region
    dfd = pd.get_dummies(df, columns=[c for c in ["station_id", "region"] if c in df.columns], drop_first=False)

    # Usuń kolumny tekstowe/dtype object
    text_cols = dfd.select_dtypes(include=["object"]).columns.tolist()
    name_like = [c for c in dfd.columns if c.startswith("station_name")]

    protected = {"split"}
    text_cols = [c for c in text_cols if c not in protected]

    to_drop = sorted(set(text_cols + name_like))
    if to_drop:
        dfd = dfd.drop(columns=to_drop)

    if "split" not in dfd.columns:
        raise ValueError("Brak kolumny 'split' w danych cech.")
    _split = dfd["split"].astype(str)
    mask_train = _split.eq("train")
    mask_val   = _split.eq("val")
    mask_test  = _split.eq("test")

    # Klimatologie konwersja i imputacja
    for clim_col in [c for c in dfd.columns if c.startswith("clim_t_mean")]:
        dfd[clim_col] = pd.to_numeric(dfd[clim_col], errors="coerce")
        train_mean = dfd.loc[mask_train, clim_col].mean()
        if pd.isna(train_mean):
            continue
        na_mask = dfd[clim_col].isna()
        dfd.loc[na_mask, clim_col] = train_mean

    feature_cols = select_feature_columns(dfd)

    rows_metrics_val_cal, rows_metrics_test_cal = [], []
    rows_metrics_baseline = []
    rows_preds_test, rows_preds_val = [], []
    rows_per_station = []

    for h in args.horizons:
        y_col = f"y_h{h}"
        if y_col not in dfd.columns:
            raise ValueError(f"Brak kolumny targetu: {y_col}")

        mask_train = dfd["split"] == "train"
        mask_val   = dfd["split"] == "val"
        mask_test  = dfd["split"] == "test"

        X_train = dfd.loc[mask_train, feature_cols]
        y_train = dfd.loc[mask_train, y_col]
        X_val   = dfd.loc[mask_val, feature_cols]
        y_val   = dfd.loc[mask_val, y_col]
        X_test  = dfd.loc[mask_test, feature_cols]
        y_test  = dfd.loc[mask_test, y_col]

        # Baselines
        pers_col = f"t_mean_lag_{h}"
        clim_col = f"clim_t_mean_h{h}"
        if pers_col not in dfd.columns:
            raise ValueError(f"Brak kolumny baseline persistence: {pers_col}")
        if clim_col not in dfd.columns:
            raise ValueError(f"Brak kolumny klimatologii dla h={h}: {clim_col}")

        baseline_val_pers  = dfd.loc[mask_val,  pers_col]
        baseline_val_clim  = dfd.loc[mask_val,  clim_col]
        baseline_test_pers = dfd.loc[mask_test, pers_col]
        baseline_test_clim = dfd.loc[mask_test, clim_col]
        baseline_val_lin7  = _baseline_linreg7_block(dfd.loc[mask_val],  h)
        baseline_test_lin7 = _baseline_linreg7_block(dfd.loc[mask_test], h)

        # Maski kompletności
        mt  = y_train.notna()
        mv  = y_val.notna()
        mte = y_test.notna()
        mv_cal = mv & baseline_val_clim.notna()

        # Model XGB
        model = xgb.XGBRegressor(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_lambda=args.reg_lambda,
            min_child_weight=args.min_child_weight,
            gamma=args.gamma,
            reg_alpha=args.reg_alpha,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=args.n_jobs if args.n_jobs != 0 else None,
            random_state=42,
        )

        ok = False
        if XGB_EarlyStopping is not None:
            try:
                model.fit(
                    X_train.loc[mt], y_train.loc[mt],
                    eval_set=[(X_val.loc[mv], y_val.loc[mv])],
                    verbose=False,
                    callbacks=[XGB_EarlyStopping(rounds=args.early_stopping_rounds, save_best=True)],
                ); ok = True
            except TypeError:
                pass
        if not ok:
            try:
                model.fit(
                    X_train.loc[mt], y_train.loc[mt],
                    eval_set=[(X_val.loc[mv], y_val.loc[mv])],
                    verbose=False,
                    early_stopping_rounds=args.early_stopping_rounds,
                ); ok = True
            except TypeError:
                pass
        if not ok:
            model.fit(X_train.loc[mt], y_train.loc[mt], eval_set=[(X_val.loc[mv], y_val.loc[mv])], verbose=False)

        # Predykcje raw
        pred_val_raw  = pd.Series(model.predict(X_val),  index=X_val.index)
        pred_test_raw = pd.Series(model.predict(X_test), index=X_test.index)

        # Kalibracja hybrydowa (OLS): y ~ pred_raw + klimatologia
        cal = LinearRegression()
        X_cal_val = np.column_stack([pred_val_raw.loc[mv_cal].values,
                                     baseline_val_clim.loc[mv_cal].values])
        cal.fit(X_cal_val, y_val.loc[mv_cal].values)

        pred_val_cal  = pd.Series(np.nan, index=X_val.index)
        pred_test_cal = pd.Series(np.nan, index=X_test.index)
        pred_val_cal.loc[mv_cal] = cal.predict(X_cal_val)
        mte_cal = baseline_test_clim.notna()
        X_cal_test = np.column_stack([pred_test_raw.loc[mte_cal].values,
                                      baseline_test_clim.loc[mte_cal].values])
        pred_test_cal.loc[mte_cal] = cal.predict(X_cal_test)

        # Metryki baselineów
        mbv_p = mv & baseline_val_pers.notna()
        mbv_c = mv & baseline_val_clim.notna()
        mbv_l = mv & baseline_val_lin7.notna()
        mbt_p = mte & baseline_test_pers.notna()
        mbt_c = mte & baseline_test_clim.notna()
        mbt_l = mte & baseline_test_lin7.notna()

        rows_metrics_baseline += [
            {"horizon": h, "split": "val",  "baseline": "persistence",  "MAE": mean_absolute_error(y_val.loc[mbv_p], baseline_val_pers.loc[mbv_p]),  "RMSE": rmse(y_val.loc[mbv_p], baseline_val_pers.loc[mbv_p])},
            {"horizon": h, "split": "val",  "baseline": "climatology",  "MAE": mean_absolute_error(y_val.loc[mbv_c], baseline_val_clim.loc[mbv_c]),  "RMSE": rmse(y_val.loc[mbv_c], baseline_val_clim.loc[mbv_c])},
            {"horizon": h, "split": "val",  "baseline": "linreg7",     "MAE": mean_absolute_error(y_val.loc[mbv_l], baseline_val_lin7.loc[mbv_l]),  "RMSE": rmse(y_val.loc[mbv_l], baseline_val_lin7.loc[mbv_l])},
            {"horizon": h, "split": "test", "baseline": "persistence",  "MAE": mean_absolute_error(y_test.loc[mbt_p], baseline_test_pers.loc[mbt_p]), "RMSE": rmse(y_test.loc[mbt_p], baseline_test_pers.loc[mbt_p])},
            {"horizon": h, "split": "test", "baseline": "climatology",  "MAE": mean_absolute_error(y_test.loc[mbt_c], baseline_test_clim.loc[mbt_c]), "RMSE": rmse(y_test.loc[mbt_c], baseline_test_clim.loc[mbt_c])},
            {"horizon": h, "split": "test", "baseline": "linreg7",     "MAE": mean_absolute_error(y_test.loc[mbt_l], baseline_test_lin7.loc[mbt_l]), "RMSE": rmse(y_test.loc[mbt_l], baseline_test_lin7.loc[mbt_l])},
        ]

        # Metryki modelu (raw / calibrated)
        rows_metrics_val_cal += [
            {"horizon": h, "variant": "raw",        "MAE": mean_absolute_error(y_val.loc[mv],  pred_val_raw.loc[mv]),  "RMSE": rmse(y_val.loc[mv],  pred_val_raw.loc[mv])},
            {"horizon": h, "variant": "calibrated", "MAE": mean_absolute_error(y_val.loc[mv_cal], pred_val_cal.loc[mv_cal]), "RMSE": rmse(y_val.loc[mv_cal], pred_val_cal.loc[mv_cal])},
        ]
        rows_metrics_test_cal += [
            {"horizon": h, "variant": "raw",        "MAE": mean_absolute_error(y_test.loc[mte], pred_test_raw.loc[mte]), "RMSE": rmse(y_test.loc[mte], pred_test_raw.loc[mte])},
            {"horizon": h, "variant": "calibrated", "MAE": mean_absolute_error(y_test.loc[mte & mte_cal], pred_test_cal.loc[mte & mte_cal]), "RMSE": rmse(y_test.loc[mte & mte_cal], pred_test_cal.loc[mte & mte_cal])},
        ]

        # Predykcje TEST (long) + etykiety
        tmp = dfd.loc[dfd["split"] == "test", ["date"]].copy()
        st_cols = [c for c in dfd.columns if c.startswith("station_id_")]
        tmp["station_id"] = dfd.loc[dfd["split"] == "test", st_cols].idxmax(axis=1).str.replace("station_id_", "", regex=False) if st_cols else ""
        tmp["station_label"] = tmp["station_id"].astype(str).map(id2name).fillna(tmp["station_id"].astype(str))
        tmp["horizon"] = h
        tmp["y_true"] = y_test.values
        tmp["y_pred_raw"] = pred_test_raw.values
        tmp["y_pred_cal"] = pred_test_cal.values
        tmp["baseline_persistence"] = baseline_test_pers.values
        tmp["baseline_climatology"] = baseline_test_clim.values
        tmp["baseline_linreg7"] = baseline_test_lin7.values
        rows_preds_test.append(tmp)

        # Predykcje VAL (long) + etykiety — zapisujemy do conformal
        tmpv = dfd.loc[dfd["split"] == "val", ["date"]].copy()
        tmpv["station_id"] = dfd.loc[dfd["split"] == "val", st_cols].idxmax(axis=1).str.replace("station_id_", "", regex=False) if st_cols else ""
        tmpv["station_label"] = tmpv["station_id"].astype(str).map(id2name).fillna(tmpv["station_id"].astype(str))
        tmpv["horizon"] = h
        tmpv["y_true"] = y_val.values
        tmpv["y_pred_raw"] = pred_val_raw.values
        tmpv["y_pred_cal"] = pred_val_cal.values
        rows_preds_val.append(tmpv)

        # Poprawa per stacja vs persistence — bez groupby.apply
        g = tmp.dropna(subset=["y_true", "y_pred_cal", "baseline_persistence"]).copy()
        g["err_model"] = (g["y_true"] - g["y_pred_cal"]).abs()
        g["err_base"]  = (g["y_true"] - g["baseline_persistence"]).abs()
        perf = (
            g.groupby(["station_id", "station_label"])[["err_model","err_base"]]
             .mean()
             .rename(columns={"err_model":"mae_model_cal","err_base":"mae_baseline"})
             .reset_index()
        )
        perf["horizon"] = h
        perf["mae_delta"] = perf["mae_baseline"] - perf["mae_model_cal"]
        rows_per_station.append(perf)

        # Wykresy per-horyzont
        perf_plot = perf.copy()
        perf_plot["station_label"] = perf_plot["station_label"].map(lambda x: format_station_label(str(x)))
        if h == 1:
            plot_improvement_bar(
                perf_plot[["station_label", "mae_delta"]],
                outdir / "plots" / f"improvement_mae_per_station_h{h}.png",
                title=f"Poprawa vs persistence — h={h}",
            )
        perf15 = select_top_mid_bottom(perf_plot, k_each=5)
        plot_improvement_bar(
            perf15[["station_label", "mae_delta"]],
            outdir / "plots" / f"improvement_mae_per_station_h{h}_15.png",
            title=f"Poprawa vs persistence — h={h} (15 stacji)",
        )

        model.save_model(outdir / "models" / f"xgb_global_h{h}.json")

    # ===== ZAPIS ARTEFAKTÓW =====
    pd.DataFrame(rows_metrics_baseline).to_csv(outdir / "metrics_test_baseline.csv", index=False)
    pd.DataFrame(rows_metrics_val_cal).to_csv(outdir / "metrics_val_calibrated.csv", index=False)
    pd.DataFrame(rows_metrics_test_cal).to_csv(outdir / "metrics_test_calibrated.csv", index=False)

    preds_test = pd.concat(rows_preds_test, ignore_index=True)
    preds_test.to_csv(outdir / "preds_test_calibrated.csv", index=False)

    preds_val = pd.concat(rows_preds_val, ignore_index=True)
    preds_val.to_csv(outdir / "preds_val_calibrated.csv", index=False)

    per_station_all = pd.concat(rows_per_station, ignore_index=True)
    per_station_all.to_csv(outdir / "improvement_vs_baseline_test.csv", index=False)

    # ===== DODATKOWE WYKRESY =====
    # 1) Poprawa vs horyzont (średnio po stacjach)
    agg = per_station_all.groupby("horizon")["mae_delta"].mean().rename("impr_degC").reset_index()
    plt.figure(figsize=(6,4))
    plt.bar(agg["horizon"].astype(int), agg["impr_degC"].astype(float))
    plt.xlabel("horyzont [dni]"); plt.ylabel("Poprawa MAE [°C]"); plt.title("Model vs persistence — średnio (TEST)")
    plt.tight_layout(); plt.savefig(outdir / "plots" / "improvement_vs_horizon.png", dpi=150); plt.close()

    # 2) Hit-rate ≤1°C i ≤2°C vs horyzont
    preds_test["abs_err"] = (preds_test["y_pred_cal"] - preds_test["y_true"]).abs()
    hit1 = preds_test.assign(hit=(preds_test["abs_err"] <= 1.0).astype(int)).groupby("horizon")["hit"].mean().rename("hit_1C").reset_index()
    hit2 = preds_test.assign(hit=(preds_test["abs_err"] <= 2.0).astype(int)).groupby("horizon")["hit"].mean().rename("hit_2C").reset_index()
    plt.figure(figsize=(6,4))
    plt.plot(hit1["horizon"].astype(int), hit1["hit_1C"].astype(float), marker="o", label="≤1°C")
    plt.plot(hit2["horizon"].astype(int), hit2["hit_2C"].astype(float), marker="o", label="≤2°C")
    plt.ylim(0,1); plt.ylabel("Udział trafień"); plt.xlabel("horyzont [dni]"); plt.title("Trafienia (TEST)"); plt.legend()
    plt.tight_layout(); plt.savefig(outdir / "plots" / "hit_rate_vs_horizon.png", dpi=150); plt.close()

    # 3) Histogram błędów (TEST)
    plt.figure(figsize=(6,4))
    plt.hist(preds_test["abs_err"].dropna().astype(float).values, bins=50)
    plt.xlabel("|błąd| [°C]"); plt.ylabel("Liczność"); plt.title("Rozkład bezwzględnego błędu — TEST")
    plt.tight_layout(); plt.savefig(outdir / "plots" / "error_hist.png", dpi=150); plt.close()

    # 4) Parity plot (h=1) — bezpieczne próbkowanie dla małych zbiorów
    p1_all = preds_test.loc[preds_test["horizon"] == 1, ["y_true", "y_pred_cal"]].dropna()
    if len(p1_all) > 0:
        n_parity = min(50000, len(p1_all))
        p1 = p1_all.sample(n=n_parity, random_state=42) if len(p1_all) > n_parity else p1_all
        plt.figure(figsize=(5,5))
        plt.scatter(p1["y_true"].astype(float).values, p1["y_pred_cal"].astype(float).values, s=3, alpha=0.4)
        lo = float(min(p1["y_true"].min(), p1["y_pred_cal"].min()))
        hi = float(max(p1["y_true"].max(), p1["y_pred_cal"].max()))
        plt.plot([lo, hi], [lo, hi])
        plt.xlabel("Rzeczywiste [°C]"); plt.ylabel("Prognoza [°C]"); plt.title("Parity (TEST, h=1)")
        plt.tight_layout(); plt.savefig(outdir / "plots" / "parity_test_h1.png", dpi=150); plt.close()
    else:
        print("[plot] Parity (h=1) pominięty — brak danych po filtrach.")

    # 5) MAE wg sezonu
    preds_test["forecast_date"] = pd.to_datetime(preds_test["date"]) + pd.to_timedelta(preds_test["horizon"], unit="D")
    preds_test["season"] = _label_season(preds_test["forecast_date"].astype("datetime64[ns]"))
    mae_season = preds_test.dropna(subset=["y_true","y_pred_cal"]).assign(err=lambda d: (d["y_true"]-d["y_pred_cal"]).abs()).groupby("season")["err"].mean().reset_index()
    order = ["DJF","MAM","JJA","SON"]
    mae_season = mae_season.set_index("season").reindex(order).reset_index()
    plt.figure(figsize=(6,4))
    plt.bar(mae_season["season"], mae_season["err"].astype(float))
    plt.xlabel("Sezon"); plt.ylabel("MAE [°C]"); plt.title("MAE wg sezonu (TEST)")
    plt.tight_layout(); plt.savefig(outdir / "plots" / "mae_by_season.png", dpi=150); plt.close()

    # 6) Top-10 / Bottom-10 (h=1) — z krótkimi etykietami
    ps1 = per_station_all[per_station_all["horizon"]==1].copy()
    if not ps1.empty:
        top10 = ps1.sort_values("mae_delta", ascending=False).head(10)
        bot10 = ps1.sort_values("mae_delta", ascending=True).head(10)
        for t in (top10, bot10):
            if "station_label" not in t.columns:
                t["station_label"] = t["station_id"].astype(str).map(id2name).fillna(t["station_id"].astype(str))
            t["station_label"] = t["station_label"].map(lambda x: format_station_label(str(x)))
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8,8))
        axes[0].barh(top10["station_label"].astype(str), top10["mae_delta"].astype(float))
        axes[0].invert_yaxis(); axes[0].set_title("Top-10 zysk MAE (h=1)")
        axes[1].barh(bot10["station_label"].astype(str), bot10["mae_delta"].astype(float))
        axes[1].invert_yaxis(); axes[1].set_title("Bottom-10 zysk MAE (h=1)")
        for ax in axes:
            ax.set_xlabel("MAE baseline − MAE model_cal [°C]")
        fig.tight_layout(); fig.savefig(outdir / "plots" / "top10_bottom10_h1.png", dpi=150); plt.close(fig)

    # 7) Zysk kalibracji (VAL i TEST)
    m_val  = pd.DataFrame(rows_metrics_val_cal)
    m_test = pd.DataFrame(rows_metrics_test_cal)

    def _gain(df):
        piv = df.pivot(index="horizon", columns="variant", values="MAE")
        piv["gain_calibration"] = piv["raw"] - piv["calibrated"]
        return piv[["gain_calibration"]].reset_index()

    gain_val  = _gain(m_val).assign(split="val")
    gain_test = _gain(m_test).assign(split="test")
    gain_all = pd.concat([gain_val, gain_test], ignore_index=True)
    gain_all.to_csv(outdir / "calibration_gain.csv", index=False)

    plt.figure(figsize=(6,4))
    for sp, g in gain_all.groupby("split"):
        plt.plot(g["horizon"], g["gain_calibration"], marker="o", label=sp.upper())
    plt.axhline(0, lw=1, color="k")
    plt.xlabel("horyzont [dni]"); plt.ylabel("Spadek MAE po kalibracji [°C]")
    plt.title("Zysk kalibracji (raw → calibrated)")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "plots" / "calibration_gain_vs_horizon.png", dpi=150); plt.close()

    params = {
        "features": str(args.features),
        "outdir": str(outdir),
        "horizons": args.horizons,
        "xgb_params": {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "reg_lambda": args.reg_lambda,
            "min_child_weight": args.min_child_weight,
            "gamma": args.gamma,
            "reg_alpha": args.reg_alpha,
            "early_stopping_rounds": args.early_stopping_rounds,
            "tree_method": "hist",
            "random_state": 42,
        },
        "restrict_to_meta": bool(args.restrict_to_meta),
        "meta_csv": args.meta_csv,
        "runtime_sec": round(time.time() - t0, 2),
    }

    with open(outdir / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    with open(outdir / "runtime.txt", "w", encoding="utf-8") as f:
        f.write(str(params["runtime_sec"]))

    print("Gotowe. Artefakty zapisane w:", args.outdir)


if __name__ == "__main__":
    main()
