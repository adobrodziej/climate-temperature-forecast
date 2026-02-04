#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Budowa globalnego zbioru cech dla prognozy t_mean (horyzont 1–7 dni) w ujęciu wielostacyjnym.
- Wczytuje dzienne dane stacyjne (parquet, partycjonowane po stacji lub pliki zbiorcze)
- Ujednolica typy, uzupełnia braki (ffill per station)
- Dodaje cechy czasowe (DOY, DOW, month, weekend, Fourier K)
- Dodaje klimatologię DOY×stacja (obliczoną wyłącznie na train ≤ train_end)
- Dodaje lagi i role dla t_mean
- Tworzy targety y_h1..y_h7 (shift -h)
- Oznacza split ∈ {train, val, test} wg dat granicznych
- Zapisuje features_global.parquet i params.json


Uruchomienie:
python build_global_features.py `
  --data_dir "E:\Final_mgr\Database\Dobowe_klimat\processed\klimat_all_clean_ds" `
  --meta_csv "E:\Final_mgr\Database\Dobowe_klimat\processed\reports\global_xgb\station_meta_clean.csv" `
  --out "E:\Final_mgr\Database\Dobowe_klimat\processed\reports\global_xgb\features_global_geoFix.parquet" `
  --train_end 2015-12-31 `
  --val_end 2019-12-31      
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True,
                   help="Katalog z danymi dziennymi (parquet), może być partycjonowany.")
    p.add_argument("--meta_csv", type=str, required=True,
                   help="CSV z metadanymi stacji: station_id, lat, lon, elev_m, region.")
    p.add_argument("--out", type=str, required=True, help="Ścieżka wyjściowa features_global.parquet")
    p.add_argument("--train_end", type=str, default="2015-12-31")
    p.add_argument("--val_end", type=str, default="2019-12-31")
    p.add_argument("--fourier_k", type=int, default=2)
    p.add_argument("--max_lag", type=int, default=14)
    p.add_argument("--roll_windows", type=int, nargs="*", default=[3, 7, 14])
    p.add_argument("--target", type=str, default="t_mean")
    p.add_argument("--horizons", type=int, nargs="*", default=[1,2,3,4,5,6,7])
    return p.parse_args()


def find_parquet_files(data_dir: Path):
    return sorted(list(data_dir.rglob("*.parquet")))


def ensure_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "station_id" in df.columns:
        df["station_id"] = df["station_id"].astype(str)
    return df


def add_time_features(df: pd.DataFrame, fourier_k: int) -> pd.DataFrame:
    df = df.copy()
    df["doy_raw"] = df["date"].dt.dayofyear
    # DOY zredukowany do 365 (eliminuje 29 lutego jako 366 ⇒ 365)
    df["doy_365"] = df["doy_raw"].clip(upper=365)
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["weekend"] = (df["dow"].isin([5,6])).astype(np.int8)
    # Fourier dla sezonowości rocznej (okres ~365.25)
    two_pi_over_T = 2.0 * math.pi / 365.25
    for k in range(1, fourier_k + 1):
        ang = two_pi_over_T * k * df["doy_365"].astype(float)
        df[f"fourier_sin_{k}"] = np.sin(ang).astype(np.float32)
        df[f"fourier_cos_{k}"] = np.cos(ang).astype(np.float32)
    return df


def add_station_meta(df: pd.DataFrame, meta_csv: Path) -> pd.DataFrame:
    meta = pd.read_csv(meta_csv)
    meta["station_id"] = meta["station_id"].astype(str)
    return df.merge(meta, on="station_id", how="left")


def compute_station_climatology(train_df: pd.DataFrame, target: str) -> pd.DataFrame:
    clm = (
        train_df.groupby(["station_id", "doy_365"], as_index=False)[target]
        .mean()
        .rename(columns={target: f"clim_{target}"})
    )
    return clm


def add_lags_and_rolls(df: pd.DataFrame, target: str, max_lag: int, roll_windows: list[int]) -> pd.DataFrame:
    df = df.sort_values(["station_id", "date"]).copy()
    grp = df.groupby("station_id", group_keys=False)
    # Lagi 1..max_lag
    for L in range(1, max_lag + 1):
        df[f"{target}_lag_{L}"] = grp[target].shift(L)
    # Role (rolling mean)
    for w in roll_windows:
        df[f"{target}_roll_{w}"] = (
            grp[target].rolling(window=w, min_periods=max(1, w//2)).mean().reset_index(level=0, drop=True)
        )
    return df


def add_targets(df: pd.DataFrame, target: str, horizons: list[int]) -> pd.DataFrame:
    df = df.sort_values(["station_id", "date"]).copy()
    grp = df.groupby("station_id", group_keys=False)
    for h in horizons:
        df[f"y_h{h}"] = grp[target].shift(-h)
    return df


def mark_splits(df: pd.DataFrame, train_end: str, val_end: str) -> pd.DataFrame:
    train_end = pd.to_datetime(train_end)
    val_end   = pd.to_datetime(val_end)

    cond_train = df["date"] <= train_end
    cond_val   = (df["date"] > train_end) & (df["date"] <= val_end)
    cond_test  = df["date"] > val_end

    split = np.select([cond_train, cond_val, cond_test], ["train", "val", "test"])
    df["split"] = pd.Series(split, index=df.index, dtype="category")
    return df


def forward_fill_by_station(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.sort_values(["station_id", "date"]).copy()
    df[cols] = df.groupby("station_id")[cols].ffill()
    return df


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    meta_csv = Path(args.meta_csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = find_parquet_files(data_dir)
    if not files:
        raise FileNotFoundError(f"Nie znaleziono plików parquet w {data_dir}")

    dfs = []
    for fp in tqdm(files, desc="Wczytywanie parquet"):
        df = pd.read_parquet(fp)
        df = ensure_dtypes(df)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Kolumny oczekiwane minimalnie
    required = ["date", "station_id", "t_mean"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Brak wymaganej kolumny: {c}")

    # Uzupełnianie braków per station (ffill) dla podstawowych zmiennych meteorologicznych
    numeric_cols = [c for c in df.columns if c not in ["date", "station_id", "station_name"]]
    df = forward_fill_by_station(df, [c for c in numeric_cols if df[c].dtype.kind in "ifu"])  # int/float/uint

    # Cechy czasowe
    df = add_time_features(df, fourier_k=args.fourier_k)

    # Metadane stacji
    df = add_station_meta(df, meta_csv)

    # Splity
    df = mark_splits(df, args.train_end, args.val_end)

    # Klimatologia z TRAIN ONLY
    clm = compute_station_climatology(df[df["split"] == "train"], args.target)
    df = df.merge(clm, on=["station_id", "doy_365"], how="left")
    # === Klimatologia dla dnia prognozy t+h (h=1..7) ===
    # Tworzymy DOY dla daty prognozy i dokładamy odpowiadającą klimatologię.
    for h in args.horizons:
        doy_h = f"doy_365_h{h}"
        df[doy_h] = (df["date"] + pd.to_timedelta(h, unit="D")).dt.dayofyear.clip(upper=365)
        clm_h = clm.rename(columns={
            "doy_365": doy_h,
            f"clim_{args.target}": f"clim_{args.target}_h{h}",
        })
        df = df.merge(clm_h, on=["station_id", doy_h], how="left")


        # porządek: nie zostawiamy pomocniczych kolumn DOY dla h
        helper_doys = [f"doy_365_h{h}" for h in args.horizons if f"doy_365_h{h}" in df.columns]
        if helper_doys:
            df = df.drop(columns=helper_doys)

    # Lagi i role dla targetu
    df = add_lags_and_rolls(df, args.target, args.max_lag, args.roll_windows)

    # Targety dla horyzontów
    df = add_targets(df, args.target, args.horizons)

    # Porządki: lekkie rzutowanie typów
    int8_cols = ["dow", "month", "weekend"]
    for c in int8_cols:
        if c in df.columns:
            df[c] = df[c].astype("int8")
    # station_id jako kategoria
    df["station_id"] = df["station_id"].astype("category")

    # Zapis danych i parametrów
    df.to_parquet(out_path, index=False)

    params = {
        "data_dir": str(data_dir),
        "meta_csv": str(meta_csv),
        "out": str(out_path),
        "train_end": args.train_end,
        "val_end": args.val_end,
        "fourier_k": args.fourier_k,
        "max_lag": args.max_lag,
        "roll_windows": args.roll_windows,
        "target": args.target,
        "horizons": args.horizons,
        "row_count": int(len(df)),
        "station_count": int(df["station_id"].nunique()),
    }
    with open(out_path.with_suffix(".params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    print(f"Zapisano: {out_path} oraz {out_path.with_suffix('.params.json')}")


if __name__ == "__main__":
    main()
