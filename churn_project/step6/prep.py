# -*- coding: utf-8 -*-
"""STEP 6 준비: 원본 CSV -> STEP1~3과 같은 규칙으로 정제·생애 분리·신규 코호트 재구성 (parquet 없이 pickle 저장)"""
import pandas as pd, numpy as np, time, os
from pathlib import Path
BASE = Path(os.environ["HOME"]) / "mnt/archive"
OUT = BASE / "churn_project/step6"
t0 = time.time()
chunks = []
for ch in pd.read_csv(BASE / "wowah_data.csv", skipinitialspace=True, chunksize=1_000_000):
    ch.columns = [c.strip().lower() for c in ch.columns]
    ch = ch.rename(columns={"class": "charclass"})
    ch["timestamp"] = pd.to_datetime(ch["timestamp"], format="%m/%d/%y %H:%M:%S")
    for c in ["race", "charclass", "zone"]:
        ch[c] = ch[c].astype(str).str.strip()
    ch["level"] = ch["level"].astype("int16"); ch["guild"] = ch["guild"].astype("int32"); ch["char"] = ch["char"].astype("int32")
    chunks.append(ch)
    print("chunk", len(chunks), round(time.time() - t0), "s", flush=True)
df = pd.concat(chunks, ignore_index=True); del chunks
for c in ["race", "charclass", "zone"]:
    df[c] = df[c].astype("category")
print("rows", len(df), flush=True)
df = df[~df.duplicated(subset=["char", "timestamp"])].sort_values(["char", "timestamp"]).reset_index(drop=True)
same = df["char"].eq(df["char"].shift())
rc = same & ((df["race"].astype(str) != df["race"].astype(str).shift()) | (df["charclass"].astype(str) != df["charclass"].astype(str).shift()))
lvd = df["level"] - df["level"].shift()
new_life = (~same) | rc | (same & (lvd <= -2))
idx = new_life.astype(int).groupby(df["char"]).cumsum() - 1
df["life_id"] = (df["char"].astype("int64") * 1000 + idx).astype("int64")
print("lives", df["life_id"].nunique(), round(time.time() - t0), "s", flush=True)
df.to_pickle(OUT / "clean.pkl")
print("saved", round(time.time() - t0), "s", flush=True)
