# -*- coding: utf-8 -*-
"""
STEP 6. 막힘 구간 분석

정의
 - 레벨 L 성장 시간 = 레벨 L 상태로 필드/던전에 머문 시간 (마을·전장·투기장 제외), 10분 스냅샷 x 10분
 - 막힘 구간 = 앞 3개 레벨 추세보다 성장 속도가 급격히 느려지는 레벨 (레벨 6 이상)
원인 구분
 - 원인1 경험치 공급 부족: 외부 데이터(레벨별 필요 경험치, 패치 2.3 이후)로 '경험치 1만당 시간'이 튀는지
                       모든 직업에서 같이 나타나는지
 - 원인2 스펙 부족: 빠른 유저-느린 유저 격차(p75/p25)가 벌어지는지, 특정 직업에서만 느려지는지
대상: STEP3과 같은 신규 코호트 (1/8~8/7 생성, 시작 지역 등장, 분석 종료 10/7)
"""
import pandas as pd, numpy as np, os
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = (Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()).parents[1]  # archive 폴더 (이 스크립트 기준 두 단계 위)
OUT = BASE / "churn_project/step6"
DATA_END = pd.Timestamp("2008-10-07 23:59:59")
CREATE_CUTOFF = DATA_END.normalize() - pd.Timedelta(days=60)
WARMUP_END = pd.Timestamp("2008-01-08")
START_ZONES = {"Durotar", "Mulgore", "Tirisfal Glades", "Eversong Woods"}
EXCLUDE_TYPES = {"City", "Battleground", "Arena"}
MIN_LV, MAX_LV = 6, 60
SNAP_H = 10 / 60
rng = np.random.default_rng(42)

# 레벨 L -> L+1 필요 경험치 (TBC 패치 2.3 이후, Wowpedia/WoWWiki 'Experience to level')
XP = dict(zip(range(1, 70), [400,900,1400,2100,2800,3600,4500,5400,6500,7600,8700,9800,11000,12300,13600,15000,16400,17800,19300,20800,
    22400,24000,25500,27200,28900,30500,32200,33900,36300,38800,41600,44600,48000,51400,55000,58700,62400,66200,70200,74300,
    78500,82800,87100,91600,96300,101000,105800,110700,115700,120900,126100,131500,137000,142500,148200,154000,159900,165800,172000,494000,
    574700,614400,650300,682300,710200,734100,753700,768900,779700]))

log = []
def say(m=""):
    print(m, flush=True); log.append(str(m))

df = pd.read_pickle(OUT / "clean.pkl")
zones = pd.read_csv(BASE / "zones.csv")
ztype = dict(zip(zones["Zone_Name"], zones["Type"]))

# ---------------- 1) 코호트 재구성 (STEP3과 동일 규칙) ----------------
first = df.groupby("life_id").first()
w = df[df["timestamp"] <= DATA_END]
gw = w.groupby("life_id")
lives = pd.DataFrame({"first_seen": first["timestamp"], "first_level": first["level"],
                      "first_zone": first["zone"].astype(str), "charclass": first["charclass"].astype(str),
                      "race": first["race"].astype(str)})
lives = lives.join(pd.DataFrame({"last_seen": gw["timestamp"].max(), "max_level": gw["level"].max()}), how="inner")
coh = lives[(lives["first_level"] == 1) & (lives["first_seen"] >= WARMUP_END) &
            (lives["first_seen"] < CREATE_CUTOFF) & (lives["first_zone"].isin(START_ZONES))].copy()
coh["churned"] = (DATA_END - coh["last_seen"]).dt.days >= 30
say(f"## 0. 코호트 재구성: {len(coh):,}명 / 이탈률 {coh['churned'].mean():.1%}  (STEP3: 7,040명 / 90.1%)")

# ---------------- 2) 레벨별 성장 시간 ----------------
rows = w[w["life_id"].isin(coh.index)].copy()
rows["ztype"] = rows["zone"].astype(str).map(ztype).fillna("Zone")
unmatched = rows.loc[~rows["zone"].astype(str).isin(ztype.keys()), "zone"].astype(str).value_counts()
say(f"zones.csv에 없는 지역 기록 {unmatched.sum():,}행 (필드로 간주): " + ", ".join(f"{k} {v}" for k, v in unmatched.head(5).items()))
city_share = rows["ztype"].isin(EXCLUDE_TYPES).mean()
say(f"마을·전장·투기장 기록 비율 {city_share:.1%} -> 성장 시간에서 제외")
grow = rows[~rows["ztype"].isin(EXCLUDE_TYPES)]
lt = grow.groupby(["life_id", "level"]).size().mul(SNAP_H).rename("hours").reset_index()
lt = lt.merge(coh[["max_level", "churned", "charclass"]], left_on="life_id", right_index=True)
# 레벨 L에 한 번도 안 찍힌 통과자는 0시간으로 포함
def level_table(sub_coh, sub_lt, min_n):
    out = []
    for L in range(MIN_LV, MAX_LV + 1):
        reached = sub_coh[sub_coh["max_level"] >= L]
        passed_ids = reached.index[reached["max_level"] > L]
        if len(passed_ids) < min_n:
            continue
        h = sub_lt[(sub_lt["level"] == L)].set_index("life_id")["hours"].reindex(passed_ids).fillna(0).values
        boots = [np.median(rng.choice(h, len(h))) for _ in range(200)]
        out.append({"level": L, "reached": len(reached), "passed": len(passed_ids),
                    "churn_here": int((reached["churned"] & (reached["max_level"] == L)).sum()),
                    "med_h": np.median(h), "ci_lo": np.percentile(boots, 2.5), "ci_hi": np.percentile(boots, 97.5),
                    "p25": np.percentile(h, 25), "p75": np.percentile(h, 75), "xp": XP[L]})
    t = pd.DataFrame(out).set_index("level")
    if t.empty:
        return t
    t["churn_rate"] = t["churn_here"] / t["reached"]
    t["h_per_10kxp"] = t["med_h"] / t["xp"] * 1e4
    t["spread"] = t["p75"] / t["p25"].replace(0, np.nan)
    trail = lambda s: s.shift(1).rolling(3, min_periods=2).median()
    t["slow_raw"] = t["med_h"] / trail(t["med_h"])
    t["slow_xp"] = t["h_per_10kxp"] / trail(t["h_per_10kxp"])
    t["spread_jump"] = t["spread"] / trail(t["spread"])
    return t

say("\n## 1. 전체 유저: 레벨별 성장 시간 / 경험치당 시간 / 이탈률")
T = level_table(coh, lt, 30)
T.round(3).to_csv(OUT / "level_table_all.csv", encoding="utf-8-sig")
# 레벨별 주 활동 지역 (위치 정보)
topz = grow.groupby(["level", "zone"], observed=True).size().reset_index(name="n")
topz = topz.sort_values("n", ascending=False).groupby("level").head(1).set_index("level")
tot = grow.groupby("level").size()
T["top_zone"] = topz["zone"].astype(str).reindex(T.index)
T["top_zone_share"] = (topz["n"] / tot).reindex(T.index)
show = T[["reached", "passed", "churn_rate", "med_h", "h_per_10kxp", "slow_raw", "slow_xp", "spread", "spread_jump", "top_zone"]]
say(show.round(2).to_string())

flag = T[(T["slow_xp"] >= 1.3) | (T["slow_raw"] >= 1.5)]
say("\n막힘 구간 후보 (경험치당 시간이 앞 3레벨 대비 1.3배 이상 or 성장 시간 1.5배 이상)")
say(flag[["passed", "med_h", "slow_raw", "slow_xp", "spread_jump", "churn_rate", "top_zone"]].round(2).to_string() if len(flag) else "  없음")

# ---------------- 3) 직업별 ----------------
say("\n## 2. 직업별: 경험치당 시간의 '자기 추세 대비' 급증 (통과자 20명 이상 레벨만)")
classes = sorted(coh["charclass"].unique())
heat, cls_rows = {}, []
for c in classes:
    tc = level_table(coh[coh["charclass"] == c], lt[lt["charclass"] == c], 20)
    if tc.empty:
        continue
    tc.round(3).to_csv(OUT / f"level_table_{c}.csv", encoding="utf-8-sig")
    heat[c] = tc["slow_xp"]
    for L, r in tc.iterrows():
        cls_rows.append({"class": c, "level": L, "passed": r["passed"], "med_h": r["med_h"], "slow_xp": r["slow_xp"],
                         "churn_rate": r["churn_rate"], "overall_slow_xp": T["slow_xp"].get(L, np.nan)})
H = pd.DataFrame(heat).T
H.round(2).to_csv(OUT / "class_slowdown_heatmap.csv", encoding="utf-8-sig")
C = pd.DataFrame(cls_rows)
C.round(3).to_csv(OUT / "class_level_long.csv", index=False, encoding="utf-8-sig")
say(f"직업별 분석 가능 최고 레벨: " + ", ".join(f"{c} {int(H.loc[c].dropna().index.max())}" for c in H.index))
spec = C[(C["slow_xp"] >= 1.4) & (C["overall_slow_xp"] < 1.2)].sort_values("slow_xp", ascending=False)
say("\n특정 직업에서만 튀는 막힘 (직업 1.4배 이상, 전체는 1.2배 미만)")
say(spec.round(2).to_string(index=False) if len(spec) else "  없음")
common = C[C["slow_xp"] >= 1.3].groupby("level")["class"].agg(["count", lambda s: ", ".join(s)])
common.columns = ["n_classes", "classes"]
say("\n여러 직업에서 같이 튀는 레벨 (직업 1.3배 이상인 직업 수)")
say(common.sort_values("n_classes", ascending=False).head(10).to_string())

# ---------------- 4) 그래프 ----------------
fig, ax = plt.subplots(2, 2, figsize=(15, 9))
a = ax[0, 0]
a.plot(T.index, T["med_h"], marker="o", ms=3, label="median hours at level (field/dungeon)")
a.fill_between(T.index, T["ci_lo"], T["ci_hi"], alpha=0.2)
a.set_xlabel("level"); a.set_ylabel("hours"); a.set_title("Time spent at each level")
a2 = a.twinx(); a2.plot(T.index, T["churn_rate"] * 100, color="tab:red", lw=1, label="churn here (%)"); a2.set_ylabel("% churned at level", color="tab:red")
a = ax[0, 1]
a.plot(T.index, T["h_per_10kxp"], marker="o", ms=3)
for L in flag.index:
    a.axvline(L, color="red", alpha=0.3)
a.set_title("Hours per 10k XP (external XP table) - red: flagged"); a.set_xlabel("level")
a = ax[1, 0]
a.plot(T.index, T["spread"], marker="o", ms=3); a.set_title("Fast-slow gap (p75 / p25 hours)"); a.set_xlabel("level")
a = ax[1, 1]
im = a.imshow(H.values, aspect="auto", cmap="RdBu_r", vmin=0.5, vmax=1.5)
a.set_yticks(range(len(H))); a.set_yticklabels(H.index)
step = max(1, len(H.columns) // 15)
a.set_xticks(range(0, len(H.columns), step)); a.set_xticklabels(H.columns[::step])
a.set_title("Class slowdown vs own trend (hours/XP ratio)"); fig.colorbar(im, ax=a)
fig.tight_layout(); fig.savefig(OUT / "01_stuck_overview.png", dpi=120); plt.close(fig)

(OUT / "summary.txt").write_text("\n".join(log), encoding="utf-8")
print("done")
