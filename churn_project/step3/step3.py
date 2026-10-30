# -*- coding: utf-8 -*-
"""
STEP 3. 신규 유저 코호트 + 이상 코호트 탐지 + 퍼널 + 잔존 곡선

1) 신규 유저 정의
   - 레벨 1로 처음 등장한 캐릭터 생애
   - 데이터 시작 직후(1/1~1/7)에 처음 보인 캐릭터는 '원래 있던 캐릭터'일 수 있어 제외
   - 연말까지 최소 60일 이상 지켜볼 수 있게, 생성일이 11/1 이전인 캐릭터만
2) 이상 코호트 탐지: 하루 신규 수가 평소보다 비정상적으로 튄 날을 찾고,
   그날 신규가 '진짜 신규'처럼 보이는지 확인 (처음부터 길드 소속 비율 등)
3) 이탈 정의: 12/31 기준 30일 이상 미접속
4) 롤링 잔존율(Dn): 생성 후 n일 이후에도 한 번이라도 접속한 비율
5) 퍼널: 생성 -> 재접속 -> Lv10 -> Lv20 -> ... -> Lv70
6) 월별 코호트 잔존 히트맵

실행: python step3.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = (Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()).parent  # churn_project 폴더
OUT_DIR = BASE / "step3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_END = pd.Timestamp("2008-12-31 23:59:59")
WARMUP_END = pd.Timestamp("2008-01-08")      # 이 전에 처음 보인 캐릭터는 기존 캐릭터일 수 있음
CREATE_CUTOFF = pd.Timestamp("2008-11-01")   # 최소 60일 관측 확보
CHURN_DAYS = 30
SPIKE_Z = 5                                  # 하루 신규 수 z-점수가 이 이상이면 '이상 급증'

log = []
def say(m=""):
    print(m); log.append(str(m))

lives = pd.read_parquet(BASE / "step2" / "lives.parquet")
sess = pd.read_parquet(BASE / "step2" / "sessions.parquet")

# ------------------------------------------------------------------
# 1) 신규 유저 후보
# ------------------------------------------------------------------
say("## 1. 신규 유저 정의")
new = lives[lives["first_level"] == 1].copy()
say(f"레벨 1로 처음 등장: {len(new):,}")
new = new[new["first_seen"] >= WARMUP_END]
say(f"데이터 시작 첫 주(1/1~1/7) 등장 제외 후: {len(new):,}")
new = new[new["first_seen"] < CREATE_CUTOFF]
say(f"생성일 11/1 이전(60일+ 관측) : {len(new):,}")
new["create_date"] = new["first_seen"].dt.normalize()

# ------------------------------------------------------------------
# 2) 이상 코호트 탐지
#    하루 신규 수를 28일 이동 중앙값과 비교 (중앙값·MAD 기반이라 급증에 덜 휘둘림)
# ------------------------------------------------------------------
say("\n## 2. 이상 코호트 탐지 (하루 신규 수 급증)")
daily = new.groupby("create_date").size()
daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"), fill_value=0)
med = daily.rolling(28, center=True, min_periods=7).median()
mad = (daily - med).abs().rolling(28, center=True, min_periods=7).median()
z = (daily - med) / (1.4826 * mad.replace(0, np.nan))
spike_days = z[z > SPIKE_Z].index
for d in spike_days:
    say(f"  {d.date()}: 신규 {daily[d]:,}명 (평소 중앙값 {med[d]:.0f}명, z={z[d]:.1f})")

# 급증일 신규가 '진짜 신규'처럼 보이는지 비교
new["first_in_guild"] = new["first_guild"] >= 0
new["is_spike"] = new["create_date"].isin(spike_days)
cmp = new.groupby("is_spike").agg(
    n=("life_id", "size"),
    first_in_guild=("first_in_guild", "mean"),
    one_session=("n_sessions", lambda s: (s == 1).mean()),
    median_play_h=("play_hours", "median"),
)
cmp.index = cmp.index.map({False: "평소 신규", True: "급증일 신규"})
say("\n급증일 신규 vs 평소 신규")
say(cmp.round(3).to_string())
say("-> 레벨 1 신규인데 처음부터 길드 소속 비율이 훨씬 높다면, 진짜 신규가 아니라 기존 캐릭터가 새 ID로 다시 매겨졌을 가능성")
cmp.to_csv(OUT_DIR / "spike_vs_normal.csv", encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(daily.index, daily.values, lw=1, label="daily new characters")
ax.plot(med.index, med.values, lw=1, ls="--", label="28d rolling median")
ax.scatter(spike_days, daily[spike_days], color="red", zorder=3, label="spike")
ax.set_title("Daily new characters"); ax.legend(); fig.tight_layout()
fig.savefig(OUT_DIR / "01_daily_new.png", dpi=130); plt.close(fig)

coh = new[~new["is_spike"]].copy()
say(f"\n급증일 제외 최종 신규 코호트: {len(coh):,}명 (제외 {int(new['is_spike'].sum()):,}명)")

# ------------------------------------------------------------------
# 3) 이탈 정의 + 롤링 잔존율
# ------------------------------------------------------------------
say(f"\n## 3. 이탈 / 잔존 (이탈 = 12/31 기준 {CHURN_DAYS}일 이상 미접속)")
coh["life_days"] = (coh["last_seen"] - coh["first_seen"]).dt.total_seconds() / 86400
coh["churned"] = (DATA_END - coh["last_seen"]).dt.days >= CHURN_DAYS
say(f"이탈률: {coh['churned'].mean():.1%}")
say(f"첫 세션만 하고 다시 안 온 비율: {(coh['n_sessions']==1).mean():.1%}")

ret_days = [1, 3, 7, 14, 30, 60]
ret = {f"D{d}": (coh["life_days"] >= d).mean() for d in ret_days}
say("롤링 잔존율 (생성 후 n일 이후에도 접속한 비율): " + ", ".join(f"{k} {v:.1%}" for k, v in ret.items()))
pd.Series(ret).rename("rolling_retention").to_csv(OUT_DIR / "retention.csv", encoding="utf-8-sig")

# ------------------------------------------------------------------
# 4) 퍼널
# ------------------------------------------------------------------
say("\n## 4. 퍼널")
stages = [("Created", np.ones(len(coh), bool)),
          ("Returned (2+ sessions)", coh["n_sessions"] >= 2)]
for lv in [10, 20, 30, 40, 50, 60, 70]:
    stages.append((f"Lv {lv}", coh["max_level"] >= lv))
rows, prev = [], None
for name, mask in stages:
    n = int(np.sum(mask))
    rows.append({"stage": name, "count": n, "pct_of_created": n / len(coh),
                 "step_conv": (n / prev) if prev else np.nan,
                 "step_drop": (prev - n) if prev else np.nan})
    prev = n
funnel = pd.DataFrame(rows)
say(funnel.round(3).to_string(index=False))
funnel.to_csv(OUT_DIR / "funnel.csv", index=False, encoding="utf-8-sig")
worst = funnel.iloc[1:].sort_values("step_conv").iloc[0]
say(f"전환율이 가장 낮은 단계: {worst['stage']} ({worst['step_conv']:.1%})")

fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(funnel["stage"][::-1], funnel["pct_of_created"][::-1] * 100)
for i, v in enumerate(funnel["pct_of_created"][::-1] * 100):
    ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
ax.set_xlabel("% of created"); ax.set_title("New character funnel"); fig.tight_layout()
fig.savefig(OUT_DIR / "02_funnel.png", dpi=130); plt.close(fig)

# ------------------------------------------------------------------
# 5) 월별 코호트 잔존 히트맵 (생성 후 n주차에도 남아 있는 비율)
# ------------------------------------------------------------------
coh["cohort_month"] = coh["first_seen"].dt.to_period("M").astype(str)
weeks = list(range(0, 9))
heat = pd.DataFrame({f"W{w}": coh.groupby("cohort_month")["life_days"].apply(lambda s: (s >= 7 * w).mean())
                     for w in weeks})
heat["n"] = coh.groupby("cohort_month").size()
heat.to_csv(OUT_DIR / "cohort_heatmap.csv", encoding="utf-8-sig")
say("\n## 5. 월별 코호트 잔존 (n주차 이후에도 접속한 비율)")
say(heat.round(2).to_string())

fig, ax = plt.subplots(figsize=(9, 5))
im = ax.imshow(heat[[f"W{w}" for w in weeks]].values, aspect="auto", cmap="Blues", vmin=0, vmax=0.5)
ax.set_xticks(range(len(weeks))); ax.set_xticklabels([f"W{w}" for w in weeks])
ax.set_yticks(range(len(heat))); ax.set_yticklabels([f"{m} (n={n})" for m, n in zip(heat.index, heat["n"])])
for i in range(len(heat)):
    for j, w in enumerate(weeks):
        ax.text(j, i, f"{heat.iloc[i][f'W{w}']*100:.0f}", ha="center", va="center", fontsize=7)
ax.set_title("Cohort retention (% still active after week n)")
fig.colorbar(im); fig.tight_layout()
fig.savefig(OUT_DIR / "03_cohort_heatmap.png", dpi=130); plt.close(fig)

coh.to_parquet(OUT_DIR / "cohort.parquet", index=False)
(OUT_DIR / "summary.txt").write_text("\n".join(log), encoding="utf-8")
print(f"\n저장 완료: {OUT_DIR}")
