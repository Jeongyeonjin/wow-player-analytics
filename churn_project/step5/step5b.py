# -*- coding: utf-8 -*-
"""
STEP 5b. 여름 하락은 서버 전체 문제인가, 신규만의 문제인가?

비교: 월 M에 접속한 유저 중 다음 달(M+1)에도 접속한 비율 (월간 잔존)
 - 기존 유저: 데이터 시작 첫 주(1/1~1/7)에 이미 접속해 있던 캐릭터
     . 만렙(70) 유저 / 성장 중 유저(70 미만)로 나눠 봄
 - 신규 유저: STEP 3 코호트 중 월 M에 생성된 캐릭터
기존 유저도 여름에 같이 빠졌다면 -> 서버 전체 요인(콘텐츠 공백 등)
기존 유저는 그대로이고 신규만 빠졌다면 -> 신규 특화 요인
분석 기간은 STEP 3과 같이 10/7까지

실행: python step5b.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = (Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()).parent  # churn_project 폴더
OUT_DIR = BASE / "step5"
DATA_END = pd.Timestamp("2008-10-07 23:59:59")

log = []
def say(m=""):
    print(m); log.append(str(m))

lives = pd.read_parquet(BASE / "step2" / "lives.parquet")
sess = pd.read_parquet(BASE / "step2" / "sessions.parquet")
coh = pd.read_parquet(BASE / "step3" / "cohort.parquet")
sess = sess[sess["start"] <= DATA_END].copy()
sess["month"] = sess["start"].dt.to_period("M")

# 월별 접속 여부 + 그 달 최고 레벨
act = sess.groupby(["life_id", "month"]).agg(level=("level_end", "max")).reset_index()
active = set(zip(act["life_id"], act["month"]))

veteran_ids = set(lives.loc[lives["first_seen"] < pd.Timestamp("2008-01-08"), "life_id"])
coh = coh.copy(); coh["cmonth"] = coh["first_seen"].dt.to_period("M")

months = pd.period_range("2008-01", "2008-09", freq="M")   # M+1 이 9월까지 (10월은 7일까지라 제외)
rows = []
for m in months[:-1]:
    nxt = m + 1
    a = act[act["month"] == m]
    vet = a[a["life_id"].isin(veteran_ids)]
    for name, grp in [("veteran_lv70", vet[vet["level"] >= 70]),
                      ("veteran_under70", vet[vet["level"] < 70])]:
        kept = np.mean([(lid, nxt) in active for lid in grp["life_id"]]) if len(grp) else np.nan
        rows.append({"month": str(m), "group": name, "n": len(grp), "next_month_retention": kept})
    new_ids = coh.loc[coh["cmonth"] == m, "life_id"]
    kept = np.mean([(lid, nxt) in active for lid in new_ids]) if len(new_ids) else np.nan
    rows.append({"month": str(m), "group": "new_created_this_month", "n": len(new_ids), "next_month_retention": kept})

res = pd.DataFrame(rows)
piv = res.pivot(index="month", columns="group", values="next_month_retention")
npiv = res.pivot(index="month", columns="group", values="n")
say("## 월간 잔존 (월 M 접속자 중 M+1에도 접속한 비율)")
say(piv.round(3).to_string())
say("\n인원")
say(npiv.to_string())
res.to_csv(OUT_DIR / "veteran_vs_new_monthly.csv", index=False, encoding="utf-8-sig")

spring = [str(p) for p in pd.period_range("2008-03", "2008-05", freq="M")]
summer = [str(p) for p in pd.period_range("2008-06", "2008-07", freq="M")]
say("\n봄(3~5월) -> 여름(6~7월) 평균 월간 잔존 변화")
for col in piv.columns:
    sp, su = piv.loc[spring, col].mean(), piv.loc[summer, col].mean()
    say(f"  {col:24s}: {sp:.1%} -> {su:.1%} ({(su-sp)*100:+.1f}%p, 상대 {(su/sp-1):+.0%})")
say("-> 기존 유저도 비슷한 비율로 하락했다면 서버 전체 요인, 신규만 크게 하락했다면 신규 특화 요인")

# 일간 접속자(DAU) 추세도 같이
dau = sess.groupby(sess["start"].dt.normalize())["life_id"].nunique()
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
for col in piv.columns:
    axes[0].plot(piv.index, piv[col] * 100, marker="o", label=col)
axes[0].set_title("Next-month retention by group"); axes[0].set_ylabel("%"); axes[0].legend(fontsize=8)
axes[0].tick_params(axis="x", rotation=45)
axes[1].plot(dau.index, dau.values, lw=0.8); axes[1].plot(dau.rolling(14, center=True).median(), lw=2)
axes[1].set_title("Daily active characters (14d median)")
fig.tight_layout(); fig.savefig(OUT_DIR / "02_veteran_vs_new.png", dpi=130); plt.close(fig)

with open(OUT_DIR / "summary.txt", "a", encoding="utf-8") as f:
    f.write("\n\n## STEP 5b\n" + "\n".join(log))
print(f"\n저장 완료: {OUT_DIR}")
