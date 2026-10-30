# -*- coding: utf-8 -*-
"""
STEP 5. 원인 검증 + A/B 테스트 설계

1) 여름 잔존 하락 분해
   - 봄(1~5월) 데이터로 학습한 모델로 각 월 코호트의 '72시간 행동 기준 예상 잔존'을 계산
   - 실제 잔존 하락 중 '초반 행동이 나빠져서' 설명되는 몫과, 설명되지 않는 몫(외부 요인)을 나눔
   - 부트스트랩으로 신뢰구간 계산
2) '오래 했는데 안 크는' 유저: 72시간 플레이 시간이 비슷한 유저끼리 레벨에 따른 잔존 비교
3) A/B 테스트 설계: 위험 중간층 대상 온보딩 개입 시, 효과 크기별 필요 표본·기간

실행: python step5.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

BASE = (Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()).parent  # churn_project 폴더
OUT_DIR = BASE / "step5"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_END = pd.Timestamp("2008-06-01")
rng = np.random.default_rng(42)

log = []
def say(m=""):
    print(m); log.append(str(m))

df = pd.read_parquet(BASE / "step4" / "features_72h.parquet")
df["retained"] = 1 - df["churn30"]
df["month"] = df["first_seen"].dt.to_period("M").astype(str)
df = df[df["month"] <= "2008-07"]          # 8월은 표본이 작아(1주일치) 제외

num = ["first_sess_h", "n_sess_72h", "play_h_72h", "level_72h", "zones_72h", "guild_72h",
       "h_to_2nd", "returned_72h", "lv_per_h", "days_72h", "create_hour", "create_weekend"]
cat = ["race", "charclass"]
model = Pipeline([("pre", ColumnTransformer([("num", StandardScaler(), num),
                                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])),
                  ("m", LogisticRegression(max_iter=2000, C=0.5))])
train = df[df["first_seen"] < TRAIN_END]
model.fit(train[num + cat], train["churn30"])
df["pred_retained"] = 1 - model.predict_proba(df[num + cat])[:, 1]

# ------------------------------------------------------------------
# 1) 월별: 실제 잔존 vs 72시간 행동으로 예상한 잔존
# ------------------------------------------------------------------
say("## 1. 월별 실제 D30 잔존 vs 72시간 행동 기준 예상 잔존 (모델은 1~5월로 학습)")
mon = df.groupby("month").agg(
    n=("retained", "size"),
    actual=("retained", "mean"),
    expected=("pred_retained", "mean"),
    level_72h=("level_72h", "mean"),
    days_72h=("days_72h", "mean"),
    returned_72h=("returned_72h", "mean"),
    first_sess_h=("first_sess_h", "mean"),
)
mon["gap(actual-expected)"] = mon["actual"] - mon["expected"]
say(mon.round(3).to_string())
mon.to_csv(OUT_DIR / "monthly_actual_vs_expected.csv", encoding="utf-8-sig")

spring = df[df["month"].between("2008-03", "2008-05")]
summer = df[df["month"].between("2008-06", "2008-07")]

def decompose(sp, su):
    d_act = su["retained"].mean() - sp["retained"].mean()
    d_exp = su["pred_retained"].mean() - sp["pred_retained"].mean()
    return d_act, d_exp

d_act, d_exp = decompose(spring, summer)
boots = []
for _ in range(1000):
    sp = spring.sample(len(spring), replace=True, random_state=rng.integers(1e9))
    su = summer.sample(len(summer), replace=True, random_state=rng.integers(1e9))
    boots.append(decompose(sp, su))
boots = np.array(boots)
ci = lambda a: (np.percentile(a, 2.5), np.percentile(a, 97.5))
say(f"\n봄(3~5월) -> 여름(6~7월) D30 잔존 변화")
say(f"  실제 변화      : {d_act*100:+.1f}%p  (95% CI {ci(boots[:,0])[0]*100:+.1f} ~ {ci(boots[:,0])[1]*100:+.1f})")
say(f"  행동으로 설명  : {d_exp*100:+.1f}%p  (95% CI {ci(boots[:,1])[0]*100:+.1f} ~ {ci(boots[:,1])[1]*100:+.1f})")
say(f"  설명 안 되는 몫: {(d_act-d_exp)*100:+.1f}%p")
if d_act < 0:
    say(f"  -> 하락의 약 {d_exp/d_act:.0%}가 '초반 72시간 참여가 약해진 것'으로 설명됨")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(mon.index, mon["actual"] * 100, marker="o", label="actual D30 retention")
ax.plot(mon.index, mon["expected"] * 100, marker="s", ls="--", label="expected from 72h behavior")
ax.axvline(4.5, color="gray", lw=0.8); ax.text(4.55, ax.get_ylim()[1] * 0.95, "train | test", fontsize=8)
ax.set_ylabel("%"); ax.set_title("Actual vs behavior-expected retention by cohort month"); ax.legend()
fig.tight_layout(); fig.savefig(OUT_DIR / "01_actual_vs_expected.png", dpi=130); plt.close(fig)

# ------------------------------------------------------------------
# 2) '오래 했는데 안 크는' 유저
# ------------------------------------------------------------------
say("\n## 2. 72시간 플레이 시간이 비슷한 유저끼리, 레벨에 따른 D30 잔존")
d2 = df[df["play_h_72h"] >= 1].copy()
d2["play_band"] = pd.cut(d2["play_h_72h"], [1, 2, 4, 8, 999], labels=["1-2h", "2-4h", "4-8h", "8h+"], include_lowest=True)
# 같은 플레이 시간 구간 안에서 레벨을 3등분 (하/중/상)
d2["level_tier"] = d2.groupby("play_band", observed=True)["level_72h"].transform(
    lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"]))
t2 = d2.pivot_table(index="play_band", columns="level_tier", values="retained", aggfunc="mean", observed=True)
n2 = d2.pivot_table(index="play_band", columns="level_tier", values="retained", aggfunc="size", observed=True)
say("D30 잔존율\n" + t2.round(3).to_string())
say("인원\n" + n2.to_string())
say("-> 같은 시간을 플레이해도 레벨이 낮은(성장이 막힌) 유저의 잔존이 낮다면 '막힘' 신호")
t2.to_csv(OUT_DIR / "stuck_users.csv", encoding="utf-8-sig")

# ------------------------------------------------------------------
# 3) A/B 테스트 설계
#    대상: 위험 점수 중간층 (예상 잔존 하위 30%~70% 구간 = 대략 D3~D6)
# ------------------------------------------------------------------
say("\n## 3. A/B 테스트 설계 (온보딩 개입: 예) 72시간 내 재접속 보상 / 레벨 10까지 가이드)")
q30, q70 = df["pred_retained"].quantile([0.3, 0.7])
mid = df[df["pred_retained"].between(q30, q70)]
p0 = mid["retained"].mean()
daily_new = df.groupby(df["first_seen"].dt.normalize()).size().mean()
daily_target = daily_new * len(mid) / len(df)
say(f"대상: 예상 잔존 {q30:.1%} ~ {q70:.1%} 구간 (전체 신규의 {len(mid)/len(df):.0%}) / 기준 D30 잔존 {p0:.1%}")
say(f"이 서버 기준 대상 유입: 하루 약 {daily_target:.1f}명")

def n_per_arm(p0, lift, alpha=0.05, power=0.8):
    p1 = p0 + lift
    z_a, z_b = norm.ppf(1 - alpha / 2), norm.ppf(power)
    pbar = (p0 + p1) / 2
    return ((z_a * np.sqrt(2 * pbar * (1 - pbar)) + z_b * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2) / lift ** 2

rows = []
for lift in [0.02, 0.03, 0.05, 0.08]:
    n = n_per_arm(p0, lift)
    days = 2 * n / daily_target + 30          # 모집 기간 + 30일 관찰
    rows.append({"효과(%p)": lift * 100, "목표 잔존": p0 + lift, "그룹당 필요 인원": int(np.ceil(n)),
                 "총 필요 인원": int(np.ceil(2 * n)), "이 서버만으로 걸리는 기간(일)": int(days)})
ab = pd.DataFrame(rows)
say(ab.round(3).to_string(index=False))
ab.to_csv(OUT_DIR / "ab_sample_size.csv", index=False, encoding="utf-8-sig")
say("-> 서버 하나의 유입으로는 작은 효과를 검증하기 어려움: 여러 서버를 묶거나, 1차 지표를 D30 대신 D7·72시간 재접속처럼 더 빨리 보이는 지표로 두는 설계가 필요")

(OUT_DIR / "summary.txt").write_text("\n".join(log), encoding="utf-8")
print(f"\n저장 완료: {OUT_DIR}")
