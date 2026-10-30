# -*- coding: utf-8 -*-
"""
WoWAH 신규 캐릭터 이탈 퍼널 분석
퀘스트: 이탈 지점을 찾아 전환 개선안 제안하기

분석 흐름
  1) 로드 & 정제
  2) 캐릭터 ID 충돌 분리 (레벨이 크게 떨어지면 다른 캐릭터로 간주)
  3) 분석 대상 = 관측 기간 중 Lv1로 새로 등장한 캐릭터 (의심 코호트 제외)
  4) 퍼널: 생성 → 재접속 → Lv10 → … → Lv70
  5) 온보딩: N일 생존율, 첫 세션 길이별 잔존율
  6) 레벨별 이탈률(hazard) + 레벨당 체류시간 → '벽' 구간 찾기
  7) 이탈 직전 지역: 평소 세션 종료 지역 대비 몇 배 많이 끝나는지(lift)
  8) 길드 가입 / 직업별 차이
  9) summary.txt 로 핵심 수치 저장
"""

# %% ============================================================
# 0. 설정
# ===============================================================
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

SHOW_PLOTS = False
if not SHOW_PLOTS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path("D:/code/game/archive")
RAW_FILE = DATA_DIR / "wowah_data.csv"
ZONES_FILE = DATA_DIR / "zones.csv"
OUT_DIR = DATA_DIR / "funnel_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOT_MIN = 10            # 1행 ≈ 10분 접속
SESSION_GAP_MIN = 30         # 이보다 길게 끊기면 새 세션
LEVEL_DROP_SPLIT = 5         # 한 번에 5레벨 이상 떨어지면 ID 충돌 → 새 캐릭터
CHURN_DAYS = 30              # 데이터 종료 N일 전까지 접속이 없으면 이탈
OBS_DAYS = 60                # 생성 후 최소 관측 기간 (우측 절단 방지)
EDGE_DAYS = 7                # 데이터 시작 직후 N일 생성분 제외
MILESTONES = [10, 20, 30, 40, 50, 60, 70]
EXCLUDE_COHORT = ("2008-10-08", "2008-10-10")   # 의심 코호트 생성일 (None이면 제외 안 함)
MIN_REACHED = 50             # 레벨별 이탈률 계산 시 최소 도달 인원
MIN_ZONE_CHURNERS = 30       # 지역 lift 계산 시 최소 이탈 인원
LEVEL_BANDS = [0, 10, 20, 30, 40, 50, 60, 70, 80]

plt.rcParams.update({
    "figure.figsize": (11, 5),
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

SUMMARY = []


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)
    SUMMARY.append(f"\n## {title}")


def note(text):
    print(text)
    SUMMARY.append(text)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def save_table(df, name):
    df.to_csv(OUT_DIR / f"{name}.csv", encoding="utf-8-sig")


# %% ============================================================
# 1. 로드 & 정제
# ===============================================================
section("1. 데이터 로드")
df = pd.read_csv(RAW_FILE, low_memory=False)
df.columns = df.columns.str.strip()
for c in ["race", "charclass", "zone"]:
    df[c] = df[c].astype(str).str.strip()

ts = df["timestamp"].astype(str).str.strip()
parsed = pd.to_datetime(ts, format="%m/%d/%y %H:%M:%S", errors="coerce")
if parsed.isna().mean() > 0.01:
    parsed = pd.to_datetime(ts, errors="coerce")
df["timestamp"] = parsed
df = df.dropna(subset=["timestamp"])
df["level"] = df["level"].astype(int)
df["guild"] = pd.to_numeric(df["guild"], errors="coerce")
df = df.sort_values(["char", "timestamp"], kind="mergesort").reset_index(drop=True)

data_start, data_end = df["timestamp"].min(), df["timestamp"].max()
note(f"행 수 {len(df):,} / 캐릭터 ID {df['char'].nunique():,} / 기간 {data_start} ~ {data_end}")
print("guild 값 상위 5개 (길드 없음 표기 확인용):",
      df["guild"].value_counts(dropna=False).head(5).to_dict())

# %% ============================================================
# 2. ID 충돌 분리 → life_id
# ===============================================================
section("2. 캐릭터 ID 충돌 분리")
lvl_diff = df.groupby("char")["level"].diff()
split = (lvl_diff <= -LEVEL_DROP_SPLIT).astype(np.int16)
df["life"] = split.groupby(df["char"]).cumsum()
df["life_id"] = df["char"].astype(np.int64) * 1000 + df["life"]
note(f"분리된 지점 {int(split.sum()):,}개 → 분석 단위 캐릭터 {df['life_id'].nunique():,}개")

life = df.groupby("life_id").agg(
    char=("char", "first"),
    first_seen=("timestamp", "min"),
    last_seen=("timestamp", "max"),
    first_level=("level", "first"),
    max_level=("level", "max"),
    n_snap=("level", "size"),
    race=("race", "first"),
    charclass=("charclass", "first"),
)
life["churned"] = life["last_seen"] <= data_end - pd.Timedelta(days=CHURN_DAYS)

# %% ============================================================
# 3. 분석 대상: 신규 캐릭터 코호트
# ===============================================================
section("3. 분석 대상 (신규 캐릭터)")
mask = (
    (life["first_level"] <= 1)
    & (life["first_seen"] >= data_start + pd.Timedelta(days=EDGE_DAYS))
    & (life["first_seen"] <= data_end - pd.Timedelta(days=OBS_DAYS))
)
excl = pd.Series(False, index=life.index)
if EXCLUDE_COHORT:
    s = pd.Timestamp(EXCLUDE_COHORT[0])
    e = pd.Timestamp(EXCLUDE_COHORT[1]) + pd.Timedelta(days=1)
    excl = (life["first_seen"] >= s) & (life["first_seen"] < e)

new = life[mask & ~excl].copy()
note(f"Lv1 신규 등장 & 관측 {OBS_DAYS}일 이상 확보: {int(mask.sum()):,}명")
note(f"의심 코호트({EXCLUDE_COHORT}) 제외: {int((mask & excl).sum()):,}명 → 최종 {len(new):,}명")
note(f"생성일 범위: {new['first_seen'].min().date()} ~ {new['first_seen'].max().date()}")
note(f"이탈 정의: {data_end.date()} 기준 {CHURN_DAYS}일 이상 미접속 → 이탈률 {new['churned'].mean():.1%}")

d = df[df["life_id"].isin(new.index)].copy()
del df, lvl_diff, split

# 세션
gap = d.groupby("life_id")["timestamp"].diff()
d["new_sess"] = gap.isna() | (gap > pd.Timedelta(minutes=SESSION_GAP_MIN))
d["sess"] = d.groupby("life_id")["new_sess"].cumsum().astype(int)
d["t_since"] = d["timestamp"] - d["life_id"].map(new["first_seen"])
d["has_guild"] = d["guild"].notna() & (d["guild"] >= 0)

new["n_sessions"] = d.groupby("life_id")["sess"].max()
new["first_sess_h"] = d[d["sess"] == 1].groupby("life_id").size() * SNAPSHOT_MIN / 60
new["active_days"] = (new["last_seen"] - new["first_seen"]).dt.total_seconds() / 86400
new["play_h"] = new["n_snap"] * SNAPSHOT_MIN / 60
new["guild_7d"] = (
    d[d["t_since"] <= pd.Timedelta(days=7)].groupby("life_id")["has_guild"].max()
    .reindex(new.index).fillna(False).astype(bool)
)
save_table(new, "cohort_lives")

# %% ============================================================
# 4. 퍼널
# ===============================================================
section("4. 퍼널 (생성 → 재접속 → 레벨 마일스톤)")
stages = [("Created", pd.Series(True, index=new.index)),
          ("Returned (2+ sessions)", new["n_sessions"] >= 2)]
stages += [(f"Lv {m}", new["max_level"] >= m) for m in MILESTONES]

rows, cum = [], pd.Series(True, index=new.index)
for name, cond in stages:
    cum = cum & cond
    rows.append((name, int(cum.sum())))
funnel = pd.DataFrame(rows, columns=["stage", "count"]).set_index("stage")
funnel["pct_of_created"] = funnel["count"] / funnel["count"].iloc[0]
funnel["step_conv"] = funnel["count"] / funnel["count"].shift(1)
funnel["step_drop_n"] = funnel["count"].shift(1) - funnel["count"]
print(funnel.to_string(float_format=lambda x: f"{x:.3f}"))
save_table(funnel, "funnel")

worst = funnel["step_conv"].iloc[1:].idxmin()
biggest = funnel["step_drop_n"].iloc[1:].idxmax()
note(f"전환율이 가장 낮은 단계: '{worst}' ({funnel.loc[worst, 'step_conv']:.1%})")
note(f"이탈 인원이 가장 많은 단계: '{biggest}' ({int(funnel.loc[biggest, 'step_drop_n']):,}명 이탈)")

fig, ax = plt.subplots(figsize=(10, 6))
y = np.arange(len(funnel))[::-1]
ax.barh(y, funnel["pct_of_created"] * 100, color="#4C72B0")
ax.set_yticks(y)
ax.set_yticklabels(funnel.index)
for yi, (_, r) in zip(y, funnel.iterrows()):
    conv = "" if pd.isna(r["step_conv"]) else f"  (step {r['step_conv']:.0%})"
    ax.text(r["pct_of_created"] * 100 + 0.5, yi, f"{r['pct_of_created']:.1%} n={int(r['count']):,}{conv}",
            va="center", fontsize=9)
ax.set_xlabel("% of created characters")
ax.set_title("New character funnel")
ax.set_xlim(0, 115)
save(fig, "01_funnel")

# %% ============================================================
# 5. 온보딩
# ===============================================================
section("5. 온보딩 (초기 이탈)")
for n in [1, 3, 7, 14, 30]:
    note(f"생성 후 {n}일 이후에도 접속한 비율: {(new['active_days'] >= n).mean():.1%}")
note(f"첫 세션만 하고 다시 안 온 비율: {(new['n_sessions'] == 1).mean():.1%}")

days = np.arange(0, OBS_DAYS + 1)
surv = [(new["active_days"] >= x).mean() for x in days]
fig, ax = plt.subplots()
ax.plot(days, np.array(surv) * 100, color="#C44E52")
ax.set_xlabel("Days since creation")
ax.set_ylabel("% still active after N days")
ax.set_title("Survival curve of new characters")
save(fig, "02_survival")

bins = [0, 0.5, 1, 2, 4, np.inf]
labels = ["<30m", "30m-1h", "1-2h", "2-4h", "4h+"]
new["first_sess_bucket"] = pd.cut(new["first_sess_h"], bins=bins, labels=labels, right=False)
fs = new.groupby("first_sess_bucket", observed=True).agg(
    n=("n_snap", "size"),
    d7_rate=("active_days", lambda s: (s >= 7).mean()),
    lv10_rate=("max_level", lambda s: (s >= 10).mean()),
)
print(fs.to_string(float_format=lambda x: f"{x:.3f}"))
save_table(fs, "first_session_vs_retention")
note("첫 세션 길이별 7일 잔존율: " + ", ".join(f"{i} {r.d7_rate:.0%}" for i, r in fs.iterrows()))

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(fs))
ax.bar(x - 0.2, fs["d7_rate"] * 100, width=0.4, label="Active after 7 days")
ax.bar(x + 0.2, fs["lv10_rate"] * 100, width=0.4, label="Reached Lv10")
ax.set_xticks(x)
ax.set_xticklabels(fs.index.astype(str))
ax.set_xlabel("First session length")
ax.set_ylabel("%")
ax.set_title("First session length vs retention")
ax.legend()
save(fig, "03_first_session")

# %% ============================================================
# 6. 레벨별 이탈률 + 레벨당 체류시간
# ===============================================================
section("6. 레벨별 이탈률 (어느 레벨에서 멈추고 떠나는가)")
lt = d.groupby(["life_id", "level"]).size().rename("snaps").reset_index()
lt["hours"] = lt["snaps"] * SNAPSHOT_MIN / 60
lt["max_level"] = lt["life_id"].map(new["max_level"])
med_h = lt[lt["level"] < lt["max_level"]].groupby("level")["hours"].median()

haz_rows = []
for L in range(1, 70):
    reached = int((new["max_level"] >= L).sum())
    churn_at = int(((new["max_level"] == L) & new["churned"]).sum())
    haz_rows.append((L, reached, churn_at))
haz = pd.DataFrame(haz_rows, columns=["level", "reached", "churned_here"]).set_index("level")
haz["churn_rate"] = haz["churned_here"] / haz["reached"]
haz["median_hours_to_pass"] = med_h
haz = haz[haz["reached"] >= MIN_REACHED]
save_table(haz, "level_churn")

top = haz.drop(index=1, errors="ignore").sort_values("churn_rate", ascending=False).head(10)
print("이탈률 상위 레벨 (Lv1 제외):")
print(top.to_string(float_format=lambda x: f"{x:.3f}"))
if 1 in haz.index:
    note(f"Lv1에서 이탈: {haz.loc[1, 'churn_rate']:.1%} (온보딩 이탈)")
note("이탈률 상위 레벨: " + ", ".join(f"Lv{i} {r.churn_rate:.1%}" for i, r in top.head(5).iterrows()))
corr = haz.drop(index=1, errors="ignore")[["churn_rate", "median_hours_to_pass"]].corr().iloc[0, 1]
note(f"레벨별 이탈률 vs 해당 레벨 통과 소요시간 상관계수: {corr:.2f}")

band_rows = []
for lo, hi in zip(LEVEL_BANDS[:-1], LEVEL_BANDS[1:]):
    if hi > 70:
        break
    entered = (new["max_level"] > lo) if lo > 0 else pd.Series(True, index=new.index)
    left = entered & (new["max_level"] <= hi) & (new["max_level"] < 70) & new["churned"]
    band_rows.append((f"{lo + 1}-{hi}", int(entered.sum()), int(left.sum())))
band = pd.DataFrame(band_rows, columns=["band", "entered", "churned_in_band"]).set_index("band")
band["churn_rate"] = band["churned_in_band"] / band["entered"]
print(band.to_string(float_format=lambda x: f"{x:.3f}"))
save_table(band, "band_churn")

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(haz.index, haz["churn_rate"] * 100, color="#C44E52", alpha=0.8, label="Churn rate at level")
ax.set_xlabel("Level")
ax.set_ylabel("Churn rate at level (%)")
ax.set_xticks(list(range(0, 71, 5)))
ax2 = ax.twinx()
ax2.plot(haz.index, haz["median_hours_to_pass"], color="#4C72B0", marker=".", label="Median hours to pass")
ax2.set_ylabel("Median hours spent to pass level")
ax2.grid(False)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper right")
ax.set_title("Where players stop: churn rate and time cost per level")
save(fig, "04_level_churn")

# %% ============================================================
# 7. 이탈 직전 지역 (lift)
# ===============================================================
section("7. 이탈 직전 지역")
band_labels = [f"{a + 1}-{b}" for a, b in zip(LEVEL_BANDS[:-1], LEVEL_BANDS[1:])]
d["band"] = pd.cut(d["level"], bins=LEVEL_BANDS, labels=band_labels)

is_sess_end = d.groupby(["life_id", "sess"]).cumcount(ascending=False) == 0
sess_end = d[is_sess_end]
final = sess_end.groupby("life_id").tail(1)
churn_final = final[final["life_id"].map(new["churned"]) & (final["life_id"].map(new["max_level"]) < 70)]
baseline = sess_end[~sess_end.index.isin(final.index)]   # 다시 돌아온 세션의 종료 지점

churn_cnt = churn_final.groupby(["band", "zone"], observed=True).size()
churn_share = churn_cnt.div(churn_final.groupby("band", observed=True).size(), level="band")
base_share = (baseline.groupby(["band", "zone"], observed=True).size()
              .div(baseline.groupby("band", observed=True).size(), level="band"))
zl = pd.DataFrame({"churn_n": churn_cnt, "churn_share": churn_share, "base_share": base_share})
zl = zl.dropna(subset=["churn_n"])
zl["base_share"] = zl["base_share"].fillna(0)
zl["lift"] = zl["churn_share"] / zl["base_share"].replace(0, np.nan)
zl = zl[zl["churn_n"] >= MIN_ZONE_CHURNERS].sort_values("lift", ascending=False)

try:
    zones = pd.read_csv(ZONES_FILE)
    zones.columns = zones.columns.str.strip()
    name_col = [c for c in zones.columns if "name" in c.lower()][0]
    type_col = [c for c in zones.columns if c.lower() == "type"][0]
    zones[name_col] = zones[name_col].astype(str).str.strip()
    zmap = zones.drop_duplicates(name_col).set_index(name_col)[type_col]
    zl["zone_type"] = zl.index.get_level_values("zone").map(zmap)
except Exception as ex:
    print(f"(zones.csv 조인 생략: {ex})")

save_table(zl, "zone_churn_lift")
print("이탈 직전 지역 lift 상위 15 (평소 세션 종료 대비 몇 배):")
print(zl.head(15).to_string(float_format=lambda x: f"{x:.3f}"))
note("lift 상위 지역: " + ", ".join(
    f"{z}({b}) x{r.lift:.1f}, n={int(r.churn_n)}" for (b, z), r in zl.head(5).iterrows()))

topz = zl.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(10, 6))
yy = np.arange(len(topz))
ax.barh(yy, topz["lift"], color="#DD8452")
ax.set_yticks(yy)
ax.set_yticklabels([f"{z} (Lv {b})" for b, z in topz.index])
ax.axvline(1, color="gray", ls="--")
ax.set_xlabel("Lift = share as final logout zone / share as normal session-end zone")
ax.set_title("Zones where players log out for the last time")
save(fig, "05_zone_lift")

# %% ============================================================
# 8. 길드 / 직업
# ===============================================================
section("8. 길드 가입 · 직업별 차이")
early = new[new["active_days"] >= 7]
g = early.groupby("guild_7d").agg(n=("n_snap", "size"),
                                  d30_rate=("active_days", lambda s: (s >= 30).mean()),
                                  lv20_rate=("max_level", lambda s: (s >= 20).mean()))
print("(7일 이상 남은 캐릭터 대상) 첫 7일 내 길드 가입 여부별:")
print(g.to_string(float_format=lambda x: f"{x:.3f}"))
save_table(g, "guild_effect")
if True in g.index and False in g.index:
    note(f"첫 7일 내 길드 가입 O: 30일 잔존 {g.loc[True, 'd30_rate']:.1%} / "
         f"X: {g.loc[False, 'd30_rate']:.1%} (상관관계일 뿐 인과 아님)")

cls = new.groupby("charclass").agg(
    n=("n_snap", "size"),
    churn_before_lv20=("max_level", lambda s: ((s < 20) & new.loc[s.index, "churned"]).mean()),
    lv60_rate=("max_level", lambda s: (s >= 60).mean()),
).sort_values("churn_before_lv20", ascending=False)
print(cls.to_string(float_format=lambda x: f"{x:.3f}"))
save_table(cls, "class_churn")

# %% ============================================================
# 9. 요약 저장
# ===============================================================
(OUT_DIR / "summary.txt").write_text("\n".join(SUMMARY), encoding="utf-8")
print(f"\n완료 → {OUT_DIR}")