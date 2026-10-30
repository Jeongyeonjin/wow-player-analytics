# -*- coding: utf-8 -*-
"""
WoWAH (World of Warcraft Avatar History) 전체 데이터 EDA 스크립트

사용법
  1) 아래 CONFIG의 경로를 본인 환경에 맞게 수정
  2) python wowah_eda.py  (또는 VS Code / Jupyter에서 # %% 셀 단위로 실행)
  3) 결과: OUT_DIR 폴더에 그래프(png)와 요약 테이블(csv) 저장

필요 패키지: pandas, numpy, matplotlib
"""

# %% ============================================================
# 0. 설정
# ===============================================================
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

DATA_DIR = Path("D:/code/game/archive")      # 데이터 폴더
RAW_FILE = DATA_DIR / "wowah_data.csv"        # 원본 로그 (gzip이어도 자동 인식)
ZONES_FILE = DATA_DIR / "zones.csv"
LOCATIONS_FILE = DATA_DIR / "locations.csv"
OUT_DIR = Path("./eda_output")

SHOW_PLOTS = False          # Jupyter에서 그래프를 바로 보려면 True
SNAPSHOT_MIN = 10           # 관측 간격(분): 1행 ≈ 10분 접속
SESSION_GAP_MIN = 30        # 이 간격보다 길게 끊기면 새 세션으로 간주
CHURN_DAYS = 30             # 데이터 마지막 날 기준 N일 이상 미접속 → 이탈
WOTLK_RELEASE = pd.Timestamp("2008-11-13")   # 리치왕의 분노 출시일

OUT_DIR.mkdir(parents=True, exist_ok=True)
if not SHOW_PLOTS:
    matplotlib.use("Agg")

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


def save(fig, name):
    """그래프 저장 (+옵션: 화면 출력)"""
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def save_table(df, name):
    df.to_csv(OUT_DIR / f"{name}.csv", encoding="utf-8-sig")


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# %% ============================================================
# 1. 데이터 로드 & 정제
# ===============================================================
section("1. 데이터 로드")

df = pd.read_csv(RAW_FILE, compression="infer")
df.columns = df.columns.str.strip()           # 원본 컬럼명 앞 공백 제거
for c in ["race", "charclass", "zone"]:
    df[c] = df[c].astype(str).str.strip()

df["timestamp"] = pd.to_datetime(df["timestamp"].str.strip(),
                                 format="%m/%d/%y %H:%M:%S")

# 메모리 절약
df["char"] = df["char"].astype("int32")
df["level"] = df["level"].astype("int8")
df["guild"] = df["guild"].astype("int32")
for c in ["race", "charclass", "zone"]:
    df[c] = df[c].astype("category")

df["has_guild"] = df["guild"] != -1
df["date"] = df["timestamp"].dt.normalize()
df["hour"] = df["timestamp"].dt.hour
df["weekday"] = df["timestamp"].dt.dayofweek          # 0=월
df["month"] = df["timestamp"].dt.to_period("M")
df["level_band"] = pd.cut(df["level"], bins=[0, 10, 20, 30, 40, 50, 60, 69, 70, 79, 80],
                          labels=["1-10", "11-20", "21-30", "31-40", "41-50", "51-60",
                                  "61-69", "70", "71-79", "80"])

print(f"shape: {df.shape}")
print(f"memory: {df.memory_usage(deep=True).sum() / 1e6:,.1f} MB")
print(df.head())


# --- 메타 테이블 정제 -------------------------------------------
def decode_unicode_escape(s):
    """'<U+9F8D><U+9AA8>' → '龍骨' 형태로 복원"""
    if not isinstance(s, str):
        return s
    return re.sub(r"<U\+([0-9A-Fa-f]{4,6})>", lambda m: chr(int(m.group(1), 16)), s)


zones = pd.read_csv(ZONES_FILE, na_values=["NA", ""])
zones["Zone_Name"] = zones["Zone_Name"].map(decode_unicode_escape).str.strip()
zones["Continent"] = zones["Continent"].replace({"Ouland": "Outland"})
zones["Zone"] = zones["Zone"].replace({"Thosand Needles": "Thousand Needles"})
zones.loc[zones["Zone_Name"] == "Stratholme", "Continent"] = "Eastern Kingdoms"
for c in ["Min_rec_level", "Max_rec_level", "Min_bot_level", "Max_bot_level", "Size"]:
    zones[c] = pd.to_numeric(zones[c], errors="coerce")
zones = zones.drop_duplicates("Zone_Name")

locations = pd.read_csv(LOCATIONS_FILE)
locations["Location_Name"] = (locations["Location_Name"].map(decode_unicode_escape)
                              .str.replace(" ", "", regex=False).str.strip())

# --- 조인 ---------------------------------------------------------
df = df.merge(
    zones[["Zone_Name", "Continent", "Area", "Type", "Controlled",
           "Min_rec_level", "Max_rec_level"]],
    left_on="zone", right_on="Zone_Name", how="left",
).drop(columns="Zone_Name")

unmatched = df.loc[df["Type"].isna(), "zone"].value_counts()
unmatched = unmatched[unmatched > 0]
print(f"\nzones 매칭률: {df['Type'].notna().mean():.2%}")
if len(unmatched):
    print("매칭 안 된 zone (상위 20):")
    print(unmatched.head(20))
    save_table(unmatched.to_frame("rows"), "unmatched_zones")

for c in ["Continent", "Type", "Controlled"]:
    df[c] = df[c].fillna("Unknown").astype("category")


# %% ============================================================
# 2. 기본 개요
# ===============================================================
section("2. 기본 개요")

overview = pd.Series({
    "rows": len(df),
    "unique_chars": df["char"].nunique(),
    "unique_guilds (excl. -1)": df.loc[df["has_guild"], "guild"].nunique(),
    "unique_zones": df["zone"].nunique(),
    "start": df["timestamp"].min(),
    "end": df["timestamp"].max(),
    "days_covered": df["date"].nunique(),
    "null_cells": int(df[["char", "level", "race", "charclass", "zone", "guild",
                          "timestamp"]].isna().sum().sum()),
    "duplicate_rows (char+timestamp)": int(df.duplicated(["char", "timestamp"]).sum()),
    "guild_member_ratio": round(df["has_guild"].mean(), 4),
})
print(overview.to_string())
save_table(overview.to_frame("value"), "01_overview")

print("\n수치 요약")
print(df[["level", "guild"]].describe().T)


# %% ============================================================
# 3. 캐릭터 단위 테이블 (이후 분석의 기준)
# ===============================================================
section("3. 캐릭터 단위 집계")

df = df.sort_values(["char", "timestamp"]).reset_index(drop=True)

chars = df.groupby("char").agg(
    race=("race", "first"),
    charclass=("charclass", "first"),
    first_seen=("timestamp", "min"),
    last_seen=("timestamp", "max"),
    first_level=("level", "first"),
    max_level=("level", "max"),
    n_obs=("timestamp", "size"),
    n_days=("date", "nunique"),
    n_zones=("zone", "nunique"),
    guild_ratio=("has_guild", "mean"),
    n_guilds=("guild", lambda s: s[s != -1].nunique()),
)
chars["play_hours"] = chars["n_obs"] * SNAPSHOT_MIN / 60
chars["lifespan_days"] = (chars["last_seen"] - chars["first_seen"]).dt.total_seconds() / 86400
chars["levels_gained"] = chars["max_level"] - chars["first_level"]
chars["ever_guild"] = chars["guild_ratio"] > 0

data_end = df["timestamp"].max()
chars["churned"] = chars["last_seen"] < data_end - pd.Timedelta(days=CHURN_DAYS)
# 데이터 시작 직후부터 있던 캐릭터는 신규가 아님 → 신규 판정은 첫 관측 레벨 1
chars["is_new"] = chars["first_level"] == 1

print(chars.describe(include="number").T.round(2))
print(f"\n이탈률(마지막 {CHURN_DAYS}일 미접속): {chars['churned'].mean():.2%}")
print(f"관측 기간 내 신규(1레벨 시작) 캐릭터: {chars['is_new'].sum():,}")
save_table(chars, "02_character_table")


# %% ============================================================
# 4. 인구 구성: 종족 / 직업 / 레벨
# ===============================================================
section("4. 종족 · 직업 · 레벨 분포")

race_cnt = chars["race"].value_counts()
class_cnt = chars["charclass"].value_counts()
print(race_cnt, "\n", class_cnt)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
race_cnt.sort_values().plot.barh(ax=axes[0], color="#4C72B0")
axes[0].set_title("Characters by Race")
class_cnt.sort_values().plot.barh(ax=axes[1], color="#55A868")
axes[1].set_title("Characters by Class")
for ax in axes:
    ax.set_xlabel("characters")
save(fig, "03_race_class")

# 종족 × 직업 (호드 진영이라 불가능한 조합은 0)
rc = pd.crosstab(chars["race"], chars["charclass"])
save_table(rc, "03_race_x_class")
fig, ax = plt.subplots(figsize=(12, 4.5))
im = ax.imshow(rc.values, cmap="Blues", aspect="auto")
ax.set_xticks(range(rc.shape[1]))
ax.set_xticklabels(rc.columns, rotation=40, ha="right")
ax.set_yticks(range(rc.shape[0]))
ax.set_yticklabels(rc.index)
for i in range(rc.shape[0]):
    for j in range(rc.shape[1]):
        v = rc.values[i, j]
        if v:
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=8,
                    color="white" if v > rc.values.max() * 0.6 else "black")
ax.set_title("Race x Class (characters)")
ax.grid(False)
fig.colorbar(im, ax=ax)
save(fig, "03_race_x_class_heatmap")

# 레벨 분포: 관측 기준 vs 캐릭터 최고레벨 기준
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
lv_obs = df["level"].value_counts().sort_index()
lv_max = chars["max_level"].value_counts().sort_index()
axes[0].bar(lv_obs.index, lv_obs.values, width=1, color="#4C72B0")
axes[0].set_title("Level distribution (observations = time spent)")
axes[1].bar(lv_max.index, lv_max.values, width=1, color="#C44E52")
axes[1].set_title("Max level per character")
for ax in axes:
    ax.set_xticks(range(0, 81, 10))
    ax.set_xlabel("level")
save(fig, "04_level_distribution")

print("\n최고레벨 구간 비율")
print(pd.cut(chars["max_level"], [0, 10, 30, 59, 60, 69, 70, 79, 80]).value_counts(normalize=True)
      .sort_index().round(4))


# %% ============================================================
# 5. 시간 패턴
# ===============================================================
section("5. 시간 패턴")

daily = df.groupby("date").agg(dau=("char", "nunique"), obs=("char", "size"))
daily["dau_7d"] = daily["dau"].rolling(7, min_periods=1).mean()
save_table(daily, "05_daily_active")

fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(daily.index, daily["dau"], alpha=0.3, label="DAU")
ax.plot(daily.index, daily["dau_7d"], lw=2, label="DAU (7d avg)")
ax.axvline(WOTLK_RELEASE, color="crimson", ls="--", lw=1)
ax.text(WOTLK_RELEASE, ax.get_ylim()[1] * 0.95, " WotLK release", color="crimson")
ax.set_title("Daily Active Characters")
ax.legend()
save(fig, "05_dau_trend")

# 월별 신규 / 활성 / 이탈 캐릭터
monthly = pd.DataFrame({
    "active": df.groupby("month")["char"].nunique(),
    "new_first_seen": chars.groupby(chars["first_seen"].dt.to_period("M")).size(),
    "last_seen": chars.groupby(chars["last_seen"].dt.to_period("M")).size(),
}).fillna(0).astype(int)
print(monthly)
save_table(monthly, "05_monthly")

# 요일 × 시간 히트맵 (평균 동시접속 = 관측 수 / 일수)
n_days_by_wd = df.groupby("weekday")["date"].nunique()
wh = df.pivot_table(index="weekday", columns="hour", values="char",
                    aggfunc="size", fill_value=0)
wh = wh.div(n_days_by_wd, axis=0)
fig, ax = plt.subplots(figsize=(13, 4))
im = ax.imshow(wh.values, cmap="YlOrRd", aspect="auto")
ax.set_yticks(range(len(wh.index)))
ax.set_yticklabels([["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i] for i in wh.index])
ax.set_xticks(range(24))
ax.set_xlabel("hour (server time)")
ax.set_title("Avg observations per day: weekday x hour")
ax.grid(False)
fig.colorbar(im, ax=ax)
save(fig, "05_weekday_hour_heatmap")


# %% ============================================================
# 6. 세션 분석 (연속 관측 = 한 세션)
# ===============================================================
section("6. 세션 분석")

gap = df.groupby("char")["timestamp"].diff()
new_session = gap.isna() | (gap > pd.Timedelta(minutes=SESSION_GAP_MIN))
df["session_id"] = new_session.cumsum()

sessions = df.groupby("session_id").agg(
    char=("char", "first"),
    start=("timestamp", "min"),
    end=("timestamp", "max"),
    n_obs=("char", "size"),
    level_start=("level", "min"),
    level_end=("level", "max"),
)
# 관측 1회 세션도 최소 SNAPSHOT_MIN 분은 접속한 것으로 간주
sessions["minutes"] = ((sessions["end"] - sessions["start"]).dt.total_seconds() / 60
                       + SNAPSHOT_MIN)
print(sessions["minutes"].describe(percentiles=[.25, .5, .75, .9, .99]).round(1))

per_char_sess = sessions.groupby("char").agg(n_sessions=("minutes", "size"),
                                             avg_session_min=("minutes", "mean"))
chars = chars.join(per_char_sess)

fig, ax = plt.subplots()
clip = sessions["minutes"].quantile(0.99)
ax.hist(sessions["minutes"].clip(upper=clip), bins=60, color="#8172B2")
ax.set_title(f"Session length (minutes, clipped at p99={clip:.0f})")
ax.set_xlabel("minutes")
save(fig, "06_session_length")
print("※ 행 단위 랜덤 샘플에서는 세션이 거의 1회 관측으로 쪼개지므로 전체 데이터에서 해석할 것")


# %% ============================================================
# 7. 지역(Zone) 분석
# ===============================================================
section("7. 지역 분석")

top_zones = df["zone"].value_counts().head(25)
fig, ax = plt.subplots(figsize=(11, 7))
(top_zones.sort_values() * SNAPSHOT_MIN / 60).plot.barh(ax=ax, color="#4C72B0")
ax.set_title("Top 25 zones by total play time")
ax.set_xlabel("hours")
save(fig, "07_top_zones")
save_table(df["zone"].value_counts().to_frame("rows"), "07_zone_counts")

# 레벨 구간별 활동 유형 비중
type_by_band = pd.crosstab(df["level_band"], df["Type"], normalize="index")
print(type_by_band.round(3))
save_table(type_by_band, "07_type_by_level_band")
fig, ax = plt.subplots(figsize=(12, 5))
type_by_band.plot.bar(stacked=True, ax=ax, width=0.85, colormap="tab10")
ax.set_title("Activity type share by level band")
ax.set_ylabel("share")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
save(fig, "07_type_by_level_band")

# 월별 대륙 비중 (확장팩 이동 확인: Outland → Northrend)
cont_month = pd.crosstab(df["month"], df["Continent"], normalize="index")
save_table(cont_month, "07_continent_by_month")
fig, ax = plt.subplots(figsize=(12, 5))
cont_month.plot.area(ax=ax, colormap="tab20", alpha=0.85)
ax.set_title("Continent share by month")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
save(fig, "07_continent_by_month")

# 권장 레벨 적합도 (도시/전장/투기장처럼 범위가 넓은 곳 제외)
fit_mask = df["Type"].isin(["Zone", "Dungeon"]) & df["Min_rec_level"].notna() \
           & (df["Max_rec_level"] - df["Min_rec_level"] < 50)
fit = df.loc[fit_mask, ["level", "Min_rec_level", "Max_rec_level", "level_band"]].copy()
fit["fit"] = np.select(
    [fit["level"] < fit["Min_rec_level"], fit["level"] > fit["Max_rec_level"]],
    ["under", "over"], default="in_range")
fit_tbl = pd.crosstab(fit["level_band"], fit["fit"], normalize="index")
print("\n레벨 구간별 권장레벨 적합도")
print(fit_tbl.round(3))
save_table(fit_tbl, "07_level_fit")


# %% ============================================================
# 8. 길드
# ===============================================================
section("8. 길드")

g = df[df["has_guild"]]
guild_size = g.groupby("guild")["char"].nunique().sort_values(ascending=False)
print(guild_size.describe().round(1))
save_table(guild_size.to_frame("members"), "08_guild_size")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].hist(guild_size, bins=50, color="#55A868")
axes[0].set_yscale("log")
axes[0].set_title("Guild size distribution (unique members)")
guild_by_lv = df.groupby("level_band", observed=True)["has_guild"].mean()
guild_by_lv.plot.bar(ax=axes[1], color="#4C72B0")
axes[1].set_title("Guild membership rate by level band")
save(fig, "08_guild")

guild_effect = chars.groupby("ever_guild").agg(
    chars=("max_level", "size"),
    churn_rate=("churned", "mean"),
    median_play_hours=("play_hours", "median"),
    median_lifespan_days=("lifespan_days", "median"),
    mean_max_level=("max_level", "mean"),
)
print("\n길드 가입 경험 여부별 비교")
print(guild_effect.round(3))
save_table(guild_effect, "08_guild_effect")


# %% ============================================================
# 9. 레벨업 속도 (관측 기간 중 1레벨에서 시작한 캐릭터)
# ===============================================================
section("9. 레벨업 속도")

new_ids = chars.index[chars["is_new"]]
nd = df[df["char"].isin(new_ids)]
start = nd.groupby("char")["timestamp"].min()

level_speed = {}
for target in [10, 20, 30, 40, 50, 60, 70, 80]:
    reached = nd[nd["level"] >= target].groupby("char").agg(
        t=("timestamp", "min"),
        # 실제 플레이 시간 = 목표 레벨 도달 전 관측 수 × 10분
    )
    if reached.empty:
        continue
    obs_before = (nd.merge(reached, left_on="char", right_index=True)
                    .query("timestamp <= t").groupby("char").size())
    level_speed[target] = pd.DataFrame({
        "calendar_days": (reached["t"] - start.loc[reached.index]).dt.total_seconds() / 86400,
        "play_hours": obs_before * SNAPSHOT_MIN / 60,
    })

speed_tbl = pd.DataFrame({
    t: {"n_chars": len(v),
        "median_calendar_days": v["calendar_days"].median(),
        "median_play_hours": v["play_hours"].median()}
    for t, v in level_speed.items()
}).T
print(speed_tbl.round(1))
save_table(speed_tbl, "09_leveling_speed")

if not speed_tbl.empty:
    fig, ax = plt.subplots()
    ax.plot(speed_tbl.index, speed_tbl["median_play_hours"], marker="o")
    ax.set_title("Median play hours to reach level (new characters)")
    ax.set_xlabel("target level")
    ax.set_ylabel("play hours")
    save(fig, "09_leveling_speed")

# 직업별 70 도달 시간
if 70 in level_speed:
    by_class = (level_speed[70].join(chars["charclass"])
                .groupby("charclass", observed=True)["play_hours"]
                .agg(["size", "median"]).sort_values("median"))
    print("\n직업별 70레벨 도달 플레이시간(중앙값)")
    print(by_class.round(1))
    save_table(by_class, "09_hours_to_70_by_class")


# %% ============================================================
# 10. 이탈 분석
# ===============================================================
section("10. 이탈 분석")

def churn_by(col):
    return chars.groupby(col, observed=True)["churned"].agg(["size", "mean"]) \
                .rename(columns={"size": "chars", "mean": "churn_rate"})

chars["max_level_band"] = pd.cut(chars["max_level"], [0, 10, 20, 40, 59, 60, 69, 70, 79, 80])
chars["play_hours_band"] = pd.qcut(chars["play_hours"], 5, duplicates="drop")

for col in ["race", "charclass", "max_level_band", "play_hours_band", "ever_guild"]:
    t = churn_by(col)
    print(f"\n[{col}]")
    print(t.round(3))
    save_table(t, f"10_churn_by_{col}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
churn_by("max_level_band")["churn_rate"].plot.bar(ax=axes[0], color="#C44E52")
axes[0].set_title("Churn rate by max level")
churn_by("charclass")["churn_rate"].sort_values().plot.barh(ax=axes[1], color="#C44E52")
axes[1].set_title("Churn rate by class")
save(fig, "10_churn")

# 신규 캐릭터 생존곡선 (첫 관측 후 N일째에도 활동 중인 비율)
new_chars = chars[chars["is_new"]].copy()
if len(new_chars):
    observable = (data_end - new_chars["first_seen"]).dt.days
    horizon = range(0, 181, 7)
    surv = []
    for d in horizon:
        base = new_chars[observable >= d]
        surv.append((base["lifespan_days"] >= d).mean() if len(base) else np.nan)
    fig, ax = plt.subplots()
    ax.plot(list(horizon), surv, marker="o", ms=3)
    ax.set_ylim(0, 1)
    ax.set_title("New character retention (still seen after N days)")
    ax.set_xlabel("days since first seen")
    save(fig, "10_new_char_retention")


# %% ============================================================
# 11. 리치왕 출시 전후 비교
# ===============================================================
section("11. WotLK 출시 전후")

window = pd.Timedelta(days=30)
pre = df[(df["timestamp"] >= WOTLK_RELEASE - window) & (df["timestamp"] < WOTLK_RELEASE)]
post = df[(df["timestamp"] >= WOTLK_RELEASE) & (df["timestamp"] < WOTLK_RELEASE + window)]

def period_summary(x):
    days = max(x["date"].nunique(), 1)
    return pd.Series({
        "avg_DAU": x.groupby("date")["char"].nunique().mean(),
        "obs_per_day": len(x) / days,
        "share_Northrend": (x["Continent"] == "Northrend").mean(),
        "share_lvl_71plus": (x["level"] > 70).mean(),
        "death_knight_chars": x.loc[x["charclass"] == "Death Knight", "char"].nunique(),
    })

wotlk = pd.DataFrame({"30d_before": period_summary(pre), "30d_after": period_summary(post)})
print(wotlk.round(3))
save_table(wotlk, "11_wotlk_before_after")


# %% ============================================================
# 12. 마무리
# ===============================================================
save_table(chars, "02_character_table")     # 세션 컬럼까지 포함해 다시 저장
section("완료")
print(f"결과 폴더: {OUT_DIR.resolve()}")
for p in sorted(OUT_DIR.iterdir()):
    print(" -", p.name)