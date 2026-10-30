# -*- coding: utf-8 -*-
"""
WoWAH 이상탐지 v2

v1 대비 변경점
  A. 데이터 이상
     - 종족·직업 변경(A5)과 레벨 역행(A2)의 겹침 분석 → "ID 충돌" 캐릭터 목록을 만들어 B 판정에서 제외
     - 입장 불가 지역(A4)은 메타데이터·버스 플레이 가능성이 커서 '경고'로 하향, 레벨 차이 분포만 확인
  B. 캐릭터 이상
     - R2: 24시간 이상 세션 중 '활동이 있는' 세션만 인정 (지역 이동 / 레벨업) → 잠수와 구분
     - R3: 1년 전체 엔트로피 → 하루 단위 '쉬는 구간(연속 오프라인 N시간) 없는 날 수'로 교체
     - 상주 캐릭터(저레벨·무성장·장시간 접속, 수집용 캐릭터 의심) 별도 라벨로 분리
     - 봇 유형 세분화: leveling / bg_farming / field_farming / other
     - v1 결과가 있으면 의심 목록 비교(유지 / 제외 / 신규)
  C. 시점 이상
     - 데이터 첫 7일(신규 급증 인공물), 마지막 7일(이탈 급증 인공물) 제외
     - 하루 중 관측된 시간 수로 '부분 수집일' 판별 → 점검·장애로 설명되는 이상 분리
     - 목요일 정기점검, 현지 일정(춘절 등) 캘린더 반영, z 임계값 5로 상향
     - 10월 8~10일 코호트 심층 분석 (하루살이 비율, ID 재매핑 매칭 테스트)

결과: D:/code/game/archive/anomaly_output_v2
필요 패키지: pandas, numpy, matplotlib / 선택: scikit-learn, statsmodels
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
OUT_DIR = DATA_DIR / "anomaly_output_v2"
V1_DIR = DATA_DIR / "anomaly_output"        # v1 결과 비교용 (없으면 건너뜀)

SNAPSHOT_MIN = 10
SESSION_GAP_MIN = 30

# --- A ---
JUMP_LEVELS = 5
JUMP_WINDOW_MIN = 60
LEVEL_DROP_EXCLUDE = 5        # 한 번에 이 레벨 이상 떨어지면 ID 충돌로 간주

# --- B ---
MIN_ACTIVE_DAYS = 7
R_DAILY_HOURS = 16            # R1 접속일 기준 일평균 접속시간
R_MARATHON_HOURS = 24         # R2 연속 세션 길이
R_MARATHON_MIN_ZONE_CHG = 3   # R2 세션 중 지역 이동 최소 횟수 (또는 레벨업 1회 이상)
REST_GAP_HOURS = 6            # R3 하루 중 이 시간 이상 연속 오프라인이면 '쉰 날'
R_NO_REST_STREAK = 3          # R3 쉬지 않은 날이 N일 "연속" (헤비유저 제외 목적)
R_LEVEL_QUANTILE = 0.005      # R4 레벨당 플레이시간 하위 0.5%
R_LEVEL_MIN_GAIN = 10
R_FARM_ZONE_RATIO = 0.8       # R5
R_FARM_MIN_HOURS = 50
RESIDENT_MAX_LEVEL = 10       # 상주 캐릭터: 이 레벨 이하 + 성장 없음 + 하루 16시간 이상
BG_BOT_RATIO = 0.4            # 봇 유형: 전장 비중 40% 이상이면 bg_farming
LEVELING_MIN_GAIN = 20        # 봇 유형: 20레벨 이상 올렸으면 leveling
IFOREST_TOP = 0.01
TIMELINES_PER_TYPE = 5
N_DROPPED_FROM_V1 = 5
N_NORMAL_COMPARE = 3

# --- C ---
Z_THRESHOLD = 5.0
OUTAGE_MIN_HOURS = 2
FULL_DAY_MIN_HOURS = 20       # 하루 중 관측된 시간이 이보다 적으면 '부분 수집일'
EDGE_DAYS = 7                 # 데이터 시작·끝 N일은 신규·이탈 판정 제외
EVENT_MATCH_DAYS = 3
MAINTENANCE_WEEKDAY = 3       # 목요일(0=월)

# 이벤트 캘린더 (날짜는 서버 현지 기준으로 확인 후 수정 권장)
EVENTS = {
    "2008-02-06": "Lunar New Year holidays start",
    "2008-02-11": "Lunar New Year holidays end",
    "2008-03-27": "Patch 2.4 (Fury of the Sunwell, local)",
    "2008-10-14": "Patch 3.0.2 (WotLK pre-patch)",
    "2008-11-13": "WotLK release",
}

# 코호트 심층 분석 구간
COHORT_START = "2008-10-08"
COHORT_END = "2008-10-10"
BASELINE_START = "2008-09-10"     # 비교용 평상시 구간 (같은 길이)
BASELINE_END = "2008-09-12"

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "timelines").mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (11, 5), "figure.dpi": 110,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
})
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 30)


def save(fig, name, sub=None):
    fig.tight_layout()
    folder = OUT_DIR / sub if sub else OUT_DIR
    fig.savefig(folder / f"{name}.png", bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def save_table(df, name, index=True):
    df.to_csv(OUT_DIR / f"{name}.csv", encoding="utf-8-sig", index=index)


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def robust_z(x):
    x = pd.Series(x, dtype="float64")
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return pd.Series(0.0, index=x.index)
    return 0.6745 * (x - med) / mad


# %% ============================================================
# 1. 로드
# ===============================================================
section("1. 데이터 로드")

df = pd.read_csv(RAW_FILE, compression="infer")
df.columns = df.columns.str.strip()
for c in ["race", "charclass", "zone"]:
    df[c] = df[c].astype(str).str.strip()
df["timestamp"] = pd.to_datetime(df["timestamp"].astype(str).str.strip(),
                                 format="%m/%d/%y %H:%M:%S")
df["char"] = df["char"].astype("int32")
df["level"] = df["level"].astype("int16")
df["guild"] = df["guild"].astype("int32")

zones = pd.read_csv(ZONES_FILE, na_values=["NA", ""])
zones["Zone_Name"] = (zones["Zone_Name"].astype(str)
                      .str.replace(r"<U\+([0-9A-Fa-f]{4,6})>",
                                   lambda m: chr(int(m.group(1), 16)), regex=True)
                      .str.strip())
zones["Min_req_level"] = pd.to_numeric(zones["Min_req_level"], errors="coerce")
zones = zones.drop_duplicates("Zone_Name")
df = df.merge(zones[["Zone_Name", "Type", "Continent", "Min_req_level"]],
              left_on="zone", right_on="Zone_Name", how="left").drop(columns="Zone_Name")
df["Type"] = df["Type"].fillna("Unknown")
for c in ["race", "charclass", "zone", "Type", "Continent"]:
    df[c] = df[c].astype("category")

df = df.sort_values(["char", "timestamp"]).reset_index(drop=True)
df["date"] = df["timestamp"].dt.normalize()
df["hour"] = df["timestamp"].dt.hour
print(f"shape: {df.shape}")


# %% ============================================================
# A. 데이터 이상
# ===============================================================
section("A. 데이터 이상")

n_dup = int(df.duplicated(["char", "timestamp"]).sum())
dup_chars = df.loc[df.duplicated(["char", "timestamp"], keep=False), "char"].unique()
df = df.drop_duplicates(["char", "timestamp"], keep="first").reset_index(drop=True)
print(f"A1 중복 관측: {n_dup:,}행 제거 (캐릭터 {len(dup_chars):,}명)")

g = df.groupby("char", sort=False)
df["prev_level"] = g["level"].shift()
df["d_level"] = df["level"] - df["prev_level"]
df["d_min"] = g["timestamp"].diff().dt.total_seconds() / 60
df["zone_code"] = df["zone"].cat.codes
df["prev_zone"] = g["zone_code"].shift()

df["a_level_drop"] = df["d_level"] < 0
df["a_level_jump"] = (df["d_level"] >= JUMP_LEVELS) & (df["d_min"] <= JUMP_WINDOW_MIN)
df["a_entry_violation"] = df["Min_req_level"].notna() & (df["level"] < df["Min_req_level"])

attr = df.groupby("char").agg(n_race=("race", "nunique"), n_class=("charclass", "nunique"))
a5_chars = set(attr.index[(attr["n_race"] > 1) | (attr["n_class"] > 1)])
a2_chars = set(df.loc[df["a_level_drop"], "char"].unique())
max_drop = (-df["d_level"]).clip(lower=0).groupby(df["char"]).max()
big_drop_chars = set(max_drop.index[max_drop >= LEVEL_DROP_EXCLUDE])

overlap = pd.DataFrame({
    "chars": [len(a5_chars), len(a2_chars), len(a5_chars & a2_chars),
              len(a2_chars - a5_chars), len(big_drop_chars),
              len(big_drop_chars - a5_chars)],
}, index=["A5 race/class change", "A2 level drop", "A5 & A2",
          "A2 only", f"drop >= {LEVEL_DROP_EXCLUDE} levels", f"drop >= {LEVEL_DROP_EXCLUDE} & not A5"])
print("\nA2·A5 겹침")
print(overlap)
save_table(overlap, "A_overlap_A2_A5")

print("\nA2 레벨 하락 폭 분포 (작으면 기록 지연, 크면 ID 충돌)")
print((-df.loc[df["a_level_drop"], "d_level"]).describe(percentiles=[.5, .75, .9]).round(1))

id_collision = a5_chars | big_drop_chars
pd.Series(sorted(id_collision), name="char").to_csv(OUT_DIR / "A_id_collision_chars.csv",
                                                    index=False, encoding="utf-8-sig")
print(f"\n→ ID 충돌 의심 캐릭터 {len(id_collision):,}명: B 판정에서 제외")

# A4는 경고로 하향: 레벨 차이만 확인
viol = df[df["a_entry_violation"]]
gap = (viol["Min_req_level"] - viol["level"])
print(f"\nA4 입장 불가 지역(경고): {len(viol):,}행 / {viol['char'].nunique():,}명")
print("   요구레벨 - 실제레벨 분포:")
print(gap.describe(percentiles=[.5, .75, .9]).round(1))
print(f"   5레벨 이내 차이 비율: {(gap <= 5).mean():.1%} (작으면 메타데이터 기준 차이 가능성)")
save_table(viol.groupby("zone", observed=True).agg(
    rows=("char", "size"), chars=("char", "nunique"),
    min_req=("Min_req_level", "first"), median_level=("level", "median"))
    .sort_values("rows", ascending=False), "A4_entry_warning_by_zone")

summary_a = pd.Series({
    "A1_duplicate_rows": n_dup,
    "A2_level_drop_rows": int(df["a_level_drop"].sum()),
    "A3_level_jump_rows": int(df["a_level_jump"].sum()),
    "A4_entry_warning_rows": len(viol),
    "A5_race_class_change_chars": len(a5_chars),
    "id_collision_excluded_chars": len(id_collision),
})
save_table(summary_a.to_frame("value"), "A_summary")


# %% ============================================================
# B. 캐릭터 피처
# ===============================================================
section("B. 캐릭터 피처")

# 세션 + 세션 내 활동량
new_sess = df["d_min"].isna() | (df["d_min"] > SESSION_GAP_MIN)
df["session_id"] = new_sess.cumsum()
df["zone_chg"] = (~new_sess) & (df["zone_code"] != df["prev_zone"])
sess = df.groupby("session_id").agg(
    char=("char", "first"), start=("timestamp", "min"), end=("timestamp", "max"),
    zone_changes=("zone_chg", "sum"), lv_min=("level", "min"), lv_max=("level", "max"))
sess["hours"] = ((sess["end"] - sess["start"]).dt.total_seconds() / 60 + SNAPSHOT_MIN) / 60
sess["level_gain"] = sess["lv_max"] - sess["lv_min"]
sess["long"] = sess["hours"] >= R_MARATHON_HOURS
sess["active_marathon"] = sess["long"] & ((sess["zone_changes"] >= R_MARATHON_MIN_ZONE_CHG)
                                          | (sess["level_gain"] >= 1))
sess["afk_marathon"] = sess["long"] & ~sess["active_marathon"]

feat = df.groupby("char").agg(
    race=("race", "first"), charclass=("charclass", "first"),
    first_seen=("timestamp", "min"), last_seen=("timestamp", "max"),
    first_level=("level", "first"), last_level=("level", "last"),
    max_level=("level", "max"), n_obs=("timestamp", "size"),
    n_days=("date", "nunique"), n_zones=("zone", "nunique"),
    first_guild=("guild", "first"), last_guild=("guild", "last"),
)
feat["guild_ratio"] = (df["guild"] != -1).groupby(df["char"]).mean()
feat["play_hours"] = feat["n_obs"] * SNAPSHOT_MIN / 60
feat["hours_per_active_day"] = feat["play_hours"] / feat["n_days"]
feat["levels_gained"] = feat["max_level"] - feat["first_level"]

sg = sess.groupby("char")
feat["max_session_hours"] = sg["hours"].max()
feat["n_active_marathon"] = sg["active_marathon"].sum()
feat["n_afk_marathon"] = sg["afk_marathon"].sum()
feat["zone_chg_per_hour"] = sg["zone_changes"].sum() / feat["play_hours"]

# 하루 단위 '쉰 구간' : 하루 24칸 중 연속 오프라인 최대 길이
pres = (df.groupby(["char", "date", "hour"]).size() > 0).unstack(fill_value=False)
pres = pres.reindex(columns=range(24), fill_value=False)
arr = pres.values
run = np.zeros(len(arr), dtype=np.int16)
best = np.zeros(len(arr), dtype=np.int16)
for h in range(24):
    run = np.where(arr[:, h], 0, run + 1)
    best = np.maximum(best, run)
day_tbl = pd.DataFrame({"online_hours": arr.sum(axis=1), "max_off_run": best},
                       index=pres.index)
day_tbl["no_rest"] = day_tbl["max_off_run"] < REST_GAP_HOURS
feat["no_rest_days"] = day_tbl["no_rest"].groupby(level=0).sum()

# 쉬지 않은 날이 '연속'으로 이어진 최대 일수 (헤비유저 vs 봇 구분)
nr = day_tbl[day_tbl["no_rest"]].reset_index()[["char", "date"]]
nr["brk"] = (nr["char"] != nr["char"].shift()) | (nr["date"].diff() != pd.Timedelta(days=1))
nr["grp"] = nr["brk"].cumsum()
feat["max_no_rest_streak"] = nr.groupby("grp").agg(char=("char", "first"), n=("date", "size")) \
    .groupby("char")["n"].max()
feat["max_no_rest_streak"] = feat["max_no_rest_streak"].fillna(0).astype(int)
del nr
feat["max_daily_online_hours"] = day_tbl["online_hours"].groupby(level=0).max()
del pres, arr

# 지역 / 활동 유형
zone_cnt = df.groupby(["char", "zone"], observed=True).size()
feat["top_zone"] = zone_cnt.groupby(level=0).idxmax().map(lambda t: t[1]).astype(str)
feat["top_zone_ratio"] = zone_cnt.groupby(level=0).max() / feat["n_obs"]
zone_type = zones.set_index("Zone_Name")["Type"]
feat["top_zone_type"] = feat["top_zone"].map(zone_type).fillna("Unknown")
for t, col in [("City", "city_ratio"), ("Battleground", "bg_ratio"),
               ("Zone", "field_ratio"), ("Dungeon", "dungeon_ratio")]:
    feat[col] = (df["Type"] == t).groupby(df["char"]).mean()

feat["hours_per_level"] = np.where(feat["levels_gained"] >= R_LEVEL_MIN_GAIN,
                                   feat["play_hours"] / feat["levels_gained"].clip(lower=1),
                                   np.nan)
feat["id_collision"] = feat.index.isin(list(id_collision))
save_table(feat, "B_char_features_all")


# %% ============================================================
# B. 규칙 판정
# ===============================================================
section("B. 규칙 판정")

cand = feat[(feat["n_days"] >= MIN_ACTIVE_DAYS) & ~feat["id_collision"]].copy()
print(f"판정 대상: {len(cand):,}명 (관측일 {MIN_ACTIVE_DAYS}일 이상, ID 충돌 제외)")

q_lvl = cand["hours_per_level"].quantile(R_LEVEL_QUANTILE)
cand["R1_long_daily"] = cand["hours_per_active_day"] >= R_DAILY_HOURS
cand["R2_active_marathon"] = cand["n_active_marathon"] >= 1
cand["R3_no_rest_streak"] = cand["max_no_rest_streak"] >= R_NO_REST_STREAK
# 참고용: 예전 기준(누적 3일)에만 걸리고 연속은 아닌 헤비유저
cand["heavy_user"] = (cand["no_rest_days"] >= 3) & ~cand["R3_no_rest_streak"]
cand["R4_fast_level"] = cand["hours_per_level"] <= q_lvl
cand["R5_farming"] = ((cand["top_zone_ratio"] >= R_FARM_ZONE_RATIO)
                      & (cand["play_hours"] >= R_FARM_MIN_HOURS)
                      & (cand["top_zone_type"] != "City"))
rules = ["R1_long_daily", "R2_active_marathon", "R3_no_rest_streak", "R4_fast_level", "R5_farming"]
cand["n_rules"] = cand[rules].sum(axis=1)
print(f"R4 임계값: 레벨당 {q_lvl:.2f}시간 이하")
print(cand[rules].sum().rename("chars").to_frame())
print("\n규칙 충족 개수")
print(cand["n_rules"].value_counts().sort_index())

# 정상 용도 / 특수 캐릭터
cand["resident"] = ((cand["max_level"] <= RESIDENT_MAX_LEVEL) & (cand["levels_gained"] == 0)
                    & (cand["hours_per_active_day"] >= R_DAILY_HOURS))
cand["bank_alt"] = (~cand["resident"] & (cand["max_level"] <= 20)
                    & (cand["top_zone_type"] == "City")
                    & (cand["top_zone_ratio"] >= R_FARM_ZONE_RATIO))
cand["afk_city"] = (~cand["resident"] & ~cand["bank_alt"] & (cand["levels_gained"] == 0)
                    & (cand["city_ratio"] >= R_FARM_ZONE_RATIO))

# Isolation Forest
iso_cols = ["hours_per_active_day", "max_session_hours", "n_active_marathon",
            "max_no_rest_streak", "max_daily_online_hours", "zone_chg_per_hour",
            "top_zone_ratio", "bg_ratio", "play_hours", "n_days", "hours_per_level"]
X = cand[iso_cols].copy()
X["hours_per_level"] = X["hours_per_level"].fillna(X["hours_per_level"].median())
for c in ["hours_per_active_day", "max_session_hours", "n_active_marathon", "max_no_rest_streak",
          "zone_chg_per_hour", "play_hours", "n_days", "hours_per_level"]:
    X[c] = np.log1p(X[c])
X = X.fillna(0)
try:
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(n_estimators=300, contamination=IFOREST_TOP, random_state=42)
    iso.fit(X.values)
    cand["iso_score"] = -iso.score_samples(X.values)
    print("\nIsolation Forest 사용")
except ImportError:
    cand["iso_score"] = X.apply(robust_z).abs().mean(axis=1)
    print("\nscikit-learn 없음 → robust z 평균 점수로 대체")
cand["iso_top"] = cand["iso_score"] >= cand["iso_score"].quantile(1 - IFOREST_TOP)

special = cand["resident"] | cand["bank_alt"] | cand["afk_city"]
bot = ((cand["n_rules"] >= 2) | ((cand["n_rules"] >= 1) & cand["iso_top"])) & ~special
cand["label"] = np.select(
    [cand["resident"], bot, cand["bank_alt"], cand["afk_city"]],
    ["resident", "bot_suspect", "bank_alt", "afk_city"], default="normal")

cand["bot_type"] = ""
is_bot = cand["label"] == "bot_suspect"
cand.loc[is_bot, "bot_type"] = np.select(
    [cand.loc[is_bot, "bg_ratio"] >= BG_BOT_RATIO,
     cand.loc[is_bot, "levels_gained"] >= LEVELING_MIN_GAIN,
     cand.loc[is_bot, "R5_farming"]],
    ["bg_farming", "leveling", "field_farming"], default="other")

print("\n최종 라벨")
print(cand["label"].value_counts())
print(f"\n헤비유저(누적 3일 이상 무휴식, 연속은 아님): {int(cand['heavy_user'].sum()):,}명 → 봇 판정에서 R3 미적용")
print("\n봇 유형")
print(cand.loc[is_bot, "bot_type"].value_counts())

profile_cols = ["play_hours", "hours_per_active_day", "max_session_hours", "n_active_marathon",
                "n_afk_marathon", "no_rest_days", "max_no_rest_streak", "zone_chg_per_hour", "bg_ratio",
                "levels_gained", "guild_ratio"]
prof = cand.groupby("label")[profile_cols].median()
prof_type = cand[is_bot].groupby("bot_type")[profile_cols].median()
print("\n라벨별 중앙값")
print(prof.round(2))
print("\n봇 유형별 중앙값")
print(prof_type.round(2))
save_table(prof, "B_label_profile")
save_table(prof_type, "B_bot_type_profile")

show_cols = ["bot_type", "race", "charclass", "first_level", "max_level", "play_hours",
             "hours_per_active_day", "max_session_hours", "n_active_marathon",
             "max_no_rest_streak", "bg_ratio", "top_zone", "n_rules", "iso_score"] + rules
suspects = cand[is_bot].sort_values(["n_rules", "iso_score"], ascending=False)
print("\n봇 의심 상위 20")
print(suspects[show_cols[:14]].head(20).round(2).to_string())
save_table(cand, "B_char_labels")
save_table(suspects[show_cols], "B_bot_suspects")
save_table(cand[cand["label"] == "resident"], "B_resident")
save_table(cand[cand["label"].isin(["bank_alt", "afk_city"])], "B_bank_afk")

# v1 비교
dropped_v1 = []
v1_file = V1_DIR / "B_bot_suspects.csv"
if v1_file.exists():
    v1 = set(pd.read_csv(v1_file, encoding="utf-8-sig").iloc[:, 0].astype(int))
    v2 = set(suspects.index)
    dropped_v1 = sorted(v1 - v2)
    cmp_tbl = pd.Series({"v1": len(v1), "v2": len(v2), "kept": len(v1 & v2),
                         "dropped_in_v2": len(v1 - v2), "new_in_v2": len(v2 - v1)})
    print("\nv1 대비 봇 의심 목록 변화")
    print(cmp_tbl)
    why = feat.loc[[c for c in dropped_v1 if c in feat.index]].copy()
    why["v2_label"] = cand["label"].reindex(why.index).fillna("excluded(days/id_collision)")
    print("\nv2에서 빠진 캐릭터의 v2 라벨")
    print(why["v2_label"].value_counts())
    save_table(why, "B_dropped_from_v1")
    save_table(cand.loc[sorted(v2 - v1)], "B_new_in_v2")

# 산점도
colors = {"normal": "#B0B0B0", "bank_alt": "#55A868", "afk_city": "#4C72B0",
          "resident": "#DD8452", "bot_suspect": "#C44E52"}
fig, ax = plt.subplots(figsize=(10, 6))
for lab in ["normal", "bank_alt", "afk_city", "resident", "bot_suspect"]:
    s = cand[cand["label"] == lab]
    ax.scatter(s["hours_per_active_day"], s["max_no_rest_streak"] + 0.5,
               s=6 if lab == "normal" else 16, alpha=0.35 if lab == "normal" else 0.85,
               color=colors[lab], label=f"{lab} ({len(s):,})")
ax.set_yscale("log")
ax.axvline(R_DAILY_HOURS, color="k", ls="--", lw=0.8)
ax.axhline(R_NO_REST_STREAK + 0.5, color="k", ls=":", lw=0.8)
ax.set_xlabel("avg hours per active day")
ax.set_ylabel(f"longest streak of days without a {REST_GAP_HOURS}h break (+0.5, log)")
ax.set_title("Character anomaly map (v2)")
ax.legend()
save(fig, "B_anomaly_scatter")


# %% ============================================================
# B. 타임라인
# ===============================================================
def plot_timeline(cid, tag):
    sub = df[df["char"] == cid]
    if sub.empty:
        return
    info = cand.loc[cid] if cid in cand.index else feat.loc[cid]
    dates = pd.date_range(sub["date"].min(), sub["date"].max(), freq="D")
    mat = (sub.groupby(["date", "hour"]).size().unstack(fill_value=0)
              .reindex(index=dates, columns=range(24), fill_value=0))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5),
                             gridspec_kw={"width_ratios": [1.2, 1.4, 1]})
    ax = axes[0]
    ax.imshow(mat.values, aspect="auto", cmap="Reds", interpolation="nearest")
    step = max(len(dates) // 10, 1)
    ax.set_yticks(range(0, len(dates), step))
    ax.set_yticklabels([d.strftime("%m-%d") for d in dates[::step]], fontsize=7)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels(range(0, 24, 3))
    ax.set_xlabel("hour")
    ax.set_title("online (date x hour)")
    ax.grid(False)
    ax = axes[1]
    ax.plot(sub["timestamp"], sub["level"], drawstyle="steps-post")
    ax.set_title("level over time")
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
    ax = axes[2]
    zc = sub["zone"].value_counts()
    zc = zc[zc > 0].head(8).sort_values()
    ax.barh([str(z) for z in zc.index], zc.values * SNAPSHOT_MIN / 60)
    ax.set_title("top zones (hours)")
    extra = f" | type={info['bot_type']}" if "bot_type" in info and info["bot_type"] else ""
    fig.suptitle(f"[{tag}] char {cid} | {info['race']} {info['charclass']} | "
                 f"lv {info['first_level']}->{info['max_level']} | {info['play_hours']:.0f}h, "
                 f"{info['hours_per_active_day']:.1f}h/day | active marathons "
                 f"{int(info['n_active_marathon'])}, no-rest days {int(info['no_rest_days'])} "
                 f"(streak {int(info['max_no_rest_streak'])}){extra}")
    save(fig, f"{tag}_{cid}", sub="timelines")


for btype, grp in suspects.groupby("bot_type"):
    for cid in grp.index[:TIMELINES_PER_TYPE]:
        plot_timeline(cid, f"bot-{btype}")
for cid in cand[cand["label"] == "resident"].sort_values("play_hours", ascending=False).index[:3]:
    plot_timeline(cid, "resident")
for cid in dropped_v1[:N_DROPPED_FROM_V1]:
    plot_timeline(cid, "dropped-v1")
heavy = cand[(cand["label"] == "normal") & (cand["n_days"] >= 30)] \
    .sort_values("play_hours", ascending=False).head(50)
for cid in heavy.sample(min(N_NORMAL_COMPARE, len(heavy)), random_state=0).index:
    plot_timeline(cid, "normal")
print(f"\n타임라인 저장: {OUT_DIR / 'timelines'}")


# %% ============================================================
# C. 시점 이상
# ===============================================================
section("C. 시점 이상")

full_days = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
hourly = df.set_index("timestamp").resample("H").size()

daily = pd.DataFrame(index=full_days)
daily["dau"] = df.groupby("date")["char"].nunique()
daily["obs"] = df.groupby("date").size()
daily["observed_hours"] = (hourly > 0).groupby(hourly.index.normalize()).sum()
daily["new_chars"] = feat.groupby(feat["first_seen"].dt.normalize()).size()
daily["gone_chars"] = feat.groupby(feat["last_seen"].dt.normalize()).size()
daily = daily.fillna({"obs": 0, "observed_hours": 0, "new_chars": 0, "gone_chars": 0})
daily["missing"] = daily["obs"] == 0
daily["partial"] = ~daily["missing"] & (daily["observed_hours"] < FULL_DAY_MIN_HOURS)
print(f"누락일 {int(daily['missing'].sum())}일, 부분 수집일(관측 {FULL_DAY_MIN_HOURS}시간 미만) "
      f"{int(daily['partial'].sum())}일")

# 장애·점검 구간
zero = hourly == 0
rid = (zero != zero.shift()).cumsum()
zs = hourly[zero].index.to_series()
outages = pd.DataFrame({"start": zs.groupby(rid[zero]).min(),
                        "hours": zs.groupby(rid[zero]).size()})
outages["end"] = outages["start"] + pd.to_timedelta(outages["hours"], unit="h")
outages = outages[outages["hours"] >= OUTAGE_MIN_HOURS].reset_index(drop=True)
outages["weekday"] = outages["start"].dt.dayofweek
outages["type"] = np.where((outages["weekday"] == MAINTENANCE_WEEKDAY) & (outages["hours"] <= 18),
                           "weekly_maintenance", "outage/collection_gap")
print("\n장애·점검 구간 유형")
print(outages["type"].value_counts())
save_table(outages, "C1_outages", index=False)

outage_days = set()
for _, r in outages.iterrows():
    for d in pd.date_range(r["start"].normalize(), (r["end"] - pd.Timedelta(seconds=1)).normalize()):
        outage_days.add(d)

# DAU: 누락·부분 수집일은 결측 처리 후 보간해서 분해
dau_clean = daily["dau"].where(~daily["missing"] & ~daily["partial"])
dau_filled = dau_clean.interpolate(limit_direction="both")
try:
    from statsmodels.tsa.seasonal import STL
    stl = STL(dau_filled, period=7, robust=True).fit()
    daily["dau_trend"] = stl.trend + stl.seasonal
    method = "STL"
except ImportError:
    tr = dau_filled.rolling(7, center=True, min_periods=1).median()
    wk = (dau_filled - tr).groupby(dau_filled.index.dayofweek).transform("median")
    daily["dau_trend"] = tr + wk
    method = "rolling median"
daily["dau_z"] = robust_z(daily["dau"] - daily["dau_trend"])
print(f"\nDAU 분해: {method}")


def resid_z(s, win=15):
    return robust_z(s - s.rolling(win, center=True, min_periods=3).median())


daily["new_z"] = resid_z(daily["new_chars"])
daily["gone_z"] = resid_z(daily["gone_chars"])
start, end = daily.index.min(), daily.index.max()
daily.loc[daily.index < start + pd.Timedelta(days=EDGE_DAYS), "new_z"] = np.nan
daily.loc[daily.index > end - pd.Timedelta(days=EDGE_DAYS), "gone_z"] = np.nan
daily.loc[daily["missing"], ["dau_z", "new_z", "gone_z"]] = np.nan

ev = pd.Series(EVENTS)
ev.index = pd.to_datetime(ev.index)
missing_days = daily.index[daily["missing"]]


def explain(d):
    reasons = []
    if daily.at[d, "partial"]:
        reasons.append(f"partial collection ({int(daily.at[d, 'observed_hours'])}h)")
    if d in outage_days:
        reasons.append("outage/maintenance day")
    if ((missing_days >= d - pd.Timedelta(days=1)) & (missing_days <= d + pd.Timedelta(days=1))).any():
        reasons.append("adjacent to missing day")
    near = ev[(ev.index >= d - pd.Timedelta(days=EVENT_MATCH_DAYS))
              & (ev.index <= d + pd.Timedelta(days=EVENT_MATCH_DAYS))]
    reasons += list(near.values)
    return "; ".join(reasons)


rows = []
for col, kind in [("dau_z", "DAU"), ("new_z", "new_chars"), ("gone_z", "gone_chars")]:
    for d, r in daily[daily[col].abs() >= Z_THRESHOLD].iterrows():
        rows.append({"date": d, "weekday": d.day_name()[:3], "metric": kind,
                     "z": round(r[col], 1), "direction": "spike" if r[col] > 0 else "drop",
                     "dau": r["dau"], "new_chars": r["new_chars"],
                     "gone_chars": r["gone_chars"], "obs_hours": r["observed_hours"]})
time_anom = pd.DataFrame(rows)
if len(time_anom):
    time_anom["explained_by"] = time_anom["date"].apply(explain)
    time_anom["explained"] = time_anom["explained_by"] != ""
    time_anom = time_anom.sort_values(["date", "metric"])
    n_ue = int((~time_anom["explained"]).sum())
    print(f"\n시점 이상 (|z| ≥ {Z_THRESHOLD}): {len(time_anom)}건 "
          f"(설명됨 {len(time_anom) - n_ue} / 설명 안 됨 {n_ue})")
    print("\n[설명 안 되는 이상]")
    print(time_anom[~time_anom["explained"]].drop(columns=["explained_by", "explained"])
          .to_string(index=False))
    print("\n[설명되는 이상]")
    print(time_anom[time_anom["explained"]][["date", "metric", "z", "explained_by"]]
          .to_string(index=False))
else:
    n_ue = 0
    print("시점 이상 없음")
save_table(time_anom, "C2_time_anomalies", index=False)
save_table(daily, "C_daily_metrics")

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
for ax, (col, zc, title) in zip(axes, [("dau", "dau_z", "DAU"),
                                       ("new_chars", "new_z", "new characters / day"),
                                       ("gone_chars", "gone_z", "last-seen characters / day")]):
    ax.plot(daily.index, daily[col], lw=1, color="#4C72B0")
    if col == "dau":
        ax.plot(daily.index, daily["dau_trend"], lw=1, color="#888888", ls="--")
    for d in daily.index[daily["missing"]]:
        ax.axvspan(d, d + pd.Timedelta(days=1), color="orange", alpha=0.3, lw=0)
    for d in daily.index[daily["partial"]]:
        ax.axvspan(d, d + pd.Timedelta(days=1), color="gold", alpha=0.15, lw=0)
    if len(time_anom):
        t = time_anom[time_anom["metric"] == {"dau": "DAU"}.get(col, col)]
        for flag, colr in [(True, "#999999"), (False, "#C44E52")]:
            tt = t[t["explained"] == flag]
            ax.scatter(tt["date"], daily.loc[tt["date"], col], color=colr, s=30, zorder=3,
                       label="explained" if flag else "unexplained")
    for d in ev.index:
        ax.axvline(d, color="purple", ls=":", lw=1)
    ax.set_title(title)
axes[0].legend(loc="upper left")
fig.suptitle("Time anomalies v2 (orange = missing, gold = partial day, purple = event)")
save(fig, "C_time_anomalies")


# %% ============================================================
# C. 코호트 심층 분석 (10월 8~10일)
# ===============================================================
section(f"C. 코호트 분석: {COHORT_START} ~ {COHORT_END}")


def cohort_stats(s, e, name):
    s, e = pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta(days=1)
    new = feat[(feat["first_seen"] >= s) & (feat["first_seen"] < e)]
    gone = feat[(feat["last_seen"] >= s) & (feat["last_seen"] < e)]
    both = new.index.intersection(gone.index)
    gone_old = gone[gone["first_seen"] < s]          # 이전부터 있다가 이 구간에 사라진 캐릭터

    # ID 재매핑 테스트: 사라진 기존 캐릭터와 같은 (종족, 직업, 길드, 레벨±1) 신규 캐릭터가 있는가
    a = gone_old.reset_index()[["char", "race", "charclass", "last_guild", "last_level"]]
    b = new.reset_index()[["char", "race", "charclass", "first_guild", "first_level"]]
    a = a[a["last_guild"] != -1]
    m = a.merge(b, left_on=["race", "charclass", "last_guild"],
                right_on=["race", "charclass", "first_guild"], suffixes=("_old", "_new"))
    m = m[(m["first_level"] - m["last_level"]).abs() <= 1]
    matched_old = m["char_old"].nunique()

    res = pd.Series({
        "new_chars": len(new),
        "gone_chars": len(gone),
        "new_and_gone_same_window": len(both),
        "new: only 1 day observed": (new["n_days"] == 1).mean(),
        "new: first_level median": new["first_level"].median(),
        "new: first_level > 10 share": (new["first_level"] > 10).mean(),
        "new: guild at first sight": (new["first_guild"] != -1).mean(),
        "new: median play hours": new["play_hours"].median(),
        "new: id_collision share": new["id_collision"].mean(),
        "gone_old (existing chars that vanished)": len(gone_old),
        "gone_old in guild": len(a),
        "gone_old matched to new (race,class,guild,lv±1)": matched_old,
        "match rate": matched_old / len(a) if len(a) else np.nan,
    }, name=name)
    return res, new, m


res_c, new_c, match_c = cohort_stats(COHORT_START, COHORT_END, "cohort")
res_b, new_b, _ = cohort_stats(BASELINE_START, BASELINE_END, "baseline")
cohort_tbl = pd.concat([res_c, res_b], axis=1)
print(cohort_tbl.round(3).to_string())
save_table(cohort_tbl, "C3_cohort_vs_baseline")
save_table(match_c, "C3_cohort_id_remap_matches", index=False)

print("\n코호트 신규 캐릭터 첫 레벨 분포")
print(pd.cut(new_c["first_level"], [0, 1, 10, 30, 60, 69, 70, 80]).value_counts().sort_index())
print("\n코호트 신규 캐릭터 종족·직업 상위")
print(new_c.groupby(["race", "charclass"], observed=True).size()
      .sort_values(ascending=False).head(10))

ids = set(new_c.index)
coh_rows = df[df["char"].isin(ids) & (df["date"] >= pd.Timestamp(COHORT_START))
              & (df["date"] <= pd.Timestamp(COHORT_END))]
print("\n코호트 신규 캐릭터의 구간 내 관측 시간대(시간별 행 수)")
print(coh_rows["hour"].value_counts().sort_index().to_string())
print("\n코호트 신규 캐릭터 체류 지역 상위")
print(coh_rows["zone"].value_counts().head(10))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (lab, nd) in zip([axes[0], axes[0]], [("cohort", new_c), ("baseline", new_b)]):
    ax.hist(nd["first_level"], bins=range(1, 82, 2), alpha=0.6, density=True, label=lab)
axes[0].set_title("first level of new characters")
axes[0].legend()
for lab, nd in [("cohort", new_c), ("baseline", new_b)]:
    axes[1].hist(np.log10(nd["play_hours"]), bins=40, alpha=0.6, density=True, label=lab)
axes[1].set_title("log10 total play hours")
axes[1].legend()
axes[2].bar(range(24), coh_rows["hour"].value_counts().reindex(range(24), fill_value=0).values)
axes[2].set_title("cohort observations by hour (in window)")
save(fig, "C3_cohort_analysis")


# %% ============================================================
# 요약
# ===============================================================
section("요약")
lab = cand["label"].value_counts()
bt = cand.loc[is_bot, "bot_type"].value_counts()
print(f"A. 중복 {n_dup:,}행 제거 / ID 충돌 의심 {len(id_collision):,}명 제외 / "
      f"입장 경고 {len(viol):,}행")
print(f"B. 판정 대상 {len(cand):,}명 → 봇 의심 {lab.get('bot_suspect', 0)} "
      f"({', '.join(f'{k} {v}' for k, v in bt.items())}) / 상주 {lab.get('resident', 0)} / "
      f"경매·은행 {lab.get('bank_alt', 0)} / 도시 잠수 {lab.get('afk_city', 0)}")
print(f"C. 누락 {int(daily['missing'].sum())}일, 부분수집 {int(daily['partial'].sum())}일, "
      f"점검·장애 {len(outages)}건 / 급변 {len(time_anom)}건 중 설명 안 됨 {n_ue}건")
print(f"   코호트 매칭률 {res_c['match rate']:.1%} vs 평상시 {res_b['match rate']:.1%} "
      f"(높으면 ID 재매핑 의심)")
print(f"\n결과 폴더: {OUT_DIR.resolve()}")