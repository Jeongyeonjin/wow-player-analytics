# -*- coding: utf-8 -*-
"""
STEP 2. 접속 세션 만들기 + 캐릭터 생애별 요약표

핵심 아이디어
 - 기록은 10분마다 '접속 중인 캐릭터'를 찍은 것
 - 같은 캐릭터가 연속된 수집 회차(round)에 계속 찍히면 같은 세션
 - 수집 회차를 2번 이상 연속으로 놓치면(= 서버는 기록 중인데 이 캐릭터가 없으면) 세션 종료
 - 서버 점검·수집 중단으로 '기록 자체가 없던 시간'은 결석으로 보지 않음
   -> 수집 공백 때문에 세션이 잘못 끊기는 것을 방지

출력
 - sessions.parquet : 세션 단위 표 (시작, 끝, 길이, 레벨 변화, 방문 지역 수)
 - lives.parquet    : 캐릭터 생애 단위 표 (첫/마지막 접속, 세션 수, 총 플레이, 길드 등)
 - summary.txt

실행: python step2.py
"""
from pathlib import Path
import pandas as pd
import numpy as np

BASE = (Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()).parent  # churn_project 폴더
IN_PATH = BASE / "step1" / "wowah_clean.parquet"
OUT_DIR = BASE / "step2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROUND_GAP_MIN = 5      # 전체 기록 시각 사이가 5분 넘게 벌어지면 '새 수집 회차'
MISS_ROUNDS = 2        # 이 캐릭터가 수집 회차를 2번 이상 연속 놓치면 세션 종료
SNAP_H = 10 / 60       # 한 번 찍힘 = 10분

log = []
def say(m=""):
    print(m); log.append(str(m))

df = pd.read_parquet(IN_PATH) if IN_PATH.exists() else pd.read_pickle(IN_PATH.with_suffix(".pkl"))
df = df.sort_values(["life_id", "timestamp"]).reset_index(drop=True)
say(f"## 입력: {len(df):,}행 / 캐릭터 생애 {df['life_id'].nunique():,}개")

# ------------------------------------------------------------------
# 1) 수집 회차(round) 만들기
#    전체 기록 시각을 정렬해서, 5분 넘게 끊기는 지점마다 새 회차로 봄
# ------------------------------------------------------------------
ts_all = np.sort(df["timestamp"].unique())
new_round = np.r_[True, np.diff(ts_all).astype("timedelta64[s]").astype(int) > ROUND_GAP_MIN * 60]
round_start = ts_all[new_round]
say(f"수집 회차 수: {len(round_start):,}회 (10분마다 1회라면 1년 약 52,700회)")

# 각 기록이 몇 번째 회차인지
df["round"] = np.searchsorted(round_start, df["timestamp"].values, side="right") - 1

# ------------------------------------------------------------------
# 2) 세션 나누기
#    같은 생애 안에서 이전 기록과의 회차 차이가 MISS_ROUNDS 이상이면 새 세션
#    회차 차이 = 그 사이에 서버가 실제로 수집한 횟수 기준이라
#    수집이 멈춘 시간은 자동으로 결석에서 빠짐
# ------------------------------------------------------------------
same = df["life_id"].eq(df["life_id"].shift())
round_gap = df["round"] - df["round"].shift()
new_sess = (~same) | (round_gap > MISS_ROUNDS)
df["sess_no"] = new_sess.astype(int).groupby(df["life_id"]).cumsum()

# 참고: 시간으로만 끊었다면 몇 개가 더 끊겼을지 (수집 공백 보정 효과)
time_gap_min = (df["timestamp"] - df["timestamp"].shift()).dt.total_seconds() / 60
naive_new = (~same) | (time_gap_min > (MISS_ROUNDS + 1) * 10)
say(f"세션 수: 회차 기준 {int(new_sess.sum()):,}개 / 단순 시간 기준이었다면 {int(naive_new.sum()):,}개")
say(f"  -> 수집 공백 때문에 잘못 끊길 뻔한 세션 {int(naive_new.sum() - new_sess.sum()):,}개를 보정")

sess = df.groupby(["life_id", "sess_no"]).agg(
    start=("timestamp", "min"),
    end=("timestamp", "max"),
    n_snap=("timestamp", "size"),
    level_start=("level", "first"),
    level_end=("level", "last"),
    n_zones=("zone", "nunique"),
    guild_end=("guild", "last"),
).reset_index()
sess["hours"] = sess["n_snap"] * SNAP_H                  # 찍힌 횟수 x 10분
sess["levels_gained"] = sess["level_end"] - sess["level_start"]
sess.to_parquet(OUT_DIR / "sessions.parquet", index=False)

say("\n## 세션 길이 분포")
q = sess["hours"].quantile([.25, .5, .75, .9, .99])
say(f"중앙값 {q[.5]:.2f}h / 25% {q[.25]:.2f}h / 75% {q[.75]:.2f}h / 90% {q[.9]:.2f}h / 99% {q[.99]:.2f}h")
say(f"10분(한 번만 찍힘) 세션 비율: {(sess['n_snap']==1).mean():.1%}")

# 시작 시각 / 요일 분포
sess["start_hour"] = sess["start"].dt.hour
sess["start_wd"] = sess["start"].dt.dayofweek   # 0=월
sess.groupby("start_hour").size().rename("sessions").to_csv(OUT_DIR / "session_start_by_hour.csv", encoding="utf-8-sig")
sess.groupby("start_wd").size().rename("sessions").to_csv(OUT_DIR / "session_start_by_weekday.csv", encoding="utf-8-sig")
peak = sess.groupby("start_hour").size().idxmax()
say(f"세션 시작이 가장 많은 시각: {peak}시")

# ------------------------------------------------------------------
# 3) 캐릭터 생애 요약표
# ------------------------------------------------------------------
first = df.groupby("life_id").first()
lives = pd.DataFrame({
    "char": first["char"],
    "race": first["race"], "charclass": first["charclass"],
    "first_seen": first["timestamp"],
    "first_level": first["level"],
    "first_zone": first["zone"],
    "first_guild": first["guild"],
})
g = df.groupby("life_id")
lives["last_seen"] = g["timestamp"].max()
lives["max_level"] = g["level"].max()
lives["n_snap"] = g.size()
lives["play_hours"] = lives["n_snap"] * SNAP_H
lives["active_days"] = g["timestamp"].apply(lambda s: s.dt.normalize().nunique())
lives["ever_guild"] = g["guild"].apply(lambda s: (s >= 0).any())

sg = sess.groupby("life_id")
lives["n_sessions"] = sg.size()
lives["first_sess_hours"] = sg["hours"].first()
lives = lives.reset_index()
lives.to_parquet(OUT_DIR / "lives.parquet", index=False)

say("\n## 캐릭터 생애 요약")
say(f"생애 수 {len(lives):,}")
say(f"레벨 1로 처음 등장한 생애(신규 후보): {(lives['first_level']==1).sum():,}")
say(f"세션 1개뿐인 생애: {(lives['n_sessions']==1).mean():.1%}")
say(f"총 플레이 시간 중앙값: {lives['play_hours'].median():.2f}h / 평균 {lives['play_hours'].mean():.1f}h")

(OUT_DIR / "summary.txt").write_text("\n".join(log), encoding="utf-8")
print(f"\n저장 완료: {OUT_DIR}")
