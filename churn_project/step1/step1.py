# -*- coding: utf-8 -*-
"""
STEP 1. 데이터 불러오기 + 품질 점검 + 분석 단위(캐릭터 생애) 만들기

데이터: World of Warcraft Avatar History (Lee et al., 2011 / Kaggle)
 - 10분마다 '그 시각에 접속해 있던 캐릭터'를 기록한 스냅샷
 - 한 줄 = (캐릭터 ID, 레벨, 종족, 직업, 지역, 길드, 시각)

이 단계에서 하는 일
 1) 원본 로드, 컬럼 정리
 2) 품질 점검 (게임 규칙상 불가능한 기록 찾기)
    - 같은 시각 중복 기록
    - 종족/직업이 바뀐 기록      (게임에서 불가능)
    - 레벨이 내려간 기록          (게임에서 불가능)
    - 짧은 시간에 레벨이 크게 뛴 기록
 3) 수집이 비어 있는 시간대 확인 (서버 점검/수집 누락)
 4) ID 충돌 분리: 하나의 ID를 서로 다른 캐릭터가 쓴 경우를
    '캐릭터 생애(life)' 단위로 나눔 -> 이후 모든 분석의 기본 단위
 5) 정제된 데이터 저장

실행: python step1.py
"""
from pathlib import Path
import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# 설정
# ------------------------------------------------------------------
DATA_DIR = (Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()).parents[1]  # archive 폴더 (이 스크립트 기준 두 단계 위)
OUT_DIR = DATA_DIR / "churn_project" / "step1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 레벨이 이만큼 이상 '떨어지면' 다른 캐릭터로 판단 (레벨은 원래 떨어질 수 없음)
LEVEL_DROP_SPLIT = 2
# 이 시간 안에 이만큼 이상 레벨이 오르면 '비정상 점프'로 표시만 함
JUMP_HOURS, JUMP_LEVELS = 1.0, 5

log = []
def say(msg=""):
    print(msg)
    log.append(str(msg))

# ------------------------------------------------------------------
# 1) 원본 로드
# ------------------------------------------------------------------
cands = [p for p in DATA_DIR.glob("*.csv") if "wowah" in p.name.lower()]
if not cands:
    raise FileNotFoundError(f"{DATA_DIR} 에 wowah 로 시작하는 csv가 없습니다. 파일명을 확인하세요.")
RAW = max(cands, key=lambda p: p.stat().st_size)  # 샘플이 아닌 전체 데이터 선택
say(f"## 1. 원본 로드: {RAW.name}")

df = pd.read_csv(RAW, skipinitialspace=True)
df.columns = [c.strip().lower() for c in df.columns]          # ' level' -> 'level'
df = df.rename(columns={"class": "charclass"})
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%y %H:%M:%S")
for c in ["race", "charclass", "zone"]:
    df[c] = df[c].astype(str).str.strip().astype("category")
df["level"] = df["level"].astype("int16")
df["guild"] = df["guild"].astype("int32")                       # -1 = 길드 없음
df["char"] = df["char"].astype("int32")

say(f"행 수 {len(df):,} / 캐릭터 ID {df['char'].nunique():,}")
say(f"기간 {df['timestamp'].min()} ~ {df['timestamp'].max()}")
say(f"컬럼 {list(df.columns)}")

# ------------------------------------------------------------------
# 2) 품질 점검
# ------------------------------------------------------------------
say("\n## 2. 품질 점검")

# 2-1) 같은 캐릭터가 같은 시각에 두 번 찍힌 기록
dup_mask = df.duplicated(subset=["char", "timestamp"], keep="first")
say(f"[중복] 같은 캐릭터·같은 시각 중복 행: {dup_mask.sum():,}")
df = df[~dup_mask].sort_values(["char", "timestamp"]).reset_index(drop=True)

# 이전 기록과 비교하기 위한 컬럼
g = df.groupby("char", sort=False)
same_char = df["char"].eq(df["char"].shift())
prev_level = df["level"].shift()
prev_race = df["race"].astype(str).shift()
prev_class = df["charclass"].astype(str).shift()
gap_h = (df["timestamp"] - df["timestamp"].shift()).dt.total_seconds() / 3600

# 2-2) 종족/직업 변경 (불가능)
rc_change = same_char & ((df["race"].astype(str) != prev_race) | (df["charclass"].astype(str) != prev_class))
say(f"[종족/직업 변경] 행 {rc_change.sum():,} / 캐릭터 {df.loc[rc_change, 'char'].nunique():,}")

# 2-3) 레벨 하락 (불가능)
lv_diff = df["level"] - prev_level
lv_drop = same_char & (lv_diff < 0)
say(f"[레벨 하락] 행 {lv_drop.sum():,} / 캐릭터 {df.loc[lv_drop, 'char'].nunique():,}")
drop_size = (-lv_diff[lv_drop]).astype(int)
say(f"    하락 폭 분포: 1레벨 {int((drop_size==1).sum())}, 2~4레벨 {int(drop_size.between(2,4).sum())}, 5레벨 이상 {int((drop_size>=5).sum())}")

# 2-4) 짧은 시간 안의 큰 레벨 점프 (표시만)
jump = same_char & (lv_diff >= JUMP_LEVELS) & (gap_h <= JUMP_HOURS) & ~rc_change
say(f"[레벨 점프] {JUMP_HOURS}시간 내 {JUMP_LEVELS}레벨 이상 상승: 행 {jump.sum():,} / 캐릭터 {df.loc[jump, 'char'].nunique():,}")

# 두 문제가 같은 캐릭터에서 겹치는지 -> 겹치면 'ID 충돌' 근거
c_rc = set(df.loc[rc_change, "char"]); c_drop = set(df.loc[lv_drop, "char"])
say(f"    종족/직업 변경 & 레벨 하락 둘 다 있는 캐릭터: {len(c_rc & c_drop):,}  (겹침이 크면 'ID 하나를 여러 캐릭터가 공유'했다는 신호)")

# ------------------------------------------------------------------
# 3) 수집 공백 확인 (서버 점검/수집 누락)
# ------------------------------------------------------------------
say("\n## 3. 수집 공백")
hourly = df.set_index("timestamp").resample("h")["char"].count()
full_idx = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
hourly = hourly.reindex(full_idx, fill_value=0)
empty_hours = hourly[hourly == 0]
daily_obs_hours = (hourly > 0).groupby(hourly.index.date).sum()
say(f"기록이 0건인 시간: {len(empty_hours):,}시간 / 전체 {len(hourly):,}시간")
say(f"24시간 다 기록된 날 {int((daily_obs_hours==24).sum())}일, 일부만 기록된 날 {int(((daily_obs_hours>0)&(daily_obs_hours<24)).sum())}일, 기록 없는 날 {int((daily_obs_hours==0).sum())}일")
daily_obs_hours.rename("observed_hours").to_csv(OUT_DIR / "daily_observed_hours.csv", encoding="utf-8-sig")

# ------------------------------------------------------------------
# 4) ID 충돌 분리 -> 캐릭터 생애(life) 단위 만들기
#    같은 ID 안에서 '종족/직업이 바뀌거나 레벨이 LEVEL_DROP_SPLIT 이상 떨어지면'
#    새 캐릭터가 시작된 것으로 보고 생애 번호를 올림
# ------------------------------------------------------------------
say("\n## 4. ID 충돌 분리")
new_life = (~same_char) | rc_change | (same_char & (lv_diff <= -LEVEL_DROP_SPLIT))
life_idx = new_life.astype(int).groupby(df["char"]).cumsum() - 1
df["life_id"] = (df["char"].astype("int64") * 1000 + life_idx).astype("int64")
n_split_points = int((new_life & same_char).sum())
say(f"분리 지점 {n_split_points:,}개 → 분석 단위(캐릭터 생애) {df['life_id'].nunique():,}개 (원래 ID {df['char'].nunique():,}개)")

# 분리 후에도 남은 1레벨 하락(노이즈) 건수
lv_diff2 = df["level"] - df.groupby("life_id")["level"].shift()
say(f"분리 후 남은 레벨 하락 행(1레벨 하락 등 경미한 노이즈): {int((lv_diff2 < 0).sum()):,}")

# ------------------------------------------------------------------
# 5) 저장
# ------------------------------------------------------------------
issues = pd.DataFrame({
    "char": df["char"], "timestamp": df["timestamp"],
    "race_class_change": rc_change, "level_drop": lv_drop, "level_jump": jump,
})
issues = issues[issues[["race_class_change", "level_drop", "level_jump"]].any(axis=1)]
issues.to_csv(OUT_DIR / "quality_issue_rows.csv", index=False, encoding="utf-8-sig")

clean_path = OUT_DIR / "wowah_clean"
try:
    df.to_parquet(clean_path.with_suffix(".parquet"), index=False)
    saved = clean_path.with_suffix(".parquet")
except Exception:
    df.to_pickle(clean_path.with_suffix(".pkl"))
    saved = clean_path.with_suffix(".pkl")
say(f"\n정제 데이터 저장: {saved}")

(OUT_DIR / "summary.txt").write_text("\n".join(log), encoding="utf-8")
print(f"\n요약 저장: {OUT_DIR / 'summary.txt'}")
