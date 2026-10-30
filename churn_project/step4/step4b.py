# -*- coding: utf-8 -*-
"""
STEP 4b. 더 어려운 조건으로 다시 평가

질문: 모델 성능이 '첫 세션 후 안 돌아온(이미 떠난) 유저를 맞힌 것'에서 대부분 나온 건 아닌가?
확인:
 A) 전체 평가 대상에서 '72시간 내 재접속 여부' 하나만 쓴 기준선 성능
 B) 72시간 안에 다시 들어온 유저만 대상으로 다시 학습·평가 (아직 안 떠난 사람 중 누가 떠날지)
    -> 운영팀이 실제로 개입할 수 있는 대상

실행: python step4b.py  (step4.py 실행 후)
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

BASE = (Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()).parent  # churn_project 폴더
OUT_DIR = BASE / "step4"
TRAIN_END = pd.Timestamp("2008-06-01")

log = []
def say(m=""):
    print(m); log.append(str(m))

df = pd.read_parquet(BASE / "step4" / "features_72h.parquet")
num = ["first_sess_h", "n_sess_72h", "play_h_72h", "level_72h", "zones_72h", "guild_72h",
       "h_to_2nd", "returned_72h", "lv_per_h", "days_72h", "create_hour", "create_weekend"]
cat = ["race", "charclass"]

def fit_eval(data, cols_num, cols_cat, label):
    tr, te = data[data["first_seen"] < TRAIN_END], data[data["first_seen"] >= TRAIN_END]
    trans = [("num", StandardScaler(), cols_num)]
    if cols_cat:
        trans.append(("cat", OneHotEncoder(handle_unknown="ignore"), cols_cat))
    m = Pipeline([("pre", ColumnTransformer(trans)), ("m", LogisticRegression(max_iter=2000, C=0.5))])
    m.fit(tr[cols_num + cols_cat], tr["churn30"])
    p = m.predict_proba(te[cols_num + cols_cat])[:, 1]
    auc = roc_auc_score(te["churn30"], p)
    say(f"  {label:40s} AUC {auc:.3f}  (평가 {len(te):,}명, 이탈률 {te['churn30'].mean():.1%})")
    return m, te, auc

say("## A) 전체 평가 대상")
fit_eval(df, ["returned_72h"], [], "기준선: 72시간 재접속 여부만")
fit_eval(df, ["returned_72h", "days_72h"], [], "재접속 + 접속 일수")
fit_eval(df, num, cat, "전체 모델")

say("\n## B) 72시간 안에 다시 들어온 유저만 (아직 안 떠난 사람)")
ret = df[df["returned_72h"] == 1]
say(f"대상 {len(ret):,}명 (전체의 {len(ret)/len(df):.0%}) / 30일 이탈률 {ret['churn30'].mean():.1%}")
num_b = [c for c in num if c != "returned_72h"]
fit_eval(ret, ["days_72h"], [], "기준선: 접속 일수만")
fit_eval(ret, ["level_72h"], [], "기준선: 72시간 레벨만")
m, te, auc = fit_eval(ret, num_b, cat, "전체 모델")

say("\n재접속 유저 대상 변수 중요도 (순열 중요도, AUC 감소량)")
pi = permutation_importance(m, te[num_b + cat], te["churn30"], scoring="roc_auc", n_repeats=10, random_state=42)
imp = pd.Series(pi.importances_mean, index=num_b + cat).sort_values(ascending=False)
say(imp.round(4).to_string())

t = pd.DataFrame({"p": m.predict_proba(te[num_b + cat])[:, 1], "y": te["churn30"].values})
t["decile"] = pd.qcut(t["p"].rank(method="first"), 10, labels=[f"D{i}" for i in range(1, 11)])
dec = t.groupby("decile", observed=True)["y"].agg(n="size", churn_rate="mean")
dec["retained_rate"] = 1 - dec["churn_rate"]
say("\n재접속 유저 대상 위험 10분위별 실제 30일 잔존")
say(dec.round(3).to_string())

(OUT_DIR / "summary_hard_eval.txt").write_text("\n".join(log), encoding="utf-8")
print(f"\n저장 완료: {OUT_DIR / 'summary_hard_eval.txt'}")
