# -*- coding: utf-8 -*-
"""
STEP 4. 조기 이탈 예측 (생성 후 72시간 행동 -> 30일 이탈 예측)

설계
 - 대상: STEP 3 최종 신규 코호트 (cohort.parquet)
 - 특징: 생성 후 72시간 안의 세션만 사용 (그 이후 정보가 섞이면 '미래를 보고 맞히는' 누수)
 - 정답: 30일 이탈 = 생성 후 30일이 되기 전에 마지막으로 접속
 - 검증: 시간 순서 분할 (1~5월 생성으로 학습 -> 6~8월 생성으로 평가)
         실제 운영처럼 '과거로 학습해서 미래 신규를 예측'하는지 확인
 - 모델: 로지스틱 회귀(해석용) vs 트리 부스팅(성능용)
 - 평가: ROC-AUC, PR-AUC + 운영 지표(위험 상위 20%에 실제 이탈자가 얼마나 몰렸나)

실행: python step4.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.inspection import permutation_importance

try:
    from lightgbm import LGBMClassifier
    def make_tree():
        return LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15,
                              min_child_samples=40, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, random_state=42, verbose=-1)
    TREE_NAME = "LightGBM"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    def make_tree():
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=42)
    TREE_NAME = "HistGradientBoosting"

BASE = (Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()).parent  # churn_project 폴더
OUT_DIR = BASE / "step4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OBS_H = 72            # 관찰 시간: 생성 후 72시간
CHURN_DAY = 30        # 30일 이탈
TRAIN_END = pd.Timestamp("2008-06-01")   # 이 전에 생성 = 학습, 이후 = 평가
SNAP_H = 10 / 60

log = []
def say(m=""):
    print(m); log.append(str(m))

coh = pd.read_parquet(BASE / "step3" / "cohort.parquet")
sess = pd.read_parquet(BASE / "step2" / "sessions.parquet")
sess = sess[sess["life_id"].isin(coh["life_id"])].merge(coh[["life_id", "first_seen"]], on="life_id")

# ------------------------------------------------------------------
# 1) 72시간 특징 만들기
# ------------------------------------------------------------------
cut = sess["first_seen"] + pd.Timedelta(hours=OBS_H)
s72 = sess[sess["start"] < cut].copy()
# 72시간 경계를 넘는 세션은 경계까지만 계산
s72["end_c"] = np.minimum(s72["end"], cut[s72.index])
s72["hours_c"] = ((s72["end_c"] - s72["start"]).dt.total_seconds() / 3600 + SNAP_H).clip(lower=SNAP_H)
s72 = s72.sort_values(["life_id", "start"])

g = s72.groupby("life_id")
feat = pd.DataFrame({
    "first_sess_h": g["hours_c"].first(),
    "n_sess_72h": g.size(),
    "play_h_72h": g["hours_c"].sum(),
    "level_72h": g["level_end"].max(),
    "zones_72h": g["n_zones"].sum(),
    "guild_72h": g["guild_end"].apply(lambda s: (s >= 0).any()).astype(int),
})
# 두 번째 세션까지 걸린 시간 (없으면 72 = 관찰 기간 내 재접속 없음)
second = s72.groupby("life_id")["start"].apply(lambda s: s.iloc[1] if len(s) > 1 else pd.NaT)
feat["h_to_2nd"] = ((second - coh.set_index("life_id")["first_seen"]).dt.total_seconds() / 3600).reindex(feat.index)
feat["returned_72h"] = feat["h_to_2nd"].notna().astype(int)
feat["h_to_2nd"] = feat["h_to_2nd"].fillna(OBS_H)
feat["lv_per_h"] = (feat["level_72h"] - 1) / feat["play_h_72h"]
# 활동한 서로 다른 날 수 (72시간 = 최대 4개 날짜)
feat["days_72h"] = g["start"].apply(lambda s: s.dt.normalize().nunique())

df = coh.set_index("life_id")[["first_seen", "life_days", "race", "charclass"]].join(feat, how="inner")
df["create_hour"] = df["first_seen"].dt.hour
df["create_weekend"] = (df["first_seen"].dt.dayofweek >= 5).astype(int)
df["race"] = df["race"].astype(str); df["charclass"] = df["charclass"].astype(str)
df["churn30"] = (df["life_days"] < CHURN_DAY).astype(int)

say(f"## 1. 데이터: {len(df):,}명 / 30일 이탈률 {df['churn30'].mean():.1%}")

# ------------------------------------------------------------------
# 2) 한 변수씩 보기: 72시간 행동과 30일 잔존의 관계
# ------------------------------------------------------------------
say("\n## 2. 72시간 행동별 30일 잔존율")
def bucket_report(col, bins, labels):
    b = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    t = df.groupby(b, observed=True).agg(n=("churn30", "size"), retained_d30=("churn30", lambda s: 1 - s.mean()))
    say(f"[{col}]\n" + t.round(3).to_string())
    return t
bucket_report("first_sess_h", [0, .5, 1, 2, 4, 99], ["<30m", "30m-1h", "1-2h", "2-4h", "4h+"])
bucket_report("n_sess_72h", [0, 1, 2, 3, 5, 999], ["1", "2", "3", "4-5", "6+"])
bucket_report("level_72h", [0, 3, 6, 10, 15, 99], ["1-3", "4-6", "7-10", "11-15", "16+"])
bucket_report("days_72h", [0, 1, 2, 3, 4], ["1일", "2일", "3일", "4일"])

# ------------------------------------------------------------------
# 3) 모델: 시간 순서 분할
# ------------------------------------------------------------------
num = ["first_sess_h", "n_sess_72h", "play_h_72h", "level_72h", "zones_72h", "guild_72h",
       "h_to_2nd", "returned_72h", "lv_per_h", "days_72h", "create_hour", "create_weekend"]
cat = ["race", "charclass"]
train = df[df["first_seen"] < TRAIN_END]
test = df[df["first_seen"] >= TRAIN_END]
say(f"\n## 3. 모델 (학습: {TRAIN_END.date()} 이전 생성 {len(train):,}명 / 평가: 이후 생성 {len(test):,}명)")
say(f"이탈률 학습 {train['churn30'].mean():.1%} / 평가 {test['churn30'].mean():.1%}")

pre = ColumnTransformer([("num", StandardScaler(), num),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
models = {
    "Baseline(첫 세션 길이만)": Pipeline([("pre", ColumnTransformer([("n", StandardScaler(), ["first_sess_h"])])),
                                      ("m", LogisticRegression(max_iter=1000))]),
    "LogisticRegression": Pipeline([("pre", pre), ("m", LogisticRegression(max_iter=2000, C=0.5))]),
    TREE_NAME: Pipeline([("pre", ColumnTransformer([("num", "passthrough", num),
                                                     ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])),
                         ("m", make_tree())]),
}
X_tr, y_tr = train[num + cat], train["churn30"]
X_te, y_te = test[num + cat], test["churn30"]
results, probs = [], {}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]
    probs[name] = p
    # PR-AUC는 '잔존(정착) 유저'를 찾는 관점으로도 계산 (소수 클래스)
    results.append({"model": name, "ROC_AUC": roc_auc_score(y_te, p),
                    "PR_AUC_churn": average_precision_score(y_te, p),
                    "PR_AUC_retained": average_precision_score(1 - y_te, 1 - p)})
res = pd.DataFrame(results)
say(res.round(3).to_string(index=False))
res.to_csv(OUT_DIR / "model_results.csv", index=False, encoding="utf-8-sig")
best = res.sort_values("ROC_AUC", ascending=False).iloc[0]["model"]
say(f"최고 모델: {best}")

# ------------------------------------------------------------------
# 4) 운영 관점: 위험 점수 구간별 실제 잔존
# ------------------------------------------------------------------
say("\n## 4. 위험 점수 10분위별 실제 30일 잔존율 (평가 데이터)")
t = pd.DataFrame({"p": probs[best], "y": y_te.values})
t["decile"] = pd.qcut(t["p"].rank(method="first"), 10, labels=[f"D{i}" for i in range(1, 11)])  # D10 = 가장 위험
dec = t.groupby("decile", observed=True).agg(n=("y", "size"), churn_rate=("y", "mean"), mean_score=("p", "mean"))
dec["retained_rate"] = 1 - dec["churn_rate"]
say(dec.round(3).to_string())
dec.to_csv(OUT_DIR / "risk_deciles.csv", encoding="utf-8-sig")

# '정착 가능성이 애매한 중간층'이 개입 대상: 이미 떠난 사람(재접속 없음)도, 확실히 남을 사람도 아닌 구간
say("-> 위험이 가장 낮은 구간은 개입 없이도 남고, 가장 높은 구간은 이미 떠난 경우가 많음."
    " 온보딩 개입은 중간 구간에서 효과가 클 가능성 (STEP 5에서 검토)")

# ------------------------------------------------------------------
# 5) 어떤 행동이 중요한가 (평가 데이터 기준 순열 중요도)
# ------------------------------------------------------------------
say("\n## 5. 변수 중요도 (순열 중요도, ROC-AUC 감소량)")
pi = permutation_importance(models[best], X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=42)
imp = pd.Series(pi.importances_mean, index=num + cat).sort_values(ascending=False)
say(imp.round(4).to_string())
imp.rename("importance").to_csv(OUT_DIR / "feature_importance.csv", encoding="utf-8-sig")

lr = models["LogisticRegression"]
names = num + list(lr.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(cat))
coef = pd.Series(lr.named_steps["m"].coef_[0], index=names).sort_values()
coef.rename("coef(+ = 이탈 위험 증가)").to_csv(OUT_DIR / "logit_coef.csv", encoding="utf-8-sig")

# ------------------------------------------------------------------
# 6) 그래프
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for name, p in probs.items():
    fpr, tpr, _ = roc_curve(y_te, p)
    axes[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_te, p):.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=0.8); axes[0].set_title("ROC (test: Jun-Aug cohorts)")
axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].legend(fontsize=8)
axes[1].bar(dec.index.astype(str), dec["retained_rate"] * 100)
axes[1].set_title(f"D30 retention by risk decile ({best})"); axes[1].set_ylabel("% retained")
top = imp.head(10)[::-1]
axes[2].barh(top.index, top.values); axes[2].set_title("Permutation importance (AUC drop)")
fig.tight_layout(); fig.savefig(OUT_DIR / "01_model_eval.png", dpi=130); plt.close(fig)

df.to_parquet(OUT_DIR / "features_72h.parquet")
(OUT_DIR / "summary.txt").write_text("\n".join(log), encoding="utf-8")
print(f"\n저장 완료: {OUT_DIR}")
