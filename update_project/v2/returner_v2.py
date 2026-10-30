# 복귀 유저 교란 통제: 로지스틱 회귀(캐릭터 군집 강건 표준오차)
# (A) 리치왕 복귀자 780명: 4~6주 뒤 정착 ~ 첫 주 행동 + 레벨·휴면 기간·휴면 전 길드·직업
# (B) 연중 복귀 이벤트: 4~6주 잔존 ~ 2일차 재접속 + 레벨·휴면 기간·휴면 전 활동량·시기  (A/B 설계의 근거 점검)
import sys, os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import common
import pandas as pd, numpy as np
OUT=os.path.dirname(os.path.abspath(__file__))+'/'

def logit(X,y,cl=None,names=None,it=50):
    X=np.asarray(X,float); y=np.asarray(y,float); b=np.zeros(X.shape[1])
    for _ in range(it):
        p=1/(1+np.exp(-X@b)); W=p*(1-p); H=X.T@(X*W[:,None]); g=X.T@(y-p); s=np.linalg.solve(H,g); b+=s
        if np.abs(s).max()<1e-9: break
    p=1/(1+np.exp(-X@b)); Hi=np.linalg.inv(X.T@(X*(p*(1-p))[:,None])); u=X*(y-p)[:,None]
    if cl is not None: u=pd.DataFrame(u).groupby(np.asarray(cl)).sum().values
    V=Hi@(u.T@u)@Hi; se=np.sqrt(np.diag(V))
    # 평균 한계효과(이진 변수는 0->1 전환 시 예측확률 차이)
    ame=[]
    for j in range(X.shape[1]):
        if set(np.unique(X[:,j]))<={0.0,1.0} and j>0:
            X1=X.copy();X1[:,j]=1;X0=X.copy();X0[:,j]=0; ame.append((1/(1+np.exp(-X1@b))-1/(1+np.exp(-X0@b))).mean())
        else: ame.append((p*(1-p)).mean()*b[j])
    return pd.DataFrame({'coef':b,'se':se,'OR':np.exp(b),'OR_lo':np.exp(b-1.96*se),'OR_hi':np.exp(b+1.96*se),'z':b/se,'AME':ame},index=names)

def design(df,cols,cats=()):
    X=df[list(cols)].astype(float).copy()
    for c in cats: X=X.join(pd.get_dummies(df[c],prefix=c,drop_first=True).astype(float))
    X.insert(0,'const',1.0); return X

pd.set_option('display.width',220)
# ---------------- (A) 리치왕 복귀자 ----------------
R=pd.read_csv(os.path.join(OUT,'..','returners_wotlk.csv'))
R['lv70']=(R.lv_start>=70).astype(int); R['lv_lt60']=(R.lv_start<60).astype(int)
R['log_dormant']=np.log(R.dormant_days); R['d2plus']=(R.days_w1>=2).astype(int); R['log_hours']=np.log1p(R.hours_w1)
R['gm_any']=(R.guildmates>0).astype(int); R['guild_before']=R.guild_before.astype(int); R['guild_w1']=R.guild_w1.astype(int)
R['settled']=R.settled.astype(int)
specs={
 'A1 길드만':['guild_w1'],
 'A2 +레벨·휴면·휴면전길드':['guild_w1','lv70','lv_lt60','log_dormant','guild_before'],
 'A3 +첫주 접속일 2일+':['guild_w1','lv70','lv_lt60','log_dormant','guild_before','d2plus'],
 'A4 +첫주 플레이시간':['guild_w1','lv70','lv_lt60','log_dormant','guild_before','d2plus','log_hours'],
}
resA={}
for k,cols in specs.items():
    X=design(R,cols,cats=['cls'] if k!='A1 길드만' else ()); t=logit(X.values,R.settled,names=X.columns)
    resA[k]=t; print(f'\n[{k}] n={len(R)}'); print(t.drop(index=[c for c in t.index if c.startswith('cls_')]).round(3).to_string())
pd.concat(resA).round(4).to_csv(OUT+'returner_logit_A.csv')
# ---------------- (B) 연중 복귀 이벤트 ----------------
o=common.load(); cd,cov,info=o['cd'],o['cov'],o['info']
first=info['first']; surge=first[(first>='2008-10-07')&(first<'2008-10-27')].index
cd=cd[~cd.char.isin(surge)].sort_values(['char','day']).reset_index(drop=True)
cd['prev']=cd.groupby('char').day.shift(1); cd['gap']=(cd.day-cd.prev).dt.days
bad=set(cov.index[cov<12])
E=cd[(cd.gap>=15)&(cd.day>='2008-02-01')&(cd.day<='2008-11-20')].copy()
E=E[~E.day.isin(bad)&~(E.day+pd.Timedelta('1D')).isin(bad)&~(E.day+pd.Timedelta('2D')).isin(bad)]
dn=cd.day.values.astype('datetime64[D]').astype(np.int64); ch=cd.char.values
start=pd.Series(np.arange(len(cd))).groupby(ch).min(); end=pd.Series(np.arange(len(cd))).groupby(ch).max()+1
def anyin(c,d0,lo,hi):
    s,e=start[c],end[c]; a=dn[s:e]; return ((a>=d0+lo)&(a<=d0+hi)).any()
def cnt(c,lo_d,hi_d):
    s,e=start[c],end[c]; a=dn[s:e]; return ((a>=lo_d)&(a<=hi_d)).sum()
ev_d=E.day.values.astype('datetime64[D]').astype(np.int64); pv=E.prev.values.astype('datetime64[D]').astype(np.int64)
E['d12']=[anyin(c,d,1,2) for c,d in zip(E.char,ev_d)]
E['long']=[anyin(c,d,28,41) for c,d in zip(E.char,ev_d)]
E['pre_days']=[cnt(c,p-27,p) for c,p in zip(E.char,pv)]
E['lv70']=(E.lv>=70).astype(int); E['lv_lt60']=(E.lv<60).astype(int); E['log_gap']=np.log(E.gap)
E['month']=E.day.dt.month.astype(str); E['d12']=E.d12.astype(int); E['long']=E.long.astype(int)
print(f'\n(B) 복귀 이벤트 {len(E):,}건 / 캐릭터 {E.char.nunique():,}명, 2일차 재접속 {E.d12.mean():.1%}, 4~6주 잔존 {E.long.mean():.1%}')
resB={}
for k,cols in {'B1 2일차만':['d12'],'B2 +레벨·휴면기간·휴면전활동·월':['d12','lv70','lv_lt60','log_gap','pre_days']}.items():
    X=design(E,cols,cats=['month'] if k.startswith('B2') else ()); t=logit(X.values,E.long,cl=E.char.values,names=X.columns)
    resB[k]=t; print(f'\n[{k}]'); print(t.drop(index=[c for c in t.index if c.startswith('month_')]).round(3).to_string())
# 하위 집단별: 70레벨만
for nm,sub in [('70레벨',E[E.lv70==1]),('60 미만',E[E.lv_lt60==1])]:
    X=design(sub,['d12','log_gap','pre_days'],cats=['month']); t=logit(X.values,sub.long,cl=sub.char.values,names=X.columns)
    resB['B3 '+nm]=t; print(f'\n[B3 {nm}만 n={len(sub):,}]'); print(t.loc[['d12','log_gap','pre_days']].round(3).to_string())
pd.concat(resB).round(4).to_csv(OUT+'returner_logit_B.csv')
