# 10/30 주 수집에서 블러드엘프 관측이 급감(점유율 35% -> 24%) -> 블러드엘프 제외 후 3.0.2 2주차 효과 재확인
import sys, os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np, p302_v2 as P
race=pd.read_pickle(os.path.join(P.common.BASE,'churn_project/step6/clean.pkl'))[['char','race']].drop_duplicates('char').set_index('char').race.astype(str)
full=P.cd
E=pd.Timestamp('2008-10-23')
d_all=P.daily(E)
P.cd=full[full.char.map(race)!='Blood Elf']; d_nobe=P.daily(E); r_nobe=P.its(E,d_nobe)
P.cd=full[full.char.map(race)=='Blood Elf']; d_be=P.daily(E)
print('주별 DAU (기존 캐릭터): 전체 / 블러드엘프 / 그 외')
print(pd.DataFrame({'all':d_all.dau,'BE':d_be.dau,'nonBE':d_nobe.dau}).resample('W-WED').mean().round(0).loc['2008-09-24':].to_string())
P.cd=full[full.char.map(race)!='Blood Elf']
cands=[t for t in pd.date_range('2008-02-21','2008-10-09',freq='W-THU') if all(not(-49<=(p-t).days<=14) for p in P.REAL)]
pl=pd.DataFrame({t.strftime('%m/%d'):P.its(t) for t in cands}).T
rows=[]
for mn in P.MODELS:
    for m in ['dau','play_hours','returners']:
        for w in ['w1','w2']:
            v=pl[(mn,m,w)]; r=r_nobe[(mn,m,w)]
            rows.append({'model':mn,'metric':m,'week':w,'real':r,'placebo_mean':v.mean(),'adj':r-v.mean(),'z':(r-v.mean())/v.std()})
R=pd.DataFrame(rows); print('\n블러드엘프 제외'); print(R.round(3).to_string(index=False))
R.to_csv(os.path.dirname(os.path.abspath(__file__))+'/p302_noBE.csv',index=False)
