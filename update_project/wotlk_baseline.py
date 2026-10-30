# 확장팩 효과의 기준 주 민감도: 출시 주(11/20~26) vs 여러 기준 주. 10/7~10/26 급증 캐릭터 제외
import pandas as pd, numpy as np
df=pd.read_pickle('churn_project/step6/clean.pkl')[['char','timestamp']]
first=df.groupby('char').timestamp.min(); surge=first[(first>='2008-10-07')&(first<'2008-10-27')].index
ts=np.sort(df.timestamp.unique()); rid=np.r_[0,np.cumsum(np.diff(ts)>np.timedelta64(5,'m'))]
df['round']=pd.Series(rid,index=ts).reindex(df.timestamp).values
rt=df.groupby('round').timestamp.min().sort_index()
w=(rt.shift(-1)-rt).dt.total_seconds().clip(upper=1800).fillna(600)/3600
df['h']=df['round'].map(w); df['day']=df.timestamp.dt.normalize()
cov=pd.Series(w.values,index=rt.dt.normalize().values).groupby(level=0).sum(); good=set(cov[cov>=20].index)
df=df[~df.char.isin(surge)]
def week(S):
    S=pd.Timestamp(S); x=df[(df.day>=S)&(df.day<S+pd.Timedelta('7D'))]
    gd=[d for d in pd.date_range(S,periods=7) if d in good]
    hrs=x[x.day.isin(gd)].h.sum()/len(gd)*7
    return {'active_chars':x.char.nunique(),'weekly_hours':hrs,'h_per_char':hrs/x.char.nunique(),'good_days':len(gd)}
L=week('2008-11-20')
rows={}
for lab,S in [('9/22-28 (평범한 주)','2008-09-22'),('9/29-10/5 (평범한 주)','2008-09-29'),('10/27-11/2 (3.0.2 이후)','2008-10-27'),('11/10-16 (출시 직전, 기존 기준)','2008-11-10')]:
    b=week(S); rows[lab]={**b,'launch_vs_base_chars':L['active_chars']/b['active_chars']-1,'launch_vs_base_hours':L['weekly_hours']/b['weekly_hours']-1}
rows['출시 주 11/20-26']=L
r=pd.DataFrame(rows).T; pd.set_option('display.width',200); print(r.round(3).to_string()); r.to_csv('update_project/wotlk_baseline_sensitivity.csv')
# 출시 후 6주 추이 (같은 기준으로)
print('\n출시 후 주별')
for k in range(6):
    S=pd.Timestamp('2008-11-20')+pd.Timedelta(f'{7*k}D'); 
    if S+pd.Timedelta('6D')>pd.Timestamp('2008-12-31'): break
    v=week(S); print(S.date(), {kk:round(vv,1) for kk,vv in v.items()})
