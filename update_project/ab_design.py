# A/B 실험 설계용 기준값: 복귀/신규 유저의 2일차 재접속률, 하루 유입량, 장기 잔존과의 관계, 필요 표본·기간
import pandas as pd, numpy as np
from math import ceil
from statistics import NormalDist
df=pd.read_pickle('churn_project/step6/clean.pkl')[['char','level','timestamp']]
df['day']=df.timestamp.dt.normalize()
first=df.groupby('char').timestamp.min()
surge=set(first[(first>='2008-10-07')&(first<'2008-10-27')].index)
cd=df.groupby(['char','day'])['level'].max().reset_index()
cd=cd[~cd.char.isin(surge)].sort_values(['char','day'])
cov=pd.read_csv('update_project/daily_metrics.csv',index_col=0,parse_dates=True)
bad=set(cov.index[cov.cov_h<12]) if 'cov_h' in cov else set()
d2=pd.read_csv('update_project/daily_metrics_nosurge.csv',index_col=0,parse_dates=True); bad=set(d2.index[d2.cov_h<12])
cd['prev']=cd.groupby('char').day.shift(1); cd['gap']=(cd.day-cd.prev).dt.days
cd['first_day']=cd.char.map(first).dt.normalize()
# 다음 접속일들
days_by=cd.groupby('char').day.apply(np.array)
def outcome(row,lo,hi):
    a=days_by[row.char]; return bool(((a>=row.day+pd.Timedelta(f'{lo}D'))&(a<=row.day+pd.Timedelta(f'{hi}D'))).any())
UPD=[pd.Timestamp(x) for x in ['2008-03-27','2008-10-23','2008-11-18']]
def period(d): return 'update(±14d)' if any(0<=(d-u).days<=14 for u in UPD) else 'normal'
# 복귀 이벤트: 14일 이상 미접속 후 접속
R=cd[(cd.gap>=15)&(cd.day>='2008-02-01')].copy()
R=R[~R.day.isin(bad)&~(R.day+pd.Timedelta('1D')).isin(bad)&~(R.day+pd.Timedelta('2D')).isin(bad)]
R['d12']=R.apply(lambda r:outcome(r,1,2),axis=1)
R['long']=R.apply(lambda r:outcome(r,28,41),axis=1); R.loc[R.day>'2008-11-20','long']=np.nan
R['lv70']=R.level>=70; R['period']=R.day.map(period)
# 사전 활동(휴면 전 28일 접속일수) -> CUPED용 공변량
def pre_days(r):
    a=days_by[r.char]; return ((a<r.prev+pd.Timedelta('1D'))&(a>r.prev-pd.Timedelta('28D'))).sum()
R['pre_days']=R.apply(pre_days,axis=1)
# 신규: 1레벨로 처음 등장
N=cd[(cd.day==cd.first_day)&(cd.level<=1)&(cd.day>='2008-01-08')].copy()
N=N[~N.day.isin(bad)&~(N.day+pd.Timedelta('1D')).isin(bad)&~(N.day+pd.Timedelta('2D')).isin(bad)]
N['d12']=N.apply(lambda r:outcome(r,1,2),axis=1); N['long']=N.apply(lambda r:outcome(r,28,41),axis=1); N.loc[N.day>'2008-11-20','long']=np.nan
N['period']=N.day.map(period)
ndays=lambda s: s.day.nunique()
pd.set_option('display.width',200)
print('## 기준값 (2일차 재접속 = 다음 날~이틀 뒤 접속 / 장기 = 4~6주 뒤 접속)')
rows=[]
for name,g in [('복귀 전체',R),('복귀 70레벨',R[R.lv70]),('신규(1레벨 생성)',N)]:
    for p in ['normal','update(±14d)']:
        x=g[g.period==p]; span=(x.day.max()-x.day.min()).days+1
        rows.append({'segment':name,'period':p,'events':len(x),'per_day':len(x)/x.day.nunique(),'d12_rate':x.d12.mean(),
                     'long_if_d12':x[x.d12].long.mean(),'long_if_not':x[~x.d12].long.mean()})
base=pd.DataFrame(rows); print(base.round(3).to_string(index=False)); base.to_csv('update_project/ab_baseline.csv',index=False)
# CUPED 분산 감소 (복귀 70레벨: 사전 접속일수와 결과의 상관)
x=R[R.lv70&(R.period=='normal')]; rho=np.corrcoef(x.pre_days,x.d12.astype(float))[0,1]
print(f'\nCUPED: 복귀 70레벨 사전 접속일수-2일차 재접속 상관 {rho:.3f} -> 필요 표본 {rho**2:.1%} 감소')
def n_arm(p0,lift,a=0.05,pw=0.8):
    z=NormalDist().inv_cdf; za,zb=z(1-a/2),z(pw); p1=p0+lift; pb=(p0+p1)/2
    return ((za*np.sqrt(2*pb*(1-pb))+zb*np.sqrt(p0*(1-p0)+p1*(1-p1)))**2)/lift**2
print('\n## 필요 표본과 기간 (유의수준 5%, 검정력 80%, 50:50 배정, 1차 지표 = 2일차 재접속, 관찰 +2일)')
out=[]
for _,r in base.iterrows():
    for lift in [0.03,0.05,0.10]:
        n=ceil(n_arm(r.d12_rate,lift)); out.append({'segment':r.segment,'period':r.period,'baseline':round(r.d12_rate,3),'MDE(%p)':int(lift*100),
            'n_per_arm':n,'days_needed':ceil(2*n/r.per_day)+2})
out=pd.DataFrame(out); print(out.to_string(index=False)); out.to_csv('update_project/ab_sample_size.csv',index=False)
