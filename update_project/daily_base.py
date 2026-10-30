# 공통 일별 지표 (10/8 급증 캐릭터 제외) + 일별/주별 복귀 유저
import pandas as pd, numpy as np
df=pd.read_pickle('churn_project/step6/clean.pkl')[['char','level','charclass','timestamp']]
first=df.groupby('char').timestamp.min()
import sys
MODE=sys.argv[1] if len(sys.argv)>1 else 'nosurge'
# 10/7~10/26: 신규 캐릭터 대량 유입 기간 (평소 하루 40개 -> 최대 1,521개)
surge=first[(first>='2008-10-07')&(first<'2008-10-27')].index
if MODE=='existing': surge=first[first>='2008-10-07'].index   # 10/7 이전에 있던 캐릭터만

ts=np.sort(df.timestamp.unique()); rid=np.r_[0,np.cumsum(np.diff(ts)>np.timedelta64(5,'m'))]
df['round']=pd.Series(rid,index=ts).reindex(df.timestamp).values
rt=df.groupby('round').timestamp.min().sort_index()
w=(rt.shift(-1)-rt).dt.total_seconds().clip(upper=1800).fillna(600)/3600
cov=pd.Series(w.values,index=rt.dt.normalize().values).groupby(level=0).sum()
df=df[~df.char.isin(surge)]
df['day']=df.timestamp.dt.normalize()
rc=df.groupby('round').size().reindex(rt.index,fill_value=0)
ccu=pd.Series(rc.values,index=rt.dt.normalize().values).groupby(level=0).mean()
cd=df.groupby(['day','char'])['level'].max().reset_index()
day=pd.DataFrame({'cov_h':cov,'ccu':ccu})
day['dau']=cd.groupby('day').size(); day['lv70']=cd.groupby('day')['level'].apply(lambda s:(s>=70).mean())
# 일별 복귀: 그날 접속, 직전 28일 미접속, 첫 등장이 28일보다 이전
cd['first']=cd.char.map(first)
cd=cd.sort_values(['char','day']); cd['prev']=cd.groupby('char').day.shift(1)
gap=(cd.day-cd.prev).dt.days
cd['ret']=(gap>28)
cd['ret70']=cd.ret&(cd.level>=70)
day['returners']=cd.groupby('day').ret.sum(); day['returners70']=cd.groupby('day').ret70.sum()
day.to_csv(f'update_project/daily_metrics_{MODE}.csv',encoding='utf-8-sig')
print(len(surge),'surge chars excluded'); 
