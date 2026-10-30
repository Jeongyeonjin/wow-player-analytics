# 3.0.2 추세 연장 방식의 가짜 패치일 검증: 같은 방법을 패치 없던 목요일들에 적용
import pandas as pd, numpy as np
d=pd.read_csv('update_project/daily_metrics_nosurge.csv',index_col=0,parse_dates=True); d=d[d.cov_h>=20]
M=['ccu','dau','lv70','returners','returners70']
def its(E):
    E=pd.Timestamp(E); pre=d.loc[E-pd.Timedelta('49D'):E-pd.Timedelta('1D')]; post=d.loc[E+pd.Timedelta('1D'):E+pd.Timedelta('6D')]
    def X(idx): t=(idx-E).days.values.astype(float); return np.c_[np.ones(len(idx)),t,np.eye(7)[idx.dayofweek][:,[1,2,4,5,6]]]
    out={}
    for m in M:
        b,*_=np.linalg.lstsq(X(pre.index),pre[m].values,rcond=None); p=X(post.index)@b
        out[m]=(post[m].values-p).mean()/p.mean()
    out['n_pre'],out['n_post']=len(pre),len(post); return out
real=its('2008-10-23')
# 패치 없던 목요일: 2.4.0(3/27), 2.4.1(4/3), 2.4.2(5/15), 2.4.3(7/17) 및 그 주변 2주, 10/7 이후 제외
patch=[pd.Timestamp(x) for x in ['2008-03-27','2008-04-03','2008-05-15','2008-07-17']]
cands=[t for t in pd.date_range('2008-02-21','2008-09-25',freq='W-THU') if all(abs((t-p).days)>14 for p in patch)]
pl=pd.DataFrame({t.date():its(t) for t in cands}).T
pl=pl[(pl.n_pre>=30)&(pl.n_post>=4)]
pd.set_option('display.width',200)
print(f'가짜 패치일 {len(pl)}개 (1주차 예측 대비 상대 차이)'); print(pl[M].round(3).to_string())
print('\n실제 3.0.2 (10/23):',{k:round(v,3) for k,v in real.items()})
print('\n가짜 날짜 중 실제보다 크거나 같은 비율 (경험적 p값)')
for m in M: print(f'  {m:12s}: 실제 {real[m]:+.1%} / 가짜 평균 {pl[m].mean():+.1%}, 최대 {pl[m].max():+.1%} -> p={((pl[m]>=real[m]).sum()+1)/(len(pl)+1):.3f}')
pl.to_csv('update_project/placebo_its.csv')
