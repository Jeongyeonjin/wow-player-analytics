# 10월 급락이 실제 플레이 감소인지, 수집 누락인지 확인
# '구멍': 같은 캐릭터가 r번째, r+2번째 라운드에는 보이는데 r+1번째에만 없는 경우 (접속 중 수집 누락)
import pandas as pd, numpy as np, os
df=pd.read_pickle('churn_project/step6/clean.pkl')[['char','timestamp','zone']]
ts=np.sort(df.timestamp.unique()); rid=np.r_[0,np.cumsum(np.diff(ts)>np.timedelta64(5,'m'))]
df['round']=pd.Series(rid,index=ts).reindex(df.timestamp).values
rt=df.groupby('round').timestamp.min()
df=df.sort_values(['char','round']); d=df.groupby('char')['round'].diff()
df['wk']=df.timestamp.dt.normalize(); df['wk']=df.wk-pd.to_timedelta((df.wk.dt.dayofweek-3)%7,unit='D')
g=df.assign(cons=(d==1),hole=(d==2)).groupby('wk')
W=pd.DataFrame({'pairs':g.cons.sum(),'holes':g.hole.sum()}); W['hole_rate']=W.holes/(W.pairs+W.holes)
# 라운드당 관측 캐릭터 수, 라운드 간격
rc=df.groupby('round').size(); rinfo=pd.DataFrame({'n':rc,'t':rt}); rinfo['wk']=rinfo.t.dt.normalize(); rinfo['wk']=rinfo.wk-pd.to_timedelta((rinfo.wk.dt.dayofweek-3)%7,unit='D')
rinfo['gap_min']=rinfo.t.diff().shift(-1).dt.total_seconds()/60
R=rinfo.groupby('wk').agg(rounds=('n','size'),ccu_mean=('n','mean'),ccu_p95=('n',lambda s:s.quantile(.95)),ccu_max=('n','max'),gap_med=('gap_min','median'))
out=W.join(R); pd.set_option('display.width',200)
print(out.loc['2008-07-01':].round(3).to_string()); out.to_csv('update_project/v2/collection_check.csv')
