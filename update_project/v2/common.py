# 공통 캐시: 캐릭터-일 단위 플레이 시간(h), 최고 레벨(lv), 일별 수집 시간(cov)
# 정의는 기존 스크립트와 동일: 수집 라운드 간격(최대 30분)을 그 라운드에 보인 캐릭터의 플레이 시간으로 계산
import pandas as pd, numpy as np, os
BASE=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..')
CACHE=os.path.join(os.path.dirname(os.path.abspath(__file__)),'_cd.pkl')
def load():
    if os.path.exists(CACHE): return pd.read_pickle(CACHE)
    df=pd.read_pickle(os.path.join(BASE,'churn_project/step6/clean.pkl'))[['char','level','charclass','guild','timestamp']]
    ts=np.sort(df.timestamp.unique()); rid=np.r_[0,np.cumsum(np.diff(ts)>np.timedelta64(5,'m'))]
    df['round']=pd.Series(rid,index=ts).reindex(df.timestamp).values
    rt=df.groupby('round').timestamp.min().sort_index()
    w=(rt.shift(-1)-rt).dt.total_seconds().clip(upper=1800).fillna(600)/3600
    df['h']=df['round'].map(w); df['day']=df.timestamp.dt.normalize()
    cov=pd.Series(w.values,index=rt.dt.normalize().values).groupby(level=0).sum()
    cd=df.groupby(['char','day']).agg(h=('h','sum'),lv=('level','max'),guild=('guild','max')).reset_index()
    info=df.groupby('char').agg(first=('timestamp','min'),cls=('charclass','first'))
    out={'cd':cd,'cov':cov,'info':info}; pd.to_pickle(out,CACHE); return out
