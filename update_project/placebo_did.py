# 2.4.0 이중차분 방식의 가짜 패치일 검증 (patch24_did.py와 같은 정의, 그림 없음)
import pandas as pd, numpy as np, sys
raw=pd.read_pickle('churn_project/step6/clean.pkl')[['char','level','timestamp']]
ts=np.sort(raw.timestamp.unique()); rid=np.r_[0,np.cumsum(np.diff(ts)>np.timedelta64(5,'m'))]
raw['round']=pd.Series(rid,index=ts).reindex(raw.timestamp).values
rt=raw.groupby('round').timestamp.min().sort_index()
w=(rt.shift(-1)-rt).dt.total_seconds().clip(upper=1800).fillna(600)/3600
raw['h']=raw['round'].map(w); raw['day']=raw.timestamp.dt.normalize()
cov=pd.Series(w.values,index=rt.dt.normalize().values).groupby(level=0).sum()
def run(MAINT, wmin=-8, wmax=4):
    M=pd.Timestamp(MAINT); A=M+pd.Timedelta('1D'); P=M-pd.Timedelta('1D')
    def wk(dd): dd=pd.DatetimeIndex(dd); return np.where(dd>=A,(dd-A).days//7,-1-(P-dd).days//7)
    gd=cov[(cov>=20)&(cov.index!=M)].index; ng=pd.Series(wk(gd),index=gd).value_counts()
    weeks=[k for k in range(wmin,wmax+1) if ng.get(k,0)>=3]
    df=raw[raw.day.isin(gd)].copy(); df['week']=wk(df.day); df=df[df.week.isin(weeks)]
    pre=df[(df.day>=M-pd.Timedelta('21D'))&(df.day<M)].sort_values('timestamp')
    lv=pre.groupby('char')['level'].last()
    grp=pd.Series(np.where(lv>=70,'T',np.where(lv>=60,'C','x')),index=lv.index); grp=grp[grp!='x']
    a=df[df.char.isin(grp.index)].groupby(['char','week']).agg(hours=('h','sum'),days=('day','nunique'))
    idx=pd.MultiIndex.from_product([grp.index,weeks],names=['char','week'])
    pan=a.reindex(idx,fill_value=0).reset_index(); pan['g']=pan.char.map(grp); pan['ng']=pan.week.map(ng)
    pan['hours']=pan.hours/pan.ng*7; pan['days']=pan.days/pan.ng*7
    W=np.array(weeks); j=list(W).index(-1); res={'nT':(grp=='T').sum(),'nC':(grp=='C').sum(),'weeks':weeks}
    for m in ['hours','days']:
        T=pan[pan.g=='T'].pivot(index='char',columns='week',values=m).values; C=pan[pan.g=='C'].pivot(index='char',columns='week',values=m).values
        c=(T-T[:,[j]]).mean(0)-(C-C[:,[j]]).mean(0); b=np.polyfit(W[W<0],c[W<0],1); adj=c-np.polyval(b,W)
        base=T[:,W<0].mean()
        for k in [0,2]:
            if k in weeks: res[f'{m}_w{k}_rel']=adj[list(W).index(k)]/base
    return res
out={}
for name,mt in [('REAL 2.4.0 (3/27)','2008-03-27'),('fake 2/21','2008-02-21'),('fake 3/06','2008-03-06'),('fake 8/21','2008-08-21'),('fake 9/04','2008-09-04')]:
    r=run(mt); out[name]=r; print(name, {k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items()})
pd.DataFrame(out).T.to_csv('update_project/placebo_did.csv')
