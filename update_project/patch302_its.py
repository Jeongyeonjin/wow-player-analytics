# 3.0.2 사전 패치 (대만 10/23 추정) 단절 시계열: 사전 추세 + 요일 효과로 예측 vs 실제
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d=pd.read_csv('update_project/daily_metrics_existing.csv',index_col=0,parse_dates=True)
d=d[d.cov_h>=20]
PRE=('2008-09-04','2008-10-22'); POST=('2008-10-24','2008-11-05'); E=pd.Timestamp('2008-10-23')
pre=d.loc[PRE[0]:PRE[1]]; post=d.loc[POST[0]:POST[1]]
def X(idx):
    t=(idx-E).days.values.astype(float); wd=np.eye(7)[idx.dayofweek][:,[1,2,4,5,6]]  # 목요일은 점검으로 항상 제외, 월요일 기준
    return np.c_[np.ones(len(idx)),t,wd]
rows=[]; fits={}
for m in ['ccu','dau','lv70','returners','returners70']:
    Xp,yp=X(pre.index),pre[m].values; b,*_=np.linalg.lstsq(Xp,yp,rcond=None)
    res=yp-Xp@b; s2=res.var(ddof=Xp.shape[1]); cov=s2*np.linalg.inv(Xp.T@Xp)
    Xq=X(post.index); pred=Xq@b; gap=post[m].values-pred
    xb=Xq.mean(0); se=np.sqrt(xb@cov@xb+s2/len(post))
    rows.append({'metric':m,'pre_mean':yp.mean(),'post_actual':post[m].mean(),'post_pred':pred.mean(),
                 'gap':gap.mean(),'gap_lo':gap.mean()-1.96*se,'gap_hi':gap.mean()+1.96*se,'rel':gap.mean()/pred.mean()})
    fits[m]=(b,pred)
    # 1주차/2주차
    for k,(a,z) in enumerate([('2008-10-24','2008-10-29'),('2008-10-31','2008-11-05')]):
        mk=(post.index>=a)&(post.index<=z); rows[-1][f'rel_w{k+1}']=gap[mk].mean()/pred[mk].mean()
r=pd.DataFrame(rows); pd.set_option('display.width',200)
print(f'사전 {len(pre)}일 / 사후 {len(post)}일 (good day만)'); print(r.round(3).to_string(index=False))
r.to_csv('update_project/p302_its.csv',index=False)
fig,ax=plt.subplots(1,3,figsize=(15,4)); C='#2a6fb0'
win=d.loc['2008-09-04':'2008-11-05']
for a,(m,t) in zip(ax,[('ccu','Avg concurrent characters'),('lv70','Share of daily actives at lv70+'),('returners','Returning characters per day (absent 28d+)')]):
    b=fits[m][0]; a.plot(win.index,win[m],'o-',color=C,ms=3,lw=1)
    a.plot(win.index,X(win.index)@b,'--',color='#777',lw=1.3,label='expected (pre-trend + weekday)')
    a.axvline(E,color='#222',lw=1.2); a.axvline(pd.Timestamp('2008-11-06'),color='#aaa',lw=1,ls=':')
    a.set_title(t,loc='left',fontsize=10); a.spines[['top','right']].set_visible(False); a.grid(axis='y',color='#eee')
    a.tick_params(axis='x',rotation=45,labelsize=8)
ax[0].legend(fontsize=8,frameon=False)
fig.suptitle('Patch 3.0.2 (TW est. 10/23): actual vs expected. Existing characters only (first seen before 10/7).',x=0.01,ha='left',fontsize=10)
fig.tight_layout(); fig.savefig('update_project/03_p302_its.png',dpi=130)
