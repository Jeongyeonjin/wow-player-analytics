import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.dates as mdates
df=pd.read_pickle('churn_project/step6/clean.pkl')[['char','level','timestamp']]
# 수집 라운드: 타임스탬프 간격 5분 초과면 새 라운드
ts=np.sort(df.timestamp.unique()); rid=np.r_[0,np.cumsum(np.diff(ts)>np.timedelta64(5,'m'))]
df['round']=pd.Series(rid,index=ts).reindex(df.timestamp).values
rnd=df.groupby('round').agg(t=('timestamp','min'),n=('char','size'))
rnd=rnd.sort_values('t'); rnd['cov']=(rnd.t.shift(-1)-rnd.t).dt.total_seconds().clip(upper=1800).fillna(0)/3600
day=rnd.groupby(rnd.t.dt.normalize()).agg(rounds=('n','size'),ccu=('n','mean'),cov_h=('cov','sum'))
d2=df.groupby(df.timestamp.dt.normalize()).agg(dau=('char','nunique'))
lv=df.groupby([df.timestamp.dt.normalize(),'char'])['level'].max().reset_index()
d2['lv70']=lv.groupby('timestamp')['level'].apply(lambda x:(x>=70).mean())
day=day.join(d2); day['ok']=day.cov_h>=20; day['outage']=day.cov_h<12   # 비교엔 20h 이상 수집된 날만, 그림 음영은 12h 미만
day.to_csv('update_project/daily_metrics.csv',encoding='utf-8-sig')
EV={'2.4.0':('2008-03-27','2008-03-28'),'3.0.3':('2008-11-06','2008-11-07'),'WotLK':('2008-11-18','2008-11-20')}
REF={'3.0.2 (ref)':'2008-10-23','ID artifact':'2008-10-08'}
rows=[]
for k,(m,a) in EV.items():
    m,a=pd.Timestamp(m),pd.Timestamp(a)
    pre=day.loc[m-pd.Timedelta('7D'):m-pd.Timedelta('1D')]; post=day.loc[a:a+pd.Timedelta('6D')]
    pre,post=pre[pre.ok],post[post.ok]
    rows.append({'event':k,'pre_days':len(pre),'post_days':len(post),
      'ccu_pre':pre.ccu.mean(),'ccu_post':post.ccu.mean(),'ccu_chg%':(post.ccu.mean()/pre.ccu.mean()-1)*100,
      'dau_pre':pre.dau.mean(),'dau_post':post.dau.mean(),'dau_chg%':(post.dau.mean()/pre.dau.mean()-1)*100,
      'lv70_pre':pre.lv70.mean()*100,'lv70_post':post.lv70.mean()*100})
r=pd.DataFrame(rows).round(1); print(r.to_string(index=False)); r.to_csv('update_project/event_1week.csv',index=False)
# 그림: 3단 small multiples, 공백일은 회색 음영
g=day.copy(); 
for c in ['ccu','dau','lv70']: g.loc[g.outage,c]=np.nan
fig,ax=plt.subplots(3,1,figsize=(13,8.5),sharex=True)
C='#2a6fb0'
spec=[('ccu','Avg concurrent characters (per collection round)',1),('dau','Daily active characters',1),('lv70','Share of daily actives at level 70+ (%)',100)]
for a,(c,t,s) in zip(ax,spec):
    a.plot(g.index,g[c]*s,color=C,lw=0.8,alpha=0.35)
    a.plot(g.index,(g[c]*s).rolling(7,min_periods=4).mean(),color=C,lw=2)
    a.set_title(t,loc='left',fontsize=10); a.grid(axis='y',color='#e5e5e5'); a.spines[['top','right']].set_visible(False)
    for dte in day.index[day.outage]: a.axvspan(dte,dte+pd.Timedelta('1D'),color='#bbbbbb',alpha=0.35,lw=0)
    for k,(m,_) in EV.items(): a.axvline(pd.Timestamp(m),color='#222',lw=1.2)
    for k,m in REF.items(): a.axvline(pd.Timestamp(m),color='#888',lw=1,ls='--')
for k,(m,_) in EV.items(): ax[0].text(pd.Timestamp(m),ax[0].get_ylim()[1],(k+' ') if k=='3.0.3' else (' '+k),ha='right' if k=='3.0.3' else 'left',va='top',fontsize=9,fontweight='bold')
for i,(k,m) in enumerate(REF.items()): ax[0].text(pd.Timestamp(m),ax[0].get_ylim()[1]*(0.92-0.07*i),(k+' '),ha='right',va='top',fontsize=8,color='#666')
ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d')); ax[2].xaxis.set_major_locator(mdates.MonthLocator())
fig.text(0.01,0.005,'Thin = daily, bold = 7-day mean. Gray bands = collection gaps (<12h collected), excluded. Dates = Taiwan server.',fontsize=8,color='#555')
fig.tight_layout(rect=(0,0.02,1,1)); fig.savefig('update_project/01_timeline.png',dpi=130)
print('outage days:',day.outage.sum(),' partial(12-20h):',(~day.ok&~day.outage).sum())
