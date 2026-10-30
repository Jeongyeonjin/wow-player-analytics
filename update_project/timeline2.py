# 1년 타임라인 v2: 10/7~10/26 대량 신규 캐릭터 제외, 대형 업데이트 3개 + 참고선
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.dates as mdates
d=pd.read_csv('update_project/daily_metrics_nosurge.csv',index_col=0,parse_dates=True)
g=d.copy(); g.loc[g.cov_h<12,['ccu','dau','lv70']]=np.nan
EV={'2.4.0 content':'2008-03-27','3.0.2 system (est.)':'2008-10-23','WotLK expansion':'2008-11-18'}
REF={'2.4.3 (riding lv30)':'2008-07-17','3.0.3':'2008-11-06'}
fig,ax=plt.subplots(3,1,figsize=(13,8.5),sharex=True); C='#2a6fb0'
spec=[('ccu','Avg concurrent characters (per collection round)',1),('dau','Daily active characters',1),('lv70','Share of daily actives at level 70+ (%)',100)]
for a,(c,t,s) in zip(ax,spec):
    a.plot(g.index,g[c]*s,color=C,lw=0.8,alpha=0.35); a.plot(g.index,(g[c]*s).rolling(7,min_periods=4).mean(),color=C,lw=2)
    a.set_title(t,loc='left',fontsize=10); a.grid(axis='y',color='#e5e5e5'); a.spines[['top','right']].set_visible(False)
    for dte in g.index[d.cov_h<12]: a.axvspan(dte,dte+pd.Timedelta('1D'),color='#bbbbbb',alpha=0.35,lw=0)
    a.axvspan(pd.Timestamp('2008-10-07'),pd.Timestamp('2008-10-27'),color='#f2c14e',alpha=0.18,lw=0)
    for m in EV.values(): a.axvline(pd.Timestamp(m),color='#222',lw=1.2)
    for m in REF.values(): a.axvline(pd.Timestamp(m),color='#999',lw=1,ls='--')
top=ax[0].get_ylim()[1]
for i,(k,m) in enumerate(EV.items()):
    right=k.startswith('3.0.2'); ax[0].text(pd.Timestamp(m),top,(k+' ') if right else (' '+k),ha='right' if right else 'left',va='top',fontsize=9,fontweight='bold')
for k,m in REF.items(): ax[0].text(pd.Timestamp(m),top*0.9,' '+k,va='top',fontsize=8,color='#666')
ax[1].text(pd.Timestamp('2008-10-08'),ax[1].get_ylim()[0],' new-char surge\n (excluded)',fontsize=8,color='#9a7300',va='bottom')
ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d')); ax[2].xaxis.set_major_locator(mdates.MonthLocator())
fig.text(0.01,0.005,'Thin = daily, bold = 7-day mean. Gray = collection gaps (<12h). Yellow = 10/7-10/26 mass new-character surge, those characters excluded. Taiwan server dates.',fontsize=8,color='#555')
fig.tight_layout(rect=(0,0.02,1,1)); fig.savefig('update_project/01_timeline.png',dpi=130)
