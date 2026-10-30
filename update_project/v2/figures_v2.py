# v2 보완 분석 그림 4장
import sys, os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
D=os.path.dirname(os.path.abspath(__file__))+'/'
C1,C2,GR='#2a6fb0','#d9822b','#9a9a9a'
def clean(a): a.spines[['top','right']].set_visible(False); a.grid(axis='y',color='#eee')

# 1) 2.4.0: 기존(S0) vs 보완(S3)
s=pd.read_csv(D+'p24_specs.csv',index_col=0); W=[-8,-7,-6,-5,-4,-3,-2,-1,0,2,3,4]
pl=pd.read_csv(D+'p24_placebo_S3.csv',index_col=0)
fig,ax=plt.subplots(1,2,figsize=(13,4.3))
for a,m,t in zip(ax,['days','hours'],['Active days / week','Play hours / week']):
    for lab,col,ls in [('S0 기존(직전 3주 선택)',GR,'--'),('S3 S1+대조군 40~69',C1,'-')]:
        b=float(s.loc[lab,m+'_base']); y=np.array([float(s.loc[lab,f'{m}_w{k}']) for k in W])/b*100
        a.plot(W,y,'o'+ls,color=col,lw=2,ms=4,label='original spec' if lab.startswith('S0') else 'revised spec (balanced panel, lv40-69 control)')
        if lab.startswith('S3'):
            lo=np.array([float(s.loc[lab,f'{m}_w{k}_lo']) for k in W])/b*100; hi=np.array([float(s.loc[lab,f'{m}_w{k}_hi']) for k in W])/b*100
            a.fill_between(W,lo,hi,color=C1,alpha=0.15,lw=0)
    v=(pl[f'{m}_w0']/pl[f'{m}_base']*100); a.axhspan(v.min(),v.max(),xmin=0,xmax=1,color=C2,alpha=0.10,lw=0)
    a.text(W[0],v.max(),' week-0 range on 7 placebo dates',color=C2,fontsize=8,va='bottom')
    a.axhline(0,color='#bbb',lw=1); a.axvline(-0.5,color='#222',lw=1.2)
    a.set_title(f'Patch 2.4.0: lv70 vs control, {t} (% of lv70 pre-mean)',loc='left',fontsize=10); a.set_xlabel('weeks from patch (week 1 missing: collection gap 4/3-4/7)'); clean(a)
ax[0].legend(fontsize=8,frameon=False,loc='upper right'); fig.tight_layout(); fig.savefig(D+'v2_01_p24.png',dpi=130); plt.close()

# 2) 3.0.2: 모형·집단별 DAU 효과 vs 가짜 날짜
a_=pd.read_csv(D+'p302_robust.csv'); b_=pd.read_csv(D+'p302_noBE.csv')
ML=['선형_49일','로그선형_49일','수준유지_14일','선형_28일']; EN={'선형_49일':'linear, 49d pre','로그선형_49일':'log-linear, 49d pre','수준유지_14일':'flat level, 14d pre','선형_28일':'linear, 28d pre*'}
fig,ax=plt.subplots(1,2,figsize=(13,4.2),sharey=True)
for a,w in zip(ax,['w1','w2']):
    for i,mn in enumerate(ML):
        r=a_[(a_.model==mn)&(a_.metric=='dau')&(a_.week==w)].iloc[0]; q=b_[(b_.model==mn)&(b_.metric=='dau')&(b_.week==w)].iloc[0]
        a.plot([r.placebo_mean-2*r.placebo_sd,r.placebo_mean+2*r.placebo_sd],[i,i],color=C2,lw=6,alpha=0.25,solid_capstyle='butt')
        a.plot(r.real,i,'o',color=C1,ms=8); a.plot(q.real,i,'D',color='#1d4e7a',ms=6,mfc='white')
    a.axvline(0,color='#bbb',lw=1); a.set_yticks(range(len(ML))); a.set_yticklabels([EN[m] for m in ML])
    a.set_title(f'Patch 3.0.2 (TW 10/23): DAU effect, week {w[1]}',loc='left',fontsize=10); a.set_xlabel('actual vs expected (relative)'); clean(a)
    a.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax[0].plot([],[],'o',color=C1,label='all existing chars'); ax[0].plot([],[],'D',color='#1d4e7a',mfc='white',label='excluding Blood Elf (collection loss in week 2)')
ax[0].plot([],[],color=C2,lw=6,alpha=0.25,label='placebo mean ± 2sd (7 dates)'); fig.legend(*ax[0].get_legend_handles_labels(),fontsize=8,frameon=False,loc='lower right',ncol=3)
fig.text(0.01,0.01,'* 28-day linear trend extrapolates the early-October drop as a trend; shown for comparison only.',fontsize=8,color='#555')
fig.tight_layout(rect=(0,0.07,1,1)); fig.savefig(D+'v2_02_p302.png',dpi=130); plt.close()

# 3) 리치왕: 주별 플레이 시간과 공백기 이전 추세
w=pd.read_csv(D+'weekly_series.csv',index_col=0,parse_dates=True).loc['2008-05-22':]
f=w.loc['2008-07-03':'2008-09-25']; t=(f.index-pd.Timestamp('2008-09-25')).days/7; b=np.polyfit(t,f.hours,1)
tt=(w.index-pd.Timestamp('2008-09-25')).days/7
fig,ax=plt.subplots(1,2,figsize=(13,4.3))
a=ax[0]; a.bar(w.index,w.hours_noDK/1000,width=5,color=C1,label='non-DK characters'); a.bar(w.index,(w.hours-w.hours_noDK)/1000,bottom=w.hours_noDK/1000,width=5,color=C2,label='Death Knights (new class)')
m=w.index>='2008-07-03'; a.plot(w.index[m],np.polyval(b,tt[m])/1000,'--',color='#222',lw=1.3,label='Jul-Sep trend (counterfactual)')
a.axvspan(pd.Timestamp('2008-10-01'),pd.Timestamp('2008-11-18'),color='#bbb',alpha=0.2,lw=0); a.text(pd.Timestamp('2008-10-03'),36,'pre-launch lull\n-33% vs trend',fontsize=8)
a.axvline(pd.Timestamp('2008-11-18'),color='#222',lw=1.2); a.set_title('Weekly play hours (k), surge characters excluded',loc='left',fontsize=10); a.set_ylim(0,62); a.legend(fontsize=8,frameon=False,loc='upper left',ncol=3); clean(a)
cc=pd.read_csv(D+'collection_check.csv',index_col=0,parse_dates=True).loc['2008-05-22':]
a=ax[1]; a.plot(cc.index,cc.hole_rate*100,'o-',color=C1,ms=3,lw=1.5,label='missed-in-session rate (%)'); a2=a.twinx(); a2.plot(cc.index,cc.gap_med,'s-',color=C2,ms=3,lw=1.2,label='median round interval (min)')
a.set_title('Collection quality: changes after 10/23 and 11/6',loc='left',fontsize=10); a.axvline(pd.Timestamp('2008-11-18'),color='#222',lw=1.2)
a.set_ylabel('missed-in-session %',color=C1); a2.set_ylabel('round interval (min)',color=C2); clean(a); a2.spines[['top']].set_visible(False)
fig.tight_layout(); fig.savefig(D+'v2_03_wotlk.png',dpi=130); plt.close()

# 4) 복귀: 통제 변수 추가에 따른 오즈비 변화
A=pd.read_csv(D+'returner_logit_A.csv',index_col=[0,1]); B=pd.read_csv(D+'returner_logit_B.csv',index_col=[0,1])
rows=[('guild in week 1',k,A.loc[(k,'guild_w1')]) for k in ['A1 길드만','A2 +레벨·휴면·휴면전길드','A3 +첫주 접속일 2일+','A4 +첫주 플레이시간']]
rows+=[('2+ active days in week 1',k,A.loc[(k,'d2plus')]) for k in ['A3 +첫주 접속일 2일+','A4 +첫주 플레이시간']]
lab2={('2+ active days in week 1','A3 +첫주 접속일 2일+'):'+ level, dormancy, guild'}
rows+=[('day-2 return (all returns, A/B metric)',k,B.loc[(k,'d12')]) for k in ['B1 2일차만','B2 +레벨·휴면기간·휴면전활동·월']]
lab={'A1 길드만':'raw','A2 +레벨·휴면·휴면전길드':'+ level, dormancy, prior guild','A3 +첫주 접속일 2일+':'+ 2+ days in week 1','A4 +첫주 플레이시간':'+ week-1 hours',
     'B1 2일차만':'raw','B2 +레벨·휴면기간·휴면전활동·월':'+ level, dormancy, prior activity, month'}
fig,a=plt.subplots(figsize=(10,4.6)); ys=list(range(len(rows)))[::-1]; colr={'guild in week 1':C2,'2+ active days in week 1':C1,'day-2 return (all returns, A/B metric)':'#1d4e7a'}
for y,(v,k,r) in zip(ys,rows):
    a.plot([r.OR_lo,r.OR_hi],[y,y],color=colr[v],lw=2); a.plot(r.OR,y,'o',color=colr[v],ms=7)
a.axvline(1,color='#999',lw=1); a.set_xscale('log'); a.set_yticks(ys); a.set_yticklabels([f'{v}: {lab2.get((v,k),lab[k])}' for v,k,_ in rows],fontsize=8)
a.set_xticks([0.5,1,2,4,8]); a.set_xticklabels(['0.5','1','2','4','8']); a.minorticks_off()
a.set_xlabel('odds ratio for staying 4-6 weeks later (95% CI, log scale)'); a.set_title('Returners: what survives after controlling for level and dormancy',loc='left',fontsize=10); clean(a)
fig.tight_layout(); fig.savefig(D+'v2_04_returner.png',dpi=130); plt.close()
print('ok')
