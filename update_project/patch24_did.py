# 패치 2.4.0 (대만 3/27 점검, 3/28 오픈) 효과: 70레벨(처치) vs 60~69레벨(대조) 이중차분
# - 수집 공백 대응: 하루 20시간 이상 수집된 날(good day)만 쓰고, 주별 값은 '7일 환산'
# - 사전 추세가 평행하지 않으면 사전 추세를 연장한 값과 비교(추세 보정)
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT='update_project/'
df=pd.read_pickle('churn_project/step6/clean.pkl')[['char','level','zone','timestamp']]
df=df[(df.timestamp>='2008-01-20')&(df.timestamp<'2008-06-13')]
ts=np.sort(df.timestamp.unique()); rid=np.r_[0,np.cumsum(np.diff(ts)>np.timedelta64(5,'m'))]
df['round']=pd.Series(rid,index=ts).reindex(df.timestamp).values
rt=df.groupby('round').timestamp.min().sort_index()
w=(rt.shift(-1)-rt).dt.total_seconds().clip(upper=1800).fillna(600)/3600
df['h']=df['round'].map(w); df['day']=df.timestamp.dt.normalize()
cov=pd.Series(w.values,index=rt.dt.normalize().values).groupby(level=0).sum()
A=pd.Timestamp('2008-03-28'); P=pd.Timestamp('2008-03-26')
def wk(dd): dd=pd.DatetimeIndex(dd); return np.where(dd>=A,(dd-A).days//7,-1-(P-dd).days//7)
gd=cov[(cov>=20)&(cov.index!=pd.Timestamp('2008-03-27'))].index
gdw=pd.Series(wk(gd),index=gd); ngood=gdw.value_counts().sort_index()
weeks=[k for k in range(-8,11) if ngood.get(k,0)>=3]
print('주별 good day 수:',ngood.loc[-8:10].to_dict())
df=df[df.day.isin(gd)]; df['week']=wk(df.day); df=df[df.week.isin(weeks)]
pre=df[(df.day>='2008-03-06')&(df.day<'2008-03-27')].sort_values('timestamp')
lv=pre.groupby('char')['level'].last()
grp=pd.Series(np.where(lv>=70,'T70',np.where(lv>=60,'C60_69','x')),index=lv.index); grp=grp[grp!='x']
print(grp.value_counts().to_dict())
x=df[df.char.isin(grp.index)].copy()
NEW={"Isle of Quel'Danas","Magisters' Terrace","Sunwell Plateau"}; x['newh']=np.where(x.zone.astype(str).isin(NEW),x.h,0)
a=x.groupby(['char','week']).agg(hours=('h','sum'),newh=('newh','sum'),days=('day','nunique'))
idx=pd.MultiIndex.from_product([grp.index,weeks],names=['char','week'])
pan=a.reindex(idx,fill_value=0).reset_index(); pan['g']=pan.char.map(grp)
pan['ng']=pan.week.map(ngood); pan['active']=(pan.hours>0).astype(float)
pan['hours']=pan.hours/pan.ng*7; pan['days']=pan.days/pan.ng*7; pan['newh']=pan.newh/pan.ng*7
post70=df[(df.week>=0)&(df.level>=70)].char.unique()
print(f"대조군 중 처치기간에 70 도달: {grp[grp=='C60_69'].index.isin(post70).mean():.1%}")
M=['hours','days','active']
means=pan.groupby(['week','g'])[M+['newh']].mean().unstack(); means.round(3).to_csv(OUT+'p24_weekly_means.csv')
mats={g:{m:pan[pan.g==g].pivot(index='char',columns='week',values=m).values for m in M} for g in ['T70','C60_69']}
W=np.array(weeks); j=list(W).index(-1); prew=W<0
def curve(T,C): return (T-T[:,[j]]).mean(0)-(C-C[:,[j]]).mean(0)
def adj(c):  # 사전(-8~-1) 추세선 연장 대비 편차
    b=np.polyfit(W[prew],c[prew],1); return c-np.polyval(b,W), b[0]
rng=np.random.default_rng(0); rows=[]; plotd={}
for m in M:
    T,C=mats['T70'][m],mats['C60_69'][m]
    raw=curve(T,C); ad,slope=adj(raw); B_raw=[];B_ad=[]
    for _ in range(500):
        tt=T[rng.integers(0,len(T),len(T))]; cc=C[rng.integers(0,len(C),len(C))]
        r=curve(tt,cc); B_raw.append(r); B_ad.append(adj(r)[0])
    B_raw,B_ad=np.array(B_raw),np.array(B_ad)
    plotd[m]=(raw,np.percentile(B_raw,2.5,0),np.percentile(B_raw,97.5,0),raw-ad)
    for k,wv in enumerate(W):
        rows.append({'metric':m,'week':wv,'did':raw[k],'did_lo':np.percentile(B_raw[:,k],2.5),'did_hi':np.percentile(B_raw[:,k],97.5),
                     'trend_adj':ad[k],'adj_lo':np.percentile(B_ad[:,k],2.5),'adj_hi':np.percentile(B_ad[:,k],97.5)})
    print(f"[{m}] 사전 추세 기울기(처치-대조, 주당): {slope:+.3f}")
es=pd.DataFrame(rows); es.round(3).to_csv(OUT+'p24_event_study.csv',index=False)
pd.set_option('display.width',220)
print('\n주별 평균'); print(means.round(2).to_string())
print('\n이벤트 스터디: 원 DiD / 추세 보정 DiD (95% CI)')
print(es.pivot(index='week',columns='metric',values=['did','trend_adj']).round(2).to_string())
base=pan[(pan.g=='T70')&(pan.week<0)][M].mean()
for lab,ws in [('0~1주',[0,1]),('2~4주',[2,3,4]),('5~10주',list(range(5,11)))]:
    s=es[es.week.isin(ws)].groupby('metric')[['did','trend_adj','adj_lo','adj_hi']].mean()
    for m in M: print(f"{lab} {m:6s}: 원 DiD {s.loc[m,'did']:+.2f} / 추세보정 {s.loc[m,'trend_adj']:+.2f} (CI {s.loc[m,'adj_lo']:+.2f}~{s.loc[m,'adj_hi']:+.2f}, 처치군 사전평균 {base[m]:.2f}, 상대 {s.loc[m,'trend_adj']/base[m]:+.0%})")
nt=pan[(pan.g=='T70')&(pan.week>=0)].groupby('week')[['newh','hours']].sum(); nt['share']=nt.newh/nt.hours
print('\n70레벨 플레이 시간 중 신규 지역 비중'); print(nt.share.round(3).to_string()); nt.to_csv(OUT+'p24_new_content_share.csv')
fig,ax=plt.subplots(1,3,figsize=(15,4.3)); C1='#2a6fb0'
lab={'hours':'Weekly play hours','days':'Active days / week','active':'Share active in week'}
for a_,m in zip(ax,M):
    raw,lo,hi,tr=plotd[m]
    a_.axhline(0,color='#bbb',lw=1); a_.axvline(-0.5,color='#222',lw=1.2)
    a_.fill_between(W,lo,hi,color=C1,alpha=0.18,lw=0); a_.plot(W,raw,'o-',color=C1,lw=2,ms=5)
    a_.plot(W,tr,'--',color='#777',lw=1.3); a_.text(W[-1],tr[-1],' pre-trend',fontsize=8,color='#555',va='center')
    a_.set_title('lv70 minus lv60-69: '+lab[m],loc='left',fontsize=10); a_.set_xlabel('weeks from patch 2.4.0 (TW 3/28), base = week -1')
    a_.spines[['top','right']].set_visible(False); a_.grid(axis='y',color='#eee')
fig.tight_layout(); fig.savefig(OUT+'02_p24_event_study.png',dpi=130)
