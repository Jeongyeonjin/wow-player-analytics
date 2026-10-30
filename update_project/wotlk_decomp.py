# 확장팩(대만 11/18) 접속자 증가 분해: 기준 주(11/10~16) vs 출시 주(11/20~26)
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
df=pd.read_pickle('churn_project/step6/clean.pkl')[['char','level','charclass','timestamp']]
ts=np.sort(df.timestamp.unique()); rid=np.r_[0,np.cumsum(np.diff(ts)>np.timedelta64(5,'m'))]
df['round']=pd.Series(rid,index=ts).reindex(df.timestamp).values
rt=df.groupby('round').timestamp.min().sort_index()
w=(rt.shift(-1)-rt).dt.total_seconds().clip(upper=1800).fillna(600)/3600
df['h']=df['round'].map(w); df['day']=df.timestamp.dt.normalize()
cov=pd.Series(w.values,index=rt.dt.normalize().values).groupby(level=0).sum()
good=set(cov[cov>=20].index)
cd=df.groupby(['char','day']).agg(h=('h','sum'),lv=('level','max')).reset_index()
info=df.groupby('char').agg(first=('timestamp','min'),cls=('charclass','first'))
info['dk']=info.cls.astype(str).str.contains('Death')
D=pd.Timedelta
def classify(S):
    S=pd.Timestamp(S); wk=cd[(cd.day>=S)&(cd.day<S+D('7D'))]
    chars=wk.char.unique(); hist=cd[cd.day<S-D('2D')]
    last=hist.groupby('char').day.max()
    f=info.loc[chars,'first']; cat=pd.Series('?',index=chars)
    l=last.reindex(chars)
    cat[l>=S-D('17D')]='1_existing'
    cat[(l<S-D('17D'))&(l>=S-D('80D'))]='2_return_short(2-11wk)'
    cat[l<S-D('80D')]='3_return_long(11wk+)'
    new=f>=S-D('2D'); cat[new&info.loc[chars,'dk'].values]='4_new_DK'; cat[new&~info.loc[chars,'dk'].values]='5_new_other'
    gd=[d for d in pd.date_range(S,periods=7) if d in good]
    hh=wk[wk.day.isin(gd)].groupby('char').h.sum().reindex(chars,fill_value=0)/len(gd)*7
    return pd.DataFrame({'cat':cat,'hours':hh})
B=classify('2008-11-10'); L=classify('2008-11-20')
tab=pd.DataFrame({'base_n':B.cat.value_counts(),'launch_n':L.cat.value_counts(),
                  'base_hours':B.groupby('cat').hours.sum(),'launch_hours':L.groupby('cat').hours.sum()}).fillna(0).sort_index()
tab.loc['TOTAL']=tab.sum()
tab['d_n']=tab.launch_n-tab.base_n; tab['d_hours']=tab.launch_hours-tab.base_hours
tab['share_of_d_n']=tab.d_n/tab.loc['TOTAL','d_n']; tab['share_of_d_hours']=tab.d_hours/tab.loc['TOTAL','d_hours']
tab['h_per_char_base']=tab.base_hours/tab.base_n; tab['h_per_char_launch']=tab.launch_hours/tab.launch_n
pd.set_option('display.width',250); print(tab.round(2).to_string()); tab.round(3).to_csv('update_project/wotlk_decomp.csv')
# 기존 유저: 같은 사람 기준 1인당 시간 변화 (기준 주와 출시 주 모두 접속)
both=B.index.intersection(L.index)
print(f"\n두 주 모두 접속한 {len(both)}명: 주간 플레이 {B.loc[both,'hours'].mean():.1f}h -> {L.loc[both,'hours'].mean():.1f}h")
# 그룹별 잔존: 출시 주 접속자가 이후 각 주에 접속했는지
weeks=[pd.Timestamp('2008-11-27')+D(f'{7*k}D') for k in range(5)]
act=cd[cd.day>='2008-11-27'][['char','day']]
ret={}
for k,s in enumerate(weeks):
    a=set(act[(act.day>=s)&(act.day<s+D('7D'))].char)
    ret[f'w+{k+1}']=L.groupby('cat').apply(lambda g: np.mean([c in a for c in g.index]))
ret=pd.DataFrame(ret); print('\n출시 주 접속자 그룹별 이후 주차 접속 비율'); print(ret.round(3).to_string()); ret.to_csv('update_project/wotlk_retention.csv')
# 비교: 확장팩 전 같은 방식 (10/27~11/2 접속자 -> 4주 뒤)
# 레벨업: 출시 주에 70이었던 캐릭터의 12/31까지 최고 레벨
L70=[c for c in L.index if cd[(cd.char==c)&(cd.day<'2008-11-19')].lv.max()==70] if False else None
lvmax=cd[cd.day>='2008-11-19'].groupby('char').lv.max()
pre_lv=cd[(cd.day<'2008-11-18')].groupby('char').lv.last()
c70=[c for c in L.index if pre_lv.get(c,0)==70]
m=lvmax.reindex(c70)
print(f"\n출시 직전 70레벨이었던 출시 주 접속자 {len(c70)}명: 12/31까지 75+ {(m>=75).mean():.0%}, 80 도달 {(m>=80).mean():.0%}")
ttl=cd[(cd.day>='2008-11-19')&(cd.lv>=80)].groupby('char').day.min()
dd=(ttl.reindex(c70).dropna()-pd.Timestamp('2008-11-19')).dt.days
print(f"80 도달자 소요일 중앙값 {dd.median():.0f}일 (p25 {dd.quantile(.25):.0f}, p75 {dd.quantile(.75):.0f})")
# 그림: 그룹별 증가 기여 (인원 / 플레이 시간)
t=tab.drop('TOTAL'); lab=[i.split('_',1)[1] for i in t.index]
fig,ax=plt.subplots(1,3,figsize=(15,4)); C='#2a6fb0'
ax[0].barh(lab,t.base_n,color='#c9c9c9',label='base week 11/10-16'); ax[0].barh(lab,t.launch_n,color=C,height=0.45,label='launch week 11/20-26')
ax[0].set_title('Active characters by group',loc='left',fontsize=10); ax[0].legend(fontsize=8,frameon=False); ax[0].invert_yaxis()
ax[1].barh(lab,t.d_hours,color=C); ax[1].axvline(0,color='#999',lw=1); ax[1].invert_yaxis()
ax[1].set_title('Change in weekly play hours (launch - base)',loc='left',fontsize=10)
for i,(nm,r) in enumerate(ret.iterrows()): ax[2].plot(range(1,6),r.values*100,'o-',lw=2,ms=4,label=nm.split('_',1)[1])
ax[2].set_title('Launch-week actives still active in week +k (%)',loc='left',fontsize=10); ax[2].legend(fontsize=7,frameon=False); ax[2].set_xlabel('weeks after launch week')
for a in ax: a.spines[['top','right']].set_visible(False)
fig.tight_layout(); fig.savefig('update_project/04_wotlk_decomp.png',dpi=130)
