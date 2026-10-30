# 확장팩 복귀 유저: 복귀 후 첫 7일 행동별로 4~6주 뒤에도 남았는지
import pandas as pd, numpy as np
z=pd.read_csv('zones.csv'); NR=set(z.loc[z.iloc[:,1]=='Northrend',z.columns[0]].astype(str))
df=pd.read_pickle('churn_project/step6/clean.pkl')[['char','level','charclass','zone','guild','timestamp']]
first=df.groupby('char').timestamp.min()
surge=first[(first>='2008-10-07')&(first<'2008-10-27')].index
ts=np.sort(df.timestamp.unique()); rid=np.r_[0,np.cumsum(np.diff(ts)>np.timedelta64(5,'m'))]
df['round']=pd.Series(rid,index=ts).reindex(df.timestamp).values
rt=df.groupby('round').timestamp.min().sort_index()
w=(rt.shift(-1)-rt).dt.total_seconds().clip(upper=1800).fillna(600)/3600
df['h']=df['round'].map(w); df['day']=df.timestamp.dt.normalize()
L0,L1=pd.Timestamp('2008-11-19'),pd.Timestamp('2008-11-27')
before=df[df.timestamp<'2008-11-05'].groupby('char').agg(last_seen=('timestamp','max'))
inL=df[(df.timestamp>=L0)&(df.timestamp<L1)]
ret_day=inL.groupby('char').day.min()
gone=df[(df.timestamp>='2008-11-05')&(df.timestamp<L0)].char.unique()
cand=ret_day.index.difference(gone).difference(surge)
cand=cand[first.reindex(cand)<'2008-11-05']                  # 기존 캐릭터만 (DK 등 신규 제외)
R=pd.DataFrame({'ret_day':ret_day.reindex(cand)}); R['dormant_days']=(R.ret_day-before.last_seen.reindex(cand).dt.normalize()).dt.days
x=df[df.char.isin(cand)].copy(); x=x.merge(R[['ret_day']],left_on='char',right_index=True)
wk1=x[(x.day>=x.ret_day)&(x.day<x.ret_day+pd.Timedelta('7D'))]
g=wk1.groupby('char')
R['hours_w1']=g.h.sum(); R['days_w1']=g.day.nunique()
R['lv_start']=g['level'].min(); R['lv_gain_w1']=g['level'].max()-g['level'].min()
R['northrend_share']=wk1.assign(nr=wk1.zone.astype(str).isin(NR)*wk1.h).groupby('char').nr.sum()/R.hours_w1
R['guild_w1']=g.guild.max()>=0
# 길드 동료: 같은 길드에서 그 주에 활동한 다른 캐릭터 수
gw=wk1[wk1.guild>=0].groupby('char').guild.agg(lambda s:s.mode().iloc[0])
act=df[(df.timestamp>=L0)&(df.timestamp<L1+pd.Timedelta('6D'))&(df.guild>=0)].groupby('guild').char.nunique()
R['guildmates']=gw.map(act).reindex(R.index).fillna(1)-1; R.loc[~R.guild_w1,'guildmates']=0
# 휴면 전 길드 (마지막 접속 시점)
lastg=df[df.timestamp<'2008-11-05'].sort_values('timestamp').groupby('char').guild.last()
R['guild_before']=lastg.reindex(R.index)>=0
# 결과: 12/18~12/31 접속 여부 (복귀 후 4~6주)
late=set(df[(df.timestamp>='2008-12-18')].char)
R['settled']=R.index.isin(late)
R['cls']=df[df.char.isin(cand)].groupby('char').charclass.first().astype(str)
R.to_csv('update_project/returners_wotlk.csv')
print(f'복귀 유저 {len(R)}명, 4~6주 뒤 접속 {R.settled.mean():.1%}')
pd.set_option('display.width',200)
def rep(col,bins=None,labels=None):
    s=R[col] if bins is None else pd.cut(R[col],bins,labels=labels,include_lowest=True)
    t=R.groupby(s,observed=True).settled.agg(n='size',settled='mean'); print(f'\n[{col}]'); print(t.round(3).to_string())
rep('dormant_days',[14,30,60,120,400],['2-4주','1-2달','2-4달','4달+'])
rep('hours_w1',[0,1,3,8,20,200],['<1h','1-3h','3-8h','8-20h','20h+'])
rep('days_w1',[0,1,2,4,7],['1일','2일','3-4일','5-7일'])
rep('lv_start',[0,59,69,70,80],['<60','60-69','70','70+'])
rep('northrend_share',[-0.01,0,0.25,0.75,1],['0','~25%','25-75%','75%+'])
rep('guild_w1'); rep('guild_before')
rep('guildmates',[-1,0,4,19,1000],['길드없음/혼자','1-4명','5-19명','20명+'])
# 교란 확인: 첫 주 플레이 시간 구간 안에서 길드 동료 효과
R['h_band']=pd.qcut(R.hours_w1.rank(method='first'),3,labels=['하','중','상'])
R['gm_band']=pd.cut(R.guildmates,[-1,0,4,1000],labels=['0','1-4','5+'])
print('\n[첫 주 플레이 시간 3분위 x 길드 동료] 정착률 (인원)')
print(R.pivot_table(index='h_band',columns='gm_band',values='settled',aggfunc='mean',observed=True).round(3).to_string())
print(R.pivot_table(index='h_band',columns='gm_band',values='settled',aggfunc='size',observed=True).to_string())
