import pandas as pd, numpy as np
df=pd.read_pickle('churn_project/step6/clean.pkl')
ts=pd.Series(sorted(df['timestamp'].dt.floor('min').unique()))
gap=ts.diff()
g=pd.DataFrame({'start':ts.shift(1),'end':ts,'hours':gap.dt.total_seconds()/3600})
g=g[g.hours>=1.5].reset_index(drop=True)
g['weekday']=g.start.dt.day_name().str[:3]
# 새 지역 첫 등장
first=df.groupby('zone',observed=True)['timestamp'].min()
first=first[first>'2008-01-15']
def newzones(e):
    z=first[(first>=e)&(first<e+pd.Timedelta('2D'))]
    return ', '.join(z.index.astype(str)[:4])+(f' 외 {len(z)-4}' if len(z)>4 else '')
g['new_zones']=g.end.apply(newzones)
# 점검 전후 3일 비교: 일 접속 캐릭터, 70레벨 비율
day=df.groupby(df.timestamp.dt.normalize()).agg(chars=('char','nunique'),lv70=('level',lambda x:(x>=70).mean()))
def comp(r):
    b=day.loc[r.start.normalize()-pd.Timedelta('3D'):r.start.normalize()-pd.Timedelta('1D')]
    a=day.loc[r.end.normalize()+pd.Timedelta('1D'):r.end.normalize()+pd.Timedelta('3D')]
    return pd.Series({'chars_chg%':(a.chars.mean()/b.chars.mean()-1)*100,'lv70_chg_pp':(a.lv70.mean()-b.lv70.mean())*100})
g=pd.concat([g,g.apply(comp,axis=1)],axis=1)
g.to_csv('update_project/downtime_all.csv',index=False,encoding='utf-8-sig')
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',60)
print('목요일 점검 시간 분포:'); print(g[g.weekday=='Thu'].hours.describe().round(2))
print(g[(g.hours>=10)|(g.weekday!='Thu')&(g.hours>=3)].round(1).to_string(index=False))
