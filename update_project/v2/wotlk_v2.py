# 리치왕 효과 기준선 재설계: '출시 직전 주' 대신 '공백기(10월~) 이전 추세'를 반사실로 사용
import sys, os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import common
import pandas as pd, numpy as np
o=common.load(); cd,cov,info=o['cd'],o['cov'],o['info']
OUT=os.path.dirname(os.path.abspath(__file__))+'/'
first=info['first']; surge=first[(first>='2008-10-07')&(first<'2008-10-27')].index
dk=info.index[info.cls.astype(str).str.contains('Death')]
good=cov[cov>=20].index
def thu(s): return s-pd.to_timedelta((s.dt.dayofweek-3)%7,unit='D')
x=cd[~cd.char.isin(surge)].copy(); x['wk']=thu(x.day)
gw=thu(pd.Series(good)).value_counts()
xs=x[x.day.isin(good)]
W=pd.DataFrame({'chars':x.groupby('wk').char.nunique(),'chars_noDK':x[~x.char.isin(dk)].groupby('wk').char.nunique(),
                'hours':xs.groupby('wk').h.sum()/gw*7,'hours_noDK':xs[~xs.char.isin(dk)].groupby('wk').h.sum()/gw*7,'gd':gw}).sort_index()
W=W[W.gd>=4]
LAUNCH=pd.date_range('2008-11-20','2008-12-25',freq='7D'); LULL=pd.date_range('2008-10-02','2008-11-13',freq='7D')
FITS={'7~9월 추세':('2008-07-03','2008-09-25'),'5~9월 추세':('2008-05-22','2008-09-25'),'9월 평균(추세 없음)':('2008-09-04','2008-09-25')}
rows=[]
for fn,(a,z) in FITS.items():
    f=W.loc[a:z]; t=(f.index-pd.Timestamp('2008-09-25')).days.values/7
    for m in ['chars','hours','chars_noDK','hours_noDK']:
        if fn.startswith('9월'): b=np.array([0,f[m].mean()]); sd=f[m].std()
        else: b=np.polyfit(t,f[m].values,1); sd=(f[m].values-np.polyval(b,t)).std(ddof=2)
        for per,idx in [('공백기 10/2~11/19',LULL),('출시 후 6주 11/20~12/31',LAUNCH),('출시 주',LAUNCH[:1])]:
            idx=[i for i in idx if i in W.index]; tt=(pd.DatetimeIndex(idx)-pd.Timestamp('2008-09-25')).days.values/7
            pred=np.polyval(b,tt); act=W.loc[idx,m].values
            rows.append({'fit':fn,'metric':m,'period':per,'actual':act.mean(),'pred':pred.mean(),'rel':act.mean()/pred.mean()-1,
                         'rel_lo':act.mean()/(pred.mean()+1.96*sd/np.sqrt(len(idx)))-1,'rel_hi':act.mean()/(pred.mean()-1.96*sd/np.sqrt(len(idx)))-1})
R=pd.DataFrame(rows); pd.set_option('display.width',220)
print(R.round(3).to_string(index=False)); R.to_csv(OUT+'wotlk_trend_effect.csv',index=False); W.to_csv(OUT+'weekly_series.csv')
# 사이클 전체 누적: 공백기 손실 vs 출시 후 증가 (7~9월 추세 기준, 플레이 시간)
f=W.loc['2008-07-03':'2008-09-25']; t=(f.index-pd.Timestamp('2008-09-25')).days.values/7; b=np.polyfit(t,f.hours.values,1)
for per,idx in [('공백기',LULL),('출시 후',LAUNCH)]:
    idx=[i for i in idx if i in W.index]; tt=(pd.DatetimeIndex(idx)-pd.Timestamp('2008-09-25')).days.values/7
    print(f'{per}: 추세 대비 누적 {(W.loc[idx,"hours"].values-np.polyval(b,tt)).sum():+,.0f}시간 ({len(idx)}주)')
# 수집 상태 점검: 라운드당 관측 상한
cc=pd.read_csv(OUT+'collection_check.csv',index_col=0,parse_dates=True)
print('\n수집 점검 (주별 구멍 비율, 라운드 간격, 라운드당 최대 관측):')
print(cc.loc['2008-09-18':,['hole_rate','gap_med','ccu_mean','ccu_p95','ccu_max']].round(3).to_string())
