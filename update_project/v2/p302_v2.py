# 3.0.2 (대만 10/23 점검, 신규 지역 The North Sea 첫 등장) 단절 시계열 보완
# - 실제·가짜 날짜 모두 같은 기준의 '기존 캐릭터' 집단 사용 (E-16일 이전 첫 등장 = 실제 날짜에서 10/7 급증 이전)
# - 모형 3종(선형/로그선형, 사전 28일/49일)과 1주차·2주차를 따로 가짜 날짜 분포와 비교
import sys, os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import common
import pandas as pd, numpy as np
o=common.load(); cd,cov,info=o['cd'],o['cov'],o['info']
OUT=os.path.dirname(os.path.abspath(__file__))+'/'
good=cov[cov>=20].index
cd=cd.sort_values(['char','day']); cd['gap']=(cd.day-cd.groupby('char').day.shift(1)).dt.days
first=info['first'].dt.normalize()
REAL=[pd.Timestamp(t) for t in ['2008-03-27','2008-04-03','2008-05-15','2008-07-17','2008-10-23','2008-11-06','2008-11-18']]

def daily(E):
    coh=first[first<E-pd.Timedelta('16D')].index
    x=cd[cd.char.isin(coh)&(cd.day>=E-pd.Timedelta('50D'))&(cd.day<=E+pd.Timedelta('14D'))]
    g=x.groupby('day')
    d=pd.DataFrame({'dau':g.size(),'play_hours':g.h.sum(),'lv70':g.lv.apply(lambda s:(s>=70).mean()),
                    'returners':g.gap.apply(lambda s:(s>28).sum())})
    d['play_hours']=d.play_hours/cov.reindex(d.index)*24      # 수집 누락 시간 보정한 하루 총 플레이 시간
    return d[d.index.isin(good)&(d.index!=E)]

def X(idx,E,trend=True):
    t=(idx-E).days.values.astype(float); wd=np.eye(7)[idx.dayofweek][:,[1,2,4,5,6]]
    return np.c_[np.ones(len(idx)),t,wd] if trend else np.c_[np.ones(len(idx)),wd]
# (사전 일수, 로그, 추세 포함)
MODELS={'선형_49일':(49,False,True),'로그선형_49일':(49,True,True),'선형_28일':(28,False,True),'수준유지_14일':(14,False,False)}
MET=['dau','play_hours','lv70','returners']
def its(E,d=None):
    E=pd.Timestamp(E); d=daily(E) if d is None else d; out={}
    for mn,(L,lg,tr) in MODELS.items():
        pre=d.loc[E-pd.Timedelta(f'{L}D'):E-pd.Timedelta('1D')]
        for wk,(a,z) in enumerate([(1,6),(8,13)]):
            post=d.loc[E+pd.Timedelta(f'{a}D'):E+pd.Timedelta(f'{z}D')]
            for m in MET:
                y=pre[m].values.astype(float); 
                if lg: y=np.log(np.maximum(y,0.5))
                b,*_=np.linalg.lstsq(X(pre.index,E,tr),y,rcond=None); p=X(post.index,E,tr)@b
                if lg: p=np.exp(p)
                out[(mn,m,f'w{wk+1}')]=(post[m].values-p).mean()/p.mean()
            out[(mn,'n_post',f'w{wk+1}')]=len(post)
        out[(mn,'n_pre','')]=len(pre)
    return out

if __name__=='__main__':
    E0=pd.Timestamp('2008-10-23'); d0=daily(E0); real=its(E0,d0)
    cands=[t for t in pd.date_range('2008-02-21','2008-10-09',freq='W-THU') if all(not(-49<=(p-t).days<=14) for p in REAL)]
    print('가짜 날짜:',[t.strftime('%m/%d') for t in cands])
    pl=pd.DataFrame({t.strftime('%m/%d'):its(t) for t in cands}).T
    rows=[]
    for mn in MODELS:
        for m in MET:
            for w in ['w1','w2']:
                v=pl[(mn,m,w)].dropna(); r=real[(mn,m,w)]
                rows.append({'model':mn,'metric':m,'week':w,'real':r,'placebo_mean':v.mean(),'placebo_sd':v.std(),
                             'placebo_max':v.max(),'z':(r-v.mean())/v.std(),'p_emp':((v>=r).sum()+1)/(len(v)+1),'n_placebo':len(v)})
    R=pd.DataFrame(rows); pd.set_option('display.width',220)
    print(R.round(3).to_string(index=False)); R.to_csv(OUT+'p302_robust.csv',index=False)
    pl.to_csv(OUT+'p302_placebo.csv')
    # 3.0.3(11/6)도 같은 방식으로 1주만 (2주차는 확장팩 직전이라 제외)
    r3=its('2008-11-06')
    print('\n3.0.3 (11/6) 1주차:',{f'{k[0]}/{k[1]}':round(v,3) for k,v in r3.items() if k[2]=='w1' and k[1] in MET})
