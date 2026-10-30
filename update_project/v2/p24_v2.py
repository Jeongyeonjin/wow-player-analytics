# 2.4.0 이중차분 보완: (1) 사전 추세가 왜 기울었는지 진단 (2) 그룹 정의를 바꿔 재추정 (3) 가짜 패치일 분포로 보정
import sys, os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import common
import pandas as pd, numpy as np
o=common.load(); cd,cov=o['cd'],o['cov']
OUT=os.path.dirname(os.path.abspath(__file__))+'/'
REAL=['2008-03-27','2008-04-03','2008-05-15','2008-07-17','2008-10-23','2008-11-06','2008-11-18']  # 대만 패치 점검일
rng=np.random.default_rng(0)

def build(MAINT, spec, wmin=-8, wmax=4):
    M=pd.Timestamp(MAINT); A=M+pd.Timedelta('1D'); P=M-pd.Timedelta('1D')
    def wk(dd): dd=pd.DatetimeIndex(dd); return np.where(dd>=A,(dd-A).days//7,-1-(P-dd).days//7)
    gd=cov[(cov>=20)&(cov.index!=M)].index; ng=pd.Series(wk(gd),index=gd).value_counts()
    weeks=[k for k in range(wmin,wmax+1) if ng.get(k,0)>=3]
    x=cd[cd.day.isin(gd)].copy(); x['week']=wk(x.day); x=x[x.week.isin(weeks)]
    if spec in ('S0','S1'):      # 패치 직전 3주(-3~-1주)에 접속한 캐릭터를 그 시점 레벨로 분류 (기존 방식)
        s=x[(x.day>=M-pd.Timedelta('21D'))&(x.day<M)].sort_values('day').groupby('char').lv.last()
        grp=pd.Series(np.where(s>=70,'T',np.where(s>=60,'C','x')),index=s.index)
    elif spec=='S3':             # S3: S1과 같되 대조군을 40~69레벨로 넓힘
        s=x[(x.day>=M-pd.Timedelta('21D'))&(x.day<M)].sort_values('day').groupby('char').lv.last()
        grp=pd.Series(np.where(s>=70,'T',np.where(s>=40,'C','x')),index=s.index)
    else:                        # S2: 분석 창 시작(-8~-6주) 레벨로 미리 분류, 패치 전 70 도달한 대조군은 제외
        e=x[x.week<=wmin+2].sort_values('day').groupby('char').lv.last()
        grp=pd.Series(np.where(e>=70,'T',np.where(e>=60,'C','x')),index=e.index)
        lvpre=x[x.week<0].groupby('char').lv.max()
        grp[(grp=='C')&(lvpre.reindex(grp.index)>=70)]='x'
    grp=grp[grp!='x']
    a=x[x.char.isin(grp.index)].groupby(['char','week']).agg(hours=('h','sum'),days=('day','nunique'))
    idx=pd.MultiIndex.from_product([grp.index,weeks],names=['char','week'])
    pan=a.reindex(idx,fill_value=0).reset_index(); pan['ng']=pan.week.map(ng)
    pan['hours']=pan.hours/pan.ng*7; pan['days']=pan.days/pan.ng*7
    if spec in ('S1','S2','S3'):      # 사전 모든 주에 1일 이상 접속한 캐릭터만 (진입·이탈 효과 제거)
        pre=pan[pan.week<0].groupby('char').hours.min(); keep=pre[pre>0].index; grp=grp[grp.index.isin(keep)]
    mats={}
    for g in 'TC':
        ch=grp[grp==g].index; p=pan[pan.char.isin(ch)]
        mats[g]={m:p.pivot(index='char',columns='week',values=m).values for m in ['hours','days']}
    return np.array(weeks),mats

def est(W,T,C):
    pre=W<0; ref=(W>=-4)&(W<0)
    c=(T-T[:,ref].mean(1,keepdims=True)).mean(0)-(C-C[:,ref].mean(1,keepdims=True)).mean(0)
    slope=np.polyfit(W[pre],c[pre],1)[0]
    return c,slope

def run(MAINT,spec,B=300,**kw):
    W,mats=build(MAINT,spec,**kw); res={'nT':len(mats['T']['hours']),'nC':len(mats['C']['hours']),'weeks':list(W)}
    for m in ['hours','days']:
        T,C=mats['T'][m],mats['C'][m]; c,s=est(W,T,C); base=T[:,W<0].mean()
        bc=[];bs=[]
        for _ in range(B):
            r=est(W,T[rng.integers(0,len(T),len(T))],C[rng.integers(0,len(C),len(C))]); bc.append(r[0]); bs.append(r[1])
        bc=np.array(bc); bs=np.array(bs)
        res[f'{m}_base']=base; res[f'{m}_preslope']=s
        if B: res[f'{m}_preslope_lo'],res[f'{m}_preslope_hi']=np.percentile(bs,[2.5,97.5])
        for k in W:
            i=list(W).index(k); res[f'{m}_w{k}']=c[i]
            if B: res[f'{m}_w{k}_lo'],res[f'{m}_w{k}_hi']=np.percentile(bc[:,i],[2.5,97.5])
    return res

if __name__=='__main__':
    pd.set_option('display.width',250)
    # 1) 진단: 사양별 사전 추세와 효과
    rows={}
    for spec,lab in [('S0','S0 기존(직전 3주 선택)'),('S1','S1 기존+사전 매주 접속'),('S2','S2 조기 분류+사전 매주 접속'),('S3','S3 S1+대조군 40~69')]:
        r=run('2008-03-27',spec); rows[lab]=r
        print(f"\n[{lab}] T {r['nT']}명 / C {r['nC']}명, 주차 {r['weeks']}")
        for m in ['hours','days']:
            print(f"  {m}: 사전 기울기 {r[m+'_preslope']:+.3f}/주 (CI {r[m+'_preslope_lo']:+.3f}~{r[m+'_preslope_hi']:+.3f}), 처치군 사전평균 {r[m+'_base']:.2f}")
            print('   주차별 DiD: '+'  '.join(f"{k}:{r[f'{m}_w{k}']:+.2f}[{r[f'{m}_w{k}_lo']:+.1f},{r[f'{m}_w{k}_hi']:+.1f}]" for k in r['weeks']))
    pd.DataFrame(rows).T.to_csv(OUT+'p24_specs.csv')
    # 2) 가짜 패치일: 창(-6~+3주) 안에 실제 패치가 없는 목요일
    real=[pd.Timestamp(t) for t in REAL]
    cands=[t for t in pd.date_range('2008-02-14','2008-10-02',freq='W-THU')
           if all(not (-42<= (p-t).days <=27) for p in real)]
    print('\n가짜 패치일 후보:',[t.strftime('%m/%d') for t in cands])
    for SP in ['S0','S1','S3']:
        print('\n== 가짜 패치일 검증',SP)
        pl={}
        for t in cands:
            r=run(t,SP,B=0,wmin=-6,wmax=3)
            if 0 not in r['weeks']: continue
            pl[t.strftime('%m/%d')]={k:v for k,v in r.items() if k!='weeks'}
        realr=run('2008-03-27',SP,B=0,wmin=-6,wmax=3)
        P=pd.DataFrame(pl).T
        for m in ['hours','days']:
            for k in [0,2,3]:
                col=f'{m}_w{k}'
                if col not in P: continue
                P[col+'_rel']=P[col]/P[m+'_base']; rv=realr[col]/realr[m+'_base']; v=P[col+'_rel'].dropna()
                print(f"{col}: 실제 {rv:+.1%} / 가짜 {len(v)}개 평균 {v.mean():+.1%}, 표준편차 {v.std():.1%}, 최대 {v.max():+.1%}, "
                      f"경험적 p={((v>=rv).sum()+1)/(len(v)+1):.3f}, 가짜 평균 차감 {rv-v.mean():+.1%}")
        P.round(4).to_csv(OUT+f'p24_placebo_{SP}.csv')
