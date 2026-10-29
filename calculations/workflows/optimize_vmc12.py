"""Correlated-sampling (r,g) search followed by independent validation.

E_loc(g) is recomputed at every target g. Reweighting E_loc(g0) alone
would be incorrect for off-diagonal operators and is explicitly avoided.
"""

from vmc_paths import PROJECT_ROOT
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json
import time
import projected_parton_exact as exact
import projected_parton_vmc as vmc
import numpy as np

ROOT=PROJECT_ROOT
OUT=ROOT/'output/vmc12-optimized'
GGRID=np.linspace(-.25,.05,13)

def ratios_summary(q,energies,ggrid,g0):
    logw=-2*(q[:,None]-q.mean())*(ggrid[None,:]-g0)
    w=np.exp(logw-logw.max(axis=0));s=w.sum(axis=0)
    e=(w*energies.real).sum(axis=0)/s
    imag=(w*energies.imag).sum(axis=0)/s
    eq=(w*q[:,None]).sum(axis=0)/s
    grad=-2*((w*energies.real*q[:,None]).sum(axis=0)/s-e*eq)
    ess=s*s/(w*w).sum(axis=0)
    estimates=[];grads=[]
    for ids in np.array_split(np.arange(len(q)),20):
        ss=s-w[ids].sum(axis=0)
        ee=((w*energies.real).sum(axis=0)-(w[ids]*energies[ids].real).sum(axis=0))/ss
        qq=((w*q[:,None]).sum(axis=0)-(w[ids]*q[ids,None]).sum(axis=0))/ss
        qe=((w*energies.real*q[:,None]).sum(axis=0)-(w[ids]*energies[ids].real*q[ids,None]).sum(axis=0))/ss
        estimates.append(ee);grads.append(-2*(qe-ee*qq))
    def jack(a):
        a=np.asarray(a);return np.sqrt(19/20*np.sum((a-a.mean(axis=0))**2,axis=0))
    se=jack(estimates);sg=jack(grads)
    return [dict(g=float(g),energy=float(en),error=float(er),gradient=float(gr),gradient_error=float(ge),ess=float(es),ess_fraction=float(es/len(q)),imaginary_energy=float(im)) for g,en,er,gr,ge,es,im in zip(ggrid,e,se,grad,sg,ess,imag)]

def worker(job):
    r,seed=job;stem=f'scan_r{r:g}_seed{seed}';jp=OUT/(stem+'.json')
    if jp.exists():return json.loads(jp.read_text())
    start=time.perf_counter();g0=-.10
    u,v,meta=exact.orbitals(12,r);state=vmc.ProjectedState(u,v,12,g0,seed=seed)
    accepted=attempted=0
    for _ in range(256*144):accepted+=state.proposal();attempted+=1
    q=np.empty(1024);energies=np.empty((1024,len(GGRID)),complex)
    for m in range(len(q)):
        for _ in range(3*144):accepted+=state.proposal();attempted+=1
        q[m]=state.q_nn()
        for k,g in enumerate(GGRID):
            state.g=g;ej,ek,_,_=state.local_energy();energies[m,k]=ej+10*ek
        state.g=g0
    result=dict(r=r,seed=seed,g0=g0,L=12,Nb=72,J=1,K=10,samples=len(q),warmup=256,stride=3,acceptance=accepted/attempted,seconds=time.perf_counter()-start,metadata=meta,grid=ratios_summary(q,energies,GGRID,g0))
    np.savez_compressed(OUT/(stem+'.npz'),Q_nn=q,energies=energies,g_grid=GGRID,g0=g0)
    jp.write_text(json.dumps(result,indent=2),encoding='utf-8');return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    # Distinct occupied seas only; .25 and .33 are the same state on L12.
    rs=[.03,.08,.15,.33,.5,.75,1.]
    jobs=[(r,seed) for r in rs for seed in (2201,2202)];records=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(worker,j) for j in jobs]):
            record=f.result();records.append(record)
            print('SCAN_DONE',record['r'],record['seed'],'seconds',round(record['seconds'],1),'best',min(record['grid'],key=lambda s:s['energy']),flush=True)
            (OUT/'scan_checkpoint.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    combined=[]
    for r in rs:
        pair=[z for z in records if z['r']==r];grid=[]
        for i,g in enumerate(GGRID):
            v=[z['grid'][i] for z in pair]
            grid.append(dict(g=float(g),energy=float(np.mean([z['energy'] for z in v])),error=float(np.linalg.norm([z['error'] for z in v])/2),gradient=float(np.mean([z['gradient'] for z in v])),gradient_error=float(np.linalg.norm([z['gradient_error'] for z in v])/2),min_ess_fraction=min(z['ess_fraction'] for z in v)))
        reliable=[z for z in grid if z['min_ess_fraction']>=.2]
        best=min(reliable,key=lambda z:z['energy'])
        # Use local quadratic interpolation to propose g, then validate with new chains.
        i=min(range(len(grid)),key=lambda j:abs(grid[j]['g']-best['g']))
        proposal=best['g']
        if 0<i<len(grid)-1:
            local=grid[i-1:i+2];a,b,c=np.polyfit([z['g'] for z in local],[z['energy'] for z in local],2)
            if a>0 and min(z['min_ess_fraction'] for z in local)>=.2:
                proposal=float(np.clip(-b/(2*a),local[0]['g'],local[-1]['g']))
        combined.append(dict(r=r,grid=grid,best_grid=best,proposed_g=proposal))
    combined.sort(key=lambda z:z['best_grid']['energy'])
    selection=[combined[0],min((z for z in combined if z['r']!=.03),key=lambda z:z['best_grid']['energy'])]
    # Avoid repeating same shell when the global winner is not the strong shell.
    if selection[0]['r']==selection[1]['r']:selection=[combined[0],next(z for z in combined if z['r']==.03)]
    result=dict(L=12,J=1,K=10,method='two independent reference chains per distinct r sea; exact target-g local energies; 20-block delete-block jackknife; ESS>=20% selection; fresh-chain validation required',records=combined,selected=selection)
    (OUT/'optimization.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('SELECTED',json.dumps([dict(r=z['r'],g=z['proposed_g'],grid=z['best_grid']) for z in selection]),flush=True)

if __name__=='__main__':main()
