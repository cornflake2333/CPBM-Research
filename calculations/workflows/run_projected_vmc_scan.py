"""Reproducible L=8, g=0 scan over all distinct closed-shell parton seas.

Uses projected_parton_exact.orbitals and projected_parton_vmc.sample without
changing either implementation. Results are checkpointed after every state.
"""

from vmc_paths import PROJECT_ROOT
from pathlib import Path
import json
import time
import numpy as np
from projected_parton_exact import closed_shell_intervals, orbitals, references
from projected_parton_vmc import sample

ROOT=PROJECT_ROOT
OUT=ROOT/'output/projected-jk/vmc8_pure_scan.json'


def summarize(records):
    rows=[]
    for k in range(1,11):
        evaluated=[]
        for entry in records:
            r=entry['result'];cov=np.asarray(r['covariance_jk_of_mean'])
            e=r['e_j']+k*r['e_k']
            se=float(np.sqrt(max(0.,cov[0,0]+2*k*cov[0,1]+k*k*cov[1,1])))
            evaluated.append((e,se,entry))
        if not evaluated:continue
        e,se,best=min(evaluated,key=lambda x:x[0])
        ref=references(64,1.,k)
        dimer=min(ref[name] for name in ('staggered_dimer','signed_dimer','plaquette','phase_dimer'))
        rows.append(dict(k=k,state_index=best['state_index'],r_lo=best['r_lo'],r_hi=best['r_hi'],
                         r=best['r'],energy=e,error=se,m_cdw_squared=best['result']['m_cdw_squared'],
                         m_cdw_squared_error=best['result']['s_pi_pi_error']/64,
                         lowest_dimer_reference=dimer,constant_reference=ref['constant_boson'],
                         difference_from_dimer=e-dimer,difference_from_constant=e-ref['constant_boson']))
    return rows


def main():
    intervals=closed_shell_intervals(8,(.37,.19))
    OUT.parent.mkdir(parents=True,exist_ok=True)
    data=dict(L=8,g=0.,twist_pi=[.37,.19],warmup=512,samples=2048,stride=2,
              expected_states=len(intervals),records=[],summary=[])
    if OUT.exists():
        previous=json.loads(OUT.read_text(encoding='utf-8'))
        if all(previous.get(k)==data[k] for k in ('L','g','twist_pi','warmup','samples','stride','expected_states')):
            data['records']=previous.get('records',[])
    completed={v['state_index'] for v in data['records']}
    start=time.perf_counter()
    print('START',len(intervals),'states;',len(completed),'already complete',flush=True)
    for index,interval in enumerate(intervals):
        if index in completed:continue
        u1,u2,metadata=orbitals(8,interval['r'],(.37,.19))
        print('STATE_START',index,interval['r_lo'],interval['r_hi'],flush=True)
        result=sample(u1,u2,8,g=0.,seed=1000+index,warmup=512,samples=2048,stride=2,return_samples=False)
        result.pop('raw_samples',None)
        data['records'].append(dict(state_index=index,r_lo=interval['r_lo'],r_hi=interval['r_hi'],
                                    r=interval['r'],metadata=metadata,result=result))
        data['records'].sort(key=lambda v:v['state_index'])
        data['summary']=summarize(data['records'])
        data['completed_states']=len(data['records'])
        data['complete']=len(data['records'])==len(intervals)
        OUT.write_text(json.dumps(data,indent=2),encoding='utf-8')
        print('STATE_DONE',index,'eJ',result['e_j'],'eK',result['e_k'],
              'm2',result['m_cdw_squared'],'accept',result['acceptance'],'seconds',result['seconds'],flush=True)
    print('SUMMARY',json.dumps(data['summary']),flush=True)
    print('FINISHED',str(OUT),'new elapsed seconds',time.perf_counter()-start,flush=True)


if __name__=='__main__':
    main()
