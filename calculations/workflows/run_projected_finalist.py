"""Independent long-chain L8 projected-JK finalist validation.

python run_projected_finalist.py
The selected r,g are fixed before these independent replicas are sampled.
"""
from pathlib import Path
import json
import numpy as np
import projected_parton_exact as exact
import projected_parton_vmc as vmc
from projected_vmc_validation import ess


def blocking(raw):
    records=[]
    for block_size in (32,64,128,256,512,1024):
        count=len(raw)//block_size
        means=raw[:count*block_size].reshape(count,block_size,6).mean(axis=1)
        cov=np.cov(means[:,:2],rowvar=False,ddof=1)/count
        records.append(dict(block_size=block_size,blocks=count,
                            energy_k_coefficient_error=float(np.sqrt(cov[1,1])),
                            energy_K10_error=float(np.sqrt(np.array([1.,10.])@cov@np.array([1.,10.]))),
                            m_cdw_squared_error=float(means[:,5].std(ddof=1)/np.sqrt(count)/64)))
    return records


def run():
    out=Path('output/projected-jk');out.mkdir(parents=True,exist_ok=True)
    u,v,meta=exact.orbitals(8,.03)
    replicas=[]
    target=out/'vmc8_finalist_replicas.json'
    for seed in (5501,5502):
        print('START',seed,flush=True)
        result=vmc.sample(u,v,8,g=-.10,seed=seed,warmup=1024,
                          samples=8192,stride=3,return_samples=True)
        raw=np.asarray(result.pop('raw_samples'))
        raw_path=out/f'vmc8_finalist_r003_gm010_seed{seed}.npz'
        np.savez_compressed(raw_path,raw_samples=raw,columns=np.array(result['sample_columns']))
        cov=np.asarray(result['covariance_jk_of_mean'])
        coeff=np.array([1.,10.])
        result.update(energy_K10=float(coeff@np.array([result['e_j'],result['e_k']])),
                      energy_K10_error=float(np.sqrt(coeff@cov@coeff)),
                      gradient_K10=float(result['gradient_j_g']+10*result['gradient_k_g']),
                      gradient_K10_error=float(np.hypot(result['gradient_j_g_error'],10*result['gradient_k_g_error'])),
                      m_cdw_squared_error=float(result['s_pi_pi_error']/64),
                      ess_e_k=ess(raw[:,1]),ess_m_cdw=ess(raw[:,5]),blocking=blocking(raw),
                      raw_file=str(raw_path))
        replicas.append(result)
        target.write_text(json.dumps(dict(metadata=meta,g=-.10,replicas=replicas),indent=2),encoding='utf-8')
        print('DONE',seed,json.dumps({k:result[k] for k in ['e_j','e_k','energy_K10','energy_K10_error',
                    'gradient_K10','gradient_K10_error','m_cdw_squared','m_cdw_squared_error','acceptance','seconds']}),flush=True)
    a,b=replicas
    cov=(np.asarray(a['covariance_jk_of_mean'])+np.asarray(b['covariance_jk_of_mean']))/4
    mean=np.array([[(r['e_j']),(r['e_k'])] for r in replicas]).mean(axis=0)
    m=(a['m_cdw_squared']+b['m_cdw_squared'])/2
    m_error=np.hypot(a['m_cdw_squared_error'],b['m_cdw_squared_error'])/2
    grad=(a['gradient_K10']+b['gradient_K10'])/2
    grad_error=np.hypot(a['gradient_K10_error'],b['gradient_K10_error'])/2
    combined=dict(e_j=float(mean[0]),e_k=float(mean[1]),covariance_jk_of_mean=cov.tolist(),
                  energy_K10=float(coeff@mean),energy_K10_error=float(np.sqrt(coeff@cov@coeff)),
                  gradient_K10=float(grad),gradient_K10_error=float(grad_error),
                  m_cdw_squared=float(m),m_cdw_squared_error=float(m_error),
                  seed_energy_difference=float(a['energy_K10']-b['energy_K10']),
                  seed_energy_difference_error=float(np.hypot(a['energy_K10_error'],b['energy_K10_error'])),
                  seed_m2_difference=float(a['m_cdw_squared']-b['m_cdw_squared']),
                  seed_m2_difference_error=float(np.hypot(a['m_cdw_squared_error'],b['m_cdw_squared_error'])),
                  seeds=[5501,5502],warmup_sweeps=1024,samples_per_seed=8192,stride_sweeps=3)
    report=dict(metadata=meta,g=-.10,selection='Fixed finalist chosen before independent replicas',
                replicas=replicas,combined=combined)
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('COMBINED',json.dumps(combined),flush=True)


if __name__=='__main__':run()
