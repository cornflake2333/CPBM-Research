"""Reproducible projected NN-Jastrow grid, using the existing VMC sampler.

Output is saved after every completed independent chain. Full raw samples
are used only to form covariance-aware errors, then removed from JSON.
"""

from vmc_paths import PROJECT_ROOT
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import json
import os

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'

import projected_parton_exact as exact
import projected_parton_vmc as vmc
import numpy as np

ROOT=PROJECT_ROOT


def worker(job):
    size,g,seed=job
    u1,u2,meta=exact.orbitals(size,.03,(.37,.19))
    result=vmc.sample(u1,u2,size,g=g,seed=seed,warmup=768,
                      samples=4096,stride=3,return_samples=True)
    raw=np.asarray(result.pop('raw_samples'))
    count=len(raw);block_size=count//20;blocks=count//block_size
    block=raw[:blocks*block_size].reshape(blocks,block_size,raw.shape[1])
    result['m_cdw_squared_error']=result['s_pi_pi_error']/(size*size)
    result['orbital_metadata']=meta
    result['combined_energy_and_gradient']=[]
    for k in range(1,11):
        e=raw[:,0]+k*raw[:,1]
        eb=block[:,:,0]+k*block[:,:,1]
        means=eb.mean(axis=1)
        energy_error=float(means.std(ddof=1)/np.sqrt(blocks))
        gradient=float(-2*((e*raw[:,4]).mean()-e.mean()*raw[:,4].mean()))
        bg=-2*((eb*block[:,:,4]).mean(axis=1)-eb.mean(axis=1)*block[:,:,4].mean(axis=1))
        gradient_error=float(bg.std(ddof=1)/np.sqrt(blocks))
        result['combined_energy_and_gradient'].append(dict(j=1.,k=k,
            energy=float(e.mean()),energy_error=energy_error,
            gradient_g=gradient,gradient_g_error=gradient_error))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    values=[-.25,-.15,-.10,-.05,0.,.10]
    jobs=[(8,g,2200+i) for i,g in enumerate(values)]
    jobs.extend([(6,0.,2300),(6,-.10,2301)])
    out=ROOT/'output/projected-jk/vmc8_jastrow_grid.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    result=dict(r=.03,twist_pi=[.37,.19],j=1.,
                g_grid_L8=values,warmup=768,samples=4096,stride=3,
                error_method='20 contiguous blocks, covariance-aware combined energies and gradients',
                records=[],pending=[list(j) for j in jobs])
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(worker,j):j for j in jobs}
        for future in as_completed(futures):
            job=futures[future]
            record=future.result()
            result['records'].append(record)
            result['records'].sort(key=lambda r:(r['L'],r['g']))
            result['pending'].remove(list(job))
            out.write_text(json.dumps(result,indent=2),encoding='utf-8')
            selected=record['combined_energy_and_gradient'][-1]
            print(json.dumps(dict(L=job[0],g=job[1],seed=job[2],
                                  energy_K10=selected['energy'],error=selected['energy_error'],
                                  gradient_K10=selected['gradient_g'],gradient_error=selected['gradient_g_error'],
                                  m2=record['m_cdw_squared'],m2_error=record['m_cdw_squared_error'],
                                  acceptance=record['acceptance'],seconds=record['seconds'],
                                  pending=len(result['pending']))),flush=True)
    print('Saved',str(out),flush=True)


if __name__=='__main__':
    main()
