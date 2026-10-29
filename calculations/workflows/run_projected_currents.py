"""Independent replicas of the projected physical-boson current correlator."""
from pathlib import Path
import argparse
import json
import numpy as np
from projected_parton_exact import orbitals
from projected_currents import sample_currents
from run_projected_correlations import gradient_with_jackknife


def run(L=8,r=.03,g=-.105,seed=7101,samples=8192,warmup=1024,stride=3):
    out=Path('output/projected-currents');out.mkdir(parents=True,exist_ok=True)
    stem=f'currents_L{L}_r{r:g}_g{g:g}_seed{seed}'
    target=out/(stem+'.json')
    if target.exists() and (out/(stem+'.npz')).exists():
        print('EXISTING',target,flush=True)
        return json.loads(target.read_text())
    u,v,metadata=orbitals(L,r)
    print('START',L,r,g,seed,samples,flush=True)
    report,arrays=sample_currents(u,v,L,g,J=1.,K=10.,seed=seed,warmup=warmup,
                                 samples=samples,stride=stride)
    gradient,error=gradient_with_jackknife(arrays['raw_energy_samples'])
    report.update(r=r,orbital_metadata=metadata,gradient_K10=gradient,gradient_K10_error=error,
                  selection='g=-0.105 interpolates previously measured opposite-sign gradients at -0.11 and -0.10; restricted ansatz optimization',
                  array_file=str(out/(stem+'.npz')))
    np.savez_compressed(out/(stem+'.npz'),**arrays)
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    keys=('seconds','energy','energy_error','gradient_K10','gradient_K10_error',
          'total_current_variance_per_site','total_current_variance_error','diamagnetic','hopping_square_per_site')
    print('DONE',json.dumps({k:report[k] for k in keys}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--size',type=int,default=8)
    p.add_argument('--r',type=float,default=.03);p.add_argument('--g',type=float,default=-.105)
    p.add_argument('--seed',type=int,default=7101);p.add_argument('--samples',type=int,default=8192)
    p.add_argument('--warmup',type=int,default=1024);p.add_argument('--stride',type=int,default=3)
    a=p.parse_args();run(a.size,a.r,a.g,a.seed,a.samples,a.warmup,a.stride)
