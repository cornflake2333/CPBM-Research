"""Supplementary independent L=6 projected-current VMC replicas."""

from vmc_paths import PROJECT_ROOT
from pathlib import Path
import json
import projected_parton_exact as exact
import projected_currents as currents
import numpy as np

def run():
    out=PROJECT_ROOT/'output/projected-currents'
    out.mkdir(parents=True,exist_ok=True)
    u,v,metadata=exact.orbitals(6,.03)
    for seed in (7301,7302):
        stem=f'currents_L6_r0.03_g-0.105_seed{seed}'
        if (out/(stem+'.json')).exists():continue
        print('START',stem,flush=True)
        report,arrays=currents.sample_currents(u,v,6,g=-.105,J=1.,K=10.,seed=seed,
                                               warmup=768,samples=4096,stride=3)
        report.update(r=.03,orbital_metadata=metadata,array_file=str(out/(stem+'.npz')))
        np.savez_compressed(out/(stem+'.npz'),**arrays)
        (out/(stem+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('DONE',stem,report['seconds'],report['total_current_variance_per_site'],flush=True)

if __name__=='__main__':run()
