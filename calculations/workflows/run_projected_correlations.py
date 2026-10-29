"""Reproducible independent projected-JK correlation replicas."""
from pathlib import Path
import json
import numpy as np
import projected_parton_exact as exact
import projected_correlations as corr


def gradient_with_jackknife(raw,j=1.,k=10.):
    energy=j*raw[:,0]+k*raw[:,1];q=raw[:,4];n=len(raw)
    gradient=-2*(np.mean(q*energy)-np.mean(q)*np.mean(energy))
    total=np.array([energy.sum(),q.sum(),(energy*q).sum()])
    replicates=[]
    for ids in np.array_split(np.arange(n),20):
        remaining=n-len(ids)
        sums=total-np.array([energy[ids].sum(),q[ids].sum(),(energy[ids]*q[ids]).sum()])
        replicates.append(-2*(sums[2]/remaining-sums[0]*sums[1]/remaining**2))
    replicates=np.array(replicates)
    error=np.sqrt(19/20*np.sum((replicates-replicates.mean())**2))
    return float(gradient),float(error)


def run():
    out=Path('output/projected-jk');out.mkdir(parents=True,exist_ok=True)
    records=[]
    setups=[(.03,-.11,6601,1024,4096),(.03,-.11,6602,1024,4096),
            (.33,-.11,6701,768,2048),(.33,-.11,6702,768,2048)]
    for r,g,seed,warmup,samples in setups:
        print('START',r,g,seed,flush=True)
        u,v,metadata=exact.orbitals(8,r)
        report,arrays=corr.sample_correlations(u,v,8,g=g,seed=seed,warmup=warmup,
                                              samples=samples,stride=3)
        stats=report['energy_stats'];coeff=np.array([1.,10.])
        cov=np.asarray(stats['covariance_jk_of_mean'])
        gradient,error=gradient_with_jackknife(arrays['raw_energy_samples'])
        stem=f'correlations_L8_r{r:g}_g{g:g}_seed{seed}'
        array_path=out/(stem+'.npz')
        np.savez_compressed(array_path,**arrays)
        report.update(r=r,orbital_metadata=metadata,array_file=str(array_path),
                      energy_K10=float(stats['e_j']+10*stats['e_k']),
                      energy_K10_error=float(np.sqrt(coeff@cov@coeff)),
                      gradient_K10=gradient,gradient_K10_error=error,
                      m_cdw_squared=float(arrays['sn'][4,4]/64),
                      m_cdw_squared_error=float(arrays['sn_error'][4,4]/64),
                      density_peak_q_index=np.unravel_index(np.argmax(arrays['sn']),arrays['sn'].shape),
                      density_peak=float(arrays['sn'].max()),
                      nb_min=float(arrays['nb'].min()),nb_max=float(arrays['nb'].max()),
                      pair_x_min=float(arrays['pair_x'].min()),pair_x_max=float(arrays['pair_x'].max()),
                      pair_y_min=float(arrays['pair_y'].min()),pair_y_max=float(arrays['pair_y'].max()),
                      pair_s_min=float(arrays['pair_s'].min()),pair_s_max=float(arrays['pair_s'].max()),
                      pair_d_min=float(arrays['pair_d'].min()),pair_d_max=float(arrays['pair_d'].max()))
        report['density_peak_q_index']=[int(x) for x in report['density_peak_q_index']]
        (out/(stem+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
        records.append(report)
        (out/'correlation_replicas.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        print('DONE',r,seed,json.dumps({key:report[key] for key in ['seconds','acceptance','energy_K10',
             'energy_K10_error','gradient_K10','gradient_K10_error','m_cdw_squared','m_cdw_squared_error',
             'density_peak_q_index','nb_min','nb_max','pair_s_min','pair_s_max','pair_d_min','pair_d_max']}),flush=True)


def run_l6():
    out=Path('output/projected-jk');out.mkdir(parents=True,exist_ok=True)
    u,v,metadata=exact.orbitals(6,.03)
    print('START L6',6801,flush=True)
    report,arrays=corr.sample_correlations(u,v,6,g=-.11,seed=6801,warmup=768,samples=4096,stride=3)
    stats=report['energy_stats'];coeff=np.array([1.,10.]);cov=np.asarray(stats['covariance_jk_of_mean'])
    gradient,error=gradient_with_jackknife(arrays['raw_energy_samples'])
    stem='correlations_L6_r0.03_g-0.11_seed6801';array_path=out/(stem+'.npz')
    np.savez_compressed(array_path,**arrays)
    report.update(r=.03,orbital_metadata=metadata,array_file=str(array_path),
                  energy_K10=float(stats['e_j']+10*stats['e_k']),energy_K10_error=float(np.sqrt(coeff@cov@coeff)),
                  gradient_K10=gradient,gradient_K10_error=error,
                  m_cdw_squared=float(arrays['sn'][3,3]/36),m_cdw_squared_error=float(arrays['sn_error'][3,3]/36),
                  density_peak_q_index=[int(x) for x in np.unravel_index(np.argmax(arrays['sn']),arrays['sn'].shape)],
                  density_peak=float(arrays['sn'].max()),nb_min=float(arrays['nb'].min()),nb_max=float(arrays['nb'].max()),
                  pair_x_min=float(arrays['pair_x'].min()),pair_x_max=float(arrays['pair_x'].max()),
                  pair_y_min=float(arrays['pair_y'].min()),pair_y_max=float(arrays['pair_y'].max()),
                  pair_s_min=float(arrays['pair_s'].min()),pair_s_max=float(arrays['pair_s'].max()),
                  pair_d_min=float(arrays['pair_d'].min()),pair_d_max=float(arrays['pair_d'].max()))
    (out/(stem+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    index=out/'correlation_replicas.json'
    records=json.loads(index.read_text()) if index.exists() else []
    records=[x for x in records if x.get('seed')!=6801]+[report]
    index.write_text(json.dumps(records,indent=2),encoding='utf-8')
    print('DONE L6',json.dumps({key:report[key] for key in ['seconds','acceptance','energy_K10','energy_K10_error',
          'm_cdw_squared','m_cdw_squared_error','gradient_K10','gradient_K10_error','nb_min','nb_max','pair_d_min','pair_d_max']}),flush=True)


if __name__=='__main__':
    run()
    run_l6()
