"""Monte Carlo measurement schedule and block accumulation."""
import time
import numpy as np
import projected_parton_vmc as vmc
from projected_correlations import observe, fourier, _array_summary

def sample_correlations(U1,U2,L,g=0.,seed=1,warmup=256,samples=1024,stride=3,progress_every=0):
    """Return (JSON-serializable report, numpy arrays) with 20 contiguous blocks.

    Observable errors are direct block errors of the Fourier estimators, not
    independent propagation of errors of individual real-space separations.
    All raw scalar energy measurements are retained; spatial data retain block
    means, enough to combine independent replicas without a large raw tensor.
    """
    if samples<40 or stride<1 or warmup<0:
        raise ValueError('Use samples>=40, stride>=1 and warmup>=0')
    start=time.perf_counter();state=vmc.ProjectedState(U1,U2,L,g,seed=seed)
    accepted=attempted=0
    for _ in range(warmup*state.N):
        accepted+=state.proposal();attempted+=1
    spatial={key:[] for key in ('cn','gb','pair')};raw=np.empty((samples,6))
    for measurement in range(samples):
        for _ in range(stride*state.N):
            accepted+=state.proposal();attempted+=1
        values=observe(state)
        for key in spatial:spatial[key].append(values[key])
        ej,ek,q,spi=state.local_energy()
        raw[measurement]=[ej.real,ek.real,ej.imag,ek.imag,q,spi]
        if progress_every and (measurement+1)%progress_every==0:
            print(f'L={L} seed={seed} measurements={measurement+1}/{samples} elapsed={time.perf_counter()-start:.1f}s',flush=True)
    arrays={};mean={};blocks={}
    for key,values in spatial.items():
        values=np.stack(values);average,errors,b=_array_summary(values)
        mean[key]=average;blocks[key]=b;arrays[key]=average;arrays['blocks_'+key]=b
        for label,error in errors.items():arrays[key+'_'+label]=error
    momentum=fourier(mean['cn'],mean['gb'],mean['pair'])
    momentum_blocks=fourier(blocks['cn'],blocks['gb'],blocks['pair'])
    for key,value in momentum.items():
        arrays[key]=value;arrays[key+'_error']=momentum_blocks[key].std(axis=0,ddof=1)/np.sqrt(20)
        arrays['blocks_'+key]=momentum_blocks[key]
    arrays['raw_energy_samples']=raw
    stats=vmc._stats(raw);stats['m_cdw_squared']=stats['s_pi_pi']/state.N
    report=dict(L=L,sites=state.N,bosons=state.M,g=float(g),seed=int(seed),warmup_sweeps=warmup,
                samples=samples,stride_sweeps=stride,acceptance=float(accepted/attempted),
                seconds=float(time.perf_counter()-start),energy_stats=stats,
                pair_definition='Delta_x(i)=b_i b_(i+x), Delta_y(i)=b_i b_(i+y); s,d=(x +/- y)/sqrt(2); site anchors, no half-bond phases',
                normalization=dict(mean_nb=float(arrays['nb'].mean()),mean_sn=float(arrays['sn'].mean()),
                                   mean_pair_x=float(arrays['pair_x'].mean()),mean_pair_y=float(arrays['pair_y'].mean())),
                boson_offdiagonal_max=float(np.max(abs(mean['gb'].ravel()[1:]))),
                pair_offdiagonal_max=float(max(np.max(abs(mean['pair'][0,0].ravel()[1:])),np.max(abs(mean['pair'][1,1].ravel()[1:])),np.max(abs(mean['pair'][0,1])),np.max(abs(mean['pair'][1,0])))))
    return report,arrays

