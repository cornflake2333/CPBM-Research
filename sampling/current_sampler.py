"""Monte Carlo measurement schedule and block accumulation."""
import time
import numpy as np
import projected_parton_vmc as vmc
from projected_currents import observe, fourier, longitudinal_transverse
from projected_correlations import _array_summary

def sample_currents(U1,U2,L,g=0.,J=1.,K=10.,seed=1,warmup=512,samples=2048,stride=3):
    if samples<40:raise ValueError('At least 40 measurements required')
    start=time.perf_counter();state=vmc.ProjectedState(U1,U2,L,g,seed=seed)
    accepted=attempted=0
    for _ in range(warmup*state.N):
        accepted+=state.proposal();attempted+=1
    data={key:[] for key in ('cc','hh','diamag','mean_current')}
    raw=np.empty((samples,6))
    for t in range(samples):
        for _ in range(stride*state.N):
            accepted+=state.proposal();attempted+=1
        for key,value in observe(state,J,True).items():data[key].append(value)
        ej,ek,q,spi=state.local_energy()
        raw[t]=ej.real,ek.real,ej.imag,ek.imag,q,spi
    arrays={}
    for key,values in data.items():
        average,errors,blocks=_array_summary(np.asarray(values))
        arrays[key]=average;arrays[key+'_error']=errors['error'];arrays['blocks_'+key]=blocks
    blocks_q=fourier(arrays['blocks_cc']);arrays['sjj']=fourier(arrays['cc'])
    arrays['blocks_sjj']=blocks_q
    arrays['sjj_error_real']=blocks_q.real.std(axis=0,ddof=1)/np.sqrt(20)
    arrays['sjj_error_imag']=blocks_q.imag.std(axis=0,ddof=1)/np.sqrt(20)
    arrays['raw_energy_samples']=raw
    long,trans=longitudinal_transverse(arrays['sjj']);bl,bt=longitudinal_transverse(blocks_q)
    arrays.update(slong=long,strans=trans,slong_error=bl.std(axis=0,ddof=1)/np.sqrt(20),
                  strans_error=bt.std(axis=0,ddof=1)/np.sqrt(20))
    stats=vmc._stats(raw);stats['m_cdw_squared']=stats['s_pi_pi']/state.N
    coef=np.array([J,K]);cov=np.asarray(stats['covariance_jk_of_mean'])
    report=dict(L=L,sites=state.N,g=float(g),J=J,K=K,seed=seed,samples=samples,
                warmup_sweeps=warmup,stride_sweeps=stride,acceptance=accepted/attempted,
                seconds=time.perf_counter()-start,energy=J*stats['e_j']+K*stats['e_k'],
                energy_error=float(np.sqrt(coef@cov@coef)),energy_stats=stats,
                current_definition='j_a(i)=i J (bdag_(i+e_a) b_i-bdag_i b_(i+e_a)); physical NN boson current',
                correlation_definition='C_ab(R)=sum_i <{j_a(i),j_b(i+R)}>/(2N); equal time, not Kubo',
                ring_caveat='Uniform vector potential cancels in ring exchange. Finite-q data include NN hopping current only.',
                total_current_variance_per_site=[float(arrays['sjj'][a,a,0,0].real) for a in range(2)],
                total_current_variance_error=[float(arrays['sjj_error_real'][a,a,0,0]) for a in range(2)],
                diamagnetic=arrays['diamag'].tolist(),diamagnetic_error=arrays['diamag_error'].tolist(),
                hopping_square_per_site=float(J*J*arrays['hh'].sum()),
                hopping_square_error=float(J*J*arrays['blocks_hh'].sum(axis=(1,2,3,4)).std(ddof=1)/np.sqrt(20)))
    return report,arrays

