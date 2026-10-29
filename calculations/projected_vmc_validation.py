"""Independent exact-enumeration/long-chain validation of projected VMC."""
from pathlib import Path
import json
import numpy as np
import projected_parton_exact as exact
import projected_parton_vmc as vmc


def ess(values):
    a=np.asarray(values,float);a=a-a.mean();n=len(a)
    var=float(np.dot(a,a)/n)
    if var<1e-24:return None
    fft=np.fft.rfft(a,n=2*n)
    corr=np.fft.irfft(fft*fft.conj(),n=2*n)[:n]/np.arange(n,0,-1)
    rho=corr/corr[0];total=0.
    for start in range(1,min(n-1,n//2),2):
        pair=float(rho[start]+rho[start+1])
        if pair<=0:break
        total+=pair
    tau=.5+total
    return dict(tau_integrated_measurements=tau,effective_samples=min(n,n/(2*tau)))


def run():
    out=Path('output/projected-jk');out.mkdir(parents=True,exist_ok=True)
    basis=exact.BosonBasis(4);records=[]
    for r,g in [(.05,0.),(.4,.1)]:
        u,v,meta=exact.orbitals(4,r)
        psi,_=basis.amplitudes(u,v)
        expected=exact.evaluate(basis.compress(psi),g,1,10)
        for seed in (1729,9811):
            result=vmc.sample(u,v,4,g,seed,warmup=512,samples=4096,stride=2,return_samples=True)
            raw=np.asarray(result.pop('raw_samples'))
            np.savez_compressed(out/f'validation_L4_r{r:g}_g{g:g}_seed{seed}.npz',raw_samples=raw)
            z={}
            for name,ref,error in [('e_j',expected['e_j_over_j'],'e_j_error'),
                                   ('e_k',expected['e_k_over_k'],'e_k_error')]:
                z[name]=(result[name]-ref)/result[error] if result[error]>1e-14 else None
            z['m_cdw_squared']=(result['m_cdw_squared']-expected['cdw_m2'])/(result['s_pi_pi_error']/16)
            record=dict(r=r,g=g,metadata=meta,seed=seed,exact=expected,vmc=result,
                        z_scores=z,ess_e_k=ess(raw[:,1]),ess_m_cdw=ess(raw[:,5]))
            records.append(record)
            (out/'vmc_validation.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
            print(json.dumps(dict(r=r,g=g,seed=seed,z_scores=z,acceptance=result['acceptance'],
                                  seconds=result['seconds'],ess_e_k=record['ess_e_k'],
                                  e_j=result['e_j'],e_k=result['e_k'],m2=result['m_cdw_squared'])),flush=True)


if __name__=='__main__':run()
