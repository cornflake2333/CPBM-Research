"""Independent physical-boson correlations by exact configuration enumeration.

Delta_a(i)=b_i b_{i+e_a}; pair[a,b,dx,dy]=N^-1 sum_i
<Delta_a^dag(i) Delta_b(i+R)>. Shared sites are handled by annihilation
followed by creation, with hard-core algebra and no fermionic signs.
The normalized s/d channels use Delta_s,d=(Delta_x +/- Delta_y)/sqrt(2).
"""

from vmc_paths import PROJECT_ROOT
from pathlib import Path
import json
import sys

ROOT=PROJECT_ROOT
if (ROOT/'tmp/plot-libs').exists():
    sys.path.insert(0,str(ROOT/'tmp/plot-libs'))
import numpy as np
from projected_parton_exact import BosonBasis,orbitals


def boson_matrix_element(basis,psi,annihilate,create,lookup=None):
    """Return <psi| prod_create b^dag prod_annihilate b |psi>.

    Sites inside a creation/annihilation product commute. Repeated sites
    inside one product vanish by the hard-core constraint. A shared site
    between the two products is permitted and gives the correct number
    operator factor. psi may be complex and need not be normalized.
    """
    annihilate=tuple(map(int,annihilate));create=tuple(map(int,create))
    if len(set(annihilate))!=len(annihilate) or len(set(create))!=len(create):
        return 0j
    if len(annihilate)!=len(create):
        return 0j  # This basis has fixed particle number.
    if lookup is None:
        lookup=np.full(1<<basis.sites,-1,dtype=np.int64)
        lookup[basis.masks]=np.arange(len(basis.masks))
    amask=sum(1<<i for i in annihilate)
    cmask=sum(1<<i for i in create)
    allowed=(basis.masks&amask)==amask
    intermediate=basis.masks^amask
    allowed&=(intermediate&cmask)==0
    source=np.flatnonzero(allowed)
    if not len(source):return 0j
    destination=lookup[intermediate[source]|cmask]
    if np.any(destination<0):raise RuntimeError('Operator escaped fixed-number basis')
    return np.vdot(psi[destination],psi[source])


def exact_correlations(basis,psi):
    """Return physical correlations plus FFT spectra for any basis wavefunction.

    Density correlation is <(n_i-nu)(n_j-nu)>, where nu=M/N, so it
    retains a density-order Bragg contribution. An additional locally
    connected version subtracts <n_i><n_j> if the state is nonuniform.
    FFT convention is sum_R exp(-iq.R) C(R), axes=(qx,qy).
    """
    psi=np.asarray(psi,dtype=complex)
    norm=float(np.vdot(psi,psi).real)
    if norm<=0:raise ValueError('Zero-norm wavefunction')
    psi=psi/np.sqrt(norm)
    size=basis.size;n=basis.sites;nu=basis.particles/n
    lookup=np.full(1<<n,-1,dtype=np.int64)
    lookup[basis.masks]=np.arange(len(basis.masks))
    occupancy=((basis.masks[:,None]>>np.arange(n))&1).astype(float)
    weights=abs(psi)**2
    density=weights@occupancy
    nn=occupancy.T@(weights[:,None]*occupancy)
    index=np.arange(n).reshape(size,size)
    nearest=[np.roll(index,-1,axis=0).ravel(),np.roll(index,-1,axis=1).ravel()]
    gb=np.zeros((size,size),complex)
    cn=np.zeros((size,size),float);cn_connected=np.zeros_like(cn)
    pair=np.zeros((2,2,size,size),complex)
    for dx in range(size):
        for dy in range(size):
            shifted=np.roll(index,(-dx,-dy),axis=(0,1)).ravel()
            cn[dx,dy]=np.mean(nn[np.arange(n),shifted]-nu*density-nu*density[shifted]+nu*nu)
            cn_connected[dx,dy]=np.mean(nn[np.arange(n),shifted]-density*density[shifted])
            for i,j in enumerate(shifted):
                gb[dx,dy]+=boson_matrix_element(basis,psi,(j,),(i,),lookup)/n
                for a in range(2):
                    for b in range(2):
                        pair[a,b,dx,dy]+=boson_matrix_element(basis,psi,
                            (j,nearest[b][j]),(i,nearest[a][i]),lookup)/n
    nb=np.fft.fft2(gb)
    sn=np.fft.fft2(cn)
    pair_fft=np.fft.fft2(pair,axes=(-2,-1))
    pair_s=(pair_fft[0,0]+pair_fft[1,1]+pair_fft[0,1]+pair_fft[1,0])/2
    pair_d=(pair_fft[0,0]+pair_fft[1,1]-pair_fft[0,1]-pair_fft[1,0])/2
    return dict(rho=density.reshape(size,size),gb=gb,cn=cn,cn_local_connected=cn_connected,
                pair=pair,nb=nb,sn=sn,pair_fft=pair_fft,pair_s=pair_s,pair_d=pair_d,
                q=2*np.pi*np.arange(size)/size)


def check_invariants(arrays,nu=.5,tol=2e-10):
    size=arrays['gb'].shape[0];n=size*size
    inverse=(-np.arange(size))%size
    gb=arrays['gb'];pair=arrays['pair'];nb=arrays['nb'];sn=arrays['sn']
    checks=dict(gb_equal_site_error=float(abs(gb[0,0]-nu)),
                density_equal_site_error=float(abs(arrays['cn'][0,0]-nu*(1-nu))),
                fixed_number_structure_zero_error=float(abs(sn[0,0])),
                nb_mean_error=float(abs(nb.mean()-nu)),
                density_spectrum_mean_error=float(abs(sn.mean()-nu*(1-nu))),
                gb_hermiticity_error=float(np.max(abs(gb-gb[np.ix_(inverse,inverse)].conj()))),
                pair_hermiticity_error=float(np.max(abs(pair-pair.transpose(1,0,2,3)[:,:,inverse[:,None],inverse[None,:]].conj()))),
                nb_min=float(nb.real.min()),sn_min=float(sn.real.min()),
                largest_spectrum_imaginary=float(max(np.max(abs(nb.imag)),np.max(abs(sn.imag)),
                    np.max(abs(arrays['pair_s'].imag)),np.max(abs(arrays['pair_d'].imag)))))
    eig=np.linalg.eigvalsh(np.moveaxis(arrays['pair_fft'],(0,1),(-2,-1)))
    checks['minimum_pair_spectrum_eigenvalue']=float(eig.min())
    checks['cdw_m2']=float(sn[size//2,size//2].real/n)
    for key,value in checks.items():
        if key.endswith('_error') or key=='largest_spectrum_imaginary':
            if value>tol:raise AssertionError(f'{key}: {value}')
    if min(checks['nb_min'],checks['sn_min'],checks['minimum_pair_spectrum_eigenvalue']) < -tol:
        raise AssertionError('A physical equal-time spectrum is not positive semidefinite')
    return checks


def main():
    basis=BosonBasis(4)
    output=ROOT/'output/projected-correlations'
    output.mkdir(parents=True,exist_ok=True)
    arrays_out={};reports=[]
    for label,r in [('strong',.03),('finite_anisotropy',.33)]:
        u1,u2,meta=orbitals(4,r,(.37,.19))
        psi,projection_norm=basis.amplitudes(u1,u2)
        g=-.11
        exponent=-g*basis.q
        psi*=np.exp(exponent-exponent.max())
        psi/=np.linalg.norm(psi)
        arrays=exact_correlations(basis,psi)
        checks=check_invariants(arrays)
        offsite=arrays['gb'].copy();offsite[0,0]=0
        pair_offsite=arrays['pair'].copy()
        pair_offsite[0,0,0,0]=0;pair_offsite[1,1,0,0]=0
        ring=basis.direct_energy(psi,j=0,k=1).real
        checks.update(max_offsite_gb=float(abs(offsite).max()),
                      max_noncontact_pair=float(abs(pair_offsite).max()),
                      pair_s_range=[float(arrays['pair_s'].real.min()),float(arrays['pair_s'].real.max())],
                      pair_d_range=[float(arrays['pair_d'].real.min()),float(arrays['pair_d'].real.max())],
                      nb_range=[float(arrays['nb'].real.min()),float(arrays['nb'].real.max())],
                      pair_contact_x=float(arrays['pair'][0,0,0,0].real),
                      pair_contact_y=float(arrays['pair'][1,1,0,0].real),e_k=ring)
        reports.append(dict(label=label,r=r,g=g,projection_norm=projection_norm,metadata=meta,checks=checks))
        arrays_out.update({f'{label}_{key}':value for key,value in arrays.items()})
        print(json.dumps(reports[-1]),flush=True)
    np.savez_compressed(output/'exact4_correlations.npz',**arrays_out)
    report=dict(size=4,bosons=8,pair_definition='Delta_a(i)=b_i*b_(i+e_a)',
                channel_normalization='Delta_s,d=(Delta_x +/- Delta_y)/sqrt(2)',
                density_definition='<(n_i-nu)(n_(i+R)-nu)>, nu=1/2',
                fourier='sum_R exp(-i q.R) C(R); array axes qx,qy',states=reports)
    (output/'exact4_correlations.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
