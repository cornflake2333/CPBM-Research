"""Independent exact NN boson current correlators on the physical Hilbert space.

j_(i,a)=i J (b^dag_(i+a) b_i-b^dag_i b_(i+a)).  We form every vector
j_(i,a)|psi> explicitly, including configurations where psi itself vanishes.
The Gram matrix is therefore <j_l j_m>, without any division by psi.
"""

from vmc_paths import PROJECT_ROOT
from pathlib import Path
import argparse
import json
import numpy as np
from projected_parton_exact import BosonBasis, orbitals, optimize_g


def current_vectors(basis, psi, j=1.):
    """Return columns j_(i,a)|psi>, T_(i,a)|psi>, with T=b^dag_j b_i+h.c.."""
    psi=np.asarray(psi,complex)
    lookup={int(mask):index for index,mask in enumerate(basis.masks)}
    shape=(len(psi),len(basis.edges))
    currents=np.zeros(shape,complex);hopping=np.zeros(shape,complex)
    for bond,(i,k) in enumerate(basis.edges):
        i,k=int(i),int(k)
        ni=(basis.masks>>i)&1;nk=(basis.masks>>k)&1
        sources=np.flatnonzero(ni!=nk)
        targets=np.array([lookup[int(basis.masks[s])^(1<<i)^(1<<k)] for s in sources])
        sign=ni[sources]-nk[sources]
        currents[targets,bond]=1j*j*sign*psi[sources]
        hopping[targets,bond]=psi[sources]
    return currents,hopping


def translation_average(basis, matrix):
    """Site-anchored tensor [a,b,Rx,Ry] from an interleaved bond Gram matrix."""
    L=basis.size;N=basis.sites
    out=np.zeros((2,2,L,L),complex)
    for a in range(2):
        for b in range(2):
            for dx in range(L):
                for dy in range(L):
                    for x in range(L):
                        for y in range(L):
                            i=x*L+y;k=((x+dx)%L)*L+(y+dy)%L
                            out[a,b,dx,dy]+=matrix[2*i+a,2*k+b]/N
    return out


def bond_fourier(cc):
    """FFT of <j_(0,a) j_(R,b)> with bond-center phases and negative exponent."""
    L=cc.shape[-1];q=2*np.pi*np.fft.fftfreq(L)
    qx,qy=np.meshgrid(q,q,indexing='ij')
    directions=np.eye(2,dtype=int)
    result=np.fft.fft2(cc,axes=(-2,-1))
    for a in range(2):
        for b in range(2):
            delta=(directions[b]-directions[a])/2
            result[a,b]*=np.exp(-1j*(qx*delta[0]+qy*delta[1]))
    return result


def exact_currents(basis, psi, j=1.,k=10.):
    """Physical exact expectations for a normalized arbitrary variational state."""
    psi=np.asarray(psi,complex);psi=psi/np.linalg.norm(psi)
    vectors,hopping=current_vectors(basis,psi,j)
    gram=vectors.conj().T@vectors
    hopping_gram=hopping.conj().T@hopping
    mean=psi.conj()@vectors
    cc_unsym=translation_average(basis,gram)
    cc=cc_unsym.real
    ccq=bond_fourier(cc)
    connected_gram=gram-np.outer(mean.real,mean.real)
    cc_connected=translation_average(basis,connected_gram).real
    ccq_connected=bond_fourier(cc_connected)
    hopping_cc=translation_average(basis,hopping_gram)
    h_mean=(psi.conj()@hopping).reshape(basis.sites,2).sum(axis=0)/basis.sites
    tau=j*h_mean.real
    hj=-j*hopping.sum(axis=1)
    hk=np.zeros(len(psi),complex)
    np.add.at(hk,basis.kd,k*psi[basis.ks])
    hpsi=hj+hk
    energy=np.vdot(psi,hpsi)
    variance=np.vdot(hpsi,hpsi).real-abs(energy)**2
    current_sum=vectors.reshape(len(psi),basis.sites,2).sum(axis=1)
    total_gram=current_sum.conj().T@current_sum/basis.sites
    total_mean=mean.reshape(basis.sites,2).sum(axis=0).real
    total_connected=total_gram.real-np.outer(total_mean,total_mean)/basis.sites
    occupied=((basis.masks[:,None]>>np.arange(basis.sites))&1)
    weights=abs(psi)**2
    contacts=[]
    for a in range(2):
        edges=basis.edges[a::2]
        contacts.append(float(j*j*np.sum(weights[:,None]*(occupied[:,edges[:,0]]+occupied[:,edges[:,1]]
                                        -2*occupied[:,edges[:,0]]*occupied[:,edges[:,1]]))/basis.sites))
    arrays=dict(cc=cc,cc_unsym=cc_unsym,ccq=ccq,cc_connected=cc_connected,ccq_connected=ccq_connected,
                current_mean_bonds=mean.reshape(basis.sites,2),
                current_gram=gram,hopping_cc=hopping_cc,tau=tau,total_gram=total_gram,
                total_connected=total_connected,
                contact_identity=np.array(contacts),q=2*np.pi*np.fft.fftfreq(basis.size))
    summary=dict(size=basis.size,sites=basis.sites,j=float(j),k=float(k),
                 energy_per_site=float(energy.real/basis.sites),energy_imag=float(energy.imag),
                 h_variance=float(variance),h_variance_per_site=float(variance/basis.sites),
                 h_j_norm_squared=float(np.vdot(hj,hj).real),
                 current_means_max_abs=float(np.max(abs(mean))),
                 current_mean_total_per_site=[float(x.real) for x in mean.reshape(basis.sites,2).mean(axis=0)],
                 diamagnetic_tau=tau.tolist(),
                 total_current_variance_per_site=total_gram.real.tolist(),
                 total_current_connected_per_site=total_connected.tolist(),
                 current_contact=np.array(contacts).tolist(),
                 min_current_gram_eigenvalue=float(np.linalg.eigvalsh(gram).min()),
                 contact_identity_error=float(max(abs(cc[a,a,0,0]-contacts[a]) for a in range(2))),
                 fourier_q0_error=float(np.max(abs(ccq[:,:,0,0]-total_gram.real))),
                 max_imaginary_unsym=float(np.max(abs(cc_unsym.imag))))
    return arrays,summary


def run(output=None,g=-.105,j=1.,k=10.):
    root=PROJECT_ROOT
    output=Path(output) if output else root/'output/projected-currents'
    output.mkdir(parents=True,exist_ok=True)
    basis=BosonBasis(4);records=[]
    for label,r in [('strong',.03),('finite_reference',.33),('strong_opt4',.03)]:
        u1,u2,metadata=orbitals(4,r)
        pure,_=basis.amplitudes(u1,u2)
        actual_g=g
        if label=='strong_opt4':actual_g=optimize_g(basis.compress(pure),j,k)['g']
        weights=np.exp(-actual_g*basis.q);psi=pure*weights;psi/=np.linalg.norm(psi)
        arrays,summary=exact_currents(basis,psi,j,k)
        summary.update(label=label,r=r,g=actual_g,orbitals=metadata)
        np.savez_compressed(output/f'exact4_{label}.npz',**arrays)
        (output/f'exact4_{label}.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        records.append(summary)
        print(json.dumps(summary),flush=True)
    (output/'exact4_summary.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    return records


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--g',type=float,default=-.105)
    parser.add_argument('--j',type=float,default=1.)
    parser.add_argument('--k',type=float,default=10.)
    args=parser.parse_args();run(args.output,args.g,args.j,args.k)
