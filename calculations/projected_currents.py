"""Equal-time *physical boson* hopping currents in projected parton states.

j_a(i)=i*J*(b^dag_(i+e_a)*b_i-b^dag_i*b_(i+e_a)). Units a=hbar=charge=1.
The measured tensor is C_ab(R)=1/(2N) sum_i <{j_a(i),j_b(i+R)}>.
It is an equal-time correlator, not the zero-frequency Kubo susceptibility.

Both hops are applied before taking the projected wavefunction ratio. This
retains excursions through zero-amplitude configurations, essential in the
row/column constrained shell. Multiplying two local current estimators would
incorrectly discard these excursions. At finite q the tensor includes only
the NN hopping current. Ring exchange has zero uniform current, but specifying
its local gauging is required for a full finite-q continuity current.
"""
from functools import lru_cache
import numpy as np
from projected_correlations import _move_values, _array_summary


@lru_cache(maxsize=None)
def geometry(L):
    N=L*L
    site=np.arange(N).reshape(L,L)
    plus=np.stack([np.roll(site,-1,axis=a).ravel() for a in range(2)])
    aa,bb,rr,ii=np.indices((2,2,N,N))
    dx,dy=np.divmod(rr,L)
    xx,yy=np.divmod(ii,L)
    jj=((xx+dx)%L)*L+(yy+dy)%L
    i=ii.ravel();j=jj.ravel();a=aa.ravel();b=bb.ravel()
    l0=i;l1=plus[a,i];m0=j;m1=plus[b,j]
    same=(l0==m0)&(l1==m1)
    shared=(l0==m0)|(l0==m1)|(l1==m0)|(l1==m1)
    return dict(plus=plus,l0=l0,l1=l1,m0=m0,m1=m1,
                same=same,shared=shared)


def observe(state,J=1.,with_hopping=False):
    """Direct sequential hard-core operator estimator at one configuration.

    Returns symmetrized real current tensor cc, directional diamagnetic
    expectations, and mean currents. hh (optional) is the analogous tensor
    of h_a(i)=b^dag_(i+e_a)*b_i+h.c.; it diagnoses hopping outside support.
    All rank-0/1/2 final configurations are evaluated without dividing by an
    intermediate wavefunction amplitude.
    """
    L,N=state.L,state.N;geo=geometry(L);occ=state.occ
    l0,l1,m0,m1=[geo[key] for key in ('l0','l1','m0','m1')]
    active=occ[l0]!=occ[l1]
    s1=np.where(occ[l0],l0,l1);t1=np.where(occ[l0],l1,l0)
    n0=occ[m0]+(m0==t1).astype(int)-(m0==s1).astype(int)
    n1=occ[m1]+(m1==t1).astype(int)-(m1==s1).astype(int)
    active&=n0!=n1
    s2=np.where(n0,m0,m1);t2=np.where(n0,m1,m0)
    transformed=np.stack([state.U[a]@state.inverse[a] for a in range(2)])
    ratios=np.zeros(len(l0),complex)
    zero=active&geo['same'];ratios[zero]=1.
    one=active&geo['shared']&(~geo['same'])
    removed=np.where(s1==t2,s2,s1)
    added=np.where(t1==s2,t2,t1)
    ratios[one]=_move_values(state,removed[one,None],added[one,None],transformed)
    two=active&(~geo['shared'])
    ratios[two]=_move_values(state,np.column_stack([s1[two],s2[two]]),
                            np.column_stack([t1[two],t2[two]]),transformed)
    # Bra action: conjugate of the two forward hopping coefficients. Their
    # product is real: (i*sigma1)*(i*sigma2)=-sigma1*sigma2.
    sign1=2*occ[l0]-1;sign2=2*n0-1
    current=(-J*J*sign1*sign2*ratios).reshape(2,2,N,N).mean(axis=-1).reshape(2,2,L,L)
    inv=(-np.arange(L))%L
    current=(current+current.swapaxes(0,1)[:,:,inv[:,None],inv[None,:]])/2
    result=dict(cc=current.real)
    if with_hopping:
        hh=ratios.reshape(2,2,N,N).mean(axis=-1).reshape(2,2,L,L)
        result['hh']=((hh+hh.swapaxes(0,1)[:,:,inv[:,None],inv[None,:]])/2).real
    diamag=np.zeros(2,complex);mean_current=np.zeros(2,complex)
    for a in range(2):
        first=np.arange(N);second=geo['plus'][a]
        valid=occ[first]!=occ[second]
        old=np.where(occ[first[valid]],first[valid],second[valid])
        new=np.where(occ[first[valid]],second[valid],first[valid])
        ratio=_move_values(state,old[:,None],new[:,None],transformed)
        diamag[a]=J*ratio.sum()/N
        mean_current[a]=(-1j*J*(2*occ[first[valid]]-1)*ratio).sum()/N
    result.update(diamag=diamag.real,mean_current=mean_current.real)
    return result


def fourier(cc):
    """Bond-centered covariance, axes (...,a,b,qx,qy), FFT negative sign.

    C_ab is real symmetrized in operators. Its Hermitian q tensor has
    C_ab(q)=C_ba(q)^*. q=2*pi*fftfreq(L), and q=0 gives <{I_a,I_b}>/(2N).
    The diagonal spectra are the same with site or bond-center anchors.
    """
    L=cc.shape[-1];q=2*np.pi*np.fft.fftfreq(L)
    qx,qy=np.meshgrid(q,q,indexing='ij')
    offsets=np.array([[.5,0.],[0.,.5]])
    result=np.fft.fft2(cc,axes=(-2,-1))
    for a in range(2):
        for b in range(2):
            d=offsets[b]-offsets[a]
            result[...,a,b,:,:]*=np.exp(-1j*(qx*d[0]+qy*d[1]))
    return result


def longitudinal_transverse(spectrum):
    """Project hopping-current tensor on lattice ktilde=(2sin(qx/2),...).

    Set q=0 to NaN: longitudinal/transverse directions are undefined there.
    This is a kinematic decomposition, not a superfluid-stiffness formula.
    """
    L=spectrum.shape[-1];q=2*np.pi*np.fft.fftfreq(L)
    x,y=np.meshgrid(2*np.sin(q/2),2*np.sin(q/2),indexing='ij')
    norm=x*x+y*y;safe=np.where(norm==0,1.,norm)
    cross=(spectrum[...,0,1,:,:]+spectrum[...,1,0,:,:]).real
    xx=spectrum[...,0,0,:,:].real;yy=spectrum[...,1,1,:,:].real
    long=(x*x*xx+y*y*yy+x*y*cross)/safe
    trans=(y*y*xx+x*x*yy-x*y*cross)/safe
    long[...,0,0]=np.nan;trans[...,0,0]=np.nan
    return long,trans


def sample_currents(*args, **kwargs):
    """Compatibility entry; sampling loop is in sampling/current_sampler.py."""
    from current_sampler import sample_currents as implementation
    return implementation(*args, **kwargs)
