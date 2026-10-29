"""Density, boson and nearest-bond-pair correlations of projected partons.

G(R)=N^-1 sum_i <b_i^dag b_(i+R)>;
P_ab(R)=N^-1 sum_i <Delta_a^dag(i) Delta_b(i+R)>,
Delta_x(i)=b_i b_(i+x), Delta_y(i)=b_i b_(i+y).
The bond anchor is site i. Fourier transforms use exp(-i q.R), with no
half-bond phase. Delta_s,d=(Delta_x +/- Delta_y)/sqrt(2).

ProjectedState determinant ratios exactly include all spectator/overlap
occupations. This module does not modify the sampling engine.
"""
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=None)
def geometry(L):
    x,y=np.indices((L,L));x=x.ravel();y=y.ravel();N=L*L
    rx,ry=np.indices((L,L));rx=rx.ravel();ry=ry.ravel()
    targets=((x[None,:]+rx[:,None])%L)*L+(y[None,:]+ry[:,None])%L
    source=np.broadcast_to(np.arange(N),(N,N)).ravel()
    plus=np.stack([((x+1)%L)*L+y,x*L+(y+1)%L])
    negative=((-np.arange(L))%L)
    return dict(targets=targets,source=source,target=targets.ravel(),plus=plus,
                negative=negative)


def _move_values(state,removed,added,occupied_inverse):
    """Vectorized projected amplitudes for valid rank-1/rank-2 moves.

    removed and added have shape (moves,rank), contain distinct sites and are
    disjoint. Caller checks occupations. Adjacent pair endpoints require both
    old-old and new-new terms in the Jastrow delta, not just the cross terms.
    """
    removed=np.asarray(removed,int);added=np.asarray(added,int)
    rank=removed.shape[1]
    counts=state.occ[state.neighbors].sum(axis=1)
    delta=counts[added].sum(axis=1)-counts[removed].sum(axis=1)
    if rank==2:
        delta+=state.adjacency[removed[:,0],removed[:,1]]+state.adjacency[added[:,0],added[:,1]]
    for a in range(rank):
        for b in range(rank):
            delta-=state.adjacency[removed[:,a],added[:,b]]
    rows=state.locator[removed]
    ratio=np.ones(len(removed),complex)
    for alpha in range(2):
        matrix=occupied_inverse[alpha]
        if rank==1:
            ratio*=matrix[added[:,0],rows[:,0]]
        else:
            ratio*=matrix[added[:,0],rows[:,0]]*matrix[added[:,1],rows[:,1]]-matrix[added[:,0],rows[:,1]]*matrix[added[:,1],rows[:,0]]
    dm=state.stagger[added].sum(axis=1)-state.stagger[removed].sum(axis=1)
    return ratio*np.exp(np.clip(-state.g*delta+getattr(state,'h',0.)*dm,-700,700))


def observe(state):
    """One exact projected local estimator, averaged over source lattice sites.

    Returns arrays cn[L,L], gb[L,L], pair[2,2,L,L]. Per-observation Hermitian
    symmetrization is unbiased and ensures real diagonal momentum densities.
    """
    L,N=state.L,state.N;geo=geometry(L)
    source,target=geo['source'],geo['target']
    transformed=np.stack([state.U[a]@state.inverse[a] for a in range(2)])
    centered=state.occ-state.M/state.N
    cn=(centered[None,:]*centered[geo['targets']]).mean(axis=1).reshape(L,L)
    values=np.zeros(N*N,complex)
    same=source==target
    values[same]=state.occ[source[same]]
    valid=(~same)&(state.occ[source]==1)&(state.occ[target]==0)
    values[valid]=_move_values(state,source[valid,None],target[valid,None],transformed)
    gb=values.reshape(N,N).mean(axis=1).reshape(L,L)
    pair=np.zeros((2,2,L,L),complex)
    for a in range(2):
        old0=source;old1=geo['plus'][a,source]
        occupied=(state.occ[old0]==1)&(state.occ[old1]==1)
        for b in range(2):
            new0=target;new1=geo['plus'][b,target]
            shared0=(old0==new0)|(old0==new1)
            shared1=(old1==new0)|(old1==new1)
            rank=(~shared0).astype(int)+(~shared1).astype(int)
            vals=np.zeros(N*N,complex)
            diag=occupied&(rank==0)
            vals[diag]=1.
            one=occupied&(rank==1)
            remove=np.where(~shared0,old0,old1)
            add=np.where((new0!=old0)&(new0!=old1),new0,new1)
            one&=state.occ[add]==0
            vals[one]=_move_values(state,remove[one,None],add[one,None],transformed)
            two=occupied&(rank==2)&(state.occ[new0]==0)&(state.occ[new1]==0)
            vals[two]=_move_values(state,np.column_stack([old0[two],old1[two]]),
                                   np.column_stack([new0[two],new1[two]]),transformed)
            pair[a,b]=vals.reshape(N,N).mean(axis=1).reshape(L,L)
    inv=geo['negative']
    gb=(gb+gb[np.ix_(inv,inv)].conj())/2
    reversed_pair=pair[:,:,inv[:,None],inv[None,:]]
    pair=(pair+reversed_pair.swapaxes(0,1).conj())/2
    return dict(cn=cn,gb=gb,pair=pair)


def fourier(cn,gb,pair):
    sn=np.fft.fft2(cn,axes=(-2,-1)).real
    nb=np.fft.fft2(gb,axes=(-2,-1)).real
    pq=np.fft.fft2(pair,axes=(-2,-1))
    px=pq[...,0,0,:,:].real if pq.ndim>4 else pq[0,0].real
    py=pq[...,1,1,:,:].real if pq.ndim>4 else pq[1,1].real
    xy=pq[...,0,1,:,:] if pq.ndim>4 else pq[0,1]
    yx=pq[...,1,0,:,:] if pq.ndim>4 else pq[1,0]
    return dict(sn=sn,nb=nb,pair_x=px,pair_y=py,
                pair_s=(px+py+(xy+yx).real)/2,
                pair_d=(px+py-(xy+yx).real)/2)


def _array_summary(raw):
    blocks=np.stack([x.mean(axis=0) for x in np.array_split(raw,20)])
    mean=raw.mean(axis=0)
    if np.iscomplexobj(raw):
        return mean,dict(error_real=blocks.real.std(axis=0,ddof=1)/np.sqrt(20),
                         error_imag=blocks.imag.std(axis=0,ddof=1)/np.sqrt(20)),blocks
    return mean,dict(error=blocks.std(axis=0,ddof=1)/np.sqrt(20)),blocks


def sample_correlations(*args, **kwargs):
    """Compatibility entry; sampling loop is in sampling/correlation_sampler.py."""
    from correlation_sampler import sample_correlations as implementation
    return implementation(*args, **kwargs)
