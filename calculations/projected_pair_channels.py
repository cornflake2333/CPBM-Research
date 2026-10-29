"""Projected pair correlations for axial and diagonal hard-core boson bonds.

Bond order is x, y, x+y, x-y.  B_a(i)=b_i b_(i+v_a), and
P_ab(R)=N^-1 sum_i <B_a^dag(i) B_b(i+R)>.  The complex determinant-ratio
convention and Hermitian symmetrization match projected_correlations.py.

The four measured fields use both orientations of each undirected bond:
  s_nn / d_x2y2 = [B_x(i)+B_x(i-x) +/- B_y(i)+/-B_y(i-y)]/2,
  s_diag / d_xy = [B_+(i)+B_+(i-x-y) +/- B_-(i)+/-B_-(i-x+y)]/2.
The spelling above means plus for the s field, minus on both y/- terms for d.
They are boson-pair fields; a fixed-number wavefunction has zero mean field.
Only correlations are measured, without an unprojected Wick approximation.
"""
from functools import lru_cache
import numpy as np
from projected_correlations import _move_values

BOND_VECTORS=((1,0),(0,1),(1,1),(1,-1))
CHANNEL_NAMES=('s_nn','d_x2y2','s_diag','d_xy')


@lru_cache(maxsize=None)
def geometry(L):
    """Cached source/target endpoints and overlap classes, independent of state."""
    if L<4 or L%2:
        raise ValueError('Use even L>=4')
    x,y=np.indices((L,L));x=x.ravel();y=y.ravel();N=L*L
    rx,ry=np.indices((L,L));rx=rx.ravel();ry=ry.ravel()
    targets=((x[None,:]+rx[:,None])%L)*L+(y[None,:]+ry[:,None])%L
    source=np.broadcast_to(np.arange(N),(N,N)).ravel()
    target=targets.ravel()
    plus=np.stack([((x+dx)%L)*L+(y+dy)%L for dx,dy in BOND_VECTORS])
    slots={}
    for a in range(4):
        old0=source;old1=plus[a,source]
        for b in range(4):
            new0=target;new1=plus[b,target]
            shared0=(old0==new0)|(old0==new1)
            shared1=(old1==new0)|(old1==new1)
            rank=(~shared0).astype(np.int8)+(~shared1).astype(np.int8)
            one=np.flatnonzero(rank==1);two=np.flatnonzero(rank==2)
            slots[a,b]=dict(
                diag=np.flatnonzero(rank==0),one=one,two=two,
                one_remove=np.where(~shared0,old0,old1)[one],
                one_add=np.where((new0!=old0)&(new0!=old1),new0,new1)[one],
                two_remove=np.column_stack([old0[two],old1[two]]),
                two_add=np.column_stack([new0[two],new1[two]]))
    return dict(source=source,target=target,targets=targets,plus=plus,slots=slots,
                negative=(-np.arange(L))%L)


def observe(state,full_matrix=True):
    """Return Hermitian P_ab(R), shape (4,4,L,L), from one configuration.

    If full_matrix=False, measure only the (x,y) and (x+y,x-y) 2x2 diagonal
    blocks.  Cross-group entries are zeros denoting *unmeasured* quantities,
    not a claim that their physical expectations vanish.  Those blocks suffice
    for all four CHANNEL_NAMES fields; mixed axial/diagonal fields need full.
    """
    L,N=state.L,state.N;geo=geometry(L)
    source=geo['source'];plus=geo['plus'];occ=state.occ
    transformed=np.stack([state.U[a]@state.inverse[a] for a in range(2)])
    pair=np.zeros((4,4,L,L),complex)
    for a in range(4):
        occupied=(occ[source]==1)&(occ[plus[a,source]]==1)
        for b in range(4):
            if not full_matrix and a//2!=b//2:
                continue
            slot=geo['slots'][a,b];vals=np.zeros(N*N,complex)
            ids=slot['diag'];ids=ids[occupied[ids]]
            vals[ids]=1.
            ids=slot['one'];valid=occupied[ids]&(occ[slot['one_add']]==0)
            vals[ids[valid]]=_move_values(state,slot['one_remove'][valid,None],
                                         slot['one_add'][valid,None],transformed)
            ids=slot['two'];added=slot['two_add']
            valid=occupied[ids]&(occ[added[:,0]]==0)&(occ[added[:,1]]==0)
            vals[ids[valid]]=_move_values(state,slot['two_remove'][valid],
                                         added[valid],transformed)
            pair[a,b]=vals.reshape(N,N).mean(axis=1).reshape(L,L)
    inv=geo['negative']
    return (pair+pair[:,:,inv[:,None],inv[None,:]].swapaxes(0,1).conj())/2


@lru_cache(maxsize=None)
def channel_factors(L):
    """Fourier coefficients f_a(q) for fields sum_a f_a(q) B_a(q)."""
    q=2*np.pi*np.fft.fftfreq(L);qx,qy=np.meshgrid(q,q,indexing='ij')
    phases=np.stack([(1+np.exp(-1j*(dx*qx+dy*qy)))/2
                     for dx,dy in BOND_VECTORS])
    factors=np.zeros((4,4,L,L),complex)
    factors[0,0]=factors[1,0]=phases[0]
    factors[0,1]=phases[1];factors[1,1]=-phases[1]
    factors[2,2]=factors[3,2]=phases[2]
    factors[2,3]=phases[3];factors[3,3]=-phases[3]
    return factors


def fourier(pair):
    """Transform pair[...,4,4,L,L] into the four momentum densities.

    Returns the complex bond-space ``matrix``, each full field ``name``, its
    ``contact_name`` baseline and ``nonlocal_name`` difference.  Contact retains
    only identical bond terms P_aa(R=0).  Its momentum dependence is a known
    shape factor; the nonlocal difference may have either sign.  Leading
    dimensions (e.g. independent sampling blocks) are preserved.
    """
    pair=np.asarray(pair)
    if pair.ndim<4 or pair.shape[-4:-2]!=(4,4) or pair.shape[-2]!=pair.shape[-1]:
        raise ValueError('Expected pair[...,4,4,L,L]')
    L=pair.shape[-1];pq=np.fft.fft2(pair,axes=(-2,-1))
    contact=np.zeros_like(pair)
    for a in range(4):
        contact[...,a,a,0,0]=pair[...,a,a,0,0]
    cq=np.fft.fft2(contact,axes=(-2,-1))
    results=dict(matrix=pq)
    for name,f in zip(CHANNEL_NAMES,channel_factors(L)):
        full=np.einsum('axy,...abxy,bxy->...xy',f.conj(),pq,f).real
        baseline=np.einsum('axy,...abxy,bxy->...xy',f.conj(),cq,f).real
        results[name]=full
        results['contact_'+name]=baseline
        results['nonlocal_'+name]=full-baseline
    return results
