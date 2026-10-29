"""Exact Gutzwiller-projected parton wavefunctions on small square lattices.

Physical amplitudes are det(U1[X])*det(U2[X])*exp(-g*Q_nn(X)).
No Gaussian Wick factorization is used. Uniform parent spectra have shell
plateaus as a function of r; all distinct occupied-orbital sets are enumerated.
"""

from vmc_paths import PROJECT_ROOT
from pathlib import Path
import argparse
import itertools
import json
import math
import sys

ROOT=PROJECT_ROOT
if (ROOT/'tmp/plot-libs').exists():sys.path.insert(0,str(ROOT/'tmp/plot-libs'))
import numpy as np


def orbitals(size,r,twist=(.37,.19),particles=None):
    """Opposite internal twists in units of pi, zero physical boson twist.

    t_parallel=1 fixes an unphysical parent energy scale; r is variational.
    Species 1 has (tx,ty)=(1,r), species 2=(r,1).
    """
    sites=size*size;particles=sites//2 if particles is None else int(particles)
    if not 0<particles<sites:raise ValueError('Need 0<particles<L**2')
    if size%2 or not 0<=r<=1:raise ValueError('Even size and 0<=r<=1 required')
    positions=np.array(list(itertools.product(range(size),repeat=2)))
    indices=positions.copy();us=[];occupied=[];gaps=[]
    for alpha in (0,1):
        sign=1 if alpha==0 else -1
        k=(2*np.pi*indices+sign*np.pi*np.array(twist))/size
        energy=-2*(np.cos(k[:,0])+r*np.cos(k[:,1])) if alpha==0 else -2*(r*np.cos(k[:,0])+np.cos(k[:,1]))
        order=np.argsort(energy,kind='stable');occ=np.sort(order[:particles])
        us.append(np.exp(1j*positions@k[occ].T)/np.sqrt(sites))
        occupied.append(occ.tolist());gaps.append(float(energy[order[particles]]-energy[order[particles-1]]))
    return us[0],us[1],dict(size=size,r=float(r),particles=particles,filling=particles/sites,twist_pi=list(twist),occupied=occupied,parent_gaps=gaps)


def modulated_orbitals(size,r,delta,twist=(.37,.19)):
    """Period-two real bond modulation, with opposite boundary-only twists."""
    sites=size*size;us=[]
    for alpha in (0,1):
        h=np.zeros((sites,sites),complex)
        base=(1,r) if alpha==0 else (r,1)
        theta=np.pi*np.array(twist)*(1 if alpha==0 else -1)
        for x,y in itertools.product(range(size),repeat=2):
            i=x*size+y
            for ell,(dx,dy) in enumerate(((1,0),(0,1))):
                coordinate=x if ell==0 else y
                j=((x+dx)%size)*size+(y+dy)%size
                hopping=base[ell]*(1+delta*(-1.)**coordinate)
                phase=np.exp(1j*theta[ell]) if coordinate==size-1 else 1.
                h[i,j]-=hopping*phase;h[j,i]-=hopping*np.conj(phase)
        _,u=np.linalg.eigh(h);us.append(u[:,:sites//2])
    return us[0],us[1]


def closed_shell_intervals(size,twist=(.37,.19)):
    indices=np.array(list(itertools.product(range(size),repeat=2)))
    bounds=[0.,1.]
    for alpha in (0,1):
        sign=1 if alpha==0 else -1
        k=(2*np.pi*indices+sign*np.pi*np.array(twist))/size
        cx,cy=np.cos(k[:,0]),np.cos(k[:,1])
        a,b=(cx,cy) if alpha==0 else (cy,cx)
        da=a[:,None]-a[None,:];db=b[:,None]-b[None,:]
        mask=abs(db)>1e-12
        roots=-da[mask]/db[mask]
        bounds.extend(roots[(roots>1e-10)&(roots<1-1e-10)].tolist())
    bounds=np.unique(np.round(bounds,13));records=[];last=None
    for lo,hi in zip(bounds[:-1],bounds[1:]):
        if hi-lo<1e-11:continue
        r=float((lo+hi)/2)
        _,_,meta=orbitals(size,r,twist)
        key=tuple(tuple(x) for x in meta['occupied'])
        if key==last:
            records[-1]['r_hi']=float(hi)
        else:
            records.append(dict(r_lo=float(lo),r_hi=float(hi),r=r,metadata=meta))
            last=key
    for record in records:
        record['r']=(record['r_lo']+record['r_hi'])/2
        record['metadata']=orbitals(size,record['r'],twist)[2]
    return records


class BosonBasis:
    def __init__(self,size=4):
        self.size=size;self.sites=size*size;self.particles=self.sites//2
        count=math.comb(self.sites,self.particles)
        if count>300000:raise ValueError('Exact enumeration limited to <=300000 configurations')
        self.configs=np.array(list(itertools.combinations(range(self.sites),self.particles)),dtype=int)
        self.masks=np.array([sum(1<<int(i) for i in row) for row in self.configs],dtype=np.int64)
        lookup={int(s):i for i,s in enumerate(self.masks)}
        x,y=np.indices((size,size));idx=x*size+y
        self.edges=np.array([(int(idx[a,b]),int(idx[(a+da)%size,(b+db)%size]))
                            for a,b in itertools.product(range(size),repeat=2) for da,db in [(1,0),(0,1)]])
        self.plaquettes=np.array([(int(idx[a,b]),int(idx[(a+1)%size,b]),int(idx[(a+1)%size,(b+1)%size]),int(idx[a,(b+1)%size]))
                                  for a,b in itertools.product(range(size),repeat=2)])
        occupied=((self.masks[:,None]>>np.arange(self.sites))&1)
        self.q=(occupied[:,self.edges[:,0]]*occupied[:,self.edges[:,1]]).sum(axis=1).astype(int)
        eta=(-1.)**(x+y).ravel()
        self.cdw_squared=(occupied@eta)**2/self.sites**2
        js=[];jd=[];ks=[];kd=[]
        for col,s in enumerate(self.masks):
            s=int(s)
            for i,j in self.edges:
                if ((s>>int(i))&1)!=((s>>int(j))&1):
                    js.append(col);jd.append(lookup[s^(1<<int(i))^(1<<int(j))])
            for plaq in self.plaquettes:
                bits=[(s>>int(i))&1 for i in plaq]
                if bits in ([1,0,1,0],[0,1,0,1]):
                    ks.append(col);kd.append(lookup[s^sum(1<<int(i) for i in plaq)])
        self.js=np.array(js);self.jd=np.array(jd);self.ks=np.array(ks);self.kd=np.array(kd)

    def amplitudes(self,u1,u2):
        psi=np.linalg.det(u1[self.configs])*np.linalg.det(u2[self.configs])
        norm=float(np.vdot(psi,psi).real)
        if norm<1e-25:raise ValueError('Projected state has vanishing norm')
        return psi/np.sqrt(norm),norm

    def compress(self,psi):
        weights=abs(psi)**2
        qmax=int(self.q.max());smax=2*qmax
        bins=np.bincount(self.q,weights=weights,minlength=qmax+1)
        cdw=np.bincount(self.q,weights=weights*self.cdw_squared,minlength=qmax+1)
        hj=np.bincount(self.q[self.js]+self.q[self.jd],
                       weights=-(psi[self.js].conj()*psi[self.jd]).real,minlength=smax+1)
        hk=np.bincount(self.q[self.ks]+self.q[self.kd],
                       weights=(psi[self.ks].conj()*psi[self.kd]).real,minlength=smax+1)
        return dict(q_weights=bins,q_cdw=cdw,j_moments=hj,k_moments=hk,sites=self.sites)

    def direct_energy(self,psi,j=1.,k=1.,g=0.):
        ex=-g*self.q;psi=psi*np.exp(ex-ex.max());norm=np.vdot(psi,psi).real
        ej=-(psi[self.js].conj()*psi[self.jd]).sum()/norm/self.sites
        ek=(psi[self.ks].conj()*psi[self.kd]).sum()/norm/self.sites
        return complex(j*ej+k*ek)


def evaluate(moments,g,j=1.,k=1.):
    q=np.arange(len(moments['q_weights']));s=np.arange(len(moments['j_moments']))
    offset=float(np.max(-2*g*q[moments['q_weights']>1e-30]))
    w=moments['q_weights']*np.exp(-2*g*q-offset)
    norm=w.sum();p=np.exp(-g*s-offset)
    ej=float(np.dot(moments['j_moments'],p)/norm/moments['sites'])
    ek=float(np.dot(moments['k_moments'],p)/norm/moments['sites'])
    energy=j*ej+k*ek
    numerator=(j*moments['j_moments']+k*moments['k_moments'])
    derivative=float(-np.dot(s*numerator,p)/norm/moments['sites']+2*energy*np.dot(q,w)/norm)
    cdw=float(np.dot(moments['q_cdw'],np.exp(-2*g*q-offset))/norm)
    return dict(g=float(g),energy=float(energy),e_j_over_j=ej,e_k_over_k=ek,
                derivative_g=derivative,cdw_m2=cdw)


def optimize_g(moments,j,k,lo=-2.,hi=4.):
    grid=np.linspace(lo,hi,97)
    values=[evaluate(moments,g,j,k) for g in grid]
    n=int(np.argmin([v['energy'] for v in values]))
    if n in (0,len(grid)-1):
        best=values[n];best.update(g_boundary=True,curvature_g=None)
        return best
    a,b=grid[n-1],grid[n+1];gold=(np.sqrt(5)-1)/2
    da=evaluate(moments,a,j,k)['derivative_g'];db=evaluate(moments,b,j,k)['derivative_g']
    if da<0<db:
        for _ in range(70):
            middle=(a+b)/2
            if evaluate(moments,middle,j,k)['derivative_g']<0:a=middle
            else:b=middle
        best=evaluate(moments,(a+b)/2,j,k)
        step=1e-4
        best.update(g_boundary=False,curvature_g=(evaluate(moments,best['g']+step,j,k)['energy']-2*best['energy']
                                                      +evaluate(moments,best['g']-step,j,k)['energy'])/step**2)
        return best
    c=b-gold*(b-a);d=a+gold*(b-a)
    for _ in range(65):
        if evaluate(moments,c,j,k)['energy']<evaluate(moments,d,j,k)['energy']:
            b=d;d=c;c=b-gold*(b-a)
        else:
            a=c;c=d;d=a+gold*(b-a)
    best=evaluate(moments,(a+b)/2,j,k)
    step=1e-4
    best.update(g_boundary=False,curvature_g=(evaluate(moments,best['g']+step,j,k)['energy']-2*best['energy']
                                                  +evaluate(moments,best['g']-step,j,k)['energy'])/step**2)
    return best


def references(sites,j,k):
    return dict(staggered_dimer=-j/2,signed_dimer=-k/4,plaquette=-k/4,
                phase_dimer=-k/4-j*j/(8*k) if k>=j/2 else -j/2+k/4,
                constant_boson=-j*sites/(sites-1)+k*sites*(sites-2)/(8*(sites-1)*(sites-3)),
                cdw_fock=0.)


def run_exact(size=4,twists=((.37,.19),(.13,.61)),out=None):
    out=Path(out) if out is not None else ROOT/'output/projected-jk'
    out.mkdir(parents=True,exist_ok=True)
    basis=BosonBasis(size);all_records=[];state_moments=[]
    for twist_index,twist in enumerate(twists):
        intervals=closed_shell_intervals(size,twist)
        print('Exact twist',twist,'unique seas',len(intervals),flush=True)
        for n,interval in enumerate(intervals):
            u1,u2,meta=orbitals(size,interval['r'],twist)
            psi,norm=basis.amplitudes(u1,u2);mom=basis.compress(psi)
            sid=f'twist{twist_index}_sea{n}'
            record=dict(state_id=sid,twist_index=twist_index,twist_pi=list(twist),
                        r_lo=interval['r_lo'],r_hi=interval['r_hi'],r=interval['r'],
                        projection_norm=norm,metadata=meta)
            coeff=evaluate(mom,0,1,1)
            record.update(e_j_over_j=coeff['e_j_over_j'],e_k_over_k=coeff['e_k_over_k'],cdw_m2=coeff['cdw_m2'])
            all_records.append(record);state_moments.append(mom)
    optimized=[]
    for k in range(1,11):
        pure=[];jastrow=[]
        for record,mom in zip(all_records,state_moments):
            pure.append(dict(record,**evaluate(mom,0,1,k)))
            jastrow.append(dict(record,**optimize_g(mom,1,k)))
        best_pure=min(pure,key=lambda x:x['energy']);best_j=min(jastrow,key=lambda x:x['energy'])
        ref=references(basis.sites,1,k)
        row=dict(j=1,k=k,best_pure=best_pure,best_jastrow=best_j,references=ref,
                 pure_by_twist=[min((v for v in pure if v['twist_index']==t),key=lambda x:x['energy']) for t in range(len(twists))],
                 jastrow_by_twist=[min((v for v in jastrow if v['twist_index']==t),key=lambda x:x['energy']) for t in range(len(twists))])
        optimized.append(row)
        print('Exact K',k,'pure',best_pure['energy'],'r interval',[best_pure['r_lo'],best_pure['r_hi']],
              'Jastrow',best_j['energy'],'g',best_j['g'],'references',ref,flush=True)
    result=dict(size=size,sites=basis.sites,particles=basis.particles,twists_pi=twists,
                approximation='Exactly projected determinant products; optional NN density Jastrow; complete physical configuration enumeration',
                states=all_records,optimized=optimized)
    (out/'exact_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez_compressed(out/'exact_moments.npz',q_weights=np.stack([v['q_weights'] for v in state_moments]),
                         q_cdw=np.stack([v['q_cdw'] for v in state_moments]),
                         j_moments=np.stack([v['j_moments'] for v in state_moments]),
                         k_moments=np.stack([v['k_moments'] for v in state_moments]))
    # Independent smooth orbital deformation: a tested stability direction,
    # not an unrestricted Hessian over all possible parent Hamiltonians.
    stability=[]
    for delta in (-.03,-.01,0,.01,.03,.05,.1,.25,.5,1.):
        psi,_=basis.amplitudes(*modulated_orbitals(size,.05,delta,twists[0]))
        mom=basis.compress(psi)
        stability.append(dict(delta=delta,pure=evaluate(mom,0,1,10),optimized=optimize_g(mom,1,10)))
    (out/'bond_stability.json').write_text(json.dumps(dict(j=1,k=10,r=.05,size=size,rows=stability),indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--size',type=int,default=4)
    p.add_argument('--output',type=Path,default=ROOT/'output/projected-jk')
    a=p.parse_args();run_exact(a.size,out=a.output)
