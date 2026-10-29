"""Determinantal VMC for exactly projected square-lattice two-parton states.

Psi(X)=det(U1[X,:])*det(U2[X,:])*exp(-g*Q_nn(X)), with fixed M
bosons on N=L**2 sites. Site index is x*L+y. Physical bosons have PBC.
The two orbital arrays may include opposite internal boundary twists.
An optional h multiplies the state by exp[h*sum_i(-1)**(x+y)*n_i].

All expectations use the projected configuration probability, never Wick
factorization. Energies returned are coefficients per site: e=J*e_j+K*e_k.
warmup and stride count sweeps, one sweep=N attempted symmetric proposals.
"""

from vmc_paths import PROJECT_ROOT
from pathlib import Path
import time
import sys

ROOT=PROJECT_ROOT
if (ROOT/'tmp/plot-libs').exists():
    sys.path.insert(0,str(ROOT/'tmp/plot-libs'))
import numpy as np


class ProjectedState:
    def __init__(self,U1,U2,L,g=0.,positions=None,seed=1,h=0.):
        if L<4 or L%2:
            raise ValueError('Use even L>=4')
        self.L=L;self.N=L*L;self.g=float(g);self.h=float(h)
        self.U=np.stack([np.asarray(U1,complex),np.asarray(U2,complex)])
        self.M=self.U.shape[-1]
        if not 0<self.M<self.N:raise ValueError('Need 0<M<N occupied orbitals')
        if self.U.shape!=(2,self.N,self.M):
            raise ValueError('Each orbital matrix must have shape (L**2,M)')
        self.rng=np.random.default_rng(seed)
        x,y=np.indices((L,L));index=(x*L+y)
        self.edges=np.column_stack((np.tile(index.ravel(),2),
                                   np.r_[np.roll(index,-1,axis=0).ravel(),np.roll(index,-1,axis=1).ravel()]))
        self.neighbors=np.column_stack([np.roll(index,s,axis=a).ravel() for a,s in [(0,-1),(0,1),(1,-1),(1,1)]])
        self.adjacency=np.zeros((self.N,self.N),dtype=np.int8)
        self.adjacency[np.arange(self.N)[:,None],self.neighbors]=1
        self.plaquettes=np.column_stack([index.ravel(),np.roll(index,-1,axis=0).ravel(),
            np.roll(np.roll(index,-1,axis=0),-1,axis=1).ravel(),np.roll(index,-1,axis=1).ravel()])
        self.stagger=(-1.)**(x+y).ravel()
        if positions is None:
            # Chiral half-filled free-fermion states have nonsingular restriction
            # to a checkerboard sublattice. This also respects chain sectors.
            if self.M==self.N//2:
                positions=np.flatnonzero((x+y).ravel()%2==0)
            else:
                positions=self._initial_positions()
        self.positions=np.array(positions,dtype=int).copy()
        if len(self.positions)!=self.M or len(np.unique(self.positions))!=self.M:
            raise ValueError('Need M distinct occupied sites')
        self.locator=np.full(self.N,-1,int)
        self.locator[self.positions]=np.arange(self.M)
        self.occ=(self.locator>=0).astype(np.int8)
        self.accepted_since_rebuild=0
        self.rebuild()

    def _initial_positions(self):
        """Find nonsingular common support; balanced starts suit quasi-1D seas.

        Initial selection is only thermalization, never part of the proposal
        distribution. Every accepted production move still uses |Psi|**2.
        """
        L=self.L;base,extra=divmod(self.M,L)
        for attempt in range(256):
            if attempt<192:
                occ=np.zeros((L,L),np.int8)
                for x in range(L):occ[x,(x+np.arange(base))%L]=1
                # Extra particles on distinct rows; columns are randomized.
                if extra:
                    rows=self.rng.permutation(L)[:extra]
                    for x in rows:
                        choices=np.flatnonzero(occ[x]==0)
                        occ[x,self.rng.choice(choices)]=1
                occ=occ[self.rng.permutation(L)][:,self.rng.permutation(L)]
                for _ in range(8*L*L):
                    x1,x2=self.rng.choice(L,2,replace=False);y1,y2=self.rng.choice(L,2,replace=False)
                    if occ[x1,y1]==occ[x2,y2] and occ[x1,y2]==occ[x2,y1] and occ[x1,y1]!=occ[x1,y2]:
                        occ[np.ix_([x1,x2],[y1,y2])]=1-occ[np.ix_([x1,x2],[y1,y2])]
                pos=np.flatnonzero(occ.ravel())
            else:
                pos=np.sort(self.rng.choice(self.N,self.M,replace=False))
            smallest=min(float(np.linalg.svd(u[pos],compute_uv=False)[-1]) for u in self.U)
            if smallest>1e-8:return pos
        raise ValueError('Could not find common nonsingular projected support')

    def rebuild(self):
        self.matrices=self.U[:,self.positions,:].copy()
        signs,logs=np.linalg.slogdet(self.matrices)
        if np.any(abs(signs)<.5) or not np.all(np.isfinite(logs)):
            raise ValueError('Configuration has zero projected amplitude')
        self.inverse=np.linalg.inv(self.matrices)
        err=float(np.max(abs(self.matrices@self.inverse-np.eye(self.M))))
        if err>1e-7:
            raise ValueError('Ill-conditioned occupied orbital matrices')
        self.accepted_since_rebuild=0

    def q_nn(self):
        return int(np.sum(self.occ[self.edges[:,0]]*self.occ[self.edges[:,1]]))

    def ratio(self,rows,new_sites):
        """Wavefunction ratio for replacing 1 or 2 labeled occupied rows."""
        rows=np.asarray(rows,int);new_sites=np.asarray(new_sites,int)
        if len(rows) not in (1,2):
            raise ValueError('Only rank-1 and rank-2 updates are supported')
        counts=self.occ[self.neighbors].sum(axis=1)
        old_sites=self.positions[rows]
        delta=float(counts[new_sites].sum()-counts[old_sites].sum())
        changes=np.r_[old_sites,new_sites]
        charges=np.r_[-np.ones(len(rows)),np.ones(len(rows))]
        delta+=.5*float(charges@self.adjacency[np.ix_(changes,changes)]@charges)
        rs=[]
        for alpha in range(2):
            r=self.U[alpha,new_sites]@self.inverse[alpha,:,rows].T
            # Advanced indexing puts the selected column dimension first.
            rs.append(np.linalg.det(r))
        dm=self.stagger[new_sites].sum()-self.stagger[old_sites].sum()
        return rs[0]*rs[1]*np.exp(np.clip(-self.g*delta+self.h*dm,-700,700)),delta,rs

    def try_move(self,rows,new_sites):
        rows=np.asarray(rows,int);new_sites=np.asarray(new_sites,int)
        if np.any(self.occ[new_sites]) or len(np.unique(new_sites))!=len(rows):
            return False
        wave_ratio,delta,ratios=self.ratio(rows,new_sites)
        if not np.isfinite(abs(wave_ratio)):
            raise FloatingPointError('Nonfinite projected ratio')
        prob=min(1.,float(abs(wave_ratio)**2))
        if self.rng.random()>=prob:
            return False
        old_sites=self.positions[rows].copy()
        for alpha in range(2):
            inv=self.inverse[alpha]
            new_rows=self.U[alpha,new_sites]
            columns=inv[:,rows].copy()
            change=new_rows@inv
            change[np.arange(len(rows)),rows]-=1
            small=new_rows@columns
            self.inverse[alpha]=inv-columns@np.linalg.solve(small,change)
        self.positions[rows]=new_sites
        self.locator[old_sites]=-1;self.locator[new_sites]=rows
        self.occ[old_sites]=0;self.occ[new_sites]=1
        self.accepted_since_rebuild+=1
        if self.accepted_since_rebuild>=128 or min(abs(r) for r in ratios)<1e-5:
            self.rebuild()
        return True

    def proposal(self):
        kind=self.rng.random()
        if kind<.45:
            row=int(self.rng.integers(self.M))
            new=int(self.rng.integers(self.N))
            while self.occ[new]:
                new=int(self.rng.integers(self.N))
            return self.try_move([row],[new])
        if kind<.65:
            row=int(self.rng.integers(self.M))
            new=int(self.neighbors[self.positions[row],self.rng.integers(4)])
            if self.occ[new]:return False
            return self.try_move([row],[new])
        # Any axis-aligned rectangle, including large ones. The proposal is
        # symmetric and connects row/column constrained determinant supports.
        x1,x2=self.rng.choice(self.L,2,replace=False)
        y1,y2=self.rng.choice(self.L,2,replace=False)
        corners=np.array([x1*self.L+y1,x2*self.L+y1,x2*self.L+y2,x1*self.L+y2])
        n=self.occ[corners]
        if n[0]==n[2] and n[1]==n[3] and n[0]!=n[1]:
            old=corners[[0,2]] if n[0] else corners[[1,3]]
            new=corners[[1,3]] if n[0] else corners[[0,2]]
            return self.try_move(self.locator[old],new)
        return False

    def local_energy(self):
        from jk_energy import local_energy
        return local_energy(self)


def local_energy(U1,U2,L,positions,g=0.):
    return ProjectedState(U1,U2,L,g,positions).local_energy()


def _stats(raw):
    from vmc_statistics import energy_statistics
    return energy_statistics(raw)


def sample(U1,U2,L,g=0.,seed=1,warmup=128,samples=512,stride=2,return_samples=False,positions=None):
    """Run VMC. Error bars are contiguous-block standard errors, not iid errors.

    Returned m_cdw_squared is S(pi,pi)/N; a finite value on one lattice does
    not by itself establish long-range density order.
    """
    if samples<20 or warmup<0 or stride<1:
        raise ValueError('Use samples>=20,warmup>=0,stride>=1')
    start=time.perf_counter()
    state=ProjectedState(U1,U2,L,g,positions,seed)
    accepted=attempted=0
    for _ in range(warmup*state.N):
        accepted+=state.proposal();attempted+=1
    raw=np.empty((samples,6))
    for index in range(samples):
        for _ in range(stride*state.N):
            accepted+=state.proposal();attempted+=1
        ej,ek,q,spi=state.local_energy()
        raw[index]=[ej.real,ek.real,ej.imag,ek.imag,q,spi]
    result=_stats(raw)
    result['m_cdw_squared']=result['s_pi_pi']/state.N
    result.update(L=L,sites=state.N,bosons=state.M,g=float(g),seed=int(seed),
                  warmup_sweeps=int(warmup),samples=int(samples),stride_sweeps=int(stride),
                  acceptance=float(accepted/attempted),seconds=float(time.perf_counter()-start),
                  final_positions=state.positions.tolist(),
                  sample_columns=['e_j','e_k','imag_e_j','imag_e_k','Q_nn','S_pi_pi'])
    if return_samples:result['raw_samples']=raw.tolist()
    return result
