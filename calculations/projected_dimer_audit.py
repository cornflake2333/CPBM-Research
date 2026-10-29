"""Independent exact Gutzwiller audit for elementary J-K trial states.

This checks only the named reference states; it does not optimize CPBM
orbitals or use an unprojected Wick energy.
"""

from vmc_paths import PROJECT_ROOT
from pathlib import Path
from itertools import combinations
import json
import sys

ROOT=PROJECT_ROOT
if (ROOT/'tmp/plot-libs').exists():
    sys.path.insert(0,str(ROOT/'tmp/plot-libs'))
import numpy as np


def dimer_orbitals(size,kind,alpha):
    n=size*size
    u=np.zeros((n,n//2))
    column=0
    for x in range(size):
        for y in range(size):
            if kind=='crossed_plaquettes':
                active=x%2==0 if alpha==0 else y%2==0
                dx,dy=(1,0) if alpha==0 else (0,1)
                sign=1
            else:
                active=(x+y)%2==0 if kind=='staggered_dimers' else y%2==0
                dx,dy=0,1
                sign=(-1)**x if kind=='signed_dimers' and alpha==0 else 1
            if active:
                i=x*size+y
                j=((x+dx)%size)*size+(y+dy)%size
                u[i,column]=1/np.sqrt(2)
                u[j,column]=sign/np.sqrt(2)
                column+=1
    if column!=n//2 or not np.allclose(u.T@u,np.eye(n//2),atol=1e-13):
        raise RuntimeError('Orbitals are not orthonormal')
    return u


def projected_amplitudes(size,kind):
    n=size*size;m=n//2
    if kind=='constant_superfluid':
        return {sum(1<<i for i in occ):1. for occ in combinations(range(n),m)}
    if kind=='checkerboard_fock':
        return {sum(1<<(x*size+y) for x in range(size) for y in range(size) if (x+y)%2==0):1.}
    u1,u2=[dimer_orbitals(size,kind,a) for a in range(2)]
    amps={}
    for occ in combinations(range(n),m):
        val=np.linalg.det(u1[list(occ)])*np.linalg.det(u2[list(occ)])
        if abs(val)>1e-13:
            amps[sum(1<<i for i in occ)]=float(val)
    return amps


def exact_coefficients(size,amps):
    norm=sum(abs(a)**2 for a in amps.values())
    ej=ek=0j
    for bits,amp in amps.items():
        for x in range(size):
            for y in range(size):
                i=x*size+y
                for dx,dy in ((1,0),(0,1)):
                    j=((x+dx)%size)*size+(y+dy)%size
                    if ((bits>>i)&1)!=((bits>>j)&1):
                        ej-=np.conj(amp)*amps.get(bits^(1<<i)^(1<<j),0.)
                square=(i,((x+1)%size)*size+y,((x+1)%size)*size+(y+1)%size,x*size+(y+1)%size)
                pattern=tuple((bits>>s)&1 for s in square)
                if pattern in ((1,0,1,0),(0,1,0,1)):
                    ek+=np.conj(amp)*amps.get(bits^sum(1<<s for s in square),0.)
    return dict(support=len(amps),norm_before_normalization=float(norm),
                eJ_over_J=float(ej.real/norm/(size*size)),
                eK_over_K=float(ek.real/norm/(size*size)),
                imaginary_error=float(max(abs(ej.imag),abs(ek.imag))))


def audit(size=4):
    if size!=4:
        raise ValueError('This exact audit intentionally uses 4x4 only.')
    result={}
    expected={'staggered_dimers':(-.5,0.),'parallel_dimers':(-.5,.25),
              'signed_dimers':(0.,-.25),'crossed_plaquettes':(0.,-.25),
              'constant_superfluid':(-16/15,224/(8*15*13)),'checkerboard_fock':(0.,0.)}
    for kind in expected:
        r=exact_coefficients(size,projected_amplitudes(size,kind))
        np.testing.assert_allclose([r['eJ_over_J'],r['eK_over_K']],expected[kind],atol=2e-12)
        result[kind]=r
    return dict(size=size,sites=size*size,bosons=size*size//2,states=result)


if __name__=='__main__':
    result=audit()
    out=ROOT/'output/projected-reference-audit'
    out.mkdir(parents=True,exist_ok=True)
    (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
