"""Bare parton Fermi-sea correlations and projected physical density correlators.

Bare intersite d^dag d is zero when sandwiched by the strict local projector.
The nonzero curves here are explicitly those of the unprojected parent state.
"""

from vmc_paths import PROJECT_ROOT
from pathlib import Path
import json,csv
import projected_parton_exact as exact
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=PROJECT_ROOT
SOURCE=ROOT/'output/vmc12-n06';OUT=ROOT/'output/parton-realspace-n06'
DIRECTIONS={'x':(1,0),'y':(0,1),'diagonal':(1,1)}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    records=json.loads((SOURCE/'summary.json').read_text())
    best=min(records,key=lambda s:s['energy']);L=best['L'];N=L*L;M=best['Nb'];nu=M/N
    twist=tuple(best['replicas'][0]['metadata']['twist_pi'])
    u,v,meta=exact.orbitals(L,best['r'],twist,particles=M)
    assert meta['occupied']==best['replicas'][0]['metadata']['occupied']
    xy=np.indices((L,L)).reshape(2,-1).T
    gs=[];checks=[]
    for alpha,U in enumerate((u,v)):
        projector=U.conj()@U.T
        k=(2*np.pi*xy+(-1)**alpha*np.pi*np.array(twist))/L
        k=k[meta['occupied'][alpha]]
        G=np.exp(1j*xy@k.T).sum(axis=1).reshape(L,L)/N
        errors=dict(orbitals_vs_plane_waves=float(np.max(abs(G.ravel()-projector[0]))),projector_idempotence=float(np.max(abs(projector@projector-projector))),density=float(abs(G[0,0]-nu)),sum_abs_squared=float(abs(np.sum(abs(G)**2)-nu)))
        assert max(errors.values())<1e-11
        gs.append(G);checks.append(errors)
    combined=np.load(SOURCE/f'combined_r{best["r"]:g}_h{best["h"]:g}.npz')
    cn=np.fft.ifft2(combined['sn']).real;gb=np.fft.ifft2(combined['nb'])
    datasets=[np.load(ROOT/r['array_file']) for r in best['replicas']]
    cb=[np.fft.ifft2(a['blocks_sn'],axes=(-2,-1)).real for a in datasets]
    bb=[np.fft.ifft2(a['blocks_nb'],axes=(-2,-1)) for a in datasets]
    def blockerr(values,component='real'):
        vals=[getattr(x,component) for x in values]
        return np.sqrt(sum(x.var(axis=0,ddof=1)/len(x) for x in vals))/len(vals)
    ce=blockerr(cb);bre=blockerr(bb);bie=blockerr(bb,'imag')
    assert abs(cn[0,0]-nu*(1-nu))<1e-10
    assert abs(gb[0,0]-nu)<1e-10
    np.savez_compressed(OUT/'correlations.npz',G_parton_parent=np.stack(gs),projected_density_connected=cn,projected_density_error=ce,projected_boson_G=gb,projected_boson_G_error_real=bre,projected_boson_G_error_imag=bie)
    with (OUT/'direction_cuts.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['direction','steps','distance','dx','dy','G1_real','G1_imag','G1_abs','G2_real','G2_imag','G2_abs','projected_Cnn','Cnn_error','projected_Gb_real','Gb_error_real','projected_Gb_imag','Gb_error_imag'])
        for name,(dx,dy) in DIRECTIONS.items():
            for ell in range(L//2+1):
                i,j=ell*dx,ell*dy
                w.writerow([name,ell,float(np.hypot(i,j)),i,j]+[value for G in gs for value in (G[i,j].real,G[i,j].imag,abs(G[i,j]))]+[cn[i,j],ce[i,j],gb[i,j].real,bre[i,j],gb[i,j].imag,bie[i,j]])
    fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for alpha,G in enumerate(gs):
        for name,(dx,dy) in DIRECTIONS.items():
            ell=np.arange(L//2+1);distance=ell*np.hypot(dx,dy);vals=G[ell*dx,ell*dy]
            axs[alpha,0].plot(distance,vals.real,'o-',label=name,ms=4)
            mask=(ell>0)&(abs(vals)>1e-12)
            axs[alpha,1].semilogy(distance[mask],abs(vals[mask]),'o-',label=name,ms=4)
        for ax in axs[alpha]:ax.set(xlabel='Real-space distance (lattice units)');ax.grid(alpha=.2);ax.legend()
        axs[alpha,0].set(title=f'Parton {alpha+1}: real part',ylabel=r'Re $G_\alpha(R)$')
        axs[alpha,1].set(title=f'Parton {alpha+1}: magnitude, R > 0',ylabel=r'$|G_\alpha(R)|$')
    fig.suptitle(f'Bare parent Fermi seas: 12 x 12, Nb={M}, r={best["r"]}, density={nu:.6f}\nNot a bare-parton correlator measured after Gutzwiller projection')
    fig.savefig(OUT/'parton_correlations.png',dpi=180);fig.savefig(OUT/'parton_correlations.svg');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for name,(dx,dy) in DIRECTIONS.items():
        ell=np.arange(1,L//2+1);i,j=ell*dx,ell*dy;distance=ell*np.hypot(dx,dy)
        axs[0].errorbar(distance,cn[i,j],yerr=ce[i,j],fmt='o-',ms=4,label=name)
        axs[1].errorbar(distance,gb[i,j].real,yerr=bre[i,j],fmt='o-',ms=4,label=name)
    for ax in axs:ax.set(xlabel='Real-space distance (lattice units)');ax.axhline(0,color='grey',lw=.7);ax.grid(alpha=.2);ax.legend()
    axs[0].set(title='Projected physical density correlations',ylabel=r'$C_n(R)=\langle(n_0-\nu)(n_R-\nu)\rangle$')
    axs[1].set(title='Projected physical boson correlations',ylabel=r'Re $\langle b_0^\dagger b_R\rangle$')
    fig.suptitle('VMC state: r=0.03, g=-0.1538, h=0; density correlations also equal parton-density correlations')
    fig.savefig(OUT/'projected_physical_correlations.png',dpi=180);plt.close(fig)
    first=dict(G1_x_abs=float(abs(gs[0][1,0])),G1_y_abs=float(abs(gs[0][0,1])),G2_x_abs=float(abs(gs[1][1,0])),G2_y_abs=float(abs(gs[1][0,1])))
    report=dict(L=L,Nb=M,filling=nu,r=best['r'],g=best['g'],h=best['h'],twist_pi=list(twist),parton_definition='G_alpha(R)=<d_alpha^dag(0)d_alpha(R)> in the unprojected Fermi sea; twisted-boundary gauge, unwrapped displacements',projected_bare_parton='P_G d_i^dag d_j P_G = 0 for i != j; at i=j it is n_b(i)',projected_density='All species alpha,beta satisfy <(n_alpha(i)-nu)(n_beta(j)-nu)>_projected=C_boson_density(i,j)',checks=checks,nearest_neighbor=first,maximum_axis_separation=L//2,exponent_fit=None,exponent_fit_reason='Only six nonzero axis separations on L12; shell oscillations and boundary twists prevent a controlled asymptotic fit')
    (OUT/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# 实空间 parton 关联：12×12，Nb=86','',
    f'采用当前最低能候选：r={best["r"]}、g={best["g"]:.6f}、h=0，填充{nu:.8f}。','',
    '主图计算费米海母态的单parton关联 G_alpha(R)=<d_alpha†(0)d_alpha(R)>=sum_(k occupied) exp(ik·R)/N。这是给定占据轨道的精确有限尺寸结果，无蒙特卡洛误差。g作用于投影后的波函数，不进入这里的自由parton母态；本图不包含规范涨落。','',
    '内部边界扭转分别为±(.37,.19)pi，总物理玻色扭转为0。这里用twisted-boundary规范，R为展开坐标；G的复相位依赖规范。换为均匀Peierls相位、周期场的规范，G乘以exp[-i s_alpha theta·R/L]，模长相同。没有将跨边界的裸场直接当作普通周期场平均。','',
    '严格投影仅保留每格点|0,0>与|1,1>。i不等于j时，单独移动一个parton会离开物理子空间，因此P_G d_alpha†(i)d_alpha(j) P_G=0。这个零不能解释为parton有能隙或物理绝缘。若要研究含规范涨落的parton传播，需要指定规范固定的Green函数，或在包含规范链接的扩展Hilbert空间定义带Wilson线的算符；当前纯玻色VMC没有直接计算它。','',
    '另一方面，投影后n1=n2=nb，故两种parton的所有同/异species密度关联都等于已测量的物理玻色密度关联Cn(R)。副图给出这些关联，并同时给出物理玻色单粒子关联Gb(R)。从各独立链的24个动量分块分别逆Fourier后计算误差，保留动量误差间的相关性。','',
    '|最近邻|沿x|沿y|','|---|---|---|',f'|母态 parton 1 的模长|{first["G1_x_abs"]:.8f}|{first["G1_y_abs"]:.8f}|',f'|母态 parton 2 的模长|{first["G2_x_abs"]:.8f}|{first["G2_y_abs"]:.8f}|','',
    '两种parton的强传播方向互换。曲线有费米占据产生的符号与振荡；弱方向以及对角线中有部分填充壳层造成的有限尺寸残余。模长低于1e-12的点在半对数图中省略，但CSV保存原值；这些干涉零点不能作为指数衰减或能隙的证据。','',
    '轴向仅能取独立距离1至6；对角线步数1至6对应几何距离sqrt(2)至6sqrt(2)。不对这六个点拟合无控制的长距离幂指数，也不把母态结果解释为投影后parton的重整化指数。','',
    '验证：平面波求和与占据轨道投影矩阵相符；投影矩阵平方等于自身；G(0)=nu，sum_R |G(R)|²=nu。物理密度Cn(0)=nu(1-nu)，Gb(0)=nu。','',
    '输出parton_correlations.png/svg、projected_physical_correlations.png、direction_cuts.csv、correlations.npz、summary.json。复现运行analyze_parton_realspace_n06.py。']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
