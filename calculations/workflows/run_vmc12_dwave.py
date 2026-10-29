"""L12 projected VMC: four-bond d-wave momentum density and contact baseline."""

from vmc_paths import PROJECT_ROOT
from pathlib import Path
import json
import csv
import projected_parton_exact as exact
import projected_correlations as corr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=PROJECT_ROOT/'output/vmc12-dwave'

def dfield(pair):
    L=pair.shape[-1];q=2*np.pi*np.fft.fftfreq(L)
    qx,qy=np.meshgrid(q,q,indexing='ij')
    f=np.stack([(1+np.exp(-1j*qx))/2,-(1+np.exp(-1j*qy))/2])
    pq=np.fft.fft2(pair,axes=(-2,-1))
    full=np.einsum('axy,...abxy,bxy->...xy',f.conj(),pq,f).real
    contact=np.zeros_like(pair)
    contact[...,0,0,0,0]=pair[...,0,0,0,0]
    contact[...,1,1,0,0]=pair[...,1,1,0,0]
    base=np.einsum('axy,...abxy,bxy->...xy',f.conj(),np.fft.fft2(contact,axes=(-2,-1)),f).real
    return full,base,full-base

def run():
    OUT.mkdir(parents=True,exist_ok=True);summaries=[];datasets=[]
    for r in (.03,.33):
        reports=[];arrays=[]
        for seed in (1201,1202):
            stem=f'L12_r{r:g}_seed{seed}';jp=OUT/(stem+'.json');ap=OUT/(stem+'.npz')
            if jp.exists() and ap.exists():
                report=json.loads(jp.read_text());a=dict(np.load(ap))
            else:
                u,v,meta=exact.orbitals(12,r)
                print('START',r,seed,flush=True)
                report,a=corr.sample_correlations(u,v,12,g=-.11,seed=seed,warmup=512,samples=2048,stride=3,progress_every=256)
                report.update(r=r,orbital_metadata=meta)
                np.savez_compressed(ap,**a);jp.write_text(json.dumps(report,indent=2),encoding='utf-8')
            reports.append(report);arrays.append(a)
        pair=np.mean([a['pair'] for a in arrays],axis=0)
        blocks=np.concatenate([a['blocks_pair'] for a in arrays])
        full,contact,nonlocal_=dfield(pair);bf,bc,bn=dfield(blocks)
        # Independent replica means; contiguous block uncertainty within each chain.
        def error(key):
            vals=[dfield(a['blocks_pair'])[key] for a in arrays]
            return np.sqrt(sum(np.var(b,axis=0,ddof=1)/len(b) for b in vals))/2
        ef,en=error(0),error(2)
        np.savez_compressed(OUT/f'combined_r{r:g}.npz',pair=pair,blocks_pair=blocks,dwave=full,dwave_error=ef,contact=contact,nonlocal_dwave=nonlocal_,nonlocal_error=en,legacy_dwave=np.mean([a['pair_d'] for a in arrays],axis=0))
        energy=[];err=[]
        for report in reports:
            s=report['energy_stats'];coef=np.array([1.,10.])
            energy.append(s['e_j']+10*s['e_k']);err.append(float(np.sqrt(coef@np.asarray(s['covariance_jk_of_mean'])@coef)))
        u,v,m=exact.orbitals(12,r);_,_,zero=exact.orbitals(12,0)
        checks=dict(number=max(abs(a['sn'][0,0]) for a in arrays),boson_sum=max(abs(a['nb'].mean()-.5) for a in arrays),density_sum=max(abs(a['sn'].mean()-.25) for a in arrays),d_pi_pi=float(abs(full[6,6])))
        assert max(checks.values())<1e-8
        # Compare independent chains and alternative 10-block errors for diagnostics.
        chain=[dfield(a['pair'])[0] for a in arrays]
        ediff=abs(energy[0]-energy[1])/np.hypot(*err)
        ten=[b.reshape(10,2,12,12).mean(axis=1) for b in (bf[:20],bf[20:])]
        e10=np.sqrt(sum(np.var(b,axis=0,ddof=1)/len(b) for b in ten))/2
        summary=dict(L=12,Nb=72,J=1,K=10,r=r,g=-.11,seeds=[1201,1202],samples=4096,warmup_sweeps_per_chain=512,stride_sweeps=3,strong_shell=m['occupied']==zero['occupied'],energy_per_site=float(np.mean(energy)),energy_error=float(np.linalg.norm(err)/2),replica_energies=energy,replica_energy_errors=err,replica_energy_difference_sigma=float(ediff),acceptance=[a['acceptance'] for a in reports],seconds=sum(a['seconds'] for a in reports),full_range=[float(full.min()),float(full.max())],nonlocal_range=[float(nonlocal_.min()),float(nonlocal_.max())],max_full_error=float(ef.max()),max_nonlocal_error=float(en.max()),max_error_10blocks=float(e10.max()),checks=checks)
        summaries.append(summary);datasets.append((full,contact,nonlocal_,ef,en))
        print('COMBINED',json.dumps(summary),flush=True)
    q=np.fft.fftshift(2*np.pi*np.fft.fftfreq(12))/np.pi
    with (OUT/'dwave_momentum.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['r','qx_over_pi','qy_over_pi','n_d','error','same_bond_contact','nonlocal_n_d','nonlocal_error'])
        for s,(full,base,nl,ef,en) in zip(summaries,datasets):
            vals=[np.fft.fftshift(a) for a in (full,ef,base,nl,en)]
            for i in range(12):
                for j in range(12):w.writerow([s['r'],q[i],q[j]]+[a[i,j] for a in vals])
    fig,axs=plt.subplots(2,3,figsize=(13,7.6),constrained_layout=True)
    for row,(s,ds) in enumerate(zip(summaries,datasets)):
        for col,(a,label) in enumerate(zip(ds[:3],['Full d-wave density','Same-bond contact contribution','Nonlocal contribution'])):
            vmax=max(abs(a.min()),abs(a.max()))
            kwargs=dict(cmap='viridis',vmin=0,vmax=max(x[0].max() for x in datasets)) if col<2 else dict(cmap='RdBu_r',vmin=-max(vmax,1e-10),vmax=max(vmax,1e-10))
            im=axs[row,col].pcolormesh(q,q,np.fft.fftshift(a).T,shading='nearest',**kwargs)
            axs[row,col].set(title=f'r={s["r"]}: {label}',xlabel=r'$q_x/\pi$',ylabel=r'$q_y/\pi$')
            fig.colorbar(im,ax=axs[row,col],shrink=.85)
    fig.suptitle(r'12$\times$12 projected VMC, $N_b=72$, $J=1$, $K=10$, $g=-0.11$'+'\n'+r'$\Delta_d(i)=[b_i b_{i+x}+b_i b_{i-x}-b_i b_{i+y}-b_i b_{i-y}]/2$')
    fig.savefig(OUT/'dwave_momentum.png',dpi=180);fig.savefig(OUT/'dwave_momentum.svg');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    path=[(i,0) for i in range(7)]+[(6,i) for i in range(1,7)]+[(i,i) for i in range(5,-1,-1)]
    for s,(full,base,nl,ef,en) in zip(summaries,datasets):
        for ax,a,e in [(axs[0],full,ef),(axs[1],nl,en)]:
            ax.errorbar(range(len(path)),[a[i,j] for i,j in path],yerr=[e[i,j] for i,j in path],marker='o',ms=3,label=f'r={s["r"]}')
    for ax,title in zip(axs,['Full d-wave momentum density','Nonlocal contribution']):
        ax.set(title=title,ylabel=r'$n_d(q)$',xticks=[0,6,12,18],xticklabels=[r'$\Gamma$',r'$X$',r'$M$',r'$\Gamma$']);ax.legend();ax.grid(alpha=.25)
    fig.savefig(OUT/'dwave_cuts.png',dpi=180);plt.close(fig)
    (OUT/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    lines=['# 12×12 投影 VMC 的 d 波配对动量密度','',
    '模型：半填充 72 个硬核玻色子，周期物理边界，J=1、K=10。试探态 Psi=det U1 det U2 exp(-g Qnn)，g=-0.11；交叉跃迁比 r=.03 和 .33，内部相反扭转 (.37,.19)pi。参数沿用此前计算，本次未重新优化，不代表已求得 12×12 最优态。','',
    '四键 d 波定义 Delta_d(i)=[b_i b_(i+x)+b_i b_(i-x)-b_i b_(i+y)-b_i b_(i-y)]/2。n_d(q)=N^-1 sum_ij exp[-iq(j-i)] <Delta_d†(i)Delta_d(j)>。通过已有的完整 2×2 键场关联矩阵变换得到；包含共享端点项，没有使用 Wick 分解。固定粒子数下 <Delta_d>=0，所画为关联动量密度。','',
    '同键接触基线为 <n_i n_(i+x)> cos²(qx/2)+<n_i n_(i+y)> cos²(qy/2)。非局域图仅扣除此项，允许为负，不是另一种必须非负的粒子数。四键算符在 (pi,pi) 恒为零，属形状因子，不能解释为物理节点证据。旧的正向两键 (Delta_x-Delta_y)/sqrt(2) 结果另存 NPZ legacy_dwave；两种定义有限动量下不同。','',
    '每个参数使用两个独立种子，每链 512 sweep 热化、2048 次测量、间隔 3 sweep；每 sweep=144 次尝试。误差来自每链 20 个连续分块；原始能量样本和空间分块均保存。另检查每链 10 块误差和独立链能量差，有限样本仍可能低估慢模误差。','',
    '|r|能量/格点|满分布范围|非局域范围|最大误差|','|---|---|---|---|---|']
    for s in summaries:lines.append(f'|{s["r"]}|{s["energy_per_site"]:.6f} ± {s["energy_error"]:.6f}|{s["full_range"]}|{s["nonlocal_range"]}|{s["max_nonlocal_error"]:.6g}|')
    lines+=['','r=.03 与 r=0 的已占据轨道集合相同，固定每行每列 6 个粒子。因此非同键的最近邻配对关联被支撑集严格禁止，四键图形完全由接触形状因子产生。r=.33 为较宽支撑的对照态，图中的非局域部分是实际测量结果；单一尺寸与有限采样不能据此确立 CPBM 或配对凝聚。','', '复现：设置 OPENBLAS_NUM_THREADS=1 后运行 run_vmc12_dwave.py。完整采样设置、归一化检查与链间诊断见 summary.json。']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':run()
