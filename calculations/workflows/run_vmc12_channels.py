"""Independent validation and multiple pair-channel measurements after (r,g) search."""

from vmc_paths import PROJECT_ROOT
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json
import time
import csv
import projected_parton_exact as exact
import projected_parton_vmc as vmc
import projected_pair_channels as pc
import numpy as np

ROOT=PROJECT_ROOT;OUT=ROOT/'output/vmc12-optimized'
CHANNELS=['s_nn','d_x2y2','s_diag','d_xy']

def worker(job):
    r,g,seed,samples=job;stem=f'final_r{r:g}_g{g:.6f}_seed{seed}'
    jp=OUT/(stem+'.json');ap=OUT/(stem+'.npz')
    if jp.exists() and ap.exists():return json.loads(jp.read_text())
    u,v,meta=exact.orbitals(12,r);state=vmc.ProjectedState(u,v,12,g,seed=seed)
    start=time.perf_counter();accepted=attempted=0
    for _ in range(1024*144):accepted+=state.proposal();attempted+=1
    pair=np.empty((samples,4,4,12,12),complex);raw=np.empty((samples,6));dens=np.empty(samples)
    for m in range(samples):
        for _ in range(3*144):accepted+=state.proposal();attempted+=1
        pair[m]=pc.observe(state,full_matrix=False)
        ej,ek,q,sp=state.local_energy();raw[m]=[ej.real,ek.real,ej.imag,ek.imag,q,sp]
        dens[m]=sp/144
        if (m+1)%512==0:print('FINAL_PROGRESS',r,seed,m+1,samples,'seconds',round(time.perf_counter()-start,1),flush=True)
    blocks=np.stack([p.mean(axis=0) for p in np.array_split(pair,20)])
    np.savez_compressed(ap,pair=pair.mean(axis=0),blocks_pair=blocks,raw_energy_samples=raw)
    stats=vmc._stats(raw);c=np.array([1.,10.]);energy=raw[:,0]+10*raw[:,1]
    grad=-2*(np.mean(energy*raw[:,4])-energy.mean()*raw[:,4].mean())
    gj=[]
    for ids in np.array_split(np.arange(samples),20):
        e=np.delete(energy,ids);q=np.delete(raw[:,4],ids)
        gj.append(-2*(np.mean(e*q)-e.mean()*q.mean()))
    graderr=float(np.sqrt(19/20*np.sum((gj-np.mean(gj))**2)))
    rec=dict(r=r,g=g,seed=seed,samples=samples,warmup=1024,stride=3,L=12,Nb=72,J=1,K=10,acceptance=accepted/attempted,seconds=time.perf_counter()-start,metadata=meta,energy=float(energy.mean()),error=float(np.sqrt(c@np.asarray(stats['covariance_jk_of_mean'])@c)),gradient=float(grad),gradient_error=graderr,m_cdw_squared=float(dens.mean()),energy_stats=stats,array_file=str(ap.relative_to(ROOT)))
    jp.write_text(json.dumps(rec,indent=2),encoding='utf-8');return rec

def main():
    opt=json.loads((OUT/'optimization.json').read_text())
    selected=opt['selected'];jobs=[(s['r'],s['proposed_g'],seed,2048) for s in selected for seed in (3301,3302)]
    records=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(worker,j) for j in jobs]):
            rec=f.result();records.append(rec);print('FINAL_DONE',json.dumps({k:rec[k] for k in ['r','g','seed','energy','error','gradient','gradient_error','seconds']}),flush=True)
            (OUT/'final_checkpoint.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summaries=[];data=[];q=2*np.pi*np.fft.fftfreq(12)/np.pi;qs=np.fft.fftshift(q)
    rows=[]
    for s in selected:
        reps=[z for z in records if z['r']==s['r']];a=[dict(np.load(ROOT/z['array_file'])) for z in reps]
        meanpair=np.mean([z['pair'] for z in a],axis=0);mean=pc.fourier(meanpair)
        b=[pc.fourier(z['blocks_pair']) for z in a]
        err={k:np.sqrt(sum(np.var(x[k],axis=0,ddof=1)/len(x[k]) for x in b))/2 for k in CHANNELS+['nonlocal_'+c for c in CHANNELS]}
        combined={**mean,**{k+'_error':v for k,v in err.items()},'pair':meanpair}
        np.savez_compressed(OUT/f'channels_r{s["r"]:g}.npz',**combined)
        e=float(np.mean([z['energy'] for z in reps]));ee=float(np.linalg.norm([z['error'] for z in reps])/2)
        grad=float(np.mean([z['gradient'] for z in reps]));ge=float(np.linalg.norm([z['gradient_error'] for z in reps])/2)
        summary=dict(r=s['r'],g=s['proposed_g'],energy=e,error=ee,gradient=grad,gradient_error=ge,samples=sum(z['samples'] for z in reps),replicas=reps,replica_energy_difference_sigma=float(abs(reps[0]['energy']-reps[1]['energy'])/np.hypot(reps[0]['error'],reps[1]['error'])),m_cdw_squared=float(np.mean([z['m_cdw_squared'] for z in reps])),channels={})
        block10=[pc.fourier(z['blocks_pair'].reshape(10,2,4,4,12,12).mean(axis=1)) for z in a]
        summary['error_10block_max']={c:float((np.sqrt(sum(np.var(x[c],axis=0,ddof=1)/10 for x in block10))/2).max()) for c in CHANNELS}
        if s['r']==.03:
            qrad=q*np.pi;qx,qy=np.meshgrid(qrad,qrad,indexing='ij')
            w=float(meanpair[2,3,0,1].real)
            predicted=w*(np.cos(qx)+np.cos(qy))
            allowed=np.zeros((2,2,12,12),bool)
            allowed[0,0,0,0]=allowed[1,1,0,0]=True
            allowed[0,1,0,1]=allowed[1,0,0,11]=True
            support=float(np.max(abs(meanpair[2:,2:][~allowed])))
            diagnostic=dict(W=w,ring_energy_coefficient=float(np.mean([z['energy_stats']['e_k'] for z in reps])),s_diag_local_ring_formula_error=float(np.max(abs(mean['nonlocal_s_diag']-predicted))),d_xy_local_ring_formula_error=float(np.max(abs(mean['nonlocal_d_xy']+predicted))),diagonal_pair_outside_local_support=support)
            assert max(diagnostic[k] for k in ['s_diag_local_ring_formula_error','d_xy_local_ring_formula_error','diagonal_pair_outside_local_support'])<1e-8
            assert abs(diagnostic['ring_energy_coefficient']-2*w)<1e-8
            summary['strong_shell_local_ring_diagnostic']=diagnostic
        for c in CHANNELS:
            i=np.unravel_index(np.argmax(mean[c]),(12,12));j=np.unravel_index(np.argmax(abs(mean['nonlocal_'+c])),(12,12))
            summary['channels'][c]=dict(full_min=float(mean[c].min()),full_max=float(mean[c].max()),peak_q_over_pi=[float(q[k]) for k in i],peak_error=float(err[c][i]),max_abs_nonlocal=float(abs(mean['nonlocal_'+c][j])),nonlocal_at_max=float(mean['nonlocal_'+c][j]),nonlocal_error=float(err['nonlocal_'+c][j]),nonlocal_q_over_pi=[float(q[k]) for k in j])
            for ix in range(12):
                for iy in range(12):rows.append([s['r'],s['proposed_g'],c,q[ix],q[iy],mean[c][ix,iy],err[c][ix,iy],mean['contact_'+c][ix,iy],mean['nonlocal_'+c][ix,iy],err['nonlocal_'+c][ix,iy]])
        summaries.append(summary);data.append(combined)
    with (OUT/'pair_channels.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['r','g','channel','qx_over_pi','qy_over_pi','density','error','contact','nonlocal','nonlocal_error']);w.writerows(rows)
    labels=['Extended s (nearest)','d x2-y2 (nearest)','s (diagonal)','d xy (diagonal)']
    for nonlocal_ in (False,True):
        fig,axs=plt.subplots(len(data),4,figsize=(15,7.5),constrained_layout=True,squeeze=False)
        for col,(c,label) in enumerate(zip(CHANNELS,labels)):
            key=('nonlocal_' if nonlocal_ else '')+c
            vmax=max(float(abs(z[key]).max()) for z in data);vmin=-vmax if nonlocal_ else 0.
            for row,(s,z) in enumerate(zip(summaries,data)):
                im=axs[row,col].pcolormesh(qs,qs,np.fft.fftshift(z[key]).T,shading='nearest',cmap='RdBu_r' if nonlocal_ else 'viridis',vmin=vmin,vmax=max(vmax,1e-12))
                axs[row,col].set(title=f'{label}\nr={s["r"]}, g={s["g"]:.4f}',xlabel=r'$q_x/\pi$',ylabel=r'$q_y/\pi$')
                fig.colorbar(im,ax=axs[row,col],shrink=.8)
        fig.suptitle('12 x 12 projected VMC: '+('nonlocal pair correlations (same-bond contact removed)' if nonlocal_ else 'pair momentum densities (including same-bond contact)'))
        name='pair_channels_nonlocal' if nonlocal_ else 'pair_channels_full'
        fig.savefig(OUT/(name+'.png'),dpi=180);fig.savefig(OUT/(name+'.svg'));plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for rec in opt['records']:
        grid=rec['grid'];valid=[z for z in grid if z['min_ess_fraction']>=.2]
        axs[0].errorbar([z['g'] for z in valid],[z['energy'] for z in valid],yerr=[z['error'] for z in valid],marker='.',label=f'r={rec["r"]}')
        if rec['r'] in [z['r'] for z in summaries]:
            axs[1].errorbar([z['g'] for z in valid],[z['gradient'] for z in valid],yerr=[z['gradient_error'] for z in valid],marker='.',label=f'r={rec["r"]} scan')
    for s in summaries:
        axs[0].errorbar([s['g']],[s['energy']],yerr=[s['error']],fmt='ks',ms=6)
        axs[1].errorbar([s['g']],[s['gradient']],yerr=[s['gradient_error']],fmt='ks',ms=6)
    axs[0].set(xlabel='NN Jastrow g',ylabel='Energy per site',title='Correlated scan; squares = independent validation');axs[0].legend(fontsize=8)
    axs[1].set(xlabel='NN Jastrow g',ylabel='d(E/N) / dg',title='Stationarity diagnostic');axs[1].axhline(0,color='grey',lw=1);axs[1].legend(fontsize=8)
    for ax in axs:ax.grid(alpha=.2)
    fig.savefig(OUT/'variational_search.png',dpi=180);plt.close(fig)
    (OUT/'final_summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    lines=['# 12×12：变分 Jastrow 与多配对形状因子','',
    'J=1、K=10、半填充72个硬核玻色子，物理周期边界，零通量。Psi=det U1 det U2 exp(-g Qnn)，两parton交叉跃迁(1,r)/(r,1)，内部扭转相反(0.37,0.19)pi。','',
    '严格 Gutzwiller 投影（n1=n2=nb）保持不变；本次优化的是NN Jastrow g，不是松弛物理约束。在硬核Hilbert空间中 onsite exp[-u sum n(n-1)]恒等于1，不能有效变分。扫描7个不同的轨道海r=.03,.08,.15,.33,.5,.75,1，以及13个g∈[-.25,.05]；不是穷尽所有可能的parton海或所有Jastrow形式。','',
    '每个r有两条独立参考链（g0=-.10），每链256 sweep热化、1024测量、3 sweep间隔。重加权权重exp[-2(g-g0)Qnn]，每个target g都重新计算局域能量。20连续块删除jackknife给出能量和梯度误差；权重ESS至少20%才用于选点。权重ESS没有扣除自相关。相关采样候选通过局部二次插值提出，再用两条全新独立链验证（各1024 sweep热化、2048次测量、间隔3 sweep）。最终误差为每链20分块的合并误差，候选梯度来自-2Cov(Qnn,Re E_loc)。','',
    '|r|候选g|独立验证能量/格点|能量梯度|','|---|---|---|---|']
    for s in summaries:lines.append(f'|{s["r"]}|{s["g"]:.6f}|{s["energy"]:.6f} ± {s["error"]:.6f}|{s["gradient"]:.4f} ± {s["gradient_error"]:.4f}|')
    lines+=['','本次候选g仅需读作约-0.10和-0.12，插值的小数位不代表统计精度。两候选梯度均在2个标准误内与零相容。r=1的网格最低点处于g下边界，不能称该态已完成无边界优化；本次总体结论只限于所搜索范围。内部扭转与有限尺寸轨道海也可能造成动量图各向异性，不能直接解释为自发破坏旋转对称性。','',
    '配对场定义（各四项除以2）：最近邻s为+x,-x,+y,-y四键同号；d_x2-y2的x键为正、y键为负；对角s为±(x+y)、±(x-y)同号；d_xy中±(x+y)为正，±(x-y)为负。所有场均由b_i b_(i+delta)组成。n_c(q)=N^-1 sum_ij exp[-iq·(j-i)]<Delta_c†(i)Delta_c(j)>。q是玻色子对的质心动量，不是内部相对动量；没有用n_b乘一个形状因子来替代配对关联。','',
    '测量正向键x,y,x+y,x-y的两个2×2关联块并精确变换至四键场；不同键组之间的交叉关联未测，本次四个通道不需要它们。包括同键、共享端点和不相交键算符。相同波函数的不同测量通道不会有不同的变分能量；本次没有把配对BCS/Pfaffian结构加入试探态。','',
    '|r|通道|最大总配对密度|最大绝对非局域部分|该点非局域误差|','|---|---|---|---|---|']
    for s in summaries:
        for c,d in s['channels'].items():lines.append(f'|{s["r"]}|{c}|{d["full_max"]:.6f}|{d["max_abs_nonlocal"]:.6f}|{d["nonlocal_error"]:.6f}|')
    lines+=['','full图包含同键接触概率及四键场的几何形状因子；nonlocal图仅扣除同一物理键的接触项，可以有负值，不能将其整体当作另一个非负动量密度。某些动量的四键场严格为零是算符恒等式。强壳层固定每行每列6粒子，最近邻非同键配对转移被禁止，但对角键的矩形翻转可保留行列计数，因而d_xy/s_diag仍可能非平凡。','',
    '强壳层进一步可解析：P_++(R)=A+ delta_R0，P_--(R)=A- delta_R0，P_+-(R)=W delta_R,y，P_-+(R)=W* delta_R,-y。W为同一单位方格的环关联，e_K=2 Re W。因此 n_sdiag/dxy(q)=A+ cos²[(qx+qy)/2]+A- cos²[(qx-qy)/2] ± Re W (cos qx+cos qy)。实测数据逐动量检查此关系，并检查其他位移全部为零。强壳层d_xy的峰来自局域环交换，没有长距离配对关联；不能将它当作CPBM证据。这里nonlocal仅表示不同物理键，并不意味着长距离。','',
    '有限尺寸峰或某个通道较强不能单独确立凝聚或CPBM；需要尺寸标度、长距离配对关联、密度序诊断和更广试探族的能量比较。当前g只是所测两parton×单NN Jastrow族内的候选，不能称全Hilbert空间最优态。独立验证未显著降低旧g=-.11的能量时，应报告精度内相容，而不能宣称优化收益。','',
    '验证：4个新配对通道测试覆盖随机复轨道、直接行列式、共享端点、Hermitian性、实空间移位与Fourier相位、解析接触项，全部通过。4×4精确全构型重加权能量误差小于2e-15，协方差梯度与独立中心差分误差小于2e-9（reweight_exact4_validation.json）。强壳层整个对角动量图与局域环公式偏差小于2e-16。final_summary.json另存每链10块与20块误差检查。','',
    '输出：optimization.json（全部扫描及ESS），final_summary.json（独立链能量与梯度），pair_channels.csv（离散动量及误差），channels_r*.npz（关联矩阵/通道），final_*.npz（原始能量与空间分块），variational_search.png，pair_channels_full.png，pair_channels_nonlocal.png。复现依次运行optimize_vmc12.py与run_vmc12_channels.py。']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('FINISHED',json.dumps([{k:s[k] for k in ['r','g','energy','error','gradient','gradient_error']} for s in summaries]),flush=True)

if __name__=='__main__':main()
