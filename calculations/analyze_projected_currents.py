"""Combine current-VMC replicas, retain connected errors, and make figures."""

from vmc_paths import PROJECT_ROOT
from pathlib import Path
import csv
import json
import sys

ROOT=PROJECT_ROOT
if (ROOT/'tmp/plot-libs').exists():sys.path.insert(0,str(ROOT/'tmp/plot-libs'))
import numpy as np
import projected_currents as currents
from run_projected_correlations import gradient_with_jackknife

OUT=ROOT/'output/projected-currents'


def load_records(L,r):
    records=[]
    for file in sorted(OUT.glob(f'currents_L{L}_r{r:g}_g-0.105_seed*.json')):
        record=json.loads(file.read_text(encoding='utf-8'))
        record['_json_file']=str(file)
        records.append(record)
    return records


def combine(records):
    if not records:raise ValueError('No replica records')
    datasets=[np.load(OUT/Path(r['array_file']).name) for r in records]
    weights=np.array([r['samples'] for r in records],float);weights/=weights.sum()
    L=records[0]['L'];N=L*L
    arrays={}
    for key in ('cc','hh','diamag','mean_current'):
        arrays[key]=sum(w*d[key] for w,d in zip(weights,datasets))
        arrays[key+'_error']=np.sqrt(sum(w*w*np.var(d['blocks_'+key],axis=0,ddof=1)/len(d['blocks_'+key])
                                        for w,d in zip(weights,datasets)))
    arrays['sjj']=currents.fourier(arrays['cc'])
    arrays['cc_connected']=arrays['cc']-np.outer(arrays['mean_current'],arrays['mean_current'])[:,:,None,None]
    arrays['sjj_connected']=currents.fourier(arrays['cc_connected'])
    errors={key:np.zeros_like(arrays[key].real) for key in ('cc_connected','sjj_connected')}
    imag_error=np.zeros_like(arrays['sjj_connected'].real)
    # Each replica remains at its original independent weight. Delete one of
    # its contiguous blocks, recompute that chain's mean, then the nonlinear
    # connected statistic. Sum independent chain jackknife variances.
    for weight,record,d in zip(weights,records,datasets):
        blocks=d['blocks_cc'];B=len(blocks);n=record['samples']
        lengths=np.array([len(x) for x in np.array_split(np.arange(n),B)])
        leave_cc=[];leave_q=[]
        for b,nb in enumerate(lengths):
            cc_new=(n*d['cc']-nb*blocks[b])/(n-nb)
            m_new=(n*d['mean_current']-nb*d['blocks_mean_current'][b])/(n-nb)
            cc=arrays['cc']+weight*(cc_new-d['cc'])
            m=arrays['mean_current']+weight*(m_new-d['mean_current'])
            connected=cc-np.outer(m,m)[:,:,None,None]
            leave_cc.append(connected);leave_q.append(currents.fourier(connected))
        leave_cc=np.array(leave_cc);leave_q=np.array(leave_q)
        errors['cc_connected']+=(B-1)/B*np.sum((leave_cc-leave_cc.mean(axis=0))**2,axis=0)
        errors['sjj_connected']+=(B-1)/B*np.sum((leave_q.real-leave_q.real.mean(axis=0))**2,axis=0)
        imag_error+=(B-1)/B*np.sum((leave_q.imag-leave_q.imag.mean(axis=0))**2,axis=0)
    arrays['cc_connected_error']=np.sqrt(errors['cc_connected'])
    arrays['sjj_connected_error_real']=np.sqrt(errors['sjj_connected'])
    arrays['sjj_connected_error_imag']=np.sqrt(imag_error)
    spectrum_blocks=[currents.fourier(d['blocks_cc']) for d in datasets]
    arrays['sjj_error_real']=np.sqrt(sum(w*w*np.var(b.real,axis=0,ddof=1)/len(b) for w,b in zip(weights,spectrum_blocks)))
    arrays['sjj_error_imag']=np.sqrt(sum(w*w*np.var(b.imag,axis=0,ddof=1)/len(b) for w,b in zip(weights,spectrum_blocks)))
    stats=[r['energy_stats'] for r in records];coef=np.array([1.,10.])
    cov=sum(w*w*np.array(s['covariance_jk_of_mean']) for w,s in zip(weights,stats))
    energy=sum(w*(s['e_j']+10*s['e_k']) for w,s in zip(weights,stats))
    gradients=[gradient_with_jackknife(d['raw_energy_samples']) for d in datasets]
    gradient=sum(w*g[0] for w,g in zip(weights,gradients))
    gradient_error=np.sqrt(sum(w*w*g[1]**2 for w,g in zip(weights,gradients)))
    ej=sum(w*s['e_j'] for w,s in zip(weights,stats));ek=sum(w*s['e_k'] for w,s in zip(weights,stats))
    hop_square=arrays['hh'].sum()
    hop_error=np.sqrt(sum(w*w*np.var(d['blocks_hh'].sum(axis=(1,2,3,4)),ddof=1)/len(d['blocks_hh'])
                          for w,d in zip(weights,datasets)))
    c=arrays['cc'];q=arrays['sjj'];contact=[float(c[a,a,0,0]) for a in range(2)]
    checks=dict(energy_diamagnetic_identity=float(abs(arrays['diamag'].sum()+ej)),
                spectrum_contact_identity=float(max(abs(q[a,a].real.mean()-contact[a]) for a in range(2))),
                tensor_hermiticity=float(np.max(abs(q-q.swapaxes(0,1).conj()))))
    if records[0]['r']<.075:
        checks.update(strong_cross_current_zero=float(np.max(abs(c[0,1]))),
                      strong_x_support_zero=float(np.max(abs(c[0,0,1:,:]))),
                      strong_y_support_zero=float(np.max(abs(c[1,1,:,1:]))),
                      adjacent_parallel_ring_identity=float(max(abs(c[0,0,0,1]-ek),abs(c[1,1,1,0]-ek))),
                      hopping_leakage_current_identity=float(abs(hop_square-q[0,0,0,0].real-q[1,1,0,0].real)))
    arrays['q']=2*np.pi*np.fft.fftfreq(L)
    summary=dict(L=L,N=N,r=records[0]['r'],g=records[0]['g'],samples=sum(r['samples'] for r in records),
                 seeds=[r['seed'] for r in records],energy=float(energy),energy_error=float(np.sqrt(coef@cov@coef)),
                 e_j=float(ej),e_k=float(ek),gradient_g=float(gradient),gradient_g_error=float(gradient_error),
                 mean_current=arrays['mean_current'].tolist(),mean_current_error=arrays['mean_current_error'].tolist(),
                 diamagnetic=arrays['diamag'].tolist(),diamagnetic_error=arrays['diamag_error'].tolist(),
                 total_raw=q[:,:,0,0].real.tolist(),
                 total_raw_error=arrays['sjj_error_real'][:,:,0,0].tolist(),
                 total_connected=arrays['sjj_connected'][:,:,0,0].real.tolist(),
                 total_connected_error=arrays['sjj_connected_error_real'][:,:,0,0].tolist(),
                 current_contact=contact,hopping_square_per_site=float(hop_square),hopping_square_error=float(hop_error),
                 checks=checks,error_method='20 contiguous blocks per independent chain; connected covariance via delete-block jackknife with fixed chain weights')
    return summary,arrays


def bz_map(ax,values,L,**kwargs):
    q=np.fft.fftshift(np.fft.fftfreq(L))*2
    edges=np.r_[q-1/L,q[-1]+1/L,q[-1]+3/L]
    image=np.pad(np.fft.fftshift(values).T,((0,1),(0,1)),mode='wrap')
    im=ax.pcolormesh(edges,edges,image,shading='flat',rasterized=True,**kwargs)
    ax.set(xlim=(-1,1),ylim=(-1,1),aspect='equal',xlabel=r'$q_x/\pi$',ylabel=r'$q_y/\pi$')
    ax.set_xticks([-1,0,1]);ax.set_yticks([-1,0,1])
    return im


def plot(cases,scaling):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'svg.fonttype':'path'})
    names=['strong','finite_reference'];colors=['#0072b2','#d55e00']
    fig,axes=plt.subplots(2,2,figsize=(10.3,9),layout='constrained')
    vmax=max(float(cases[n][1]['sjj_connected'][a,a].real.max()) for n in names for a in range(2))
    for row,name in enumerate(names):
        s,z=cases[name]
        for a in range(2):
            im=bz_map(axes[row,a],z['sjj_connected'][a,a].real,s['L'],cmap='magma',vmin=0,vmax=vmax)
            axes[row,a].set_title((r'$S^{c}_{j_xj_x}(\mathbf{q})$' if a==0 else r'$S^{c}_{j_yj_y}(\mathbf{q})$')+f" | r={s['r']:g}")
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8,label=r'Equal-time hopping-current covariance / $J^2$')
    fig.suptitle('Projected boson currents | 8 x 8, half filling, J=1, K=10, g=-0.105\nNN hopping current at finite q; complete uniform transport current at q=0',fontsize=12)
    fig.savefig(OUT/'projected_current_momentum.png',dpi=200)
    fig.savefig(OUT/'projected_current_momentum.svg');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11.6,8),layout='constrained')
    for name,color in zip(names,colors):
        s,z=cases[name];L=s['L'];rr=np.arange(L//2+1);label=f"r={s['r']:g}"
        for a in range(2):
            c=z['cc_connected'][a,a,0,:] if a==0 else z['cc_connected'][a,a,:,0]
            e=z['cc_connected_error'][a,a,0,:] if a==0 else z['cc_connected_error'][a,a,:,0]
            axes[0,a].errorbar(rr,c[rr],yerr=e[rr],marker='o',capsize=3,label=label,color=color)
            q=z['q']/np.pi
            sq=z['sjj_connected'][a,a,0,:].real if a==0 else z['sjj_connected'][a,a,:,0].real
            se=z['sjj_connected_error_real'][a,a,0,:] if a==0 else z['sjj_connected_error_real'][a,a,:,0]
            order=np.argsort(q);axes[1,a].errorbar(q[order],sq[order],yerr=se[order],marker='o',capsize=3,label=label,color=color)
    for a in range(2):
        direction='x' if a==0 else 'y';transverse='y' if a==0 else 'x'
        axes[0,a].set(title=rf'$j_{direction}$ correlation along $\hat{{{transverse}}}$',xlabel='Transverse separation / lattice spacing',ylabel=r'$C^c_{jj}(R)/J^2$')
        axes[1,a].set(title='Transverse momentum cut, longitudinal q=0',xlabel=rf'$q_{transverse}/\pi$',ylabel=r'$S^c_{jj}(q)/J^2$')
        for ax in axes[:,a]:ax.axhline(0,color='.5',linewidth=.7);ax.legend(frameon=False);ax.grid(alpha=.18)
    fig.suptitle('Equal-time current correlations and finite-size momentum cuts',fontsize=13)
    fig.savefig(OUT/'projected_current_realspace.png',dpi=200);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.7),layout='constrained')
    for a,color in enumerate(colors):
        x=[s['L'] for s in scaling];y=[s['total_connected'][a][a] for s in scaling]
        err=[s['total_connected_error'][a][a] for s in scaling]
        axes[0].errorbar(x,y,yerr=err,marker='o',capsize=4,color=color,label=r'$\langle (\delta I_'+('x' if a==0 else 'y')+r')^2\rangle/N$')
    axes[0].set(xlabel='L (L=4 exact; L=6,8 VMC)',ylabel=r'Uniform equal-time variance / $J^2$',title='Strong-shell size comparison, matched g=-0.105',xticks=[4,6,8])
    axes[0].legend(frameon=False);axes[0].grid(alpha=.2)
    positions=np.arange(2);width=.27
    strong,reference=[cases[n][0] for n in names]
    for i,(s,color) in enumerate(zip([strong,reference],colors)):
        values=[s['total_connected'][0][0],s['diamagnetic'][0]]
        errors=[s['total_connected_error'][0][0],s['diamagnetic_error'][0]]
        axes[1].bar(positions+(i-.5)*width,values,width,yerr=errors,capsize=4,label=f"r={s['r']:g}",color=color)
    axes[1].set(xticks=positions,xticklabels=[r'$\langle (\delta I_x)^2\rangle/N$',r'$\langle \tau_x\rangle/N$'],ylabel='Value (J=1)',title='Two distinct inputs; neither is the Drude weight')
    axes[1].legend(frameon=False);axes[1].axhline(0,color='.5',linewidth=.7)
    fig.suptitle('Transport diagnostics of a variational wavefunction: no conductivity inferred',fontsize=13)
    fig.savefig(OUT/'projected_current_transport_diagnostics.png',dpi=200);plt.close(fig)


def write_report(cases,scaling,validation):
    s,z=cases['strong'];r,_=cases['finite_reference']
    extension=next(v['hopping_extension'] for v in validation if v.get('label')=='strong_opt4_hopping_extension')
    fmt=lambda x,e:f'{x:.6f} ± {e:.6f}'
    lines=['# 投影玻色子 J–K 模型的等时流–流关联','',
           '使用与前次变分搜索相同的投影 Slater 行列式乘积及最近邻密度 Jastrow 波函数，J=1、K=10、半填充、物理玻色子周期边界。主结果为 L=8，r=0.03。g=-0.105 由先前 g=-0.11 与 -0.10 的梯度变号区间内插选取，属于近似优化参数；这不是全变分空间内已经收敛的鞍点。r=0.33 是同一 g 下的有限各向异性参考态，未单独优化 g。','',
           r'以格距、玻色子电荷和 ħ 为 1，测量物理玻色子的最近邻跃迁电流：',
           r'$$j_\alpha(i)=iJ[b^\dagger_{i+\hat\alpha}b_i-b^\dagger_i b_{i+\hat\alpha}],\qquad I_\alpha=\sum_i j_\alpha(i).$$','',
           r'$$C_{\alpha\beta}(R)=\frac{1}{2N}\sum_i\langle\{j_\alpha(i),j_\beta(i+R)\}\rangle,\quad C^c_{\alpha\beta}=C_{\alpha\beta}-\bar j_\alpha\bar j_\beta.$$','',
           r'$$S^c_{\alpha\beta}(q)=\sum_R e^{-iq\cdot[R+(\hat\beta-\hat\alpha)/2]}C^c_{\alpha\beta}(R),\quad S^c_{\alpha\beta}(0)=\frac{\langle\{\delta I_\alpha,\delta I_\beta\}\rangle}{2N}.$$','',
           '有限 q 图只包含最近邻跃迁电流。环交换的两个相反方向跃迁在均匀矢势下 Peierls 相位抵消，因此其均匀输运电流为零，q=0 总电流是完整的。若研究完整的有限 q 规范响应，还须规定环交换的局域 Peierls 路径并加入相应环流。','',
           '## 数值结果','',
           '| L=8 | 低能强各向异性态 r=0.03 | 参考态 r=0.33 |','|---|---:|---:|',
           f"| 测量次数（各两条独立链） | {s['samples']} | {r['samples']} |",
           f"| 能量/格点 | {fmt(s['energy'],s['energy_error'])} | {fmt(r['energy'],r['energy_error'])} |",
           f"| d(E/N)/dg | {fmt(s['gradient_g'],s['gradient_g_error'])} | {fmt(r['gradient_g'],r['gradient_g_error'])} |",
           f"| 平均 jx | {fmt(s['mean_current'][0],s['mean_current_error'][0])} | {fmt(r['mean_current'][0],r['mean_current_error'][0])} |",
           f"| 平均 jy | {fmt(s['mean_current'][1],s['mean_current_error'][1])} | {fmt(r['mean_current'][1],r['mean_current_error'][1])} |",
           f"| Var(Ix)/N | {fmt(s['total_connected'][0][0],s['total_connected_error'][0][0])} | {fmt(r['total_connected'][0][0],r['total_connected_error'][0][0])} |",
           f"| Var(Iy)/N | {fmt(s['total_connected'][1][1],s['total_connected_error'][1][1])} | {fmt(r['total_connected'][1][1],r['total_connected_error'][1][1])} |",
           f"| 抗磁项 tau_x/N | {fmt(s['diamagnetic'][0],s['diamagnetic_error'][0])} | {fmt(r['diamagnetic'][0],r['diamagnetic_error'][0])} |",
           f"| 抗磁项 tau_y/N | {fmt(s['diamagnetic'][1],s['diamagnetic_error'][1])} | {fmt(r['diamagnetic'][1],r['diamagnetic_error'][1])} |",'',
           '参考态使用相反内部边界扭转，保证物理周期性，但有限尺寸占据海不必具有时间反演对称性。其非零平均电流已在 connected 结果中扣除；不能把参考态的持久电流解读为自发输运序。','',
           '![电流动量关联](projected_current_momentum.png)','',
           '## 强各向异性态中的空间结构','',
           '每一行、每一列的粒子数固定使单次跃迁把态带出原占据扇区，因此单粒子关联及平均跃迁动能为零。两次方向相反的跃迁能够返回原扇区，故等时流–流关联非零。具体选择定则为 Cxy(R)=0，Cxx(R) 仅能在 Rx=0 时非零，Cyy(R) 仅能在 Ry=0 时非零。对应 Sxx 与 qx 无关，Syy 与 qy 无关。','',
           r'同键接触项满足 $C_{\alpha\alpha}(0)=J^2\langle n_i+n_{i+\alpha}-2n_i n_{i+\alpha}\rangle$。相邻平行键满足 $C_{xx}(\hat y)=C_{yy}(\hat x)=J^2 e_K$，其中 $E/N=J e_J+K e_K$。本次低能态相邻平行电流的负关联约为 '+f"{s['e_k']:.6f} J²；它与环交换能逐样本一致。",'',
           '![实空间及动量切线](projected_current_realspace.png)','',
           '| L，固定 g=-0.105 | Var(Ix)/N | Var(Iy)/N |','|---|---:|---:|']
    for item in scaling:
        lines.append(f"| {item['L']} | {fmt(item['total_connected'][0][0],item['total_connected_error'][0][0])} | {fmt(item['total_connected'][1][1],item['total_connected_error'][1][1])} |")
    lines += ['', '三个尺寸只给出有限尺寸比较，未对热力学极限作拟合。','',
              '![输运诊断](projected_current_transport_diagnostics.png)','',
              '## 能否据此判断金属、绝缘体或超流','',
              '等时电流涨落含有全部激发能上的矩阵元权重，不能确定权重是否位于低频。以真实非简并基态为前提，记 Re σxx(ω)=πDxx δ(ω)+σreg(ω)，有',
              r'$$D_{xx}=\frac{\langle\tau_x\rangle}{N}-\frac{2}{N}\sum_{n\ne0}\frac{|\langle n|I_x|0\rangle|^2}{E_n-E_0},\qquad \tau_x=-H_{Jx}.$$','',
              r'而 $S^c_{xx}(0)=N^{-1}\sum_{n\ne0}|\langle n|I_x|0\rangle|^2$ 没有激发能分母。平衡态光学求和规则为 $\int_0^\infty d\omega\,\mathrm{Re}\,\sigma_{xx}(\omega)=\pi\langle\tau_x\rangle/(2N)$（零频 δ 函数计半权重）。当前试探态并非 H 的本征态，因此这些基态谱公式不能直接套用来构造正定光学电导。','',
              '尤其，低能态的抗磁项严格为零，而电流涨落为正。这个组合不能支持“电导严格为零”的结论：它指出原试探族禁止了单粒子虚跃迁，尚不足以描述完整的输运响应。固定波函数或只在同一个行列扇区内重优化得到的扭转能量曲率也会人为平坦，不能据此宣布 Drude 权重为零。','',
              r'可直接验证遗漏的能量下降方向。令 $T=H_J/J$，$|\Psi_\epsilon\rangle\propto(1-\epsilon T)|\Psi\rangle$。由于环交换保持行列粒子数而单次跃迁离开此扇区，',
              r'$$\left.\frac{d(E/N)}{d\epsilon}\right|_0=-\frac{2J}{N}\langle T^2\rangle<0,\quad \frac{\langle H_J^2\rangle}{N}=\frac{\langle I_x^2+I_y^2\rangle}{N}.$$','',
              f"本次 L=8 的遗漏跃迁范数 <H_J²>/N={fmt(s['hopping_square_per_site'],s['hopping_square_error'])}。在独立的 L=4 全物理希尔伯特空间中，先将 g 优化到 -0.114176242（g 梯度为零、曲率为正），再把变分子空间扩为 span{{Ψ,H_JΨ}}，能量/格点仍从 {extension['trial_energy_per_site']:.9f} 降到 {extension['improved_energy_per_site']:.9f}。这证明目前的参数优化只是在受限试探族内完成。",'',
              '后续计算输运应先扩充试探族以允许这些虚跃迁，再求物理边界扭转下重优化的能量曲率，或构建投影激发态/虚时电流关联以估计低频谱。当前结果证实存在非零均匀电流涨落及反向关联，但尚不能确定 σDC、Drude 权重或超流刚度，也不能据此确认稳定玻色金属相。','',
              '## 验证与复现','',
              '测量器在每个样本上连续作用两个硬核玻色子电流算符后才求波函数比值，因此保留中间态在波函数节点之外的贡献。没有把两个局域电流估计值相乘。L=4 全配置直接算符作用与 VMC 局域估计值精确加权平均的最大误差约 1.4×10⁻¹⁴；还检查了随机复轨道、键接触恒等式、环交换恒等式和傅里叶归一化。误差使用每条链的 20 个连续块，connected 方差额外用删块 jackknife 传播平均电流扣除误差。','',
              '程序：projected_currents.py、run_projected_currents.py、run_projected_currents_sizes.py、projected_exact_currents.py、validate_projected_currents.py、analyze_projected_currents.py。结果包括独立副本 JSON/NPZ、合并 NPZ、current_momentum_distributions.csv 和 current_summary.json。','',
              '线性响应与刚度定义参考：[Scalapino, White, Zhang, PRL 68, 2830](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.68.2830)；[Watanabe et al., General Properties of Non-linear Optical Conductivities](https://link.springer.com/article/10.1007/s10955-020-02654-5)。']
    (OUT/'current_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run():
    cases={}
    for name,r in [('strong',.03),('finite_reference',.33)]:
        records=load_records(8,r)
        if len(records)!=2:raise RuntimeError(f'Expected two L8 replicas for {name}, found {len(records)}')
        cases[name]=combine(records)
        np.savez_compressed(OUT/f'combined_L8_{name}.npz',**cases[name][1])
    exact=json.loads((OUT/'exact4_strong.json').read_text(encoding='utf-8'))
    exact_summary=dict(L=4,total_connected=exact['total_current_connected_per_site'],total_connected_error=[[0.,0.],[0.,0.]])
    scaling=[exact_summary]
    l6=load_records(6,.03)
    if len(l6)==2:
        s,z=combine(l6);scaling.append(s);np.savez_compressed(OUT/'combined_L6_strong.npz',**z)
    scaling.append(cases['strong'][0])
    summary={name:value[0] for name,value in cases.items()};summary['strong_size_comparison']=scaling
    (OUT/'current_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (OUT/'current_momentum_distributions.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['case','L','r','g','qx_pi','qy_pi','a','b','S_raw_real','S_raw_imag','S_connected_real','S_connected_imag','error_real','error_imag'])
        for name,(s,z) in cases.items():
            for ix,qx in enumerate(z['q']/np.pi):
                for iy,qy in enumerate(z['q']/np.pi):
                    for a in range(2):
                        for b in range(2):
                            ix4=a,b,ix,iy;raw=z['sjj'][ix4];conn=z['sjj_connected'][ix4]
                            writer.writerow([name,s['L'],s['r'],s['g'],qx,qy,'xy'[a],'xy'[b],raw.real,raw.imag,conn.real,conn.imag,z['sjj_connected_error_real'][ix4],z['sjj_connected_error_imag'][ix4]])
    validation=json.loads((OUT/'vmc_estimator_exact_validation.json').read_text(encoding='utf-8'))
    plot(cases,scaling);write_report(cases,scaling,validation)
    print(json.dumps(summary,indent=2),flush=True)
    return summary

if __name__=='__main__':run()
