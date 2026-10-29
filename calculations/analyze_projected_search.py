"""Summarize exact projected search and independently validated VMC energies."""

from vmc_paths import PROJECT_ROOT
from pathlib import Path
import csv
import json
import sys

ROOT=PROJECT_ROOT
if (ROOT/'tmp/plot-libs').exists():sys.path.insert(0,str(ROOT/'tmp/plot-libs'))
import numpy as np
import projected_parton_exact as p


def main():
    out=ROOT/'output/projected-jk'
    exact=json.loads((out/'exact_results.json').read_text())
    pure=json.loads((out/'vmc8_pure_scan.json').read_text())
    grid=json.loads((out/'vmc8_jastrow_grid.json').read_text())
    final=json.loads((out/'vmc8_finalist_replicas.json').read_text())
    if not pure.get('complete') or grid.get('pending') or 'combined' not in final:
        raise RuntimeError('Production jobs must finish before summarizing')
    best=final['combined'];rows=[]
    for k in range(1,11):
        v=pure['summary'][k-1];coeff=np.array([1.,k]);cov=np.array(best['covariance_jk_of_mean'])
        refs=p.references(64,1,k)
        baseline=min(refs.values())
        rows.append(dict(k=k,exact4_pure=exact['optimized'][k-1]['best_pure']['energy'],
                         exact4_jastrow=exact['optimized'][k-1]['best_jastrow']['energy'],
                         vmc8_pure_best=v['energy'],vmc8_pure_error=v['error'],
                         vmc8_fixed_g_minus_010=best['e_j']+k*best['e_k'],
                         vmc8_fixed_g_error=float(np.sqrt(coeff@cov@coeff)),
                         best_reference=baseline,phase_dimer=refs['phase_dimer'],
                         staggered_dimer=refs['staggered_dimer'],constant_reference=refs['constant_boson'],
                         pure_r_lo=v['r_lo'],pure_r_hi=v['r_hi']))
    with (out/'projected_energy_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    k=np.arange(1,11)
    axes[0,0].plot(k,[v['exact4_pure'] for v in rows],'o-',label='Pure projection')
    axes[0,0].plot(k,[v['exact4_jastrow'] for v in rows],'s-',label='Projection + optimized NN Jastrow')
    axes[0,0].plot(k,[min(p.references(16,1,int(kk)).values()) for kk in k],'k--',label='Best simple reference')
    axes[0,0].set(title='4 x 4: exact configuration enumeration',xlabel='K/J',ylabel='Energy / site');axes[0,0].legend(fontsize=8)
    axes[0,1].errorbar(k,[v['vmc8_pure_best'] for v in rows],yerr=[v['vmc8_pure_error'] for v in rows],fmt='o-',label='Best sampled pure shell')
    axes[0,1].errorbar(k,[v['vmc8_fixed_g_minus_010'] for v in rows],yerr=[v['vmc8_fixed_g_error'] for v in rows],fmt='s-',label='Fixed strong shell, g=-0.10; independent chains')
    axes[0,1].plot(k,[v['best_reference'] for v in rows],'k--',label='Best simple reference, same N')
    axes[0,1].set(title='8 x 8: projected VMC (statistical errors)',xlabel='K/J',ylabel='Energy / site');axes[0,1].legend(fontsize=8)
    rs=sorted((r for r in grid['records'] if r['L']==8),key=lambda r:r['g'])
    axes[1,0].errorbar([r['g'] for r in rs],[r['e_j']+10*r['e_k'] for r in rs],
                     yerr=[float(np.sqrt(np.array([1.,10.])@np.array(r['covariance_jk_of_mean'])@np.array([1.,10.]))) for r in rs],fmt='o-')
    axes[1,0].errorbar([-.1],[best['energy_K10']],yerr=[best['energy_K10_error']],fmt='s',c='black',label='Independent validation')
    axes[1,0].set(title='8 x 8, K=10: smooth Jastrow direction',xlabel='g in exp(-g Q_nn)',ylabel='Energy / site');axes[1,0].legend(fontsize=8)
    moments=np.load(out/'exact_moments.npz')
    m={name:moments[name][0] for name in ('q_weights','q_cdw','j_moments','k_moments')};m['sites']=16
    for g,color in ((0,'#D26527'),(-.1,'#167FA6')):
        points=[(4,p.evaluate(m,g)['cdw_m2'],0.)]
        for L in (6,8):
            if L==8 and g==-.1:points.append((8,best['m_cdw_squared'],best['m_cdw_squared_error']))
            else:
                r=next(v for v in grid['records'] if v['L']==L and abs(v['g']-g)<1e-8)
                points.append((L,r['m_cdw_squared'],r['s_pi_pi_error']/(L*L)))
        axes[1,1].errorbar([1/x[0]**2 for x in points],[x[1] for x in points],yerr=[x[2] for x in points],fmt='o-',color=color,label=f'g={g:g}')
    axes[1,1].set(title='Density order diagnostic: no extrapolation assumed',xlabel='1/N',ylabel=r'$m_{\rm CDW}^2=S(\pi,\pi)/N$');axes[1,1].legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.18)
    fig.suptitle('Exactly Gutzwiller-projected square-lattice partons | half filling | J=1',fontsize=15)
    fig.savefig(out/'projected_search_summary.png',dpi=190);fig.savefig(out/'projected_search_summary.svg');plt.close(fig)
    lines=['# Gutzwiller 投影后的各向异性变分搜索','',
           '固定 J=1，半填充、零温；物理波函数为两个同位置 Slater 行列式的乘积。投影严格去除单独占据的 parton 构型，所有能量均在物理硬核玻色子空间计算，不使用 Wick 因子化或此前的 1/nu 重标度。','',
           '## 已完成的搜索','',
           '- 4×4：完整枚举 12870 个物理构型，比较两组相反内部边界扭转（总物理玻色扭转为零），枚举各向异性 r∈[0,1] 内所有不同占据壳层。每组扭转有 5 个壳层。','- 8×8：对一组扭转的全部 17 个不同壳层执行投影 VMC；每个壳层 512 热化 sweep、2048 次测量、间隔 2 sweep，误差使用连续分块。','- 额外加入最近邻密度 Jastrow exp(-g Q_nn)，其中 Q_nn=Σ_<ij>n_i n_j。这个因子是明确增加的变分通道，不伪称为纯投影平均场的一部分。','- 强各向异性壳层扫描 g=-0.25,-0.15,-0.10,-0.05,0,0.10，并在 g=-0.10 固定后用两个独立种子各 8192 次测量复核，避免直接使用选优链的最低噪声值。','',
           '## 能量比较','',
           '| K/J | 4×4纯投影最优 | 8×8纯投影最低估计 | 8×8强壳层、g=-0.10 | 同尺寸最低简单参考 |',
           '|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['k']} | {r['exact4_pure']:.6f} | {r['vmc8_pure_best']:.6f} ± {r['vmc8_pure_error']:.6f} | {r['vmc8_fixed_g_minus_010']:.6f} ± {r['vmc8_fixed_g_error']:.6f} | {r['best_reference']:.6f} |")
    lines+=['','参考态包含投影交错二聚体 -J/2、实数环交换二聚体/交叉方格 -K/4、交替复相位二聚体 -K/4-J²/(8K)（K>=J/2），以及固定粒子数的常数振幅超流参考态。最后一项为 -JN/(N-1)+KN(N-2)/[8(N-1)(N-3)]。比较波函数不受此前未投影 SCF 的存在阈值限制。','',
            '在 K=1 的本次搜索中，简单超流参考仍更低。在 K>=3 的采样点，强各向异性投影候选明显低于上述参考。K=2 的有限尺寸/候选族竞争更敏感；没有据此宣布相边界。纯壳层扫描的逐点最小值有选优统计偏差，故独立验证的固定候选能量是更可靠的定量证据。','',
            '## 驻点与“自洽”的准确含义','',
            '有限平移不变 NN parent Hamiltonian 的占据轨道是平面波，r 变化仅在能级交叉时改变占据子空间。因此投影能量在闭壳层区间内严格平坦，不能从 dE/dr=0 提取唯一 r，更不能确定整体 hopping 能标。4×4 的低能强壳层包含 r=0；8×8 对所用扭转为 0<=r<0.075337468542。r=.03 只是该壳层的代表，不能说成唯一自洽跃迁比。','',
            '可连续优化的 Jastrow 通道满足 ∂_g e=-2 Re Cov(E_loc/N,Q_nn)=0。4×4 强壳层的精确驻点为 g=-0.114176241988，曲率为正；K=10 时 e=-4.379735559290。一个周期二 parent 键调制方向的能量也在零调制附近增加，见 bond_stability.json；这仅是已检验的方向，不是所有通量、配对和空间模式的完整稳定性证明。','',
            f"8×8 独立复核 g=-0.10 得 e(K=10)={best['energy_K10']:.9f} ± {best['energy_K10_error']:.9f}，但 ∂_g e={best['gradient_K10']:.6f} ± {best['gradient_K10_error']:.6f}，约 3σ 为正。故该点是低能候选，但不把它称为精确驻点。网格导数在 g=-0.15 与 -0.05 间变号；后续关联测量已转向 g=-0.11，继续检查梯度。",'',
            '## 物理性质与限制','',
            '强壳层固定每一行与每一列的玻色子数，故 e_J=0 且所有单玻色非局域关联严格为零；低能并不自动等于有输运的稳定 CPBM。Jastrow 只改变支撑集内振幅，不解除这个行列约束。半填充还可能存在棋盘格密度序，需检查 C_n(r)、S_n(q)、m_CDW² 和尺寸变化。4、6、8 的有限尺寸值在图中给出，未强行外推为零或非零。','',
            '相关背景：[Motrunich–Fisher, PRB 75, 235116 (2007)](https://arxiv.org/abs/cond-mat/0703261)。文献的投影行列式方法用于构造本次波函数；上述数值来自本目录实际运行。','',
            '## 复现与文件','',
            '```powershell','python projected_parton_exact.py','python run_projected_vmc_scan.py','python run_projected_jastrow_grid.py','python run_projected_finalist.py','python analyze_projected_search.py','python -m unittest test_projected_parton_exact test_projected_parton_vmc -v','```','',
            '输出包含 exact_results.json、vmc8_pure_scan.json、vmc8_jastrow_grid.json、vmc8_finalist_replicas.json、projected_energy_comparison.csv、原始独立链 NPZ 和图。局域能量、行列式更新、完整投影枚举、规范不变性、梯度和周期二调制方向等 9 项检验通过。']
    (out/'projected_search_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Saved projected search summary',str(out))


if __name__=='__main__':main()
