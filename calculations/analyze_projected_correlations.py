"""Combine independent projected-VMC correlation replicas and plot physical spectra."""

from vmc_paths import PROJECT_ROOT
from pathlib import Path
import csv
import json
import sys

ROOT=PROJECT_ROOT
if (ROOT/'tmp/plot-libs').exists():sys.path.insert(0,str(ROOT/'tmp/plot-libs'))
import numpy as np

OUT=ROOT/'output/projected-correlations'
SOURCE=ROOT/'output/projected-jk'


def combine(records):
    datasets=[np.load(ROOT/r['array_file']) for r in records]
    weights=np.array([r['samples'] for r in records],float);weights/=weights.sum()
    arrays={}
    for key in ('cn','gb','pair','sn','nb','pair_x','pair_y','pair_s','pair_d'):
        arrays[key]=sum(w*z[key] for w,z in zip(weights,datasets))
        blocksets=[z['blocks_'+key] for z in datasets]
        if np.iscomplexobj(arrays[key]):
            arrays[key+'_error_real']=np.sqrt(sum(w*w*np.var(b.real,axis=0,ddof=1)/len(b) for w,b in zip(weights,blocksets)))
            arrays[key+'_error_imag']=np.sqrt(sum(w*w*np.var(b.imag,axis=0,ddof=1)/len(b) for w,b in zip(weights,blocksets)))
        else:
            arrays[key+'_error']=np.sqrt(sum(w*w*np.var(b,axis=0,ddof=1)/len(b) for w,b in zip(weights,blocksets)))
    stats=[r['energy_stats'] for r in records]
    coeff=np.array([1.,10.])
    covariance=sum(w*w*np.asarray(s['covariance_jk_of_mean']) for w,s in zip(weights,stats))
    ej=sum(w*s['e_j'] for w,s in zip(weights,stats));ek=sum(w*s['e_k'] for w,s in zip(weights,stats))
    L=records[0]['L'];N=L*L
    q=2*np.pi*np.fft.fftfreq(L)
    def peak(key):
        index=np.unravel_index(np.argmax(arrays[key]),arrays[key].shape)
        return dict(index=[int(x) for x in index],q_over_pi=[float(q[x]/np.pi) for x in index],
                    value=float(arrays[key][index]),error=float(arrays[key+'_error'][index]))
    checks=dict(mean_nb_error=float(abs(arrays['nb'].mean()-.5)),
                mean_density_spectrum_error=float(abs(arrays['sn'].mean()-.25)),
                density_equal_site_error=float(abs(arrays['cn'][0,0]-.25)),
                fixed_number_structure_zero=float(abs(arrays['sn'][0,0])),
                pair_x_contact_error=float(abs(arrays['pair_x'].mean()-(.25+arrays['cn'][1,0]))),
                pair_y_contact_error=float(abs(arrays['pair_y'].mean()-(.25+arrays['cn'][0,1]))))
    if max(checks.values())>1e-9:raise RuntimeError('Fourier normalization or contact check failed: '+str(checks))
    summary=dict(L=L,N=N,r=records[0]['r'],g=records[0]['g'],seeds=[r['seed'] for r in records],
                 samples=int(sum(r['samples'] for r in records)),energy_K10=float(ej+10*ek),
                 energy_K10_error=float(np.sqrt(coeff@covariance@coeff)),e_j=float(ej),e_k=float(ek),
                 gradient_K10=float(sum(w*r['gradient_K10'] for w,r in zip(weights,records))),
                 gradient_K10_error=float(np.sqrt(sum(w*w*r['gradient_K10_error']**2 for w,r in zip(weights,records)))),
                 m_cdw_squared=float(arrays['sn'][L//2,L//2]/N),
                 m_cdw_squared_error=float(arrays['sn_error'][L//2,L//2]/N),
                 density_peak=peak('sn'),boson_peak=peak('nb'),pair_s_peak=peak('pair_s'),pair_d_peak=peak('pair_d'),
                 nb_range=[float(arrays['nb'].min()),float(arrays['nb'].max())],
                 pair_s_range=[float(arrays['pair_s'].min()),float(arrays['pair_s'].max())],
                 pair_d_range=[float(arrays['pair_d'].min()),float(arrays['pair_d'].max())],
                 pair_x_population=float(arrays['pair_x'].mean()),pair_y_population=float(arrays['pair_y'].mean()),
                 boson_offdiagonal_max=float(np.max(abs(arrays['gb'].ravel()[1:]))),
                 pair_noncontact_max=float(max(np.max(abs(arrays['pair'][0,0].ravel()[1:])),np.max(abs(arrays['pair'][1,1].ravel()[1:])),np.max(abs(arrays['pair'][0,1])),np.max(abs(arrays['pair'][1,0])))),
                 ring_correlator_real=float(ek/2),checks=checks)
    return summary,arrays


def bz_map(ax,values,L,**kwargs):
    q=np.fft.fftshift(np.fft.fftfreq(L))*2
    edges=np.r_[q-1/L,q[-1]+1/L,q[-1]+3/L]
    image=np.pad(np.fft.fftshift(values).T,((0,1),(0,1)),mode='wrap')
    im=ax.pcolormesh(edges,edges,image,shading='flat',rasterized=True,**kwargs)
    ax.set(xlim=(-1,1),ylim=(-1,1),aspect='equal',xlabel=r'$q_x/\pi$',ylabel=r'$q_y/\pi$')
    ax.set_xticks([-1,0,1]);ax.set_yticks([-1,0,1])
    return im


def make_plots(cases,scaling):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'savefig.facecolor':'white','svg.fonttype':'path'})
    names=['strong','finite_reference'];keys=['sn','nb','pair_s','pair_d']
    titles=[r'Density $S_n(\mathbf{q})$',r'Boson $n_b(\mathbf{q})$',r'Bond-pair $n_s(\mathbf{q})$',r'Bond-pair $n_d(\mathbf{q})$']
    fig,axes=plt.subplots(2,4,figsize=(15.7,8.1),layout='constrained')
    vmax=[max(float(cases[n][1][key].max()) for n in names) for key in keys]
    vmax[2]=vmax[3]=max(vmax[2],vmax[3])
    for row,name in enumerate(names):
        report,arrays=cases[name]
        for col,key in enumerate(keys):
            im=bz_map(axes[row,col],arrays[key],report['L'],cmap='magma',vmin=0,vmax=vmax[col])
            if row==0:axes[row,col].set_title(titles[col],fontsize=12)
            if col==0:axes[row,col].set_ylabel(f"r={report['r']:g}, g={report['g']:g}\n"+r'$q_y/\pi$')
            if row==1:fig.colorbar(im,ax=axes[:,col],shrink=.72,label=key.replace('_',' '))
    fig.suptitle('Projected VMC | 8 x 8, half filling, J=1, K=10\nTop: low-energy strong shell; bottom: finite-anisotropy reference',fontsize=14)
    fig.savefig(OUT/'projected_momentum_distributions.png',dpi=200)
    fig.savefig(OUT/'projected_momentum_distributions.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
    max_offsite=max(float(np.max(abs(np.delete(cases[n][1]['cn'].ravel(),0)))) for n in names)
    for col,name in enumerate(names):
        report,arrays=cases[name];L=report['L']
        cn=arrays['cn'].copy();cn[0,0]=np.nan
        im=axes[col].imshow(np.fft.fftshift(cn).T,origin='lower',
                           extent=[-L/2-.5,L/2-.5,-L/2-.5,L/2-.5],cmap='RdBu_r',vmin=-max_offsite,vmax=max_offsite)
        axes[col].set(title=f"Connected density C_n(R), r={report['r']:g}\nC_n(0)=1/4 (origin masked)",xlabel=r'$R_x$',ylabel=r'$R_y$')
        fig.colorbar(im,ax=axes[col],shrink=.8)
    for name,color in [('strong','#167CA6'),('finite_reference','#D0682E')]:
        report,arrays=cases[name];L=report['L'];distance=np.arange(1,L//2+1)
        axes[2].errorbar(distance,arrays['cn'][distance,0],yerr=arrays['cn_error'][distance,0],fmt='o-',color=color,label=f"r={report['r']:g}, x axis")
        axes[2].errorbar(distance,arrays['cn'][distance,distance],yerr=arrays['cn_error'][distance,distance],fmt='s--',color=color,label=f"r={report['r']:g}, diagonal")
    axes[2].axhline(0,c='#888888',lw=1);axes[2].set(xlabel='Separation m',ylabel=r'$C_n(\mathbf{R})$',title='Axis and diagonal cuts')
    axes[2].legend(fontsize=8);axes[2].grid(alpha=.2)
    fig.savefig(OUT/'projected_density_correlations.png',dpi=200);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for key,label in [('m2',r'$m_{\rm CDW}^2$'),('spi',r'$S_n(\pi,\pi)$')]:
        col=0 if key=='m2' else 1
        axes[col].errorbar([1/r['N'] for r in scaling],[r[key] for r in scaling],yerr=[r[key+'_error'] for r in scaling],fmt='o-',color='#167CA6')
        for r in scaling:axes[col].annotate(f"L={r['L']}",(1/r['N'],r[key]),xytext=(5,6),textcoords='offset points',fontsize=9)
        axes[col].set(xlabel='1/N',ylabel=label,title='Matched g=-0.11; no infinite-size fit assumed');axes[col].grid(alpha=.2)
    fig.savefig(OUT/'projected_density_scaling.png',dpi=190);plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    records=json.loads((SOURCE/'correlation_replicas.json').read_text())
    cases={}
    for name,r in [('strong',.03),('finite_reference',.33)]:
        subset=[v for v in records if v['L']==8 and abs(v['r']-r)<1e-8]
        if len(subset)!=2:raise RuntimeError('Two completed L8 replicas required for '+name)
        cases[name]=combine(subset)
        np.savez_compressed(OUT/f'combined_L8_{name}.npz',**cases[name][1])
    exact=np.load(OUT/'exact4_correlations.npz')
    scaling=[dict(L=4,N=16,m2=float(exact['strong_sn'][2,2].real/16),m2_error=0.,spi=float(exact['strong_sn'][2,2].real),spi_error=0.)]
    l6=[v for v in records if v['L']==6 and abs(v['r']-.03)<1e-8]
    if not l6:raise RuntimeError('Matched-g L6 result is still pending')
    report6,array6=combine(l6)
    np.savez_compressed(OUT/'combined_L6_strong.npz',**array6)
    for r in (report6,cases['strong'][0]):
        scaling.append(dict(L=r['L'],N=r['N'],m2=r['m_cdw_squared'],m2_error=r['m_cdw_squared_error'],
                            spi=r['m_cdw_squared']*r['N'],spi_error=r['m_cdw_squared_error']*r['N']))
    summaries={name:pair[0] for name,pair in cases.items()}
    result=dict(j=1,k=10,half_filling=.5,pair_definition='Delta_x(i)=b_i b_(i+x), Delta_y(i)=b_i b_(i+y); s,d=(x +/- y)/sqrt(2); Fourier about site anchors',
                cases=summaries,size_scaling=scaling,estimator_validation='Exact 4x4 full operator enumeration and weighted local estimators agree within 3e-14, including complex finite-anisotropy state.')
    (OUT/'correlation_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    rows=[]
    for name,(r,a) in cases.items():
        L=r['L'];q=2*np.pi*np.fft.fftfreq(L)
        for ix in range(L):
            for iy in range(L):
                row=dict(case=name,qx_over_pi=q[ix]/np.pi,qy_over_pi=q[iy]/np.pi)
                for key in ('sn','nb','pair_x','pair_y','pair_s','pair_d'):
                    row[key]=float(a[key][ix,iy]);row[key+'_error']=float(a[key+'_error'][ix,iy])
                rows.append(row)
    with (OUT/'momentum_distributions.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    make_plots(cases,scaling)
    s=summaries['strong'];ref=summaries['finite_reference']
    lines=['# 投影 VMC：密度关联与有效配对动量分布','',
           '本次实际测量 J=1、K=10、半填充、周期物理边界的投影波函数 Psi=det U1 det U2 exp(-g Q_nn)。内部扭转相反，总物理玻色扭转为零。强壳层用 r=.03，有限各向异性参考用 r=.33，均取驻点附近 g=-.11。后者是对照态，没有另行声称其变分参数已最优。','',
           '## 算符定义','',
           r'$$C_n(R)=N^{-1}\sum_i\langle(n_i-1/2)(n_{i+R}-1/2)\rangle,\qquad S_n(q)=\sum_R e^{-iqR}C_n(R).$$','',
           r'$$G_b(R)=N^{-1}\sum_i\langle b_i^\dagger b_{i+R}\rangle,\qquad n_b(q)=\sum_R e^{-iqR}G_b(R).$$','',
           r'$$\Delta_x(i)=b_i b_{i+x},\quad\Delta_y(i)=b_i b_{i+y},\quad\Delta_{s/d}(i)=[\Delta_x(i)\pm\Delta_y(i)]/\sqrt2.$$','',
           r'$$P_{ab}(R)=N^{-1}\sum_i\langle\Delta_a^\dagger(i)\Delta_b(i+R)\rangle,\qquad n_{s/d}(q)=\langle\Delta_{s/d}^\dagger(q)\Delta_{s/d}(q)\rangle.$$','',
           'Fourier 使用格点锚点 i，不额外加入半键位移相位。s/d 在本文指上述 x/y 正方向键场的加减组合。固定粒子数态中 <b> 与 <Delta> 为零，所画的是关联定义的动量密度，并不是非零的平均场异常期望值。若 b 表示原始费米子对，n_b 本身也是原费米子对的动量分布。','',
           '## 8×8 独立链合并结果','',
           '| 量 | 强各向异性 r=.03 | 有限各向异性参考 r=.33 |','|---|---:|---:|',
           f"| 每格点能量 | {s['energy_K10']:.6f} ± {s['energy_K10_error']:.6f} | {ref['energy_K10']:.6f} ± {ref['energy_K10_error']:.6f} |",
           f"| m_CDW² | {s['m_cdw_squared']:.6f} ± {s['m_cdw_squared_error']:.6f} | {ref['m_cdw_squared']:.6f} ± {ref['m_cdw_squared_error']:.6f} |",
           f"| n_b(q) 范围 | {s['nb_range'][0]:.6f}—{s['nb_range'][1]:.6f} | {ref['nb_range'][0]:.6f}—{ref['nb_range'][1]:.6f} |",
           f"| n_s(q) 范围 | {s['pair_s_range'][0]:.6f}—{s['pair_s_range'][1]:.6f} | {ref['pair_s_range'][0]:.6f}—{ref['pair_s_range'][1]:.6f} |",
           f"| n_d(q) 范围 | {s['pair_d_range'][0]:.6f}—{s['pair_d_range'][1]:.6f} | {ref['pair_d_range'][0]:.6f}—{ref['pair_d_range'][1]:.6f} |",'',
           f"强壳层的实部环关联 Re<b1† b3† b2 b4>={s['ring_correlator_real']:.6f}，仍显著为负。有限 r 参考在 K=10 的能量也低于相位优化二聚体参考 -2.5125，但高于强壳层；这不构成完整相图或稳定 CPBM 证明。",'',
           '## 密度序诊断','',
           '| L | m_CDW²=S_n(pi,pi)/N | S_n(pi,pi) |','|---:|---:|---:|']
    for r in scaling:lines.append(f"| {r['L']} | {r['m2']:.6f} ± {r['m2_error']:.6f} | {r['spi']:.6f} ± {r['spi_error']:.6f} |")
    lines+=['','密度谱在 (pi,pi) 有峰（第一 BZ 的角点与其等价）。该峰说明棋盘格密度关联明显；m_CDW² 随尺寸下降，4、6、8 三个尺寸不足以判断热力学极限是否仍非零，没有把均匀平均密度当作排除 CDW 的证据。','',
            '## 为什么低能态的两个配对动量图都是平的','',
            '强壳层严格固定每行和每列粒子数。单粒子在不同格点移动会改变至少一个行/列粒子数，因此 G_b(R)=nu delta_R0，n_b(q)=1/2。最近邻两玻色对的不同键之间移动也改变行/列端点计数，故 P_xx 与 P_yy 仅保留同键接触项，P_xy=0。于是 n_s=n_d=(P_x+P_y)/2，所有动量完全相同。这个平坦结果由态的支撑集决定，VMC 验证它而不是人为设置它。','',
            '它说明该强壳层 ansatz 没有单玻色子或最近邻双玻色对的非对角长程序；有限局域配对概率不等于配对凝聚。对角线玻色对绕方格转向仍保存行列计数，因此负环关联与这些平坦动量图并不矛盾。有限 r=.33 参考的支撑集更宽，动量分布有可分辨结构，但有限峰值仍不能证明凝聚或 CPBM。','',
            '强壳层还应满足 S_n(qx,0)=S_n(0,qy)=0。全部图使用离散允许动量，不以插值制造 Bose surface 或尖峰。投影后 n_b(q) 可以超过 1，不施加此前未投影近似的上界。有限参考态的峰位依赖此小尺寸下的内部边界扭转，不能把偏离零动量的最大采样格点直接解释成有限动量凝聚或 PDW。','',
            '## 精度、驻点与复现','',
            '强壳层每个 L8 种子：1024 热化 sweep、4096 次测量、间隔 3 sweep；两个独立种子合并。参考态每种子 768 热化、2048 测量、间隔 3 sweep。匹配 g 的 L6 为独立链。误差由每条链 20 个连续分块得到，并在动量空间直接对块做 Fourier，再合并独立链；没有将各实空间误差误当作独立传播。','',
            f"强壳层 g=-.11 的能量梯度为 {s['gradient_K10']:.6f} ± {s['gradient_K10_error']:.6f}，此前 g=-.10 独立长链为 +.141110 ± .047237。两端导数变号，把投影变分驻点进一步夹在约 [-.11,-.10]；不把任何端点冒称精确驻点，也不从闭壳层平台推断唯一 hopping 比。",'',
            '归一化检查：C_n(0)=1/4、平均 S_n=1/4、S_n(0)=0、平均 n_b=1/2，以及平均 n_pair_x/y 等于对应最近邻占据概率。精确 4×4 全物理算符枚举与加权局域估计在包含复振幅的 r=.33 状态上相符至 3e-14，覆盖不相交、共享一个格点和相同键的两体操作。','',
            '```powershell','python projected_exact_correlations.py','python run_projected_correlations.py','python analyze_projected_correlations.py','python -m unittest test_projected_correlations -v','```','',
            '输出：projected_momentum_distributions.png（密度、单玻色、s/d 配对动量图）、projected_density_correlations.png（实空间关联）、projected_density_scaling.png（尺寸诊断）、momentum_distributions.csv、correlation_summary.json、combined_*.npz。原始分块数组与能量样本保存在 output/projected-jk/correlations_*.npz。']
    (OUT/'correlation_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
