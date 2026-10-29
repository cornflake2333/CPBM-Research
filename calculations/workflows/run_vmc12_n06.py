"""L12 Nb86 variational search, explicit CDW competitors, and pair observables."""

from vmc_paths import PROJECT_ROOT
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,time,csv,zipfile
import projected_parton_exact as exact
import projected_parton_vmc as vmc
import projected_correlations as corr
import projected_pair_channels as pc
from optimize_vmc12 import ratios_summary
import numpy as np

ROOT=PROJECT_ROOT;OUT=ROOT/'output/vmc12-n06'
L=12;N=144;M=86;NU=M/N
GG=np.linspace(-.30,.10,17)
CHANNELS=list(pc.CHANNEL_NAMES)

def checkpoint(path,obj):path.write_text(json.dumps(obj,indent=2),encoding='utf-8')

def scan_worker(job):
    r,h,seed,g0,samples=job
    stem=f'scan_N{M}_r{r:g}_h{h:g}_seed{seed}';jp=OUT/(stem+'.json')
    if jp.exists():
        rec=json.loads(jp.read_text());assert rec['samples']==samples and rec['g0']==g0;return rec
    start=time.perf_counter();u,v,meta=exact.orbitals(L,r,particles=M)
    state=vmc.ProjectedState(u,v,L,g0,h=h,seed=seed);acc=att=0
    for _ in range(384*N):acc+=state.proposal();att+=1
    q=np.empty(samples);energies=np.empty((samples,len(GG)),complex)
    for m in range(samples):
        for _ in range(3*N):acc+=state.proposal();att+=1
        q[m]=state.q_nn()
        for k,g in enumerate(GG):
            state.g=float(g);ej,ek,_,_=state.local_energy();energies[m,k]=ej+10*ek
        state.g=g0
    rec=dict(r=r,h=h,seed=seed,L=L,Nb=M,filling=NU,g0=g0,samples=samples,warmup=384,stride=3,acceptance=acc/att,seconds=time.perf_counter()-start,metadata=meta,grid=ratios_summary(q,energies,GG,g0))
    np.savez_compressed(OUT/(stem+'.npz'),Q_nn=q,energies=energies,g_grid=GG,g0=g0)
    checkpoint(jp,rec);return rec

def combine_scan(reps):
    rows=[]
    for i,g in enumerate(GG):
        v=[z['grid'][i] for z in reps];nr=len(v)
        rows.append(dict(g=float(g),energy=float(np.mean([z['energy'] for z in v])),error=float(np.linalg.norm([z['error'] for z in v])/nr),gradient=float(np.mean([z['gradient'] for z in v])),gradient_error=float(np.linalg.norm([z['gradient_error'] for z in v])/nr),min_ess_fraction=min(z['ess_fraction'] for z in v)))
    valid=[z for z in rows if z['min_ess_fraction']>=.25]
    best=min(valid,key=lambda z:z['energy']);proposal=best['g'];brackets=[]
    for a,b in zip(valid[:-1],valid[1:]):
        if a['gradient']<=0<=b['gradient'] and abs(a['g']-best['g'])<=.075:
            root=a['g']-a['gradient']*(b['g']-a['g'])/(b['gradient']-a['gradient'])
            brackets.append((abs(root-best['g']),root))
    if brackets:proposal=float(min(brackets)[1])
    return dict(r=reps[0]['r'],h=reps[0]['h'],grid=rows,best=best,proposed_g=proposal,gradient_root_found=bool(brackets),replicas=reps)

def run_scan_jobs(jobs):
    recs=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(scan_worker,j) for j in jobs]):
            r=f.result();recs.append(r);b=min((z for z in r['grid'] if z['ess_fraction']>=.25),key=lambda z:z['energy'])
            print('SCAN_DONE',r['r'],r['h'],r['seed'],'time',round(r['seconds'],1),'best',json.dumps(b),flush=True)
    return recs

def search():
    # Enumerate this specified grid, and remove identical occupied orbital sets.
    grid=[0.,.03,.08,.15,.20,.25,.33,.40,.50,.60,.75,.90,1.]
    states=[];keys=set();aliases=[]
    for r in grid:
        _,_,meta=exact.orbitals(L,r,particles=M);key=tuple(tuple(x) for x in meta['occupied'])
        if key in keys:aliases.append(r);continue
        keys.add(key);states.append(r)
    print('DISTINCT_SEAS',states,'duplicate_grid_points',aliases,flush=True)
    reps=run_scan_jobs([(r,0.,seed,-.10,768) for r in states for seed in (4101,4102)])
    uniform=[combine_scan([z for z in reps if z['r']==r]) for r in states]
    uniform.sort(key=lambda z:z['best']['energy']);best=uniform[0]
    checkpoint(OUT/'uniform_search.json',dict(Nb=M,filling=NU,grid=grid,duplicate_points=aliases,records=uniform))
    # Optimize g again at each explicit staggered density factor h.
    # Psi_h = Psi_0 exp[h sum_i (-1)^(x+y) n_i]; h is a wavefunction parameter.
    reps=run_scan_jobs([(best['r'],h,seed,best['proposed_g'],768) for h in (.10,.20,.40) for seed in (4201,4202)])
    cdw=[combine_scan([z for z in reps if z['h']==h]) for h in (.10,.20,.40)]
    cdw.sort(key=lambda z:z['best']['energy'])
    # The next genuinely different finite-r sea also needs fresh-chain checks.
    runner=next(z for z in uniform if z['r']>=.20 and z['r']!=best['r'])
    selected=[best,cdw[0],runner]
    result=dict(L=L,Nb=M,requested_filling=.6,actual_filling=NU,J=1,K=10,uniform=uniform,cdw=cdw,selected=selected)
    checkpoint(OUT/'optimization.json',result)
    print('SELECTED',json.dumps([dict(r=s['r'],h=s['h'],g=s['proposed_g'],energy=s['best']['energy']) for s in selected]),flush=True)
    return result

def one_body(state):
    geo=corr.geometry(L);source,target=geo['source'],geo['target']
    values=np.zeros(N*N,complex);same=source==target;values[same]=state.occ[source[same]]
    valid=(~same)&(state.occ[source]==1)&(state.occ[target]==0)
    transformed=np.stack([state.U[a]@state.inverse[a] for a in range(2)])
    values[valid]=corr._move_values(state,source[valid,None],target[valid,None],transformed)
    gb=values.reshape(N,N).mean(axis=1).reshape(L,L);rev=(-np.arange(L))%L
    gb=(gb+gb[np.ix_(rev,rev)].conj())/2
    return np.fft.fft2(gb).real

def final_worker(job):
    r,g,h,seed=job;stem=f'final_N{M}_r{r:g}_g{g:.8f}_h{h:g}_seed{seed}'
    jp=OUT/(stem+'.json');ap=OUT/(stem+'.npz');samples=3072
    if jp.exists() and ap.exists():return json.loads(jp.read_text())
    u,v,meta=exact.orbitals(L,r,particles=M);state=vmc.ProjectedState(u,v,L,g,h=h,seed=seed)
    start=time.perf_counter();acc=att=0
    for _ in range(1024*N):acc+=state.proposal();att+=1
    pair=np.empty((samples,4,4,L,L),complex);sn=np.empty((samples,L,L));nb=np.empty_like(sn);raw=np.empty((samples,7))
    for m in range(samples):
        for _ in range(3*N):acc+=state.proposal();att+=1
        pair[m]=pc.observe(state,full_matrix=False);nb[m]=one_body(state)
        nq=np.fft.fft2(state.occ.reshape(L,L)-NU);sn[m]=abs(nq)**2/N
        ej,ek,q,sp=state.local_energy();st=float(state.stagger@state.occ)
        raw[m]=[ej.real,ek.real,ej.imag,ek.imag,q,sp,st]
        if (m+1)%512==0:print('FINAL_PROGRESS',r,h,seed,m+1,samples,round(time.perf_counter()-start,1),flush=True)
    arrays=dict(raw_energy_samples=raw)
    for name,values in [('pair',pair),('sn',sn),('nb',nb)]:
        arrays[name]=values.mean(axis=0);arrays['blocks_'+name]=np.stack([a.mean(axis=0) for a in np.array_split(values,24)])
    np.savez_compressed(ap,**arrays)
    energy=raw[:,0]+10*raw[:,1];em=np.array([z.mean() for z in np.array_split(energy,24)])
    def covgrad(o,factor):return factor*(np.mean(energy*o)-energy.mean()*o.mean())
    grads=[covgrad(raw[:,4],-2),covgrad(raw[:,6],2)];jk=[]
    for ids in np.array_split(np.arange(samples),24):
        a=np.delete(raw,ids,axis=0);e=a[:,0]+10*a[:,1]
        jk.append([-2*(np.mean(e*a[:,4])-e.mean()*a[:,4].mean()),2*(np.mean(e*a[:,6])-e.mean()*a[:,6].mean())])
    jk=np.array(jk);ge=np.sqrt(23/24*np.sum((jk-jk.mean(axis=0))**2,axis=0))
    rec=dict(L=L,Nb=M,filling=NU,J=1,K=10,r=r,g=g,h=h,seed=seed,samples=samples,warmup=1024,stride=3,seconds=time.perf_counter()-start,acceptance=acc/att,energy=float(energy.mean()),error=float(em.std(ddof=1)/np.sqrt(24)),grad_g=float(grads[0]),grad_g_error=float(ge[0]),grad_h=float(grads[1]),grad_h_error=float(ge[1]),staggered_density_mean=float(raw[:,6].mean()/N),imaginary_energy=float(np.mean(raw[:,2]+10*raw[:,3])),metadata=meta,array_file=str(ap.relative_to(ROOT)))
    checkpoint(jp,rec);return rec

def finals(opt):
    jobs=[(s['r'],s['proposed_g'],s['h'],seed) for s in opt['selected'] for seed in (5101,5102)]
    reps=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(final_worker,j) for j in jobs]):
            rec=f.result();reps.append(rec);print('FINAL_DONE',json.dumps({k:rec[k] for k in ['r','h','g','seed','energy','error','grad_g','grad_g_error']}),flush=True)
            checkpoint(OUT/'final_checkpoint.json',reps)
    return reps

def analyze(opt,reps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    q=2*np.pi*np.fft.fftfreq(L)/np.pi;qs=np.fft.fftshift(q);summaries=[];data=[]
    for candidate in opt['selected']:
        rs=[r for r in reps if r['h']==candidate['h'] and r['r']==candidate['r']]
        arr=[dict(np.load(ROOT/r['array_file'])) for r in rs]
        meanpair=np.mean([a['pair'] for a in arr],axis=0);spectra=pc.fourier(meanpair)
        blocks=[pc.fourier(a['blocks_pair']) for a in arr]
        for key in CHANNELS+['nonlocal_'+c for c in CHANNELS]:
            spectra[key+'_error']=np.sqrt(sum(x[key].var(axis=0,ddof=1)/24 for x in blocks))/2
        for key in ['sn','nb']:
            spectra[key]=np.mean([a[key] for a in arr],axis=0)
            spectra[key+'_error']=np.sqrt(sum(a['blocks_'+key].var(axis=0,ddof=1)/24 for a in arr))/2
        spectra['pair']=meanpair
        # Number, hard-core variance, one-body density, and identical-bond sums.
        checks=dict(number=float(abs(spectra['sn'][0,0])),density_variance=float(abs(spectra['sn'].mean()-NU*(1-NU))),boson_sum=float(abs(spectra['nb'].mean()-NU)))
        assert max(checks.values())<1e-8
        peak=np.unravel_index(np.argmax(spectra['sn']),(L,L))
        summary=dict(L=L,Nb=M,filling=NU,r=candidate['r'],h=candidate['h'],g=candidate['proposed_g'],samples=sum(z['samples'] for z in rs),energy=float(np.mean([z['energy'] for z in rs])),error=float(np.linalg.norm([z['error'] for z in rs])/2),grad_g=float(np.mean([z['grad_g'] for z in rs])),grad_g_error=float(np.linalg.norm([z['grad_g_error'] for z in rs])/2),grad_h=float(np.mean([z['grad_h'] for z in rs])),grad_h_error=float(np.linalg.norm([z['grad_h_error'] for z in rs])/2),replicas=rs,checks=checks,m_cdw_squared=float(spectra['sn'][6,6]/N),m_cdw_squared_error=float(spectra['sn_error'][6,6]/N),density_peak_q_over_pi=[float(q[i]) for i in peak],density_peak=float(spectra['sn'][peak]),density_peak_error=float(spectra['sn_error'][peak]),nb_range=[float(spectra['nb'].min()),float(spectra['nb'].max())],channels={})
        summary['g_stationary_within_2sigma']=bool(abs(summary['grad_g'])<=2*summary['grad_g_error'])
        summary['replica_energy_difference_sigma']=float(abs(rs[0]['energy']-rs[1]['energy'])/np.hypot(rs[0]['error'],rs[1]['error']))
        e12=[];cdw12=[]
        for a in arr:
            raw=a['raw_energy_samples'];ev=raw[:,0]+10*raw[:,1]
            e12.append(np.array([x.mean() for x in np.array_split(ev,12)]).std(ddof=1)/np.sqrt(12))
            cdw12.append(a['blocks_sn'].reshape(12,2,L,L).mean(axis=1)[:,6,6].std(ddof=1)/np.sqrt(12)/N)
        summary['block_check_12_vs_24']=dict(energy_error_12=float(np.linalg.norm(e12)/2),energy_error_24=summary['error'],m_cdw_error_12=float(np.linalg.norm(cdw12)/2),m_cdw_error_24=summary['m_cdw_squared_error'])
        for c in CHANNELS:
            idx=np.unravel_index(np.argmax(spectra[c]),(L,L))
            summary['channels'][c]=dict(peak=float(spectra[c][idx]),error=float(spectra[c+'_error'][idx]),q_over_pi=[float(q[i]) for i in idx],nonlocal_range=[float(spectra['nonlocal_'+c].min()),float(spectra['nonlocal_'+c].max())])
        name=f'combined_r{candidate["r"]:g}_h{candidate["h"]:g}'
        np.savez_compressed(OUT/(name+'.npz'),**spectra);data.append(spectra);summaries.append(summary)
    checkpoint(OUT/'summary.json',summaries)
    with (OUT/'momentum_distributions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['r','g','h','qx_over_pi','qy_over_pi','Sn','Sn_error','nb','nb_error']+sum(([c,c+'_error','nonlocal_'+c] for c in CHANNELS),[]))
        for s,z in zip(summaries,data):
            for i in range(L):
                for j in range(L):w.writerow([s['r'],s['g'],s['h'],q[i],q[j],z['sn'][i,j],z['sn_error'][i,j],z['nb'][i,j],z['nb_error'][i,j]]+sum(([z[c][i,j],z[c+'_error'][i,j],z['nonlocal_'+c][i,j]] for c in CHANNELS),[]))
    def maps(keys,labels,filename,signed=False):
        fig,axs=plt.subplots(len(data),len(keys),figsize=(3.6*len(keys),3.5*len(data)),constrained_layout=True,squeeze=False)
        for col,(key,label) in enumerate(zip(keys,labels)):
            vmax=max(float(abs(a[key]).max()) for a in data);vmin=-vmax if signed else 0
            for row,(s,z) in enumerate(zip(summaries,data)):
                im=axs[row,col].pcolormesh(qs,qs,np.fft.fftshift(z[key]).T,shading='nearest',cmap='RdBu_r' if signed else 'viridis',vmin=vmin,vmax=max(vmax,1e-12))
                axs[row,col].set(title=f'{label}: r={s["r"]}, h={s["h"]}\ng={s["g"]:.4f}',xlabel=r'$q_x/\pi$',ylabel=r'$q_y/\pi$');fig.colorbar(im,ax=axs[row,col],shrink=.8)
        fig.suptitle(f'12 x 12 projected VMC: Nb={M}, density={NU:.6f}, J=1, K=10')
        fig.savefig(OUT/(filename+'.png'),dpi=180);fig.savefig(OUT/(filename+'.svg'));plt.close(fig)
    maps(['sn','nb'],['Density structure factor','Boson momentum density'],'density_and_bosons')
    maps(CHANNELS,['Nearest s','d x2-y2','Diagonal s','d xy'],'pair_channels_full')
    maps(['nonlocal_'+c for c in CHANNELS],['Nearest s','d x2-y2','Diagonal s','d xy'],'pair_channels_nonlocal',True)
    bestidx=int(np.argmin([s['energy'] for s in summaries]));best=summaries[bestidx];z=data[bestidx]
    fig,axs=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
    for ax,key,label in zip(axs,['sn','nb','d_xy'],['Density structure factor','Boson momentum density','d xy pair momentum density']):
        im=ax.pcolormesh(qs,qs,np.fft.fftshift(z[key]).T,shading='nearest',cmap='viridis')
        ax.set(title=label,xlabel=r'$q_x/\pi$',ylabel=r'$q_y/\pi$');fig.colorbar(im,ax=ax,shrink=.85)
    fig.suptitle(f'Lowest validated candidate: Nb={M}, r={best["r"]}, g={best["g"]:.4f}, h={best["h"]}')
    fig.savefig(OUT/'best_state_momentum.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for rec in opt['uniform']:
        vv=[a for a in rec['grid'] if a['min_ess_fraction']>=.25]
        axs[0].errorbar([v['g'] for v in vv],[v['energy'] for v in vv],yerr=[v['error'] for v in vv],marker='.',label=f'r={rec["r"]}')
    for rec in [opt['uniform'][0]]+sorted(opt['cdw'],key=lambda z:z['h']):
        vv=[a for a in rec['grid'] if a['min_ess_fraction']>=.25]
        axs[1].errorbar([v['g'] for v in vv],[v['energy'] for v in vv],yerr=[v['error'] for v in vv],marker='.',label=f'h={rec["h"]}')
    for ax in axs:ax.set(xlabel='NN Jastrow g',ylabel='Energy per site');ax.grid(alpha=.2);ax.legend(fontsize=8)
    axs[0].set_title('Uniform-parent orbital search');axs[1].set_title('Explicit checkerboard density factors')
    fig.savefig(OUT/'variational_search.png',dpi=180);plt.close(fig)
    report=['# 12×12、目标填充0.6的投影VMC','',f'固定粒子数Nb={M}，实际填充{NU:.8f}；0.6×144=86.4非整数。本次取最近整数86，未将其写作精确0.6。J=1、K=10、硬核玻色子、物理PBC、零物理通量。','',
    'Psi=det U1 det U2 exp[-g Qnn+h Mpi]，Mpi=sum_i(-1)^(x+y)n_i。严格Gutzwiller约束保持；g为NN Jastrow参数，h为波函数中的交错密度参数，均不是给物理Hamiltonian额外增加外场。两个parton使用交叉跃迁(1,r)/(r,1)，内部相反扭转(.37,.19)pi。','',
    'r网格=0,.03,.08,.15,.20,.25,.33,.40,.50,.60,.75,.90,1，按占据轨道去重后逐态扫描g∈[-.30,.10]的17点。每态2个独立参考种子、各384 sweep热化/768测量/间隔3 sweep。目标g的局域能量逐点重算，以exp[-2(g-g0)Qnn]重加权；权重ESS至少25%才用于选点。20块删除jackknife给能量与梯度误差，协方差梯度零点提出g候选。此网格并未穷尽全部轨道海或其他波函数族。','',
    '对最低能均匀母态另扫描h=.10,.20,.40，并在每个h重新优化g。只在这个r上检查此类CDW竞争态，不是对所有密度波形或所有Q的全局优化。全密度结构因子用于观察其他Q的关联，但单尺寸不能证明/排除IDW、CDW或相分离。','',
    '最终每个候选2条全新链，每链1024 sweep热化、3072测量、间隔3 sweep，24连续块合并误差。每sweep=144次提议。原始能量、Qnn、交错密度与空间分块保留。','',
    '|r|g|h|能量/格点|g梯度|mCDW²|','|---|---|---|---|---|---|']
    for s in summaries:report.append(f'|{s["r"]}|{s["g"]:.5f}|{s["h"]}|{s["energy"]:.6f} ± {s["error"]:.6f}|{s["grad_g"]:.4f} ± {s["grad_g_error"]:.4f}|{s["m_cdw_squared"]:.6f} ± {s["m_cdw_squared_error"]:.6f}|')
    report+=['','Cn以实际平均密度nu为中心；Sn(0)=0，BZ平均Sn=nu(1-nu)，BZ平均nb=nu，逐候选验证。mCDW²=Sn(pi,pi)/N；h非零态的结构因子包含有意引入的静态密度调制，不能把它直接当自发序。','',
    '独立验证后若g梯度偏离零超过2个标准误，用初扫梯度斜率估计曲率作有界Newton修正，并对新g重新采样；每个候选至多修正两次。summary.json明确记录最终g梯度是否在2σ内与零相容，并检查两链差异及12/24分块误差。g小数位不是优化参数的统计精度；r也可能代表占据轨道平台。','',
    'h=.20/.40的部分g扫描最低点或可靠ESS区间接近边界，不能称这些CDW分支已全局优化，也不能用其当前较高能量排除其他CDW态。r=0在这个粒子数处有部分填充的简并单粒子能级，其轨道选择也有任意性。本次仅比较所列变分参数与固定内部扭转。','',
    '四键配对定义：NN s/d=(b_i b_(i+x)+b_i b_(i-x) ± b_i b_(i+y) ± b_i b_(i-y))/2；对角s/dxy同理将x,y替换为x+y,x-y。完整n_pair为配对场关联密度；nonlocal只扣同键接触项，可为负，不等价于全连通vertex。未使用Wick分解。半填充强壳层的固定行列粒子数结论未套用本次填充。','',
    '投影VMC给出变分能量上界，不是12×12精确基态，不单凭能量/配对峰宣称CPBM。单填充不能计算压缩率或排除相分离；粒子空穴对称将86/144对应58/144，但本次未混合这两个固定粒子数扇区。','',
    '验证：test_vmc_filling测试非半填充M=6/10的完整行列式比值、NN/环局域能量，覆盖非零h；此前VMC与配对通道测试全部通过。','',
    '输出optimization.json、uniform_search.json、summary.json、momentum_distributions.csv、combined*.npz、原始scan/final*.npz以及四幅PNG/SVG。复现运行run_vmc12_n06.py；保留缓存将跳过已经完成的采样。']
    (OUT/'report.md').write_text('\n'.join(report),encoding='utf-8')
    print('RESULTS',json.dumps([{k:s[k] for k in ['r','g','h','energy','error','grad_g','grad_g_error','m_cdw_squared','density_peak_q_over_pi']} for s in summaries]),flush=True)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    opt=json.loads((OUT/'optimization.json').read_text()) if (OUT/'optimization.json').exists() else search()
    if len(opt['selected'])==2:
        runner=next(z for z in opt['uniform'] if z['r']>=.20 and z['r']!=opt['selected'][0]['r'])
        opt['selected'].append(runner)
    # Refine a candidate if a fresh validation rejects g stationarity.
    # The scan covariance slope supplies a positive local curvature estimate.
    if (OUT/'summary.json').exists():
        saved=json.loads((OUT/'summary.json').read_text())
        for candidate in opt['selected']:
            match=[s for s in saved if s['r']==candidate['r'] and s['h']==candidate['h'] and abs(s['g']-candidate['proposed_g'])<1e-9]
            history=candidate.setdefault('gradient_refinements',[])
            if match and len(history)<2:
                s=match[0]
                if abs(s['grad_g'])>2*s['grad_g_error']:
                    local=[a for a in candidate['grid'] if abs(a['g']-s['g'])<.055 and a['min_ess_fraction']>=.25]
                    if len(local)>=3:
                        curv=float(np.polyfit([a['g'] for a in local],[a['gradient'] for a in local],1)[0])
                        if curv>0:
                            newg=float(np.clip(s['g']-s['grad_g']/curv,s['g']-.03,s['g']+.03))
                            history.append(dict(previous_g=s['g'],gradient=s['grad_g'],error=s['grad_g_error'],curvature=curv,new_g=newg))
                            candidate['proposed_g']=newg
                            print('REFINE_G',candidate['r'],candidate['h'],history[-1],flush=True)
    checkpoint(OUT/'optimization.json',opt)
    reps=finals(opt);analyze(opt,reps)
    saved=json.loads((OUT/'summary.json').read_text())
    for candidate in opt['selected']:
        s=next(z for z in saved if z['r']==candidate['r'] and z['h']==candidate['h'])
        if abs(s['grad_g'])>2*s['grad_g_error'] and len(candidate.get('gradient_refinements',[]))<2:
            local=[a for a in candidate['grid'] if abs(a['g']-s['g'])<.055 and a['min_ess_fraction']>=.25]
            if len(local)>=3 and np.polyfit([a['g'] for a in local],[a['gradient'] for a in local],1)[0]>0:
                print('VALIDATION_REQUIRES_G_REFINEMENT',candidate['r'],candidate['h'],flush=True)
                return main()
    paths=list(OUT.glob('*'))+list(ROOT.rglob('*.py'))+[ROOT/'README.md',ROOT/'requirements.txt']
    with zipfile.ZipFile(ROOT/'output/vmc12-n06-results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            if p.is_file():z.write(p,p.relative_to(ROOT))
    print('FINISHED',flush=True)

if __name__=='__main__':main()
