# 投影 Parton VMC：采样、计算与运行入口

本目录整理自 Bose Metal 研究脚本（2026-10-29）。原工作区脚本和历史结果保留原位；此处按职责拆分代码，使用本目录自己的 `output/`。物理算法、模型符号、随机种子和默认扫描参数保持原设定。

## 目录职责

```text
vmc-scripts-2026-10-29/
├── sampling/                  # 波函数状态、Metropolis更新、通用测量采样循环
│   ├── projected_parton_vmc.py
│   ├── correlation_sampler.py
│   └── current_sampler.py
├── calculations/              # 轨道、局域能量、统计、关联算符、精确基准与绘图
│   ├── jk_energy.py
│   ├── vmc_statistics.py
│   ├── projected_parton_exact.py
│   ├── projected_correlations.py
│   ├── projected_pair_channels.py
│   ├── projected_currents.py
│   └── workflows/             # 参数搜索、任务专用测量、独立链验证和输出调度
├── runners/                   # 轻量运行入口：转发到统一启动器
├── tests/                     # 单元测试与VMC/精确枚举对照验证
├── run.py                     # 统一启动器
├── vmc_paths.py               # 归档根目录定义
├── tasks.json                 # 任务名到实现文件的映射
├── requirements.txt           # numpy、matplotlib
├── manifest.json              # 当前文件路径及SHA256
├── manifest_original.json     # 拆分前归档记录
└── output/                    # 运行后生成；归档不附带历史数据
```

### 采样与计算的边界

`sampling/projected_parton_vmc.py` 保存投影波函数状态、行列式比值、逆矩阵更新和对称 Metropolis 提议。局域能量已抽到 `calculations/jk_energy.py`，统计已抽到 `calculations/vmc_statistics.py`。密度/配对和流关联的通用采样循环分别抽到 `sampling/correlation_sampler.py`、`sampling/current_sampler.py`；算符估计和 Fourier 变换留在 `calculations/`。旧函数名作为兼容转发保留，避免破坏依赖。

批量工作流仍保留任务专用的采样 worker、参数网格和保存流程，例如非半填充联合扫描与四通道测量；这些代码集中在 `calculations/workflows/`。`runners/` 只负责启动，不复制物理算法。此整理未将所有历史工作流改写成一个新的通用框架。

## 环境安装

推荐 Python 3.10 或更高版本。以下命令在本归档根目录执行：

```powershell
python -m pip install -r requirements.txt
python run.py --list
python run.py check-imports
python run.py test
python run.py smoke_layout
```

启动器默认将 BLAS/OMP 线程数设为1（用户已有设置优先），配置模块路径，并将当前工作目录定位到归档根目录。工作流中的并行采样通常使用2个进程；Windows 多进程入口均由 `if __name__ == '__main__'` 保护。

也可以从其他目录传入 `run.py` 的完整路径，或直接执行对应的 `runners/任务名.py`。不要直接执行 `calculations/` 或 `sampling/` 内的实现文件；它们需要启动器配置导入路径。

## 当前推荐流程：12×12、目标填充0.6

```powershell
python run.py run_vmc12_n06
python run.py analyze_parton_realspace_n06
```

等价运行入口：

```powershell
python runners/run_vmc12_n06.py
python runners/analyze_parton_realspace_n06.py
```

模型为硬核玻色子 `H=-J Σ(b†_i b_j+h.c.)+K Σ(b†_1 b_2 b†_3 b_4+h.c.)`，默认 `J=1,K=10`、零物理通量、周期边界。严格投影满足 `n1=n2=nb`。试探态为 `det U1 det U2 exp[-g Qnn+h Mπ]`；`g` 为最近邻 Jastrow 参数，`h` 是波函数中的交错密度参数，物理 Hamiltonian 没有额外施加交错势。

`0.6×144=86.4` 非整数，固定 `Nb=86`，实际填充为 `0.597222...`。工作流按轨道海去重，扫描跃迁比和g，比较CDW因子，使用新链验证，并在g梯度明显非零时最多作两次修正。参数常量与网格在 `calculations/workflows/run_vmc12_n06.py`；历史脚本并不都提供命令行参数，不能假定 `--L` 或 `--filling` 已实现。

主要输出：

|目录/文件|内容|
|---|---|
|`output/vmc12-n06/optimization.json`|变分扫描、候选与梯度修正记录|
|`output/vmc12-n06/summary.json`|独立链能量、误差、密度峰与配对通道|
|`output/vmc12-n06/scan_*.npz`|扫描能量与重加权数据|
|`output/vmc12-n06/final_*.npz`|原始能量、密度变量和空间分块|
|`output/vmc12-n06/combined_*.npz`|合并后的密度/单玻色子/四配对通道|
|`output/vmc12-n06/*png`|变分曲线、密度与配对动量图|
|`output/parton-realspace-n06/`|母态parton及投影后物理关联的距离图|

实空间诊断依赖第一步的 `summary.json`、合并数组及原始分块。母态的非零parton单粒子关联是未投影费米海结果；严格投影后的裸跨格点单parton算符不保持约束，不能把母态曲线当作投影后规范动力学传播。

## 半填充与历史流程

|用途|运行顺序|
|---|---|
|12×12 半填充优化|`python run.py optimize_vmc12` → `python run.py run_vmc12_channels`|
|12×12 固定参数d波对照|`python run.py run_vmc12_dwave`|
|L8轨道海扫描|`python run.py run_projected_vmc_scan`|
|Jastrow扫描及候选复核|`python run.py run_projected_jastrow_grid` → `python run.py run_projected_finalist`|
|密度/玻色子/配对关联|`python run.py projected_exact_correlations` → `python run.py run_projected_correlations` → `python run.py analyze_projected_correlations`|
|流关联|`python run.py run_projected_currents` 或 `run_projected_currents_sizes`；完整历史绘图还需对应精确基准和各尺寸数据|
|流关联精确基准|`python run.py projected_exact_currents`|
|搜索结果绘图|`python run.py analyze_projected_search`（需历史扫描输出）|

历史图脚本可能同时读取多个尺寸/对照文件，应查看其 `main`/`run` 的输入路径；仅执行一个入口不保证产生所有历史图。静态流关联不等于有限温Kubo电导率。本归档不包含Julia ED、电导率或纯平均场工程。

## 测试和小规模验算

`python run.py test` 自动发现 `tests/test_*.py`，覆盖非半填充、行列式比值、逆矩阵更新、局域J/K能量、粒子空穴变换、配对重叠和Fourier相位等。`check-imports` 只验证导入，不会启动生产采样。`validate_projected_correlations`、`validate_projected_currents` 等任务属于额外小簇对照，不会被普通测试发现自动运行。

本次目录拆分另验证了 Windows 多进程导入与小规模采样；没有重跑生产扫描。

## 路径、缓存与结果复用

所有入口将输出统一定位到本归档的 `output/`，与原工作区隔离。若复用历史结果，请将原工作区对应的完整 `output/子目录` 复制到本目录，并保留相对层级以及JSON引用的原始NPZ文件。原有结果不会自动复制。

部分入口按文件名读取缓存，缓存并非都全面核验采样设置。修改粒子数、波函数、种子或样本数后，建议把已有对应输出目录另行备份并使用干净的输出目录运行，避免混用旧结果。

`manifest_original.json` 用于追溯拆分前文件；`manifest.json` 对应当前结构，重构后的文件哈希与原脚本不同。README中的命令均以本目录为项目根目录。
