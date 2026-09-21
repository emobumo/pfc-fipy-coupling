# CLAUDE.md

PFC2D + FiPy 耦合：恒压注浆条件下 Bingham 浆体在固定废石堆中的变饱和迁移。
本文件是对 Claude 的硬约束；与 AGENTS.md、docs/physical_model.md 一致并互为补充。

## 硬性环境约束（不可违反）

- 运行时是 **PFC 5.0 自带的 CPython 2.7.9**（捆绑 numpy/scipy/fipy）。所有 `src/`
  代码必须兼容 Python 2.7：
  - 禁止 f-string；
  - 除 `print(...)` 函数式写法外，禁止任何 Python 3 专属语法（无 `nonlocal`、
    无类型注解、无 `yield from`、无字典/集合推导以外的新语法依赖）；
  - 禁止 `pathlib`（用 `os.path`）；
  - 禁止 `pip install`（运行时不可装包，依赖以 `requirements.txt` 记录为准）。
- 运行与测试统一通过：
  - 跑脚本：`powershell -File scripts\run_local.ps1 scripts\smoke_test_src.py`
  - 跑测试：`powershell -File scripts\run_local.ps1 -m unittest discover -s tests -v`
  - 该脚本自动定位 PFC 自带 Python 并设置 `PYTHONPATH`；PFC 装在别处时用
    `$env:PFC_PYTHON` 覆盖。
- FiPy 侧可独立于 PFC 运行测试：`src/pfc_adapter/particle_writer.py` 与
  `porosity_reader.py` 对 `itasca` 为**惰性导入**，无 PFC 环境时自动跳过/回退，
  不得改成顶层导入。

## 物理约束（与 AGENTS.md 一致，未经用户明确变更不得突破）

- 废石骨架刚性固定，不引入流固耦合变形、不引入流体驱动的颗粒运动。
- 孔隙率仅在初始化时从 PFC 传递一次，**段内**不再动态更新。唯一例外是**段间凝固**
  （`src/models/slurry_transport/stage_update.py`）：单段注浆结束、浆体硬化后，已充填
  孔隙被固结浆体永久占据，`n_{k+1} = n_k(1-S_k)`。它只在段与段之间调用，绝不进入
  marching 循环；骨架本身仍然刚性。段内降低 n 或 k 等于堵死输浆通道，是明确的错误
  （台阶 7 的 `test_porosity_is_frozen_within_a_stage` 守这条线）。
- 过程定位为**变饱和浆体迁移**，不是全饱和渗流。
- 本构框架为**广义非线性 Darcy**，不得退回线性饱和 Darcy 作为最终模型
  （linear 模式仅作为回归基线保留）。
- 不采用 Richards–van Genuchten 土壤水模型。
- 首版不含堵塞（clogging 默认关闭，仅保留扩展点）。
- 首版目标控制方程：
  - 质量守恒：`n·∂S/∂t + ∇·q = 0`
  - 广义 Darcy：`q = -(k·k_r(S)/μ_p)·max(0, 1-λ/|∇Φ|)·∇Φ`，`Φ = p + ρgz`，
    λ 为启动压力梯度。

## 工作规则

- **FiPy 侧改动**：可自行用 `run_local.ps1` 跑测试验证后再交付。
- **PFC 侧改动**（`src/pfc_adapter/`、`pfc/`、`scripts/diagnose_*` 等依赖
  itasca 的代码）：只能做静态检查，改完必须明确提醒用户在 PFC 中手动验证。
- 改动遵循**最小 diff**，不做顺带重构。
- 高风险改动（时间步进、收敛准则、边界条件、颗粒-网格映射、单位换算）必须
  先征得用户同意（见 AGENTS.md）。
- `reference_cases/` 仅作参考，不修改、不引用进新代码。

## 验证阶梯（tests/test_verification_ladder.py）

| 台阶 | 内容 | 判据 | 当前状态 |
|------|------|------|----------|
| 1 | 一维线性 Darcy 退化：λ=0、均匀 k/n、全饱和、水平一维、两端定压 | 稳态压力与解析直线 p(x)=p0(1-x/L) 最大相对误差 <1%；流量 q0=(k/μ)(p0/L) 相对误差 <1% | 通过 |
| 2 | 一维 Bingham 停滞：λ>0 瞬态注浆（threshold 模式 + 面迁移率 + S≡1 退化回归） | 锋面停滞于 L_max = p0/λ（面化后实测 0.620 vs 0.600 m，误差 3.3%，容差 5%）；剖面逼近 max(0, p0−λx)（偏差 2.8% p0） | 通过 |
| 2b | 一维变饱和充填：初始 S=0 干堆，充填闭合（未充填单元罚至空气压，显式面流入充填）；锋面跟踪 Gustafson–Stille 解析时间曲线 t(x_f)=(n/Mλ)[L_max·ln(L_max/(L_max−x_f))−x_f] | 注浆期内 S 锋面 vs GS 解析曲线最大偏差 <5% L_max（实测 ≤0.3%，至 0.65 L_max）；V_in(t) 单调；守恒漂移 <1e-5（实测 1.2e-15，通量场记账）；clip 严格为 0。注：x_f→L_max 仅 t→∞ 渐近达到；屈服边缘 Picard 不收敛致长时间伪蠕动（第五轮求解器议题） | 通过 |
| 3 | 径向 Gustafson–Stille 基准：四分之一对称角部钻孔（r0=0.1 m），沿 +x 轴取样（S≡1 退化回归） | 停滞半径 vs I_max=r0+p0/λ（面化后实测 0.700 vs 0.700 m，误差 0.0%，容差 8%）；轴向剖面偏差 2.6% p0 | 通过 |
| 4a | 光滑解网格收敛阶：非均质 k(x)=M₀(1+ax/L) 线性 Darcy，曲率对数解析解，N=20/40/80/160（线性解会被 2 阶格式精确复现，故必须用曲率解） | L2 相对误差收敛阶 >1.8（实测 1.98；Roache GCI：表观阶 1.993、GCI_fine 0.0022%、渐近比 0.9999） | 通过 |
| 4b | 含停滞锋面网格收敛阶：Bingham 停滞，dt∝dx 同步细化，N=25/50/100 | 停滞位置误差单调收敛、阶 >0.8（实测 1.0；锥点不光滑+过冲冻结使阶 ≈1 而非 2，是锐锋固有性质；剖面 L2 因锥点对齐/残余而不可靠，仅诊断不断言） | 通过 |
| 7 | 段间凝固守恒：分段前进式，`n_{k+1} = max(n_k(1-S_k), floor)`，1D 30 单元、3 段；穿孔管全长出浆，源随孔延伸（非移动窗口），钻进增量 2.5 m 须大于 L_max≈2.1 m | 闭合规则逐单元 bit 精确；`Σ(n_k−n_{k+1})V + floor_clip == Σ n_k S_k V`（机器精度）；胶结体 λ/原生 λ >100（实测 428）；clip 永不生效；`occupied_cumulative + remaining_capacity == pore_capacity_initial`；段内 n 逐步 bit 不变；硬化体积 vs 流动实际送入相对差 <5%（实测 ≪）；剩余容量单调递减；已胶结管段输出量占比 <1e-6（实测 1e-34，完全屏蔽） | 通过 |

状态取值：未开始 / 失败 / 通过。每完成或修改一个台阶，同步更新本表。
验证阶梯：台阶 1/2/2b/3/4a/4b/7 全部通过，全套 40 测试零 skip。

**胶结体本构（台阶 7 发现并已修复，2026-09-21）**：`permeability_clip_min` 原为 1.0e-10 m²。
它在 φ<0.095 才生效，而所有原生废石场都在 0.12~0.45（真实堆最低 0.144），所以一直休眠；
段间凝固第一次造出 n=1e-3 的胶结单元后，它立刻成了主导参数——不仅把"胶结体 vs 原生"的
渗透率反差封顶在 69 倍，更因为 `λ = 2τ₀/√(8k/n)` 里 k 在分子、n 在分母，**把启动压力梯度
算反了**：实测 λ_胶结/λ_原生 = 0.48，胶结体比周围岩体更容易起流。

已改为 **1.0e-18**（比律在孔隙率地板处的取值低两个数量级，永不生效，退回纯数值护栏）。
于是胶结体渗透率由 `stage_porosity_floor` 经律决定：k(1e-3) ≈ 1.25e-16 m²，是固结水泥浆的
正确量级；λ_胶结/λ_原生 回到 428。`src/coupling/porosity_to_permeability.py` 里两处
`.get()` 兜底值同步改为 1e-18，避免出现第二个隐藏 clip 值。

- **对三个工程算例可证明惰性**：它们的 k 最低 3.81e-10，无单元触到 1e-10，结果不变，
  `cases/README.md` 的回归锚点依然有效。
- **条件数实测无代价**：同一台阶 7 分段序列在两个 clip 下 75 次压力求解、最大 12 次
  Picard、均值 3.76 完全一致，而 k 跨度从 69 倍变到 5.5e7 倍。胶结面经调和平均后近似断开，
  退出活跃方程组而不是把它变刚。**该结论来自 30 单元一维夹具，二维多段算例需另行确认。**
- 两个测试守这条线：`test_cemented_ground_resists_start_up_far_more_than_virgin`（方向）与
  `test_permeability_clip_never_binds_in_the_operating_range`（clip 必须始终低于律）。
