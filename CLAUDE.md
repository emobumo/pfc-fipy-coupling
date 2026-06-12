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
- 孔隙率仅在初始化时从 PFC 传递一次，之后不再动态更新。
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
| 2 | 一维 Bingham 停滞：λ>0 瞬态注浆（porous_bingham threshold 模式，λ=2τ₀/√(8k/φ)） | 锋面停滞于 L_max = p0/λ（实测 0.610 vs 0.600 m，误差 1.7%）；剖面逼近 max(0, p0−λx)（偏差 0.6% p0） | 通过 |
| 3 | 径向 Gustafson–Stille 基准：四分之一对称角部钻孔（r0=0.1 m），沿 +x 轴取样 | 停滞半径 vs I_max=r0+p0/λ（实测 0.680 vs 0.700 m，误差 2.9%，容差 8%）；轴向剖面 vs p0−λ(r−r0)（偏差 3.6% p0） | 通过 |
| 4 | 网格收敛阶 | 网格逐次加密，L2 误差收敛阶接近理论值 | 未开始（skipTest 占位） |

状态取值：未开始 / 失败 / 通过。每完成或修改一个台阶，同步更新本表。
