# PFC2D + FiPy：废石散体恒压注浆的 Bingham 浆体变饱和迁移

博士论文《废石散体灌浆充填渗流规律及其胶结充填体力学承载机制研究与应用》第 4 章的数值模型。
废石骨架刚性固定、结构以等效孔隙率场输入（可来自 PFC2D 颗粒模型），FiPy 求解广义非线性
Darcy + Bingham 屈服的变饱和浆体迁移，支持分段前进式注浆的段间凝固。

## 从哪里读起

| 想知道 | 看 |
|---|---|
| 硬约束、控制方程、验证阶梯、现行结论速查 | `CLAUDE.md` |
| 各专题的实验记录、推翻过程与局限 | `docs/results_log.md` |
| 工程算例的设定与回归锚点数字 | `cases/README.md` |
| 充填闭合（IMPES）的设计与被否方案 | `docs/saturation_design.md` |
| 开发规则（与 CLAUDE.md 互补） | `AGENTS.md` |

## 目录

| 路径 | 内容 |
|---|---|
| `src/models/slurry_transport/` | 控制方程与时间推进（`equations.py`）、变量与参数（`variables.py`）、段间凝固（`stage_update.py`） |
| `src/coupling/` | 孔隙率 → 渗透率律（`porosity_to_permeability.py`）；PFC 耦合驱动（`driver.py`） |
| `src/fipy_adapter/`、`src/pfc_adapter/` | 网格构建；PFC 颗粒读写（`itasca` 惰性导入，无 PFC 时自动跳过） |
| `src/structure/` | 结构孔隙率场（梯度、贯通带、随机场） |
| `src/analysis/` | 充填诊断（可达域 / 三层评价域 / 空区分类）与检查点续跑 |
| `cases/` | 工程算例：真实颗粒堆、梯度场、孔底出浆、通道跑浆、分段序列、招金断面、块石绕流空区 |
| `scripts/` | 扫描、对照与后处理：诊断出图、Picard 扫描、招金筛选与求解、黏度时变界定、无重力对照、参数敏感性、分段试算、出图；不跑求解器的解析估算（`forchheimer_penetration`、`param_window`、`real_param_selfcheck`、`model1_checkup`、`bin_porosity_csv`）；台阶 4a 的 Roache GCI（`gci_4a`，验证阶梯表中的 GCI 数字只由它复现） |
| `tests/` | 100 个测试（验证阶梯、充填、重力、渗透率、结构场、诊断、检查点、招金断面） |
| `pfc/` | PFC 端入口（在 PFC 中 restore 模型后 `call` 运行） |
| `data/particles.csv` | 真实颗粒数据 |
| `reference_cases/` | 降雨入渗参考算例（只读，不引用进新代码） |
| `docs/archive/` | 早期设计文档（已过期，仅作历史记录） |
| `outputs/` | 运行结果（不进版本管理） |

## 运行

运行时是 PFC 5.0 自带的 CPython 2.7.9（捆绑 numpy/scipy/fipy），不需要也不允许 `pip install`；
所有代码须兼容 Python 2.7。`run_local.ps1` 自动定位该解释器并设置 `PYTHONPATH`（PFC 装在别处时
用 `$env:PFC_PYTHON` 覆盖），线性求解器后端默认 scipy。

```powershell
# 全套测试（约 15 分钟）
powershell -File scripts\run_local.ps1 -m unittest discover -s tests -v
# 跑一个算例或脚本
powershell -File scripts\run_local.ps1 cases\inclined_hole_gradient.py --only main
powershell -File scripts\run_local.ps1 scripts\gravity_control.py
```

长算例都带检查点：崩溃后重跑同一条命令即从断点续上。

## 专利线

面连通强度的专利原型已于 2026-09 冻结：代码状态见 tag `patent-prototype-v1` 与
`patent-prototype-v1-figures`；附图脚本、证据数据与打包成品已迁出到本机
`D:\work\patent-archive\`（独立目录，按其 README 可重新生成附图）。
