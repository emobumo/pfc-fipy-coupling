# 仓库现状审查（打包附件）

本文件是给外部审阅者的**现状说明**，不是设计文档。它存在的唯一目的：让你在读代码前
就知道**哪些内容是过期的、哪些是死代码**，不必浪费精力重新发现，也不会基于错误的
现状给出建议。

写于 2026-07-14，对应提交 `37572cc`。

---

## 一、先读这一条：docs/ 已经严重落后于代码

**`docs/` 下的三份文档写于实现之前，从未更新。不要把它们当作当前设计的描述。**

具体地：

- `docs/problem_definition.md` 和 `docs/coupling_contract.md` 仍把**“顶部占位注浆口”**
  （`inlet_zone_center_x` / `inlet_core_width_x` 那套参数）描述为当前的注浆边界。
  **实际的核心算例用的是侧面斜孔 + 孔底段内部 Dirichlet 源**，是完全不同的机制。
- `docs/physical_model.md` 只有概念性表述（“possible Bingham-type yield effect”），
  **没有任何已实现的控制方程**。
- **校准渗透率律、变饱和充填输运、斜孔内部源**这三项核心工作，docs/ 里一个字都没有。

**代码的真实现状以 `CLAUDE.md`（含精确控制方程与验证阶梯表）和 `cases/README.md`
（三个算例的设定与结果）为准。** `docs/saturation_design.md` 是唯一仍然准确的技术
文档（IMPES 充填闭合的设计与被否方案记录）。

另：同一组“固定假设”（骨架刚性、孔隙率只传一次、变饱和、广义 Darcy、不用
Richards–van Genuchten、首版无堵塞）在 README.md / AGENTS.md / CLAUDE.md /
physical_model.md / problem_definition.md / coupling_contract.md 中**重复了六遍**。

---

## 二、死代码：约 260 行，从未被任何测试、算例或脚本触发

已做全仓库反查确认无调用者。**保留它们是历史沉淀，不是设计意图。**

| 位置 | 规模 | 说明 |
|---|---|---|
| `compute_porous_bingham_fields` + `get_porous_bingham_eps`（tanh 激活路径） | ~65 行 | `porous_bingham_activation="tanh"` 全仓库无人设置 |
| `_update_yield_latch` + `enable_yield_latch` | ~55 行 | 该开关无人开启（代码注释自称“几乎惰性”） |
| `update_filling_from_flux` + `legacy_filling_mode` | ~45 行 | 该开关无人开启 |
| `gated_bingham` 整条流变路径 | ~40 行 | 仅 `scripts/diagnose_bingham_multistep.py` 使用；验证阶梯与三个算例都不碰 |
| `_compute_net_inflow_from_flux` 的两级 fallback | ~30 行 | 两个裸 `except Exception: pass` 兜底，FiPy 主路径正常时永不触发 |
| `read_cell_porosity_once` | ~18 行 | 向后兼容包装，无人调用 |
| `pressure_coeff_form="cell"` 分支 | ~10 行 | 所有算例与默认值均为 `"face"` |

合计约占 `src/models/slurry_transport/equations.py`（1318 行）的 20%。

**尚未删除**：删流变路径按 CLAUDE.md 属于高风险改动，等审阅意见后再定。

---

## 三、一个结构性事实：PFC ↔ FiPy 闭环从未端到端跑通

这是本项目最容易被误解的一点。

- `pfc/run_coupling.py`（PFC 侧入口）**只做孔隙率读取**。脚本自己写明流动求解与
  回写“故意还没跑”，因为注浆边界当时还是占位值。
- `src/coupling/driver.py` 的默认配置（顶部注浆口、x∈[-7.5,7.5] y∈[0,40] 域）与
  `cases/` 里的斜孔算例（侧面斜孔、x∈[-15,15] y∈[0,30] 域、CSV 输入）是**两套互不
  相干的配置**。三个算例**全部绕过 driver.py**，自行组装 state。

也就是说：物理引擎经过六轮验证、算例产出了核心成果，但立项时设想的
“PFC 提供结构 → FiPy 求解 → 回写颗粒”这条链，**目前只跑通了第一段**。
研究成果（非均匀诱发不对称扩散）来自纯 FiPy 侧，与 PFC 的耦合只体现为一次性的
颗粒 CSV 输入。

---

## 四、当前真实状态（可信）

**引擎**：验证阶梯已闭合，28 个测试零 skip（台阶 1/2/2b/3/4a/4b，见 CLAUDE.md 表格）。
光滑解收敛约 2 阶（Roache GCI 确认），锐锋约 1 阶（锥点不光滑，锐锋固有）。

**核心科研成果**（`cases/`，三个算例，确定性、可复现）：

1. `inclined_hole_grouting.py` —— 真实颗粒堆基准。数值扩散 8.5 m vs 解析
   L_max = p0/λ ≈ 8.2 m，**自洽性验证通过**。该堆近均质（φ≈0.184，σ=0.014），
   看不到非均匀效应。
2. `inclined_hole_gradient.py` —— 人为“下密上松”梯度场。**孔隙率梯度翻转了重力偏置**：
   均质 up/down = 0.90（重力下拽）→ 梯度 1.26（浆体逆重力往松散区走）。梯度强度扫描
   1.22/1.26/1.35 单调。
3. `inclined_hole_toe.py` —— 孔底段出浆（符合孔口管封口工艺）。**纯梯度不对称 1.22**，
   与整条线源的 1.26 几乎一致 —— 两种完全不同的源几何给出同样的 ~1.2 上偏，
   **排除了源几何假象**，证明不对称是孔隙率梯度的真实物理。

**关键修正**：渗透率律从 Kozeny-Carman 换成校准式 `k = 9.4e-8·φ³/(1-φ)²`。KC 对分米级
粗粒废石高估控流渗透率约 3 个量级，导致 Bn≈0.02、屈服自停滞机制在数值上形同虚设。
校准后停滞机制自洽激活。

**已知限制**：2.5 m 网格下 8–10 m 的扩散尺度只有 3–4 个单元，up/down 比值带量化噪声，
**1.22 的可靠精度约 ±0.1**。出定量结果前需加密到 1.0/0.5 m。梯度场是人为构造的解析场，
不是真实 PFC 结构 —— 它演示机理，不复现实测堆体。

---

## 五、环境硬约束（影响任何代码建议）

运行时是 **PFC 5.0 自带的 CPython 2.7.9**（捆绑 numpy 1.9.2 / scipy 0.15.1 / fipy）。
所有 `src/` 代码必须兼容 Python 2.7：无 f-string、无类型注解、无 pathlib、
**不能 pip install**（运行时无法装包）。任何“换个库/升级依赖”的建议在这里都不可行。
