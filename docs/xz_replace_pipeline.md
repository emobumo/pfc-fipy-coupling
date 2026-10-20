# 新终态替换流程（30 m 采场）

2026-10-20。依据：主线“S1 答复与替换准备”第二部分。新终态 .sav 预计 10 月 21 日到。

**目的**：新终态到了之后，从 .sav 导出开始，按固定顺序重跑 PFC 塌落分析、第一步、第二步 A、第二步 B、求解器算例、check_4_8 与 4.8 全部图件。几何改动全部写在配置文件里，代码不用改。

---

## 〇、配置文件

**位置**：`scripts/xz_config/`。
- `xz_config.py`：读取模块；
- `stope_40m.json`：现终态，备用；
- `stope_30m.json`：新终态的模板。

**选择方式**：环境变量 `XZ_CONFIG`，取 `40m`（默认）、`30m`，或一个 .json 文件的路径。

**输出目录**：各配置写入带后缀的目录，40 m 的结果不会被覆盖。
- 40 m 后缀为空，目录与现在相同；
- 30 m 写入 `outputs/pfc_collapse_30m`、`outputs/xz_prejudge_30m`、`data/xz_prejudge_30m`、`outputs/xz_solver_30m`、`outputs/check_4_8_30m`、`figures/pfc_collapse_30m`、`figures/xz_prejudge_30m`、`figures/xz_step2b_30m`。

**配置项与原硬编码的对应**：

| 配置项 | 含义 | 原来写死在 |
|---|---|---|
| `pfc.export_dir`、`state_fmt`、`initial` / `final` / `prev` / `earlier`、`states_time`、`final_label` | 导出目录、各档名称、终态标注 | `analyze.py`、`prejudge.py`、`pfc_collapse/plot.py`、`check_4_8/prewrite.py` |
| `stope.ore_ref`、`dip_deg`、`true_thick`、`y_floor`、`y_top` | 采场顶点由这些参数算出（`xz_config.stope_polygon`） | `analyze.py`、`pfc_collapse/plot.py` |
| `stope.origin` | 坐标原点，即采场底板下盘角点（载入时校核它与顶点一致） | `prejudge.py` |
| `analysis.box`、`profile_y`、`profile_s` | PFC 像素统计范围、剖面范围 | `analyze.py`、`prejudge.py`、`step2a.py`、`step2b.py` |
| `n_hw` | 完整上盘的圆盘孔隙率（碎胀换算用） | `prejudge.py` |
| `domain` | 0.25 m 计算域 | `prejudge.py` |
| `drifts.spacing`、`count`、`dx`、`half` | 巷道间距、条数、距下盘接触面的水平距离、断面半宽 | `prejudge.py`、`step2a.py` |
| `basal_s`、`low_end_dt` | 下盘松散底层厚度；底层下端取“底层沿倾向最低 `low_end_dt` m” | `prejudge.py`、`step2a.py`、`step2b.py`、`check_4_8/prewrite.py` |
| `solver.low_t` | 0.5 m 网格上的底层下端阈值；`null` 时自动取 0.25 m 底层最低 t + `low_end_dt` | `run_cases.py` |
| `solver.pad`、`corner_box`、`cases`、`ftz_from` | 裁剪余量、原框式角部（不引用）、算例表、次正规数置零起始步 | `run_cases.py`、`check_4_8/*.py` |
| `view.pfc`、`xz`、`cases` | 图幅 | `pfc_collapse/plot.py`、`xz_prejudge/plot.py`、`plot_cases.py` |

**由几何自动导出、不需要配置的量**：
- 孔口：巷道朝矿体一侧边壁中点；
- 钻孔：每条巷道 −30°、0° 两孔，孔口管 2 m；
- 下端角部的新定义：低于巷道 1 孔口的高程，或沿倾向低于巷道 1 孔口；
- 计算域裁剪：散体外包矩形加 `pad`。

**40 m 回归检验（2026-10-20）**：在 40 m 配置下重跑全部确定性部分，求解器不重跑。

| 输出 | 结果 |
|---|---|
| 文本：`analysis.txt`、`prejudge.txt`、`step2a.txt`、`step2b.txt`、`crop_check.txt`、`cases.txt`、`check_4_8/prewrite.txt`、`check_4_8/runs.txt` | 与现有文件**逐字节相同** |
| npz：`data/xz_prejudge/*.npz` 五个，`outputs/` 中七个 | **逐数组相同**（npz 是 zip，内含写入时间，字节本来就不会相同） |
| 图：PNG | **逐字节相同** |
| 图：SVG | 只差在 matplotlib 随机生成的元素 id 和定义顺序；嵌入图像与全部路径相同，与改动前每次重绘的情况一样 |

**`stope_30m.json` 中待新终态确认的项**（文件内 `pending` 一栏）：
- 存档序列：各档名称与终态标注；
- 采场几何：参考点、底板与顶面高程、原点。模板按主线所说取竖向高 30 m，底板仍在 y 40，顶面在 y 70；
- `analysis.box`、`domain`、`view` 必须把新腔体包在内，四周留 ≥ 10 m 岩体。`prejudge.py` 会打印余量；
- `n_hw`：取自新的 `analysis.txt`；
- 第二阶段的算例表（最多 13 个），由主线给出。

**巷道**：模板为 4 条，距采场底板 0、10、20、30 m，`count = 4`。若某条巷道的钻孔打不到散体，`prejudge.py` 直接报错停下，不会静默少一条。40 m 配置的 `count` 为 `null`，即沿用原来的规则：一直加到钻孔打不到散体为止。

---

## 一、步骤

下面所有命令都在仓库根目录执行，先设 `$env:XZ_CONFIG='30m'`。凡是确定性步骤都连跑两遍、逐字节比对。

| # | 步骤 | 命令 | 输入 | 输出 | 耗时（估计） |
|---|---|---|---|---|---|
| 0 | 填配置 | 按新开挖脚本与 .sav 填 `stope_30m.json` 的 `pending` 各项；`python scripts\xz_config\xz_config.py 30m` 校核原点与顶点 | 新 .sav、开挖脚本 | `stope_30m.json` | 0.5 h（人工） |
| 1 | 导出（PFC2D 6.0） | 设 `PFCX_SAVES`（`initial`、`final`、`prev`、`earlier` 与 `states_time` 各档，以分号分隔）、`PFCX_OUT=E:\pfc_collapse_work\export_30m`，运行 `pfc2d600_console.exe call scripts\pfc_collapse\run_export.p2dat`；再设 `PFCX_SAVE=<终态 .sav>`，运行 `run_fishvars.p2dat` | 新 .sav 的副本（原件只读） | `export_30m\*_balls.csv` / `_contacts.csv` / `_walls.csv` / `_meta.txt`、`fishvars_<终态>.txt` | 约 10 档，15～30 min |
| 2 | PFC 塌落分析 | `run_local.ps1 scripts\pfc_collapse\analyze.py`，然后 `plot.py` | 第 1 步导出 | `outputs\pfc_collapse_30m\analysis.txt` 与 npz；`figures\pfc_collapse_30m\` | 约 5 min × 2 遍 |
| 2a | 回填配置 | 按 `analysis.txt` 回填 `n_hw`，核对 `box` 与 `profile_y` / `profile_s` 的范围 | `analysis.txt` | `stope_30m.json` | 10 min |
| 3 | 第一步 | `scripts\xz_prejudge\prejudge.py`，然后 `plot.py` | 导出、配置 | `outputs\xz_prejudge_30m\prejudge.txt`、`data\xz_prejudge_30m\base_geometry_K1.25.npz`、`fig_xz_1～6` | 约 5 min × 2 |
| 4 | 第二步 A | `step2a.py`，然后 `plot.py --v3` | 同上 | `step2a.txt`、`base_geometry_K1.25_v3.npz`、`*_v3` 图 | 约 6 min × 2 |
| 5 | 第二步 B | `step2b.py`，然后 `plot_step2b.py` | 同上 | `step2b.txt`、`base_geometry_K{1.15,1.25,1.40}_v4.npz`、`fig_b1～b3` | 约 8 min × 2 |
| 6 | 分步可达域 | `seq_reach.py` | v4 npz | `outputs\xz_prejudge_30m\seq_reach.txt` | 见第四节 |
| 7 | 裁剪核对 | `run_cases.py --check` | v4 npz | `outputs\xz_solver_30m\crop_check.txt` | 1 min |
| 8 | 求解器算例 | `launch_batches.ps1 -Config 30m -Batches "..." -CapHours H`（第二节） | v4 npz、算例表 | `result_*.json`、`final_*.npz`、`snap_*.npz`、`diag_*.csv`、`wall_*.json` | 第二节 |
| 9 | 汇总与图 | `run_cases.py --summary`（两遍），然后 `plot_cases.py` | 第 8 步 | `cases.txt`、`summary.npz`、`fig_b4_*` | 2 min |
| 10 | 写前核对 | `check_4_8\prewrite.py`、`runs.py`（各两遍），然后 `plot.py --v4` | 第 1～9 步 | `outputs\check_4_8_30m\prewrite.txt`、`runs.txt`；`fig_xz_2/5_*_v4` | 约 5 min × 2 |
| 11 | 锚点 | 编号不变，在 `anchors_ch4.csv` 末列后新增一列“新终态（30 m）数值”，写入新值，旧值保留 | 第 2～10 步 | `docs/anchors_ch4.csv` | 1 h（人工） |
| 12 | 报告 | 按 `xz_step2b.md` 与 `check_4_8_prewrite.md` 的格式写新终态的报告 | 全部 | `docs/` | 主线定 |

**依赖关系**：
- 第 2 步只依赖第 1 步。
- 第 3～5 步依赖第 2a 步回填的 `n_hw`，彼此之间相互独立，可以并行。在 40 m 上，第 2～5 步四个并行共 8.1 min。
- 第 6、7 步依赖第 5 步。
- 第 8 步依赖第 5、7 步。
- 第 10 步依赖第 1 步与第 9 步。

**确定性部分合计**：两遍约 1 h（不含人工填配置）。求解器算例另计，见第二节。

**4.8 图件对应**：

| 图号 | 文件 |
|---|---|
| 4.8-2 | `figures\xz_prejudge_30m\fig_xz_2_porosity_v4`、`fig_xz_5_preq_v4` |
| 4.8-3 | `figures\xz_step2b_30m\fig_b4_*` |
| 4.8-4 | `figures\xz_step2b_30m\fig_b3_zones_compare` |

---

## 二、求解器并行（第二阶段最多 13 个单线程算例）

**本机**：
- CPU：i9-14900HX，24 个物理核、32 个逻辑核，限频 4200 MHz（`machine-instability` 记录的 CPU 退化问题仍在）；
- 内存：31.7 GB。

**单算例内存**：0.5 m 裁剪域（21 730 个单元）峰值工作集 163 MB、峰值私有内存 157 MB（`outlet_trial.py` 实测）。13 个算例合计约 2.1 GB，内存不构成限制。

**并行测速**（2026-10-20，40 m 终态，S4 设置的前 150 步，单线程、置零自第 1 步；测速期间另有一个同类进程在跑）：

| 同时运行的进程数 | 150 步墙钟 | 相对单独运行（186.4 s） | 吞吐（算例 × 速度） |
|---|---|---|---|
| 1 | 186.4 s（原 S4 记录） | 1.00 | 1.0 |
| 8（7 个测速 + 1） | 270～272 s | 1.45 | 5.5 |
| 14（13 个测速 + 1） | 384～386 s | 2.07 | 6.8 |

CPU 是 8 个性能核加 16 个能效核，进程多于 8 个后每个进程都变慢，但总吞吐仍在增加。

**统一设置**：
- 每个算例从第 1 步起单线程（`MKL_NUM_THREADS = OMP_NUM_THREADS = 1`），次正规数置零自第 1 步，中途不改。
- `run_cases.py --cases` 对 40 m 以外的配置强制检查这两项，不满足就拒绝启动。
- 理由见 `docs/xz_step2b.md` 3.1 运行说明 4：线程设置不同时，时间步分支翻转会把舍入差放大到 10⁻³ 量级。

**启动方式**：`scripts\xz_solver\launch_batches.ps1`。
- 每批一个进程，批内各算例串行；
- 每批有自己的墙钟账本 `wall_<批名>.json` 和检查点 `ckpt_<算例>`；
- 崩溃或重启后，用同一条命令再启动一次，各批从检查点续跑；
- 已完成的算例自动跳过。

**分批方案与墙钟**：单个算例单独运行的墙钟取 40 m 的实测：S1 6.33 h、S3 2.86 h、S4 6.41 h、S5 7.42 h。S2 全程置零时估计约 6.5 h。即典型 6～7.5 h，最长约 7.5 h。

| 方案 | 做法 | 估计墙钟 |
|---|---|---|
| **甲（建议）** | 13 个算例一次全部启动，每批 1 个算例：`-Batches "S1;S2;...;S13" -CapHours 24` | 最长算例 7.5 h × 2.07 ≈ **16 h** |
| 乙 | 分两波，每波 7、6 个（每波 7 个进程时为 1.45 倍） | 2 × 7.5 h × 1.45 ≈ 22 h |
| 丙 | 串行 | 13 × 约 6.5 h ≈ 85 h |

- 30 m 采场的散体比 40 m 小，算例步数可能更少。以上按 40 m 估计，偏保守。
- `-CapHours 24` 是单个算例的墙钟上限，到了就停（结果标注原因），防止个别算例因数值蠕变迟迟不停。
- 风险：本机 CPU 有退化记录。满载运行时若蓝屏，重启后用同一条 `launch_batches.ps1` 命令续跑，从最近的检查点（每 200 步）接上，最多损失 200 步。
- 续跑后的结果与不中断时逐位相同：检查点续跑在 2026-09 已验证逐位一致；线程与置零设置不变。

---

## 三、分步可达域（`scripts/xz_prejudge/seq_reach.py`）

**做法**（确定性，0.25 m，p₀ = 2 MPa，v4）：
- 按施工顺序逐孔计算可达域。
- 每注完一孔，就把该孔在当前域上的可达域设为胶结（n = 10⁻³，不透浆），再算下一孔。
- 已被胶结的出浆单元不再出浆，即胶结管段自封堵（4.6）。
- 三种顺序：
  - 不计先后：单孔可达域的并集，即现行的 2 MPa 分区；
  - 自下而上：巷道 1→4，每条巷道先 −30° 后 0°；
  - 自上而下：巷道 4→1，同上。

**输出**：`outputs/xz_prejudge<后缀>/seq_reach.txt`，包括：
- 最终覆盖的面积与占比（散体、上盘侧、内部）；
- 各孔分步可达域的面积，及其与单孔可达域之比；
- 被胶结的出浆单元数；
- 各孔分步可达域是否含底层下端；
- 比并集少掉的分块（面积、形心、范围、所在分区）。

**40 m 终态试算（备用）**：三档 K × 两情景，两遍逐字节相同（MD5 962e001e…），单遍约 18 min。主要数字见回报。

---

## 四、出矿口出流边界（`scripts/xz_solver/outlet_trial.py`）

**结论：只改 scripts/ 即可实现，不需要改 `src/`。**

**做法**：
- src 的充填闭合本来就把 S < 阈值的“未充填”单元用强隐式汇钉在空气压 0，流入的浆液按通量记入其饱和度（`_update_fill_mask`、`update_saturation_from_flux`）。
- 所以只要让出口单元（0.5 m 网格上的底层下端，8 个单元）每步结束时把 S 置回 0，它就始终是压力为 0 的出流单元。
- 该步流进出口的浆液计为出流，用的是求解器同一通量场（−div q · V · dt）。
- 另以出口单元复位前所含浆量（n S V）交叉核对，两者逐位相同，说明出口处没有截断损失。

**试算**：40 m 终态，最低巷道 −30° 孔 2 MPa（S4 设置），单线程，置零自第 1 步，不设停注判据；`outlet_trial.py --steps 500` 与 `--steps 3500`。

| 步 | t (s) | V_in | 散体内储浆增量 | 压缩储存 | 累计出流 | \|V_in − (储浆 + 出流)\| / V_in | 完整账目相对差 | 出口最大压力 |
|---|---|---|---|---|---|---|---|---|
| 500 | 5.08 | 9.518 | 9.515 | 0.0036 | 0 | 3.8×10⁻⁴ | 1.2×10⁻⁹ | 0 Pa |
| 2300 | 49.48 | 30.470 | 30.450 | 0.0146 | 0.0053 | 4.8×10⁻⁴ | 3.7×10⁻¹⁰ | 0.25 Pa |
| 3000 | 86.82 | 38.917 | 38.248 | 0.0186 | 0.651 | 4.8×10⁻⁴ | 2.9×10⁻¹⁰ | 0.35 Pa |
| 3500 | 122.81 | 45.079 | 43.593 | 0.0213 | **1.465** | 4.7×10⁻⁴ | **2.5×10⁻¹⁰** | 0.38 Pa |

（体积单位 m³/m；完整账目 = 储浆增量 + 压缩储存 + 出流 + 截断量，截断量 −2.9×10⁻⁵。）

- **质量守恒**：
  - 完整账目闭合到 10⁻¹⁰～10⁻⁹，是浮点误差量级。
  - 按主线口径（只计散体内储浆与累计出流），相对差约 4.7×10⁻⁴，正好是浆液的压缩储存，与是否设出口无关：500 步时出口还没有出流，这一项已经是 3.8×10⁻⁴。
- **出口处压力**：在 1 Pa 以下，即被钉在 0。
- **到达出口之前与原求解器逐位相同**：第 2289 步之前，V_in 与 `final_S4.npz` 每一步都逐位相同。这是因为空的出口单元本来就被钉在 0。浆液从第 2289 步开始进入出口单元，此后两者才分开。原 S4 在第 2402 步记为“到达底层下端”（首个下端单元 S ≥ 0.5）。
- **时间对照**：出流开始后注入速率略高于原 S4。到 t ≈ 120 s，原 S4 的 V_in 为 43.71 m³/m（t = 118.5 s），设出口时为 45.08 m³/m（t = 122.8 s），其中 1.46 m³/m 已流出。

**若主线采用**：
- 在 `run_cases.run_case` 中加同样的出口处理，按配置开关（例如 `solver.outlet: true`）启用。
- 指标中增加“累计出流”，并以出流开始的步作为“到达出矿口”。
- 不改 `src/`。

`outlet_trial.py` 里“首次出流”的判据在 3500 步试算之后改为“出口单元净流入大于 0 的首步”。原判据“单步内出口单元 S ≥ 0.5”因为出口每步清零，从不触发。
