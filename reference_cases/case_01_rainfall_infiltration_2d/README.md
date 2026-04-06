# case_01_rainfall_infiltration_2d

## 案例作用
二维降雨入渗土柱耦合案例，包含 PFC2D 建样、重力平衡、FiPy 网格求解和结果回写颗粒。

## 文件说明
- pfc/01_sample.p2dat：建样并保存 sample
- pfc/02_balance.p2dat：重力平衡并保存 balance
- py/01_mesh.py：创建 FiPy 网格和基本变量，并从 balance 生成 mesh 状态
- py/02_set_up.py：初始化水力变量
- py/03_solver.py：单步 FiPy 求解
- py/04_cal_value.py：把场变量映射回颗粒
- pfc/03_cal.p2dat：耦合主循环

## 原始来源
由 Word 文档《降雨入渗土柱试验二维案例》拆分整理。

## 当前状态
第一版为“按原始逻辑还原”，未做结构性重构。

## 注意事项
Python 各脚本共享运行上下文，暂时不能随意单独运行。