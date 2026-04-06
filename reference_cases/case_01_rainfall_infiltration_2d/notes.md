# notes

- 该案例先运行 01_sample，再运行 02_balance。
- 01_mesh.py 会建立 FiPy 网格，并恢复 balance。
- 03_cal.p2dat 是总控脚本。
- 02_set_up.py、03_solver.py、04_cal_value.py 依赖共享变量，暂时不能随意独立改写。
- 第一阶段目标是保真还原，不做结构重构。