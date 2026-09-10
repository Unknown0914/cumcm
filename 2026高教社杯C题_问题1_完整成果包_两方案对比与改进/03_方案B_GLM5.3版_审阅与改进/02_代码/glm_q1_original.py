# -*- coding: utf-8 -*-
"""2026 CUMCM C题 问题1：微网单日计划购电策略（LP）

【GLM-5.3 方案原始代码 —— 逐字复现版】
来源：《2026年高教社杯全国大学生数学建模竞赛 C题 问题1求解文档》(GLM-5.3, 2026-09-10)

本文件除以下两处"仅为能运行"的必要改动外，与原文代码完全一致：
  1) pd.read_excel('附件1.xlsx') -> 绝对路径（附件位于 D 盘）；
  2) 输出文件写到 output/ 子目录，避免与改进版混淆。
其余变量命名、约束写法、表 1 索引 idx=[60,72,...]、导出结构等均保持原样，
以便真实检验该方案能否直接运行、结果是否正确。
"""
import os
import numpy as np, pandas as pd, pulp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'results')
os.makedirs(OUT, exist_ok=True)
SRC = r'D:\download\CUMCM2026Problems\C题\附件\附件1.xlsx'

# ---------- 1. 数据 ----------
df = pd.read_excel(SRC)
df = df.rename(columns={df.columns[0]:'时间', df.columns[1]:'电价',
                        df.columns[2]:'负载', df.columns[3]:'光伏'})
pi = df['电价'].to_numpy(float); L = df['负载'].to_numpy(float)
v  = df['光伏'].to_numpy(float); T = len(df); dt = 1/6
E0, Emin, Emax, Pmax, eta = 6000.0, 1200.0, 10800.0, 5000.0, 0.90
# ---------- 2. LP模型 ----------
prob = pulp.LpProblem('C2026_Q1', pulp.LpMinimize)
p = [pulp.LpVariable(f'p{t}', lowBound=0) for t in range(T)]
c = [pulp.LpVariable(f'c{t}', lowBound=0, upBound=Pmax) for t in range(T)]
f = [pulp.LpVariable(f'f{t}', lowBound=0, upBound=Pmax) for t in range(T)]
d = [pulp.LpVariable(f'd{t}', lowBound=0) for t in range(T)]
E = [pulp.LpVariable(f'E{t}', lowBound=Emin, upBound=Emax) for t in range(T)]
prob += pulp.lpSum(pi[t]*p[t]*dt for t in range(T))          # 目标：购电费最小
for t in range(T):
    prob += p[t] + v[t] + f[t] == L[t] + c[t] + d[t]          # 供需平衡
    Ein = E0 if t == 0 else E[t-1]
    prob += E[t] == Ein + eta*c[t]*dt - f[t]*dt/eta           # 储量动态
prob += E[T-1] == E0                                           # 0:00与24:00储电量相同
prob.solve(pulp.PULP_CBC_CMD(msg=False)); Z = pulp.value(prob.objective)
# ---------- 3. 结果与自检 ----------
pv = np.array([x.value() for x in p]); cv = np.array([x.value() for x in c])
fv = np.array([x.value() for x in f]); dv = np.array([x.value() for x in d])
Ev = np.array([x.value() for x in E])
print('状态:', pulp.LpStatus[prob.status])
print(f'全天购电量={pv.sum()*dt:,.4f} kWh  全天购电费={Z:,.4f} 元')
print(f'弃电总量={dv.sum()*dt:,.4f} kWh；同时充放max(c·f)={(cv*fv).max():.2e}')
print(f'SOC范围[{Ev.min():.2f},{Ev.max():.2f}]，E24={Ev[-1]:.2f}')
# ---------- 4. 论文表1 ----------
slots = ['10:00-10:10','12:00-12:10','14:00-14:10','16:00-16:10','18:00-18:10','20:00-20:10']
idx = [60, 72, 84, 96, 108, 120]
tab1 = pd.DataFrame({'时间段':slots, '购电量(kWh)':np.round(pv[idx]*dt,4)})
tab1.loc[len(tab1)] = ['全天购电量(kWh)', round(pv.sum()*dt,4)]
tab1.loc[len(tab1)] = ['全天购电费(元)',  round(Z,4)]
print('\n表1\n', tab1.to_string(index=False))
# ---------- 5. 论文表2（每4小时=24个时段） ----------
bands = ['0:00-4:00','4:00-8:00','8:00-12:00','12:00-16:00','16:00-20:00','20:00-24:00']
rows = [[b, round(cv[i*24:(i+1)*24].sum()*dt,4), round(fv[i*24:(i+1)*24].sum()*dt,4)]
        for i, b in enumerate(bands)]
tab2 = pd.DataFrame(rows, columns=['时间段','充电量(kWh)','放电量(kWh)'])
tab2.loc[len(tab2)] = ['0:00 储电量', E0, '']
tab2.loc[len(tab2)] = ['24:00 储电量', round(Ev[-1],4), '']
print('\n表2\n', tab2.to_string(index=False))
# ---------- 6. 导出 result1.xlsx ----------
out_xlsx = os.path.join(OUT, 'glm_original_result1.xlsx')
with pd.ExcelWriter(out_xlsx, engine='openpyxl') as w:
    pd.DataFrame({'时间':df['时间'], '购电量':np.round(pv*dt,4)}
                 ).to_excel(w, sheet_name='计划购电量', index=False)
    tab2.to_excel(w, sheet_name='充放电量', index=False)
print('\nresult1.xlsx 已生成（请对照附件5模板核对列名与时间格式）')

# 追加：把原始解落盘，供后续审计使用（不改动原逻辑）
np.savez(os.path.join(OUT, 'glm_original_solution.npz'),
         p=pv, c=cv, f=fv, d=dv, E=Ev, pi=pi, L=L, v=v)
