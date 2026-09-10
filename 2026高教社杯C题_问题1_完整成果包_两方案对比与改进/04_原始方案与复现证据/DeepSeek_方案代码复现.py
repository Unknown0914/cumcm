# -*- coding: utf-8 -*-
"""2026 CUMCM C题 问题1：微网单日计划购电策略（LP）—— DeepSeek 版原始代码逐字复现

来源：桌面《2026高教社杯 C题第一问解答.pdf》（DeepSeek 生成，2026-09-10）第 4--6 页代码块。

本文件除以下"仅为能运行"的必要改动外，与原文代码完全一致：
  1) pd.read_excel('附件1.xlsx') -> 绝对路径；
  2) pd.ExcelWriter('result1.xlsx') -> 写到 results/ 子目录；
  3) 追加把解落盘（np.savez）与关键指标打印，便于后续审计。
其余（变量顺序、约束构造、bounds/A_ub 写法、表 1 按 P60 取值等）均保持原样，
以便真实检验该方案能否直接运行、结果是否正确。
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'results')
os.makedirs(OUT, exist_ok=True)
SRC = r'D:\download\CUMCM2026Problems\C题\附件\附件1.xlsx'

# ========== 1. 读取附件1 ==========
df = pd.read_excel(SRC)
# 假设列名依次为：时间、电价、小区负载、光伏发电预测功率
df.columns = ['时间', '电价', '小区负载', '光伏发电预测功率']
c = df['电价'].values.astype(float)
Load = df['小区负载'].values.astype(float)
PV = df['光伏发电预测功率'].values.astype(float)
n = len(c)          # 144
dt = 1/6            # 小时
E0 = 6000.0         # 初始储电量 kWh
Emax = 10800.0
Emin = 1200.0
Pmax = 5000.0
eta = 0.9
# ========== 2. 构造线性规划 ==========
# 决策变量顺序: P_buy(n), P_ch(n), P_dis(n), E(n)
# 总变量数 4*n
obj = np.zeros(4*n)
obj[:n] = c * dt    # 目标：购电费用最小
A_eq = []
b_eq = []
# 2.1 功率平衡: P_buy + PV + P_dis = Load + P_ch
for t in range(n):
    row = np.zeros(4*n)
    row[t] = 1            # P_buy
    row[n+t] = -1         # -P_ch
    row[2*n+t] = 1        # P_dis
    A_eq.append(row)
    b_eq.append(Load[t] - PV[t])
# 2.2 储能动态: E_t - E_{t-1} - eta*P_ch*dt + P_dis*dt/eta = 0
for t in range(n):
    row = np.zeros(4*n)
    row[3*n+t] = 1        # E_t
    if t > 0:
        row[3*n+t-1] = -1 # -E_{t-1}
    row[n+t] = -eta * dt          # -eta*P_ch*dt
    row[2*n+t] = dt / eta         # +P_dis*dt/eta
    A_eq.append(row)
    if t == 0:
        b_eq.append(E0)   # E_0 = 6000
    else:
        b_eq.append(0)
# 2.3 末储电量 E_{n-1} = 6000
row = np.zeros(4*n)
row[3*n+n-1] = 1
A_eq.append(row)
b_eq.append(6000)
A_ub = []
b_ub = []
# 2.4 充放电功率上限
for t in range(n):
    row = np.zeros(4*n)
    row[n+t] = 1
    A_ub.append(row); b_ub.append(Pmax)
    row = np.zeros(4*n)
    row[2*n+t] = 1
    A_ub.append(row); b_ub.append(Pmax)
# 2.5 储能容量上下限
for t in range(n):
    row = np.zeros(4*n)
    row[3*n+t] = 1
    A_ub.append(row); b_ub.append(Emax)
    row = np.zeros(4*n)
    row[3*n+t] = -1
    A_ub.append(row); b_ub.append(-Emin)
# 变量下界
bounds = [(0, None)] * n + [(0, Pmax)] * n + [(0, Pmax)] * n + [(Emin, Emax)] * n
# ========== 3. 求解 ==========
res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
              bounds=bounds, method='highs')
if res.success:
    x = res.x
    P_buy = x[:n]
    P_ch = x[n:2*n]
    P_dis = x[2*n:3*n]
    E = x[3*n:4*n]
    total_buy = np.sum(P_buy) * dt
    total_cost = np.sum(c * P_buy) * dt
    print('全天购电量: %.4f kWh' % total_buy)
    print('全天购电费: %.4f 元' % total_cost)
    # ========== 4. 保存到 result1.xlsx ==========
    with pd.ExcelWriter(os.path.join(OUT, 'ds_original_result1.xlsx')) as writer:
        # 计划购电量
        df_buy = pd.DataFrame({
            '时间': df['时间'],
            '购电量(kWh)': np.round(P_buy * dt, 4)
        })
        df_buy.to_excel(writer, sheet_name='计划购电量', index=False)
        # 充放电量：按指定时间段汇总
        periods = [(0,24), (24,48), (48,72), (72,96), (96,120), (120,144)]
        period_names = ['0:00-4:00','4:00-8:00','8:00-12:00',
                        '12:00-16:00','16:00-20:00','20:00-24:00']
        rows = []
        for (s, e), name in zip(periods, period_names):
            ch = np.sum(P_ch[s:e]) * dt
            dis = np.sum(P_dis[s:e]) * dt
            rows.append([name, round(ch,4), round(dis,4)])
        rows.append(['0:00储电量', round(E0,4), ''])
        rows.append(['24:00储电量', round(E[-1],4), ''])
        df_cd = pd.DataFrame(rows, columns=['时间段','充电量(kWh)','放电量(kWh)'])
        df_cd.to_excel(writer, sheet_name='充放电量', index=False)
    print('结果已保存至 result1.xlsx')
else:
    print('求解失败:', res.message)

# ---- 以下为审计追加（不影响原逻辑）----
if res.success:
    net = Load - PV
    balance = P_buy + PV + P_dis - (Load + P_ch)
    Ef = np.concatenate([[E0], E[:-1]])
    dyn = E - (Ef + eta * P_ch * dt - P_dis * dt / eta)
    print('\n[审计] 状态:', res.message)
    print('[审计] 功率平衡 max|residual| = %.3e kW' % np.max(np.abs(balance)))
    print('[审计] 储能动态 max|residual| = %.3e kWh' % np.max(np.abs(dyn)))
    print('[审计] 储电量范围 [%.4f, %.4f]，E_T = %.4f' % (E.min(), E.max(), E[-1]))
    print('[审计] 购电为负的时段数 = %d（负值即向电网售电，模型未禁止）' % int(np.sum(P_buy < -1e-6)))
    print('[审计] 同时充放时段数 = %d' % int(np.sum((P_ch > 1e-5) & (P_dis > 1e-5))))
    print('[审计] Σmax(0,-net)*dt = %.4f kWh（中午富余总量，被迫由储能全额消纳）'
          % (np.clip(-net, 0, None).sum() * dt))
    print('[审计] 充电总量 = %.4f kWh，放电总量 = %.4f kWh'
          % (P_ch.sum() * dt, P_dis.sum() * dt))
    print('\n[审计] 表1 按原文索引（1-based t=60,72,84,96,108,120）取值：')
    for lb, t in zip(['10:00-10:10', '12:00-12:10', '14:00-14:10',
                      '16:00-16:10', '18:00-18:10', '20:00-20:10'],
                     [60, 72, 84, 96, 108, 120]):
        print('        %s -> t=%d -> %.4f kWh' % (lb, t, P_buy[t-1] * dt))
    print('[审计] 表1 若按"时段末端"约定（t=61,73,85,97,109,121）取值：')
    for lb, t in zip(['10:00-10:10', '12:00-12:10', '14:00-14:10',
                      '16:00-16:10', '18:00-18:10', '20:00-20:10'],
                     [61, 73, 85, 97, 109, 121]):
        print('        %s -> t=%d -> %.4f kWh' % (lb, t, P_buy[t-1] * dt))
    np.savez(os.path.join(OUT, 'ds_original_solution.npz'),
             p=P_buy, c=P_ch, f=P_dis, E=E, pi=c, L=Load, v=PV)
