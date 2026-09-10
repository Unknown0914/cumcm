# -*- coding: utf-8 -*-
"""
q1_sensitivity.py —— 关键参数敏感性分析

考察储能"最大充放电功率 P_max"与"储电量上界 E_max"对最优化结果的影响，
用于说明储能配置对购电费用与光伏消纳的作用。
"""
import sys, os, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import linprog
from common import load_attachment1, DT, N, RES_DIR
from q1_model import build_lp, IDX_E, IDX_G, IDX_D

df = load_attachment1()
price, load, pv = df['price'].to_numpy(), df['load'].to_numpy(), df['pv'].to_numpy()


def run(p_max=5000.0, e_max=10800.0, e_min=1200.0):
    c, A_eq, b_eq, bounds = build_lp(price, load, pv, allow_curtail=True)
    bounds[IDX_G:IDX_G + N] = [(0.0, p_max)] * N
    bounds[IDX_D:IDX_D + N] = [(0.0, p_max)] * N
    bounds[IDX_E:IDX_E + N] = [(e_min, e_max)] * N
    r = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    assert r.success, r.message
    b = r.x[:N]
    ch = r.x[IDX_G:IDX_G + N]
    dis = r.x[IDX_D:IDX_D + N]
    u = r.x[3 * N:4 * N]
    return {
        'p_max': p_max, 'e_max': e_max,
        'buy_kWh': float(np.sum(b) * DT),
        'cost': float(np.sum(price * b) * DT),
        'charge_kWh': float(np.sum(ch) * DT),
        'discharge_kWh': float(np.sum(dis) * DT),
        'curtail_kWh': float(np.sum(u) * DT),
    }


rows = []
print('--- 灵敏度 1：最大充放电功率 P_max（E_max=10800）---')
print('P_max  购电量kWh   购电费元   充电kWh   放电kWh   弃光kWh')
for pm in [0, 1250, 2500, 3750, 5000, 6250, 7500]:
    d = run(p_max=float(pm))
    d['kind'] = 'p_max'
    rows.append(d)
    print('%6.0f %10.2f %10.2f %9.2f %9.2f %8.2f' % (
        pm, d['buy_kWh'], d['cost'], d['charge_kWh'], d['discharge_kWh'], d['curtail_kWh']))

print('\n--- 灵敏度 2：储电量上界 E_max（P_max=5000, E_min=1200）---')
print('E_max  购电量kWh   购电费元   充电kWh   放电kWh   弃光kWh')
for em in [6000, 7200, 8400, 9600, 10800, 12000]:
    d = run(e_max=float(em))
    d['kind'] = 'e_max'
    rows.append(d)
    print('%6.0f %10.2f %10.2f %9.2f %9.2f %8.2f' % (
        em, d['buy_kWh'], d['cost'], d['charge_kWh'], d['discharge_kWh'], d['curtail_kWh']))

with open(os.path.join(RES_DIR, 'q1_sensitivity.json'), 'w', encoding='utf-8') as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print('\nwritten results/q1_sensitivity.json')
