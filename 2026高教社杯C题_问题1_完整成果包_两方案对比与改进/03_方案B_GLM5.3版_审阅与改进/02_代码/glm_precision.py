# -*- coding: utf-8 -*-
"""CBC 默认容差 vs 收紧容差 vs HiGHS：精度对比（用于报告中的求解器精度讨论）。"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import load_attachment1, RES_DIR
from glm_model import solve_pulp, solve_highs

df = load_attachment1()
p, l, v = df['price'].to_numpy(), df['load'].to_numpy(), df['pv'].to_numpy()
idx = [60, 72, 84, 96, 108, 120]
res = {}
for tag, r in (('CBC-default', solve_pulp(p, l, v)),
               ('CBC-tight  ', solve_pulp(p, l, v, tight=True)),
               ('HiGHS      ', solve_highs(p, l, v))):
    bal = np.max(np.abs(r['buy_kW'] + v + r['dis_kW'] - (l + r['ch_kW'] + r['curtail_kW'])))
    dyn = np.max(np.abs(r['E_kWh'] - (np.concatenate([[6000.0], r['E_kWh'][:-1]])
                                      + 0.9 * r['ch_kW'] * (1 / 6) - r['dis_kW'] * (1 / 6) / 0.9)))
    print('%s 购电费=%.6f 元  平衡残差=%.2e kW  动态残差=%.2e kWh' % (tag, r['total_cost'], bal, dyn))
    print('             表1(6 个区间)= %s' % np.round(r['buy_kWh'][idx], 6))
    res[tag.strip()] = {'cost': r['total_cost'], 'bal': float(bal), 'dyn': float(dyn),
                        'table1': [float(x) for x in r['buy_kWh'][idx]]}
import json
with open(os.path.join(RES_DIR, 'glm_solver_precision.json'), 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print('\nwritten results/glm_solver_precision.json')
