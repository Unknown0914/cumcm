# -*- coding: utf-8 -*-
"""
glm_sensitivity.py —— 问题1（GLM 方案路线）敏感性分析

S1 效率口径对比（GLM 在"结果检验要点 5"中建议）
     double      : eta_c = eta_d = 0.9          往返效率 81%（GLM 主口径）
     single      : eta_c = 0.9, eta_d = 1.0      仅充电侧 90%（GLM 建议的对照口径）
     roundtrip90 : eta_c = eta_d = sqrt(0.9)     往返效率 90%
S2 弃电变量处理方式对比（GLM 在"结果检验要点 3"中建议的验证）
     d <= v（修正版） / d 无上界（GLM 原式） / d ≡ 0（固定弃电为 0 重解）
S3 最大充放电功率 P_max
S4 储电量上界 E_max

所有子问题同时用 PuLP-CBC 与 scipy-HiGHS 求解，交叉验证最优值。
"""
import sys, os, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import load_attachment1, DT, RES_DIR
from glm_model import solve_pulp, solve_highs, ETA_PRESETS

df = load_attachment1()
price, load, pv = df['price'].to_numpy(), df['load'].to_numpy(), df['pv'].to_numpy()

rows = []


def both(tag, **kw):
    a = solve_pulp(price, load, pv, **kw)
    b = solve_highs(price, load, pv, **kw)
    rec = {
        'group': tag, 'eta_c': a['eta_c'], 'eta_d': a['eta_d'],
        'p_max': a['p_max'], 'e_max': a['e_max'],
        'cost_cbc': a['total_cost'], 'cost_highs': b['total_cost'],
        'buy_kWh': a['total_buy_kWh'],
        'charge_kWh': a['total_charge_kWh'], 'discharge_kWh': a['total_discharge_kWh'],
        'curtail_kWh': a['total_curtail_kWh'],
        'delta_yuan': abs(a['total_cost'] - b['total_cost']),
    }
    rows.append(rec)
    return rec


print('=' * 88)
print('S1 效率口径对比（GLM 建议的敏感性分析）')
print('=' * 88)
print('%-14s %-10s %-10s %14s %14s %12s %12s %10s' %
      ('口径', 'eta_c', 'eta_d', '购电费-CBC(元)', '购电费-HiGHS(元)', '充电kWh', '放电kWh', '弃光kWh'))
for name, (ec, ed) in ETA_PRESETS.items():
    r = both('eta:' + name, eta_c=ec, eta_d=ed)
    print('%-14s %-10.4f %-10.4f %14.4f %14.4f %12.2f %12.2f %10.2f' %
          (name, ec, ed, r['cost_cbc'], r['cost_highs'], r['charge_kWh'], r['discharge_kWh'],
           r['curtail_kWh']))

print()
print('=' * 88)
print('S2 弃电变量处理方式对比（GLM 建议的 d≡0 验证）')
print('=' * 88)
for name, kw in (('d<=v（修正版）', dict(curtail_upper=True)),
                 ('d 无上界（GLM 原式）', dict(curtail_upper=False)),
                 ('d≡0（固定为0重解）', dict(allow_curtail=False))):
    r = both('curtail:' + name, **kw)
    print('%-22s 购电费 CBC %14.6f 元 | HiGHS %14.6f 元 | 弃光 %10.4f kWh  | 价差 %.2e' %
          (name, r['cost_cbc'], r['cost_highs'], r['curtail_kWh'], r['delta_yuan']))

print()
print('=' * 88)
print('S3 最大充放电功率 P_max（eta=0.9/0.9, E_max=10800）')
print('=' * 88)
print('%10s %14s %12s %10s' % ('P_max(kW)', '购电费(元)', '弃光(kWh)', 'HiGHS校验'))
for pm in [0, 1250, 2500, 3750, 5000, 6250, 7500]:
    r = both('pmax', p_max=float(pm))
    print('%10.0f %14.4f %12.4f %10.2e' % (pm, r['cost_cbc'], r['curtail_kWh'], r['delta_yuan']))

print()
print('=' * 88)
print('S4 储电量上界 E_max（eta=0.9/0.9, P_max=5000, E_min=1200）')
print('=' * 88)
print('%10s %14s %12s %10s' % ('E_max(kWh)', '购电费(元)', '弃光(kWh)', 'HiGHS校验'))
for em in [6000, 7200, 8400, 9600, 10800, 12000]:
    r = both('emax', e_max=float(em))
    print('%10.0f %14.4f %12.4f %10.2e' % (em, r['cost_cbc'], r['curtail_kWh'], r['delta_yuan']))

with open(os.path.join(RES_DIR, 'glm_sensitivity.json'), 'w', encoding='utf-8') as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print('\nwritten results/glm_sensitivity.json  (%d 组实验)' % len(rows))
