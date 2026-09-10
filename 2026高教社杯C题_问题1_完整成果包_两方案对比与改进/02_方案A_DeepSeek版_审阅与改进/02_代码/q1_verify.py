# -*- coding: utf-8 -*-
"""
q1_verify.py —— 问题1 解的独立校验（含 result1.xlsx 回读核对）

校验清单：
  V1  功率平衡      b_t + (PV_t - u_t) + d_t = L_t + g_t
  V2  储能动态      E_t = E_{t-1} + η_c g_t Δt - d_t Δt / η_d
  V3  储电量上下界  1200 ≤ E_t ≤ 10800
  V4  充放电功率界  0 ≤ g_t, d_t ≤ 5000
  V5  购电非负      b_t ≥ 0
  V6  弃光界        0 ≤ u_t ≤ PV_t
  V7  初末储电量    E_0 = E_T = 6000
  V8  充放电互斥    不存在同时 g_t>0 与 d_t>0
  V9  目标复算      Σ c_t b_t Δt 与求解器目标值一致
  V10 电量守恒      Σ b Δt = Σ (L-PV) Δt + (Σ g Δt - Σ d Δt)
  V11 result1.xlsx  回读与计算结果逐行一致
"""
import sys, os, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import openpyxl

from common import (load_attachment1, DT, N, E_INIT, E_MIN, E_MAX, P_MAX,
                    ETA_C, ETA_D, RES_DIR, OUT_DIR, TPL_RESULT1,
                    summarize_periods_bounds)
from q1_model import solve

TOL = 1e-6
lines = []
ok_all = True


def check(name, ok, detail=''):
    global ok_all
    ok_all = ok_all and ok
    lines.append('[%s] %s %s' % ('PASS' if ok else 'FAIL', name, detail))
    return ok


def main():
    df = load_attachment1()
    price, load, pv = df['price'].to_numpy(), df['load'].to_numpy(), df['pv'].to_numpy()
    r = solve(price, load, pv, allow_curtail=True, allow_sell=False)
    assert r['success']

    b, g, d, u, E = r['buy_kW'], r['ch_kW'], r['dis_kW'], r['curtail_kW'], r['E_kWh']

    # V1 功率平衡
    m1 = np.max(np.abs(b + (pv - u) + d - (load + g)))
    check('V1 功率平衡 b+(PV-u)+d = L+g', m1 < 1e-6, 'max|residual| = %.3e kW' % m1)

    # V2 储能动态
    Eprev = np.concatenate([[E_INIT], E[:-1]])
    m2 = np.max(np.abs(E - (Eprev + ETA_C * g * DT - d * DT / ETA_D)))
    check('V2 储能动态 E_t = E_{t-1}+η_c gΔt-dΔt/η_d', m2 < 1e-6, 'max|residual| = %.3e kWh' % m2)

    # V3 储电量界
    check('V3 储电量界 1200 ≤ E_t ≤ 10800',
          E.min() >= E_MIN - TOL and E.max() <= E_MAX + TOL,
          'E ∈ [%.4f, %.4f]' % (E.min(), E.max()))

    # V4 功率界
    check('V4 充放电功率 0 ≤ g,d ≤ 5000',
          g.min() >= -TOL and g.max() <= P_MAX + TOL and d.min() >= -TOL and d.max() <= P_MAX + TOL,
          'g_max=%.2f, d_max=%.2f' % (g.max(), d.max()))

    # V5 购电非负
    check('V5 购电非负 b_t ≥ 0', b.min() >= -TOL, 'b_min = %.3e kW' % b.min())

    # V6 弃光界
    check('V6 弃光 0 ≤ u_t ≤ PV_t', u.min() >= -TOL and np.all(u <= pv + TOL),
          'u_max=%.3f, 弃光总量=%.4f kWh' % (u.max(), float(u.sum() * DT)))

    # V7 初末储电量
    check('V7 初末储电量 E_0 = E_T = 6000',
          abs(E_INIT - 6000) < TOL and abs(E[-1] - E_INIT) < 1e-4,
          'E_0 = %.4f, E_T = %.4f' % (E_INIT, E[-1]))

    # V8 充放电互斥
    n_both = int(np.sum((g > 1e-5) & (d > 1e-5)))
    check('V8 充放电互斥（无同时充放电时段）', n_both == 0, '同时充放时段数 = %d' % n_both)

    # V9 目标复算
    obj = float(np.sum(price * b) * DT)
    check('V9 目标函数复算', abs(obj - r['total_cost']) < 1e-4,
          '复算 %.6f 元 vs 求解器 %.6f 元' % (obj, r['total_cost']))

    # V10 电量守恒
    lhs = float(b.sum() * DT)
    rhs = float((load - pv).sum() * DT) + float(g.sum() * DT - d.sum() * DT)
    check('V10 电量守恒 ΣbΔt = Σ(L-PV)Δt + (ΣgΔt-ΣdΔt)',
          abs(lhs - rhs) < 1e-6, '%.6f vs %.6f kWh' % (lhs, rhs))

    # 效率关系：η_c·ΣgΔt = ΣdΔt/η_d
    lhs2 = ETA_C * float(g.sum() * DT)
    rhs2 = float(d.sum() * DT) / ETA_D
    check('V10b 效率闭合 η_c·ΣgΔt = ΣdΔt/η_d', abs(lhs2 - rhs2) < 1e-4,
          '%.6f vs %.6f kWh' % (lhs2, rhs2))

    # V11 result1.xlsx 回读
    xls = os.path.join(OUT_DIR, 'result1.xlsx')
    if os.path.exists(xls):
        wb = openpyxl.load_workbook(xls)
        ws = wb['计划购电量']
        err = 0.0
        for i in range(N):
            v = ws.cell(row=i + 2, column=2).value
            err = max(err, abs(float(v) - float(r['buy_kWh'][i])))
        check('V11a result1.xlsx 计划购电量逐行一致 (%d 行)' % N, err < 1e-4, 'max|diff| = %.2e' % err)

        ws2 = wb['充放电量']
        tb2 = [dict(interval=n, charge_kWh=float(r['ch_kWh'][a - 1:b_].sum()),
                    discharge_kWh=float(r['dis_kWh'][a - 1:b_].sum()))
               for n, a, b_ in summarize_periods_bounds()]
        e2 = max(abs(float(ws2.cell(row=k + 2, column=2).value) - tb2[k]['charge_kWh']) for k in range(6))
        e3 = max(abs(float(ws2.cell(row=k + 2, column=3).value) - tb2[k]['discharge_kWh']) for k in range(6))
        e4 = abs(float(ws2.cell(row=2, column=5).value) - E_INIT)
        e5 = abs(float(ws2.cell(row=3, column=5).value) - float(E[-1]))
        check('V11b result1.xlsx 充放电量表一致', max(e2, e3, e4, e5) < 1e-4,
              'max|diff| = %.2e (充电), %.2e (放电), %.2e/%.2e (储电量)' % (e2, e3, e4, e5))

        # 模板结构保持
        check('V11c result1.xlsx 保持模板工作表与行数',
              wb.sheetnames == ['计划购电量', '充放电量'] and ws.max_row == 145 and ws2.max_row == 7,
              'sheets=%s, 计划购电量行数=%d, 充放电量行数=%d' % (wb.sheetnames, ws.max_row, ws2.max_row))
    else:
        check('V11 result1.xlsx 存在', False, '未找到 %s' % xls)

    out = '\n'.join(lines)
    print(out)
    with open(os.path.join(RES_DIR, 'q1_validation.txt'), 'w', encoding='utf-8') as f:
        f.write('问题1 解的校验报告\n' + '=' * 60 + '\n' + out + '\n')
    print('\n总体结果:', 'ALL PASS' if ok_all else 'HAS FAILURES')
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
