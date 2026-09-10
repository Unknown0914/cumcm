# -*- coding: utf-8 -*-
"""
glm_verify.py —— 问题1（GLM 方案路线）解的独立校验

V1 功率平衡   V2 储能动态   V3 储电量界   V4 功率界   V5 购电非负
V6 弃电界（含 GLM 原方案缺失的 d<=v）      V7 首尾储电量   V8 充放电互斥
V9 目标复算   V10 电量守恒   V10b 效率闭合
V11 result1.xlsx 严格套模板（表头/行数/A列标签/数值逐行）
V12 两套求解器交叉验证（PuLP-CBC vs scipy-HiGHS）
V13 GLM 原式（d 无上界）与修正版最优值一致
V14 与第一轮（scipy-HiGHS 独立成果包）结果一致
"""
import sys, os, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import openpyxl

from common import (load_attachment1, DT, N, E_INIT, E_MIN, E_MAX, P_MAX,
                    RES_DIR, OUT_DIR, TPL_RESULT1, summarize_periods_bounds)
from glm_model import solve_pulp, solve_highs

lines, ok_all = [], True


def check(name, ok, detail=''):
    global ok_all
    ok_all = ok_all and bool(ok)
    lines.append('[%s] %s %s' % ('PASS' if ok else 'FAIL', name, detail))


def main():
    df = load_attachment1()
    price, load, pv = df['price'].to_numpy(), df['load'].to_numpy(), df['pv'].to_numpy()
    r = solve_pulp(price, load, pv)
    p, c, f, d, E = r['buy_kW'], r['ch_kW'], r['dis_kW'], r['curtail_kW'], r['E_kWh']

    m1 = np.max(np.abs(p + pv + f - (load + c + d)))
    check('V1 功率平衡 p+v+f = L+c+d', m1 < 1e-3, 'max|residual| = %.3e kW（CBC 容差量级）' % m1)

    Eprev = np.concatenate([[E_INIT], E[:-1]])
    m2 = np.max(np.abs(E - (Eprev + 0.9 * c * DT - f * DT / 0.9)))
    check('V2 储能动态 E_t = E_{t-1}+0.9cΔt-fΔt/0.9', m2 < 1e-3, 'max|residual| = %.3e kWh' % m2)

    check('V3 储电量界 1200 ≤ E_t ≤ 10800', E.min() >= E_MIN - 1e-4 and E.max() <= E_MAX + 1e-4,
          'E ∈ [%.4f, %.4f]' % (E.min(), E.max()))
    check('V4 充放电功率 0 ≤ c,f ≤ 5000',
          c.min() >= -1e-6 and c.max() <= P_MAX + 1e-4 and f.min() >= -1e-6 and f.max() <= P_MAX + 1e-4,
          'c_max=%.2f, f_max=%.2f' % (c.max(), f.max()))
    check('V5 购电非负 p_t ≥ 0', p.min() >= -1e-6, 'p_min=%.3e kW' % p.min())
    check('V6 弃电界 0 ≤ d_t ≤ v_t（GLM 原文缺失的约束）',
          d.min() >= -1e-6 and np.all(d <= pv + 1e-6),
          'd_max=%.4f kW，弃电总量 %.4f kWh' % (d.max(), float(d.sum() * DT)))
    check('V7 首尾储电量 E_0 = E_T = 6000', abs(E[-1] - E_INIT) < 1e-3,
          'E_0=%.4f, E_T=%.4f' % (E_INIT, E[-1]))
    n_both = int(np.sum((c > 1e-5) & (f > 1e-5)))
    check('V8 充放电互斥（LP 自动满足）', n_both == 0, '同时充放时段数 = %d' % n_both)
    obj = float((price * p * DT).sum())
    check('V9 目标函数复算', abs(obj - r['total_cost']) < 1e-4, '%.6f 元' % obj)
    lhs = float(p.sum() * DT)
    rhs = float((load - pv).sum() * DT) + float(c.sum() * DT - f.sum() * DT)
    check('V10 电量守恒 ΣpΔt = Σ(L-v)Δt + (ΣcΔt-ΣfΔt)', abs(lhs - rhs) < 1e-3,
          '%.6f vs %.6f kWh' % (lhs, rhs))
    check('V10b 效率闭合 0.9·ΣcΔt = ΣfΔt/0.9',
          abs(0.9 * float(c.sum() * DT) - float(f.sum() * DT) / 0.9) < 1e-3,
          '%.6f kWh' % (0.9 * float(c.sum() * DT)))

    # V11 模板合规性
    xls = os.path.join(OUT_DIR, 'result1.xlsx')
    tpl = openpyxl.load_workbook(TPL_RESULT1)
    wb = openpyxl.load_workbook(xls)
    if os.path.exists(xls):
        same_sheets = wb.sheetnames == tpl.sheetnames
        ws, wst = wb['计划购电量'], tpl['计划购电量']
        ws2, ws2t = wb['充放电量'], tpl['充放电量']
        hdr_same = ([ws.cell(1, cc).value for cc in (1, 2)] == [wst.cell(1, cc).value for cc in (1, 2)]
                    and [ws2.cell(1, cc).value for cc in range(1, 6)] ==
                        [ws2t.cell(1, cc).value for cc in range(1, 6)])
        label_same = all(ws.cell(i + 2, 1).value == wst.cell(i + 2, 1).value for i in range(N))
        dims_same = (ws.max_row, ws.max_column, ws2.max_row, ws2.max_column) == \
                    (wst.max_row, wst.max_column, ws2t.max_row, ws2t.max_column)
        err = max(abs(float(ws.cell(i + 2, 2).value) - float(r['buy_kWh'][i])) for i in range(N))
        check('V11a 工作表名与表头与模板一致', same_sheets and hdr_same,
              'sheets=%s, 表头 %s' % (wb.sheetnames, [ws.cell(1, cc).value for cc in (1, 2)]))
        check('V11b A 列 144 个时间标签与模板逐行一致', label_same, '计划购电量 A2..A145')
        check('V11c 维度与模板一致 (145×2 / 7×5)', dims_same,
              '实际 %d×%d / %d×%d' % (ws.max_row, ws.max_column, ws2.max_row, ws2.max_column))
        check('V11d 购电量数值与解逐行一致', err < 1e-4, 'max|diff| = %.2e' % err)
        tb2 = [dict(charge=float(r['ch_kWh'][a - 1:b].sum()), dis=float(r['dis_kWh'][a - 1:b].sum()))
               for _, a, b in summarize_periods_bounds()]
        e2 = max(abs(float(ws2.cell(k + 2, 2).value) - tb2[k]['charge']) for k in range(6))
        e3 = max(abs(float(ws2.cell(k + 2, 3).value) - tb2[k]['dis']) for k in range(6))
        e4 = abs(float(ws2.cell(2, 5).value) - E_INIT)
        e5 = abs(float(ws2.cell(3, 5).value) - float(E[-1]))
        check('V11e 充放电量/储电量与解一致', max(e2, e3, e4, e5) < 1e-4,
              'max|diff| = %.2e' % max(e2, e3, e4, e5))
    else:
        check('V11 result1.xlsx 存在', False, xls)

    # V12 交叉验证
    rh = solve_highs(price, load, pv)
    check('V12 两套求解器交叉验证（PuLP-CBC vs scipy-HiGHS）',
          abs(rh['total_cost'] - r['total_cost']) < 1e-4,
          '%.6f vs %.6f 元，差 %.2e' % (r['total_cost'], rh['total_cost'],
                                        abs(rh['total_cost'] - r['total_cost'])))

    # V13 GLM 原式
    rr = solve_pulp(price, load, pv, curtail_upper=False)
    check('V13 GLM 原式（d 无上界）与修正版最优值一致',
          abs(rr['total_cost'] - r['total_cost']) < 1e-4,
          '%.6f vs %.6f 元' % (rr['total_cost'], r['total_cost']))

    # V14 与第一轮（独立成果包）一致
    ref = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       'cumcm2026C_wq', 'results', 'q1_summary.json')
    if os.path.exists(ref):
        with open(ref, encoding='utf-8') as fh:
            refj = json.load(fh)
        check('V14 与第一轮独立成果（scipy-HiGHS）一致',
              abs(refj['total_cost_yuan'] - r['total_cost']) < 1e-3,
              '%.6f vs %.6f 元' % (refj['total_cost_yuan'], r['total_cost']))
    else:
        lines.append('[SKIP] V14 与第一轮独立成果比对 —— 未找到参考文件 %s（换机运行时可忽略）' % ref)
        return _finish()

    return _finish()


def _finish():
    out = '\n'.join(lines)
    print(out)
    with open(os.path.join(RES_DIR, 'glm_validation.txt'), 'w', encoding='utf-8') as fh:
        fh.write('问题1（GLM 方案路线）解的校验报告\n' + '=' * 64 + '\n' + out + '\n')
    print('\n总体结果:', 'ALL PASS' if ok_all else 'HAS FAILURES')
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
