# -*- coding: utf-8 -*-
"""
glm_solve.py —— 问题1（GLM 方案路线）求解 + 严格套模板导出

运行：  python code/glm_solve.py

输出：
  output/result1.xlsx              改进版结果文件（严格套用附件5 模板）
  output/result1_glm_original.xlsx GLM 原始代码导出的文件（对照用）
  results/glm_timeseries.csv       144 时段全部决策量与状态量
  results/glm_table1.csv           论文表1
  results/glm_table2.csv           论文表2
  results/glm_summary.json         关键指标与两套求解器交叉验证结果
"""
import sys, os, io, json, csv, shutil
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import openpyxl

from common import (load_attachment1, DT, N, E_INIT, RES_DIR, OUT_DIR,
                    TPL_RESULT1, TABLE1_CLOCKS, clock_to_t, t_to_interval,
                    summarize_periods_bounds)
from glm_model import solve_pulp, solve_highs

# 论文表1：指定 10 分钟区间 -> 0-based 时段下标（"时段末端"约定，见报告 §3）
IDX_TABLE1 = [clock_to_t(c) - 1 for c in TABLE1_CLOCKS]


def aggregate(res):
    t1 = [{'interval': '%s-%s' % (c, t_to_interval(clock_to_t(c)).split('-')[1]),
           't': clock_to_t(c), 'buy_kWh': float(res['buy_kWh'][clock_to_t(c) - 1])}
          for c in TABLE1_CLOCKS]
    t2 = [{'interval': nm,
           'charge_kWh': float(res['ch_kWh'][a - 1:b].sum()),
           'discharge_kWh': float(res['dis_kWh'][a - 1:b].sum())}
          for nm, a, b in summarize_periods_bounds()]
    return t1, t2


def write_result1(res, table2, path):
    """严格按附件5 模板填写：保留工作表名、表头、A 列标签与行序。"""
    shutil.copyfile(TPL_RESULT1, path)
    wb = openpyxl.load_workbook(path)

    ws = wb['计划购电量']                      # 模板 A1:B145，A 列为 "0:10-0:20"…
    for i in range(N):
        ws.cell(row=i + 2, column=2).value = round(float(res['buy_kWh'][i]), 4)

    ws = wb['充放电量']                        # 模板 A1:E7
    for k, row in enumerate(table2):
        ws.cell(row=k + 2, column=2).value = round(row['charge_kWh'], 4)
        ws.cell(row=k + 2, column=3).value = round(row['discharge_kWh'], 4)
    ws.cell(row=2, column=5).value = round(float(E_INIT), 4)
    ws.cell(row=3, column=5).value = round(float(res['E_T']), 4)
    wb.save(path)
    return path


def write_csvs(res, t1, t2, price, load, pv):
    p = os.path.join(RES_DIR, 'glm_timeseries.csv')
    with open(p, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(['t', 'interval', 'price_yuan_per_kWh', 'load_kW', 'pv_kW', 'net_load_kW',
                    'buy_kW', 'buy_kWh', 'ch_kW', 'ch_kWh', 'dis_kW', 'dis_kWh',
                    'curtail_kW', 'curtail_kWh', 'E_end_kWh'])
        for i in range(N):
            w.writerow([i + 1, t_to_interval(i + 1), '%.4f' % price[i], '%.4f' % load[i],
                        '%.4f' % pv[i], '%.4f' % (load[i] - pv[i]),
                        '%.6f' % res['buy_kW'][i], '%.6f' % res['buy_kWh'][i],
                        '%.6f' % res['ch_kW'][i], '%.6f' % res['ch_kWh'][i],
                        '%.6f' % res['dis_kW'][i], '%.6f' % res['dis_kWh'][i],
                        '%.6f' % res['curtail_kW'][i], '%.6f' % res['curtail_kWh'][i],
                        '%.6f' % res['E_kWh'][i]])

    with open(os.path.join(RES_DIR, 'glm_table1.csv'), 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh); w.writerow(['时间段', '购电量(kWh)'])
        for r in t1:
            w.writerow([r['interval'], '%.4f' % r['buy_kWh']])
        w.writerow(['全天购电量(kWh)', '%.4f' % res['total_buy_kWh']])
        w.writerow(['全天购电费(元)', '%.4f' % res['total_cost']])

    with open(os.path.join(RES_DIR, 'glm_table2.csv'), 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh); w.writerow(['时间段', '充电量(kWh)', '放电量(kWh)'])
        for r in t2:
            w.writerow([r['interval'], '%.4f' % r['charge_kWh'], '%.4f' % r['discharge_kWh']])
        w.writerow(['0:00储电量(kWh)', '%.4f' % E_INIT, ''])
        w.writerow(['24:00储电量(kWh)', '%.4f' % res['E_T'], ''])


def main():
    df = load_attachment1()
    price, load, pv = df['price'].to_numpy(), df['load'].to_numpy(), df['pv'].to_numpy()

    res_cbc = solve_pulp(price, load, pv)                 # GLM 路线（已修正 d<=v）
    res_hig = solve_highs(price, load, pv)                # 交叉验证
    res_glm_raw = solve_pulp(price, load, pv, curtail_upper=False)   # 复现 GLM 原式（d 无上界）

    t1, t2 = aggregate(res_cbc)
    xlsx = write_result1(res_cbc, t2, os.path.join(OUT_DIR, 'result1.xlsx'))
    write_csvs(res_cbc, t1, t2, price, load, pv)

    base_curve = (price * np.clip(load - pv, 0, None) * DT).sum()   # 无储能且不弃光（下界参考）
    summary = {
        'solver': res_cbc['solver'],
        'total_buy_kWh': res_cbc['total_buy_kWh'],
        'total_cost_yuan': res_cbc['total_cost'],
        'avg_price': res_cbc['total_cost'] / res_cbc['total_buy_kWh'],
        'total_charge_kWh': res_cbc['total_charge_kWh'],
        'total_discharge_kWh': res_cbc['total_discharge_kWh'],
        'storage_loss_kWh': res_cbc['total_charge_kWh'] - res_cbc['total_discharge_kWh'],
        'total_curtail_kWh': res_cbc['total_curtail_kWh'],
        'pv_total_kWh': res_cbc['pv_total_kWh'],
        'pv_used_kWh': res_cbc['pv_used_kWh'],
        'pv_utilization': res_cbc['pv_used_kWh'] / res_cbc['pv_total_kWh'],
        'load_total_kWh': res_cbc['load_total_kWh'],
        'E_min': res_cbc['E_min'], 'E_max': res_cbc['E_max'],
        'E_0': res_cbc['E0'], 'E_T': res_cbc['E_T'],
        'cross_check': {
            'pulp_cbc_cost': res_cbc['total_cost'],
            'scipy_highs_cost': res_hig['total_cost'],
            'abs_diff_yuan': abs(res_cbc['total_cost'] - res_hig['total_cost']),
            'pulp_cbc_buy_kWh': res_cbc['total_buy_kWh'],
            'scipy_highs_buy_kWh': res_hig['total_buy_kWh'],
        },
        'glm_original_form': {          # d 无上界的 GLM 原式
            'cost': res_glm_raw['total_cost'],
            'buy_kWh': res_glm_raw['total_buy_kWh'],
            'curtail_kWh': res_glm_raw['total_curtail_kWh'],
        },
        'table1': t1, 'table2': t2,
    }
    with open(os.path.join(RES_DIR, 'glm_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('求解器:', res_cbc['solver'], '| 状态:', res_cbc['status'])
    print('全天购电量 = %.4f kWh' % summary['total_buy_kWh'])
    print('全天购电费 = %.4f 元 (均价 %.4f 元/kWh)' % (summary['total_cost_yuan'], summary['avg_price']))
    print('储能充/放 = %.4f / %.4f kWh，循环损耗 %.4f kWh'
          % (summary['total_charge_kWh'], summary['total_discharge_kWh'], summary['storage_loss_kWh']))
    print('光伏消纳 = %.4f / %.4f kWh (%.2f%%)，弃光 %.4f kWh'
          % (summary['pv_used_kWh'], summary['pv_total_kWh'], summary['pv_utilization'] * 100,
             summary['total_curtail_kWh']))
    print('储电量 [%.2f, %.2f]，0:00=%.2f，24:00=%.2f'
          % (summary['E_min'], summary['E_max'], summary['E_0'], summary['E_T']))
    print('交叉验证：PuLP-CBC %.6f 元 vs scipy-HiGHS %.6f 元，差 %.2e 元'
          % (summary['cross_check']['pulp_cbc_cost'], summary['cross_check']['scipy_highs_cost'],
             summary['cross_check']['abs_diff_yuan']))
    print('GLM 原式（d 无上界）购电费 = %.6f 元，弃光 %.4f kWh -> 与修正版%s'
          % (res_glm_raw['total_cost'], res_glm_raw['total_curtail_kWh'],
             '完全一致' if abs(res_glm_raw['total_cost'] - res_cbc['total_cost']) < 1e-6 else '不同'))
    print('已写出:', xlsx)


if __name__ == '__main__':
    main()
