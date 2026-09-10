# -*- coding: utf-8 -*-
"""
q1_figures.py —— 问题1 论文用图表（300 dpi PNG + 矢量 PDF）

Fig1 全天电价与净负荷曲线
Fig2 小区负载与光伏发电功率曲线
Fig3 最优计划购电功率与储能充放电功率（含电价背景）
Fig4 储能储电量轨迹与充放电功率（含上下限）
Fig5 各 10 分钟时段购电量与电价
Fig6 有/无储能方案对比
Fig7 全天能量来源构成（堆叠面积）
"""
import sys, os, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

from common import load_attachment1, DT, N, FIG_DIR, RES_DIR, E_INIT, E_MIN, E_MAX
from q1_model import solve

# ---------------------------------------------------------------------------
# 字体：优先使用系统可用的中文字体
# ---------------------------------------------------------------------------
_CAND = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Source Han Sans SC', 'SimSun']
_avail = {f.name for f in font_manager.fontManager.ttflist}
_PICK = next((c for c in _CAND if c in _avail), None)
if _PICK:
    plt.rcParams['font.sans-serif'] = [_PICK]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

C_PRICE = '#c0392b'
C_LOAD = '#2c3e50'
C_PV = '#e67e22'
C_BUY = '#2980b9'
C_CH = '#27ae60'
C_DIS = '#8e44ad'
C_E = '#16a085'

HOURS = np.arange(1, N + 1) * DT          # 时段右端点，单位 h（0.1667 ... 24.0）


def _save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG_DIR, '%s.%s' % (name, ext)),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('figure ->', name)


def _xticks(ax):
    ax.set_xticks(range(0, 25, 2))
    ax.set_xlim(0, 24)
    ax.set_xlabel('时刻 (h)')


def fig1(price, net):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.fill_between(HOURS, 0, net, where=(net >= 0), color='#95a5a6', alpha=0.35,
                    label='正净负荷（需购电/放电）')
    ax.fill_between(HOURS, 0, net, where=(net < 0), color='#e74c3c', alpha=0.35,
                    label='负净负荷（光伏富余）')
    ax.plot(HOURS, net, color=C_LOAD, lw=1.6, label='净负荷 $L_t-PV_t$')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylabel('净负荷 (kW)')
    ax2 = ax.twinx()
    ax2.step(HOURS, price, where='post', color=C_PRICE, lw=1.8, label='电价')
    ax2.set_ylabel('电价 (元/kWh)', color=C_PRICE)
    ax2.tick_params(axis='y', colors=C_PRICE)
    ax2.grid(False)
    _xticks(ax)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=9, framealpha=0.9)
    ax.set_title('图1  全天电价与净负荷曲线（附件1）')
    _save(fig, 'fig1_price_netload')


def fig2(load, pv):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(HOURS, load, color=C_LOAD, lw=1.8, label='小区负载 $L_t$')
    ax.plot(HOURS, pv, color=C_PV, lw=1.8, label='光伏发电预测 $PV_t$')
    ax.fill_between(HOURS, 0, pv, color=C_PV, alpha=0.18)
    ax.fill_between(HOURS, pv, load, where=(load > pv), color='#bdc3c7', alpha=0.35,
                    label='需由电网/储能补足')
    ax.set_ylabel('功率 (kW)')
    _xticks(ax)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title('图2  小区负载与光伏发电预测功率')
    _save(fig, 'fig2_load_pv')


def fig3(price, buy, ch, dis):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                  gridspec_kw={'height_ratios': [1.15, 1]})
    ax.fill_between(HOURS, 0, buy, color=C_BUY, alpha=0.55, label='计划购电功率 $b_t$')
    ax.plot(HOURS, buy, color=C_BUY, lw=1.4)
    ax.set_ylabel('购电功率 (kW)')
    ax.legend(loc='upper left', fontsize=9)
    ax3 = ax.twinx()
    ax3.step(HOURS, price, where='post', color=C_PRICE, lw=1.4, alpha=0.85, label='电价')
    ax3.set_ylabel('电价 (元/kWh)', color=C_PRICE)
    ax3.tick_params(axis='y', colors=C_PRICE)
    ax3.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=9)
    ax.set_title('图3  最优计划购电策略')

    ax2.bar(HOURS, ch, width=DT * 0.92, color=C_CH, label='充电功率 $g_t$')
    ax2.bar(HOURS, -dis, width=DT * 0.92, color=C_DIS, label='放电功率 $d_t$')
    ax2.axhline(0, color='k', lw=0.8)
    ax2.set_ylabel('储能充(+)/放(-)功率 (kW)')
    ax2.legend(loc='upper left', fontsize=9, ncol=2)
    _xticks(ax2)
    _save(fig, 'fig3_dispatch')


def fig4(ch, dis, E):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(HOURS, E, color=C_E, lw=2.0, label='储电量 $E_t$')
    ax.axhline(E_MAX, color='#7f8c8d', ls='--', lw=1.2, label='上限 10800 kWh')
    ax.axhline(E_MIN, color='#7f8c8d', ls=':', lw=1.2, label='下限 1200 kWh')
    ax.plot([0], [E_INIT], 'o', color=C_E, ms=6)
    ax.annotate('$E_0=6000$', (0, E_INIT), textcoords='offset points', xytext=(6, -14))
    ax.plot([24], [E[-1]], 's', color=C_E, ms=6)
    ax.annotate('$E_{24}=6000$', (24, E[-1]), textcoords='offset points', xytext=(-52, 8))
    ax.set_ylabel('储电量 (kWh)')
    ax.set_ylim(0, 12000)
    ax2 = ax.twinx()
    ax2.bar(HOURS, ch, width=DT * 0.92, color=C_CH, alpha=0.45)
    ax2.bar(HOURS, -dis, width=DT * 0.92, color=C_DIS, alpha=0.45)
    ax2.set_ylabel('充(+)/放(-)功率 (kW)')
    ax2.grid(False)
    ax2.axhline(0, color='k', lw=0.5)
    ax.legend(loc='lower left', fontsize=9)
    _xticks(ax)
    ax.set_title('图4  储能储电量轨迹与充放电功率')
    _save(fig, 'fig4_soc')


def fig5(price, buy_kwh):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(HOURS, buy_kwh, width=DT * 0.92, color=C_BUY, alpha=0.8, label='购电量 (kWh/10min)')
    ax.set_ylabel('购电量 (kWh)')
    ax2 = ax.twinx()
    ax2.step(HOURS, price, where='post', color=C_PRICE, lw=1.6, label='电价')
    ax2.set_ylabel('电价 (元/kWh)', color=C_PRICE)
    ax2.tick_params(axis='y', colors=C_PRICE)
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=9)
    _xticks(ax)
    ax.set_title('图5  各 10 分钟时段购电量与电价')
    _save(fig, 'fig5_buy_bars')


def fig6(summary):
    base = summary['baseline_no_storage']
    labels = ['全天购电量\n(kWh)', '全天购电费\n(元)', '弃光电量\n(kWh)']
    a = [base['total_buy_kWh'], base['total_cost'], base['total_curtail_kWh']]
    b = [summary['total_buy_kWh'], summary['total_cost_yuan'], summary['total_curtail_kWh']]
    x = np.arange(3)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    for k, ax in enumerate(axes):
        bars = ax.bar([0, 1], [a[k], b[k]], width=0.55, color=['#95a5a6', C_BUY])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['无储能', '有储能(最优)'])
        ax.set_title(labels[k], fontsize=11)
        for r, v in zip(bars, [a[k], b[k]]):
            ax.annotate('%.0f' % v, (r.get_x() + r.get_width() / 2, v),
                        textcoords='offset points', xytext=(0, 4), ha='center', fontsize=9)
        ax.margins(y=0.18)
    fig.suptitle('图6  有/无储能方案对比', y=1.02)
    _save(fig, 'fig6_compare')


def fig7(load, pv, buy, ch, dis):
    """全天能量来源构成（按 10 分钟区间堆叠，单位 kW）。"""
    from_grid = buy
    from_pv = np.minimum(pv, load + ch)             # 光伏实际直供/充电部分
    from_dis = dis
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.stackplot(HOURS, from_grid, from_pv, from_dis,
                 colors=[C_BUY, C_PV, C_DIS], alpha=0.75,
                 labels=['外网购电', '光伏发电', '储能放电'])
    ax.plot(HOURS, load + ch, color='k', lw=1.4, label='总用电需求 $L_t+g_t$')
    ax.set_ylabel('功率 (kW)')
    _xticks(ax)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title('图7  全天能量来源构成')
    _save(fig, 'fig7_energy_mix')


def fig8():
    """储能功率/容量对购电费与弃光的敏感性。"""
    with open(os.path.join(RES_DIR, 'q1_sensitivity.json'), encoding='utf-8') as f:
        rows = json.load(f)
    pm = [r for r in rows if r['kind'] == 'p_max' and r['p_max'] > 0]
    em = [r for r in rows if r['kind'] == 'e_max']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))

    ax = axes[0]
    x = [r['p_max'] for r in pm]
    ax.plot(x, [r['cost'] for r in pm], 'o-', color=C_BUY, label='购电费 (元)')
    ax.set_xlabel('最大充放电功率 $P_{max}$ (kW)')
    ax.set_ylabel('购电费 (元)', color=C_BUY)
    ax.tick_params(axis='y', colors=C_BUY)
    ax2 = ax.twinx()
    ax2.plot(x, [r['curtail_kWh'] for r in pm], 's--', color=C_PV, label='弃光 (kWh)')
    ax2.set_ylabel('弃光电量 (kWh)', color=C_PV)
    ax2.tick_params(axis='y', colors=C_PV)
    ax2.grid(False)
    ax.set_title('储能功率的影响')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8)

    ax = axes[1]
    x = [r['e_max'] for r in em]
    ax.plot(x, [r['cost'] for r in em], 'o-', color=C_BUY, label='购电费 (元)')
    ax.set_xlabel('储电量上界 $E_{max}$ (kWh)')
    ax.set_ylabel('购电费 (元)', color=C_BUY)
    ax.tick_params(axis='y', colors=C_BUY)
    ax2 = ax.twinx()
    ax2.plot(x, [r['curtail_kWh'] for r in em], 's--', color=C_PV, label='弃光 (kWh)')
    ax2.set_ylabel('弃光电量 (kWh)', color=C_PV)
    ax2.tick_params(axis='y', colors=C_PV)
    ax2.grid(False)
    ax.set_title('储能容量的影响')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8)

    fig.suptitle('图8  储能配置对购电费用与光伏消纳的敏感性', y=1.03)
    _save(fig, 'fig8_sensitivity')


def main():
    df = load_attachment1()
    price, load, pv = df['price'].to_numpy(), df['load'].to_numpy(), df['pv'].to_numpy()
    r = solve(price, load, pv)
    assert r['success']
    with open(os.path.join(RES_DIR, 'q1_summary.json'), encoding='utf-8') as f:
        summary = json.load(f)

    print('中文字体:', _PICK or '未找到，将使用默认字体')
    fig1(price, load - pv)
    fig2(load, pv)
    fig3(price, r['buy_kW'], r['ch_kW'], r['dis_kW'])
    fig4(r['ch_kW'], r['dis_kW'], r['E_kWh'])
    fig5(price, r['buy_kWh'])
    fig6(summary)
    fig7(load, pv, r['buy_kW'], r['ch_kW'], r['dis_kW'])
    fig8()


if __name__ == '__main__':
    main()
