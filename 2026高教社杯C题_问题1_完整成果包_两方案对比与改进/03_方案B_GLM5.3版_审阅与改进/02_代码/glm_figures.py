# -*- coding: utf-8 -*-
"""
glm_figures.py —— 问题1（GLM 方案路线）论文用图表（300 dpi PNG + 矢量 PDF）

fig1 全天电价与净负荷曲线
fig2 小区负载与光伏发电功率
fig3 最优计划购电策略（购电功率 + 储能充放电）
fig4 储能储电量轨迹与充放电功率
fig5 各 10 分钟时段购电量与电价
fig6 有/无储能方案对比
fig7 效率口径敏感性（GLM 建议的对照口径）
fig8 储能功率/容量敏感性
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
from glm_model import solve_pulp, solve_highs

_CAND = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Source Han Sans SC', 'SimSun']
_avail = {f.name for f in font_manager.fontManager.ttflist}
_PICK = next((c for c in _CAND if c in _avail), None)
if _PICK:
    plt.rcParams['font.sans-serif'] = [_PICK]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

C_PRICE, C_LOAD, C_PV = '#c0392b', '#2c3e50', '#e67e22'
C_BUY, C_CH, C_DIS, C_E = '#2980b9', '#27ae60', '#8e44ad', '#16a085'
H = np.arange(1, N + 1) * DT


def _save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG_DIR, '%s.%s' % (name, ext)), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('figure ->', name)


def _xticks(ax):
    ax.set_xticks(range(0, 25, 2)); ax.set_xlim(0, 24); ax.set_xlabel('时刻 (h)')


def fig1(price, net):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.fill_between(H, 0, net, where=(net >= 0), color='#95a5a6', alpha=0.35, label='正净负荷')
    ax.fill_between(H, 0, net, where=(net < 0), color='#e74c3c', alpha=0.35, label='负净负荷（光伏富余）')
    ax.plot(H, net, color=C_LOAD, lw=1.6, label='净负荷 $L_t-v_t$')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylabel('净负荷 (kW)')
    ax2 = ax.twinx()
    ax2.step(H, price, where='post', color=C_PRICE, lw=1.8, label='电价')
    ax2.set_ylabel('电价 (元/kWh)', color=C_PRICE); ax2.tick_params(axis='y', colors=C_PRICE)
    ax2.grid(False); _xticks(ax)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=9)
    ax.set_title('图1  全天电价与净负荷曲线（附件1）')
    _save(fig, 'fig1_price_netload')


def fig2(load, pv):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(H, load, color=C_LOAD, lw=1.8, label='小区负载 $L_t$')
    ax.plot(H, pv, color=C_PV, lw=1.8, label='光伏发电预测 $v_t$')
    ax.fill_between(H, 0, pv, color=C_PV, alpha=0.18)
    ax.fill_between(H, pv, load, where=(load > pv), color='#bdc3c7', alpha=0.35, label='需电网/储能补足')
    ax.set_ylabel('功率 (kW)'); _xticks(ax)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title('图2  小区负载与光伏发电预测功率')
    _save(fig, 'fig2_load_pv')


def fig3(price, buy, ch, dis):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                  gridspec_kw={'height_ratios': [1.15, 1]})
    ax.fill_between(H, 0, buy, color=C_BUY, alpha=0.55, label='计划购电功率 $p_t$')
    ax.plot(H, buy, color=C_BUY, lw=1.4)
    ax.set_ylabel('购电功率 (kW)')
    ax3 = ax.twinx()
    ax3.step(H, price, where='post', color=C_PRICE, lw=1.4, alpha=0.85, label='电价')
    ax3.set_ylabel('电价 (元/kWh)', color=C_PRICE); ax3.tick_params(axis='y', colors=C_PRICE)
    ax3.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()
    ax.legend(h1 + h3, l1 + l3, loc='upper left', fontsize=9)
    ax.set_title('图3  最优计划购电策略（GLM 路线：PuLP+CBC）')
    ax2.bar(H, ch, width=DT * 0.92, color=C_CH, label='充电功率 $c_t$')
    ax2.bar(H, -dis, width=DT * 0.92, color=C_DIS, label='放电功率 $f_t$')
    ax2.axhline(0, color='k', lw=0.8)
    ax2.set_ylabel('储能充(+)/放(-)功率 (kW)')
    ax2.legend(loc='upper left', fontsize=9, ncol=2)
    _xticks(ax2)
    _save(fig, 'fig3_dispatch')


def fig4(ch, dis, E):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(H, E, color=C_E, lw=2.0, label='储电量 $E_t$')
    ax.axhline(E_MAX, color='#7f8c8d', ls='--', lw=1.2, label='上限 10800 kWh')
    ax.axhline(E_MIN, color='#7f8c8d', ls=':', lw=1.2, label='下限 1200 kWh')
    ax.plot([0], [E_INIT], 'o', color=C_E, ms=6)
    ax.annotate('$E_0=6000$', (0, E_INIT), textcoords='offset points', xytext=(6, -14))
    ax.plot([24], [E[-1]], 's', color=C_E, ms=6)
    ax.annotate('$E_{24}=6000$', (24, E[-1]), textcoords='offset points', xytext=(-52, 8))
    ax.set_ylabel('储电量 (kWh)'); ax.set_ylim(0, 12000)
    ax2 = ax.twinx()
    ax2.bar(H, ch, width=DT * 0.92, color=C_CH, alpha=0.45)
    ax2.bar(H, -dis, width=DT * 0.92, color=C_DIS, alpha=0.45)
    ax2.set_ylabel('充(+)/放(-)功率 (kW)'); ax2.grid(False); ax2.axhline(0, color='k', lw=0.5)
    ax.legend(loc='lower left', fontsize=9); _xticks(ax)
    ax.set_title('图4  储能储电量轨迹与充放电功率')
    _save(fig, 'fig4_soc')


def fig5(price, buy_kwh):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(H, buy_kwh, width=DT * 0.92, color=C_BUY, alpha=0.8, label='购电量 (kWh/10min)')
    ax.set_ylabel('购电量 (kWh)')
    ax2 = ax.twinx()
    ax2.step(H, price, where='post', color=C_PRICE, lw=1.6, label='电价')
    ax2.set_ylabel('电价 (元/kWh)', color=C_PRICE); ax2.tick_params(axis='y', colors=C_PRICE)
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=9); _xticks(ax)
    ax.set_title('图5  各 10 分钟时段购电量与电价')
    _save(fig, 'fig5_buy_bars')


def fig6(no_storage):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    with open(os.path.join(RES_DIR, 'glm_summary.json'), encoding='utf-8') as f:
        s = json.load(f)
    labels = ['全天购电量\n(kWh)', '全天购电费\n(元)', '弃光电量\n(kWh)']
    a = [no_storage['buy'], no_storage['cost'], no_storage['curtail']]
    b = [s['total_buy_kWh'], s['total_cost_yuan'], s['total_curtail_kWh']]
    for k, ax in enumerate(axes):
        bars = ax.bar([0, 1], [a[k], b[k]], width=0.55, color=['#95a5a6', C_BUY])
        ax.set_xticks([0, 1]); ax.set_xticklabels(['无储能', '有储能(最优)'])
        ax.set_title(labels[k], fontsize=11)
        for r, v in zip(bars, [a[k], b[k]]):
            ax.annotate('%.0f' % v, (r.get_x() + r.get_width() / 2, v),
                        textcoords='offset points', xytext=(0, 4), ha='center', fontsize=9)
        ax.margins(y=0.18)
    fig.suptitle('图6  有/无储能方案对比', y=1.02)
    _save(fig, 'fig6_compare')


def fig7():
    with open(os.path.join(RES_DIR, 'glm_sensitivity.json'), encoding='utf-8') as f:
        rows = json.load(f)
    eta = [r for r in rows if r['group'].startswith('eta:')]
    names = ['双侧 90/90\n(往返 81%)', '单侧 90/100\n(往返 90%)', '往返 90%\n(94.87/94.87)']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    bars = ax.bar(range(3), [r['cost_cbc'] for r in eta], width=0.55,
                  color=['#2980b9', '#e67e22', '#27ae60'])
    ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('全天购电费 (元)')
    for r_, v in zip(bars, [r['cost_cbc'] for r in eta]):
        ax.annotate('%.0f' % v, (r_.get_x() + r_.get_width() / 2, v),
                    textcoords='offset points', xytext=(0, 4), ha='center', fontsize=9)
    ax.margins(y=0.16); ax.set_title('效率口径对购电费的影响')
    ax = axes[1]
    x = np.arange(3)
    ax.bar(x - 0.19, [r['charge_kWh'] for r in eta], width=0.36, color=C_CH, label='充电量')
    ax.bar(x + 0.19, [r['discharge_kWh'] for r in eta], width=0.36, color=C_DIS, label='放电量')
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('电量 (kWh)'); ax.legend(fontsize=9)
    ax.set_title('效率口径对充放电量的影响')
    fig.suptitle('图7  效率口径敏感性（GLM 建议的对照口径）', y=1.02)
    _save(fig, 'fig7_eta_sensitivity')


def fig8():
    with open(os.path.join(RES_DIR, 'glm_sensitivity.json'), encoding='utf-8') as f:
        rows = json.load(f)
    pm = [r for r in rows if r['group'] == 'pmax' and r['p_max'] > 0]
    em = [r for r in rows if r['group'] == 'emax']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for ax, data, key, xlabel in (
            (axes[0], pm, 'p_max', '最大充放电功率 $P_{max}$ (kW)'),
            (axes[1], em, 'e_max', '储电量上界 $E_{max}$ (kWh)')):
        xs = [r[key] for r in data]
        ax.plot(xs, [r['cost_cbc'] for r in data], 'o-', color=C_BUY, label='购电费 (元)')
        ax.set_xlabel(xlabel); ax.set_ylabel('购电费 (元)', color=C_BUY)
        ax.tick_params(axis='y', colors=C_BUY)
        ax2 = ax.twinx()
        ax2.plot(xs, [r['curtail_kWh'] for r in data], 's--', color=C_PV, label='弃光 (kWh)')
        ax2.set_ylabel('弃光电量 (kWh)', color=C_PV); ax2.tick_params(axis='y', colors=C_PV)
        ax2.grid(False)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8)
    axes[0].set_title('储能功率的影响'); axes[1].set_title('储能容量的影响')
    fig.suptitle('图8  储能配置对购电费用与光伏消纳的敏感性', y=1.03)
    _save(fig, 'fig8_sensitivity')


def main():
    df = load_attachment1()
    price, load, pv = df['price'].to_numpy(), df['load'].to_numpy(), df['pv'].to_numpy()
    r = solve_pulp(price, load, pv)
    ns = solve_highs(price, load, pv, p_max=0.0)
    print('中文字体:', _PICK or '未找到')
    fig1(price, load - pv)
    fig2(load, pv)
    fig3(price, r['buy_kW'], r['ch_kW'], r['dis_kW'])
    fig4(r['ch_kW'], r['dis_kW'], r['E_kWh'])
    fig5(price, r['buy_kWh'])
    fig6({'buy': ns['total_buy_kWh'], 'cost': ns['total_cost'], 'curtail': ns['total_curtail_kWh']})
    fig7()
    fig8()


if __name__ == '__main__':
    main()
