# -*- coding: utf-8 -*-
"""
q1_model.py —— 问题1 的线性规划模型与求解

数学模型（单日、144 个 10 分钟时段，Δt = 1/6 h）

符号
    t = 1..T (T=144)            时段编号
    c_t   (元/kWh)              第 t 时段电价
    L_t   (kW)                  第 t 时段小区负载
    PV_t  (kW)                  第 t 时段光伏发电预测功率
    b_t >= 0 (kW)               计划购电功率         —— 决策
    g_t, d_t >= 0 (kW)          储能充、放电功率      —— 决策
    u_t >= 0 (kW)               弃光功率 (curtail)   —— 决策
    E_t   (kWh)                 第 t 时段末储电量      —— 决策
    E0 = 6000, E∈[1200,10800], 充电/放电功率 <= 5000
    eta_c = eta_d = 0.9

目标
    min  F = Σ_t c_t · b_t · Δt                        (全天购电费用)

约束
    (1) 功率平衡    b_t + (PV_t - u_t) + d_t = L_t + g_t,  ∀t
    (2) 储能动态    E_t = E_{t-1} + eta_c · g_t · Δt - d_t · Δt / eta_d,  ∀t
    (3) 边界条件    E_0 = 6000,  E_T = 6000            (0:00 与 24:00 储电量相同)
    (4) 储电量界    1200 <= E_t <= 10800
    (5) 功率界      0 <= g_t, d_t <= 5000;  0 <= u_t <= PV_t;  b_t >= 0

说明：
  * 引入弃光变量 u_t 是必要的。中午净负荷最低约 -2118 kW，储能受容量上界
    10800 kWh 限制无法全部吸收富余光伏，若不允许弃光（且题目未允许售电），
    模型将不可行。
  * 目标函数只含 b_t，因而 u_t 在最优解中只出现在"零成本"位置；求解器会
    自动在"多买电"和"多弃光"之间选择更省钱的方案（买电必花钱，故优先弃光）。
  * 效率 eta_c·eta_d = 0.81 < 1 使得同一时段同时充放电必然劣于只做其一，
    因此松弛后的 LP 其最优解自动满足充放电互斥，无需引入 0-1 变量。
"""
import numpy as np
from scipy.optimize import linprog

from common import (DT, N, E_INIT, E_MIN, E_MAX, P_MAX, ETA_C, ETA_D)

# 决策变量分块下标（每个块长度 N）
IDX_B, IDX_G, IDX_D, IDX_U, IDX_E = 0, N, 2 * N, 3 * N, 4 * N
NVAR = 5 * N


def _block(offset):
    """给定变量块的起始偏移量，返回该块的切片。"""
    return slice(offset, offset + N)


def build_lp(price, load, pv, allow_curtail=True, allow_sell=False):
    """构造 LP 的 c / A_eq / b_eq / bounds。返回 (c, A_eq, b_eq, bounds)。

    allow_curtail : 是否引入弃光变量（False 时把弃光上界固定为 0）
    allow_sell    : 是否允许余电上网（购电功率下界取 -inf）
    """
    T = len(price)
    assert T == N, T

    # ---- 目标：min Σ c_t b_t Δt ----
    c = np.zeros(NVAR)
    c[_block(IDX_B)] = price * DT
    if allow_sell:
        # 售电按同一电价结算（仅用于对照方案 C）
        pass

    A_eq, b_eq = [], []

    # ---- (1) 功率平衡  b_t - g_t + d_t - u_t = L_t - PV_t ----
    for t in range(T):
        row = np.zeros(NVAR)
        row[IDX_B + t] = 1.0
        row[IDX_G + t] = -1.0
        row[IDX_D + t] = 1.0
        row[IDX_U + t] = -1.0
        A_eq.append(row)
        b_eq.append(load[t] - pv[t])

    # ---- (2) 储能动态 + (3) 初始储电量 ----
    for t in range(T):
        row = np.zeros(NVAR)
        row[IDX_E + t] = 1.0                     # E_t
        if t > 0:
            row[IDX_E + t - 1] = -1.0            # -E_{t-1}
        row[IDX_G + t] = -ETA_C * DT             # -eta_c·g_t·Δt
        row[IDX_D + t] = DT / ETA_D              # +d_t·Δt/eta_d
        A_eq.append(row)
        b_eq.append(E_INIT if t == 0 else 0.0)

    # ---- (3') 末端储电量 E_T = E_INIT ----
    row = np.zeros(NVAR)
    row[IDX_E + T - 1] = 1.0
    A_eq.append(row)
    b_eq.append(E_INIT)

    # ---- bounds ----
    b_lo = -np.inf if allow_sell else 0.0
    bounds = (
        [(b_lo, None)] * T                    # b_t 购电功率
        + [(0.0, P_MAX)] * T                  # g_t 充电功率
        + [(0.0, P_MAX)] * T                  # d_t 放电功率
        + [(0.0, float(pv[t]) if allow_curtail else 0.0) for t in range(T)]  # u_t
        + [(E_MIN, E_MAX)] * T                # E_t
    )
    return c, np.array(A_eq), np.array(b_eq), bounds


def solve(price, load, pv, allow_curtail=True, allow_sell=False, return_raw=False):
    """求解问题1 的 LP，返回结构化结果字典。"""
    c, A_eq, b_eq, bounds = build_lp(price, load, pv, allow_curtail, allow_sell)
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    out = {
        'success': bool(res.success),
        'message': res.message,
        'objective': float(res.fun) if res.success else float('nan'),
        'allow_curtail': allow_curtail,
        'allow_sell': allow_sell,
    }
    if not res.success:
        return out
    x = res.x
    b = x[_block(IDX_B)]          # 购电功率 kW
    g = x[_block(IDX_G)]          # 充电功率 kW
    d = x[_block(IDX_D)]          # 放电功率 kW
    u = x[_block(IDX_U)]          # 弃光功率 kW
    E = x[_block(IDX_E)]          # 各时段末储电量 kWh

    out.update({
        'buy_kW': b, 'ch_kW': g, 'dis_kW': d, 'curtail_kW': u,
        'E_kWh': E, 'E0': E_INIT,
        'buy_kWh': b * DT, 'ch_kWh': g * DT, 'dis_kWh': d * DT, 'curtail_kWh': u * DT,
        'total_buy_kWh': float(np.sum(b) * DT),
        'total_cost': float(np.sum(price * b) * DT),
        'total_charge_kWh': float(np.sum(g) * DT),
        'total_discharge_kWh': float(np.sum(d) * DT),
        'total_curtail_kWh': float(np.sum(u) * DT),
        'pv_used_kWh': float(np.sum(pv - u) * DT),
        'pv_total_kWh': float(np.sum(pv) * DT),
        'load_total_kWh': float(np.sum(load) * DT),
    })
    if return_raw:
        out['raw'] = res
    return out


def solve_no_storage(price, load, pv):
    """基准方案 A：不装储能（储能功率为 0），余电全部弃掉。"""
    return solve(price, load, pv, allow_curtail=True, allow_sell=False)


if __name__ == '__main__':
    from common import load_attachment1
    df = load_attachment1()
    r = solve(df['price'].to_numpy(), df['load'].to_numpy(), df['pv'].to_numpy())
    print('success', r['success'], r['message'])
    print('全天购电量 = %.4f kWh' % r['total_buy_kWh'])
    print('全天购电费 = %.4f 元' % r['total_cost'])
    print('总充电量 = %.4f kWh, 总放电量 = %.4f kWh' % (r['total_charge_kWh'], r['total_discharge_kWh']))
    print('弃光量 = %.4f kWh' % r['total_curtail_kWh'])
