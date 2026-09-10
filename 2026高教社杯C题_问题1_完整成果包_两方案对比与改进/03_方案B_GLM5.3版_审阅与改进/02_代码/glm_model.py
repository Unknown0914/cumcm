# -*- coding: utf-8 -*-
"""
glm_model.py —— 问题1 模型（GLM-5.3 方案路线：PuLP + CBC）

本文件在 GLM 原方案基础上做了三处**建模层面的修正**，其余保持其变量体系
（p 购电、c 充电、f 放电、d 弃电、E 储电量）：

  修正 1（严谨性）：补上弃电上界  0 <= d_t <= v_t。
      GLM 原文仅写 d_t >= 0。虽然本题最优解 d* ≡ 0（故不影响数值结果，
      见 results/audit_glm.txt 的 D 节），但缺上界时模型允许"无中生有的弃电"
      （v_t = 0 的时段也可能出现 d_t > 0），对任意数据不稳健。

  修正 2（口径可切换）：把充、放电效率参数化 eta_c / eta_d，
      以支持 GLM 建议的"单侧效率口径"敏感性分析：
        - 'double'      : eta_c = eta_d = 0.9   （GLM 主口径，往返 81%）
        - 'single'      : eta_c = 0.9, eta_d = 1.0 （GLM 建议的对照口径）
        - 'roundtrip90' : eta_c = eta_d = sqrt(0.9) ≈ 0.9487（往返 90%）

  修正 3（弃电可关闭）：allow_curtail=False 时把 d 固定为 0，
      用于验证 GLM 自己提出的"若弃电总量为 0，可将 d 固定为 0 重解"的结论。

同时提供 scipy/HiGHS 的等价实现 solve_highs()，用作交叉验证（CBC 默认解的
平衡残差约 1e-5 kW，HiGHS 可达 1e-12 kWh，两者的最优值一致到 1e-6 元）。
"""
import numpy as np
from scipy.optimize import linprog

from common import DT, N, E_INIT, E_MIN, E_MAX, P_MAX

# 效率口径预设
ETA_PRESETS = {
    'double':      (0.9, 0.9),                 # 双侧 90%，往返 81%（GLM 主口径）
    'single':      (0.9, 1.0),                 # 仅充电侧 90%（GLM 建议的对照口径）
    'roundtrip90': (0.9 ** 0.5, 0.9 ** 0.5),   # 往返 90%
}


def _res_pack(p, c, f, d, E, price, load, pv, **extra):
    out = {
        'buy_kW': p, 'ch_kW': c, 'dis_kW': f, 'curtail_kW': d, 'E_kWh': E,
        'buy_kWh': p * DT, 'ch_kWh': c * DT, 'dis_kWh': f * DT, 'curtail_kWh': d * DT,
        'total_buy_kWh': float(p.sum() * DT),
        'total_cost': float((price * p * DT).sum()),
        'total_charge_kWh': float(c.sum() * DT),
        'total_discharge_kWh': float(f.sum() * DT),
        'total_curtail_kWh': float(d.sum() * DT),
        'pv_total_kWh': float(pv.sum() * DT),
        'pv_used_kWh': float((pv - d).sum() * DT),
        'load_total_kWh': float(load.sum() * DT),
        'E0': float(E_INIT), 'E_T': float(E[-1]),
        'E_min': float(E.min()), 'E_max': float(E.max()),
    }
    out.update(extra)
    return out


# ---------------------------------------------------------------------------
# 路线 A：PuLP + CBC（GLM 原方案）
# ---------------------------------------------------------------------------
def solve_pulp(price, load, pv, eta_c=0.9, eta_d=0.9, allow_curtail=True,
               curtail_upper=True, tight=False,
               p_max=None, e_min=None, e_max=None):
    """用 PuLP + CBC 求解问题 1。

    curtail_upper : True 时施加 0 <= d_t <= v_t（修正 1）；False 则复现 GLM 原式。
    tight         : 是否把 CBC 原始/对偶容差收紧到 1e-9。
    p_max/e_min/e_max : 可覆盖附录1 的储能参数，用于敏感性分析。
    """
    import pulp

    pmax = P_MAX if p_max is None else float(p_max)
    emin = E_MIN if e_min is None else float(e_min)
    emax_ = E_MAX if e_max is None else float(e_max)
    T = len(price)
    prob = pulp.LpProblem('C2026_Q1_glm', pulp.LpMinimize)
    ub_d = (lambda t: float(pv[t]) if (allow_curtail and curtail_upper) else
            (1e9 if allow_curtail else 0.0))
    p = [pulp.LpVariable('p%d' % t, lowBound=0) for t in range(T)]
    c = [pulp.LpVariable('c%d' % t, lowBound=0, upBound=pmax) for t in range(T)]
    f = [pulp.LpVariable('f%d' % t, lowBound=0, upBound=pmax) for t in range(T)]
    d = [pulp.LpVariable('d%d' % t, lowBound=0, upBound=ub_d(t)) for t in range(T)]
    E = [pulp.LpVariable('E%d' % t, lowBound=emin, upBound=emax_) for t in range(T)]

    prob += pulp.lpSum(price[t] * p[t] * DT for t in range(T))
    for t in range(T):
        prob += p[t] + pv[t] + f[t] == load[t] + c[t] + d[t]
        E_in = E_INIT if t == 0 else E[t - 1]
        prob += E[t] == E_in + eta_c * c[t] * DT - f[t] * DT / eta_d
    prob += E[T - 1] == E_INIT

    opts = ['primalTolerance', '1e-9', 'dualTolerance', '1e-9'] if tight else []
    solver = pulp.PULP_CBC_CMD(msg=False, options=opts)
    prob.solve(solver)

    assert pulp.LpStatus[prob.status] == 'Optimal', pulp.LpStatus[prob.status]
    val = lambda vs: np.array([v.value() for v in vs], dtype=float)
    pv_, cv, fv, dv, Ev = val(p), val(c), val(f), val(d), val(E)
    return _res_pack(pv_, cv, fv, dv, Ev, price, load, pv,
                     solver='PuLP-CBC', status=pulp.LpStatus[prob.status],
                     objective=float(pulp.value(prob.objective)), T=T,
                     eta_c=eta_c, eta_d=eta_d, p_max=pmax, e_max=emax_)


# ---------------------------------------------------------------------------
# 路线 B：scipy + HiGHS（交叉验证）
# ---------------------------------------------------------------------------
IDX_P, IDX_C, IDX_F, IDX_D, IDX_E = 0, N, 2 * N, 3 * N, 4 * N
NVAR = 5 * N


def build_lp_highs(price, load, pv, eta_c=0.9, eta_d=0.9, allow_curtail=True,
                   p_max=None, e_min=None, e_max=None, curtail_upper=True):
    pmax = P_MAX if p_max is None else float(p_max)
    emin = E_MIN if e_min is None else float(e_min)
    emax_ = E_MAX if e_max is None else float(e_max)
    T = len(price)
    cobj = np.zeros(NVAR)
    cobj[IDX_P:IDX_P + T] = price * DT
    A_eq, b_eq = [], []
    for t in range(T):        # p - c + f - d = L - v
        row = np.zeros(NVAR)
        row[IDX_P + t] = 1.0; row[IDX_C + t] = -1.0
        row[IDX_F + t] = 1.0; row[IDX_D + t] = -1.0
        A_eq.append(row); b_eq.append(load[t] - pv[t])
    for t in range(T):        # E_t - E_{t-1} - eta_c c dt + f dt/eta_d = [E0]
        row = np.zeros(NVAR)
        row[IDX_E + t] = 1.0
        if t > 0:
            row[IDX_E + t - 1] = -1.0
        row[IDX_C + t] = -eta_c * DT
        row[IDX_F + t] = DT / eta_d
        A_eq.append(row); b_eq.append(E_INIT if t == 0 else 0.0)
    row = np.zeros(NVAR); row[IDX_E + T - 1] = 1.0
    A_eq.append(row); b_eq.append(E_INIT)

    bounds = ([(0.0, None)] * T + [(0.0, pmax)] * T + [(0.0, pmax)] * T
              + [(0.0, (float(pv[t]) if curtail_upper else 1e9) if allow_curtail else 0.0)
                 for t in range(T)]
              + [(emin, emax_)] * T)
    return cobj, np.array(A_eq), np.array(b_eq), bounds


def solve_highs(price, load, pv, eta_c=0.9, eta_d=0.9, allow_curtail=True,
                p_max=None, e_min=None, e_max=None, curtail_upper=True):
    cobj, A_eq, b_eq, bounds = build_lp_highs(price, load, pv, eta_c, eta_d, allow_curtail,
                                              p_max, e_min, e_max, curtail_upper)
    r = linprog(cobj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    assert r.success, r.message
    x = r.x
    T = len(price)
    return _res_pack(x[IDX_P:IDX_P + T], x[IDX_C:IDX_C + T], x[IDX_F:IDX_F + T],
                     x[IDX_D:IDX_D + T], x[IDX_E:IDX_E + T], price, load, pv,
                     solver='scipy-HiGHS', status=r.message, objective=float(r.fun), T=T,
                     eta_c=eta_c, eta_d=eta_d,
                     p_max=5000.0 if p_max is None else float(p_max),
                     e_max=10800.0 if e_max is None else float(e_max))


if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    from common import load_attachment1
    df = load_attachment1()
    price, load, pv = df['price'].to_numpy(), df['load'].to_numpy(), df['pv'].to_numpy()
    for tag, fn in (('PuLP-CBC', solve_pulp), ('scipy-HiGHS', solve_highs)):
        r = fn(price, load, pv)
        print('%-12s 购电量 %.4f kWh  购电费 %.6f 元  弃电 %.4f kWh  E24=%.4f'
              % (tag, r['total_buy_kWh'], r['total_cost'], r['total_curtail_kWh'], r['E_T']))
