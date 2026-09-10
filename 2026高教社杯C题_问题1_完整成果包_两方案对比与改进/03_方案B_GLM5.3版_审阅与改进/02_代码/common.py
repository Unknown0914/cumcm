# -*- coding: utf-8 -*-
"""
common.py —— 2026 高教社杯 C 题 问题1 公共模块

职责：
  1. 统一管理路径与物理参数（储能容量/功率/效率、时间粒度）；
  2. 读取附件1（电价、小区负载、光伏发电预测功率）；
  3. 提供"时段编号 <-> 时刻/区间标签"的换算工具。

时段约定（重要）：
  附件1 给出 144 个采样点，其"时间"列依次为 0:10, 0:20, ..., 24:00，
  第 t 个采样点的功率被视为区间 [10(t-1), 10t) 分钟内的平均功率，
  即采样时刻是该 10 分钟区间的**右端点**。于是
      t = 1   -> 0:00-0:10
      t = 144 -> 23:50-24:00(0:00+1)
  这一约定可同时保证：0:00 的储电量为 E0、24:00 的储电量为 E144，与题
  目"0:00 和 24:00 储电量相同"的约束自洽。
"""
import os
import pandas as pd

# ----------------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # cumcm2026C_wq
DATA_DIR = os.path.join(ROOT, 'data')
FIG_DIR = os.path.join(ROOT, 'figures')
OUT_DIR = os.path.join(ROOT, 'output')
RES_DIR = os.path.join(ROOT, 'results')
for _d in (DATA_DIR, FIG_DIR, OUT_DIR, RES_DIR):
    os.makedirs(_d, exist_ok=True)

SRC_A1 = r'D:\download\CUMCM2026Problems\C题\附件\附件1.xlsx'
TPL_RESULT1 = r'D:\download\CUMCM2026Problems\C题\附件\附件5\result1.xlsx'

# ----------------------------------------------------------------------------
# 物理参数（附录1）
# ----------------------------------------------------------------------------
DT = 1.0 / 6.0            # 时间步长，小时（10 分钟）
N = 144                   # 全天时段数
E_INIT = 6000.0           # 2025-01-01 0:00 初始储电量 kWh
E_MIN = 1200.0            # 储电量下界 kWh
E_MAX = 10800.0           # 储电量上界 kWh
E_CAP = 12000.0           # 储能最大容量 kWh（E_MAX 为其 90%）
P_MAX = 5000.0            # 最大充/放电功率 kW
ETA_C = 0.9               # 充电效率
ETA_D = 0.9               # 放电效率


# ----------------------------------------------------------------------------
# 时段标签工具
# ----------------------------------------------------------------------------
def fmt_clock(minutes):
    """把"距 0:00 的分钟数"格式化为标签：1440 -> '0:00+1'，1440+ -> '0:10+1'。"""
    day, rem = divmod(int(minutes), 1440)
    hh, mm = divmod(rem, 60)
    base = '%d:%02d' % (hh, mm)
    return base + '+1' if day == 1 else base


def t_to_right_label(t):
    """时段 t（1-based）的右端点时刻标签，如 t=1 -> '0:10'，t=144 -> '0:00+1'。"""
    return fmt_clock(10 * t)


def t_to_interval(t):
    """时段 t（1-based）对应的 10 分钟区间标签，如 t=61 -> '10:00-10:10'。"""
    return '%s-%s' % (fmt_clock(10 * (t - 1)), fmt_clock(10 * t))


def clock_to_t(label):
    """把区间左端点标签换算为时段号，如 '10:00' -> 61（即 [10:00,10:10) 所属时段）。"""
    hh, mm = label.split(':')
    minutes = int(hh) * 60 + int(mm)
    assert minutes % 10 == 0 and 0 <= minutes < 1440, label
    return minutes // 10 + 1


# ----------------------------------------------------------------------------
# 数据读取
# ----------------------------------------------------------------------------
def load_attachment1(path=SRC_A1):
    """读取附件1，返回含列 [t, t_end_min, price, load, pv] 的 DataFrame。

    附件1 的"时间"列前 60 行为 Excel 时间序列值(float)，第 61 行起为文本
    （如 '10:10'），因此这里不解析该列，直接用行序构造时段编号，避免类型
    混淆；行序即时间顺序（0:10, 0:20, ..., 24:00），已在数据核对中验证。
    """
    df = pd.read_excel(path)
    df = df.iloc[:, :4]
    df.columns = ['time_raw', 'price', 'load', 'pv']
    for col in ('price', 'load', 'pv'):
        df[col] = pd.to_numeric(df[col], errors='raise').astype(float)
    df = df.reset_index(drop=True)
    df.insert(0, 't', range(1, len(df) + 1))
    df['t_end_min'] = df['t'] * 10
    df['net_load'] = df['load'] - df['pv']          # 净负荷 L - PV
    df['interval'] = [t_to_interval(t) for t in df['t']]
    assert len(df) == N, '附件1 应有 %d 行，实际 %d' % (N, len(df))
    return df


def summarize_periods_bounds():
    """返回表2 需要的 6 个 4 小时区间的 (名称, 起止时段号) —— 均含端点，1-based。"""
    return [
        ('0:00-4:00', 1, 24),
        ('4:00-8:00', 25, 48),
        ('8:00-12:00', 49, 72),
        ('12:00-16:00', 73, 96),
        ('16:00-20:00', 97, 120),
        ('20:00-24:00', 121, 144),
    ]


TABLE1_CLOCKS = ['10:00', '12:00', '14:00', '16:00', '18:00', '20:00']


if __name__ == '__main__':
    d = load_attachment1()
    print(d.head(3).to_string())
    print(d.tail(3).to_string())
    print('price range', d.price.min(), d.price.max())
    print('net_load range', d.net_load.min(), d.net_load.max())
