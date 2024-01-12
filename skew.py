# -*- coding: utf-8 -*-
"""
@Create : 2023/11/7 9:20
@Author : fu Yurong
@Package: skew.py
@Project : option_facts
@software: PyCharm
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from facts_tools import prepare_data, get_plots
import matplotlib.pyplot as plt


class SKEW:

    def __init__(self, df):
        self.df = df

    @staticmethod
    def pivot_table(data):
        """
            data: 近月 / 次近月， dataframe
            功能：
                以执行价为索引， 计算call和put的差值
        """
        data = data.pivot_table(index='strike_price', columns='type', values='option_price')

        data['c_p'] = data['C'] - data['P']
        data['abs_diff'] = np.abs(data['c_p'])
        return data

    @staticmethod
    def get_params(data, dt, t_days, rf, fixed=False, vix_type=None):
        """
            功能：
                计算所需参数
                step：
                    找到平值期权(以min(abs(call-put))作为近似)的执行价格作为K_0， 计算F_0
                    修正k_0 = max(strike_price < F_0)
                    基于修正k_0筛选合约
                    得到的筛选合约计算delta_k_i

            data: index=['strike_price]， columns=[p, c, c-p, abs(c-p)], pivot_table
            Tdays: 剩余到期时间
            dt: 当天时间(用于确定当天利率， 固定使用0.0003，影响也不大)

            return:
                k_0_init: 当天平值期权的执行价格
                r: 当天利率
                F: 基于k_0_init计算的无套利远期价格
                k_0: 小于F的最大的执行价格
                Q: dataframe, columns=['selected_opts', 'k_i', 'delta_k_i']
        """

        # min(call - put)作为平值期权近似
        k_0_init = data.sort_values(by=['abs_diff']).index[0]
        if fixed:
            r_ = rf
        else:
            # shibor
            pass
        # 无套利远期价格
        f_0 = k_0_init + np.exp(t_days / 365 * r_) * data.loc[k_0_init, 'c_p']

        # 小于F_0的最大的执行价作为修正k_0
        k_0 = np.max(data.index[data.index < f_0])

        # 筛选合约
        # 执行价大于(小于)k_0的数据, 连续两个零买价停止
        q = data[['C', 'P']].copy()

        if vix_type == 'call':
            q.loc[q.index >= k_0, 'selected_opts'] = q.loc[q.index >= k_0, 'C']

        elif vix_type == 'put':
            q.loc[q.index < k_0, 'selected_opts'] = q.loc[q.index < k_0, 'P']

        else:
            q.loc[q.index >= k_0, 'selected_opts'] = q.loc[q.index >= k_0, 'C']
            q.loc[q.index < k_0, 'selected_opts'] = q.loc[q.index < k_0, 'P']

        q = q['selected_opts'].reset_index()  # columns=['strike_price', 'selected_opts']

        # 计算delta_k_i
        q['delta_k'] = q['strike_price'].rolling(3, center=True) \
            .apply(lambda x: (x.iloc[-1] - x.iloc[0]) / 2)
        q.loc[q.index[0], 'delta_k'] = q['strike_price'].iloc[1] - q['strike_price'].iloc[0]
        q.loc[q.index[-1], 'delta_k'] = q['strike_price'].iloc[-1] - q['strike_price'].iloc[-2]

        return k_0_init, r_, f_0, k_0, q

    @staticmethod
    def cal_skew(t, r, f_0, k_0, q):
        """
            计算偏度
            T: 有效期限
            r: 无风险利率
            F: 远期价格水平
            K_0: 平值期权执行价格

        """
        e1 = -(1 + np.log(f_0 / k_0) - f_0 / k_0)
        e2 = 2 * np.log(k_0 / f_0) * (f_0 / k_0 - 1) + 0.5 * np.square(np.log(k_0 / f_0))
        e3 = 3 * np.square(np.log(k_0 / f_0)) * (np.log(k_0 / f_0) / 3 - 1 + f_0 / k_0)

        p1 = np.exp(r * t) * (-np.sum(q['selected_opts'] * q['delta_k'] / np.square(q['strike_price']))) + e1
        p2_part_1 = (1 - np.log(q['strike_price'] / f_0))
        p2 = np.exp(r * t) * (np.sum(2 * p2_part_1 * q['selected_opts'] * q['delta_k'] / np.square(q['strike_price']))) + e2
        p3_part_1 = 2 * np.log(q['strike_price'] / f_0) - np.square(np.log(q['strike_price'] / f_0))
        p3 = np.exp(r * t) * (np.sum(3 * p3_part_1 * q['selected_opts'] * q['delta_k'] / np.square(q['strike_price']))) + e3

        s = (p3 - 3 * p1 * p2 + 2 * p1**3) / (p2 - p1**2)**(3/2)

        return s

    @staticmethod
    def cal_skew_index(s_1, s_2, t_1, t_2):
        """
            s_1: 近月偏度
            s_2: 次近月偏度
            T_1: 近月剩余到期
            T_2: 次近月到期时间
        """

        w = (t_2 - 30 / 365) / (t_2 - t_1)
        skew = 100 - 10 * (w * s_1 + (1 - w) * s_2)
        return skew

    def get_df_(self, product_type=None):
        """
            df: 期权相关数据
            product_type: 标的类型, IF, IH, IM
        """

        # 筛选合约
        df_ = self.df.copy()
        df_ = df_[df_['product'] == '{}'.format(product_type)]

        skew = {}
        for dt, data in tqdm(df_.groupby('datetime')):
            # data: 同一天，同一标的，不同到期日，不同档位期权合约
            # 筛选近月 / 次近月合约(不参与末日轮, 剩余到期日>7)
            all_t = data[data['time_diff'] > 7]['time_diff'].unique()
            all_t.sort()

            if len(all_t) >= 2:
                # 数据够区分近月，次近月
                t_fm, t_sfm = all_t[: 2] / 365
                t_days_fm, t_days_sfm = all_t[: 2]
            else:
                # 默认近月 == 次近月
                t_days_fm = all_t[0]
                t_days_sfm = all_t[0]
                t_fm = all_t[0] / 365
                t_sfm = all_t[0] / 365

            data_fm, data_sfm = data[data['time_diff'] == t_days_fm], data[data['time_diff'] == t_days_sfm]

            # 转化为透视表，以便后续计算
            # index=['strike_price'], columns=['c', 'p', 'c_p', 'abs_diff]
            data_fm_ = self.pivot_table(data_fm)
            data_sfm_ = self.pivot_table(data_sfm)

            # 计算所需参数
            k_0_init_fm, r_fm, f_0_fm, k_0_fm, q_fm = self.get_params(data=data_fm_,
                                                                      dt=None,
                                                                      t_days=t_days_fm,
                                                                      rf=0.0003,
                                                                      fixed=True,
                                                                      vix_type='all')
            k_0_init_sfm, r_sfm, f_0_sfm, k_0_sfm, q_sfm = self.get_params(data=data_sfm_,
                                                                           dt=None,
                                                                           t_days=t_days_fm,
                                                                           rf=0.0003,
                                                                           fixed=True,
                                                                           vix_type='all')

            # 计算近月, 次近月偏度
            s_1 = self.cal_skew(t=t_fm,                  # 剩余有效到期期限
                                r=r_fm,                  # 无风险收益率
                                f_0=f_0_fm,              # 无套利远期价格
                                k_0=k_0_fm,              # 小于F_0的最大的执行价
                                q=q_fm)                  # columns=['strike_price', 'option_price', 'delta_k_i']

            s_2 = self.cal_skew(t=t_sfm,                 # 剩余有效到期期限
                                r=r_sfm,                 # 无风险收益率
                                f_0=f_0_sfm,             # 无套利远期价格
                                k_0=k_0_sfm,             # 小于F_0的最大的执行价
                                q=q_sfm)                 # columns=['strike_price', 'selected_opts', 'delta_k_i']

            skew[dt] = self.cal_skew_index(s_1=s_1,
                                           s_2=s_2,
                                           t_1=t_fm,
                                           t_2=t_sfm,
                                           )

        df_skew = pd.Series(skew, name='skew').dropna()

        return df_skew

