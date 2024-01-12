# -*- coding: utf-8 -*-
"""
@Create : 2023/11/7 9:20
@Author : fu Yurong
@Package: vix.py
@Project : option_facts
@software: PyCharm
"""

import pandas as pd
import numpy as np
from tqdm import tqdm


class VIX:

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
    def cal_sigma(t, r, f, k_0, q):
        """
            计算sigma
        """
        sigma_part1 = (2 / t) * np.sum(np.exp(r * t) * q['selected_opts'] * q['delta_k'] / np.square(q['strike_price']))
        sigma_part2 = (1 / t) * (f / k_0 - 1)**2

        return sigma_part1 + sigma_part2

    @staticmethod
    def cal_vix(sigma_1, sigma_2, t_1, t_2):
        """
            近月: sigma_1, T1
            次近月: sigma_2, T2
        """
        vix_part_1 = (t_1 * sigma_1) * (t_2 - 30 / 365) / (t_2 - t_1)
        vix_part_2 = (t_2 * sigma_2) * (30 / 365 - t_1) / (t_2 - t_1)
        return np.sqrt((365 / 30) * (vix_part_1 + vix_part_2))

    def get_res_vix(self, df, product_type=None, vix_type=None):
        """
           df: 期权有关数据
           product_type: 标的股指类型
           option_type: 期权类型, call / put
        """

        df_ = df.copy()
        df_ = df_[df_['product'] == '{}'.format(product_type)]

        vix_res = {}

        for dt, data in tqdm(df_.groupby('datetime')):
            # data: 同一天， 同一标的，不同到期日，不同档位的期权
            # step_1: 筛选近月, 次近合约

            # 大于7, 剔除末日期权(临近到期日, 价格波动大但是流动性差)
            # 剩余存续期较小的两个作为近月, 次近月合约

            all_t = data[data['time_diff'] > 7]['time_diff'].unique()
            all_t.sort()
            # print(all_T[0])

            if len(all_t) >= 2:
                t_fm, t_sfm = all_t[: 2] / 365
                t_days_fm, t_days_sfm = all_t[: 2]
            else:
                # 存在当天数据不足的情况，无法区分近月,次近月合约
                # 默认近月, 次近月相同
                # vix = 2 * same

                t_fm = all_t[0] / 365
                t_sfm = all_t[0] / 365

                t_days_fm = all_t[0]
                t_days_sfm = all_t[0]

            data_fm, data_sfm = data[data['time_diff'] == t_days_fm], data[data['time_diff'] == t_days_sfm]

            # 透视表, 以便后续操作
            # index=['strike_price'], columns=['c', 'p', 'c_p', 'abs_diff]
            data_fm_ = self.pivot_table(data_fm.copy())
            data_sfm_ = self.pivot_table(data_sfm.copy())

            # 计算所需参数
            k_0_init_fm, r_fm, f_0_fm, k_0_fm, q_fm = self.get_params(data=data_fm_,
                                                                      dt=None,
                                                                      t_days=t_days_fm,
                                                                      rf=0.0003,
                                                                      fixed=True,
                                                                      vix_type=vix_type)
            k_0_init_sfm, r_sfm, f_0_sfm, k_0_sfm, q_sfm = self.get_params(data=data_sfm_,
                                                                           dt=None,
                                                                           t_days=t_days_fm,
                                                                           rf=0.0003,
                                                                           fixed=True,
                                                                           vix_type=vix_type)
            # 计算近月, 次近月sigma
            sigma_1 = self.cal_sigma(t=t_fm, r=r_fm, f=f_0_fm, k_0=k_0_fm, q=q_fm)
            sigma_2 = self.cal_sigma(t=t_sfm, r=r_sfm, f=f_0_sfm, k_0=k_0_sfm, q=q_sfm)

            # 计算VIX
            vix_res[dt] = self.cal_vix(sigma_1=sigma_1, sigma_2=sigma_2, t_1=t_fm, t_2=t_sfm)

        df_vix = pd.Series(vix_res, name='VIX{}'.format(vix_type)).dropna()

        return df_vix

    def get_df_(self, product_type=None):
        """
            return: dataframe, columns=['vix_call', 'vix_put', 'vix', 'vc_vp'], index=datetime
        """
        vix_call = self.get_res_vix(df=self.df,
                                    product_type='{}'.format(product_type),
                                    vix_type='call',
                                    )
        vix_put = self.get_res_vix(df=self.df,
                                   product_type='{}'.format(product_type),
                                   vix_type='put',
                                   )
        vix_all = self.get_res_vix(df=self.df,
                                   product_type='{}'.format(product_type),
                                   )
        df_vix = pd.concat([vix_call, vix_put, vix_all], axis=1)
        df_vix['vc_vp'] = df_vix['VIXcall'] - df_vix['VIXput']

        return df_vix



