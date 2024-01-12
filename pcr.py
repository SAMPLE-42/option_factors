# -*- coding: utf-8 -*-
"""
@Create : 2023/11/7 9:20
@Author : fu Yurong
@Package: pcr.py
@Project : option_facts
@software: PyCharm
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

from facts_tools import sorted_data


class PCR:

    def __init__(self, df):
        self.df = df

    def get_pcr_res(self, product_type=None, pcr_type=None, data_type=None):
        """
            df: 期权相关数据
            product_type: 标的类型, IF, IH, IM
            data_type: 合约类型， fm, sfm, fm+sfm, all
            pcr_type: volume / oi
        """

        # 筛选合约
        df_ = self.df.copy()
        df_ = df_[df_['product'] == '{}'.format(product_type)]

        pcr = {}
        for dt, data in tqdm(df_.groupby('datetime')):

            data['oi'] = (data['open_oi'] + data['close_oi']) / 2
            # data: 同一天，同一标的，不同到期日，不同档位期权合约
            # 筛选近月 / 次近月合约(不参与末日轮, 剩余到期日>7)
            all_t = data[data['time_diff'] > 7]['time_diff'].unique()
            all_t.sort()

            if len(all_t) >= 2:
                # 数据够区分近月，次近月

                t_days_fm, t_days_sfm = all_t[: 2]
            else:
                # 默认近月 == 次近月
                t_days_fm = all_t[0]
                t_days_sfm = all_t[0]

            data_fm, data_sfm = data[data['time_diff'] == t_days_fm], data[data['time_diff'] == t_days_sfm]
            data_fm_sfm = data[(data['time_diff'] == t_days_sfm) | (data['time_diff'] == t_days_fm)]

            # 转化为透视表，以便后续计算
            # index=['strike_price'], columns=['c', 'p', 'c_p', 'abs_diff]

            data_fm_ = data_fm.pivot_table(index='strike_price',
                                           columns='type',
                                           values='{}'.format(pcr_type))

            data_sfm_ = data_sfm.pivot_table(index='strike_price',
                                             columns='type',
                                             values='{}'.format(pcr_type))

            data_fm_sfm_ = data_fm_sfm.pivot_table(index='strike_price',
                                                   columns='type',
                                                   values='{}'.format(pcr_type))

            data_all = data.pivot_table(index='strike_price',
                                        columns='type',
                                        values='{}'.format(pcr_type))

            if data_type == 'fm':
                # 只使用近月合约
                call_volume = data_fm_['C'].sum()
                put_volume = data_fm_['P'].sum()
                pcr[dt] = put_volume / call_volume if call_volume != 0 else np.nan

            elif data_type == 'sfm':
                # 只使用次近月合约
                call_volume = data_sfm_['C'].sum()
                put_volume = data_sfm_['P'].sum()
                pcr[dt] = put_volume / call_volume if call_volume != 0 else np.nan

            elif data_type == 'fm+sfm':
                # 同时使用近月，次近月合约
                call_volume = data_fm_sfm_['C'].sum()
                put_volume = data_fm_sfm_['P'].sum()
                pcr[dt] = put_volume / call_volume if call_volume != 0 else np.nan

            else:
                # 不对合约做任何筛选
                call_volume = data_all['C'].sum()
                put_volume = data_all['P'].sum()
                pcr[dt] = put_volume / call_volume if call_volume != 0 else np.nan

        df_pcr = pd.Series(pcr, name='pcr_{}'.format(data_type)).dropna()

        return df_pcr

    def get_df_(self, product_type=None, pcr_type=None):
        """
            df: 期权数据
            product_type: 标的类型
            pcr_type: volume / oi
        """
        df_pcr_fm = self.get_pcr_res(product_type=product_type, data_type='fm', pcr_type=pcr_type)
        df_pcr_sfm = self.get_pcr_res(product_type=product_type, data_type='sfm', pcr_type=pcr_type)
        df_pcr_fm_sfm = self.get_pcr_res(product_type=product_type, data_type='fm+sfm', pcr_type=pcr_type)
        df_pcr_all = self.get_pcr_res(product_type=product_type, data_type='all', pcr_type=pcr_type)
        df_pcr_ = pd.concat([df_pcr_fm, df_pcr_sfm, df_pcr_fm_sfm, df_pcr_all], axis=1)

        return df_pcr_

