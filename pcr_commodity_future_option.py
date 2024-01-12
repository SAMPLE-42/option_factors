# -*- coding: utf-8 -*-
"""
@Create : 2023/12/5 9:20
@Author : fu Yurong
@Package: pcr.py
@Project : option_facts
@software: PyCharm

该模块的计算和股指期货期权的区别在于主次合约的筛选
股指期货近月就是主力合约，次近月就是次主力
但是商品期货未必
所以此处识别主力和次主力
以主力合约映射表作为当天主力
以剔除主力合约的标的，以持仓量最大作为次主力
"""

import pandas as pd
import numpy as np
from tqdm import tqdm


class PCR_commodity_future:

    def __init__(self, df_option, df_future):
        self.df_option = df_option
        self.df_future = df_future

    def get_merge_opts_future(self, contracts_type=None):

        df_option = self.df_option.copy()
        df_all_future = self.df_future.copy()

        # print(df_option.columns)
        # print(df_all_future.columns)

        df_option['datetime'] = pd.to_datetime(df_option['datetime'])
        df_all_future['datetime'] = pd.to_datetime(df_all_future['datetime']).dt.date
        df_all_future['datetime'] = pd.to_datetime(df_all_future['datetime'])

        df_main_future = df_all_future[df_all_future['main_sign'] == 1]
        df_sub_future = df_all_future[df_all_future['sub_sign'] == 1]

        if contracts_type == 'main':
            df_future_need = df_main_future

        elif contracts_type == 'sub_main':
            df_future_need = df_sub_future

        elif contracts_type == 'all':
            df_future_need = df_all_future

        else:
            raise TypeError

        df_merge = pd.merge(
            df_option,
            df_future_need[['datetime', 'close', 'instrument']],
            on=['datetime', 'instrument']
        )

        df_merge.rename(columns={'close_y': 'close',
                                 'close_x': 'option_price'},
                        inplace=True)
        df_merge['type'] = df_merge['option_class'].apply(lambda x: x[0])
        df_merge['expire_datetime'] = pd.to_datetime(df_merge['expire_datetime'])
        df_merge['time_diff'] = (df_merge['expire_datetime'] - df_merge['datetime']).apply(lambda x: x.days)

        return df_merge

    def get_pcr_res(self, product_type=None, pcr_type=None, data_type=None):
        """
            df: 期权相关数据
            product_type: 标的类型, IF, IH, IM
            data_type: 合约类型， fm, sfm, fm+sfm, all
            pcr_type: volume / oi
        """

        # 筛选合约
        df_ = self.get_merge_opts_future(contracts_type=data_type)
        df_.rename(columns={'product': 'Product'}, inplace=True)
        df_ = df_[df_['Product'] == '{}'.format(product_type)]

        pcr = {}
        for dt, data in tqdm(df_.groupby('datetime')):

            data['oi'] = (data['open_oi'] + data['close_oi']) / 2
            # data: 同一天，同一标的，不同到期日，不同档位期权合约
            # 筛选近月 / 次近月合约(不参与末日轮, 剩余到期日>7)
            data = data[data['time_diff'] > 7]

            # 转化为透视表，以便后续计算
            # index=['strike_price'], columns=['c', 'p', 'c_p', 'abs_diff]

            data_all = data.pivot_table(index='strike_price',
                                        columns='type',
                                        values='{}'.format(pcr_type))

            # 检查 'C' 列和 'P' 列是否存在
            if 'C' in data_all.columns and 'P' in data_all.columns:
                call_volume = data_all['C'].sum()
                put_volume = data_all['P'].sum()
            else:
                # 处理 'C' 或 'P' 列不存在的情况
                if 'C' in data_all.columns:
                    call_volume = data_all['C'].sum()
                    put_volume = 0  # 设置 put_volume 为零或者其他适当的值
                elif 'P' in data_all.columns:
                    put_volume = data_all['P'].sum()
                    call_volume = 0  # 设置 call_volume 为零或者其他适当的值
                else:
                    # 如果 'C' 和 'P' 列都不存在，处理其他情况
                    call_volume = 0
                    put_volume = 0

            # print(f"{dt}: {call_volume}_{put_volume}_{put_volume / call_volume}")

            pcr[dt] = put_volume / call_volume if call_volume != 0 else np.nan

        df_pcr = pd.Series(pcr, name='pcr_{}'.format(data_type)).dropna()
        # print(df_pcr)

        return df_pcr

    def get_df_(self, product_type=None, pcr_type=None):
        """
            df: 期权数据
            product_type: 标的类型
            pcr_type: volume / oi
        """
        df_pcr_main = self.get_pcr_res(product_type=product_type, pcr_type=pcr_type, data_type='main')
        df_pcr_sub_main = self.get_pcr_res(product_type=product_type, pcr_type=pcr_type, data_type='sub_main')
        df_pcr_all = self.get_pcr_res(product_type=product_type, pcr_type=pcr_type, data_type='all')

        df_pcr_ = pd.concat([df_pcr_main, df_pcr_sub_main, df_pcr_all], axis=1)

        return df_pcr_
