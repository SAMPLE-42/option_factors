# -*- coding: utf-8 -*-
"""
@Create : 2023/11/7 9:20
@Author : fu Yurong
@Package: main.py
@Project : option_facts
@software: PyCharm
"""

import pandas as pd
from iv import IV
from vix import VIX
from pcr import PCR
from skew import SKEW
from facts_tools import prepare_data, get_plots


def cal_facts(data_iv,
              option_data,
              future_data,
              facts_type,
              facts_name,
              product_type,
              pcr_type,
              facts,
              columns_to_plot,
              plots=False,
              add_iv=False,
              save_data=False,
              save_plots=False,
              ):
    """
        df_iv, option_data, future_data: 所需数据, dataframe
        facts_type: ['pcr', 'iv', 'vix', 'skew'], str
        facts_name: ['SKEW', VIX', 'IV', 'PCR_OI', 'PCR_VOLUME']
        product_type: ['IF', 'IH', 'IM'], str
        pcr_type: ['oi', 'volume']
        facts: VIX(), IV(), PCR(), SKEW(),
        columns_to_plot:
            VIX: ['VIXNone', 'VIXcall', 'VIXput', 'vc_vp', 'close']
            PCR: ['pcr_fm', 'pcr_sfm', 'pcr_fm+sfm', 'pcr_all', close]
            SKEW: ['skew', 'close']
            IV: ['iv', 'close']
        plots: 绘图, bools
        save_data: 存储数据, bools
        save_plots: 存储图片, bools
    """

    df_iv = data_iv.copy()
    df_iv['datetime'] = pd.to_datetime(df_iv['datetime'])
    option_data['datetime'] = pd.to_datetime(option_data['datetime'])
    df_iv_ = pd.merge(df_iv, option_data[['symbol', 'Product', 'datetime']], on=['symbol', 'datetime'], how='left')

    # 计算facts
    if facts_type == 'pcr':
        res = facts.get_df_(product_type=product_type, pcr_type=pcr_type)
    elif facts_type == 'vix':
        res = facts.get_df_(product_type=product_type)
    elif facts_type == 'skew':
        res = facts.get_df_(product_type=product_type)
    else:
        raise TypeError

    if plots:
        get_plots(df_facts=res,
                  df_iv=df_iv_,
                  df_futures_data=future_data,
                  product_type=product_type,
                  columns_to_plot=columns_to_plot,
                  add_iv=add_iv,
                  facts_name=facts_name,
                  save_data=save_data,
                  save_plots=save_plots
                  )
    else:
        pass

    return res


if __name__ == '__main__':

    # df_future_data = pd.read_csv(r"D:\download\data\futureData.csv")
    # df_main_info = pd.read_csv(r"D:\download\data\mainInfo.csv")
    # df_option_info = pd.read_csv(r"D:\download\data\option_info.csv")  # instrument_id是唯一字段， 表中存在重复情况
    # df_option_data = pd.read_csv(r"D:\download\data\optionData.csv")
    # df_iv = pd.read_csv(r"D:\python_projects\pythonProject1\深度学习\work\iv_2023_11_2.csv", index_col=0)



    df_option_data.rename(columns={'close': 'option_price',
                                   'TYPE': 'type'},
                          inplace=True)

    df_option_info.rename(columns={'instrument_id': 'symbol',
                                   'close': 'future_price'},
                          inplace=True)

    df_future_data.rename(columns={'instrument': 'underlying_future',
                                   'trade_date': 'datetime'},
                          inplace=True)

    df_sorted = prepare_data(df_opt_data=df_option_data,
                             df_info=df_option_info,
                             df_fut_data=df_future_data,
                             rf=0.0003,
                             fixed=True)

    # df_data: 所需字段
    # ['datetime', 'option_price', 'underlying_future', 'strike_price', 'type', 'expire_datetime', 'symbol', 'Product', 'volume', 'open_oi', 'close_oi']
    # datetime, expire_datetime: 要求为datetime
    # Pr
    df_facts = pd.read_csv('   ')           # ['datetime', 'facts', '']


    facts_1 = PCR(df=df_sorted)
    facts_2 = VIX(df=df_sorted)
    facts_3 = SKEW(df=df_sorted)
    facts_4 = IV(iterations_num=100, precision_cap=0.0001, r=0.0003)

    product_type_list = ['IH', 'IF', 'IM']
    for p in product_type_list:
        cal_facts(data_iv=df_iv,
                  option_data=df_option_data,
                  future_data=df_future_data,
                  facts_type='pcr',
                  facts_name='PCR_volume',
                  product_type=p,
                  pcr_type='volume',
                  facts=facts_1,
                  columns_to_plot=None,
                  plots=True,
                  add_iv=False,
                  save_data=True,
                  save_plots=False,
                  )


