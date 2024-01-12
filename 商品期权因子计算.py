# -*- coding: utf-8 -*-
"""
@Create : 2023/11/30 9:20
@Author : fu Yurong
@Package: 商品期权因子计算.py
@Project : option_facts
@software: PyCharm
"""

import pandas as pd
from iv import IV
from vix import VIX
from pcr_commodity_future_option import PCR_commodity_future
from skew import SKEW
from facts_tools import get_plot, sorted_data
import os
import numpy as np
from tqdm import tqdm


def cal_facts(
              facts_type,
              facts_name,
              product_type,
              pcr_type,
              facts,
              df_future,
              columns_to_plot,
              plots=False,
              save_data=False,
              save_plots=False,
              dir_roots=r'D:\实习\量道投资实习\test_res',
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

    # 计算facts
    if facts_type == 'pcr':
        res = facts.get_df_(product_type=product_type, pcr_type=pcr_type)
    elif facts_type == 'vix':
        res = facts.get_df_(product_type=product_type)
    elif facts_type == 'skew':
        res = facts.get_df_(product_type=product_type)
    elif facts_type == 'iv':
        res = facts.get_df_(product_type=product_type)
    else:
        raise TypeError

    res = pd.DataFrame(res)
    res.reset_index(inplace=True, names='datetime')
    res['datetime'] = pd.to_datetime(res['datetime'])
    df_future_ = df_future.copy()

    df_future_ = df_future_[df_future_['Product'] == product_type]
    df_future_['datetime'] = pd.to_datetime(df_future_['datetime'])

    # 价格后复权(等差复权)
    adjust = df_future_['close'].shift(1) - df_future_['open']
    df_future_['adjust'] = np.where(
        df_future_['instrument_main'] != df_future_['instrument_main'].shift(1),
        adjust, 0)
    df_future_['adj_close'] = df_future_['close'] + df_future_['adjust'].cumsum()

    m = pd.merge(
        res,
        df_future_[['datetime', 'adj_close']],
        on=['datetime']
    )

    m.rename(columns={'adj_close': 'close'}, inplace=True)

    if not os.path.exists(dir_roots):
        os.makedirs(dir_roots)

    file_save_path = os.path.join(dir_roots, f'{facts_type}_{pcr_type}', 'data')
    plot_save_path = os.path.join(dir_roots, f'{facts_type}_{pcr_type}', 'plots')

    if not m.empty:
        if plots:

            if not os.path.exists(plot_save_path):
                os.makedirs(plot_save_path)

            get_plot(df_facts=m, product_type=product_type, columns_to_plot=columns_to_plot,
                     facts_name=facts_name, save_plots=save_plots, save_path=plot_save_path)

        if save_data:

            if not os.path.exists(file_save_path):
                os.makedirs(file_save_path)

            file_path = os.path.join(file_save_path, f"{product_type}_{facts_name}.csv")
            m.to_csv(file_path)

    else:

        return res


if __name__ == '__main__':


    # 所需字段
    # ['datetime', 'option_price', 'underlying_future', 'strike_price', 'type', 'expire_datetime', 'symbol', 'Product', 'volume', 'open_oi', 'close_oi']
    # datetime, expire_datetime: 要求为datetime

    # 期权数据(和映射表合并)
    df_opts = pd.read_csv(r"D:\python_projects\pythonProject1\深度学习\work\option_facts\codes\data\need_data.csv")

    # 主力合约映射表
    df_main_info_ = pd.read_csv(r"D:\实习\量道投资实习\data\mainInfo.csv")
    # 商品期货行情表
    df_future_data_ = pd.read_csv(r"D:\实习\量道投资实习\data\futureData.csv")


    sorted_future_data = sorted_data(df_future=df_future_data_,
                                     df_main_info=df_main_info_,
                                     sub_main_type='open_interest'
                                     )

    # sorted_future_data.to_csv('sorted_future_data.csv')


    # 剔除指数
    sorted_future_data = sorted_future_data[sorted_future_data['is_index'] == 0]

    # 主力合约对应期权数据
    df_opts['datetime'] = pd.to_datetime(df_opts['datetime'])
    sorted_future_data['datetime'] = pd.to_datetime(sorted_future_data['datetime']).dt.date
    sorted_future_data['datetime'] = pd.to_datetime(sorted_future_data['datetime'])

    df_opts.rename(columns={'Product': 'product'}, inplace=True)
    df_opts['type'] = df_opts['option_class'].apply(lambda x: x[0])
    df_opts['type'] = df_opts['type'].astype(str)
    df_opts['type'] = df_opts['type'].apply(lambda x: x.upper())

    df_opts_need = pd.merge(
                            df_opts,
                            sorted_future_data[['datetime', 'close', 'instrument']],
                            on=['datetime', 'instrument']
                            )
    df_opts_need.rename(columns={'close_y': 'close',
                                 'close_x': 'option_price'},
                                  inplace=True)
    df_opts_need['type'] = df_opts_need['option_class'].apply(lambda x: x[0])
    df_opts_need['expire_datetime'] = pd.to_datetime(df_opts_need['expire_datetime'])
    df_opts_need['time_diff'] = (df_opts_need['expire_datetime'] - df_opts_need['datetime']).apply(lambda x: x.days)


    # sorted_future_data.to_csv('main_and_sub.csv')

    facts_1 = PCR_commodity_future(df_option=df_opts, df_future=sorted_future_data)
    facts_2 = VIX(df=df_opts_need)
    facts_3 = SKEW(df=df_opts_need)
    facts_4 = IV(iterations_num=2, precision_cap=0.01, r=0.0003, df=df_opts_need)

    product_type_list = sorted_future_data['Product'].unique()

    df_future = sorted_future_data[(sorted_future_data['Product'] == 'I') & (sorted_future_data['main_sign'] == 1)]
    cal_facts(df_future=df_future,
              facts_type='pcr',  # ['pcr', 'vix', 'skew']
              facts_name='pcr',
              product_type='I',
              pcr_type='volume',  # ['volume', 'oi']
              facts=facts_1,  # 传入的实例化因子,
              columns_to_plot=None,  # 默认对所有因子可视化
              plots=True,  # 是否可视化图片
              save_data=True,  # 是否保存数据
              save_plots=True,  # 是否保存图片
              )

    """

    for p in tqdm(product_type_list):

        print(p)

        try:

            # df_iv = facts_4.get_df_(df=df_facts)
            # 传入主力合约
            df_future = sorted_future_data[(sorted_future_data['Product'] == p) & (sorted_future_data['main_sign'] == 1)]
            cal_facts(df_future=df_future,
                      facts_type='iv',                 # ['pcr', 'vix', 'skew']
                      facts_name='iv',
                      product_type=p,
                      pcr_type=None,                # ['volume', 'oi']
                      facts=facts_4,                    # 传入的实例化因子,
                      columns_to_plot=None,             # 默认对所有因子可视化
                      plots=True,                       # 是否可视化图片
                      save_data=True,                  # 是否保存数据
                      save_plots=True,                 # 是否保存图片
                      )

        except Exception as e:

            print(f"Error processing product {p}: {e}")
        """