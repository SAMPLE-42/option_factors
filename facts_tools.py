# -*- coding: utf-8 -*-
"""
@Create : 2023/11/7 9:20
@Author : fu Yurong
@Package: facts_tools.py
@Project : option_facts
@software: PyCharm
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def prepare_data(df_opt_data, df_info, df_fut_data, rf, fixed=False):
    """提取所需数据"""

    # #(symbol) = 3744 = len(df_option_info['symbol].unique())
    df_merge_ = pd.merge(df_opt_data, df_info, on=['symbol'])

    df_merge_['expire_datetime'] = pd.to_datetime(df_merge_['expire_datetime'])
    df_merge_['datetime'] = pd.to_datetime(df_merge_['datetime'])
    df_fut_data['datetime'] = pd.to_datetime(df_fut_data['datetime'])

    df_merge_.rename(columns={'instrument_x': 'instrument'}, inplace=True)
    print(df_merge_.columns)

    # 筛选所需字段
    use_filds = ['datetime', 'option_price', 'underlying_future',
                 'strike_price', 'type', 'expire_datetime', 'symbol',
                 'Product_x', 'volume', 'open_oi', 'close_oi']

    opt_data = df_merge_[use_filds]

    opt_data['time_diff'] = (opt_data['expire_datetime'] - opt_data['datetime']).apply(lambda x: x.days)
    opt_data['time_diff_days'] = opt_data['time_diff'] / 365

    # 判断是否使用固定利率
    if fixed:
        opt_data['r'] = rf

    else:
        # 采用 shibor
        # 读取数据，合并，对遗漏天数进行填充
        pass

    # df_future_data中数据缺失
    # #(symbol) < len(df_option_info['symbol].unique())
    res = pd.merge(opt_data,
                   df_fut_data[['underlying_future', 'close', 'datetime']],
                   on=['datetime', 'underlying_future'])

    res.sort_values(['symbol', 'datetime'], inplace=True)
    res.rename(columns={'close': 'future_price',
                        'Product_x': 'product'},
               inplace=True)

    return res


def get_plots(df_facts, df_iv, df_futures_data, product_type=None, columns_to_plot=None, add_iv=False, facts_name=None, save_data=False, save_plots=False):
    """
        可视化
        product_type: ['IH', 'IF', IM]
        columns_to_plot: ['VIXNone', 'VIXcall', 'VIXput', 'vc_vp', 'iv', 'future_price']
    """

    res = {}
    iv_res = {}

    for dt, group in df_futures_data[df_futures_data['Product'] == '{}'.format(product_type)].groupby(['datetime']):
        # 计算期货合约收盘价的均值
        # 实际上应该对主力合约进行筛选形成序列数据
        res[dt] = group['close'].mean()

    df_res = pd.Series(res, name='close').dropna()
    df_merge = pd.merge(
        df_facts,
        df_res,
        left_index=True,
        right_index=True,
        how='left'
        )
    df_merge.reset_index(inplace=True)

    df_iv_ = df_iv[df_iv['Product'] == '{}'.format(product_type)]

    for dt, df_group in df_iv_.groupby(['datetime']):
        iv_res[dt] = df_group['iv'].mean()

    iv_df = pd.DataFrame(pd.Series(iv_res, name='iv').dropna())
    iv_df.reset_index(inplace=True)

    # 剔除极端值
    iv_df.replace([np.inf, -np.inf], 0, inplace=True)
    iv_df.drop(iv_df['iv'].idxmin(), inplace=True)

    df_merge_ = pd.merge(
                iv_df,
                df_merge,
                on=['index'],
                how='left'
                )

    # df_merge_.to_csv('{}_vix_iv_future.csv'.format(product_type))
    # columns = [index, iv, vix_call, vix_put, vix, close]

    df_merge_.set_index(['index'], inplace=True)

    if add_iv:
        data_to_plot = df_merge_
    else:
        df_merge.set_index(['index'], inplace=True)
        data_to_plot = df_merge

    if save_data:
        data_to_plot.to_csv(f'{facts_name}_{product_type}.csv')

    # 创建一个图形和两个坐标轴
    fig, ax = plt.subplots(figsize=(15, 6))

    # 设置第一个 y 轴（左侧）和绘制第一列数据
    ax.set_xlabel('time')
    ax.set_ylabel('{}'.format(facts_name), color='tab:blue')
    ax2 = ax.twinx()
    ax2.set_ylabel('price', color='tab:red')

    # 为每个列指定不同的颜色
    colors = ['c', 'r', 'b', 'g', 'm', 'y']

    if columns_to_plot is None:
        columns_to_plot = data_to_plot.columns

    for i, column in enumerate(columns_to_plot):

        if column in data_to_plot.columns[: -1]:
            ax.plot(data_to_plot.index, data_to_plot[column], label=column, color=colors[i])
            ax.legend(loc='upper left')

        elif column == 'close':
            ax2.plot(data_to_plot.index, data_to_plot[column], label='future_price', color=colors[i])
            ax2.legend(loc='upper right')

    ax.grid()

    # 添加标题
    plt.title(f'{facts_name}_{product_type} AND Underlying Products Price')

    # 显示图形
    if save_plots:

        plt.savefig(f'{facts_name}_{product_type}.png', bbox_inches='tight', dpi=200)

    plt.show()

def get_plot(df_facts, product_type=None, columns_to_plot=None, facts_name=None, save_plots=False, save_path=None):

    """
        可视化
        product_type: ['IH', 'IF', IM]
        columns_to_plot: ['VIXNone', 'VIXcall', 'VIXput', 'vc_vp', 'iv', 'future_price']
    """

    res = {}
    iv_res = {}

    df_data = df_facts.copy()
    df_data.set_index(['datetime'], drop=True, inplace=True)
    df_data.sort_values(['datetime'], inplace=True)
    df_data = df_data.interpolate(method='linear')
    df_data = df_data.fillna(df_data.mean())
    # df_merge_.to_csv('{}_vix_iv_future.csv'.format(product_type))
    # columns = [index, iv, vix_call, vix_put, vix, close]

    # df_data.set_index(['index'], inplace=True)
    print(df_data)

    # 创建一个图形和两个坐标轴
    fig, ax = plt.subplots(figsize=(15, 6))

    # 设置第一个 y 轴（左侧）和绘制第一列数据
    ax.set_xlabel('time')
    ax.set_ylabel('{}'.format(facts_name), color='tab:blue')
    ax2 = ax.twinx()
    ax2.set_ylabel('price', color='tab:red')

    # 为每个列指定不同的颜色
    colors = ['c', 'r', 'b', 'g', 'm', 'y']

    if columns_to_plot is None:
        columns_to_plot = df_data.columns
        print(columns_to_plot)

    for i, column in enumerate(columns_to_plot):

        if column in df_data.columns[: -1]:
            ax.plot(df_data[column], label=column, color=colors[i])
            ax.legend(loc='upper left')

        elif column == 'close':
            ax2.plot(df_data[column], label='future_price', color=colors[i])
            ax2.legend(loc='upper right')

    ax.grid(True)

    # 添加标题
    plt.title(f'{facts_name}_{product_type} AND Underlying Products Price')

    file_path = os.path.join(save_path, f'{facts_name}_{product_type}.png')

    # 显示图形
    if save_plots:

        plt.savefig(file_path, bbox_inches='tight', dpi=200)

    plt.show()


# 定义函数判断是否为指数合约
def is_index_contract(contract_code):
    if pd.notna(contract_code):
        # 获取品种代码和后面的数字部分
        variety_code = contract_code[:-4]
        number_part = contract_code[-4:]

        # 判断是否为指数
        return all(c.isdigit() for c in number_part) and len(set(number_part)) == 1
    else:
        return False  # 如果是缺失值，不属于指数合约

def sorted_data(df_future, df_main_info, sub_main_type):
    """
        标记主力合约和次主力合约
        次主力挑选依据: open_interest, volume
    """

    future_data = df_future.copy()
    main_info = df_main_info.copy()

    main_info.rename(columns={'product': 'Product'}, inplace=True)

    df_future_merge = pd.merge(
        future_data,
        main_info[['TRADE_DAYS', 'Product', 'instrument', 'contract']],
        on=['Product', 'TRADE_DAYS'],
        how='right'
    )

    df_future_merge.rename(columns={'instrument_x': 'instrument',
                                    'instrument_y': 'instrument_main'},
                           inplace=True)

    df_future_merge['main_sign'] = np.where(
        (df_future_merge['instrument'] == df_future_merge['instrument_main']), 1, 0)

    # 标识指数
    df_future_merge['is_index'] = df_future_merge['instrument'].apply(lambda x: 1 if is_index_contract(x) else 0)

    df_future_merge['sub_sign'] = 0
    df_ = df_future_merge[(df_future_merge['main_sign'] == 0) & (df_future_merge['is_index'] == 0)]
    idx_2 = df_.groupby(['trade_date', 'Product'])[f'{sub_main_type}'].idxmax()

    df_future_merge.loc[idx_2, 'sub_sign'] = 1

    df_future_merge['exchange_id'] = df_future_merge['contract'].apply(lambda x: x[x.find('.'):].replace('.', ''))
    df_future_merge['underlying_symbol'] = df_future_merge['exchange_id'] + '.' + df_future_merge['instrument']
    df_future_merge.rename(columns={'trade_date': 'datetime'}, inplace=True)
    df_future_merge['datetime'] = pd.to_datetime(df_future_merge['datetime']).dt.date

    return df_future_merge


