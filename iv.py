# -*- coding: utf-8 -*-
"""
@Create : 2023/11/7 9:20
@Author : fu Yurong
@Package: iv.py
@Project : option_facts
@software: PyCharm
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from numba import jit


class IV:
    """
        基于b-s公式，使用牛顿迭代法计算隐含波动率
            s_0: 期初价格
            k: 执行价格
            r: 无风险收益率
            sigma: 期权波动率
            T: 时间
            q: 股息率
            price_type: c / p
    """

    def __init__(self, iterations_num, precision_cap, r, df):
        # columns = ['datetime', 'option_price', 'underlying_future',
        #           'strike_price', 'type', 'expire_datetime', 'symbol',
        #            'Product_x', 'volume', 'open_oi', 'close_oi']
        self.iterations_num = iterations_num        # 迭代次数
        self.precision_cap = precision_cap          # 精度阈值
        self.r = r                                  # 无风险收益率
        self.df = df

    @staticmethod
   #  @jit(nopython=False)
    def bs_price(s_0, k, r, sigma, t, q=0.0, price_type=None):
        """
            B-S公式
        """
        d_1 = (np.log(s_0 / k) + (r - q + sigma*sigma/2.) * t) / (sigma * np.sqrt(t))
        d_2 = (np.log(s_0 / k) + (r - q - sigma*sigma/2.) * t) / (sigma * np.sqrt(t))

        n = norm.pdf
        N = norm.cdf

        if price_type == 'C':
            # 看涨期权
            price = s_0 * np.exp(-q * t) * N(d_1) - k * np.exp(-r * t) * N(d_2)
        elif price_type == 'P':
            # 看跌期权
            price = k * np.exp(-r * t) * N(-d_2) - s_0 * np.exp(-q * t) * N(-d_1)
        else:
            raise TypeError

        return price

    @staticmethod
   #  @jit(nopython=False)
    def bs_vega(s_0, k, r, sigma, t, q=0.0):
        """
            vega = f'(x)
        """
        n = norm.pdf
        N = norm.cdf
        d_1 = (np.log(s_0 / k) + (r - q + sigma*sigma/2.) * t) / (sigma * np.sqrt(t))
        return s_0 * np.sqrt(t) * n(d_1)

   #  @jit(nopython=False)
    def cal_vol(self, target_value, s_0, k, t, r, price_type=None):
        """
            牛顿迭代法计算iv
        """

        max_iterations = self.iterations_num
        precision = self.precision_cap

        # 初始化sigma
        sigma = 0.5

        for i in range(0, max_iterations):
            # 理论价格(bs公式价格)
            price = self.bs_price(s_0, k, r, sigma, t, q=0.0, price_type=price_type)
            vega = self.bs_vega(s_0, k, r, sigma, t, q=0.0)

            # 检查分母是否为零或无效值
            if np.abs(vega) < 1e-10 or not np.isfinite(vega):
                # 处理分母为零或无效值的情况
                # 可以返回默认值或者引发异常
                return np.nan  # 或者 raise ValueError("分母无效")

            price = price
            diff = target_value - price

            # print(f"第{i}迭代, vol:{sigma}, diff:{diff}")

            if np.abs(diff) < precision:

                return sigma

            else:
                # f(x) / f'(x)
                sigma += diff / vega

        # 未找到值，返回迄今为止的最佳猜测
        return sigma

    def get_df_(self, product_type):
        """
            iv计算
        """

        df = self.df.copy()
        df.rename(columns={'close': 'future_price'}, inplace=True)
        print(df.info())
        df = df[df['product'] == product_type]
        df['iv'] = df.apply(lambda row: self.cal_vol(
            target_value=row['option_price'],           # 期权价格
            s_0=row['future_price'],                    # 标的价格
            k=row['strike_price'],                      # 执行价格
            t=row['time_diff'] / 365,                   # 剩余到期时间
            r=self.r,                                   # 无风险收益率
            price_type=row['type']                      # 期权类型
        ), axis=1)
        print(df)

        return df





