# option_factors
本项目主要实现期权常见因子及其衍生因子的构造（VIX， VIX_call， VIX_put， vc_vp，pcr_oi， pcr_volume， skew）

## 文件简介
  iv.py: 基于B-S公式反解得到的隐含波动率  
  
  vix.py:基于芝加哥商品期货交易所的白皮书计算的无模型的隐含波动率，也称之为恐慌指数，包括衍生指标（VIX_CALL, VIX_PUT, VC_VP）  
  
  pcr.py:基于volume / oi计算的pcr指标  
  
  skew.py:基于芝加哥商品期货交易所的白皮书计算的无模型的偏度指数  
  
  facts_tools.py:包括生成因子和标的的可视化以及主力合约和次主力合约的筛选  
  

## 代码细节
  主力合约和次主力合约的筛选规则：使用主力合约映射表得到主力合约（标签为1），次主力合约为剔除主力合约后，合约持仓量最大的合约<bar>
  计算的结果均剔除末日期权<bar>
  得到的结果未进行标准化<bar>
