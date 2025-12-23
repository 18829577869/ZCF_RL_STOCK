"""
V11 实时预测系统 - 全功能集成版（道通科技688208专用）
整合 V7、V9、V10 的所有功能：
1. V7功能：技术指标、多数据源、LLM解释、成本模型、PPO强化学习
2. V9功能：LSTM/GRU、注意力机制、动态参数优化、自动学习优化
3. V10功能：Transformer、多模态处理、实时可视化、全息动态模型

设计理念：多模型协同工作，智能融合决策
专用标的：道通科技688208（科创板，汽车电子）
"""

from real_time_predict_v11_118013 import *  # 复用全部V11逻辑

# ==================== 道通科技专用配置覆盖 ====================

# 使用道通科技专用 PPO 模型
MODEL_PATH = "ppo_stock_v7_688208.zip"  # V11使用专用PPO模型（道通科技688208）

# 实盘/半实盘主要决策标的：道通科技
STOCK_CODE = 'sh.688208'  # 股票代码（道通科技）

# 补充名称映射（防止未命名时显示代码）
if 'get_stock_name' in globals():
    # 直接扩展原有映射
    _orig_get_stock_name = get_stock_name

    def get_stock_name(code):
        stock_name_map_ext = {
            'sh.688208': '道通科技',
        }
        # 先查扩展映射，再回退原函数
        if code in stock_name_map_ext:
            return stock_name_map_ext[code]
        return _orig_get_stock_name(code)


