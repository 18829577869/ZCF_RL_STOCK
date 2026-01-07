# -*- coding: utf-8 -*-
"""
移除股票备份文件 - Python格式
备份日期: 2025-01-05
来源文件: batch_predict_v18_ma_alignment.py
说明: 以下9只股票已从STOCK_LIST中移除，如需恢复可直接复制使用
"""

# ==================== 被移除的股票列表（Python代码格式）====================

REMOVED_STOCKS = [
    # 3   sh.688208 道通科技 → 使用道通转债(118013)模型
    {'code': 'sh.688208', 'name': '道通科技', 'model': 'ppo_stock_v7_118013.zip', 'rank': 3, 'sharpe': 2.87, 'return': 33.70, 'drawdown': 2.42, 'strategy': '🔵 稳健型'},  # 排名3：夏普2.87，收益率+33.70%，回撤2.42%
    
    # 11  sz.002335 科华数据 → 使用自身002335专用模型（如果有），否则使用高澜股份(300499)模型
    {'code': 'sz.002335', 'name': '科华数据', 'model': 'ppo_stock_v7_300499.zip', 'rank': 11, 'sharpe': 2.28, 'return': 103.71, 'drawdown': 14.82, 'strategy': '🟡 进取型'},  # 排名11：夏普2.28，收益率+103.71%，回撤14.82%（电力设备板块，使用高澜模型）
    
    # 12  sz.000777 中核科技 → 使用汉威科技(300007)模型
    {'code': 'sz.000777', 'name': '中核科技', 'model': 'ppo_stock_v7_300007.zip', 'rank': 12, 'sharpe': 2.22, 'return': 38.03, 'drawdown': 8.28, 'strategy': '🔵 稳健型'},  # 排名12：黄金组合：高夏普、低回撤、收益稳，夏普2.22，收益率+38.03%，回撤8.28%
    
    # 13  sz.002851 麦格米特 → 使用自身002851专用模型
    {'code': 'sz.002851', 'name': '麦格米特', 'model': 'ppo_stock_v7_002851.zip', 'rank': 13, 'sharpe': 2.20, 'return': 33.24, 'drawdown': 7.18, 'strategy': '🟢 均衡型'},  # 排名13：夏普2.20，收益率+33.24%，回撤7.18%
    
    # 14  sh.118013 道通转债 → 使用自身118013专用模型
    {'code': 'sh.118013', 'name': '道通转债', 'model': 'ppo_stock_v7_118013.zip', 'rank': 14, 'sharpe': 2.12, 'return': 33.41, 'drawdown': 7.41, 'strategy': '🟢 均衡型'},  # 排名14：夏普2.12，收益率+33.41%，回撤7.41%
    
    # 15  sz.300153 科泰电源 → 使用自身300153专用模型
    {'code': 'sz.300153', 'name': '科泰电源', 'model': 'ppo_stock_v7_300153.zip', 'rank': 15, 'sharpe': 2.08, 'return': 47.51, 'drawdown': 4.52, 'strategy': '🔵 稳健型'},  # 排名15：夏普2.08，收益率+47.51%，回撤4.52%
    
    # 23  sz.300900 广联航空 → 使用航天电器(002025)模型（军工/航空航天板块）
    {'code': 'sz.300900', 'name': '广联航空', 'model': 'ppo_stock_v7_002025.zip', 'rank': 23, 'sharpe': 1.45, 'return': 44.62, 'drawdown': 20.69, 'strategy': '🟡 进取型'},  # 排名23：本次新增，夏普1.45，收益率+44.62%，回撤20.69%
    
    # 27  sh.601012 隆基绿能 → 使用自身601012专用模型
    {'code': 'sh.601012', 'name': '隆基绿能', 'model': 'ppo_stock_v7_601012.zip', 'rank': 27, 'sharpe': 1.07, 'return': 12.19, 'drawdown': 7.77, 'strategy': '🟢 均衡型'},  # 排名27：V11特色模型，夏普1.07，收益率+12.19%，回撤7.77%
    
    # 28  sh.600363 联创光电 → 使用通用模型
    {'code': 'sh.600363', 'name': '联创光电', 'model': 'ppo_stock_v7.zip', 'rank': 28, 'sharpe': 0.88, 'return': 1.80, 'drawdown': 1.00, 'strategy': '🟢 均衡型'},  # 排名28：第二份榜单，夏普0.88，收益率+1.80%，回撤1.00%
]

# ==================== 使用说明 ====================
# 如需恢复这些股票到 batch_predict_v18_ma_alignment.py 的 STOCK_LIST 中：
# 1. 打开 batch_predict_v18_ma_alignment.py
# 2. 找到 STOCK_LIST 定义位置（约第1419行）
# 3. 将 REMOVED_STOCKS 中的条目复制到 STOCK_LIST 列表的适当位置
# 4. 保存文件即可

# ==================== 统计信息 ====================
REMOVED_COUNT = len(REMOVED_STOCKS)  # 9只
REMOVED_DATE = "2025-01-05"
SOURCE_FILE = "batch_predict_v18_ma_alignment.py"

if __name__ == "__main__":
    print(f"备份信息:")
    print(f"  移除日期: {REMOVED_DATE}")
    print(f"  来源文件: {SOURCE_FILE}")
    print(f"  移除股票数量: {REMOVED_COUNT}只")
    print(f"\n被移除的股票列表:")
    for idx, stock in enumerate(REMOVED_STOCKS, 1):
        print(f"  {idx}. {stock['name']}({stock['code']}) | 夏普{stock['sharpe']} | 收益+{stock['return']}% | 回撤{stock['drawdown']}% | {stock['strategy']}")

