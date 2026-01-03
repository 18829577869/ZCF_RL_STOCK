# backtest_002706_with_002837_model.py - 良信股份使用英维克模型回测并记录操作
# -*- coding: utf-8 -*-
"""
使用英维克(002837)模型对良信股份(002706)进行回测，并记录所有操作到文件

止损策略：
- 如果当日跌幅大于3%且有持仓，则强制全卖出（止损）
"""
from stable_baselines3 import PPO
from stock_env_v6 import StockTradingEnv
import os
import numpy as np
import pandas as pd
from datetime import datetime
import csv

# 模型路径
MODEL_PATH = "ppo_stock_v7_002837.zip"

# 股票信息
STOCK_CODE = "sz.002706"
STOCK_NAME = "良信股份"
TEST_FILE = "stockdata_v7_002837/test/sz.002706.良信股份.csv"

# 初始资金
INITIAL_BALANCE = 50000  # 5万元

# 操作记录文件
OUTPUT_DIR = "backtest_operation_records"
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
OPERATION_RECORD_FILE = os.path.join(OUTPUT_DIR, f"backtest_002706_002837_operations_{timestamp}.csv")

print("="*70)
print("良信股份(002706)使用英维克(002837)模型回测")
print("="*70)
print(f"模型文件: {MODEL_PATH}")
print(f"股票代码: {STOCK_CODE}")
print(f"股票名称: {STOCK_NAME}")
print(f"数据文件: {TEST_FILE}")
print(f"初始资金: {INITIAL_BALANCE:,.0f} 元")
print(f"操作记录文件: {OPERATION_RECORD_FILE}")
print("="*70)

# 检查文件是否存在
if not os.path.exists(MODEL_PATH):
    print(f"\n[错误] 模型文件不存在: {MODEL_PATH}")
    exit(1)

if not os.path.exists(TEST_FILE):
    print(f"\n[错误] 测试数据文件不存在: {TEST_FILE}")
    exit(1)

# 加载模型
print(f"\n[加载] 正在加载模型: {MODEL_PATH}")
try:
    model = PPO.load(MODEL_PATH)
    print(f"[成功] 模型加载成功")
except Exception as e:
    print(f"[错误] 模型加载失败: {e}")
    exit(1)

# 读取数据文件以获取日期信息
print(f"\n[加载] 正在加载数据文件: {TEST_FILE}")
df_data = pd.read_csv(TEST_FILE)
df_data['date'] = pd.to_datetime(df_data['date'])
print(f"[成功] 数据加载成功，共 {len(df_data)} 条记录")

# 创建环境
print(f"\n[创建] 正在创建回测环境...")
try:
    env = StockTradingEnv(TEST_FILE, initial_balance=INITIAL_BALANCE)
    obs, _ = env.reset()
    print(f"[成功] 环境创建成功")
except Exception as e:
    print(f"[错误] 环境创建失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 初始化操作记录
operation_records = []

# 开始回测
print("\n" + "="*70)
print("开始回测...")
print("="*70 + "\n")

step_count = 0
done = False

# 记录操作前的状态
prev_trade_count = 0
stop_profit_count = 0  # 止损触发次数（保持变量名，实际是止损）

while not done:
    # 获取当前日期和价格
    current_row = df_data.iloc[env.current_step]
    current_date = current_row['date']
    current_price = float(current_row['close'])
    current_pct_chg = float(current_row['pctChg'])  # 当日涨跌幅
    
    # 记录操作前的状态
    balance_before = env.balance
    shares_before = env.shares_held
    net_worth_before = env.net_worth
    
    # 预测动作
    action, _ = model.predict(obs, deterministic=True)
    
    # 将 action 转换为整数（处理 numpy 数组的情况）
    if isinstance(action, np.ndarray):
        action = int(action.item())
    elif isinstance(action, (list, tuple)):
        action = int(action[0])
    else:
        action = int(action)
    
    # 止损策略：如果当日跌幅大于3%且有持仓，则全卖出
    original_action = action
    stop_loss_triggered = False
    if current_pct_chg < -3.0 and shares_before > 0:
        action = 6  # 强制全卖出
        stop_loss_triggered = True
        stop_profit_count += 1  # 保持变量名一致，实际是止损计数
    
    # 动作映射
    action_map = {
        0: "持有",
        1: "买入10%",
        2: "买入25%",
        3: "买入50%",
        4: "买入100%",
        5: "卖出50%",
        6: "卖出100%"
    }
    action_name = action_map.get(action, "未知")
    if stop_loss_triggered:
        action_name = f"{action_name} (止损触发:跌幅{current_pct_chg:.2f}%<-3%)"
    
    # 执行动作
    obs, reward, done, truncated, info = env.step(action)
    
    # 记录操作后的状态
    balance_after = env.balance
    shares_after = env.shares_held
    net_worth_after = env.net_worth
    
    # 检查是否有新交易
    if len(env.trade_history) > prev_trade_count:
        # 有新交易发生
        new_trades = env.trade_history[prev_trade_count:]
        for trade in new_trades:
            operation_records.append({
                '日期': current_date.strftime('%Y-%m-%d'),
                '时间戳': current_date.strftime('%Y-%m-%d %H:%M:%S'),
                '股票代码': STOCK_CODE,
                '股票名称': STOCK_NAME,
                '操作类型': '买入' if trade['action'] == 'BUY' else '卖出',
                '动作': action_name,
                '当日涨跌幅(%)': f"{current_pct_chg:.2f}",
                '止损触发': '是' if stop_loss_triggered else '否',
                '原始动作': action_map.get(original_action, "未知") if stop_loss_triggered else '',
                '价格': f"{trade['price']:.2f}",
                '数量': f"{trade['shares']:.0f}",
                '金额': f"{trade['price'] * trade['shares']:.2f}",
                '手续费': f"{trade['fee']:.2f}",
                '操作前持仓': f"{shares_before:.0f}",
                '操作前资金': f"{balance_before:.2f}",
                '操作后持仓': f"{shares_after:.0f}",
                '操作后资金': f"{balance_after:.2f}",
                '总资产': f"{net_worth_after:.2f}",
                '步骤': trade['step']
            })
        prev_trade_count = len(env.trade_history)
    elif action != 0:  # 如果不是持有动作，也记录（即使没有实际交易）
        # 记录决策（可能因为资金不足等原因未执行）
        operation_records.append({
            '日期': current_date.strftime('%Y-%m-%d'),
            '时间戳': current_date.strftime('%Y-%m-%d %H:%M:%S'),
            '股票代码': STOCK_CODE,
            '股票名称': STOCK_NAME,
            '操作类型': '决策',
            '动作': action_name,
            '当日涨跌幅(%)': f"{current_pct_chg:.2f}",
            '止损触发': '是' if stop_loss_triggered else '否',
            '原始动作': action_map.get(original_action, "未知") if stop_loss_triggered else '',
            '价格': f"{current_price:.2f}",
            '数量': "0",
            '金额': "0.00",
            '手续费': "0.00",
            '操作前持仓': f"{shares_before:.0f}",
            '操作前资金': f"{balance_before:.2f}",
            '操作后持仓': f"{shares_after:.0f}",
            '操作后资金': f"{balance_after:.2f}",
            '总资产': f"{net_worth_after:.2f}",
            '步骤': env.current_step,
            '备注': '未执行（可能资金不足或持仓不足）'
        })
    
    step_count += 1
    if step_count % 50 == 0:
        print(f"回测进度: {step_count} 步... (当前日期: {current_date.strftime('%Y-%m-%d')}, 总资产: {env.net_worth:.2f})", end='\r')

print(f"\n回测完成，共 {step_count} 步")

# 获取最终统计
stats = env.get_stats()
print("\n" + "="*70)
print("回测统计结果")
print("="*70)
print(f"最终净值: {stats['final_net_worth']:,.2f} 元")
print(f"总收益率: {stats['total_return']:+.2f}%")
print(f"最大回撤: {stats['max_drawdown']:.2f}%")
print(f"夏普比率: {stats['sharpe_ratio']:.2f}")
print(f"交易次数: {stats['num_trades']}")
print(f"胜率: {stats['win_rate']:.2f}%")
print(f"风险事件: {stats['risk_events']} 次")
print(f"总天数: {stats['total_days']} 天")
print(f"止损触发: {stop_profit_count} 次 (跌幅<-3%全卖出)")

# 保存操作记录
if len(operation_records) > 0:
    print(f"\n[保存] 正在保存操作记录...")
    df_operations = pd.DataFrame(operation_records)
    df_operations.to_csv(OPERATION_RECORD_FILE, index=False, encoding='utf-8-sig')
    print(f"[成功] 操作记录已保存到: {OPERATION_RECORD_FILE}")
    print(f"       共记录 {len(operation_records)} 条操作")
    
    # 统计信息
    buy_count = len([r for r in operation_records if r['操作类型'] == '买入'])
    sell_count = len([r for r in operation_records if r['操作类型'] == '卖出'])
    decision_count = len([r for r in operation_records if r['操作类型'] == '决策'])
    stop_loss_ops = len([r for r in operation_records if r.get('止损触发', '否') == '是'])
    
    print(f"\n操作统计:")
    print(f"  买入次数: {buy_count}")
    print(f"  卖出次数: {sell_count}")
    print(f"  决策次数: {decision_count}")
    print(f"  止损触发次数: {stop_profit_count} 次 (跌幅<-3%)")
    print(f"  总操作数: {len(operation_records)}")
else:
    print(f"\n[警告] 没有记录到任何操作")

# 保存回测统计结果
stats_file = os.path.join(OUTPUT_DIR, f"backtest_002706_002837_stats_{timestamp}.csv")
stats_df = pd.DataFrame([{
    '股票代码': STOCK_CODE,
    '股票名称': STOCK_NAME,
    '模型': '英维克(002837)模型',
    '初始资金': INITIAL_BALANCE,
    '最终净值': stats['final_net_worth'],
    '总收益率(%)': stats['total_return'],
    '最大回撤(%)': stats['max_drawdown'],
    '夏普比率': stats['sharpe_ratio'],
    '交易次数': stats['num_trades'],
    '胜率(%)': stats['win_rate'],
    '风险事件': stats['risk_events'],
    '总天数': stats['total_days'],
    '操作记录数': len(operation_records),
    '止损触发次数': stop_profit_count,
    '止损策略': '跌幅<-3%全卖出'
}])
stats_df.to_csv(stats_file, index=False, encoding='utf-8-sig')
print(f"\n[保存] 回测统计已保存到: {stats_file}")

print("\n" + "="*70)
print("[完成] 回测完成！")
print("="*70)
print(f"[模型] 使用模型: {MODEL_PATH}")
print(f"[标的] {STOCK_NAME}({STOCK_CODE})")
print(f"[记录] 操作记录文件: {OPERATION_RECORD_FILE}")
print(f"[统计] 统计结果文件: {stats_file}")

