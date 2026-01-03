# backtest_300007_model.py - 使用汉威科技300007模型对7只股票进行回测
# -*- coding: utf-8 -*-
"""
使用汉威科技300007的已有模型（ppo_stock_v7_300007.zip）对以下股票进行回测：
1. 汉威科技300007
2. 安培龙301413
3. 中核科技000777
4. 永鼎股份600105
5. 兆威机电003021
6. 大业股份603278
7. 中国卫星600118
"""
from stable_baselines3 import PPO
from stock_env_v6 import StockTradingEnv
import os
import numpy as np
import pandas as pd
from datetime import datetime

# 模型路径
MODEL_PATH = "ppo_stock_v7_300007.zip"

# 初始资金（匹配实盘操作）
INITIAL_BALANCE = 20000  # 2万元

# 回测股票列表
BACKTEST_STOCKS = [
    {'code': 'sz.300007', 'name': '汉威科技', 'test_file': 'stockdata_v7_300007/test/sz.300007.汉威科技.csv'},
    {'code': 'sz.301413', 'name': '安培龙', 'test_file': 'stockdata_v7_300007/test/sz.301413.安培龙.csv'},
    {'code': 'sz.000777', 'name': '中核科技', 'test_file': 'stockdata_v7_300007/test/sz.000777.中核科技.csv'},
    {'code': 'sh.600105', 'name': '永鼎股份', 'test_file': 'stockdata_v7_300007/test/sh.600105.永鼎股份.csv'},
    {'code': 'sz.003021', 'name': '兆威机电', 'test_file': 'stockdata_v7_300007/test/sz.003021.兆威机电.csv'},
    {'code': 'sh.603278', 'name': '大业股份', 'test_file': 'stockdata_v7_300007/test/sh.603278.大业股份.csv'},
    {'code': 'sh.600118', 'name': '中国卫星', 'test_file': 'stockdata_v7_300007/test/sh.600118.中国卫星.csv'},
]

print("="*70)
print("使用汉威科技300007模型进行回测")
print("="*70)
print(f"模型文件: {MODEL_PATH}")
print(f"初始资金: {INITIAL_BALANCE:,.0f} 元")
print(f"回测股票数量: {len(BACKTEST_STOCKS)} 只")
print("="*70)

# 检查模型文件是否存在
if not os.path.exists(MODEL_PATH):
    print(f"\n[错误] 模型文件不存在: {MODEL_PATH}")
    print("请先训练汉威科技300007的模型")
    exit(1)

# 加载模型
print(f"\n[加载] 正在加载模型: {MODEL_PATH}")
try:
    model = PPO.load(MODEL_PATH)
    print(f"[成功] 模型加载成功")
except Exception as e:
    print(f"[错误] 模型加载失败: {e}")
    exit(1)

# 开始回测
print("\n" + "="*70)
print("开始回测...")
print("="*70 + "\n")

all_stats = []
category_stats = {}
hanwei_keji_stats = None

# 优先测试汉威科技300007
test_stocks_sorted = []
hanwei_keji_stock = None
for stock in BACKTEST_STOCKS:
    if stock['code'] == 'sz.300007':
        hanwei_keji_stock = stock
        test_stocks_sorted.append(stock)
        break

# 其他股票
for stock in BACKTEST_STOCKS:
    if stock != hanwei_keji_stock:
        test_stocks_sorted.append(stock)

for stock_info in test_stocks_sorted:
    stock_code = stock_info['code']
    stock_name = stock_info['name']
    test_file = stock_info['test_file']
    
    if not os.path.exists(test_file):
        print(f"[警告] 测试文件不存在: {test_file}")
        print(f"       跳过 {stock_name}({stock_code}) 的回测\n")
        continue
    
    try:
        print(f"[回测] {stock_name}({stock_code})")
        print(f"        数据文件: {os.path.basename(test_file)}")
        
        # 创建环境
        env = StockTradingEnv(test_file, initial_balance=INITIAL_BALANCE)
        obs, _ = env.reset()
        done = False
        
        # 运行回测
        step_count = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            step_count += 1
            if step_count % 50 == 0:
                print(f"        回测进度: {step_count} 步...", end='\r')
        
        # 获取统计数据
        stats = env.get_stats()
        stats['code'] = stock_code
        stats['name'] = stock_name
        stats['file'] = test_file
        
        # 检查是否是汉威科技300007
        is_hanwei_keji = (stock_code == 'sz.300007')
        if is_hanwei_keji:
            stats['is_core'] = True
            hanwei_keji_stats = stats
            print("\n" + "="*70)
            print("🎯 [核心标的] 汉威科技300007回测结果")
            print("="*70)
        else:
            stats['is_core'] = False
        
        all_stats.append(stats)
        
        # 按分类统计（简化，根据股票名称判断）
        category = '传感器' if '300007' in stock_code or '301413' in stock_code else \
                   '核电设备' if '000777' in stock_code else \
                   '通信设备' if '600105' in stock_code else \
                   '精密制造' if '003021' in stock_code else \
                   '金属制品' if '603278' in stock_code else \
                   '航天军工' if '600118' in stock_code else '其他'
        
        if category not in category_stats:
            category_stats[category] = []
        category_stats[category].append(stats)
        
        core_mark = "🎯 [核心]" if is_hanwei_keji else ""
        print(f"{core_mark}[{category}] {stock_name}({stock_code})")
        print(f"   最终净值: {stats['final_net_worth']:,.2f} 元")
        print(f"   总收益率: {stats['total_return']:+.2f}%")
        print(f"   最大回撤: {stats['max_drawdown']:.2f}%")
        print(f"   夏普比率: {stats['sharpe_ratio']:.2f}")
        print(f"   交易次数: {stats['num_trades']}")
        print(f"   胜率: {stats['win_rate']:.2f}%")
        print(f"   风险事件: {stats['risk_events']} 次")
        print()
        
    except Exception as e:
        print(f"[错误] {stock_name}({stock_code}) 回测失败: {e}\n")
        import traceback
        traceback.print_exc()

# 整体统计
if len(all_stats) > 0:
    print("="*70)
    print("[整体统计]")
    print("="*70)
    
    avg_return = np.mean([s['total_return'] for s in all_stats])
    avg_drawdown = np.mean([s['max_drawdown'] for s in all_stats])
    avg_sharpe = np.mean([s['sharpe_ratio'] for s in all_stats])
    avg_win_rate = np.mean([s['win_rate'] for s in all_stats])
    total_trades = sum([s['num_trades'] for s in all_stats])
    total_risk_events = sum([s['risk_events'] for s in all_stats])
    
    print(f"平均收益率: {avg_return:+.2f}%")
    print(f"平均最大回撤: {avg_drawdown:.2f}%")
    print(f"平均夏普比率: {avg_sharpe:.2f}")
    print(f"平均胜率: {avg_win_rate:.2f}%")
    print(f"总交易次数: {total_trades}")
    print(f"总风险事件: {total_risk_events} 次")
    print(f"测试标的数: {len(all_stats)}")
    
    # 汉威科技300007专项统计
    if hanwei_keji_stats:
        print("\n" + "="*70)
        print("🎯 [核心标的专项统计] 汉威科技300007")
        print("="*70)
        print(f"最终净值: {hanwei_keji_stats['final_net_worth']:,.2f} 元")
        print(f"总收益率: {hanwei_keji_stats['total_return']:+.2f}%")
        print(f"最大回撤: {hanwei_keji_stats['max_drawdown']:.2f}%")
        print(f"夏普比率: {hanwei_keji_stats['sharpe_ratio']:.2f}")
        print(f"交易次数: {hanwei_keji_stats['num_trades']}")
        print(f"胜率: {hanwei_keji_stats['win_rate']:.2f}%")
        print(f"风险事件: {hanwei_keji_stats['risk_events']} 次")
    
    # 分类统计
    print("\n" + "="*70)
    print("[分类统计]")
    print("="*70)
    
    for category, stats_list in category_stats.items():
        if len(stats_list) == 0:
            continue
        
        cat_avg_return = np.mean([s['total_return'] for s in stats_list])
        cat_avg_drawdown = np.mean([s['max_drawdown'] for s in stats_list])
        cat_avg_sharpe = np.mean([s['sharpe_ratio'] for s in stats_list])
        cat_trades = sum([s['num_trades'] for s in stats_list])
        
        print(f"\n[{category}] ({len(stats_list)}只)")
        print(f"  平均收益率: {cat_avg_return:+.2f}%")
        print(f"  平均最大回撤: {cat_avg_drawdown:.2f}%")
        print(f"  平均夏普比率: {cat_avg_sharpe:.2f}")
        print(f"  总交易次数: {cat_trades}")
    
    # 最佳/最差
    print("\n" + "="*70)
    best = max(all_stats, key=lambda x: x['total_return'])
    worst = min(all_stats, key=lambda x: x['total_return'])
    
    print(f"\n[最佳] {best['name']}({best['code']})")
    print(f"   收益率: {best['total_return']:+.2f}%")
    print(f"   回撤: {best['max_drawdown']:.2f}%")
    print(f"   夏普: {best['sharpe_ratio']:.2f}")
    
    print(f"\n[最差] {worst['name']}({worst['code']})")
    print(f"   收益率: {worst['total_return']:+.2f}%")
    print(f"   回撤: {worst['max_drawdown']:.2f}%")
    print(f"   夏普: {worst['sharpe_ratio']:.2f}")
    
    # 保存回测结果到CSV
    results_df = pd.DataFrame([
        {
            '股票代码': s['code'],
            '股票名称': s['name'],
            '最终净值': s['final_net_worth'],
            '总收益率(%)': s['total_return'],
            '最大回撤(%)': s['max_drawdown'],
            '夏普比率': s['sharpe_ratio'],
            '交易次数': s['num_trades'],
            '胜率(%)': s['win_rate'],
            '风险事件': s['risk_events'],
        }
        for s in all_stats
    ])
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file = f'backtest_300007_model_results_{timestamp}.csv'
    results_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"\n[保存] 回测结果已保存到: {csv_file}")

else:
    print("\n[警告] 没有成功完成任何回测")

print("\n" + "="*70)
print("[完成] 回测完成！")
print("="*70)
print(f"[模型] 使用模型: {MODEL_PATH}")
print(f"[回测] 回测股票数: {len(all_stats)} 只")
print("\n[说明] 本回测使用汉威科技300007的模型对所有7只股票进行测试")









