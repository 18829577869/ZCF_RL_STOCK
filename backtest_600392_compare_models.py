# backtest_600392_compare_models.py - 盛和资源600392对比英维克模型和金力永磁模型回测
# -*- coding: utf-8 -*-
"""
对比盛和资源600392使用英维克模型和金力永磁模型的表现
"""
from stable_baselines3 import PPO
from stock_env_v6 import StockTradingEnv
import os
import numpy as np

# 股票信息
STOCK_CODE = "sh.600392"
STOCK_NAME = "盛和资源"

# 测试数据文件（优先使用金力永磁模型目录的数据）
TEST_FILE_PATHS = [
    "stockdata_v7_300748/test/sh.600392.盛和资源.csv",  # 金力永磁模型目录
    "stockdata_v7_000831/test/sh.600392.盛和资源.csv",  # 中国稀土模型目录（备用）
    "stockdata_v7_002837/test/sh.600392.盛和资源.csv",  # 英维克模型目录（备用）
]

# 模型配置
MODELS = [
    {
        'name': '英维克002837模型',
        'path': 'ppo_stock_v7_002837.zip',
        'description': '英维克002837专用模型'
    },
    {
        'name': '金力永磁300748模型',
        'path': 'ppo_stock_v7_300748.zip',
        'description': '金力永磁300748专用模型（盛和资源优选）'
    }
]

# 初始资金
INITIAL_BALANCE = 20000  # 2万元（与训练时一致）

print("="*70)
print(f"盛和资源({STOCK_CODE})模型对比回测")
print("="*70)
print(f"股票名称: {STOCK_NAME}")
print(f"股票代码: {STOCK_CODE}")
print(f"初始资金: {INITIAL_BALANCE:,.0f} 元")
print("="*70)

# 查找测试数据文件
test_file = None
for test_path in TEST_FILE_PATHS:
    if os.path.exists(test_path):
        test_file = test_path
        print(f"\n[找到] 测试数据文件: {test_file}")
        break

if not test_file:
    print(f"\n[错误] 未找到测试数据文件，尝试的路径:")
    for path in TEST_FILE_PATHS:
        print(f"  - {path}")
    exit(1)

# 回测结果
results = []

# 对每个模型进行回测
for model_config in MODELS:
    model_path = model_config['path']
    model_name = model_config['name']
    
    print("\n" + "="*70)
    print(f"回测模型: {model_name}")
    print(f"模型文件: {model_path}")
    print("="*70)
    
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"[警告] 模型文件不存在: {model_path}")
        print(f"[跳过] 跳过此模型的回测")
        continue
    
    # 加载模型
    try:
        print(f"[加载] 正在加载模型...")
        model = PPO.load(model_path)
        print(f"[成功] 模型加载成功")
    except Exception as e:
        print(f"[错误] 模型加载失败: {e}")
        continue
    
    # 创建环境
    try:
        print(f"[创建] 正在创建回测环境...")
        env = StockTradingEnv(test_file, initial_balance=INITIAL_BALANCE)
        obs, _ = env.reset()
        print(f"[成功] 环境创建成功")
    except Exception as e:
        print(f"[错误] 环境创建失败: {e}")
        import traceback
        traceback.print_exc()
        continue
    
    # 开始回测
    print(f"[回测] 开始回测...")
    done = False
    step_count = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        
        # 确保action是整数
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)
        
        obs, reward, done, truncated, _ = env.step(action)
        step_count += 1
        
        if step_count % 50 == 0:
            print(f"  进度: {step_count} 步... (总资产: {env.net_worth:.2f})", end='\r')
    
    print(f"\n[完成] 回测完成，共 {step_count} 步")
    
    # 获取统计结果
    stats = env.get_stats()
    
    result = {
        '模型名称': model_name,
        '模型文件': model_path,
        '最终净值': stats['final_net_worth'],
        '总收益率(%)': stats['total_return'],
        '最大回撤(%)': stats['max_drawdown'],
        '夏普比率': stats['sharpe_ratio'],
        '交易次数': stats['num_trades'],
        '胜率(%)': stats['win_rate'],
        '风险事件': stats['risk_events'],
        '总天数': stats['total_days']
    }
    
    results.append(result)
    
    # 显示结果
    print("\n" + "-"*70)
    print(f"【{model_name}】回测结果:")
    print("-"*70)
    print(f"  最终净值: {stats['final_net_worth']:,.2f} 元")
    print(f"  总收益率: {stats['total_return']:+.2f}%")
    print(f"  最大回撤: {stats['max_drawdown']:.2f}%")
    print(f"  夏普比率: {stats['sharpe_ratio']:.2f}")
    print(f"  交易次数: {stats['num_trades']}")
    print(f"  胜率: {stats['win_rate']:.2f}%")
    print(f"  风险事件: {stats['risk_events']} 次")
    print(f"  总天数: {stats['total_days']} 天")

# 对比结果
if len(results) > 0:
    print("\n" + "="*70)
    print("模型对比总结")
    print("="*70)
    
    print(f"\n{'模型':<25} {'夏普比率':<12} {'总收益率':<12} {'最大回撤':<12}")
    print("-"*70)
    
    for result in results:
        print(f"{result['模型名称']:<25} {result['夏普比率']:<12.2f} {result['总收益率(%)']:>+11.2f}% {result['最大回撤(%)']:>11.2f}%")
    
    # 找出最佳模型
    if len(results) >= 2:
        best_sharpe = max(results, key=lambda x: x['夏普比率'])
        best_return = max(results, key=lambda x: x['总收益率(%)'])
        best_drawdown = min(results, key=lambda x: x['最大回撤(%)'])
        
        print("\n" + "-"*70)
        print("最佳表现:")
        print(f"  最高夏普比率: {best_sharpe['模型名称']} (夏普: {best_sharpe['夏普比率']:.2f})")
        print(f"  最高收益率: {best_return['模型名称']} (收益率: {best_return['总收益率(%)']:+.2f}%)")
        print(f"  最低回撤: {best_drawdown['模型名称']} (回撤: {best_drawdown['最大回撤(%)']:.2f}%)")
        
        # 综合推荐（优先考虑夏普比率）
        print("\n" + "-"*70)
        print("💡 推荐:")
        if best_sharpe['夏普比率'] > 0:
            print(f"   ✅ 推荐使用: {best_sharpe['模型名称']}")
            print(f"      理由: 夏普比率最高 ({best_sharpe['夏普比率']:.2f})，风险调整后收益最优")
            print(f"      收益率: {best_sharpe['总收益率(%)']:+.2f}%")
            print(f"      最大回撤: {best_sharpe['最大回撤(%)']:.2f}%")
        else:
            print(f"   ⚠️  所有模型表现不佳，建议重新训练或调整策略")
    
    print("\n" + "="*70)
    print("[完成] 对比回测完成！")
    print("="*70)
else:
    print("\n[错误] 没有成功完成任何模型的回测")

