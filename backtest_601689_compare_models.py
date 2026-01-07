"""
拓普集团（601689）模型对比回测脚本
比较不同模型在拓普集团上的表现
"""
import os
import numpy as np
from stable_baselines3 import PPO
from stock_env_v6 import StockTradingEnv

# 测试数据
test_file = 'stockdata_v7_601689/test/sh.601689.拓普集团.csv'

# 初始资金（与训练时一致）
INITIAL_BALANCE = 20000  # 2万元

# 要对比的模型列表
models_to_test = [
    {
        'name': '拓普集团自身模型',
        'path': 'ppo_stock_v7_601689.zip',
        'description': '拓普集团601689专用模型'
    },
    {
        'name': '金力永磁模型（当前使用）',
        'path': 'ppo_stock_v7_300748.zip',
        'description': '金力永磁300748模型'
    },
    {
        'name': '三花智控模型',
        'path': 'ppo_stock_v7_002050.zip',
        'description': '三花智控002050模型（如果存在）'
    },
]

print("="*70)
print("拓普集团（601689）模型对比回测")
print("="*70)

# 检查测试数据
if not os.path.exists(test_file):
    print(f"[错误] 测试数据文件不存在: {test_file}")
    print("请先运行: python get_stock_data_v11_601689.py")
    exit(1)

print(f"\n[测试数据] {test_file}")
print(f"[初始资金] {INITIAL_BALANCE:,} 元\n")

results = []

for model_info in models_to_test:
    model_path = model_info['path']
    model_name = model_info['name']
    
    print(f"[测试] {model_name}")
    print(f"  模型文件: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"  ⚠️  模型文件不存在，跳过\n")
        continue
    
    try:
        # 加载模型
        model = PPO.load(model_path)
        print(f"  ✅ 模型加载成功")
        
        # 运行回测
        env = StockTradingEnv(test_file, initial_balance=INITIAL_BALANCE)
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
        
        stats = env.get_stats()
        
        result = {
            'model_name': model_name,
            'model_path': model_path,
            'final_net_worth': stats['final_net_worth'],
            'total_return': stats['total_return'],
            'max_drawdown': stats['max_drawdown'],
            'sharpe_ratio': stats['sharpe_ratio'],
            'num_trades': stats['num_trades'],
            'win_rate': stats['win_rate'],
            'risk_events': stats['risk_events']
        }
        
        results.append(result)
        
        print(f"  ✅ 回测完成")
        print(f"     最终净值: {stats['final_net_worth']:,.2f} 元")
        print(f"     总收益率: {stats['total_return']:+.2f}%")
        print(f"     最大回撤: {stats['max_drawdown']:.2f}%")
        print(f"     夏普比率: {stats['sharpe_ratio']:.2f}")
        print(f"     交易次数: {stats['num_trades']}")
        print(f"     胜率: {stats['win_rate']:.2f}%")
        print()
        
    except Exception as e:
        print(f"  ❌ 回测失败: {e}\n")
        import traceback
        traceback.print_exc()

# 对比分析
if len(results) > 0:
    print("="*70)
    print("### 拓普集团（601689）模型对比回测结果")
    print("="*70)
    print()
    print("| 模型 | 夏普比率 | 总收益率 | 最大回撤 | 交易次数 | 胜率 |")
    print("|------|---------|---------|---------|---------|------|")
    
    for result in results:
        print(f"| **{result['model_name']}** | **{result['sharpe_ratio']:.2f}** | "
              f"**{result['total_return']:+.2f}%** | **{result['max_drawdown']:.2f}%** | "
              f"{result['num_trades']} | {result['win_rate']:.2f}% |")
    
    print()
    print("### 对比分析")
    print()
    
    # 找出最佳模型
    best_sharpe = max(results, key=lambda x: x['sharpe_ratio'])
    best_return = max(results, key=lambda x: x['total_return'])
    best_drawdown = min(results, key=lambda x: x['max_drawdown'])
    
    print(f"1. **最佳夏普比率**: {best_sharpe['model_name']}")
    print(f"   - 夏普比率: {best_sharpe['sharpe_ratio']:.2f}")
    print(f"   - 总收益率: {best_sharpe['total_return']:+.2f}%")
    print(f"   - 最大回撤: {best_sharpe['max_drawdown']:.2f}%")
    print()
    
    print(f"2. **最高收益率**: {best_return['model_name']}")
    print(f"   - 总收益率: {best_return['total_return']:+.2f}%")
    print(f"   - 夏普比率: {best_return['sharpe_ratio']:.2f}")
    print(f"   - 最大回撤: {best_return['max_drawdown']:.2f}%")
    print()
    
    print(f"3. **最小回撤**: {best_drawdown['model_name']}")
    print(f"   - 最大回撤: {best_drawdown['max_drawdown']:.2f}%")
    print(f"   - 总收益率: {best_drawdown['total_return']:+.2f}%")
    print(f"   - 夏普比率: {best_drawdown['sharpe_ratio']:.2f}")
    print()
    
    # 推荐
    print("### 推荐")
    print()
    if best_sharpe['sharpe_ratio'] > 1.5:
        print(f"**推荐使用 {best_sharpe['model_name']}**:")
        print(f"- 夏普比率最高（{best_sharpe['sharpe_ratio']:.2f}），风险调整后收益更优")
        print(f"- 总收益率: {best_sharpe['total_return']:+.2f}%")
        print(f"- 最大回撤: {best_sharpe['max_drawdown']:.2f}%")
    else:
        print("**建议**:")
        print("- 所有模型的夏普比率都较低，建议重新训练或优化模型")
        print("- 如果必须选择，建议使用夏普比率最高的模型")
    
    print()
    print("="*70)
else:
    print("\n[错误] 没有成功完成任何回测")

