"""
V11 航天工程603698专用回测脚本
直接使用已有模型进行回测，不重新训练
"""
import os
import numpy as np
from stable_baselines3 import PPO
from stock_env_v6 import StockTradingEnv

# 模型路径
MODEL_PATH = "ppo_stock_v7_603698.zip"

# 测试数据目录
test_dir = 'stockdata_v7_603698/test'

# 初始资金（与训练时一致）
INITIAL_BALANCE = 50000  # 5万元

print("="*70)
print("V11 航天工程603698专用回测脚本")
print("="*70)

# 检查模型文件
if not os.path.exists(MODEL_PATH):
    print(f"[错误] 模型文件不存在: {MODEL_PATH}")
    print("请先运行训练脚本生成模型")
    exit(1)

# 检查测试数据目录
if not os.path.exists(test_dir):
    print(f"[错误] 测试数据目录不存在: {test_dir}")
    print("请先运行: python get_stock_data_v11_603698.py")
    exit(1)

# 加载模型
print(f"\n[加载模型] {MODEL_PATH}")
try:
    model = PPO.load(MODEL_PATH)
    print("[成功] 模型加载成功")
except Exception as e:
    print(f"[错误] 模型加载失败: {e}")
    exit(1)

# 获取测试文件
test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]
test_files = sorted([f for f in test_files if os.path.exists(f)])

if len(test_files) == 0:
    print(f"[错误] 测试数据目录中没有找到CSV文件: {test_dir}")
    exit(1)

print(f"\n[找到测试文件] {len(test_files)} 个")

# 优先找到航天工程603698的文件
hangtian_gongcheng_test = None
for test_file in test_files:
    if '603698' in test_file or '航天工程' in test_file:
        hangtian_gongcheng_test = test_file
        break

# 排序：航天工程优先
test_files_sorted = []
if hangtian_gongcheng_test:
    test_files_sorted.append(hangtian_gongcheng_test)
    for test_file in test_files:
        if test_file != hangtian_gongcheng_test:
            test_files_sorted.append(test_file)
else:
    test_files_sorted = test_files

print("\n" + "="*70)
print("开始分类回测...")
print("="*70 + "\n")

all_stats = []
category_stats = {}
hangtian_gongcheng_stats = None

for test_file in test_files_sorted:
    if not os.path.exists(test_file):
        print(f"[警告] 文件不存在: {test_file}")
        continue
    
    try:
        print(f"[回测] {os.path.basename(test_file)}")
        
        env = StockTradingEnv(test_file, initial_balance=INITIAL_BALANCE)
        obs, _ = env.reset()
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
        
        stats = env.get_stats()
        stats['file'] = test_file
        stats['name'] = env.stock_info.get('name', '未知')
        
        # 检查是否是航天工程603698
        is_hangtian_gongcheng = ('603698' in test_file or '航天工程' in test_file)
        if is_hangtian_gongcheng:
            stats['is_core'] = True
            hangtian_gongcheng_stats = stats
            print("="*70)
            print("🎯 [核心标的] 航天工程603698回测结果")
            print("="*70)
        else:
            stats['is_core'] = False
        
        all_stats.append(stats)
        
        # 按分类统计
        category = stats.get('category', '未知')
        if category not in category_stats:
            category_stats[category] = []
        category_stats[category].append(stats)
        
        name = os.path.basename(test_file).replace('.csv', '')
        core_mark = "🎯 [核心]" if is_hangtian_gongcheng else ""
        print(f"{core_mark}[{category}|{stats.get('volatility', '未知')}波动] {stats['name']}")
        print(f"   最终净值: {stats['final_net_worth']:,.2f} 元")
        print(f"   总收益率: {stats['total_return']:+.2f}%")
        print(f"   最大回撤: {stats['max_drawdown']:.2f}%")
        print(f"   夏普比率: {stats['sharpe_ratio']:.2f}")
        print(f"   交易次数: {stats['num_trades']}")
        print(f"   胜率: {stats['win_rate']:.2f}%")
        print(f"   风险事件: {stats['risk_events']} 次")
        print()
        
    except Exception as e:
        print(f"[错误] {test_file} 回测失败: {e}")
        import traceback
        traceback.print_exc()
        print()

# 整体统计
if len(all_stats) > 0:
    print("="*70)
    print("[整体统计]")
    print("="*70)
    
    avg_return = np.mean([s['total_return'] for s in all_stats])
    avg_drawdown = np.mean([s['max_drawdown'] for s in all_stats])
    avg_sharpe = np.mean([s['sharpe_ratio'] for s in all_stats])
    avg_win_rate = np.mean([s['win_rate'] for s in all_stats])
    
    print(f"平均收益率: {avg_return:+.2f}%")
    print(f"平均最大回撤: {avg_drawdown:.2f}%")
    print(f"平均夏普比率: {avg_sharpe:.2f}")
    print(f"平均胜率: {avg_win_rate:.2f}%")
    
    # 核心标的统计
    if hangtian_gongcheng_stats:
        print("\n" + "="*70)
        print("🎯 [核心标的] 航天工程603698详细统计")
        print("="*70)
        print(f"最终净值: {hangtian_gongcheng_stats['final_net_worth']:,.2f} 元")
        print(f"总收益率: {hangtian_gongcheng_stats['total_return']:+.2f}%")
        print(f"最大回撤: {hangtian_gongcheng_stats['max_drawdown']:.2f}%")
        print(f"夏普比率: {hangtian_gongcheng_stats['sharpe_ratio']:.2f}")
        print(f"交易次数: {hangtian_gongcheng_stats['num_trades']}")
        print(f"胜率: {hangtian_gongcheng_stats['win_rate']:.2f}%")
        print(f"风险事件: {hangtian_gongcheng_stats['risk_events']} 次")
    
    # 按分类统计
    if len(category_stats) > 0:
        print("\n" + "="*70)
        print("[按分类统计]")
        print("="*70)
        for category, stats_list in category_stats.items():
            cat_returns = [s['total_return'] for s in stats_list]
            cat_drawdowns = [s['max_drawdown'] for s in stats_list]
            print(f"\n{category} ({len(stats_list)}只):")
            print(f"  平均收益率: {np.mean(cat_returns):+.2f}%")
            print(f"  平均最大回撤: {np.mean(cat_drawdowns):.2f}%")

print("\n" + "="*70)
print("回测完成！")
print("="*70)
print(f"模型文件: {MODEL_PATH}")
print(f"测试文件数: {len(all_stats)}")
print("\n💡 提示: 回测结果已显示，可以用于评估模型性能")
