# train_v11_1A0001.py - V11上证指数1A0001专用训练
# -*- coding: utf-8 -*-
"""
V11 上证指数1A0001专用版特点：
1. 专门针对上证指数1A0001（sh.000001）进行训练优化
2. 初始资金10万元（指数交易通常资金量较大）
3. 优先使用上证指数1A0001进行训练和评估
4. 包含相关指数确保更好的泛化能力
5. 训练后的模型可用于V11全功能集成版实时预测
"""
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stock_env_v6 import StockTradingEnv  # 复用V6环境（V11兼容）
import random
import os
import numpy as np
import pandas as pd

# 扫描V7_1A0001训练数据（V11使用V7的数据格式）
train_dir = 'stockdata_v7_1A0001/train'
test_dir = 'stockdata_v7_1A0001/test'

if not os.path.exists(train_dir):
    print(f"[错误] 训练数据目录不存在: {train_dir}")
    print("请先运行: python get_stock_data_v11_1A0001.py")
    exit(1)

stock_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.csv')]
test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]

stock_files = sorted([f for f in stock_files if os.path.exists(f)])
test_files = sorted([f for f in test_files if os.path.exists(f)])

# 优先找到上证指数1A0001的文件
shangzheng_index_file = None
for f in stock_files:
    if '000001' in f or '上证指数' in f or '1A0001' in f:
        shangzheng_index_file = f
        break

print("="*70)
print("V11 上证指数1A0001专用版 - 训练启动")
print("="*70)
print(f"找到 {len(stock_files)} 只训练标的")
print(f"找到 {len(test_files)} 只测试标的")

if shangzheng_index_file:
    print(f"✅ 核心标的: 上证指数1A0001 - {shangzheng_index_file}")
else:
    print(f"⚠️  警告: 未找到上证指数1A0001的训练数据！")

if len(stock_files) == 0:
    print("[错误] 没有找到训练数据！")
    print("请先运行: python get_stock_data_v11_1A0001.py")
    exit(1)

# 列出所有找到的文件
print("\n训练数据文件:")
for i, f in enumerate(stock_files, 1):
    name = os.path.basename(f)
    is_core = '000001' in f or '上证指数' in f or '1A0001' in f
    mark = "🎯 [核心]" if is_core else "  "
    print(f"  {mark} {i}. {name}")

# V11_1A0001特殊配置：初始资金10万元（指数交易通常资金量较大）
INITIAL_BALANCE_V11_1A0001 = 100000  # 10万初始资金

def make_env():
    """随机选择标的创建环境，优先使用上证指数1A0001"""
    # 40%概率使用上证指数1A0001，60%概率使用其他指数
    if shangzheng_index_file and random.random() < 0.4:
        selected_file = shangzheng_index_file
        # 指数使用最小交易单位1（指数点），而不是100股
        env = StockTradingEnv(selected_file, initial_balance=INITIAL_BALANCE_V11_1A0001, min_trade_unit=1)
    else:
        selected_file = random.choice(stock_files)
        # 股票使用默认最小交易单位100股
        env = StockTradingEnv(selected_file, initial_balance=INITIAL_BALANCE_V11_1A0001)
    return env

def make_eval_env():
    """评估环境（优先使用上证指数1A0001）"""
    if shangzheng_index_file:
        # 指数使用最小交易单位1（指数点），而不是100股
        return StockTradingEnv(shangzheng_index_file, initial_balance=INITIAL_BALANCE_V11_1A0001, min_trade_unit=1)
    else:
        return StockTradingEnv(stock_files[0], initial_balance=INITIAL_BALANCE_V11_1A0001)

print("\n" + "="*70)
print("开始训练【V11 上证指数1A0001专用版】")
print("="*70)
print("核心特点：")
print("  [核心] 专门针对上证指数1A0001优化")
print("  [配置] 初始资金: 10万元（指数交易通常资金量较大）")
print("  [配置] 包含上证指数1A0001及相关指数")
print("  [策略] 训练时40%概率使用上证指数1A0001")
print("  [策略] 评估时优先使用上证指数1A0001")
print("  [兼容] 训练后的模型可用于V11全功能集成版")
print("  [保留] V6差异化风险策略")
print("  [保留] V5风险感知机制")
print("="*70 + "\n")

# 创建训练环境
train_env = DummyVecEnv([make_env for _ in range(16)])
eval_env = DummyVecEnv([make_eval_env])

# 回调
os.makedirs('./models_v7_1A0001/', exist_ok=True)
os.makedirs('./logs_v7_1A0001/eval/', exist_ok=True)

checkpoint_callback = CheckpointCallback(
    save_freq=100000 // 16,
    save_path='./models_v7_1A0001/',
    name_prefix='ppo_stock_v7_1A0001'
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./models_v7_1A0001/best/',
    log_path='./logs_v7_1A0001/eval/',
    eval_freq=50000 // 16,
    deterministic=True,
    render=False
)

# PPO模型（针对上证指数1A0001优化，兼容V11）
model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    n_steps=2048,
    batch_size=256,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.02,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log="./logs_v7_1A0001/"
)

print("开始训练 2,500,000 步...")
print("💡 提示: 训练过程中会优先使用上证指数1A0001数据")
print("💡 提示: 训练后的模型可用于V11全功能集成版实时预测")
print("💡 提示: 训练时间较长，请耐心等待...")
print()

model.learn(
    total_timesteps=2_500_000,
    callback=[checkpoint_callback, eval_callback],
    progress_bar=True
)

model.save("ppo_stock_v7_1A0001.zip")
print("\n[成功] 训练完成！模型已保存：ppo_stock_v7_1A0001.zip")
print("[提示] 可以在V11实时预测脚本中使用此模型")

# 回测评估（优先评估上证指数1A0001）
print("\n" + "="*70)
print("开始分类回测...")
print("="*70 + "\n")

all_stats = []
category_stats = {}
shangzheng_index_stats = None

# 优先测试上证指数1A0001
test_files_sorted = []
if shangzheng_index_file:
    # 找到对应的测试文件
    shangzheng_index_test = None
    for test_file in test_files:
        if '000001' in test_file or '上证指数' in test_file or '1A0001' in test_file:
            shangzheng_index_test = test_file
            test_files_sorted.append(test_file)
            break
    
    # 其他文件
    for test_file in test_files:
        if test_file != shangzheng_index_test:
            test_files_sorted.append(test_file)
else:
    test_files_sorted = test_files

for test_file in test_files_sorted:
    if not os.path.exists(test_file):
        print(f"[警告] 文件不存在: {test_file}")
        continue
    
    try:
        # 检查是否是指数文件，指数使用最小交易单位1（指数点）
        is_index = ('000001' in test_file or '上证指数' in test_file or '1A0001' in test_file or 
                   '000016' in test_file or '000300' in test_file or '399001' in test_file or '399006' in test_file)
        min_trade_unit = 1 if is_index else 100  # 指数用1，股票用100
        
        env = StockTradingEnv(test_file, initial_balance=INITIAL_BALANCE_V11_1A0001, min_trade_unit=min_trade_unit)
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
        
        stats = env.get_stats()
        stats['file'] = test_file
        stats['name'] = env.stock_info.get('name', '未知')
        
        # 检查是否是上证指数1A0001
        is_shangzheng_index = ('000001' in test_file or '上证指数' in test_file or '1A0001' in test_file)
        if is_shangzheng_index:
            stats['is_core'] = True
            shangzheng_index_stats = stats
            print("="*70)
            print("🎯 [核心标的] 上证指数1A0001回测结果")
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
        core_mark = "🎯 [核心]" if is_shangzheng_index else ""
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
        print(f"[错误] {test_file} 测试失败: {e}\n")
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
    
    print(f"平均收益率: {avg_return:+.2f}%")
    print(f"平均最大回撤: {avg_drawdown:.2f}%")
    print(f"平均夏普比率: {avg_sharpe:.2f}")
    print(f"平均胜率: {avg_win_rate:.2f}%")
    
    # 核心标的统计
    if shangzheng_index_stats:
        print("\n" + "="*70)
        print("🎯 [核心标的] 上证指数1A0001详细统计")
        print("="*70)
        print(f"最终净值: {shangzheng_index_stats['final_net_worth']:,.2f} 元")
        print(f"总收益率: {shangzheng_index_stats['total_return']:+.2f}%")
        print(f"最大回撤: {shangzheng_index_stats['max_drawdown']:.2f}%")
        print(f"夏普比率: {shangzheng_index_stats['sharpe_ratio']:.2f}")
        print(f"交易次数: {shangzheng_index_stats['num_trades']}")
        print(f"胜率: {shangzheng_index_stats['win_rate']:.2f}%")
        print(f"风险事件: {shangzheng_index_stats['risk_events']} 次")
    
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
print("训练和回测完成！")
print("="*70)
print("模型文件: ppo_stock_v7_1A0001.zip")
print("最佳模型: ./models_v7_1A0001/best/best_model.zip")
print("日志目录: ./logs_v7_1A0001/")
print("\n💡 提示: 现在可以使用 real_time_predict_v11_1A0001.py 进行实时预测")

