# train_v11_002241.py - V11歌尔股份002241专用训练
# -*- coding: utf-8 -*-
"""
V11 歌尔股份002241专用版特点：
1. 专门针对歌尔股份002241进行训练优化
2. 初始资金5万元（匹配实盘操作）
3. 优先使用歌尔股份002241进行训练和评估
4. 包含相关股票确保更好的泛化能力
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

# 扫描V7_002241训练数据（V11使用V7的数据格式）
train_dir = 'stockdata_v7_002241/train'
test_dir = 'stockdata_v7_002241/test'

if not os.path.exists(train_dir):
    print(f"[错误] 训练数据目录不存在: {train_dir}")
    print("请先运行: python get_stock_data_v11_002241.py")
    exit(1)

stock_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.csv')]
test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]

stock_files = sorted([f for f in stock_files if os.path.exists(f)])
test_files = sorted([f for f in test_files if os.path.exists(f)])

# 优先找到歌尔股份002241的文件
geer_file = None
for f in stock_files:
    if '002241' in f or '歌尔股份' in f:
        geer_file = f
        break

print("="*70)
print("V11 歌尔股份002241专用版 - 训练启动")
print("="*70)
print(f"找到 {len(stock_files)} 只训练标的")
print(f"找到 {len(test_files)} 只测试标的")

if geer_file:
    print(f"✅ 核心标的: 歌尔股份002241 - {geer_file}")
else:
    print(f"⚠️  警告: 未找到歌尔股份002241的训练数据！")

if len(stock_files) == 0:
    print("[错误] 没有找到训练数据！")
    print("请先运行: python get_stock_data_v11_002241.py")
    exit(1)

# 加载元数据
metadata_file = 'stockdata_v7_002241/metadata_v7_002241.csv'
if os.path.exists(metadata_file):
    metadata = pd.read_csv(metadata_file)
    print(f"\n[元数据] 已加载")
    print(metadata[['name', 'category', 'volatility', 'style', 'priority']].to_string(index=False))
    
    # 检查歌尔股份002241
    geer_meta = metadata[metadata['code'] == 'sz.002241']
    if len(geer_meta) > 0:
        print(f"\n[核心] 歌尔股份002241元数据:")
        print(geer_meta[['name', 'category', 'volatility', 'style']].to_string(index=False))
else:
    print(f"\n[警告] 元数据文件不存在: {metadata_file}")

# V11_002241特殊配置：初始资金5万元（匹配实盘）
INITIAL_BALANCE_V11_002241 = 50000  # 5万初始资金，匹配实盘操作

def make_env():
    """随机选择标的创建环境，优先使用歌尔股份002241"""
    # 30%概率使用歌尔股份002241，70%概率使用其他股票
    if geer_file and random.random() < 0.3:
        selected_file = geer_file
    else:
        selected_file = random.choice(stock_files)
    env = StockTradingEnv(selected_file, initial_balance=INITIAL_BALANCE_V11_002241)
    return env

def make_eval_env():
    """评估环境（优先使用歌尔股份002241）"""
    if geer_file:
        return StockTradingEnv(geer_file, initial_balance=INITIAL_BALANCE_V11_002241)
    else:
        return StockTradingEnv(stock_files[0], initial_balance=INITIAL_BALANCE_V11_002241)

print("\n" + "="*70)
print("开始训练【V11 歌尔股份002241专用版】")
print("="*70)
print("核心特点：")
print("  [核心] 专门针对歌尔股份002241优化")
print("  [配置] 初始资金: 5万元（匹配实盘操作）")
print("  [配置] 包含歌尔股份002241及相关股票")
print("  [策略] 训练时30%概率使用歌尔股份002241")
print("  [策略] 评估时优先使用歌尔股份002241")
print("  [兼容] 训练后的模型可用于V11全功能集成版")
print("  [保留] V6差异化风险策略")
print("  [保留] V5风险感知机制")
print("="*70 + "\n")

# 创建训练环境
train_env = DummyVecEnv([make_env for _ in range(16)])
eval_env = DummyVecEnv([make_eval_env])

# 回调
os.makedirs('./models_v7_002241/', exist_ok=True)
os.makedirs('./logs_v7_002241/eval/', exist_ok=True)

checkpoint_callback = CheckpointCallback(
    save_freq=100000 // 16,
    save_path='./models_v7_002241/',
    name_prefix='ppo_stock_v7_002241'
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./models_v7_002241/best/',
    log_path='./logs_v7_002241/eval/',
    eval_freq=50000 // 16,
    deterministic=True,
    render=False
)

# PPO模型（针对歌尔股份002241优化，兼容V11）
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
    tensorboard_log="./logs_v7_002241/"
)

print("开始训练 2,500,000 步...")
print("💡 提示: 训练过程中会优先使用歌尔股份002241数据")
print("💡 提示: 训练后的模型可用于V11全功能集成版实时预测")
model.learn(
    total_timesteps=2_500_000,
    callback=[checkpoint_callback, eval_callback],
    progress_bar=True
)

model.save("ppo_stock_v7_002241.zip")
print("\n[成功] 训练完成！模型已保存：ppo_stock_v7_002241.zip")
print("[提示] 可以在V11实时预测脚本中使用此模型")

# 回测评估（优先评估歌尔股份002241）
print("\n" + "="*70)
print("开始分类回测...")
print("="*70 + "\n")

all_stats = []
category_stats = {}
geer_stats = None

# 优先测试歌尔股份002241
test_files_sorted = []
if geer_file:
    # 找到对应的测试文件
    geer_test = None
    for test_file in test_files:
        if '002241' in test_file or '歌尔股份' in test_file:
            geer_test = test_file
            test_files_sorted.append(test_file)
            break
    
    # 其他文件
    for test_file in test_files:
        if test_file != geer_test:
            test_files_sorted.append(test_file)
else:
    test_files_sorted = test_files

for test_file in test_files_sorted:
    if not os.path.exists(test_file):
        print(f"[警告] 文件不存在: {test_file}")
        continue
    
    try:
        env = StockTradingEnv(test_file, initial_balance=INITIAL_BALANCE_V11_002241)
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
        
        stats = env.get_stats()
        stats['file'] = test_file
        stats['name'] = env.stock_info.get('name', '未知')
        
        # 检查是否是歌尔股份002241
        is_geer = ('002241' in test_file or '歌尔股份' in test_file)
        if is_geer:
            stats['is_core'] = True
            geer_stats = stats
            print("="*70)
            print("🎯 [核心标的] 歌尔股份002241回测结果")
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
        core_mark = "🎯 [核心]" if is_geer else ""
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
    total_trades = sum([s['num_trades'] for s in all_stats])
    total_risk_events = sum([s['risk_events'] for s in all_stats])
    
    print(f"平均收益率: {avg_return:+.2f}%")
    print(f"平均最大回撤: {avg_drawdown:.2f}%")
    print(f"平均夏普比率: {avg_sharpe:.2f}")
    print(f"平均胜率: {avg_win_rate:.2f}%")
    print(f"总交易次数: {total_trades}")
    print(f"总风险事件: {total_risk_events} 次")
    print(f"测试标的数: {len(all_stats)}")
    
    # 歌尔股份002241专项统计
    if geer_stats:
        print("\n" + "="*70)
        print("🎯 [核心标的专项统计] 歌尔股份002241")
        print("="*70)
        print(f"最终净值: {geer_stats['final_net_worth']:,.2f} 元")
        print(f"总收益率: {geer_stats['total_return']:+.2f}%")
        print(f"最大回撤: {geer_stats['max_drawdown']:.2f}%")
        print(f"夏普比率: {geer_stats['sharpe_ratio']:.2f}")
        print(f"交易次数: {geer_stats['num_trades']}")
        print(f"胜率: {geer_stats['win_rate']:.2f}%")
        print(f"风险事件: {geer_stats['risk_events']} 次")
    
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
    
    print(f"\n[最佳] {best['name']} ({best.get('category', '未知')})")
    print(f"   收益率: {best['total_return']:+.2f}%")
    print(f"   回撤: {best['max_drawdown']:.2f}%")
    print(f"   夏普: {best['sharpe_ratio']:.2f}")
    
    print(f"\n[最差] {worst['name']} ({worst.get('category', '未知')})")
    print(f"   收益率: {worst['total_return']:+.2f}%")
    print(f"   回撤: {worst['max_drawdown']:.2f}%")
    print(f"   夏普: {worst['sharpe_ratio']:.2f}")

print("\n" + "="*70)
print("[完成] 所有测试完成！")
print("="*70)
print(f"[保存] 模型: ppo_stock_v7_002241.zip")
print(f"[日志] 训练日志: ./logs_v7_002241/")
print(f"[模型] 检查点: ./models_v7_002241/")
print(f"\n[提示] 使用 tensorboard --logdir=./logs_v7_002241/ 查看训练曲线")
print("\n[V11_002241特色] 专门针对歌尔股份002241优化，初始资金5万，匹配实盘操作！")
print("\n[使用] 训练完成后，可以使用以下命令进行实时预测：")
print("  python real_time_predict_v11_002241.py")
print("  python real_time_predict_v12_002241.py")
print("\n[说明] V11全功能集成版支持多模型融合决策，包括：")
print("  - PPO强化学习模型（本训练脚本生成）")
print("  - LSTM/GRU时间序列预测")
print("  - Transformer模型")
print("  - 全息动态模型")
print("  - 智能融合决策系统")

