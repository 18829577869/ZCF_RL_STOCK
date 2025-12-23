# train_v11_118013.py - V11道通转债118013专用训练
# -*- coding: utf-8 -*-
"""
V11 道通转债118013专用版特点：
1. 专门针对道通转债118013进行训练优化
2. 初始资金2万元（匹配实盘操作）
3. 优先使用道通转债118013进行训练和评估
4. 包含相关可转债和正股确保更好的泛化能力
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

# 扫描V7_118013训练数据（V11使用V7的数据格式）
train_dir = 'stockdata_v7_118013/train'
test_dir = 'stockdata_v7_118013/test'

if not os.path.exists(train_dir):
    print(f"[错误] 训练数据目录不存在: {train_dir}")
    print("请先运行: python get_stock_data_v11_118013.py")
    exit(1)

stock_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.csv')]
test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]

# 过滤数据不足的文件（模拟环境创建时的数据处理过程）
def is_valid_data_file(file_path):
    """检查文件是否有足够的数据（模拟stock_env_v6的数据处理）"""
    try:
        df = pd.read_csv(file_path)
        
        # 检查是否有必要的列
        required_cols = ['date', 'open', 'high', 'low', 'close', 'preclose', 'volume', 'amount',
                        'turn', 'pctChg', 'peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ']
        if not all(col in df.columns for col in required_cols):
            return False
        
        # 模拟环境的数据处理过程
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values('date').reset_index(drop=True)
        
        # 转换数据类型（模拟环境处理）
        base_columns = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount',
                        'turn', 'pctChg', 'peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ']
        df[base_columns] = df[base_columns].apply(pd.to_numeric, errors='coerce')
        
        # 模拟计算技术指标（简化版，只检查数据是否足够）
        # 实际环境会计算MA5、MA20等，需要更多数据
        
        # 模拟dropna（这是关键，会减少数据量）
        df = df.dropna().reset_index(drop=True)
        
        # stock_env_v6要求：len(self.df) >= history_window + 50
        # history_window默认是5，所以至少需要55条数据
        # 但考虑到技术指标计算（MA20需要20条），我们要求至少100条数据更安全
        min_required = 100  # 保守估计，确保有足够数据
        
        if len(df) < min_required:
            print(f"  [跳过] {os.path.basename(file_path)}: 数据不足（处理后仅{len(df)}条，需要至少{min_required}条）")
            return False
        
        return True
    except Exception as e:
        print(f"  [跳过] {os.path.basename(file_path)}: 验证失败 - {e}")
        return False

# 过滤有效文件
stock_files = sorted([f for f in stock_files if os.path.exists(f) and is_valid_data_file(f)])
test_files = sorted([f for f in test_files if os.path.exists(f) and is_valid_data_file(f)])

# 优先找到道通转债118013的文件（精确匹配，避免匹配到118001等）
daotong_convertible_file = None
for f in stock_files:
    # 精确匹配118013，使用文件名精确匹配
    filename = os.path.basename(f)
    # 优先匹配包含"sh.118013"或"118013"但不包含"118001"的文件
    if ('sh.118013' in filename or ('118013' in filename and '118001' not in filename)):
        daotong_convertible_file = f
        break
    # 如果文件名包含"道通转债"
    if '道通转债' in filename:
        daotong_convertible_file = f
        break

print("="*70)
print("V11 道通转债118013专用版 - 训练启动")
print("="*70)
print(f"找到 {len(stock_files)} 只训练标的")
print(f"找到 {len(test_files)} 只测试标的")

if daotong_convertible_file:
    print(f"✅ 核心标的: 道通转债118013 - {daotong_convertible_file}")
else:
    print(f"⚠️  警告: 未找到道通转债118013的训练数据！")

if len(stock_files) == 0:
    print("[错误] 没有找到训练数据！")
    print("请先运行: python get_stock_data_v11_118013.py")
    exit(1)

# 加载元数据
metadata_file = 'stockdata_v7_118013/metadata_v7_118013.csv'
if os.path.exists(metadata_file):
    metadata = pd.read_csv(metadata_file)
    print(f"\n[元数据] 已加载")
    print(metadata[['name', 'category', 'volatility', 'style', 'priority']].to_string(index=False))
    
    # 检查道通转债118013
    daotong_convertible_meta = metadata[metadata['code'] == 'sh.118013']
    if len(daotong_convertible_meta) > 0:
        print(f"\n[核心] 道通转债118013元数据:")
        print(daotong_convertible_meta[['name', 'category', 'volatility', 'style']].to_string(index=False))
else:
    print(f"\n[警告] 元数据文件不存在: {metadata_file}")

# V11_118013特殊配置：初始资金5万元（匹配实盘）
INITIAL_BALANCE_V11_118013 = 50000  # 5万初始资金，匹配实盘操作

def make_env():
    """随机选择标的创建环境，优先使用道通转债118013"""
    max_retries = 10  # 最多重试10次
    for _ in range(max_retries):
        try:
            # 30%概率使用道通转债118013，70%概率使用其他股票
            if daotong_convertible_file and random.random() < 0.3:
                selected_file = daotong_convertible_file
            else:
                selected_file = random.choice(stock_files)
            env = StockTradingEnv(selected_file, initial_balance=INITIAL_BALANCE_V11_118013)
            return env
        except ValueError as e:
            # 如果数据不足，尝试其他文件
            if "数据不足" in str(e):
                continue
            raise
        except Exception as e:
            # 其他错误也重试
            continue
    # 如果所有重试都失败，使用第一个有效文件
    if daotong_convertible_file:
        return StockTradingEnv(daotong_convertible_file, initial_balance=INITIAL_BALANCE_V11_118013)
    return StockTradingEnv(stock_files[0], initial_balance=INITIAL_BALANCE_V11_118013)

def make_eval_env():
    """评估环境（优先使用道通转债118013）"""
    if daotong_convertible_file:
        return StockTradingEnv(daotong_convertible_file, initial_balance=INITIAL_BALANCE_V11_118013)
    else:
        return StockTradingEnv(stock_files[0], initial_balance=INITIAL_BALANCE_V11_118013)

print("\n" + "="*70)
print("开始训练【V11 道通转债118013专用版】")
print("="*70)
print("核心特点：")
print("  [核心] 专门针对道通转债118013优化")
print("  [配置] 初始资金: 5万元（匹配实盘操作）")
print("  [配置] 包含道通转债118013及相关可转债和正股")
print("  [策略] 训练时30%概率使用道通转债118013")
print("  [策略] 评估时优先使用道通转债118013")
print("  [兼容] 训练后的模型可用于V11全功能集成版")
print("  [保留] V6差异化风险策略")
print("  [保留] V5风险感知机制")
print("="*70 + "\n")

# 创建训练环境
train_env = DummyVecEnv([make_env for _ in range(16)])
eval_env = DummyVecEnv([make_eval_env])

# 回调
os.makedirs('./models_v7_118013/', exist_ok=True)
os.makedirs('./logs_v7_118013/eval/', exist_ok=True)

checkpoint_callback = CheckpointCallback(
    save_freq=100000 // 16,
    save_path='./models_v7_118013/',
    name_prefix='ppo_stock_v7_118013'
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./models_v7_118013/best/',
    log_path='./logs_v7_118013/eval/',
    eval_freq=50000 // 16,
    deterministic=True,
    render=False
)

# PPO模型（针对道通转债118013优化，兼容V11）
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
    tensorboard_log="./logs_v7_118013/"
)

print("开始训练 2,500,000 步...")
print("💡 提示: 训练过程中会优先使用道通转债118013数据")
print("💡 提示: 训练后的模型可用于V11全功能集成版实时预测")
model.learn(
    total_timesteps=2_500_000,
    callback=[checkpoint_callback, eval_callback],
    progress_bar=True
)

model.save("ppo_stock_v7_118013.zip")
print("\n[成功] 训练完成！模型已保存：ppo_stock_v7_118013.zip")
print("[提示] 可以在V11实时预测脚本中使用此模型")

# 回测评估（优先评估道通转债118013）
print("\n" + "="*70)
print("开始分类回测...")
print("="*70 + "\n")

all_stats = []
category_stats = {}
daotong_convertible_stats = None

# 优先测试道通转债118013
test_files_sorted = []
if daotong_convertible_file:
    # 找到对应的测试文件
    daotong_convertible_test = None
    for test_file in test_files:
        if '118013' in test_file or '道通转债' in test_file:
            daotong_convertible_test = test_file
            test_files_sorted.append(test_file)
            break
    
    # 其他文件
    for test_file in test_files:
        if test_file != daotong_convertible_test:
            test_files_sorted.append(test_file)
else:
    test_files_sorted = test_files

for test_file in test_files_sorted:
    if not os.path.exists(test_file):
        print(f"[警告] 文件不存在: {test_file}")
        continue
    
    try:
        env = StockTradingEnv(test_file, initial_balance=INITIAL_BALANCE_V11_118013)
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
        
        stats = env.get_stats()
        stats['file'] = test_file
        stats['name'] = env.stock_info.get('name', '未知')
        
        # 检查是否是道通转债118013
        is_daotong_convertible = ('118013' in test_file or '道通转债' in test_file)
        if is_daotong_convertible:
            stats['is_core'] = True
            daotong_convertible_stats = stats
            print("="*70)
            print("🎯 [核心标的] 道通转债118013回测结果")
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
        core_mark = "🎯 [核心]" if is_daotong_convertible else ""
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
    
    # 道通转债118013专项统计
    if daotong_convertible_stats:
        print("\n" + "="*70)
        print("🎯 [核心标的专项统计] 道通转债118013")
        print("="*70)
        print(f"最终净值: {daotong_convertible_stats['final_net_worth']:,.2f} 元")
        print(f"总收益率: {daotong_convertible_stats['total_return']:+.2f}%")
        print(f"最大回撤: {daotong_convertible_stats['max_drawdown']:.2f}%")
        print(f"夏普比率: {daotong_convertible_stats['sharpe_ratio']:.2f}")
        print(f"交易次数: {daotong_convertible_stats['num_trades']}")
        print(f"胜率: {daotong_convertible_stats['win_rate']:.2f}%")
        print(f"风险事件: {daotong_convertible_stats['risk_events']} 次")
    
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
print(f"[保存] 模型: ppo_stock_v7_118013.zip")
print(f"[日志] 训练日志: ./logs_v7_118013/")
print(f"[模型] 检查点: ./models_v7_118013/")
print(f"\n[提示] 使用 tensorboard --logdir=./logs_v7_118013/ 查看训练曲线")
print("\n[V11_118013特色] 专门针对道通转债118013优化，初始资金5万，匹配实盘操作！")
print("\n[使用] 训练完成后，可以使用以下命令进行实时预测：")
print("  python real_time_predict_v11_118013.py")
print("\n[说明] V11全功能集成版支持多模型融合决策，包括：")
print("  - PPO强化学习模型（本训练脚本生成）")
print("  - LSTM/GRU时间序列预测")
print("  - Transformer模型")
print("  - 全息动态模型")
print("  - 智能融合决策系统")

