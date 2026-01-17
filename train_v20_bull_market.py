"""
V20训练脚本 - 牛市牛股训练版
使用历史牛市牛股数据进行模型训练，提高模型在牛市中的表现

特点：
1. 使用历史牛市牛股数据（2007年、2015年、2019-2021年等）
2. 针对牛市市场环境优化模型参数
3. 提高模型在牛市中的预测准确性和收益表现
"""

import os
import time
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stock_env_v9 import StockTradingEnvV9
import random
import numpy as np

# 设置输出编码
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# === 配置参数 ===
# V20使用牛市牛股数据目录
STOCK_FILES = []
train_dir = 'stockdata_v20_bull_market/train'
test_dir = 'stockdata_v20_bull_market/test'

# 扫描牛市牛股训练数据
if os.path.exists(train_dir):
    stock_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.csv')]
    STOCK_FILES = sorted([f for f in stock_files if os.path.exists(f)])
    print(f"✅ 找到 {len(STOCK_FILES)} 只牛市牛股训练数据")
    for f in STOCK_FILES[:5]:  # 显示前5个
        print(f"   - {os.path.basename(f)}")
    if len(STOCK_FILES) > 5:
        print(f"   ... 还有 {len(STOCK_FILES) - 5} 只股票")
else:
    print(f"⚠️  牛市牛股训练数据目录不存在: {train_dir}")
    print("请先运行: python get_bull_market_stocks_data.py")
    # 如果牛市牛股数据不存在，尝试使用通用数据
    train_dir_fallback = 'stockdata_v7/train'
    if os.path.exists(train_dir_fallback):
        stock_files = [os.path.join(train_dir_fallback, f) for f in os.listdir(train_dir_fallback) if f.endswith('.csv')]
        STOCK_FILES = sorted([f for f in stock_files if os.path.exists(f)])
        print(f"⚠️  使用备用训练数据: {train_dir_fallback}")
        print(f"   找到 {len(STOCK_FILES)} 只股票")
    else:
        print("❌ 没有找到任何训练数据！")
        sys.exit(1)

# 检查并过滤存在的文件
STOCK_FILES = [f for f in STOCK_FILES if os.path.exists(f)]

if len(STOCK_FILES) == 0:
    print("❌ 没有找到训练数据！")
    print("请先运行: python get_bull_market_stocks_data.py")
    sys.exit(1)

TOTAL_TIMESTEPS = 3_000_000  # 300万步
N_ENVS = 8  # 并行环境数
SAVE_FREQ = 100_000  # 每10万步保存一次

# PPO 超参数（针对牛市优化）
PPO_PARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,  # 适度探索
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "verbose": 1,
    "tensorboard_log": "./logs_v20_bull_market/"
}

# 模型保存路径
MODEL_DIR = "models_v20_bull_market"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PPO_PARAMS["tensorboard_log"], exist_ok=True)


def make_env(stock_files, rank, seed=0):
    def _init():
        stock_file = random.choice(stock_files)
        env = StockTradingEnvV9(
            data_file=stock_file,
            initial_balance=100000,
            llm_provider="deepseek",
            enable_llm_cache=True,
            llm_weight=0.05  # LLM权重5%
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    print("="*70)
    print("V20牛市牛股训练版 - 训练启动")
    print("="*70)
    print(f"训练股票数量: {len(STOCK_FILES)}")
    print(f"总训练步数: {TOTAL_TIMESTEPS:,}")
    print(f"并行环境数: {N_ENVS}")
    print(f"保存频率: 每 {SAVE_FREQ:,} 步")
    print("="*70)
    
    # 创建环境
    if N_ENVS > 1:
        env = SubprocVecEnv([make_env(STOCK_FILES, i) for i in range(N_ENVS)])
    else:
        env = DummyVecEnv([make_env(STOCK_FILES, 0)])
    
    # 创建模型
    model = PPO(
        "MlpPolicy",
        env,
        **PPO_PARAMS
    )
    
    # 回调函数
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=MODEL_DIR,
        name_prefix="ppo_stock_v20_bull_market"
    )
    
    # 评估环境（使用第一只股票）
    eval_env = DummyVecEnv([make_env(STOCK_FILES, 0)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODEL_DIR, "best"),
        log_path=os.path.join(PPO_PARAMS["tensorboard_log"], "eval"),
        eval_freq=SAVE_FREQ,
        deterministic=True,
        render=False
    )
    
    # 开始训练
    print("\n🚀 开始训练...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # 保存最终模型
    final_model_path = os.path.join(MODEL_DIR, "ppo_stock_v20_bull_market_final.zip")
    model.save(final_model_path)
    
    elapsed_time = time.time() - start_time
    print("\n" + "="*70)
    print("✅ 训练完成！")
    print(f"训练时间: {elapsed_time/3600:.2f} 小时")
    print(f"最终模型保存路径: {final_model_path}")
    print(f"最佳模型保存路径: {os.path.join(MODEL_DIR, 'best', 'best_model.zip')}")
    print("="*70)


if __name__ == "__main__":
    main()







