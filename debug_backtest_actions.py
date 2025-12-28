"""
调试回测：检查模型预测的action分布
"""
import os
import numpy as np
from stable_baselines3 import PPO
from stock_env_v6 import StockTradingEnv

# 加载模型
model_path = "ppo_stock_v7_1A0001.zip"
if not os.path.exists(model_path):
    print(f"[错误] 模型文件不存在: {model_path}")
    exit(1)

print(f"[加载模型] {model_path}")
model = PPO.load(model_path)

# 测试文件
test_dir = 'stockdata_v7_1A0001/test'
test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]
test_files = sorted([f for f in test_files if os.path.exists(f)])

if len(test_files) == 0:
    print(f"[错误] 测试文件不存在: {test_dir}")
    exit(1)

print(f"\n📊 找到 {len(test_files)} 个测试文件\n")

# 测试每个文件
for test_file in test_files[:3]:  # 只测试前3个
    print("="*70)
    print(f"测试文件: {os.path.basename(test_file)}")
    print("="*70)
    
    try:
        # 检查是否是指数文件，指数使用最小交易单位1（指数点）
        is_index = ('000001' in test_file or '上证指数' in test_file or '1A0001' in test_file or 
                   '000016' in test_file or '000300' in test_file or '399001' in test_file or '399006' in test_file)
        min_trade_unit = 1 if is_index else 100  # 指数用1，股票用100
        
        env = StockTradingEnv(test_file, initial_balance=100000, min_trade_unit=min_trade_unit)
        obs, _ = env.reset()
        done = False
        
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        action_list = []
        step_count = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            # 确保action是整数
            if isinstance(action, np.ndarray):
                action = int(action.item())
            else:
                action = int(action)
            
            action_counts[action] = action_counts.get(action, 0) + 1
            action_list.append(action)
            
            # 检查风险控制
            current_price = float(env.df.iloc[env.current_step]['close'])
            risk_level, warnings = env._assess_risk_level()
            
            original_action = action
            if risk_level >= env.risk_threshold and action in [1, 2, 3]:
                action = 0
                if original_action != action:
                    print(f"  步骤 {step_count}: 风险控制阻止买入 (原action={original_action}, risk_level={risk_level:.2f}, threshold={env.risk_threshold})")
            
            obs, reward, done, truncated, _ = env.step(action)
            step_count += 1
            
            if step_count % 50 == 0:
                print(f"  步骤 {step_count}: action分布 = {action_counts}, 持仓={env.shares_held:.0f}, 资金={env.balance:.2f}")
        
        stats = env.get_stats()
        
        print(f"\n[最终统计]")
        print(f"  总步数: {step_count}")
        print(f"  Action分布:")
        action_names = {0: "持有", 1: "买入25%", 2: "买入50%", 3: "买入100%", 4: "卖出25%", 5: "卖出50%", 6: "卖出100%"}
        for act, count in sorted(action_counts.items()):
            pct = count / step_count * 100 if step_count > 0 else 0
            print(f"    {act} ({action_names[act]}): {count} 次 ({pct:.1f}%)")
        print(f"  交易次数: {stats['num_trades']}")
        print(f"  交易历史长度: {len(env.trade_history)}")
        print(f"  最终净值: {stats['final_net_worth']:,.2f} 元")
        print(f"  总收益率: {stats['total_return']:+.2f}%")
        print()
        
        # 显示前20个action
        print(f"  前20个action: {action_list[:20]}")
        print()
        
    except Exception as e:
        print(f"[错误] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        print()

