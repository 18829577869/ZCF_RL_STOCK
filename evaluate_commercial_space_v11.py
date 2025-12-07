# evaluate_commercial_space_v11.py - 商业航天板块V11模型评估脚本
# -*- coding: utf-8 -*-
"""
商业航天板块V11模型评估脚本：
1. 使用V11模型评估商业航天板块股票
2. 根据收益率和夏普比率筛选出5只最佳股票
3. 支持多种模型选择（优先使用V11模型）
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stock_env_v6 import StockTradingEnv

# 设置输出编码
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# 初始资金（与V11一致）
INITIAL_BALANCE = 20000  # 2万元

# 数据目录
DATA_DIR = 'stockdata_v7_commercial_space/test'

# 模型路径（按优先级）
MODEL_PATHS = [
    "ppo_stock_v7_300749.zip",  # 顶固集创专用模型
    "ppo_stock_v7_301017.zip",  # 漱玉平民专用模型
    "ppo_stock_v7.zip",  # 通用V7模型
    "ppo_stock_v9.zip",  # V9模型
    "ppo_stock_v8_fixed.zip",  # V8模型
    "ppo_stock_v6.zip",  # V6模型
]

def find_model():
    """查找可用的模型"""
    for model_path in MODEL_PATHS:
        if os.path.exists(model_path):
            return model_path
    
    # 尝试查找models目录下的最佳模型
    possible_dirs = [
        "models_v7_300749/best",
        "models_v7_301017/best",
        "models_v7/best",
        "models_v9/best",
        "models_v8/best",
    ]
    
    for model_dir in possible_dirs:
        best_model = os.path.join(model_dir, "best_model.zip")
        if os.path.exists(best_model):
            return best_model
    
    return None

def evaluate_stock(model, stock_file, initial_balance=INITIAL_BALANCE):
    """评估单只股票"""
    try:
        env = StockTradingEnv(stock_file, initial_balance=initial_balance)
        obs, _ = env.reset()
        done = False
        
        daily_returns = []
        prev_net_worth = initial_balance
        
        actions_taken = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated
            
            # 记录日收益率
            current_net_worth = env.net_worth
            daily_return = (current_net_worth - prev_net_worth) / prev_net_worth if prev_net_worth > 0 else 0
            daily_returns.append(daily_return)
            prev_net_worth = current_net_worth
            
            if isinstance(action, (int, np.integer)):
                actions_taken.append(int(action))
            else:
                actions_taken.append(int(action.item()))
        
        # 获取统计数据
        stats = env.get_stats()
        
        # 计算夏普比率（如果环境没有提供）
        if 'sharpe_ratio' not in stats or stats['sharpe_ratio'] == 0:
            if len(daily_returns) > 0:
                daily_returns_array = np.array(daily_returns)
                if daily_returns_array.std() > 0:
                    sharpe_ratio = (daily_returns_array.mean() / daily_returns_array.std()) * np.sqrt(252)
                else:
                    sharpe_ratio = 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = stats['sharpe_ratio']
        
        # 提取股票信息
        stock_name = env.stock_info.get('name', '未知')
        stock_code = env.stock_info.get('code', '未知')
        
        return {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'stock_file': stock_file,
            'final_net_worth': stats.get('final_net_worth', env.net_worth),
            'total_return': stats.get('total_return', (env.net_worth - initial_balance) / initial_balance * 100),
            'max_drawdown': stats.get('max_drawdown', 0),
            'sharpe_ratio': sharpe_ratio,
            'num_trades': stats.get('num_trades', 0),
            'win_rate': stats.get('win_rate', 0),
            'risk_events': stats.get('risk_events', 0),
            'category': stats.get('category', '未知'),
            'volatility': stats.get('volatility', '未知'),
            'actions': actions_taken,
            'daily_returns': daily_returns,
            'success': True
        }
    
    except Exception as e:
        print(f"  [错误] 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            'stock_file': stock_file,
            'stock_name': os.path.basename(stock_file).replace('.csv', ''),
            'success': False,
            'error': str(e)
        }

def main():
    print("\n" + "="*70)
    print("商业航天板块 - V11模型评估")
    print("="*70 + "\n")
    
    # 检查数据目录
    if not os.path.exists(DATA_DIR):
        print(f"[错误] 数据目录不存在: {DATA_DIR}")
        print("请先运行: python get_stock_data_v11_commercial_space.py")
        return
    
    # 查找测试数据文件
    test_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    test_files = sorted([f for f in test_files if os.path.exists(f)])
    
    if len(test_files) == 0:
        print(f"[错误] 测试数据目录为空: {DATA_DIR}")
        print("请先运行: python get_stock_data_v11_commercial_space.py")
        return
    
    print(f"找到 {len(test_files)} 只测试标的\n")
    
    # 查找模型
    model_path = find_model()
    if model_path is None:
        print("[错误] 未找到可用的模型文件！")
        print("请确保存在以下模型之一：")
        for path in MODEL_PATHS:
            print(f"  - {path}")
        print("\n或者运行训练脚本生成模型")
        return
    
    print(f"[加载] 模型: {model_path}\n")
    try:
        model = PPO.load(model_path)
        print("✅ 模型加载成功\n")
    except Exception as e:
        print(f"[错误] 模型加载失败: {e}")
        return
    
    # 评估所有股票
    print("="*70)
    print("开始评估...")
    print("="*70 + "\n")
    
    results = []
    for i, stock_file in enumerate(test_files, 1):
        stock_name = os.path.basename(stock_file).replace('.csv', '')
        print(f"[{i}/{len(test_files)}] 评估: {stock_name}")
        
        result = evaluate_stock(model, stock_file, INITIAL_BALANCE)
        results.append(result)
        
        if result.get('success', False):
            print(f"  ✅ 最终净值: {result['final_net_worth']:,.0f} 元")
            print(f"  ✅ 收益率: {result['total_return']:+.2f}%")
            print(f"  ✅ 最大回撤: {result['max_drawdown']:.2f}%")
            print(f"  ✅ 夏普比率: {result['sharpe_ratio']:.2f}")
            print(f"  ✅ 交易次数: {result['num_trades']}")
            print(f"  ✅ 胜率: {result['win_rate']:.2f}%")
            print()
        else:
            print(f"  ❌ 评估失败: {result.get('error', '未知错误')}\n")
    
    # 筛选成功评估的股票
    success_results = [r for r in results if r.get('success', False)]
    
    if len(success_results) == 0:
        print("[错误] 没有成功评估任何股票！")
        return
    
    # 计算综合评分（收益率权重60%，夏普比率权重40%）
    for result in success_results:
        # 归一化收益率（假设-50%到+100%的范围）
        normalized_return = (result['total_return'] + 50) / 150
        normalized_return = max(0, min(1, normalized_return))  # 限制在[0, 1]
        
        # 归一化夏普比率（假设-2到+5的范围）
        normalized_sharpe = (result['sharpe_ratio'] + 2) / 7
        normalized_sharpe = max(0, min(1, normalized_sharpe))  # 限制在[0, 1]
        
        # 综合评分
        result['composite_score'] = normalized_return * 0.6 + normalized_sharpe * 0.4
    
    # 按综合评分排序
    success_results.sort(key=lambda x: x['composite_score'], reverse=True)
    
    # 显示所有结果
    print("="*70)
    print("评估结果汇总")
    print("="*70 + "\n")
    
    df_all = pd.DataFrame([
        {
            '股票代码': r['stock_code'],
            '股票名称': r['stock_name'],
            '收益率(%)': f"{r['total_return']:+.2f}",
            '夏普比率': f"{r['sharpe_ratio']:.2f}",
            '最大回撤(%)': f"{r['max_drawdown']:.2f}",
            '交易次数': r['num_trades'],
            '胜率(%)': f"{r['win_rate']:.2f}",
            '综合评分': f"{r['composite_score']:.3f}"
        }
        for r in success_results
    ])
    
    print(df_all.to_string(index=False))
    print()
    
    # 筛选前5只股票
    top_5 = success_results[:5]
    
    print("="*70)
    print("🎯 推荐5只股票（按综合评分排序）")
    print("="*70 + "\n")
    
    for i, result in enumerate(top_5, 1):
        print(f"第 {i} 名: {result['stock_name']} ({result['stock_code']})")
        print(f"  收益率: {result['total_return']:+.2f}%")
        print(f"  夏普比率: {result['sharpe_ratio']:.2f}")
        print(f"  最大回撤: {result['max_drawdown']:.2f}%")
        print(f"  交易次数: {result['num_trades']}")
        print(f"  胜率: {result['win_rate']:.2f}%")
        print(f"  综合评分: {result['composite_score']:.3f}")
        print(f"  类别: {result.get('category', '未知')}")
        print(f"  波动性: {result.get('volatility', '未知')}")
        print()
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存所有结果
    csv_file_all = f"results_commercial_space_all_{timestamp}.csv"
    df_all.to_csv(csv_file_all, index=False, encoding='utf-8-sig')
    
    # 保存前5只股票
    df_top5 = pd.DataFrame([
        {
            '排名': i,
            '股票代码': r['stock_code'],
            '股票名称': r['stock_name'],
            '收益率(%)': r['total_return'],
            '夏普比率': r['sharpe_ratio'],
            '最大回撤(%)': r['max_drawdown'],
            '交易次数': r['num_trades'],
            '胜率(%)': r['win_rate'],
            '综合评分': r['composite_score'],
            '类别': r.get('category', '未知'),
            '波动性': r.get('volatility', '未知')
        }
        for i, r in enumerate(top_5, 1)
    ])
    
    csv_file_top5 = f"results_commercial_space_top5_{timestamp}.csv"
    df_top5.to_csv(csv_file_top5, index=False, encoding='utf-8-sig')
    
    print("="*70)
    print("结果已保存")
    print("="*70)
    print(f"全部结果: {csv_file_all}")
    print(f"前5只股票: {csv_file_top5}")
    print()
    
    # 统计信息
    print("="*70)
    print("统计信息")
    print("="*70 + "\n")
    
    avg_return = np.mean([r['total_return'] for r in success_results])
    avg_sharpe = np.mean([r['sharpe_ratio'] for r in success_results])
    avg_drawdown = np.mean([r['max_drawdown'] for r in success_results])
    
    print(f"平均收益率: {avg_return:+.2f}%")
    print(f"平均夏普比率: {avg_sharpe:.2f}")
    print(f"平均最大回撤: {avg_drawdown:.2f}%")
    print(f"成功评估: {len(success_results)}/{len(test_files)} 只")
    print()
    
    # 前5只的平均值
    if len(top_5) > 0:
        top5_avg_return = np.mean([r['total_return'] for r in top_5])
        top5_avg_sharpe = np.mean([r['sharpe_ratio'] for r in top_5])
        top5_avg_drawdown = np.mean([r['max_drawdown'] for r in top_5])
        
        print(f"\n前5只股票平均值:")
        print(f"  平均收益率: {top5_avg_return:+.2f}%")
        print(f"  平均夏普比率: {top5_avg_sharpe:.2f}")
        print(f"  平均最大回撤: {top5_avg_drawdown:.2f}%")
    
    print("\n" + "="*70)
    print("评估完成！")
    print("="*70)
    print(f"\n💡 提示: 可以使用V11实时预测脚本对选中的股票进行预测")
    print(f"   python real_time_predict_v11.py")
    print(f"\n💡 推荐使用的5只股票代码:")
    for i, r in enumerate(top_5, 1):
        print(f"   {i}. {r['stock_code']} - {r['stock_name']}")
    print()

if __name__ == "__main__":
    main()

