# test_models_for_002025.py - 测试不同模型在航天电器上的回测表现
"""
测试多个模型在航天电器(sz.002025)上的回测表现，选择最佳模型
"""

import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stock_env_v6 import StockTradingEnv

# 设置中文字体
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# 航天电器测试数据文件
TEST_DATA_FILE = 'stockdata_v7_002025/test/sz.002025.航天电器.csv'

# 候选模型列表（根据代码中的配置和相似股票选择）
CANDIDATE_MODELS = [
    {
        'name': '通用模型',
        'path': 'ppo_stock_v7.zip',
        'description': '通用PPO模型（当前使用）'
    },
    {
        'name': '航天电器专用模型',
        'path': 'ppo_stock_v7_002025.zip',
        'description': '航天电器002025专用模型'
    },
    {
        'name': '英维克模型(002837)',
        'path': 'ppo_stock_v7_002837.zip',
        'description': '英维克002837专用模型 - 🏆 双料冠军（夏普3.25）'
    },
    {
        'name': '铂科新材模型(300811)',
        'path': 'ppo_stock_v7_300811.zip',
        'description': '铂科新材300811专用模型 - ⭐ V16特色模型（夏普2.61）'
    },
    {
        'name': '高澜股份模型(300499)',
        'path': 'ppo_stock_v7_300499.zip',
        'description': '高澜股份300499专用模型 - 零回撤之王'
    },
    {
        'name': '鸿远电子模型(603267)',
        'path': 'ppo_stock_v7_603267.zip',
        'description': '鸿远电子603267专用模型（使用航天电器模型时表现好）'
    },
    {
        'name': '歌尔股份模型(002241)',
        'path': 'ppo_stock_v7_002241.zip',
        'description': '歌尔股份002241专用模型（夏普2.51）'
    },
    {
        'name': '圣邦股份模型(300661)',
        'path': 'ppo_stock_v7_300661.zip',
        'description': '圣邦股份300661专用模型'
    },
    {
        'name': '上海瀚讯模型(300762)',
        'path': 'ppo_stock_v7_300762.zip',
        'description': '上海瀚讯300762专用模型（夏普1.83）'
    },
    {
        'name': '宏达电子模型(300726)',
        'path': 'ppo_stock_v7_300726.zip',
        'description': '宏达电子300726专用模型（夏普1.71）'
    },
    {
        'name': '科泰电源模型(300153)',
        'path': 'ppo_stock_v7_300153.zip',
        'description': '科泰电源300153专用模型（夏普2.08）'
    },
    {
        'name': '汉威科技模型(300007)',
        'path': 'ppo_stock_v7_300007.zip',
        'description': '汉威科技300007专用模型（夏普1.72）'
    },
]

def test_model(model_path, data_file):
    """测试单个模型"""
    try:
        if not os.path.exists(model_path):
            return {
                'success': False,
                'error': f'模型文件不存在: {model_path}'
            }
        
        # 加载模型
        model = PPO.load(model_path)
        
        # 创建环境
        initial_balance = 20000.0  # 默认初始资金2万元
        env = StockTradingEnv(data_file, initial_balance=initial_balance)
        obs, _ = env.reset()
        done = False
        
        # 执行回测
        step_count = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            step_count += 1
        
        # 获取统计数据
        stats = env.get_stats()
        
        return {
            'success': True,
            'final_net_worth': stats.get('final_net_worth', 0),
            'total_return': stats.get('total_return', 0),
            'max_drawdown': stats.get('max_drawdown', 0),
            'sharpe_ratio': stats.get('sharpe_ratio', 0),
            'num_trades': stats.get('num_trades', 0),
            'win_rate': stats.get('win_rate', 0),
            'risk_events': stats.get('risk_events', 0),
            'total_days': stats.get('total_days', 0),
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def compare_models_for_002025():
    """对比所有候选模型在航天电器上的表现"""
    
    if not os.path.exists(TEST_DATA_FILE):
        print(f"❌ 测试数据文件不存在: {TEST_DATA_FILE}")
        print("   请确保已生成航天电器的测试数据")
        return
    
    print("=" * 80)
    print("📊 航天电器(sz.002025)模型回测对比")
    print("=" * 80)
    print(f"测试数据: {TEST_DATA_FILE}")
    print(f"候选模型数: {len(CANDIDATE_MODELS)}")
    print("=" * 80)
    print()
    
    results = []
    
    for idx, model_info in enumerate(CANDIDATE_MODELS, 1):
        model_name = model_info['name']
        model_path = model_info['path']
        description = model_info['description']
        
        print(f"[{idx}/{len(CANDIDATE_MODELS)}] 测试模型: {model_name}")
        print(f"   描述: {description}")
        print(f"   路径: {model_path}")
        
        result = test_model(model_path, TEST_DATA_FILE)
        
        if result['success']:
            results.append({
                'name': model_name,
                'path': model_path,
                'description': description,
                **result
            })
            print(f"   ✅ 收益率: {result['total_return']:+.2f}%")
            print(f"   ✅ 夏普比率: {result['sharpe_ratio']:.2f}")
            print(f"   ✅ 最大回撤: {result['max_drawdown']:.2f}%")
            print(f"   ✅ 交易次数: {result['num_trades']}")
            print(f"   ✅ 胜率: {result['win_rate']:.2f}%")
        else:
            print(f"   ❌ 测试失败: {result.get('error', '未知错误')}")
        
        print()
    
    if len(results) == 0:
        print("❌ 没有成功测试的模型！")
        return
    
    # 按夏普比率排序（优先），如果夏普相同则按收益率排序
    results_sorted = sorted(results, 
                          key=lambda x: (x['sharpe_ratio'] if x['sharpe_ratio'] is not None else -999, 
                                       x['total_return']), 
                          reverse=True)
    
    # 打印对比结果
    print("=" * 80)
    print("📈 模型回测对比结果（按夏普比率排序）")
    print("=" * 80)
    print(f"{'排名':<6} {'模型名称':<25} {'收益率':<12} {'夏普比率':<12} {'最大回撤':<12} {'胜率':<10}")
    print("-" * 80)
    
    for rank, result in enumerate(results_sorted, 1):
        print(f"{rank:<6} {result['name']:<25} "
              f"{result['total_return']:>+10.2f}%  "
              f"{result['sharpe_ratio']:>10.2f}  "
              f"{result['max_drawdown']:>10.2f}%  "
              f"{result['win_rate']:>8.2f}%")
    
    print("=" * 80)
    print()
    
    # 推荐最佳模型
    if len(results_sorted) > 0:
        best_model = results_sorted[0]
        print("🏆 推荐最佳模型:")
        print(f"   模型名称: {best_model['name']}")
        print(f"   模型路径: {best_model['path']}")
        print(f"   描述: {best_model['description']}")
        print(f"   收益率: {best_model['total_return']:+.2f}%")
        print(f"   夏普比率: {best_model['sharpe_ratio']:.2f}")
        print(f"   最大回撤: {best_model['max_drawdown']:.2f}%")
        print(f"   交易次数: {best_model['num_trades']}")
        print(f"   胜率: {best_model['win_rate']:.2f}%")
        print()
        
        # 与当前模型对比
        current_model = next((r for r in results if '通用模型' in r['name']), None)
        if current_model and current_model['name'] != best_model['name']:
            print("📊 与当前模型对比:")
            print(f"   当前模型: {current_model['name']}")
            print(f"   收益率提升: {best_model['total_return'] - current_model['total_return']:+.2f}%")
            print(f"   夏普比率提升: {best_model['sharpe_ratio'] - current_model['sharpe_ratio']:+.2f}")
            print(f"   回撤变化: {best_model['max_drawdown'] - current_model['max_drawdown']:+.2f}%")
            print()
        
        # 保存结果到文件
        output_file = 'model_test_results_002025.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("航天电器(sz.002025)模型回测对比结果\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"测试数据: {TEST_DATA_FILE}\n")
            f.write(f"测试时间: {pd.Timestamp.now()}\n\n")
            f.write(f"{'排名':<6} {'模型名称':<25} {'收益率':<12} {'夏普比率':<12} {'最大回撤':<12} {'胜率':<10}\n")
            f.write("-" * 80 + "\n")
            for rank, result in enumerate(results_sorted, 1):
                f.write(f"{rank:<6} {result['name']:<25} "
                       f"{result['total_return']:>+10.2f}%  "
                       f"{result['sharpe_ratio']:>10.2f}  "
                       f"{result['max_drawdown']:>10.2f}%  "
                       f"{result['win_rate']:>8.2f}%\n")
            f.write("\n" + "=" * 80 + "\n")
            f.write("🏆 推荐最佳模型:\n")
            f.write(f"   模型名称: {best_model['name']}\n")
            f.write(f"   模型路径: {best_model['path']}\n")
            f.write(f"   收益率: {best_model['total_return']:+.2f}%\n")
            f.write(f"   夏普比率: {best_model['sharpe_ratio']:.2f}\n")
            f.write(f"   最大回撤: {best_model['max_drawdown']:.2f}%\n")
        
        print(f"💾 结果已保存到: {output_file}")
    
    print("\n✅ 测试完成！")

if __name__ == '__main__':
    compare_models_for_002025()

