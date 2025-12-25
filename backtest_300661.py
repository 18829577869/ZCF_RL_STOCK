"""
圣邦股份回测脚本
针对：圣邦股份(sz.300661)
使用对应的PPO模型进行回测，获取收益、夏普和回撤信息
"""

import os
import sys
import json
import datetime
import glob
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stock_env_v6 import StockTradingEnv

# 股票配置（V11版本）
TARGET_STOCK = {
    'code': 'sz.300661',
    'name': '圣邦股份',
    'model': 'ppo_stock_v7_300661.zip',
    'data_dir': 'stockdata_v7_300661'
}

# 初始资金
INITIAL_BALANCE = 100000

def get_stock_data_file(stock_code):
    """获取股票数据文件路径"""
    code_parts = stock_code.split('.')
    code_number = code_parts[-1] if len(code_parts) > 1 else stock_code.replace('.', '')
    
    possible_paths = [
        f"stockdata_v7_300661/train/{stock_code}.*.csv",
        f"stockdata_v7_300661/test/{stock_code}.*.csv",
        f"stockdata_v7_{code_number}/train/{stock_code}.*.csv",
        f"stockdata_v7_{code_number}/test/{stock_code}.*.csv",
        f"stockdata_v7/train/{stock_code}.csv",
        f"stockdata_v7/test/{stock_code}.csv",
    ]
    
    for path in possible_paths:
        if '*' in path:
            matches = glob.glob(path)
            if matches:
                return matches[0]
        else:
            if os.path.exists(path):
                return path
    
    return None

def calculate_sharpe_ratio(returns, risk_free_rate=0.03):
    """计算夏普比率（年化，假设252个交易日）"""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / 252
    sharpe = np.sqrt(252) * np.mean(excess_returns) / np.std(returns)
    return sharpe

def calculate_max_drawdown(net_worth_history):
    """计算最大回撤"""
    if not net_worth_history or len(net_worth_history) < 2:
        return {'max_drawdown_pct': 0.0, 'max_drawdown_value': 0.0, 'peak': 0.0, 'trough': 0.0}
    
    peak = net_worth_history[0]
    max_dd = 0.0
    max_dd_value = 0.0
    trough = peak
    
    for nw in net_worth_history:
        if nw > peak:
            peak = nw
        dd = peak - nw
        dd_pct = (dd / peak * 100) if peak > 0 else 0.0
        
        if dd_pct > max_dd:
            max_dd = dd_pct
            max_dd_value = dd
            trough = nw
    
    return {
        'max_drawdown_pct': max_dd,
        'max_drawdown_value': max_dd_value,
        'peak': peak,
        'trough': trough
    }

def backtest_stock(stock_info):
    """对股票进行回测"""
    stock_code = stock_info['code']
    stock_name = stock_info['name']
    model_path = stock_info['model']
    
    try:
        print(f"\n{'='*70}")
        print(f"📊 开始回测: {stock_name} ({stock_code})")
        print(f"{'='*70}")
        print(f"   模型: {model_path}")
        
        # 获取数据文件
        print(f"\n🔍 正在查找 {stock_name} ({stock_code}) 的数据文件...")
        data_file = get_stock_data_file(stock_code)
        if not data_file:
            print(f"   ⚠️  未找到数据文件")
            return {
                'stock_code': stock_code,
                'stock_name': stock_name,
                'status': 'failed',
                'error': '未找到数据文件'
            }
        else:
            print(f"   ✅ 找到数据文件: {data_file}")
        
        # 加载模型
        print(f"   📥 加载模型: {model_path}")
        if not os.path.exists(model_path):
            print(f"   ❌ 模型文件不存在: {model_path}")
            return {
                'stock_code': stock_code,
                'stock_name': stock_name,
                'status': 'failed',
                'error': f'模型文件不存在: {model_path}'
            }
        
        try:
            model = PPO.load(model_path)
            print(f"   ✅ 模型加载成功")
        except Exception as e:
            print(f"   ❌ 模型加载失败: {e}")
            return {
                'stock_code': stock_code,
                'stock_name': stock_name,
                'status': 'failed',
                'error': f'模型加载失败: {str(e)}'
            }
        
        # 创建环境
        print(f"   📥 创建交易环境...")
        try:
            env = StockTradingEnv(data_file, initial_balance=INITIAL_BALANCE)
            print(f"   ✅ 环境创建成功，数据点: {len(env.df)}")
        except Exception as e:
            print(f"   ❌ 环境创建失败: {e}")
            return {
                'stock_code': stock_code,
                'stock_name': stock_name,
                'status': 'failed',
                'error': f'环境创建失败: {str(e)}'
            }
        
        # 执行回测
        print(f"   🚀 开始执行回测...")
        obs, _ = env.reset()
        done = False
        
        net_worth_history = [INITIAL_BALANCE]
        daily_returns = []
        prev_net_worth = INITIAL_BALANCE
        
        step_count = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            net_worth_history.append(env.net_worth)
            
            if prev_net_worth > 0:
                daily_return = (env.net_worth - prev_net_worth) / prev_net_worth
                daily_returns.append(daily_return)
            
            prev_net_worth = env.net_worth
            step_count += 1
            
            if step_count % 50 == 0:
                print(f"      步骤 {step_count}/{len(env.df)}: 净值={env.net_worth:.2f}, 收益={(env.net_worth-INITIAL_BALANCE):.2f}")
        
        print(f"   ✅ 回测完成，共 {step_count} 步")
        
        # 计算指标
        print(f"   📊 计算回测指标...")
        
        final_net_worth = env.net_worth
        total_return = (final_net_worth - INITIAL_BALANCE) / INITIAL_BALANCE * 100
        
        trading_days = len(net_worth_history) - 1
        if trading_days > 0:
            annual_return = ((final_net_worth / INITIAL_BALANCE) ** (252 / trading_days) - 1) * 100
        else:
            annual_return = 0.0
        
        max_drawdown_info = calculate_max_drawdown(net_worth_history)
        
        if len(daily_returns) > 0:
            sharpe_ratio = calculate_sharpe_ratio(np.array(daily_returns))
        else:
            sharpe_ratio = 0.0
        
        result = {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'status': 'success',
            'initial_balance': float(INITIAL_BALANCE),
            'final_net_worth': float(final_net_worth),
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'max_drawdown_pct': float(max_drawdown_info['max_drawdown_pct']),
            'sharpe_ratio': float(sharpe_ratio),
            'trading_days': int(trading_days),
            'steps': int(step_count),
        }
        
        print(f"\n   📊 回测结果:")
        print(f"      初始资金: {INITIAL_BALANCE:,.2f} 元")
        print(f"      最终净值: {final_net_worth:,.2f} 元")
        print(f"      总收益率: {total_return:+.2f}%")
        print(f"      年化收益率: {annual_return:+.2f}%")
        print(f"      最大回撤: {max_drawdown_info['max_drawdown_pct']:.2f}%")
        print(f"      夏普比率: {sharpe_ratio:.4f}")
        print(f"      交易天数: {trading_days} 天")
        
        return result
        
    except Exception as e:
        print(f"   ❌ 回测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'status': 'failed',
            'error': str(e)
        }

def main():
    """主函数"""
    print("="*70)
    print("圣邦股份回测脚本")
    print("="*70)
    print(f"\n目标股票: {TARGET_STOCK['name']} ({TARGET_STOCK['code']})")
    print(f"模型: {TARGET_STOCK['model']}")
    print("="*70 + "\n")
    
    result = backtest_stock(TARGET_STOCK)
    
    # 打印汇总
    print("\n" + "="*70)
    print("📊 回测结果汇总")
    print("="*70)
    
    if result['status'] == 'success':
        print(f"\n股票名称: {result['stock_name']}")
        print(f"股票代码: {result['stock_code']}")
        print(f"总收益率: {result['total_return']:+.2f}%")
        print(f"年化收益率: {result['annual_return']:+.2f}%")
        print(f"最大回撤: {result['max_drawdown_pct']:.2f}%")
        print(f"夏普比率: {result['sharpe_ratio']:.4f}")
        print(f"交易天数: {result['trading_days']} 天")
        print("状态: ✅ 成功")
    else:
        print(f"\n股票名称: {result['stock_name']}")
        print(f"股票代码: {result['stock_code']}")
        print(f"状态: ❌ 失败")
        print(f"错误: {result.get('error', '未知错误')}")
    
    # 保存结果
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f"backtest_300661_results_{timestamp}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n📁 详细结果已保存到: {result_file}")
    print("="*70)

if __name__ == '__main__':
    main()

