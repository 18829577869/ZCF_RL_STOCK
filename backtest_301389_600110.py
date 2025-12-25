"""
隆扬电子和诺德股份回测脚本
针对：隆扬电子(sz.301389)、诺德股份(sh.600110)
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

# 两只股票配置（V11版本）
TARGET_STOCKS = [
    {'code': 'sz.301389', 'name': '隆扬电子', 'model': 'ppo_stock_v7_301389.zip', 'data_dir': 'stockdata_v7_301389_600110'},
    {'code': 'sh.600110', 'name': '诺德股份', 'model': 'ppo_stock_v7_600110.zip', 'data_dir': 'stockdata_v7_301389_600110'},
]

# 初始资金
INITIAL_BALANCE = 100000

def get_stock_data_file(stock_code):
    """
    获取股票数据文件路径
    尝试多个可能的路径
    """
    # 提取股票代码的数字部分（如 sh.600110 -> 600110）
    code_parts = stock_code.split('.')
    code_number = code_parts[-1] if len(code_parts) > 1 else stock_code.replace('.', '')
    
    # 可能的路径列表（按优先级排序）
    possible_paths = [
        # 格式1: stockdata_v7_301389_600110/train/{stock_code}.{name}.csv (V11专用目录)
        f"stockdata_v7_301389_600110/train/{stock_code}.*.csv",
        f"stockdata_v7_301389_600110/test/{stock_code}.*.csv",
        # 格式2: stockdata_v7_{code}/train/{stock_code}.{name}.csv
        f"stockdata_v7_{code_number}/train/{stock_code}.*.csv",
        f"stockdata_v7_{code_number}/test/{stock_code}.*.csv",
        # 格式3: stockdata_v7/train/{stock_code}.csv
        f"stockdata_v7/train/{stock_code}.csv",
        f"stockdata_v7/test/{stock_code}.csv",
        # 格式4: stockdata_v7_realtime/{stock_code}.csv
        f"stockdata_v7_realtime/{stock_code}.csv",
        f"stockdata/{stock_code}.csv",
        # 格式5: 带下划线的格式
        f"stockdata_v7_realtime/{stock_code.replace('.', '_')}.csv",
    ]
    
    # 先尝试精确匹配的路径
    for path in possible_paths:
        if '*' in path:
            # 使用glob模式匹配
            matches = glob.glob(path)
            if matches:
                return matches[0]  # 返回第一个匹配的文件
        else:
            if os.path.exists(path):
                return path
    
    # 尝试根据股票代码查找文件（递归搜索）
    code_clean = stock_code.replace('.', '_')
    code_number_clean = code_number
    
    # 搜索包含股票代码的目录
    search_patterns = [
        f"stockdata_v7_{code_number}",
        f"*{code_number}*",
    ]
    
    for pattern in search_patterns:
        for dir_path in glob.glob(pattern):
            if os.path.isdir(dir_path):
                # 在目录中查找CSV文件
                for subdir in ['train', 'test', '']:
                    search_dir = os.path.join(dir_path, subdir) if subdir else dir_path
                    if os.path.isdir(search_dir):
                        for file in os.listdir(search_dir):
                            if file.endswith('.csv') and (code_number in file or stock_code in file):
                                return os.path.join(search_dir, file)
    
    # 最后尝试全局搜索
    for root, dirs, files in os.walk('.'):
        # 跳过一些不需要搜索的目录
        skip_dirs = ['node_modules', '__pycache__', '.git', 'venv', 'env', '.venv', 'trade_history']
        if any(skip_dir in root for skip_dir in skip_dirs):
            continue
        for file in files:
            if file.endswith('.csv'):
                # 检查文件名是否包含股票代码
                if code_number in file or stock_code in file or code_clean in file:
                    file_path = os.path.join(root, file)
                    # 确保是股票数据文件（包含日期、价格等列）
                    try:
                        df_test = pd.read_csv(file_path, nrows=1, encoding='utf-8-sig')
                        # 检查是否包含必要的列
                        required_cols = ['close', 'date']
                        if any(col in df_test.columns for col in required_cols):
                            return file_path
                    except Exception as e:
                        # 如果读取失败，尝试其他编码
                        try:
                            df_test = pd.read_csv(file_path, nrows=1, encoding='gbk')
                            required_cols = ['close', 'date']
                            if any(col in df_test.columns for col in required_cols):
                                return file_path
                        except:
                            continue
    
    return None

def calculate_sharpe_ratio(returns, risk_free_rate=0.03):
    """
    计算夏普比率（年化，假设252个交易日）
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 252
    sharpe = np.sqrt(252) * np.mean(excess_returns) / np.std(returns)
    return sharpe

def calculate_max_drawdown(net_worth_history):
    """
    计算最大回撤
    """
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
    """
    对单只股票进行回测
    """
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
            print(f"   ⚠️  未找到数据文件，尝试的路径包括:")
            code_parts = stock_code.split('.')
            code_number = code_parts[-1] if len(code_parts) > 1 else stock_code.replace('.', '')
            print(f"      - stockdata_v7_301389_600110/train/{stock_code}.*.csv")
            print(f"      - stockdata_v7_301389_600110/test/{stock_code}.*.csv")
            print(f"      - 以及其他可能的路径...")
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
            # 对于可转债等特殊标的，可能需要特殊处理
            # 先检查数据文件，看是否需要预处理
            df_check = pd.read_csv(data_file, nrows=10)
            # 检查关键列是否存在
            required_cols = ['date', 'close', 'open', 'high', 'low']
            missing_cols = [col for col in required_cols if col not in df_check.columns]
            if missing_cols:
                print(f"   ⚠️  数据文件缺少必要列: {missing_cols}")
                return {
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'status': 'failed',
                    'error': f'数据文件缺少必要列: {missing_cols}'
                }
            
            env = StockTradingEnv(data_file, initial_balance=INITIAL_BALANCE)
            print(f"   ✅ 环境创建成功，数据点: {len(env.df)}")
            
            # 检查数据是否足够
            if len(env.df) < 50:
                print(f"   ⚠️  数据点不足（{len(env.df)} < 50），可能影响回测准确性")
        except ValueError as e:
            if "数据不足" in str(e):
                print(f"   ⚠️  数据不足，尝试检查数据文件...")
                # 尝试读取数据文件，检查问题
                try:
                    df_test = pd.read_csv(data_file)
                    print(f"      原始数据行数: {len(df_test)}")
                    # 检查哪些列有NaN
                    nan_counts = df_test.isna().sum()
                    high_nan_cols = nan_counts[nan_counts > len(df_test) * 0.5].index.tolist()
                    if high_nan_cols:
                        print(f"      高NaN列: {high_nan_cols}")
                        print(f"      提示: 可转债等特殊标的可能缺少财务指标，这是正常的")
                except:
                    pass
            print(f"   ❌ 环境创建失败: {e}")
            return {
                'stock_code': stock_code,
                'stock_name': stock_name,
                'status': 'failed',
                'error': f'环境创建失败: {str(e)}'
            }
        except Exception as e:
            print(f"   ❌ 环境创建失败: {e}")
            import traceback
            traceback.print_exc()
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
        early_stop_reason = None
        while not done:
            # 使用模型预测动作
            action, _ = model.predict(obs, deterministic=True)
            
            # 执行动作
            obs, reward, done, truncated, info = env.step(action)
            
            # 记录净值
            net_worth_history.append(env.net_worth)
            
            # 计算日收益率
            if prev_net_worth > 0:
                daily_return = (env.net_worth - prev_net_worth) / prev_net_worth
                daily_returns.append(daily_return)
            
            prev_net_worth = env.net_worth
            step_count += 1
            
            # 检查是否提前结束（止损）
            if done and step_count < len(env.df) - 1:
                # 检查是否因为止损而结束
                drawdown_tolerance = getattr(env, 'drawdown_tolerance', 0.3)
                if env.net_worth < INITIAL_BALANCE * (1 - drawdown_tolerance):
                    early_stop_reason = f"触发止损（回撤超过{drawdown_tolerance*100:.0f}%）"
            
            if step_count % 50 == 0:
                print(f"      步骤 {step_count}/{len(env.df)}: 净值={env.net_worth:.2f}, 收益={(env.net_worth-INITIAL_BALANCE):.2f}")
        
        if early_stop_reason:
            print(f"   ⚠️  回测提前结束: {early_stop_reason}")
        print(f"   ✅ 回测完成，共 {step_count} 步（总数据点: {len(env.df)}）")
        
        # 计算指标
        print(f"   📊 计算回测指标...")
        
        # 总收益率
        final_net_worth = env.net_worth
        total_return = (final_net_worth - INITIAL_BALANCE) / INITIAL_BALANCE * 100
        
        # 年化收益率（假设252个交易日）
        trading_days = len(net_worth_history) - 1
        if trading_days > 0:
            annual_return = ((final_net_worth / INITIAL_BALANCE) ** (252 / trading_days) - 1) * 100
        else:
            annual_return = 0.0
        
        # 最大回撤
        max_drawdown_info = calculate_max_drawdown(net_worth_history)
        
        # 夏普比率
        if len(daily_returns) > 0:
            sharpe_ratio = calculate_sharpe_ratio(np.array(daily_returns))
        else:
            sharpe_ratio = 0.0
        
        # 构建结果
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
            'max_drawdown_info': max_drawdown_info
        }
        
        # 打印结果
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
    """
    主函数：执行所有股票的回测
    """
    print("="*70)
    print("两只股票回测脚本")
    print("="*70)
    print("\n目标股票:")
    for stock in TARGET_STOCKS:
        print(f"  - {stock['name']} ({stock['code']}) - 模型: {stock['model']}")
    print("="*70 + "\n")
    
    results = []
    
    for stock_info in TARGET_STOCKS:
        result = backtest_stock(stock_info)
        results.append(result)
    
    # 打印汇总
    print("\n" + "="*70)
    print("📊 回测结果汇总")
    print("="*70)
    print(f"\n{'股票名称':<10} | {'股票代码':<12} | {'总收益率':<10} | {'年化收益率':<10} | {'最大回撤':<10} | {'夏普比率':<10} | {'状态':<8}")
    print("-" * 90)
    
    for res in results:
        if res['status'] == 'success':
            print(f"{res['stock_name']:<10} | {res['stock_code']:<12} | {res['total_return']:<10.2f}% | {res['annual_return']:<10.2f}% | {res['max_drawdown_pct']:<10.2f}% | {res['sharpe_ratio']:<10.4f} | ✅ 成功")
        else:
            error_msg = res.get('error', '未知错误')
            print(f"{res['stock_name']:<10} | {res['stock_code']:<12} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | ❌ 失败       ({error_msg})")
    
    print("-" * 90)
    
    # 保存结果到JSON文件
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f"backtest_301389_600110_results_{timestamp}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n📁 详细结果已保存到: {result_file}")
    print("="*70)

if __name__ == '__main__':
    main()

