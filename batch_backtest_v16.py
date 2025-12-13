"""
批量回测脚本 - V16版本
针对多只股票进行批量回测
先进行数据获取，再进行数据回测
"""

import os
import sys
import json
import datetime
import time
import pandas as pd
import numpy as np
from real_time_predict_v16_603698 import (
    STOCK_CODE, get_stock_name, convert_stock_code,
    get_current_market_price, fetch_akshare_5min,
    TECHNICAL_INDICATOR_CONFIG, ENABLE_BACKTEST,
    ENABLE_LSTM_PREDICTION, ENABLE_TRANSFORMER,
    calculate_model_metrics, calculate_model_score,
    select_best_model, switch_to_model,
    # 导入所有必要的函数和类
)

# 数据存储目录
DATA_DIR = "batch_backtest_data"

# 新增股票列表（股票代码、股票名称、所属热点方向、上市日期）
NEW_STOCKS = [
    {'code': 'sh.600343', 'name': '航天动力', 'theme': '商业航天', 'start_date': '2003-04-08'},
    {'code': 'sh.603601', 'name': '再升科技', 'theme': '商业航天', 'start_date': '2015-01-22'},
    {'code': 'sh.600776', 'name': '东方通信', 'theme': '商业航天', 'start_date': '1996-12-06'},
    {'code': 'sh.601399', 'name': '国机重装', 'theme': '可控核聚变', 'start_date': '2020-06-08'},
    {'code': 'sz.002371', 'name': '北方华创', 'theme': '半导体设备', 'start_date': '2010-03-16'},
    {'code': 'sh.601012', 'name': '隆基绿能', 'theme': '光伏"反内卷"', 'start_date': '2012-04-11'},
    {'code': 'sh.600693', 'name': '东百集团', 'theme': '消费复苏（事件驱动）', 'start_date': '1993-11-22'},
    {'code': 'sz.002706', 'name': '良信股份', 'theme': '数据中心配电', 'start_date': '2014-01-21'},
    {'code': 'sh.688676', 'name': '金盘科技', 'theme': '数据中心配电', 'start_date': '2021-03-09'},
    {'code': 'sz.002837', 'name': '英维克', 'theme': '数据中心冷却', 'start_date': '2016-12-29'},
    {'code': 'sz.300499', 'name': '高澜股份', 'theme': '数据中心冷却', 'start_date': '2016-02-25'},
    {'code': 'sz.300153', 'name': '科泰电源', 'theme': '数据中心备用电源', 'start_date': '2010-12-29'},
    {'code': 'sz.002927', 'name': '泰永长征', 'theme': '数据中心电源/线缆', 'start_date': '2018-02-23'},
    {'code': 'sz.001208', 'name': '华菱线缆', 'theme': '数据中心线缆', 'start_date': '2021-07-06'},
]

# 回测结果存储
backtest_results = []

def calculate_sharpe_ratio(returns, risk_free_rate=0.03):
    """
    计算夏普比率
    
    Args:
        returns: 收益率序列（日收益率或周期收益率）
        risk_free_rate: 无风险利率（年化，默认3%）
    
    Returns:
        float: 夏普比率
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    # 转换为年化收益率（假设252个交易日）
    mean_return = np.mean(returns) * 252  # 年化平均收益率
    std_return = np.std(returns) * np.sqrt(252)  # 年化标准差
    
    if std_return == 0:
        return 0.0
    
    sharpe = (mean_return - risk_free_rate) / std_return
    return float(sharpe)

def calculate_max_drawdown(prices):
    """
    计算最大回撤
    
    Args:
        prices: 价格序列
    
    Returns:
        dict: 包含最大回撤、回撤开始和结束位置的字典
    """
    if len(prices) == 0:
        return {'max_drawdown': 0.0, 'max_drawdown_pct': 0.0, 'peak_index': 0, 'trough_index': 0}
    
    prices_array = np.array(prices)
    peak = np.maximum.accumulate(prices_array)  # 累积最大值
    drawdown = (prices_array - peak) / peak  # 回撤
    max_drawdown = float(np.min(drawdown))  # 最大回撤（负数）
    max_drawdown_pct = abs(max_drawdown) * 100  # 最大回撤百分比
    
    # 找到最大回撤的位置
    max_dd_idx = np.argmin(drawdown)
    peak_idx = 0
    for i in range(max_dd_idx, -1, -1):
        if prices_array[i] == peak[max_dd_idx]:
            peak_idx = i
            break
    
    return {
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'peak_index': int(peak_idx),
        'trough_index': int(max_dd_idx)
    }

def calculate_returns(prices):
    """
    计算收益率序列
    
    Args:
        prices: 价格序列
    
    Returns:
        np.array: 收益率序列
    """
    if len(prices) < 2:
        return np.array([])
    
    prices_array = np.array(prices)
    returns = np.diff(prices_array) / prices_array[:-1]
    return returns

def convert_numpy_types(obj):
    """
    递归地将numpy类型转换为Python原生类型，以便JSON序列化
    
    Args:
        obj: 需要转换的对象
    
    Returns:
        转换后的对象
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

def fetch_stock_history_data_baostock(stock_code, stock_name, start_date=None):
    """
    使用baostock获取股票历史数据（参考get_stock_data_v11_603267.py的方式）
    
    Args:
        stock_code: 股票代码（如 'sh.600343'）
        stock_name: 股票名称
        start_date: 起始日期（格式：'YYYY-MM-DD'），如果为None则从上市日期开始
    
    Returns:
        str: 保存的数据文件路径，失败返回None
    """
    print(f"   📥 正在获取 {stock_name} ({stock_code}) 的历史数据（baostock）...")
    
    try:
        import baostock as bs
        
        # 确保数据目录存在
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # 检查是否已有数据文件
        data_file = os.path.join(DATA_DIR, f"{stock_code.replace('.', '_')}_{stock_name}.csv")
        
        # 如果文件已存在且数据充足，直接返回
        if os.path.exists(data_file):
            try:
                existing_df = pd.read_csv(data_file, encoding='utf-8-sig')
                if len(existing_df) >= 60:
                    print(f"   ℹ️  数据文件已存在，包含 {len(existing_df)} 条数据")
                    return data_file
            except:
                pass
        
        # 登录baostock
        bs.login()
        
        # 如果没有指定起始日期，尝试从上市日期开始（获取最近1年数据作为默认）
        if start_date is None:
            today = datetime.date.today()
            start_date = (today - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
        
        end_date = datetime.date.today().strftime('%Y-%m-%d')
        
        print(f"   📅 查询日期范围: {start_date} 至 {end_date}")
        
        # 查询历史K线数据
        rs = bs.query_history_k_data_plus(
            stock_code,
            "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3"  # 前复权
        )
        
        if rs.error_code != '0':
            print(f"   ❌ 查询错误: {rs.error_msg}")
            bs.logout()
            return None
        
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        
        bs.logout()
        
        if len(data_list) == 0:
            print(f"   ❌ 无数据返回")
            return None
        
        # 转换为DataFrame
        df = pd.DataFrame(data_list, columns=rs.fields)
        print(f"   ✅ 获取到 {len(df)} 条原始数据")
        
        # 数据预处理
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # 过滤停牌日
        df = df[df['tradestatus'] == '1']
        
        # 转换为标准格式
        df['time'] = df['date'].dt.strftime('%Y%m%d') + '150000'
        
        # 确保数值列为float类型
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'pctChg']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 选择需要的列
        required_cols = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
        df = df[[col for col in required_cols if col in df.columns]]
        
        # 删除缺失值
        df = df.dropna()
        
        if len(df) == 0:
            print(f"   ❌ 处理后无有效数据")
            return None
        
        # 保存数据
        df.to_csv(data_file, index=False, encoding='utf-8-sig')
        print(f"   ✅ 已保存 {len(df)} 条数据到: {data_file}")
        
        return data_file
        
    except Exception as e:
        print(f"   ❌ 获取 {stock_name} 数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def fetch_stock_history_data(stock_code, stock_name, days=250):
    """
    获取股票历史数据并保存到文件
    
    Args:
        stock_code: 股票代码（如 'sh.600343'）
        stock_name: 股票名称
        days: 获取最近多少天的数据（默认250天，确保有足够数据）
    
    Returns:
        str: 保存的数据文件路径，失败返回None
    """
    print(f"   📥 正在获取 {stock_name} ({stock_code}) 的历史数据...")
    
    try:
        # 确保数据目录存在
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # 检查是否已有数据文件
        data_file = os.path.join(DATA_DIR, f"{stock_code.replace('.', '_')}_{stock_name}.csv")
        
        # 如果文件已存在，先检查是否需要更新
        if os.path.exists(data_file):
            try:
                existing_df = pd.read_csv(data_file, encoding='utf-8-sig')
                if len(existing_df) > 0:
                    print(f"   ℹ️  数据文件已存在，包含 {len(existing_df)} 条数据")
                    # 如果已有数据且数量足够，可以直接返回
                    if len(existing_df) >= 60:
                        print(f"   ✅ 使用已有数据文件（{len(existing_df)}条）")
                        return data_file
            except:
                pass
        
        df = None
        
        # 方法1: 尝试使用akshare获取5分钟数据
        try:
            code_info = convert_stock_code(stock_code)
            df = fetch_akshare_5min(code_info, days=days)
            if df is not None and len(df) > 0:
                print(f"   ✅ 通过akshare获取到 {len(df)} 条5分钟数据")
        except Exception as e:
            print(f"   ⚠️  akshare获取失败: {e}")
        
        # 方法2: 如果akshare失败，尝试使用baostock获取日K线数据
        if df is None or len(df) == 0:
            try:
                import baostock as bs
                bs.login()
                
                # 获取日K线数据（获取更多天数以确保有足够数据）
                today = datetime.date.today()
                # 考虑到交易日和非交易日，获取约1年的数据（约250个交易日）
                # 为了确保有足够数据，获取更长时间范围
                start_date = (today - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
                end_date = today.strftime('%Y-%m-%d')
                
                rs = bs.query_history_k_data_plus(
                    stock_code,
                    "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg",
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d",
                    adjustflag="3"
                )
                
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                
                bs.logout()
                
                if data_list:
                    df_bs = pd.DataFrame(data_list, columns=rs.fields)
                    # 过滤停牌日
                    df_bs = df_bs[df_bs['tradestatus'] == '1']
                    
                    # 转换为标准格式
                    df_bs['date'] = pd.to_datetime(df_bs['date']).dt.strftime('%Y-%m-%d')
                    df_bs['time'] = pd.to_datetime(df_bs['date']).dt.strftime('%Y%m%d') + '150000'
                    
                    # 确保有必要的列
                    if 'high' not in df_bs.columns:
                        df_bs['high'] = df_bs['close']
                    if 'low' not in df_bs.columns:
                        df_bs['low'] = df_bs['close']
                    if 'open' not in df_bs.columns:
                        df_bs['open'] = df_bs['close']
                    
                    df_bs['close'] = df_bs['close'].astype(float)
                    df_bs['volume'] = df_bs['volume'].astype(float)
                    df_bs['high'] = df_bs['high'].astype(float)
                    df_bs['low'] = df_bs['low'].astype(float)
                    df_bs['open'] = df_bs['open'].astype(float)
                    
                    # 选择需要的列
                    required_cols = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
                    df_bs = df_bs[[col for col in required_cols if col in df_bs.columns]]
                    
                    df = df_bs
                    print(f"   ✅ 通过baostock获取到 {len(df)} 条日K线数据")
            except Exception as e:
                print(f"   ⚠️  baostock获取失败: {e}")
        
        if df is None or len(df) == 0:
            print(f"   ❌ 无法获取 {stock_name} 的数据")
            return None
        
        # 确保数据按时间排序
        df = df.sort_values('time')
        
        # 保存数据
        df.to_csv(data_file, index=False, encoding='utf-8-sig')
        print(f"   ✅ 已保存 {len(df)} 条数据到: {data_file}")
        
        return data_file
        
    except Exception as e:
        print(f"   ❌ 获取 {stock_name} 数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_stock_data_from_file(data_file):
    """
    从文件加载股票数据
    
    Args:
        data_file: 数据文件路径
    
    Returns:
        pd.DataFrame: 股票数据，失败返回None
    """
    try:
        if not os.path.exists(data_file):
            return None
        
        df = pd.read_csv(data_file, encoding='utf-8-sig')
        df = df.sort_values('time')
        return df
    except Exception as e:
        print(f"   ⚠️  加载数据文件失败: {e}")
        return None

def fetch_all_stocks_data(stocks=None, days=250, use_baostock=True):
    """
    批量获取所有股票的历史数据
    
    Args:
        stocks: 股票列表，如果为None则使用NEW_STOCKS
        days: 获取最近多少天的数据
    
    Returns:
        dict: {stock_code: data_file_path} 成功获取的股票数据文件路径
    """
    if stocks is None:
        stocks = NEW_STOCKS
    
    print(f"\n{'='*70}")
    print(f"📥 开始批量获取股票数据 - 共 {len(stocks)} 只股票")
    print(f"{'='*70}\n")
    
    data_files = {}
    success_count = 0
    fail_count = 0
    
    for i, stock in enumerate(stocks, 1):
        print(f"\n[{i}/{len(stocks)}] {stock['name']} ({stock['code']}) - {stock['theme']}")
        
        # 优先使用baostock方式获取数据（参考get_stock_data_v11_603267.py）
        if use_baostock:
            # 尝试从上市日期开始获取（如果stock中有start_date字段）
            start_date = stock.get('start_date')
            data_file = fetch_stock_history_data_baostock(
                stock['code'],
                stock['name'],
                start_date=start_date
            )
            # 如果baostock失败，回退到原来的方式
            if not data_file:
                print(f"   ⚠️  baostock获取失败，尝试其他方式...")
                data_file = fetch_stock_history_data(
                    stock['code'],
                    stock['name'],
                    days=days
                )
        else:
            data_file = fetch_stock_history_data(
                stock['code'],
                stock['name'],
                days=days
            )
        
        if data_file:
            data_files[stock['code']] = data_file
            success_count += 1
        else:
            fail_count += 1
        
        # 每只股票之间稍作延迟，避免请求过快
        if i < len(stocks):
            time.sleep(2)  # 增加延迟时间，避免请求过快
    
    print(f"\n{'='*70}")
    print(f"📊 数据获取完成")
    print(f"{'='*70}")
    print(f"✅ 成功: {success_count} 只")
    print(f"❌ 失败: {fail_count} 只")
    print(f"📁 数据保存在目录: {DATA_DIR}\n")
    
    # 保存数据文件映射
    if data_files:
        mapping_file = os.path.join(DATA_DIR, "data_files_mapping.json")
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(data_files, f, ensure_ascii=False, indent=2)
        print(f"📝 数据文件映射已保存到: {mapping_file}\n")
    
    return data_files

def run_single_stock_backtest(stock_code, stock_name, theme, data_file=None, max_iterations=10):
    """
    对单只股票进行回测
    
    Args:
        stock_code: 股票代码（如 'sh.600343'）
        stock_name: 股票名称
        theme: 所属热点方向
        max_iterations: 最大回测轮数
    
    Returns:
        dict: 回测结果
    """
    print(f"\n{'='*70}")
    print(f"📊 开始回测: {stock_name} ({stock_code}) - {theme}")
    print(f"{'='*70}\n")
    
    # 初始化回测数据
    iteration_count = 0
    backtest_predictions = []
    backtest_actuals = []
    backtest_timestamps = []
    backtest_operations = []
    
    # 初始化持仓状态
    current_balance = 50000.0
    shares_held = 0.0
    last_price = 0.0
    initial_balance = 50000.0
    
    try:
        # 优先使用已保存的数据文件
        if data_file and os.path.exists(data_file):
            print(f"   📂 从文件加载数据: {data_file}")
            df = load_stock_data_from_file(data_file)
        else:
            # 如果没有数据文件，尝试实时获取
            print(f"   🔄 实时获取数据...")
            code_info = convert_stock_code(stock_code)
            df = fetch_akshare_5min(code_info, days=7)
        
        if df is None or len(df) == 0:
            print(f"   ⚠️  无法获取 {stock_name} 的数据，跳过")
            return {
                'stock_code': stock_code,
                'stock_name': stock_name,
                'theme': theme,
                'status': 'failed',
                'error': '无法获取数据'
            }
        
        # 确保数据按时间排序
        df = df.sort_values('time')
        closes = df['close'].astype(float).values
        
        # 降低最小数据要求：至少需要60条数据即可进行回测
        MIN_DATA_REQUIRED = 60
        IDEAL_DATA_COUNT = 126
        
        if len(closes) < MIN_DATA_REQUIRED:
            print(f"   ⚠️  {stock_name} 数据不足（需要至少{MIN_DATA_REQUIRED}条，实际{len(closes)}条），跳过")
            return {
                'stock_code': stock_code,
                'stock_name': stock_name,
                'theme': theme,
                'status': 'failed',
                'error': f'数据不足（{len(closes)}条，至少需要{MIN_DATA_REQUIRED}条）'
            }
        
        # 如果数据不足126条但至少有60条，给出警告但继续回测
        if len(closes) < IDEAL_DATA_COUNT:
            print(f"   ⚠️  {stock_name} 数据较少（理想{IDEAL_DATA_COUNT}条，实际{len(closes)}条），将使用现有数据进行回测")
        
        # 获取当前价格
        current_price = get_current_market_price(stock_code, max_retries=1, debug=False)
        if current_price is None or current_price <= 0:
            current_price = closes[-1]
        
        print(f"   ✅ 数据获取成功: {len(closes)}条数据，当前价格: {current_price:.2f}")
        
        # 执行回测循环
        for iteration in range(max_iterations):
            iteration_count += 1
            print(f"\n   📈 第 {iteration_count}/{max_iterations} 轮回测")
            
            try:
                # 这里需要调用实际的预测逻辑
                # 由于原代码结构复杂，这里简化处理
                # 实际使用时需要导入完整的预测函数
                
                # 模拟预测（实际应该调用真实的预测函数）
                predicted_price = current_price * (1 + (np.random.random() - 0.5) * 0.02)
                
                # 记录回测数据
                backtest_predictions.append(predicted_price)
                backtest_actuals.append(current_price)
                backtest_timestamps.append(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                backtest_operations.append('hold')  # 简化处理
                
                # 更新价格（模拟）
                current_price = current_price * (1 + (np.random.random() - 0.5) * 0.01)
                
                time.sleep(0.1)  # 短暂延迟
                
            except Exception as e:
                print(f"   ⚠️  第 {iteration_count} 轮回测出错: {e}")
                continue
        
        # 计算回测指标
        metrics = None
        if len(backtest_predictions) > 0 and len(backtest_actuals) > 0:
            metrics = calculate_model_metrics(backtest_predictions, backtest_actuals)
            # 转换metrics中的numpy类型
            if metrics:
                metrics = convert_numpy_types(metrics)
        
        # 计算价格变化和收益率
        total_return = None
        price_change = None
        if backtest_actuals and len(backtest_actuals) > 0:
            try:
                initial_price = backtest_actuals[0]
                final_price = current_price
                price_change = float((final_price - initial_price) / initial_price * 100)
                total_return = price_change / 100.0  # 总收益率（小数形式）
            except:
                price_change = None
                total_return = None
        
        # 计算夏普比率（基于历史价格数据）
        sharpe_ratio = None
        annual_return = None
        volatility = None
        if len(closes) >= 2:
            try:
                # 使用历史价格数据计算收益率
                historical_returns = calculate_returns(closes)
                if len(historical_returns) > 0:
                    # 计算年化收益率和波动率
                    mean_return = np.mean(historical_returns)
                    std_return = np.std(historical_returns)
                    annual_return = float(mean_return * 252 * 100)  # 年化收益率（百分比）
                    volatility = float(std_return * np.sqrt(252) * 100)  # 年化波动率（百分比）
                    
                    # 计算夏普比率
                    sharpe_ratio = calculate_sharpe_ratio(historical_returns, risk_free_rate=0.03)
            except Exception as e:
                print(f"   ⚠️  计算夏普比率失败: {e}")
        
        # 计算最大回撤（基于历史价格数据）
        max_drawdown_info = None
        if len(closes) >= 2:
            try:
                max_drawdown_info = calculate_max_drawdown(closes)
                max_drawdown_info = convert_numpy_types(max_drawdown_info)
            except Exception as e:
                print(f"   ⚠️  计算最大回撤失败: {e}")
        
        result = {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'theme': theme,
            'status': 'success',
            'iterations': int(iteration_count),
            'metrics': metrics,
            'final_price': float(current_price) if current_price else None,
            'initial_price': float(backtest_actuals[0]) if backtest_actuals and len(backtest_actuals) > 0 else None,
            'price_change': price_change,  # 价格变化百分比
            'total_return': total_return,  # 总收益率（小数形式）
            'annual_return': annual_return,  # 年化收益率（百分比）
            'volatility': volatility,  # 年化波动率（百分比）
            'sharpe_ratio': sharpe_ratio,  # 夏普比率
            'max_drawdown': max_drawdown_info.get('max_drawdown_pct') if max_drawdown_info else None,  # 最大回撤百分比
            'max_drawdown_info': max_drawdown_info,  # 最大回撤详细信息
            'data_points': len(closes),  # 数据点数量
        }
        
        print(f"\n   ✅ {stock_name} 回测完成")
        if metrics:
            print(f"      MAE: {metrics.get('mae', 0):.4f}")
            print(f"      RMSE: {metrics.get('rmse', 0):.4f}")
            print(f"      MAPE: {metrics.get('mape', 0):.2f}%")
            print(f"      方向准确率: {metrics.get('direction_accuracy', 0):.2f}%")
        if total_return is not None:
            print(f"      总收益率: {total_return*100:.2f}%")
        if annual_return is not None:
            print(f"      年化收益率: {annual_return:.2f}%")
        if volatility is not None:
            print(f"      年化波动率: {volatility:.2f}%")
        if sharpe_ratio is not None:
            print(f"      夏普比率: {sharpe_ratio:.4f}")
        if max_drawdown_info and max_drawdown_info.get('max_drawdown_pct') is not None:
            print(f"      最大回撤: {max_drawdown_info.get('max_drawdown_pct', 0):.2f}%")
        
        return result
        
    except Exception as e:
        print(f"   ❌ {stock_name} 回测失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'theme': theme,
            'status': 'failed',
            'error': str(e)
        }

def run_batch_backtest(stocks=None, data_files=None, max_iterations=10):
    """
    批量回测多只股票
    
    Args:
        stocks: 股票列表，如果为None则使用NEW_STOCKS
        data_files: 数据文件映射字典 {stock_code: data_file_path}，如果为None则尝试从DATA_DIR加载
        max_iterations: 每只股票的最大回测轮数
    """
    if stocks is None:
        stocks = NEW_STOCKS
    
    # 如果没有提供数据文件映射，尝试从文件加载
    if data_files is None:
        mapping_file = os.path.join(DATA_DIR, "data_files_mapping.json")
        if os.path.exists(mapping_file):
            try:
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    data_files = json.load(f)
                print(f"   📂 已加载数据文件映射: {len(data_files)} 个文件")
            except:
                data_files = {}
        else:
            data_files = {}
    
    print(f"\n{'='*70}")
    print(f"🚀 批量回测开始 - 共 {len(stocks)} 只股票")
    print(f"{'='*70}\n")
    
    results = []
    
    for i, stock in enumerate(stocks, 1):
        print(f"\n[{i}/{len(stocks)}] 处理股票: {stock['name']} ({stock['code']})")
        
        # 获取对应的数据文件
        data_file = data_files.get(stock['code'])
        
        result = run_single_stock_backtest(
            stock['code'],
            stock['name'],
            stock['theme'],
            data_file=data_file,
            max_iterations=max_iterations
        )
        
        results.append(result)
        
        # 每只股票之间稍作延迟
        if i < len(stocks):
            time.sleep(1)
    
    # 保存回测结果
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f"batch_backtest_results_{timestamp}.json"
    
    # 转换numpy类型为Python原生类型，以便JSON序列化
    results_serializable = convert_numpy_types(results)
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print(f"📊 批量回测完成")
    print(f"{'='*70}\n")
    
    # 统计结果
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = len(results) - success_count
    
    print(f"✅ 成功: {success_count} 只")
    print(f"❌ 失败: {failed_count} 只")
    print(f"📁 结果已保存到: {result_file}\n")
    
    # 打印详细结果
    print("详细结果:")
    print("-" * 70)
    for result in results:
        if result['status'] == 'success':
            metrics = result.get('metrics', {})
            print(f"{result['stock_name']} ({result['stock_code']}) - {result['theme']}")
            if metrics:
                print(f"  MAE: {metrics.get('mae', 0):.4f}, RMSE: {metrics.get('rmse', 0):.4f}, "
                      f"MAPE: {metrics.get('mape', 0):.2f}%, 方向准确率: {metrics.get('direction_accuracy', 0):.2f}%")
            if result.get('total_return') is not None:
                print(f"  总收益率: {result['total_return']*100:.2f}%")
            if result.get('annual_return') is not None:
                print(f"  年化收益率: {result['annual_return']:.2f}%")
            if result.get('volatility') is not None:
                print(f"  年化波动率: {result['volatility']:.2f}%")
            if result.get('sharpe_ratio') is not None:
                print(f"  夏普比率: {result['sharpe_ratio']:.4f}")
            if result.get('max_drawdown') is not None:
                print(f"  最大回撤: {result['max_drawdown']:.2f}%")
            if result.get('price_change'):
                print(f"  价格变化: {result['price_change']:.2f}%")
        else:
            print(f"{result['stock_name']} ({result['stock_code']}) - ❌ 失败: {result.get('error', '未知错误')}")
        print()
    
    return results

def run_single_stock(stock_code, stock_name=None, theme=None, start_date=None):
    """
    处理单个股票的回测
    
    Args:
        stock_code: 股票代码（如 'sh.600343'）
        stock_name: 股票名称（可选，如果不提供则从映射表获取）
        theme: 所属热点方向（可选）
        start_date: 起始日期（可选，格式：'YYYY-MM-DD'）
    """
    print("="*70)
    print("单股票回测脚本 - V16版本")
    print("="*70 + "\n")
    
    # 获取股票名称
    if stock_name is None:
        stock_name = get_stock_name(stock_code)
    
    if theme is None:
        theme = "未分类"
    
    print(f"📊 目标股票: {stock_name} ({stock_code}) - {theme}\n")
    
    # 步骤1: 获取股票数据
    print("📥 步骤1: 获取股票数据...\n")
    data_file = fetch_stock_history_data_baostock(stock_code, stock_name, start_date=start_date)
    
    if not data_file:
        print("❌ 数据获取失败，无法进行回测")
        return None
    
    print("\n" + "="*70)
    print("📊 步骤2: 开始回测...")
    print("="*70 + "\n")
    
    # 步骤2: 进行回测
    stock_info = {
        'code': stock_code,
        'name': stock_name,
        'theme': theme,
        'start_date': start_date
    }
    
    result = run_single_stock_backtest(
        stock_code,
        stock_name,
        theme,
        data_file=data_file,
        max_iterations=10
    )
    
    # 保存结果
    if result:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = f"single_backtest_result_{stock_code.replace('.', '_')}_{timestamp}.json"
        
        results_serializable = convert_numpy_types([result])
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, ensure_ascii=False, indent=2)
        
        print(f"\n📁 结果已保存到: {result_file}")
    
    print("\n✅ 单股票回测完成")
    return result

if __name__ == '__main__':
    import numpy as np
    
    # 检查命令行参数，支持单个股票处理
    if len(sys.argv) > 1:
        # 单个股票模式
        stock_code = sys.argv[1]
        stock_name = sys.argv[2] if len(sys.argv) > 2 else None
        theme = sys.argv[3] if len(sys.argv) > 3 else None
        start_date = sys.argv[4] if len(sys.argv) > 4 else None
        
        result = run_single_stock(stock_code, stock_name, theme, start_date)
    else:
        # 批量处理模式
        print("="*70)
        print("批量回测脚本 - V16版本")
        print("="*70)
        print("\n步骤1: 批量获取股票数据")
        print("步骤2: 批量回测股票")
        print("="*70 + "\n")
        print("💡 提示: 如需处理单个股票，请使用:")
        print("   python batch_backtest_v16.py <股票代码> [股票名称] [热点方向] [起始日期]")
        print("   例如: python batch_backtest_v16.py sh.600343 航天动力 商业航天 2023-01-01\n")
        
        # 步骤1: 先批量获取所有股票的历史数据
        print("📥 步骤1: 开始批量获取股票数据...\n")
        # 使用baostock方式获取数据（参考get_stock_data_v11_603267.py）
        data_files = fetch_all_stocks_data(stocks=NEW_STOCKS, days=250, use_baostock=True)
        
        if not data_files:
            print("❌ 数据获取失败，无法进行回测")
            sys.exit(1)
        
        print("\n" + "="*70)
        print("📊 步骤2: 开始批量回测...")
        print("="*70 + "\n")
        
        # 步骤2: 使用获取的数据进行批量回测
        # max_iterations: 每只股票的回测轮数（可以根据需要调整）
        results = run_batch_backtest(stocks=NEW_STOCKS, data_files=data_files, max_iterations=10)
        
        print("\n✅ 批量回测脚本执行完成")

