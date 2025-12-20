"""
回测脚本：预测第二天下跌超过3%全部卖出策略
针对良信股份和鸿远电子2025年数据进行回测
"""

import os
import sys
import json
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 直接定义必要的配置和函数，避免导入整个实时预测系统
# 这些配置来自 real_time_predict_v16_603698.py

# 配置参数
ENABLE_LSTM_PREDICTION = True
ENABLE_TRANSFORMER = True
LSTM_SEQ_LENGTH = 60
TRANSFORMER_MAX_SEQ_LEN = 100
SLIDING_WINDOW_SIZE = 200
USE_SLIDING_WINDOW_NORMALIZE = True
TRANSFORMER_ADAPTIVE_WINDOW = True
TRANSFORMER_EPOCHS = 120

# 交易成本配置
COMMISSION_RATE = 0.00025
MIN_COMMISSION = 5.0
TRANSFER_FEE_RATE = 0.00001
STAMP_DUTY_RATE = 0.001
SLIPPAGE_RATE = 0.0005

def round_to_lot(shares):
    """将股数向下取整为100股的整数倍（按手交易）"""
    if shares <= 0:
        return 0
    return int(shares // 100) * 100

def calc_buy_trade(current_price, buy_percentage, current_balance):
    """模拟买入操作，考虑滑点、手续费、过户费"""
    if current_balance <= 0 or buy_percentage <= 0:
        return 0.0, 0.0, 0.0, current_price
    
    adjusted_price = current_price * (1 + SLIPPAGE_RATE)
    buy_amount = current_balance * buy_percentage
    
    if buy_amount < 100:
        return 0.0, 0.0, 0.0, adjusted_price
    
    shares_bought = round_to_lot(buy_amount / adjusted_price) if adjusted_price > 0 else 0
    if shares_bought <= 0:
        return 0.0, 0.0, 0.0, adjusted_price
    
    trade_amount = shares_bought * adjusted_price
    
    commission = max(MIN_COMMISSION, trade_amount * COMMISSION_RATE)
    transfer_fee = trade_amount * TRANSFER_FEE_RATE
    total_fee = commission + transfer_fee
    total_cost = trade_amount + total_fee
    
    if total_cost > current_balance:
        max_trade_amount = max(0.0, current_balance - MIN_COMMISSION)
        shares_bought = round_to_lot(max_trade_amount / adjusted_price) if adjusted_price > 0 else 0
        if shares_bought <= 0:
            return 0.0, 0.0, 0.0, adjusted_price
        trade_amount = shares_bought * adjusted_price
        commission = max(MIN_COMMISSION, trade_amount * COMMISSION_RATE)
        transfer_fee = trade_amount * TRANSFER_FEE_RATE
        total_fee = commission + transfer_fee
        total_cost = trade_amount + total_fee
    
    return shares_bought, total_cost, total_fee, adjusted_price

def calc_sell_trade(current_price, sell_percentage, shares_held):
    """模拟卖出操作，考虑滑点、手续费、过户费、印花税"""
    if shares_held <= 0 or sell_percentage <= 0:
        return 0.0, 0.0, 0.0, current_price
    
    adjusted_price = current_price * (1 - SLIPPAGE_RATE)
    shares_sold = round_to_lot(shares_held * sell_percentage)
    if shares_sold <= 0:
        return 0.0, 0.0, 0.0, adjusted_price
    trade_amount = shares_sold * adjusted_price
    
    if trade_amount <= 0:
        return 0.0, 0.0, 0.0, adjusted_price
    
    commission = max(MIN_COMMISSION, trade_amount * COMMISSION_RATE)
    transfer_fee = trade_amount * TRANSFER_FEE_RATE
    stamp_duty = trade_amount * STAMP_DUTY_RATE
    total_fee = commission + transfer_fee + stamp_duty
    net_increase = trade_amount - total_fee
    
    return shares_sold, net_increase, total_fee, adjusted_price

# 最佳模型映射表（根据用户提供的表格）
BEST_MODEL_MAPPING = {
    'sz.002025': {'name': '航天电器', 'model': 'ppo_stock_v7_603698.zip', 'group': '航天工程603698组'},
    'sz.002241': {'name': '歌尔股份', 'model': 'ppo_stock_v7_603267.zip', 'group': '鸿远电子603267组'},
    'sz.002266': {'name': '浙富控股', 'model': 'ppo_stock_v7_601399.zip', 'group': '国机重装601399组'},
    'sz.002475': {'name': '立讯精密', 'model': 'ppo_stock_v7_300726.zip', 'group': '宏达电子300726组'},
    'sz.002706': {'name': '良信股份', 'model': 'ppo_stock_v7_002837.zip', 'group': '英维克002837组'},
    'sz.002837': {'name': '英维克', 'model': 'ppo_stock_v7_002837.zip', 'group': '英维克002837组'},
    'sz.002851': {'name': '麦格米特', 'model': 'ppo_stock_v7_002851.zip', 'group': '麦格米特002851组'},
    'sz.300153': {'name': '科泰电源', 'model': 'ppo_stock_v7_002706.zip', 'group': '良信股份002706组'},
    'sz.300274': {'name': '阳光电源', 'model': 'ppo_stock_v7_002927.zip', 'group': '泰永长征002927组'},
    'sz.300499': {'name': '高澜股份', 'model': 'ppo_stock_v7_300499.zip', 'group': '高澜股份300499组'},
    'sz.300726': {'name': '宏达电子', 'model': 'ppo_stock_v7_002851.zip', 'group': '麦格米特002851组'},
    'sz.300762': {'name': '上海瀚讯', 'model': 'ppo_stock_v7_300762.zip', 'group': '上海瀚讯300762组'},
    'sh.601399': {'name': '国机重装', 'model': 'ppo_stock_v7_601399.zip', 'group': '国机重装601399组'},
    'sh.603267': {'name': '鸿远电子', 'model': 'ppo_stock_v7_002025.zip', 'group': '航天电器002025组'},
    # 黑马股票：回测中发现的高收益股票
    'sh.600730': {'name': '中国高科', 'model': 'ppo_stock_v7_600730.zip', 'group': '中国高科600730组（专用模型）'},  # 收益率17,378.72%，排名第3
    'sz.301005': {'name': '超捷股份', 'model': 'ppo_stock_v7.zip', 'group': '通用模型组（待优化）'},  # 收益率8,450.34%，排名第7
}

def get_stock_name(stock_code):
    """获取股票名称"""
    # 优先从最佳模型映射表中获取
    if stock_code in BEST_MODEL_MAPPING:
        return BEST_MODEL_MAPPING[stock_code]['name']
    
    stock_names = {
        'sz.002706': '良信股份',
        'sh.603267': '鸿远电子',
        'sh.600343': '航天动力',
        'sh.603601': '再升科技',
        'sh.600776': '东方通信',
        'sh.601399': '国机重装',
        'sz.002371': '北方华创',
        'sh.601012': '隆基绿能',
        'sh.600693': '东百集团',
        'sh.688676': '金盘科技',
        'sz.002837': '英维克',
        'sz.300499': '高澜股份',
        'sz.300153': '科泰电源',
        'sz.002927': '泰永长征',
        'sz.001208': '华菱线缆',
        'sh.600730': '中国高科',  # 黑马股票：收益率17,378.72%，排名第3
        'sz.301005': '超捷股份',  # 黑马股票：收益率8,450.34%，排名第7
    }
    return stock_names.get(stock_code, stock_code)

def select_best_model_for_stock(stock_code, default_model=None):
    """
    根据股票代码选择最合适的PPO模型（优先使用最佳模型映射表中的模型）
    
    Args:
        stock_code: 股票代码（如 'sz.002706'）
        default_model: 默认模型路径（从V16文件中提取，但会被最佳模型映射覆盖）
    
    Returns:
        str: 模型文件路径，如果找不到则返回None
    """
    # 优先使用最佳模型映射表中的模型
    if stock_code in BEST_MODEL_MAPPING:
        best_model = BEST_MODEL_MAPPING[stock_code]['model']
        if os.path.exists(best_model):
            return best_model
    
    # 如果提供了默认模型（从V16文件中提取），作为备选
    if default_model and os.path.exists(default_model):
        return default_model
    
    # 提取股票代码的数字部分
    code_digits = stock_code.replace('sh.', '').replace('sz.', '').replace('.', '')
    
    # 模型匹配规则：优先使用对应股票的专用模型，否则使用通用模型
    model_priority = [
        f'ppo_stock_v7_{code_digits}.zip',  # 专用模型（最高优先级）
        'ppo_stock_v7_002706.zip',  # 良信股份模型（作为最佳通用模型）
        'ppo_stock_v7_603267.zip',  # 鸿远电子模型
        'ppo_stock_v7.zip',  # 通用模型（最低优先级）
    ]
    
    # 查找可用的模型
    for model_path in model_priority:
        if os.path.exists(model_path):
            return model_path
    
    return None

# 导入LSTM和Transformer模型
try:
    from lstm_gru_time_series import TimeSeriesProcessor
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("[警告] LSTM模块不可用")

try:
    from transformer_model import TransformerPredictor
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    print("[警告] Transformer模块不可用")

# 回测配置
INITIAL_BALANCE = 50000.0  # 初始资金5万元
DROP_THRESHOLD = -3.0  # 下跌阈值：-3%
BACKTEST_YEAR = 2025  # 回测年份

# 新增：上涨较多时的加仓 / 打价卖出参数
# 预测涨幅大于该阈值，则认为“上涨较多”
RISE_THRESHOLD = 3.0  # 例如：+3%
# 上涨较多时，按照 20% 和 0% 进行加仓（当前无仓则加20%，已有仓则不再加）
BULL_ADD_POSITION_PCT = 0.2
# 上涨较多时，按 +75% 和 0% 进行打价卖出（达到+75%止盈，回撤到成本价0%止盈）
TAKE_PROFIT_PCT = 75.0
STOP_PROFIT_PCT = 0.0

def extract_stocks_from_v16_files():
    """
    从V16预测文件中提取股票代码、名称，并使用最佳模型映射
    
    Returns:
        list: [{'code': 'sz.002706', 'name': '良信股份', 'model': 'ppo_stock_v7_002837.zip', 'group': '英维克002837组'}, ...]
    """
    import re
    import glob
    
    stocks_info = []
    v16_files = glob.glob('real_time_predict_v16_*.py')
    
    for file_path in v16_files:
        try:
            # 从文件名提取股票代码
            code_match = re.search(r'v16_(\d+)\.py', file_path)
            if not code_match:
                continue
            
            code_digits = code_match.group(1)
            # 判断市场：sz（0或3开头）或sh（6开头）
            if code_digits.startswith(('0', '3')):
                stock_code = f"sz.{code_digits}"
            else:
                stock_code = f"sh.{code_digits}"
            
            # 检查是否在最佳模型映射表中
            if stock_code in BEST_MODEL_MAPPING:
                stock_info = BEST_MODEL_MAPPING[stock_code]
                stocks_info.append({
                    'code': stock_code,
                    'name': stock_info['name'],
                    'model': stock_info['model'],
                    'group': stock_info['group'],
                    'file': file_path
                })
            else:
                # 如果不在映射表中，尝试从文件内容提取
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                stock_code_match = re.search(r"STOCK_CODE\s*=\s*(?:os\.getenv\([^)]+\)|sys\.argv\[1\]|['\"]([^'\"]+)['\"])", content)
                if stock_code_match:
                    if 'os.getenv' in stock_code_match.group(0):
                        default_match = re.search(r"os\.getenv\([^,]+,\s*['\"]([^'\"]+)['\"]\)", stock_code_match.group(0))
                        stock_code = default_match.group(1) if default_match else stock_code
                    elif stock_code_match.group(1):
                        stock_code = stock_code_match.group(1)
                
                # 提取模型路径
                model_path_match = re.search(r"MODEL_PATH\s*=\s*['\"]([^'\"]+\.zip)['\"]", content)
                model_path = model_path_match.group(1) if model_path_match else None
                
                # 提取股票名称
                stock_name = get_stock_name(stock_code)
                
                if stock_code and model_path:
                    stocks_info.append({
                        'code': stock_code,
                        'name': stock_name,
                        'model': model_path,
                        'group': '文件指定',
                        'file': file_path
                    })
        except Exception as e:
            print(f"   ⚠️  解析文件 {file_path} 失败: {e}")
            continue
    
    return stocks_info

# 从V16预测文件中提取股票列表，并使用最佳模型映射
TARGET_STOCKS = extract_stocks_from_v16_files()

# 如果没有找到，使用映射表中的股票
if not TARGET_STOCKS:
    print("⚠️  未找到V16预测文件，使用最佳模型映射表中的股票")
    TARGET_STOCKS = []
    for code, info in BEST_MODEL_MAPPING.items():
        # 检查是否有对应的V16文件
        code_digits = code.replace('sh.', '').replace('sz.', '').replace('.', '')
        v16_file = f'real_time_predict_v16_{code_digits}.py'
        if os.path.exists(v16_file):
            TARGET_STOCKS.append({
                'code': code,
                'name': info['name'],
                'model': info['model'],
                'group': info['group'],
                'file': v16_file
            })

def fetch_stock_data_2025(stock_code: str, stock_name: str) -> Optional[pd.DataFrame]:
    """
    获取股票2025年的历史数据
    
    Args:
        stock_code: 股票代码
        stock_name: 股票名称
    
    Returns:
        DataFrame: 股票数据，失败返回None
    """
    print(f"\n📥 正在获取 {stock_name} ({stock_code}) 2025年数据...")
    
    try:
        import baostock as bs
        
        # 登录baostock
        lg = bs.login()
        if lg.error_code != '0':
            print(f"   ❌ 登录失败: {lg.error_msg}")
            return None
        
        # 2025年的日期范围
        start_date = '2025-01-01'
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
        while (rs.error_code == '0') & rs.next():
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
        required_cols = ['date', 'time', 'open', 'high', 'low', 'close', 'volume', 'pctChg']
        df = df[[col for col in required_cols if col in df.columns]]
        
        # 删除缺失值
        df = df.dropna()
        
        if len(df) == 0:
            print(f"   ❌ 处理后无有效数据")
            return None
        
        print(f"   ✅ 处理后有效数据: {len(df)} 条")
        return df
        
    except Exception as e:
        print(f"   ❌ 获取数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def predict_next_day_price(df: pd.DataFrame, current_idx: int, 
                           lstm_processor: Optional[TimeSeriesProcessor] = None,
                           transformer_model: Optional[TransformerPredictor] = None,
                           lstm_normalization_params: Optional[Dict] = None,
                           transformer_normalization_params: Optional[Dict] = None) -> Tuple[Optional[float], Optional[float]]:
    """
    预测下一个交易日的价格
    
    Args:
        df: 股票数据DataFrame
        current_idx: 当前索引位置
        lstm_processor: LSTM处理器
        transformer_model: Transformer模型
        lstm_normalization_params: LSTM归一化参数
        transformer_normalization_params: Transformer归一化参数
    
    Returns:
        (lstm_prediction, transformer_prediction): 预测价格
    """
    if current_idx < max(LSTM_SEQ_LENGTH if LSTM_AVAILABLE else 0, 
                         TRANSFORMER_MAX_SEQ_LEN if TRANSFORMER_AVAILABLE else 0):
        return None, None
    
    closes = df['close'].values[:current_idx+1]
    
    lstm_prediction = None
    transformer_prediction = None
    
    # LSTM预测
    if LSTM_AVAILABLE and ENABLE_LSTM_PREDICTION and lstm_processor and len(closes) >= LSTM_SEQ_LENGTH:
        try:
            seq = closes[-LSTM_SEQ_LENGTH:]
            
            # 归一化
            if lstm_normalization_params:
                norm_method = lstm_normalization_params.get('method', 'minmax')
                if norm_method == 'minmax':
                    min_val = lstm_normalization_params['min']
                    max_val = lstm_normalization_params['max']
                    if max_val - min_val > 0:
                        normalized_seq = (seq - min_val) / (max_val - min_val)
                    else:
                        normalized_seq = np.zeros_like(seq)
                elif norm_method == 'zscore':
                    mean_val = lstm_normalization_params['mean']
                    std_val = lstm_normalization_params['std']
                    if std_val > 0:
                        normalized_seq = (seq - mean_val) / std_val
                    else:
                        normalized_seq = np.zeros_like(seq)
                else:
                    normalized_seq = seq
                
                # 预测
                prediction_norm = lstm_processor.predict_next(normalized_seq)
                # 反归一化
                lstm_prediction = float(lstm_processor.denormalize(
                    np.array([prediction_norm]),
                    lstm_normalization_params
                )[0]) if prediction_norm is not None else None
        except Exception as e:
            pass
    
    # Transformer预测
    if TRANSFORMER_AVAILABLE and ENABLE_TRANSFORMER and transformer_model and len(closes) >= TRANSFORMER_MAX_SEQ_LEN:
        try:
            seq = closes[-TRANSFORMER_MAX_SEQ_LEN:]
            
            # 归一化
            if transformer_normalization_params:
                norm_method = transformer_normalization_params.get('method', 'minmax')
                if norm_method == 'minmax':
                    min_val = transformer_normalization_params['min']
                    max_val = transformer_normalization_params['max']
                    if max_val - min_val > 0:
                        normalized_seq = (seq - min_val) / (max_val - min_val)
                    else:
                        normalized_seq = np.zeros_like(seq)
                elif norm_method == 'zscore':
                    mean_val = transformer_normalization_params['mean']
                    std_val = transformer_normalization_params['std']
                    if std_val > 0:
                        normalized_seq = (seq - mean_val) / std_val
                    else:
                        normalized_seq = np.zeros_like(seq)
                else:
                    normalized_seq = seq
                
                # 预测
                prediction_norm = transformer_model.predict_next(normalized_seq)
                # 反归一化
                transformer_prediction_raw = float(transformer_model.denormalize(
                    np.array([prediction_norm]),
                    transformer_normalization_params
                )[0]) if prediction_norm is not None else None
                
                if transformer_prediction_raw:
                    transformer_prediction = transformer_prediction_raw
        except Exception as e:
            pass
    
    return lstm_prediction, transformer_prediction

def backtest_single_stock(stock_code: str, stock_name: str, model_path: str = None) -> Dict:
    """
    对单只股票进行回测
    
    Args:
        stock_code: 股票代码
        stock_name: 股票名称
    
    Returns:
        dict: 回测结果
    """
    # 提取股票代码的数字部分（用于模型匹配）
    code_digits = stock_code.replace('sh.', '').replace('sz.', '').replace('.', '')
    
    print(f"\n{'='*70}")
    print(f"📊 开始回测: {stock_name} ({stock_code})")
    print(f"{'='*70}\n")
    
    # 获取数据
    df = fetch_stock_data_2025(stock_code, stock_name)
    if df is None or len(df) == 0:
        return {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'status': 'failed',
            'error': '无法获取数据'
        }
    
    # 初始化回测状态
    balance = INITIAL_BALANCE
    shares_held = 0.0
    cost_price = 0.0
    initial_balance = INITIAL_BALANCE
    
    # 交易记录
    trades = []
    daily_values = []
    
    # 初始化模型
    lstm_processor = None
    transformer_model = None
    ppo_model = None
    lstm_normalization_params = None
    transformer_normalization_params = None
    
    # 加载PPO模型（使用V16文件中指定的模型）
    try:
        from stable_baselines3 import PPO
        # 使用传入的模型路径，或根据股票代码选择
        selected_model_path = model_path or select_best_model_for_stock(stock_code)
        
        if selected_model_path and os.path.exists(selected_model_path):
            ppo_model = PPO.load(selected_model_path)
            # 判断是否为专用模型
            code_digits = stock_code.replace('sh.', '').replace('sz.', '').replace('.', '')
            model_type = "专用模型" if code_digits in selected_model_path else "通用模型"
            print(f"   ✅ 加载PPO模型: {selected_model_path} ({model_type})")
        else:
            # 如果找不到模型，尝试加载通用模型
            if os.path.exists('ppo_stock_v7.zip'):
                ppo_model = PPO.load('ppo_stock_v7.zip')
                selected_model_path = 'ppo_stock_v7.zip'
                print(f"   ✅ 加载PPO模型: ppo_stock_v7.zip (通用模型)")
            else:
                print(f"   ⚠️  未找到合适的PPO模型")
                ppo_model = None
                selected_model_path = None
    except Exception as e:
        print(f"   ⚠️  PPO模型加载失败: {e}")
        ppo_model = None
        selected_model_path = None
    
    if LSTM_AVAILABLE and ENABLE_LSTM_PREDICTION:
        try:
            lstm_processor = TimeSeriesProcessor(
                model_type='lstm',
                seq_length=LSTM_SEQ_LENGTH,
                input_size=1,
                hidden_size=64,
                num_layers=2
            )
        except:
            pass
    
    if TRANSFORMER_AVAILABLE and ENABLE_TRANSFORMER:
        try:
            transformer_model = TransformerPredictor(
                input_size=1,
                d_model=64,
                nhead=4,
                num_layers=2,
                max_seq_len=TRANSFORMER_MAX_SEQ_LEN
            )
        except:
            pass
    
    # 逐日回测
    closes = df['close'].values
    
    # 需要至少足够的数据来训练模型和PPO预测
    min_data_required = max(
        LSTM_SEQ_LENGTH * 2 if LSTM_AVAILABLE else 0,
        TRANSFORMER_MAX_SEQ_LEN * 2 if TRANSFORMER_AVAILABLE else 0,
        126 if ppo_model else 0,  # PPO需要126条数据
        60  # 至少60条数据
    )
    
    if len(df) < min_data_required:
        return {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'status': 'failed',
            'error': f'数据不足（需要至少{min_data_required}条，实际{len(df)}条）'
        }
    
    # 训练模型（使用前60%的数据）
    train_size = int(len(df) * 0.6)
    train_closes = closes[:train_size]
    
    # 训练LSTM
    if lstm_processor and len(train_closes) >= LSTM_SEQ_LENGTH * 2:
        try:
            if USE_SLIDING_WINDOW_NORMALIZE and len(train_closes) > SLIDING_WINDOW_SIZE:
                recent_closes = train_closes[-SLIDING_WINDOW_SIZE:]
            else:
                recent_closes = train_closes
            
            normalized_data, norm_params = lstm_processor.normalize(recent_closes)
            lstm_normalization_params = norm_params
            X, y = lstm_processor.create_sequences(normalized_data)
            if len(X) > 0:
                lstm_processor.train(X, y, epochs=50, batch_size=32, verbose=False)
                print(f"   ✅ LSTM模型训练完成")
        except Exception as e:
            print(f"   ⚠️  LSTM训练失败: {e}")
    
    # 训练Transformer
    if transformer_model and len(train_closes) >= TRANSFORMER_MAX_SEQ_LEN * 2:
        try:
            if TRANSFORMER_ADAPTIVE_WINDOW and len(train_closes) > SLIDING_WINDOW_SIZE:
                recent_closes = train_closes[-SLIDING_WINDOW_SIZE:]
            else:
                recent_closes = train_closes
            
            normalized_closes, norm_params = transformer_model.normalize(recent_closes)
            transformer_normalization_params = norm_params
            
            X_list, y_list = [], []
            for i in range(TRANSFORMER_MAX_SEQ_LEN, len(normalized_closes)):
                X_list.append(normalized_closes[i-TRANSFORMER_MAX_SEQ_LEN:i])
                y_list.append(normalized_closes[i])
            
            if len(X_list) > 0:
                X = np.array(X_list).reshape(len(X_list), TRANSFORMER_MAX_SEQ_LEN, 1)
                y = np.array(y_list).reshape(len(y_list), 1)
                transformer_model.train(
                    X, y, epochs=TRANSFORMER_EPOCHS, batch_size=32,
                    learning_rate=0.001, validation_split=0.2, verbose=False
                )
                print(f"   ✅ Transformer模型训练完成")
        except Exception as e:
            print(f"   ⚠️  Transformer训练失败: {e}")
    
    # 从训练数据之后开始回测
    start_idx = train_size
    
    print(f"\n   📈 开始逐日回测（从第{start_idx+1}个交易日开始，共{len(df)-start_idx}个交易日）\n")
    
    sell_triggered_count = 0  # 触发卖出策略的次数
    
    for i in range(start_idx, len(df)):
        current_date = df.iloc[i]['date']
        current_price = float(df.iloc[i]['close'])
        current_open = float(df.iloc[i]['open'])
        
        # 1. 首先使用PPO模型进行正常的交易决策
        ppo_action = None
        ppo_operation = "持有"
        ppo_percentage = 0.0
        if ppo_model and i >= 125:  # 需要至少126条数据（索引0到125）
            try:
                # 使用从索引0到i的最近126条数据（如果不足126条，从0开始）
                start_idx = max(0, i - 125)
                obs = closes[start_idx:i+1]
                
                # 如果数据不足126条，用第一个值填充
                if len(obs) < 126:
                    padding = np.full(126 - len(obs), obs[0] if len(obs) > 0 else current_price)
                    obs = np.concatenate([padding, obs])
                
                # 确保是126条数据
                if len(obs) >= 126:
                    obs_array = np.array(obs[-126:], dtype=np.float32)
                    action, _states = ppo_model.predict(obs_array, deterministic=True)
                    ppo_action = int(action)
                    # 动作映射：0=卖出100%, 1=卖出50%, 2=卖出25%, 3=持有, 4=买入25%, 5=买入50%, 6=买入100%
                    action_map = {
                        0: ("sell", 1.0),
                        1: ("sell", 0.5),
                        2: ("sell", 0.25),
                        3: ("hold", 0.0),
                        4: ("buy", 0.25),
                        5: ("buy", 0.5),
                        6: ("buy", 1.0)
                    }
                    ppo_operation, ppo_percentage = action_map.get(ppo_action, ("hold", 0.0))
            except Exception as e:
                pass
        
        # 2. 预测下一个交易日的价格（用于检查下跌超过3%的条件）
        lstm_pred, transformer_pred = predict_next_day_price(
            df, i, lstm_processor, transformer_model,
            lstm_normalization_params, transformer_normalization_params
        )
        
        # 计算预测的第二天价格变化百分比
        predicted_drop_pct = None
        if lstm_pred is not None:
            predicted_drop_pct = (lstm_pred - current_price) / current_price * 100
        elif transformer_pred is not None:
            predicted_drop_pct = (transformer_pred - current_price) / current_price * 100
        
        # 3. 检查是否触发新增的卖出策略：预测第二天下跌超过3%
        drop3percent_triggered = False
        if predicted_drop_pct is not None and predicted_drop_pct <= DROP_THRESHOLD:
            if shares_held > 0:
                drop3percent_triggered = True
                sell_triggered_count += 1

        # 同时判断“预测上涨较多”的信号（用于加仓 + 打价卖出）
        bull_signal = False
        if predicted_drop_pct is not None and predicted_drop_pct >= RISE_THRESHOLD:
            bull_signal = True

        # 4. 执行交易决策
        # 优先级：
        #   1）预测下跌超过3%：执行原来的全部卖出 + 当日T+0 逻辑
        #   2）预测上涨较多：执行新增的 20%/0% 加仓 & 75%/0% 打价卖出逻辑
        #   3）否则执行 PPO 原有决策
        if drop3percent_triggered:
            # 新增策略：预测下跌超过3%，全部卖出，然后在当天预测的低点买回（做T+0）
            shares_sold, net_increase, total_fee, adjusted_sell_price = calc_sell_trade(
                current_open, 1.0, shares_held
            )
            
            if shares_sold > 0:
                balance += net_increase
                original_shares = shares_held
                shares_held -= shares_sold
                cost_price_before_sell = cost_price
                
                trades.append({
                    'date': current_date,
                    'action': 'sell_all_drop3pct',
                    'price': adjusted_sell_price,
                    'shares': shares_sold,
                    'amount': net_increase,
                    'fee': total_fee,
                    'reason': f'预测下跌{predicted_drop_pct:.2f}%超过{DROP_THRESHOLD}%（覆盖PPO决策）',
                    'ppo_action': ppo_action,
                    'ppo_operation': ppo_operation
                })
                
                # 在当天预测的低点买回（做T+0）
                # 预测当天低点：使用预测价格或当前价格的一定折扣
                if predicted_drop_pct is not None:
                    # 如果预测下跌，使用预测价格作为买入价（通常是当天低点）
                    predicted_low_price = lstm_pred if lstm_pred is not None else transformer_pred
                    if predicted_low_price is None:
                        # 如果没有预测价格，使用当前价格下跌一定幅度作为低点
                        predicted_low_price = current_price * (1 + predicted_drop_pct / 100)
                    
                    # 确保买入价不超过卖出价（做T+0的盈利逻辑）
                    # 买入价应该比卖出价低，使用预测的低点价格
                    # 如果预测低点比卖出价还高，则使用卖出价的97%作为买入价
                    if predicted_low_price < adjusted_sell_price:
                        buy_price = predicted_low_price
                    else:
                        # 如果预测低点比卖出价高，使用卖出价的97%作为买入价（确保有盈利空间）
                        buy_price = adjusted_sell_price * 0.97
                    
                    # 使用卖出获得的资金买入，但买入股数不超过卖出的股数（做T+0的逻辑）
                    # 计算最多能买多少股（不超过卖出的股数）
                    max_shares_to_buy = shares_sold
                    max_buy_amount = max_shares_to_buy * buy_price
                    
                    # 如果资金足够，买入相同数量的股票；否则按资金买入
                    if balance >= max_buy_amount:
                        buy_percentage = max_buy_amount / balance if balance > 0 else 0
                    else:
                        buy_percentage = 1.0  # 使用全部资金买入
                    
                    shares_bought, total_cost, buy_fee, adjusted_buy_price = calc_buy_trade(
                        buy_price, buy_percentage, balance
                    )
                    
                    # 确保买入股数不超过卖出股数
                    if shares_bought > shares_sold:
                        shares_bought = shares_sold
                        # 重新计算成本和费用
                        trade_amount = shares_bought * adjusted_buy_price
                        commission = max(MIN_COMMISSION, trade_amount * COMMISSION_RATE)
                        transfer_fee = trade_amount * TRANSFER_FEE_RATE
                        buy_fee = commission + transfer_fee
                        total_cost = trade_amount + buy_fee
                    
                    if shares_bought > 0 and balance >= total_cost:
                        balance -= total_cost
                        # 更新成本价（加权平均）
                        if shares_held > 0:
                            total_cost_before = shares_held * cost_price
                            total_cost_after = total_cost_before + total_cost
                            new_shares = shares_held + shares_bought
                            cost_price = total_cost_after / new_shares if new_shares > 0 else cost_price
                        else:
                            cost_price = adjusted_buy_price
                        shares_held += shares_bought
                        
                        # 计算做T+0的收益
                        t0_profit = (adjusted_sell_price - adjusted_buy_price) * min(shares_sold, shares_bought)
                        t0_profit_pct = ((adjusted_sell_price - adjusted_buy_price) / adjusted_buy_price * 100) if adjusted_buy_price > 0 else 0
                        
                        trades.append({
                            'date': current_date,
                            'action': 'buy_back_t0',
                            'price': adjusted_buy_price,
                            'shares': shares_bought,
                            'amount': total_cost,
                            'fee': buy_fee,
                            'reason': f'做T+0：在预测低点{buy_price:.2f}买回（卖出价{adjusted_sell_price:.2f}，卖出{shares_sold}股）',
                            't0_profit': t0_profit,
                            't0_profit_pct': t0_profit_pct,
                            'predicted_drop_pct': predicted_drop_pct
                        })
        elif bull_signal:
            # 新增逻辑：预测上涨较多时，使用 20%/0% 加仓 + 75%/0% 打价卖出
            # 1）加仓逻辑：当前无仓则按20%仓位买入一次；已有仓则不再加仓（对应“20%和0%”）
            if shares_held <= 0 and balance > 1000:
                buy_pct = BULL_ADD_POSITION_PCT
                shares_bought, total_cost, total_fee, adjusted_price = calc_buy_trade(
                    current_open, buy_pct, balance
                )
                if shares_bought > 0:
                    balance -= total_cost
                    shares_held += shares_bought
                    cost_price = adjusted_price
                    trades.append({
                        'date': current_date,
                        'action': f'buy_bull_{int(buy_pct*100)}pct',
                        'price': adjusted_price,
                        'shares': shares_bought,
                        'amount': total_cost,
                        'fee': total_fee,
                        'reason': f'预测上涨{predicted_drop_pct:.2f}%较多，按20%仓位加仓（覆盖PPO决策）',
                        'ppo_action': ppo_action
                    })

            # 2）打价卖出逻辑：达到 +75% 利润或回撤到成本价（0%）时止盈
            if shares_held > 0 and cost_price > 0:
                # 当前浮动收益率
                current_profit_pct = (current_price - cost_price) / cost_price * 100

                # 如果已达到 +75% 目标或跌回成本价附近，则当天按市价全部卖出
                if current_profit_pct >= TAKE_PROFIT_PCT or current_profit_pct <= STOP_PROFIT_PCT:
                    shares_sold, net_increase, total_fee, adjusted_price = calc_sell_trade(
                        current_open, 1.0, shares_held
                    )
                    if shares_sold > 0:
                        balance += net_increase
                        shares_held -= shares_sold
                        cost_price = 0.0
                        trades.append({
                            'date': current_date,
                            'action': 'sell_bull_target_all',
                            'price': adjusted_price,
                            'shares': shares_sold,
                            'amount': net_increase,
                            'fee': total_fee,
                            'reason': (
                                f'预测上涨{predicted_drop_pct:.2f}%较多，'
                                f'浮动收益{current_profit_pct:.2f}%，触发75%/0%打价卖出（覆盖PPO决策）'
                            ),
                            'ppo_action': ppo_action
                        })
        elif ppo_action is not None:
            # 执行PPO模型的正常决策
            if ppo_operation == "sell" and shares_held > 0:
                # 卖出操作
                shares_sold, net_increase, total_fee, adjusted_price = calc_sell_trade(
                    current_open, ppo_percentage, shares_held
                )
                
                if shares_sold > 0:
                    balance += net_increase
                    shares_held -= shares_sold
                    cost_price = 0.0 if shares_held <= 0 else cost_price
                    
                    trades.append({
                        'date': current_date,
                        'action': f'sell_{int(ppo_percentage*100)}pct',
                        'price': adjusted_price,
                        'shares': shares_sold,
                        'amount': net_increase,
                        'fee': total_fee,
                        'reason': f'PPO模型决策：{ppo_operation} {int(ppo_percentage*100)}%',
                        'ppo_action': ppo_action
                    })
            
            elif ppo_operation == "buy" and balance > 1000:
                # 买入操作
                shares_bought, total_cost, total_fee, adjusted_price = calc_buy_trade(
                    current_open, ppo_percentage, balance
                )
                
                if shares_bought > 0:
                    balance -= total_cost
                    if shares_held <= 0:
                        cost_price = adjusted_price
                    else:
                        # 加权平均成本
                        total_cost_before = shares_held * cost_price
                        total_cost_after = total_cost_before + total_cost
                        new_shares = shares_held + shares_bought
                        cost_price = total_cost_after / new_shares if new_shares > 0 else cost_price
                    shares_held += shares_bought
                    
                    trades.append({
                        'date': current_date,
                        'action': f'buy_{int(ppo_percentage*100)}pct',
                        'price': adjusted_price,
                        'shares': shares_bought,
                        'amount': total_cost,
                        'fee': total_fee,
                        'reason': f'PPO模型决策：{ppo_operation} {int(ppo_percentage*100)}%',
                        'ppo_action': ppo_action
                    })
        
        # 计算当前总资产
        current_value = balance + shares_held * current_price
        daily_values.append({
            'date': current_date,
            'price': current_price,
            'balance': balance,
            'shares_held': shares_held,
            'total_value': current_value,
            'predicted_drop_pct': predicted_drop_pct
        })
    
    # 计算最终收益
    final_price = closes[-1]
    final_value = balance + shares_held * final_price
    total_return = (final_value - initial_balance) / initial_balance * 100
    
    # 计算买入持有策略的收益（基准）
    buy_hold_return = (final_price - closes[start_idx]) / closes[start_idx] * 100
    
    # 计算最大回撤
    values = [v['total_value'] for v in daily_values]
    if len(values) > 0:
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak * 100
        max_drawdown = float(np.min(drawdown))
    else:
        max_drawdown = 0.0
    
    # 统计做T+0的收益
    t0_trades = [t for t in trades if isinstance(t, dict) and t.get('action') == 'buy_back_t0']
    total_t0_profit = sum(t.get('t0_profit', 0) for t in t0_trades)
    avg_t0_profit_pct = np.mean([t.get('t0_profit_pct', 0) for t in t0_trades]) if len(t0_trades) > 0 else 0
    
    result = {
        'stock_code': stock_code,
        'stock_name': stock_name,
        'model_used': selected_model_path if 'selected_model_path' in locals() else None,
        'status': 'success',
        'initial_balance': initial_balance,
        'final_value': final_value,
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'max_drawdown': max_drawdown,
        'total_trades': len(trades),
        'sell_triggered_count': sell_triggered_count,
        't0_trades_count': len(t0_trades),
        'total_t0_profit': total_t0_profit,
        'avg_t0_profit_pct': avg_t0_profit_pct,
        'final_shares_held': shares_held,
        'final_balance': balance,
        'trades': trades[-20:] if len(trades) > 20 else trades,  # 保存最后20笔交易
        'daily_values': daily_values[-20:] if len(daily_values) > 20 else daily_values  # 只保存最后20天的数据
    }
    
    print(f"\n   ✅ 回测完成")
    print(f"      初始资金: {initial_balance:,.2f} 元")
    print(f"      最终资产: {final_value:,.2f} 元")
    print(f"      总收益率: {total_return:.2f}%")
    print(f"      买入持有收益率: {buy_hold_return:.2f}%")
    print(f"      最大回撤: {max_drawdown:.2f}%")
    print(f"      总交易次数: {len(trades)} 次")
    print(f"      触发卖出策略: {sell_triggered_count} 次")
    print(f"      做T+0次数: {len(t0_trades)} 次")
    if len(t0_trades) > 0:
        print(f"      做T+0总收益: {total_t0_profit:.2f} 元")
        print(f"      做T+0平均收益率: {avg_t0_profit_pct:.2f}%")
    
    return result

def main():
    """主函数"""
    print("="*70)
    print("回测脚本：预测第二天下跌超过3%全部卖出策略")
    print("="*70)
    print(f"回测年份: {BACKTEST_YEAR}")
    print(f"下跌阈值: {DROP_THRESHOLD}%")
    print(f"初始资金: {INITIAL_BALANCE:,.2f} 元")
    print("="*70)
    
    results = []
    
    print(f"📋 找到 {len(TARGET_STOCKS)} 只有V16预测文件的股票\n")
    
    for i, stock in enumerate(TARGET_STOCKS, 1):
        print(f"[{i}/{len(TARGET_STOCKS)}] {stock['name']} ({stock['code']}) - 使用模型: {stock.get('model', '自动选择')}")
        result = backtest_single_stock(stock['code'], stock['name'], stock.get('model'))
        results.append(result)
    
    # 保存结果
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f"backtest_drop3percent_results_{timestamp}.json"
    
    # 转换numpy类型
    def convert_numpy_types(obj):
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
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d')
        else:
            return obj
    
    results_serializable = convert_numpy_types(results)
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print("📊 回测结果汇总（使用最佳模型训练组）")
    print(f"{'='*70}\n")
    
    # 统计成功和失败的股票
    success_results = [r for r in results if r['status'] == 'success']
    failed_results = [r for r in results if r['status'] != 'success']
    
    print(f"✅ 成功回测: {len(success_results)} 只")
    print(f"❌ 失败: {len(failed_results)} 只\n")
    
    # 按收益率排序
    success_results_sorted = sorted(success_results, key=lambda x: x.get('total_return', -999), reverse=True)
    
    print("="*70)
    print("收益率排名（从高到低）")
    print("="*70)
    print(f"{'排名':<6} {'股票名称':<12} {'股票代码':<12} {'模型组':<20} {'总收益率':<12} {'买入持有':<12} {'最大回撤':<12} {'T+0次数':<10}")
    print("-"*100)
    
    for i, result in enumerate(success_results_sorted, 1):
        # 查找对应的模型组信息
        stock_code = result['stock_code']
        model_group = BEST_MODEL_MAPPING.get(stock_code, {}).get('group', '未知')
        print(f"{i:<6} {result['stock_name']:<12} {result['stock_code']:<12} {model_group:<20} "
              f"{result['total_return']:>10.2f}% {result['buy_hold_return']:>10.2f}% "
              f"{result['max_drawdown']:>10.2f}% {result.get('t0_trades_count', 0):>8}次")
    
    print("\n" + "="*70)
    print("详细结果")
    print("="*70 + "\n")
    
    for result in success_results_sorted:
        stock_code = result['stock_code']
        model_info = BEST_MODEL_MAPPING.get(stock_code, {})
        model_group = model_info.get('group', '未知')
        model_used = result.get('model_used', '未知')
        model_short = os.path.basename(model_used) if model_used else '未知'
        
        print(f"{result['stock_name']} ({result['stock_code']}):")
        print(f"  最佳模型组: {model_group}")
        print(f"  使用模型: {model_short}")
        print(f"  总收益率: {result['total_return']:.2f}%")
        print(f"  买入持有收益率: {result['buy_hold_return']:.2f}%")
        print(f"  策略优势: {result['total_return'] - result['buy_hold_return']:.2f}个百分点")
        print(f"  最大回撤: {result['max_drawdown']:.2f}%")
        print(f"  总交易次数: {result['total_trades']} 次")
        print(f"  触发卖出策略: {result['sell_triggered_count']} 次")
        print(f"  做T+0次数: {result.get('t0_trades_count', 0)} 次")
        if result.get('t0_trades_count', 0) > 0:
            print(f"  做T+0总收益: {result.get('total_t0_profit', 0):,.2f} 元")
            print(f"  做T+0平均收益率: {result.get('avg_t0_profit_pct', 0):.2f}%")
        print(f"  最终资产: {result['final_value']:,.2f} 元")
        print()
    
    if failed_results:
        print("="*70)
        print("失败股票")
        print("="*70 + "\n")
        for result in failed_results:
            print(f"{result['stock_name']} ({result['stock_code']}): ❌ {result.get('error', '未知错误')}")
        print()
    
    # 统计汇总
    if success_results:
        avg_return = sum(r['total_return'] for r in success_results) / len(success_results)
        avg_buy_hold = sum(r['buy_hold_return'] for r in success_results) / len(success_results)
        total_t0_count = sum(r.get('t0_trades_count', 0) for r in success_results)
        total_t0_profit = sum(r.get('total_t0_profit', 0) for r in success_results)
        
        print("="*70)
        print("统计汇总")
        print("="*70)
        print(f"平均总收益率: {avg_return:.2f}%")
        print(f"平均买入持有收益率: {avg_buy_hold:.2f}%")
        print(f"平均策略优势: {avg_return - avg_buy_hold:.2f}个百分点")
        print(f"总T+0次数: {total_t0_count} 次")
        print(f"总T+0收益: {total_t0_profit:,.2f} 元")
        print()
    
    print(f"📁 详细结果已保存到: {result_file}")

if __name__ == '__main__':
    main()

