"""
V12模型批量回测脚本
对指定的10只股票使用V12模型进行回测

股票列表：
1. 工业富联 - sh.601138
2. 华勤技术 - sh.603296
3. 通富微电 - sz.002156
4. 香农芯创 - sz.300475
5. 壹石通 - sh.688733
6. 亚威股份 - sz.002559
7. 鸿博股份 - sz.002229
8. 中富通 - sz.300560
9. 中电港 - sz.001287
10. 中际旭创 - sz.300308
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import datetime
import json
from collections import defaultdict

warnings.filterwarnings('ignore', category=DeprecationWarning)

# 导入V12相关模块
try:
    from technical_indicators import TechnicalIndicators
    TECHNICAL_INDICATORS_AVAILABLE = True
except ImportError:
    TECHNICAL_INDICATORS_AVAILABLE = False
    print("[警告] 技术指标模块不可用")

try:
    from lstm_gru_time_series import TimeSeriesProcessor
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("[警告] LSTM/GRU模块不可用")

try:
    from transformer_model import TransformerPredictor
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    print("[警告] Transformer模块不可用")

try:
    from stable_baselines3 import PPO
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False
    print("[警告] PPO模型不可用")

# 抑制Gym警告
warnings.filterwarnings('ignore', message='.*Gym has been unmaintained.*')
warnings.filterwarnings('ignore', message='.*upgrade to Gymnasium.*')

# ==================== 配置参数 ====================

# 股票列表
STOCKS = [
    {"code": "sh.601138", "name": "工业富联", "baostock": "sh.601138", "akshare": "601138"},
    {"code": "sh.603296", "name": "华勤技术", "baostock": "sh.603296", "akshare": "603296"},
    {"code": "sz.002156", "name": "通富微电", "baostock": "sz.002156", "akshare": "002156"},
    {"code": "sz.300475", "name": "香农芯创", "baostock": "sz.300475", "akshare": "300475"},
    {"code": "sh.688733", "name": "壹石通", "baostock": "sh.688733", "akshare": "688733"},
    {"code": "sz.002559", "name": "亚威股份", "baostock": "sz.002559", "akshare": "002559"},
    {"code": "sz.002229", "name": "鸿博股份", "baostock": "sz.002229", "akshare": "002229"},
    {"code": "sz.300560", "name": "中富通", "baostock": "sz.300560", "akshare": "300560"},
    {"code": "sz.001287", "name": "中电港", "baostock": "sz.001287", "akshare": "001287"},
    {"code": "sz.300308", "name": "中际旭创", "baostock": "sz.300308", "akshare": "300308"},
    {"code": "sh.513130", "name": "恒生科技ETF", "baostock": "sh.513130", "akshare": "513130"},
]

# V12配置
MODEL_PATH = "ppo_stock_v7.zip"  # PPO模型路径
INITIAL_BALANCE = 50000.0  # 初始资金

# V12模型配置
ENABLE_LSTM_PREDICTION = True
ENABLE_TRANSFORMER = True
LSTM_SEQ_LENGTH = 60
TRANSFORMER_MAX_SEQ_LEN = 100
TRANSFORMER_EPOCHS = 120
USE_SLIDING_WINDOW_NORMALIZE = True
SLIDING_WINDOW_SIZE = 500
TRANSFORMER_ADAPTIVE_WINDOW = True
TRANSFORMER_TREND_AWARE = True
TRANSFORMER_POST_PROCESS = True
TRANSFORMER_PRICE_POSITION_THRESHOLD = 0.75

# 交易成本
COMMISSION_RATE = 0.00025
MIN_COMMISSION = 5.0
TRANSFER_FEE_RATE = 0.00001
STAMP_DUTY_RATE = 0.001
SLIPPAGE_RATE = 0.0005

# 模型权重
MODEL_WEIGHTS = {
    'ppo': 0.4,
    'lstm': 0.2,
    'transformer': 0.2,
    'holographic': 0.2
}

# ==================== 工具函数 ====================

def round_to_lot(shares):
    """将股数向下取整为100股的整数倍"""
    if shares <= 0:
        return 0
    return int(shares // 100) * 100

def calc_buy_trade(current_price, buy_percentage, current_balance):
    """计算买入交易"""
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
    """计算卖出交易"""
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

def map_action_to_operation(action):
    """将动作映射到具体操作"""
    actions = {
        0: "卖出 100%",
        1: "卖出 50%",
        2: "卖出 25%",
        3: "持有",
        4: "买入 25%",
        5: "买入 50%",
        6: "买入 100%"
    }
    return actions.get(action, "未知动作")

def action_to_percentage(action):
    """将动作转换为交易百分比"""
    if action == 0:
        return -1.0  # 卖出100%
    elif action == 1:
        return -0.5  # 卖出50%
    elif action == 2:
        return -0.25  # 卖出25%
    elif action == 3:
        return 0.0  # 持有
    elif action == 4:
        return 0.25  # 买入25%
    elif action == 5:
        return 0.5  # 买入50%
    elif action == 6:
        return 1.0  # 买入100%
    return 0.0

def fuse_multi_model_predictions(ppo_action, lstm_prediction, transformer_prediction, current_price):
    """融合多模型预测结果"""
    if ppo_action is None:
        ppo_action = 3  # 默认持有
    
    # 如果有价格预测，根据预测方向调整动作
    predictions = []
    if lstm_prediction is not None:
        predictions.append(lstm_prediction)
    if transformer_prediction is not None:
        predictions.append(transformer_prediction)
    
    if predictions:
        avg_prediction = np.mean(predictions)
        price_change_pct = (avg_prediction - current_price) / current_price * 100
        
        # 如果预测方向与PPO动作冲突，进行调整
        if price_change_pct < -2.0 and ppo_action >= 4:  # 预测下跌但PPO建议买入
            if ppo_action == 6:
                final_action = 3  # 调整为持有
            elif ppo_action == 5:
                final_action = 4  # 降低买入力度
            else:
                final_action = ppo_action
        elif price_change_pct > 2.0 and ppo_action <= 2:  # 预测上涨但PPO建议卖出
            if ppo_action <= 1:
                final_action = 3  # 调整为持有
            else:
                final_action = ppo_action
        else:
            final_action = ppo_action
    else:
        final_action = ppo_action
    
    return final_action

def load_stock_data(stock_code, stock_name):
    """加载股票历史数据"""
    # 尝试从本地文件加载
    possible_paths = [
        f"stockdata/test/{stock_code}.{stock_name}.csv",
        f"stockdata/train/{stock_code}.{stock_name}.csv",
        f"stockdata_v7/test/{stock_code}.{stock_name}.csv",
        f"stockdata_v7/train/{stock_code}.{stock_name}.csv",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if 'date' in df.columns and 'close' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                    return df
            except Exception as e:
                print(f"   ⚠️ 读取文件失败 {path}: {e}")
                continue
    
    # 如果本地文件不存在，尝试从baostock获取
    try:
        import baostock as bs
        bs_code = stock_code
        lg = bs.login()
        if lg.error_code == '0':
            # 获取最近2年的数据
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime('%Y-%m-%d')
            
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount,preclose",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="3"
            )
            
            if rs.error_code == '0':
                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())
                
                if data_list:
                    df = pd.DataFrame(data_list, columns=rs.fields)
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                    # 确保close列是数值类型
                    df['close'] = pd.to_numeric(df['close'], errors='coerce')
                    df = df.dropna(subset=['close'])
                    bs.logout()
                    return df
            bs.logout()
    except Exception as e:
        print(f"   ⚠️ 从baostock获取数据失败: {e}")
    
    return None

# ==================== 回测主函数 ====================

def backtest_stock(stock_info, ppo_model, lstm_processor, transformer_model, tech_indicators):
    """对单只股票进行回测"""
    stock_code = stock_info['code']
    stock_name = stock_info['name']
    
    print(f"\n{'='*70}")
    print(f"📊 开始回测: {stock_name} ({stock_code})")
    print(f"{'='*70}")
    
    # 加载数据
    df = load_stock_data(stock_code, stock_name)
    if df is None or len(df) < 126:
        print(f"❌ 数据不足，跳过 {stock_name}")
        return None
    
    print(f"✅ 数据加载成功，共 {len(df)} 条记录")
    print(f"   日期范围: {df['date'].min()} 至 {df['date'].max()}")
    
    # 初始化状态
    initial_balance = INITIAL_BALANCE
    current_balance = initial_balance
    shares_held = 0.0
    net_worth_history = [initial_balance]
    trade_history = []
    actions_taken = []
    
    # 价格序列
    closes = df['close'].astype(float).values
    dates = df['date'].values
    
    # 模型训练状态
    lstm_trained = False
    transformer_trained = False
    lstm_normalization_params = None
    transformer_normalization_params = None
    
    # 从第126个交易日开始（需要足够的历史数据）
    start_idx = 126
    print(f"\n📈 开始回测（从第 {start_idx+1} 个交易日开始）")
    
    for i in range(start_idx, len(closes)):
        try:
            current_price = float(closes[i])
            current_date = dates[i]
            
            # 获取历史价格序列（用于PPO模型）
            hist_prices = closes[max(0, i-126):i+1]
            if len(hist_prices) < 126:
                continue
            
            # ========== PPO模型预测 ==========
            ppo_action = None
            if ppo_model:
                try:
                    obs = np.array(hist_prices[-126:], dtype=np.float32)
                    action, _ = ppo_model.predict(obs, deterministic=True)
                    ppo_action = int(action)
                except Exception as e:
                    pass
            
            # ========== LSTM预测 ==========
            lstm_prediction = None
            if lstm_processor and ENABLE_LSTM_PREDICTION:
                try:
                    if not lstm_trained and len(closes[:i+1]) >= LSTM_SEQ_LENGTH * 2:
                        # 训练LSTM模型
                        recent_closes = closes[max(0, i-SLIDING_WINDOW_SIZE):i+1]
                        if len(recent_closes) < SLIDING_WINDOW_SIZE:
                            recent_closes = closes[:i+1]
                        
                        normalized_data, norm_params = lstm_processor.normalize(recent_closes)
                        lstm_normalization_params = norm_params
                        X, y = lstm_processor.create_sequences(normalized_data)
                        if len(X) > 0:
                            lstm_processor.train(X, y, epochs=50, batch_size=32, verbose=False)
                            lstm_trained = True
                    
                    if lstm_trained and lstm_normalization_params:
                        # 预测
                        seq = closes[max(0, i-LSTM_SEQ_LENGTH):i+1]
                        if len(seq) >= LSTM_SEQ_LENGTH:
                            # 使用训练时的归一化参数
                            norm_method = lstm_normalization_params.get('method', 'minmax')
                            if norm_method == 'minmax':
                                min_val = lstm_normalization_params['min']
                                max_val = lstm_normalization_params['max']
                                if max_val - min_val > 0:
                                    normalized_seq = (seq - min_val) / (max_val - min_val)
                                else:
                                    normalized_seq = np.zeros_like(seq)
                            else:
                                normalized_seq = seq
                            
                            prediction_norm = lstm_processor.predict_next(normalized_seq[-LSTM_SEQ_LENGTH:])
                            if prediction_norm is not None:
                                lstm_prediction = float(lstm_processor.denormalize(
                                    np.array([prediction_norm]),
                                    lstm_normalization_params
                                )[0])
                except Exception as e:
                    pass
            
            # ========== Transformer预测 ==========
            transformer_prediction = None
            if transformer_model and ENABLE_TRANSFORMER:
                try:
                    if not transformer_trained and len(closes[:i+1]) >= TRANSFORMER_MAX_SEQ_LEN * 2:
                        # 训练Transformer模型
                        recent_closes = closes[max(0, i-SLIDING_WINDOW_SIZE):i+1]
                        if len(recent_closes) < SLIDING_WINDOW_SIZE:
                            recent_closes = closes[:i+1]
                        
                        # 自适应窗口
                        if TRANSFORMER_ADAPTIVE_WINDOW:
                            price_position = (current_price - np.min(recent_closes)) / (np.max(recent_closes) - np.min(recent_closes) + 1e-8)
                            if price_position > TRANSFORMER_PRICE_POSITION_THRESHOLD:
                                adaptive_window = max(int(len(recent_closes) * 0.6), TRANSFORMER_MAX_SEQ_LEN * 2)
                                recent_closes = recent_closes[-adaptive_window:]
                        
                        normalized_closes, norm_params = transformer_model.normalize(recent_closes)
                        transformer_normalization_params = norm_params
                        
                        X_list, y_list = [], []
                        for j in range(TRANSFORMER_MAX_SEQ_LEN, len(normalized_closes)):
                            X_list.append(normalized_closes[j-TRANSFORMER_MAX_SEQ_LEN:j])
                            y_list.append(normalized_closes[j])
                        
                        if len(X_list) > 0:
                            X = np.array(X_list).reshape(len(X_list), TRANSFORMER_MAX_SEQ_LEN, 1)
                            y = np.array(y_list).reshape(len(y_list), 1)
                            transformer_model.train(
                                X, y, epochs=TRANSFORMER_EPOCHS, batch_size=32,
                                learning_rate=0.001, validation_split=0.2, verbose=False
                            )
                            transformer_trained = True
                    
                    if transformer_trained and transformer_normalization_params:
                        # 预测
                        seq = closes[max(0, i-TRANSFORMER_MAX_SEQ_LEN):i+1]
                        if len(seq) >= TRANSFORMER_MAX_SEQ_LEN:
                            # 使用训练时的归一化参数
                            norm_method = transformer_normalization_params.get('method', 'minmax')
                            if norm_method == 'minmax':
                                min_val = transformer_normalization_params['min']
                                max_val = transformer_normalization_params['max']
                                if max_val - min_val > 0:
                                    normalized_seq = (seq - min_val) / (max_val - min_val)
                                else:
                                    normalized_seq = np.zeros_like(seq)
                            else:
                                normalized_seq = seq
                            
                            prediction_norm = transformer_model.predict_next(normalized_seq[-TRANSFORMER_MAX_SEQ_LEN:])
                            if prediction_norm is not None:
                                transformer_prediction_raw = float(transformer_model.denormalize(
                                    np.array([prediction_norm]),
                                    transformer_normalization_params
                                )[0])
                                
                                # 趋势感知和后处理
                                transformer_prediction = transformer_prediction_raw
                                if TRANSFORMER_TREND_AWARE and len(closes[:i+1]) >= 10:
                                    short_trend = (closes[i] - closes[max(0, i-5)]) / closes[max(0, i-5)] if closes[max(0, i-5)] > 0 else 0
                                    mid_trend = (closes[i] - closes[max(0, i-10)]) / closes[max(0, i-10)] if closes[max(0, i-10)] > 0 else 0
                                    momentum = (short_trend + mid_trend) / 2
                                    
                                    if momentum > 0.01 and transformer_prediction < current_price:
                                        trend_adjustment = min(momentum * 0.3, 0.05)
                                        transformer_prediction = transformer_prediction * (1 + trend_adjustment)
                except Exception as e:
                    pass
            
            # ========== 融合决策 ==========
            final_action = fuse_multi_model_predictions(
                ppo_action, lstm_prediction, transformer_prediction, current_price
            )
            actions_taken.append(final_action)
            
            # ========== 执行交易 ==========
            action_pct = action_to_percentage(final_action)
            
            if action_pct > 0:  # 买入
                buy_pct = action_pct
                shares_bought, cost, fee, trade_price = calc_buy_trade(
                    current_price, buy_pct, current_balance
                )
                if shares_bought > 0:
                    current_balance -= cost
                    shares_held += shares_bought
                    trade_history.append({
                        'date': current_date,
                        'action': 'BUY',
                        'shares': shares_bought,
                        'price': trade_price,
                        'amount': cost,
                        'fee': fee
                    })
            elif action_pct < 0:  # 卖出
                sell_pct = abs(action_pct)
                shares_sold, proceeds, fee, trade_price = calc_sell_trade(
                    current_price, sell_pct, shares_held
                )
                if shares_sold > 0:
                    shares_held -= shares_sold
                    current_balance += proceeds
                    trade_history.append({
                        'date': current_date,
                        'action': 'SELL',
                        'shares': shares_sold,
                        'price': trade_price,
                        'amount': proceeds,
                        'fee': fee
                    })
            
            # 计算当前净值
            current_net_worth = current_balance + shares_held * current_price
            net_worth_history.append(current_net_worth)
            
        except Exception as e:
            continue
    
    # 计算回测指标
    if len(net_worth_history) == 0:
        return None
    
    final_net_worth = net_worth_history[-1]
    total_return = (final_net_worth / initial_balance - 1) * 100
    
    # 最大回撤
    peak = initial_balance
    max_drawdown = 0.0
    for nw in net_worth_history:
        if nw > peak:
            peak = nw
        drawdown = (peak - nw) / peak * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # 夏普比率
    returns = np.diff(net_worth_history) / net_worth_history[:-1]
    sharpe_ratio = 0.0
    if len(returns) > 0 and np.std(returns) > 0:
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
    
    # 基准收益（买入持有）
    initial_price = float(closes[start_idx])
    final_price = float(closes[-1])
    buy_hold_return = (final_price / initial_price - 1) * 100
    
    # 交易统计
    num_trades = len(trade_history)
    buy_trades = [t for t in trade_history if t['action'] == 'BUY']
    sell_trades = [t for t in trade_history if t['action'] == 'SELL']
    
    # 胜率（简化计算：比较买卖价格）
    win_count = 0
    if len(sell_trades) > 0 and len(buy_trades) > 0:
        # 简单计算：卖出价格高于买入价格算盈利
        for sell_trade in sell_trades:
            # 找到对应的买入交易（简化：使用最近的买入价格）
            recent_buys = [t for t in buy_trades if t['date'] < sell_trade['date']]
            if recent_buys:
                avg_buy_price = np.mean([t['price'] for t in recent_buys[-5:]])  # 使用最近5次买入的平均价
                if sell_trade['price'] > avg_buy_price:
                    win_count += 1
        win_rate = (win_count / len(sell_trades) * 100) if len(sell_trades) > 0 else 0.0
    else:
        win_rate = 0.0
    
    stats = {
        'stock_code': stock_code,
        'stock_name': stock_name,
        'initial_balance': initial_balance,
        'final_net_worth': final_net_worth,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'buy_hold_return': buy_hold_return,
        'excess_return': total_return - buy_hold_return,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'total_days': len(closes) - start_idx,
        'net_worth_history': net_worth_history,
        'trade_history': trade_history
    }
    
    print(f"\n✅ 回测完成")
    print(f"   最终净值: {final_net_worth:,.2f} 元")
    print(f"   总收益率: {total_return:+.2f}%")
    print(f"   买入持有收益: {buy_hold_return:+.2f}%")
    print(f"   超额收益: {total_return - buy_hold_return:+.2f}%")
    print(f"   最大回撤: {max_drawdown:.2f}%")
    print(f"   夏普比率: {sharpe_ratio:.2f}")
    print(f"   交易次数: {num_trades}")
    print(f"   胜率: {win_rate:.2f}%")
    
    return stats

# ==================== 主程序 ====================

def main():
    print("\n" + "="*70)
    print("🚀 V12模型批量回测系统")
    print("="*70)
    print(f"回测股票数量: {len(STOCKS)}")
    print(f"初始资金: {INITIAL_BALANCE:,.0f} 元")
    print("="*70)
    
    # 加载PPO模型
    ppo_model = None
    if PPO_AVAILABLE:
        if os.path.exists(MODEL_PATH):
            try:
                ppo_model = PPO.load(MODEL_PATH)
                print(f"✅ PPO模型加载成功: {MODEL_PATH}")
            except Exception as e:
                print(f"⚠️  PPO模型加载失败: {e}")
        else:
            print(f"⚠️  PPO模型文件不存在: {MODEL_PATH}")
    
    # 初始化LSTM处理器
    lstm_processor = None
    if LSTM_AVAILABLE and ENABLE_LSTM_PREDICTION:
        try:
            lstm_processor = TimeSeriesProcessor(
                seq_length=LSTM_SEQ_LENGTH,
                hidden_size=64,
                model_type='lstm_attention'
            )
            print("✅ LSTM处理器初始化成功")
        except Exception as e:
            print(f"⚠️  LSTM处理器初始化失败: {e}")
    
    # 初始化Transformer模型
    transformer_model = None
    if TRANSFORMER_AVAILABLE and ENABLE_TRANSFORMER:
        try:
            transformer_model = TransformerPredictor(
                input_size=1,
                d_model=64,
                nhead=4,
                num_encoder_layers=3,
                num_decoder_layers=3,
                max_seq_len=TRANSFORMER_MAX_SEQ_LEN
            )
            print("✅ Transformer模型初始化成功")
        except Exception as e:
            print(f"⚠️  Transformer模型初始化失败: {e}")
    
    # 初始化技术指标
    tech_indicators = None
    if TECHNICAL_INDICATORS_AVAILABLE:
        try:
            tech_indicators = TechnicalIndicators()
            print("✅ 技术指标模块初始化成功")
        except Exception as e:
            print(f"⚠️  技术指标模块初始化失败: {e}")
    
    print("\n" + "="*70)
    print("开始批量回测...")
    print("="*70)
    
    # 批量回测
    all_results = []
    for stock_info in STOCKS:
        try:
            result = backtest_stock(stock_info, ppo_model, lstm_processor, transformer_model, tech_indicators)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"❌ {stock_info['name']} 回测失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 汇总结果
    if len(all_results) > 0:
        print("\n" + "="*70)
        print("📊 回测汇总结果")
        print("="*70)
        
        # 创建汇总表格
        summary_data = []
        for result in all_results:
            summary_data.append({
                '股票代码': result['stock_code'],
                '股票名称': result['stock_name'],
                '最终净值': f"{result['final_net_worth']:,.2f}",
                '总收益率(%)': f"{result['total_return']:+.2f}",
                '买入持有收益(%)': f"{result['buy_hold_return']:+.2f}",
                '超额收益(%)': f"{result['excess_return']:+.2f}",
                '最大回撤(%)': f"{result['max_drawdown']:.2f}",
                '夏普比率': f"{result['sharpe_ratio']:.2f}",
                '交易次数': result['num_trades'],
                '胜率(%)': f"{result['win_rate']:.2f}",
            })
        
        summary_df = pd.DataFrame(summary_data)
        print("\n" + summary_df.to_string(index=False))
        
        # 计算平均指标
        avg_return = np.mean([r['total_return'] for r in all_results])
        avg_excess_return = np.mean([r['excess_return'] for r in all_results])
        avg_max_drawdown = np.mean([r['max_drawdown'] for r in all_results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in all_results])
        avg_win_rate = np.mean([r['win_rate'] for r in all_results])
        total_trades = sum([r['num_trades'] for r in all_results])
        
        print("\n" + "="*70)
        print("📈 平均指标")
        print("="*70)
        print(f"平均收益率: {avg_return:+.2f}%")
        print(f"平均超额收益: {avg_excess_return:+.2f}%")
        print(f"平均最大回撤: {avg_max_drawdown:.2f}%")
        print(f"平均夏普比率: {avg_sharpe:.2f}")
        print(f"平均胜率: {avg_win_rate:.2f}%")
        print(f"总交易次数: {total_trades}")
        
        # 保存结果
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"backtest_v12_results_{timestamp}.csv"
        summary_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n✅ 结果已保存: {output_file}")
        
        # 保存详细结果到JSON
        json_file = f"backtest_v12_results_{timestamp}.json"
        # 移除net_worth_history和trade_history以减小文件大小
        json_results = []
        for r in all_results:
            json_result = r.copy()
            json_result.pop('net_worth_history', None)
            json_result.pop('trade_history', None)
            json_results.append(json_result)
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        print(f"✅ 详细结果已保存: {json_file}")
    else:
        print("\n❌ 没有成功完成回测的股票")
    
    print("\n" + "="*70)
    print("✅ 批量回测完成")
    print("="*70)

if __name__ == '__main__':
    main()

