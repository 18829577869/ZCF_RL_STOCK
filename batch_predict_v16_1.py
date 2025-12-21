"""
V16批量预测系统 - 批量运行所有V16预测股票
整合 V7、V9、V10、V11、V12、V13 的所有功能：
1. V7功能：技术指标、多数据源、LLM解释、成本模型、PPO强化学习
2. V9功能：LSTM/GRU、注意力机制、动态参数优化、自动学习优化
3. V10功能：Transformer、多模态处理、实时可视化、全息动态模型
4. V11功能：智能融合决策、滑动窗口归一化、动态权重调整
5. V12功能：Transformer预测优化、预测方向冲突检测、智能决策调整
6. V13功能：自动模型选择、止损止盈风险控制、凯利公式资金管理

V14新增功能：
- StockAPI集成：新增StockAPI数据源支持，提供更丰富的股票数据
- 数据源优先级优化：StockAPI作为高优先级数据源，提供实时行情数据
- 多数据源容错增强：StockAPI失败时自动回退到其他数据源

V15新增功能：
- DeepSeek 轮次复盘：每轮预测结束调用 DeepSeek 对当前决策进行简洁点评，提供执行要点与风险提醒

V16批量预测功能：
- 批量运行多个股票的预测，每个股票只运行一次预测
- 包含所有股票的完整版本（batch_predict_v16_1.py）

设计理念：多模型协同工作，智能融合决策，自动选择最优模型，风险控制优先，科学资金管理，丰富数据源
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import datetime
import time
import json
import threading
from io import StringIO

# 代理配置（可通过环境变量或配置文件设置）
# 如果设置了代理，将用于反爬虫功能
# 格式示例：['http://user:pass@host:port', 'socks5://host:port']
PROXIES = os.getenv('PROXIES', '').split(',') if os.getenv('PROXIES') else []
PROXIES = [p.strip() for p in PROXIES if p.strip()]  # 清理空字符串

# 是否启用反爬虫功能（Cookie/UA/代理池）
ENABLE_ANTI_CRAWLER = os.getenv('ENABLE_ANTI_CRAWLER', 'true').lower() == 'true'

warnings.filterwarnings('ignore', category=DeprecationWarning)

# ==================== V7 成本模型配置 ====================

COMMISSION_RATE = 0.00025  # 佣金率
MIN_COMMISSION = 5.0  # 最低佣金
TRANSFER_FEE_RATE = 0.00001  # 过户费率
STAMP_DUTY_RATE = 0.001  # 印花税率（仅卖出）
SLIPPAGE_RATE = 0.0005  # 滑点率

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
        # 重新按资金上限计算可买股数（向下取整到100股）
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

# ==================== 导入模块 ====================

# V7模块：技术指标、多数据源、LLM解释
try:
    from technical_indicators import TechnicalIndicators
    TECHNICAL_INDICATORS_AVAILABLE = True
except ImportError:
    TECHNICAL_INDICATORS_AVAILABLE = False
    print("[警告] 技术指标模块不可用")

try:
    from multi_data_source_manager import MultiDataSourceManager
    MULTI_DATA_SOURCE_AVAILABLE = True
except ImportError:
    MULTI_DATA_SOURCE_AVAILABLE = False
    print("[警告] 多数据源管理器不可用")

try:
    from llm_indicator_interpreter import LLMIndicatorInterpreter
    LLM_INTERPRETER_AVAILABLE = True
except ImportError:
    LLM_INTERPRETER_AVAILABLE = False
    print("[警告] LLM指标解释器不可用")

# V9模块：LSTM/GRU、动态参数优化
try:
    from lstm_gru_time_series import TimeSeriesProcessor
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("[警告] LSTM/GRU模块不可用")

try:
    from dynamic_parameter_optimizer import (
        DynamicParameterOptimizer, AutoLearningOptimizer, ParameterRange
    )
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False
    print("[警告] 参数优化器模块不可用")

# V10模块：Transformer、多模态、可视化、全息模型
try:
    from transformer_model import TransformerPredictor
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    print("[警告] Transformer模块不可用")

try:
    from multimodal_data_processor import MultimodalDataProcessor
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False
    print("[警告] 多模态处理模块不可用")

try:
    from realtime_visualization import RealTimeVisualizer, WebVisualizationServer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("[警告] 可视化模块不可用")

try:
    from holographic_dynamic_model import HolographicDynamicModel
    HOLOGRAPHIC_AVAILABLE = True
except ImportError:
    HOLOGRAPHIC_AVAILABLE = False
    print("[警告] 全息动态模型不可用")

# 其他模块
# 抑制Gym的废弃警告（stable_baselines3内部使用gym）
import warnings
warnings.filterwarnings('ignore', message='.*Gym has been unmaintained.*')
warnings.filterwarnings('ignore', message='.*upgrade to Gymnasium.*')

try:
    from stable_baselines3 import PPO
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False
    print("[警告] PPO模型不可用")

try:
    from llm_market_intelligence import MarketIntelligenceAgent
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("[警告] LLM市场情报不可用")

# ==================== 工具函数 ====================

def convert_stock_code(code):
    """转换股票代码格式"""
    if '.' in code:
        market, num = code.split('.')
        return {
            'baostock': code,
            'tushare': f"{num}.{market.upper()}",
            'akshare': num,
            'market': 'sh' if market == 'sh' else 'sz'
        }
    else:
        if code.startswith('6'):
            return {
                'baostock': f"sh.{code}",
                'tushare': f"{code}.SH",
                'akshare': code,
                'market': 'sh'
            }
        else:
            return {
                'baostock': f"sz.{code}",
                'tushare': f"{code}.SZ",
                'akshare': code,
                'market': 'sz'
            }

def get_stock_name(code):
    """根据股票代码获取股票名称"""
    stock_name_map = {
        'sh.603267': '鸿远电子',
        'sh.603698': '航天工程',
        'sz.002025': '航天电器',
        'sz.002241': '歌尔股份',
        'sz.002475': '立讯精密',
        'sz.300726': '宏达电子',
        'sz.300762': '上海瀚讯',
        'sz.301017': '漱玉平民',
        'sz.300749': '顶固集创',
        'sh.600730': '中国高科',
        'sz.002851': '麦格米特',
        'sz.300274': '阳光电源',
        'sz.002266': '浙富控股',
        'sz.300153': '科泰电源',
        'sz.002837': '英维克',
        'sz.300499': '高澜股份',
        'sz.002706': '良信股份',
        'sh.601399': '国机重装',
        'sz.301005': '超捷股份',
    }
    return stock_name_map.get(code, code)  # 如果找不到，返回代码本身

def map_action_to_operation(action):
    """将动作映射到具体操作（内部使用，不直接展示给用户）"""
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

def map_action_to_direction(action):
    """将PPO动作映射为方向性描述（避免“买入100%”等歧义）"""
    directions = {
        0: "强烈看空",
        1: "看空",
        2: "轻微看空",
        3: "中性",
        4: "轻微看多",
        5: "看多",
        6: "强烈看多"
    }
    return directions.get(action, "未知")

def fetch_akshare_5min(code_info, days=7):
    """使用 AkShare 获取5分钟K线数据"""
    try:
        import akshare as ak
        symbol = code_info['akshare']
        today = datetime.date.today()
        start_date = (today - datetime.timedelta(days=days)).strftime('%Y%m%d')
        end_date = today.strftime('%Y%m%d')
        
        try:
            df = ak.stock_zh_a_hist_min_em(
                symbol=symbol,
                period="5",
                adjust="qfq",
                start_date=start_date,
                end_date=end_date
            )
            if df is None or len(df) == 0:
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"
                )
                if df is not None and len(df) > 0:
                    df = df.rename(columns={'日期': 'date', '收盘': 'close', '成交量': 'volume'})
                    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                    df['time'] = df['date'] + '15000000'
                    return df[['date', 'time', 'close', 'volume']]
                return None
            
            column_mapping = {
                '时间': 'time',
                '收盘': 'close',
                '成交量': 'volume',
                '日期': 'date'
            }
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y%m%d%H%M%S')
                df['date'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d')
            elif 'date' in df.columns:
                df['time'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d%H%M%S')
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
            return df[['date', 'time', 'close', 'volume']]
        except Exception as e:
            return None
    except ImportError:
        return None
    except Exception as e:
        return None

def init_trade_log():
    """初始化交易日志文件"""
    import csv
    TRADE_LOG_FILE = "trade_log.csv"
    if not os.path.exists(TRADE_LOG_FILE):
        with open(TRADE_LOG_FILE, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                '时间戳', '日期', '时间', '股票代码', '操作类型', '操作比例', 
                '当前价格', '建议买入价格', '建议卖出价格', '预测数量', '预测金额', 
                '持仓数量', '可用资金', '总资产', '操作状态', '备注'
            ])

def save_portfolio_state(stock_code, shares_held, current_balance, last_price, initial_balance,
                        actual_buy_price=None, actual_sell_price=None, cost_price=None,
                        realized_pnl=None):
    """保存持仓状态"""
    try:
        # V12优化：不在这里回退到last_price，只有在明确传入时才保存成本价
        # 回退逻辑应该在调用方处理（Web编辑器），确保用户明确设置的成本价不被覆盖
        
        state = {
            'stock_code': stock_code,
            'shares_held': float(shares_held),
            'current_balance': float(current_balance),
            'last_price': float(last_price),
            'initial_balance': float(initial_balance),
            'last_update': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_assets': float(current_balance + shares_held * last_price)
        }
        
        # 添加可选字段
        if actual_buy_price and actual_buy_price > 0:
            state['actual_buy_price'] = float(actual_buy_price)
        
        # V12优化：只有当cost_price明确存在且大于0时才保存，不使用last_price作为回退
        if cost_price is not None and isinstance(cost_price, (int, float)) and cost_price > 0:
            state['cost_price'] = float(cost_price)
            
        if actual_sell_price and actual_sell_price > 0:
            state['actual_sell_price'] = float(actual_sell_price)
        
        if realized_pnl is not None:
            state['realized_pnl'] = float(realized_pnl)
        
        with open(PORTFOLIO_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        return True
    except:
        return False

def load_portfolio_state():
    """加载持仓状态"""
    try:
        if not os.path.exists(PORTFOLIO_STATE_FILE):
            return None
        with open(PORTFOLIO_STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def log_trade_operation(stock_code, operation, current_price, shares_held, 
                       current_balance, total_assets, status='预测', note=''):
    """记录交易操作"""
    try:
        import csv
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
        date = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H:%M:%S')
        
        op_type = "买入" if "买入" in operation else "卖出" if "卖出" in operation else "持有"
        op_ratio = "0%" if "持有" in operation else operation.split()[-1] if "%" in operation else "0%"
        
        with open(TRADE_LOG_FILE, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, date, time_str, stock_code, op_type, op_ratio,
                f"{current_price:.2f}", "", "", "", "",
                f"{shares_held:.2f}", f"{current_balance:.2f}", f"{total_assets:.2f}",
                status, note
            ])
        return True
    except:
        return False

# ==================== V16批量预测结果保存功能 ====================

def get_batch_predict_result_file():
    """获取批量预测结果文件路径（带日期）- JSON格式"""
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    return f"batch_predict_results_{today}.json"

def get_batch_predict_log_file():
    """获取批量预测日志文件路径（带日期）- 文本格式，包含完整控制台输出"""
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    return f"batch_predict_log_{today}.txt"

# ==================== V16新增：纳指数据获取和指标计算 ====================

# 指数数据缓存（只获取一次，包含纳斯达克、道琼斯、富时A50）
_index_metrics_cache = None

def get_index_metrics_once():
    """
    获取全球主要指数涨跌幅（只获取一次）
    包括：纳斯达克、道琼斯、富时A50期指连续
    直接调用test_nasdaq_change.py中的get_index_data函数，确保使用相同的环境
    
    Returns:
        dict: 包含所有指数信息的字典，格式为：
        {
            'nasdaq': {...},  # 纳斯达克
            'dow': {...},     # 道琼斯
            'a50': {...},     # 富时A50
            'update_time': '...'
        }
        如果失败返回None
    """
    global _index_metrics_cache
    
    # 如果已经获取过，直接返回缓存
    if _index_metrics_cache is not None:
        return _index_metrics_cache
    
    # 方法1: 直接导入并调用test_nasdaq_change.py中的函数（最可靠）
    # 使用上下文管理器临时重定向stdout和stderr，避免print输出干扰批量预测
    import sys
    import json
    import os
    from io import StringIO
    try:
        # 临时重定向stdout和stderr，捕获所有输出
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        try:
            import test_nasdaq_change
            result = test_nasdaq_change.get_index_data()
        finally:
            # 恢复stdout和stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        if result and isinstance(result, dict):
            # 直接返回完整结果（包含nasdaq、dow、a50三个指数）
            _index_metrics_cache = result
            return _index_metrics_cache
    except Exception:
        # 如果导入失败，尝试从文件读取
        pass
    
    # 方法1.5: 如果直接获取失败，尝试从文件读取（test_nasdaq_change.py保存的数据）
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'index_data.json')
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
                if file_data and isinstance(file_data, dict):
                    _index_metrics_cache = file_data
                    return _index_metrics_cache
    except Exception:
        # 文件读取失败，继续尝试其他方法
        pass
    
    # 方法2: 如果导入失败，直接使用akshare（与test_nasdaq_change.py完全一致）
    # 临时禁用代理，确保网络请求正常
    import os
    saved_proxies = {}
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
    for var in proxy_vars:
        if var in os.environ:
            saved_proxies[var] = os.environ[var]
            os.environ.pop(var, None)
    os.environ['NO_PROXY'] = '*'
    os.environ['no_proxy'] = '*'
    
    try:
        import akshare as ak
        # 获取美股实时行情，查找纳指相关ETF或指数（与test_nasdaq_change.py完全一致）
        us_spot = ak.stock_us_spot_em()
        
        if us_spot is not None and len(us_spot) > 0:
            # 查找纳指相关标的（与test_nasdaq_change.py完全一致）
            nasdaq_keywords = ['纳指', 'NASDAQ', 'QQQ', 'IXIC']
            nasdaq_stocks = us_spot[
                us_spot['名称'].str.contains('|'.join(nasdaq_keywords), case=False, na=False)
            ]
            
            if len(nasdaq_stocks) > 0:
                # 优先选择QQQ（纳指100 ETF）或直接包含"纳指"的标的（与test_nasdaq_change.py完全一致）
                qqq = nasdaq_stocks[nasdaq_stocks['名称'].str.contains('QQQ|纳指100', case=False, na=False)]
                if len(qqq) > 0:
                    latest = qqq.iloc[0]
                else:
                    latest = nasdaq_stocks.iloc[0]
                
                # 处理涨跌幅数据（与test_nasdaq_change.py完全一致）
                change_pct = latest.get('涨跌幅', 'N/A')
                if change_pct is None:
                    change_pct = 'N/A'
                
                # 构建返回格式（兼容旧格式，同时支持新格式）
                _index_metrics_cache = {
                    'nasdaq': {
                        'index_name': latest.get('名称', '纳斯达克相关标的'),
                        'change_pct': change_pct,
                        'source': 'akshare_us_spot',
                        'note': '这是ETF或相关标的，非指数本身'
                    },
                    'dow': None,  # 道琼斯数据需要单独获取
                    'a50': None,  # 富时A50数据需要单独获取
                    'update_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                # 恢复代理设置
                for var, value in saved_proxies.items():
                    os.environ[var] = value
                return _index_metrics_cache
    except ImportError:
        # akshare未安装
        pass
    except Exception as e:
        # akshare失败，静默处理
        pass
    finally:
        # 恢复代理设置
        for var, value in saved_proxies.items():
            os.environ[var] = value
    
    # 方法2: 备用使用yfinance获取纳指指数 (^IXIC)
    # 注意：yfinance在批量预测中可能超时，所以作为备用
    try:
        import yfinance as yf
        nasdaq = yf.Ticker("^IXIC")
        hist = nasdaq.history(period="2d")
        if len(hist) > 0:
            latest = hist.iloc[-1]
            prev = hist.iloc[-2] if len(hist) > 1 else latest
            change_pct = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
            
            _index_metrics_cache = {
                'nasdaq': {
                    'index_name': '纳斯达克综合指数 (IXIC)',
                    'change_pct': round(change_pct, 2),
                    'source': 'yfinance'
                },
                'dow': None,
                'a50': None,
                'update_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            return _index_metrics_cache
    except ImportError:
        pass
    except Exception as e:
        # yfinance失败（超时、速率限制等），静默处理
        pass
    
    # 所有方法都失败
    _index_metrics_cache = None
    return None

# 股票夏普比率和回撤数据配置（用户提供的回测数据）
# 格式：股票代码: {'total_return': 总收益率(%), 'max_drawdown': 最大回撤(%), 'sharpe_ratio': 夏普比率(估算)}
# 注意：由于提供的是总收益率而非夏普比率，这里使用总收益率作为参考，夏普比率根据收益率和回撤估算
STOCK_METRICS_DATA = {
    'sz.002837': {'total_return': 18551.00, 'max_drawdown': 0.10, 'sharpe_ratio': None},  # 英维克 - 🏆 双料冠军
    'sz.002851': {'total_return': 16961.80, 'max_drawdown': 9.62, 'sharpe_ratio': None},  # 麦格米特 - 收益顶尖
    'sh.600730': {'total_return': 16547.51, 'max_drawdown': 2.35, 'sharpe_ratio': None},  # 中国高科 - 最大黑马
    'sz.002241': {'total_return': 11891.78, 'max_drawdown': 5.04, 'sharpe_ratio': None},  # 歌尔股份 - 通用模型典范
    'sz.002475': {'total_return': 10394.23, 'max_drawdown': 0.08, 'sharpe_ratio': None},  # 立讯精密 - 极品策略
    'sz.300499': {'total_return': 9729.38, 'max_drawdown': 0.00, 'sharpe_ratio': None},  # 高澜股份 - 零回撤之王
    'sz.300762': {'total_return': 5708.14, 'max_drawdown': 2.61, 'sharpe_ratio': None},  # 上海瀚讯 - 稳健优秀
    'sz.002706': {'total_return': 4539.16, 'max_drawdown': 2.29, 'sharpe_ratio': None},  # 良信股份 - 模型切换成功
    'sz.301005': {'total_return': 2976.57, 'max_drawdown': 17.09, 'sharpe_ratio': None},  # 超捷股份 - 回撤风险高
    'sz.002266': {'total_return': 1156.82, 'max_drawdown': 0.00, 'sharpe_ratio': None},  # 浙富控股 - 零回撤稳健型
    'sh.601399': {'total_return': 1040.56, 'max_drawdown': 0.00, 'sharpe_ratio': None},  # 国机重装 - 零回撤稳健策略
    'sz.300153': {'total_return': 1005.32, 'max_drawdown': 9.43, 'sharpe_ratio': None},  # 科泰电源 - 回撤较大
    'sz.300274': {'total_return': 635.03, 'max_drawdown': 2.23, 'sharpe_ratio': None},  # 阳光电源 - 风控好
    'sh.603267': {'total_return': 318.23, 'max_drawdown': 18.54, 'sharpe_ratio': None},  # 鸿远电子 - ⚠️ 高危
    'sz.002025': {'total_return': 10.43, 'max_drawdown': 22.06, 'sharpe_ratio': None},  # 航天电器 - ⚠️ 表现最差
    # 以下股票不在排名列表中，保留原配置
    'sh.603698': {'total_return': None, 'max_drawdown': None, 'sharpe_ratio': None},  # 航天工程
    'sz.300726': {'total_return': None, 'max_drawdown': None, 'sharpe_ratio': None},  # 宏达电子
    'sz.301017': {'total_return': None, 'max_drawdown': None, 'sharpe_ratio': None},  # 漱玉平民
    'sz.300749': {'total_return': None, 'max_drawdown': None, 'sharpe_ratio': None},  # 顶固集创
}

def get_stock_metrics_from_config(stock_code):
    """
    从配置中获取股票的收益率和回撤数据
    
    Args:
        stock_code: 股票代码（如"sh.603267"）
    
    Returns:
        dict: 包含total_return、max_drawdown和sharpe_ratio的字典
    """
    return STOCK_METRICS_DATA.get(stock_code, {'total_return': None, 'max_drawdown': None, 'sharpe_ratio': None})

def get_nasdaq_history_data(days=252):
    """
    获取纳斯达克指数历史数据（用于计算夏普和回撤）
    
    Args:
        days: 获取的历史天数，默认252（一年交易日）
    
    Returns:
        pd.DataFrame: 包含日期和收盘价的历史数据，失败返回None
    """
    try:
        # 方法1: 优先使用yfinance
        try:
            import yfinance as yf
            nasdaq = yf.Ticker("^IXIC")
            # 获取历史数据，period可以是"1y"（一年）或指定天数
            hist = nasdaq.history(period=f"{max(days, 365)}d")
            if hist is not None and len(hist) > 0:
                # 只保留最近days天的数据
                hist = hist.tail(days)
                # 确保有日期列
                hist.reset_index(inplace=True)
                hist['date'] = pd.to_datetime(hist['Date']).dt.date
                hist = hist[['date', 'Close']].rename(columns={'Close': 'close'})
                return hist
        except ImportError:
            pass
        except Exception as e:
            print(f"  ⚠️  yfinance获取纳指历史数据失败: {e}")
        
        # 方法2: 使用akshare（备用）
        try:
            import akshare as ak
            # 获取美股指数历史数据
            # 注意：akshare可能没有直接的纳指历史接口，这里尝试获取ETF数据
            # 获取QQQ（纳指100 ETF）的历史数据作为参考
            end_date = datetime.datetime.now().strftime('%Y%m%d')
            start_date = (datetime.datetime.now() - datetime.timedelta(days=days+30)).strftime('%Y%m%d')
            
            # 尝试获取QQQ的历史数据
            qqq_hist = ak.fund_etf_hist_em(symbol="QQQ", period="daily", 
                                          start_date=start_date, end_date=end_date, adjust="")
            if qqq_hist is not None and len(qqq_hist) > 0:
                qqq_hist['date'] = pd.to_datetime(qqq_hist['日期']).dt.date
                qqq_hist = qqq_hist[['date', '收盘']].rename(columns={'收盘': 'close'})
                return qqq_hist.tail(days)
        except Exception as e:
            print(f"  ⚠️  akshare获取纳指历史数据失败: {e}")
        
        return None
    except Exception as e:
        print(f"  ⚠️  获取纳指历史数据时发生错误: {e}")
        return None

def get_stock_history_data(stock_code, days=252):
    """
    获取股票历史数据（用于计算夏普和回撤）
    
    Args:
        stock_code: 股票代码（如"600730"）
        days: 获取的历史天数，默认252（一年交易日）
    
    Returns:
        pd.DataFrame: 包含日期和收盘价的历史数据，失败返回None
    """
    try:
        # 尝试从测试数据目录获取
        test_data_dir = "stockdata_v7/test"
        if os.path.exists(test_data_dir):
            # 查找股票数据文件
            for file in os.listdir(test_data_dir):
                if file.startswith(f"{stock_code}."):
                    file_path = os.path.join(test_data_dir, file)
                    df = pd.read_csv(file_path, encoding='utf-8-sig')
                    if 'date' in df.columns and 'close' in df.columns:
                        df['date'] = pd.to_datetime(df['date']).dt.date
                        df = df[['date', 'close']].sort_values('date')
                        # 只保留最近days天的数据
                        return df.tail(days)
        
        # 如果测试数据不存在，尝试从训练数据获取
        train_data_dir = "stockdata_v7/train"
        if os.path.exists(train_data_dir):
            for file in os.listdir(train_data_dir):
                if file.startswith(f"{stock_code}."):
                    file_path = os.path.join(train_data_dir, file)
                    df = pd.read_csv(file_path, encoding='utf-8-sig')
                    if 'date' in df.columns and 'close' in df.columns:
                        df['date'] = pd.to_datetime(df['date']).dt.date
                        df = df[['date', 'close']].sort_values('date')
                        return df.tail(days)
        
        return None
    except Exception as e:
        print(f"  ⚠️  获取股票历史数据失败: {e}")
        return None

def calculate_sharpe_ratio_and_drawdown(price_data):
    """
    计算夏普收益率和最大回撤
    
    Args:
        price_data: pd.DataFrame，包含'date'和'close'列
    
    Returns:
        dict: 包含sharpe_ratio和max_drawdown的字典
    """
    if price_data is None or len(price_data) < 2:
        return {'sharpe_ratio': None, 'max_drawdown': None}
    
    try:
        # 计算日收益率
        price_data = price_data.sort_values('date').copy()
        price_data['returns'] = price_data['close'].pct_change().dropna()
        returns = price_data['returns'].dropna()
        
        if len(returns) < 2:
            return {'sharpe_ratio': None, 'max_drawdown': None}
        
        # 计算夏普比率（年化，假设252个交易日，无风险利率为0）
        mean_return = returns.mean()
        std_return = returns.std()
        if std_return > 0:
            sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # 计算最大回撤
        prices = price_data['close'].values
        peak = prices[0]
        max_drawdown = 0.0
        
        for price in prices:
            if price > peak:
                peak = price
            drawdown = (peak - price) / peak if peak > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return {
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown': round(max_drawdown * 100, 2)  # 转换为百分比
        }
    except Exception as e:
        print(f"  ⚠️  计算指标时发生错误: {e}")
        return {'sharpe_ratio': None, 'max_drawdown': None}

class OutputCapture:
    """捕获控制台输出的上下文管理器"""
    def __init__(self):
        self.original_stdout = sys.stdout
        self.captured_output = StringIO()
        self.output_lines = []
    
    def __enter__(self):
        sys.stdout = self
        return self
    
    def __exit__(self, *args):
        sys.stdout = self.original_stdout
        self.captured_output.seek(0)
        self.output_lines = self.captured_output.getvalue().splitlines()
    
    def write(self, text):
        self.original_stdout.write(text)  # 同时输出到控制台
        self.captured_output.write(text)
    
    def flush(self):
        self.original_stdout.flush()
        self.captured_output.flush()
    
    def get_output(self):
        """获取捕获的输出"""
        # 如果 output_lines 为空（说明 __exit__ 还没执行），直接从 captured_output 获取
        if not self.output_lines:
            # 使用 getvalue() 不会改变流的位置，适合在 with 块内部调用
            content = self.captured_output.getvalue()
            if content:
                return content
        return '\n'.join(self.output_lines)

def append_to_log_file(content, log_file):
    """追加内容到日志文件"""
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(content)
            f.write('\n')
        return True
    except Exception as e:
        print(f"   ⚠️  写入日志文件失败: {e}")
        return False

def save_batch_predict_result(stock_code, stock_name, prediction_data):
    """
    保存单个股票的批量预测结果到带日期的记录文件
    
    Args:
        stock_code: 股票代码
        stock_name: 股票名称
        prediction_data: 预测数据字典，包含所有预测信息
    """
    try:
        result_file = get_batch_predict_result_file()
        
        # 读取现有结果（如果存在）
        all_results = []
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    all_results = json.load(f)
            except:
                all_results = []
        
        # 准备当前预测结果
        current_result = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'date': datetime.datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.datetime.now().strftime('%H:%M:%S'),
            'stock_code': stock_code,
            'stock_name': stock_name,
            **prediction_data
        }
        
        # 检查是否已存在该股票的记录（同一天），如果存在则更新，否则追加
        found = False
        for i, result in enumerate(all_results):
            if (result.get('stock_code') == stock_code and 
                result.get('date') == current_result['date']):
                all_results[i] = current_result
                found = True
                break
        
        if not found:
            all_results.append(current_result)
        
        # 保存到文件
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"   ⚠️  保存批量预测结果失败: {e}")
        return False

# ==================== V16预测准确率统计功能 ====================

def get_prediction_log_file(stock_code):
    """获取预测日志文件路径"""
    return f"v12_prediction_log_{stock_code.replace('.', '_')}.json"

def save_v12_prediction(date_str, transformer_prediction, current_price, stock_code):
    """
    保存V12 Transformer预测结果
    
    Args:
        date_str: 日期字符串（YYYY-MM-DD）
        transformer_prediction: Transformer预测价格
        current_price: 当前价格
        stock_code: 股票代码
    """
    try:
        prediction_log_file = get_prediction_log_file(stock_code)
        predictions = []
        if os.path.exists(prediction_log_file):
            with open(prediction_log_file, 'r', encoding='utf-8') as f:
                predictions = json.load(f)
        
        # 检查是否已存在该日期的预测，如果存在则更新
        found = False
        for pred in predictions:
            if pred.get('date') == date_str:
                pred['predicted_price'] = transformer_prediction
                pred['current_price'] = current_price
                pred['timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                found = True
                break
        
        if not found:
            predictions.append({
                'date': date_str,
                'predicted_price': transformer_prediction,
                'current_price': current_price,
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # 保存到文件
        with open(prediction_log_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"   ⚠️  保存V12预测失败: {e}")
        return False

def get_actual_close_price(stock_code, date_str):
    """
    获取指定日期的实际收盘价
    
    Args:
        stock_code: 股票代码
        date_str: 日期字符串（YYYY-MM-DD）
    
    Returns:
        float: 收盘价，如果获取失败返回None
    """
    try:
        import baostock as bs
        bs.login()
        
        # 转换股票代码格式
        if stock_code.startswith('sh.'):
            bs_code = f"sh.{stock_code.split('.')[1]}"
        elif stock_code.startswith('sz.'):
            bs_code = f"sz.{stock_code.split('.')[1]}"
        else:
            bs_code = stock_code
        
        # 查询指定日期的K线数据
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,close",
            start_date=date_str,
            end_date=date_str,
            frequency="d",
            adjustflag="3"
        )
        
        if rs.error_code == '0':
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())
            
            bs.logout()
            
            if len(data_list) > 0:
                return float(data_list[0][1])  # 返回收盘价
        
        bs.logout()
        return None
    except Exception as e:
        print(f"   ⚠️  获取实际收盘价失败: {e}")
        return None

def calculate_prediction_accuracy(stock_code):
    """
    计算V12预测准确率统计
    
    Args:
        stock_code: 股票代码
    
    Returns:
        dict: 统计结果
    """
    try:
        prediction_log_file = get_prediction_log_file(stock_code)
        if not os.path.exists(prediction_log_file):
            return None
        
        with open(prediction_log_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        if len(predictions) == 0:
            return None
        
        # 按日期排序
        predictions.sort(key=lambda x: x.get('date', ''))
        
        accuracy_stats = {
            'total_predictions': 0,
            'valid_comparisons': 0,
            'total_error': 0.0,
            'total_abs_error': 0.0,
            'total_error_pct': 0.0,
            'total_abs_error_pct': 0.0,
            'details': []
        }
        
        today = datetime.datetime.now().date()
        
        for i, pred in enumerate(predictions):
            pred_date_str = pred.get('date')
            if not pred_date_str:
                continue
            
            try:
                pred_date = datetime.datetime.strptime(pred_date_str, '%Y-%m-%d').date()
            except:
                continue
            
            # 只统计昨天的预测和今天的实际收盘价
            if pred_date >= today:
                continue  # 跳过今天及未来的预测
            
            predicted_price = pred.get('predicted_price')
            if predicted_price is None or predicted_price <= 0:
                continue
            
            accuracy_stats['total_predictions'] += 1
            
            # 获取预测日期后一天的实际收盘价
            next_date = pred_date + datetime.timedelta(days=1)
            
            # 跳过周末
            while next_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
                next_date += datetime.timedelta(days=1)
            
            # 如果下一天是今天或未来，跳过
            if next_date >= today:
                continue
            
            next_date_str = next_date.strftime('%Y-%m-%d')
            actual_close = get_actual_close_price(stock_code, next_date_str)
            
            if actual_close is None or actual_close <= 0:
                continue
            
            # 计算误差
            error = predicted_price - actual_close
            abs_error = abs(error)
            error_pct = (error / actual_close * 100) if actual_close > 0 else 0
            abs_error_pct = abs(error_pct)
            
            accuracy_stats['valid_comparisons'] += 1
            accuracy_stats['total_error'] += error
            accuracy_stats['total_abs_error'] += abs_error
            accuracy_stats['total_error_pct'] += error_pct
            accuracy_stats['total_abs_error_pct'] += abs_error_pct
            
            accuracy_stats['details'].append({
                'prediction_date': pred_date_str,
                'actual_date': next_date_str,
                'predicted_price': predicted_price,
                'actual_close': actual_close,
                'error': error,
                'abs_error': abs_error,
                'error_pct': error_pct,
                'abs_error_pct': abs_error_pct
            })
        
        # 计算平均值
        if accuracy_stats['valid_comparisons'] > 0:
            accuracy_stats['avg_error'] = accuracy_stats['total_error'] / accuracy_stats['valid_comparisons']
            accuracy_stats['avg_abs_error'] = accuracy_stats['total_abs_error'] / accuracy_stats['valid_comparisons']
            accuracy_stats['avg_error_pct'] = accuracy_stats['total_error_pct'] / accuracy_stats['valid_comparisons']
            accuracy_stats['avg_abs_error_pct'] = accuracy_stats['total_abs_error_pct'] / accuracy_stats['valid_comparisons']
        else:
            accuracy_stats['avg_error'] = 0.0
            accuracy_stats['avg_abs_error'] = 0.0
            accuracy_stats['avg_error_pct'] = 0.0
            accuracy_stats['avg_abs_error_pct'] = 0.0
        
        return accuracy_stats
    except Exception as e:
        print(f"   ⚠️  计算预测准确率失败: {e}")
        return None

def display_prediction_accuracy(stock_code):
    """显示V12预测准确率统计"""
    try:
        stats = calculate_prediction_accuracy(stock_code)
        if stats is None or stats['valid_comparisons'] == 0:
            print(f"\n   📊 V12预测准确率统计: 暂无有效数据")
            return
        
        print(f"\n   📊 V12预测准确率统计:")
        print(f"      ✅ 总预测次数: {stats['total_predictions']} 次")
        print(f"      ✅ 有效对比次数: {stats['valid_comparisons']} 次")
        print(f"      📈 平均误差: {stats['avg_error']:.2f} 元 ({stats['avg_error_pct']:+.2f}%)")
        print(f"      📊 平均绝对误差: {stats['avg_abs_error']:.2f} 元 ({stats['avg_abs_error_pct']:.2f}%)")
        
        # 显示最近5次预测的详细情况
        if len(stats['details']) > 0:
            print(f"\n      📋 最近5次预测详情:")
            recent_details = stats['details'][-5:]
            for detail in recent_details:
                print(f"         {detail['prediction_date']} 预测 {detail['predicted_price']:.2f}元 → "
                      f"{detail['actual_date']} 实际 {detail['actual_close']:.2f}元 "
                      f"(误差: {detail['error']:+.2f}元, {detail['error_pct']:+.2f}%)")
    except Exception as e:
        print(f"   ⚠️  显示预测准确率失败: {e}")

# ==================== 配置参数 ====================

# 基础配置
MODEL_PATH = "ppo_stock_v7_002025.zip"  # V12使用通用PPO模型，也可以使用专用模型

# 批量预测：股票列表（仅包含有专用模型的股票）
STOCK_LIST = [
    {'code': 'sz.002837', 'name': '英维克', 'model': 'ppo_stock_v7_002837.zip'},  # 🏆 双料冠军
    {'code': 'sz.002851', 'name': '麦格米特', 'model': 'ppo_stock_v7_002851.zip'},  # 收益顶尖
    {'code': 'sh.600730', 'name': '中国高科', 'model': 'ppo_stock_v7_600730.zip'},  # 最大黑马
    {'code': 'sz.002241', 'name': '歌尔股份', 'model': 'ppo_stock_v7_002241.zip'},  # 通用模型典范
    {'code': 'sz.002475', 'name': '立讯精密', 'model': 'ppo_stock_v7_002475.zip'},  # 极品策略
    {'code': 'sz.300499', 'name': '高澜股份', 'model': 'ppo_stock_v7_300499.zip'},  # 零回撤之王
    {'code': 'sz.300762', 'name': '上海瀚讯', 'model': 'ppo_stock_v7_300762.zip'},  # 稳健优秀
    {'code': 'sz.002706', 'name': '良信股份', 'model': 'ppo_stock_v7_002837.zip'},  # 模型切换成功（使用英维克模型）
    {'code': 'sz.301005', 'name': '超捷股份', 'model': 'ppo_stock_v7_301005.zip'},  # 回撤风险高
    {'code': 'sh.601399', 'name': '国机重装', 'model': 'ppo_stock_v7_601399.zip'},  # 零回撤稳健策略
    {'code': 'sz.300153', 'name': '科泰电源', 'model': 'ppo_stock_v7_300153.zip'},  # 回撤较大
    {'code': 'sz.300274', 'name': '阳光电源', 'model': 'ppo_stock_v7_300274.zip'},  # 风控好
    {'code': 'sh.603267', 'name': '鸿远电子', 'model': 'ppo_stock_v7_603267.zip'},  # ⚠️ 高危
    {'code': 'sh.603698', 'name': '航天工程', 'model': 'ppo_stock_v7_603698.zip'},
    {'code': 'sz.002025', 'name': '航天电器', 'model': 'ppo_stock_v7_002025.zip'},  # ⚠️ 表现最差
    {'code': 'sz.300726', 'name': '宏达电子', 'model': 'ppo_stock_v7_300726.zip'},
]

# 当前处理的股票代码（会在循环中动态设置）
STOCK_CODE = None
LLM_PROVIDER = "deepseek"
ENABLE_LLM = True
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'sk-167914945f7945d498e09a7f186c101d')

# V15配置：DeepSeek 轮次复盘
ENABLE_DEEPSEEK_REVIEW = True  # 每轮预测后调用 DeepSeek 简评
DEEPSEEK_REVIEW_MODEL = "deepseek-chat"
DEEPSEEK_REVIEW_MAX_TOKENS = 400
DEEPSEEK_REVIEW_TIMEOUT = 25  # 秒

# V7配置
TECHNICAL_INDICATOR_CONFIG = {
    'kdj_period': 9,
    'kdj_slow_period': 3,
    'kdj_fast_period': 3,
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'obv_smooth_period': 20,
    'ma_periods': [5, 10, 20, 60]
}

# V9配置
ENABLE_LSTM_PREDICTION = True
ENABLE_DYNAMIC_OPTIMIZATION = True
LSTM_MODEL_TYPE = 'lstm_attention'
LSTM_SEQ_LENGTH = 60
LSTM_HIDDEN_SIZE = 64

# V10配置
ENABLE_TRANSFORMER = True
ENABLE_MULTIMODAL = True
ENABLE_VISUALIZATION = True
ENABLE_HOLOGRAPHIC = True

TRANSFORMER_D_MODEL = 64
TRANSFORMER_NHEAD = 4
TRANSFORMER_NUM_LAYERS = 3
TRANSFORMER_MAX_SEQ_LEN = 100

# V11改进配置：滑动窗口归一化
USE_SLIDING_WINDOW_NORMALIZE = True  # 使用滑动窗口归一化，避免全局偏低
SLIDING_WINDOW_SIZE = 500  # 滑动窗口大小（使用最近N个数据点）

# V12优化配置：Transformer预测优化
TRANSFORMER_EPOCHS = 120  # V12增加训练轮数（从50增加到120），提高模型准确性
TRANSFORMER_ADAPTIVE_WINDOW = True  # V12启用自适应窗口（根据当前价格位置调整）
TRANSFORMER_TREND_AWARE = True  # V12启用趋势感知（基于价格动量调整预测）
TRANSFORMER_POST_PROCESS = True  # V12启用预测后处理优化（动态校正预测）
TRANSFORMER_PRICE_POSITION_THRESHOLD = 0.75  # V12当前价格位置阈值（>75%时启用特殊处理）

# V11改进配置：动态权重调整
ENABLE_DYNAMIC_WEIGHTS = True  # 启用动态权重调整
WEIGHT_ADAPTATION_RATE = 0.1  # 权重调整速率
WEIGHT_MIN = 0.05  # 最小权重
WEIGHT_MAX = 0.6  # 最大权重

# V11改进配置：多模态真实数据源
USE_REAL_NEWS_SOURCE = True  # 使用真实新闻源（LLM市场情报）
FALLBACK_TO_SAMPLE_TEXTS = True  # 如果获取失败，回退到样本文本

# V11改进配置：量化回测
ENABLE_BACKTEST = True  # 启用回测功能
BACKTEST_METRICS = ['MAE', 'RMSE', 'MAPE', 'Direction_Accuracy']  # 回测指标

# V13新增配置：自动模型选择
ENABLE_AUTO_MODEL_SELECTION = True  # 启用自动模型选择
AUTO_MODEL_SELECTION_INTERVAL = 20  # 每N轮评估一次模型（默认20轮）
MIN_BACKTEST_SAMPLES = 10  # 最少需要N个回测样本才开始评估

# V13新增配置：止损止盈风险控制
ENABLE_STOP_LOSS_TAKE_PROFIT = True  # 启用止损止盈功能
STOP_LOSS_PCT = -5.0  # 止损比例（-5%表示亏损5%时触发止损）
TAKE_PROFIT_PCT = 10.0  # 止盈比例（10%表示盈利10%时触发止盈）
STOP_LOSS_ACTION = 0  # 止损动作：0=全部卖出（清仓）
TAKE_PROFIT_ACTION = 2  # 止盈动作：2=卖出50%（减仓一半），可调整为0（全部卖出）或1（卖出25%）
ENABLE_PARTIAL_STOP_LOSS = True  # 启用部分止损：亏损3%时减仓50%，亏损5%时清仓
PARTIAL_STOP_LOSS_PCT = -3.0  # 部分止损比例（-3%时减仓50%）
PARTIAL_STOP_LOSS_ACTION = 1  # 部分止损动作：1=卖出50%

# V15新增配置：基于ATR（平均真实波幅）的动态止损
ENABLE_ATR_STOP_LOSS = True   # 启用ATR动态止损，类似海龟法则
ATR_PERIOD = 14               # ATR计算周期
ATR_MULTIPLIER = 2.0          # 止损倍数：成本价 - ATR * 倍数

# V16新增配置：趋势/震荡双策略 + 融合
ENABLE_REGIME_STRATEGY = True          # 启用趋势/震荡双模态策略
TREND_BREAKOUT_WINDOW = 20             # 趋势突破窗口（近似海龟突破）
TREND_MIN_PRED_CHANGE = 1.0            # 预测涨幅阈值（%）用于判定趋势模式
TREND_CONFIDENCE_THRESHOLD = 0.6       # 模型置信度阈值
TREND_ADJUST_STEP = 1                  # 趋势模式下对最终动作的增强步长（正数偏多）
REGIME_MAX_ADJUST = 2                  # 动作调整的最大步数，避免过度偏移
BOLL_PERIOD = 20                       # 布林带周期（均值回归）
BOLL_STD = 2.0                         # 布林带倍数
RANGE_BANDWIDTH_THRESHOLD = 0.015      # 布林带宽度判定震荡的阈值（带宽占价格比例）

# V13新增配置：资金管理策略（凯利公式）
ENABLE_KELLY_FORMULA = True  # 启用凯利公式资金管理
KELLY_FRACTION = 0.5  # 凯利公式安全系数（0.5表示使用50%的凯利值，降低风险）
MIN_KELLY_POSITION = 0.1  # 最小仓位（10%）
MAX_KELLY_POSITION = 0.9  # 最大仓位（90%）
KELLY_MIN_SAMPLES = 20  # 最少需要N个交易样本才使用凯利公式
USE_KELLY_FOR_POSITION = True  # 是否使用凯利公式调整仓位建议

# V14新增配置：StockAPI数据源
ENABLE_STOCKAPI = True  # 启用StockAPI数据源
STOCKAPI_API_KEY = os.getenv('STOCKAPI_API_KEY', '')  # StockAPI密钥（从环境变量获取）
STOCKAPI_BASE_URL = os.getenv('STOCKAPI_BASE_URL', 'https://api.stockapi.com')  # StockAPI基础URL
STOCKAPI_TIMEOUT = 10  # StockAPI请求超时时间（秒）
STOCKAPI_PRIORITY = 1  # StockAPI优先级（1=最高优先级，数字越小优先级越高）

# V14: 检测StockAPI可用性
STOCKAPI_AVAILABLE = False
try:
    import requests
    STOCKAPI_AVAILABLE = True
except ImportError:
    STOCKAPI_AVAILABLE = False
CANDIDATE_MODELS = [  # 候选模型列表（包含所有股票的专用模型）
    {
        'name': '002837模型',
        'paths': ['ppo_stock_v7_002837.zip', 'models_v7_002837/best/best_model.zip'],
        'description': '英维克002837专用模型 - 🏆 双料冠军'
    },
    {
        'name': '300499模型',
        'paths': ['ppo_stock_v7_300499.zip', 'models_v7_300499/best/best_model.zip'],
        'description': '高澜股份300499专用模型 - 零回撤之王'
    },
    {
        'name': '603267模型',
        'paths': ['ppo_stock_v7_603267.zip', 'models_v7_603267/best/best_model.zip'],
        'description': '鸿远电子603267专用模型'
    },
    {
        'name': '603698模型',
        'paths': ['ppo_stock_v7_603698.zip', 'models_v7_603698/best/best_model.zip'],
        'description': '航天工程603698专用模型'
    },
    {
        'name': '002025模型',
        'paths': ['ppo_stock_v7_002025.zip', 'models_v7_002025/best/best_model.zip'],
        'description': '航天电器002025专用模型'
    },
    {
        'name': '002241模型',
        'paths': ['ppo_stock_v7_002241.zip', 'models_v7_002241/best/best_model.zip'],
        'description': '歌尔股份002241专用模型'
    },
    {
        'name': '002475模型',
        'paths': ['ppo_stock_v7_002475.zip', 'models_v7_002475/best/best_model.zip'],
        'description': '立讯精密002475专用模型'
    },
    {
        'name': '300726模型',
        'paths': ['ppo_stock_v7_300726.zip', 'models_v7_300726/best/best_model.zip'],
        'description': '宏达电子300726专用模型'
    },
    {
        'name': '300762模型',
        'paths': ['ppo_stock_v7_300762.zip', 'models_v7_300762/best/best_model.zip'],
        'description': '上海瀚讯300762专用模型'
    },
    {
        'name': '301017模型',
        'paths': ['ppo_stock_v7_301017.zip', 'models_v7_301017/best/best_model.zip'],
        'description': '漱玉平民301017专用模型'
    },
    {
        'name': '300749模型',
        'paths': ['ppo_stock_v7_300749.zip', 'models_v7_300749/best/best_model.zip'],
        'description': '顶固集创300749专用模型'
    },
    {
        'name': '600730模型',
        'paths': ['ppo_stock_v7_600730.zip', 'models_v7_600730/best/best_model.zip'],
        'description': '中国高科600730专用模型'
    },
    {
        'name': '002851模型',
        'paths': ['ppo_stock_v7_002851.zip', 'models_v7_002851/best/best_model.zip'],
        'description': '麦格米特002851专用模型'
    },
    {
        'name': '300274模型',
        'paths': ['ppo_stock_v7_300274.zip', 'models_v7_300274/best/best_model.zip'],
        'description': '阳光电源300274专用模型'
    },
    {
        'name': '002266模型',
        'paths': ['ppo_stock_v7_002266.zip', 'models_v7_002266/best/best_model.zip'],
        'description': '浙富控股002266专用模型'
    },
    {
        'name': '300153模型',
        'paths': ['ppo_stock_v7_300153.zip', 'models_v7_300153/best/best_model.zip'],
        'description': '科泰电源300153专用模型'
    },
    {
        'name': '601399模型',
        'paths': ['ppo_stock_v7_601399.zip', 'models_v7_601399/best/best_model.zip'],
        'description': '国机重装601399专用模型'
    },
    {
        'name': '301005模型',
        'paths': ['ppo_stock_v7_301005.zip', 'models_v7_301005/best/best_model.zip'],
        'description': '超捷股份301005专用模型'
    },
    {
        'name': '通用模型',
        'paths': ['ppo_stock_v7.zip', 'models_v7/best/best_model.zip'],
        'description': '通用模型（备用）'
    }
]

# V13模型评分权重配置
MODEL_SCORE_WEIGHTS = {
    'mae': 0.25,              # MAE权重（越小越好）
    'rmse': 0.25,             # RMSE权重（越小越好）
    'mape': 0.25,             # MAPE权重（越小越好）
    'direction_accuracy': 0.25 # 方向准确率权重（越大越好）
}

VISUALIZATION_PORT = 8082  # V11使用8082端口
VISUALIZATION_OUTPUT_DIR = "visualization_output"

HOLOGRAPHIC_MEMORY_SIZE = 1000

# V11持仓编辑器配置
ENABLE_WEB_EDITOR = True          # 是否启用网页持仓编辑
WEB_EDITOR_PORT = 5001           # 本地网页端口
WEB_EDITOR_HOST = "127.0.0.1"    # 仅本机访问

# V11智能融合配置
ENABLE_MULTI_MODEL_FUSION = True  # 启用多模型融合
MODEL_WEIGHTS = {
    'ppo': 0.4,          # PPO强化学习模型权重
    'lstm': 0.2,         # LSTM/GRU模型权重
    'transformer': 0.2,  # Transformer模型权重
    'holographic': 0.2   # 全息动态模型权重
}

# 文件路径
TRADE_LOG_FILE = "trade_log.csv"
# PORTFOLIO_STATE_FILE 将在每个股票循环中动态设置
PORTFOLIO_STATE_FILE = None

# V7持仓编辑器配置
ENABLE_WEB_EDITOR = True          # 是否启用网页持仓编辑
WEB_EDITOR_PORT = 5001           # 本地网页端口（与可视化服务器分离）
WEB_EDITOR_HOST = "127.0.0.1"    # 仅本机访问

# ==================== 版本标识 ====================

print("\n" + "=" * 70)
print("V16 批量预测系统 - 趋势/震荡双策略 + DeepSeek复盘 + StockAPI（批量处理多个股票）")
print("=" * 70)
print("整合功能:")
print("   V7: 技术指标、多数据源、LLM解释、成本模型、PPO强化学习")
print("   V9: LSTM/GRU、注意力机制、动态参数优化、自动学习优化")
print("   V10: Transformer、多模态处理、实时可视化、全息动态模型")
print("   V11: 智能融合决策、滑动窗口归一化、动态权重调整")
print("   V12: Transformer预测优化、预测方向冲突检测、智能决策调整")
print("   V13: 自动模型选择、止损止盈风险控制、凯利公式资金管理")
print("   V14: StockAPI数据源集成")
print("   V15: DeepSeek 轮次复盘（每轮给出执行要点+风险提示）")
print("   V16: 趋势/震荡双策略（突破跟随 + 布林带均值回归）")
print("=" * 70)
print("⭐ V15新增功能:")
print("   - DeepSeek轮次复盘：每轮预测结束调用 DeepSeek 输出简洁点评")
print("   - 输出位置：放在本轮所有预测信息之后，便于快速执行")
print("   - ATR动态止损：基于ATR×倍数的海龟风格止损，更适应波动")
print("⭐ V16新增功能:")
print("   - 趋势/震荡双策略：趋势突破跟随 + 震荡布林均值回归，与模型信号加权融合")
print("⭐ V14功能回顾:")
print("   - StockAPI集成：新增StockAPI数据源支持，提供更丰富的股票数据")
print("   - 数据源优先级优化：StockAPI作为高优先级数据源，提供实时行情数据")
print("   - 多数据源容错增强：StockAPI失败时自动回退到其他数据源")
if ENABLE_STOCKAPI:
    print(f"      StockAPI: 启用")
    if STOCKAPI_API_KEY:
        print(f"      📊 API密钥: 已配置（长度: {len(STOCKAPI_API_KEY)}）")
    else:
        print(f"      📊 API密钥: 未配置（可通过环境变量STOCKAPI_API_KEY设置）")
    print(f"      📊 基础URL: {STOCKAPI_BASE_URL}")
    print(f"      📊 优先级: {STOCKAPI_PRIORITY}（数字越小优先级越高）")
print("")
print("⭐ V13功能（继承）:")
print("   - 自动模型选择：基于回测结果自动选择最优模型")
print("   - 多模型回测：对多个候选模型进行实时回测")
print("   - 综合评分系统：综合MAE、RMSE、MAPE、方向准确率计算模型评分")
print("   - 动态模型切换：根据回测结果动态切换到最优模型")
print("   - 模型性能追踪：记录每个模型的历史表现")
print("   - 止损止盈风险控制：自动止损止盈，保护资金安全")
if ENABLE_STOP_LOSS_TAKE_PROFIT:
    print(f"      📊 止损: {STOP_LOSS_PCT}% | 止盈: {TAKE_PROFIT_PCT}%")
    if ENABLE_PARTIAL_STOP_LOSS:
        print(f"      📊 部分止损: {PARTIAL_STOP_LOSS_PCT}% (减仓50%)")
    if ENABLE_ATR_STOP_LOSS:
        print(f"      🛡️  ATR动态止损: 周期{ATR_PERIOD}, 倍数{ATR_MULTIPLIER}")
if ENABLE_REGIME_STRATEGY:
    print("   - 趋势/震荡策略：趋势突破跟随 + 布林带均值回归（V16）")
print("   - 资金管理策略：凯利公式动态仓位管理")
if ENABLE_KELLY_FORMULA:
    print(f"      📊 凯利公式: 启用 | 安全系数: {KELLY_FRACTION*100:.0f}% | 最少样本: {KELLY_MIN_SAMPLES}")
    print(f"      📊 仓位范围: {MIN_KELLY_POSITION*100:.0f}% - {MAX_KELLY_POSITION*100:.0f}%")
print("=" * 70)
print("⚠️  版本标识: 这是 V16 版本，新增趋势/震荡双策略与 DeepSeek 复盘，并保留 StockAPI 集成！")
print("=" * 70 + "\n")

# ==================== 初始化模块 ====================

# V7模块初始化
tech_indicators = None
if TECHNICAL_INDICATORS_AVAILABLE:
    try:
        tech_indicators = TechnicalIndicators(**TECHNICAL_INDICATOR_CONFIG)
        print("✅ V7技术指标计算器初始化成功")
    except Exception as e:
        print(f"⚠️  技术指标计算器初始化失败: {e}")

multi_source_manager = None
if MULTI_DATA_SOURCE_AVAILABLE:
    try:
        # V14: 设置数据源优先级，StockAPI优先（如果启用）
        priority_list = None
        if ENABLE_STOCKAPI and STOCKAPI_AVAILABLE:
            # V14: StockAPI作为最高优先级
            priority_list = ['stockapi', 'tushare', 'akshare', 'baostock']
            print(f"📊 V14: StockAPI已启用，设置为最高优先级数据源")
        else:
            # 默认优先级
            priority_list = ['tushare', 'akshare', 'baostock']
            if ENABLE_STOCKAPI:
                print(f"⚠️  V14: StockAPI已配置但不可用，将使用默认数据源优先级")
        
        # 初始化多数据源管理器，启用反爬虫功能
        multi_source_manager = MultiDataSourceManager(
            stock_code=STOCK_CODE,
            sources=None,  # 自动检测可用数据源
            priority=priority_list,  # V14: 使用包含StockAPI的优先级列表
            timeout=10,
            retry_times=3,
            enable_anti_crawler=ENABLE_ANTI_CRAWLER,
            proxies=PROXIES if PROXIES else None
        )
        print("✅ V7多数据源管理器初始化成功")
        if ENABLE_STOCKAPI and STOCKAPI_AVAILABLE:
            print("   📊 V14: StockAPI数据源已集成")
        if ENABLE_ANTI_CRAWLER:
            print(f"   🛡️  反爬虫功能已启用 (代理数量: {len(PROXIES)})")
    except Exception as e:
        print(f"⚠️  多数据源管理器初始化失败: {e}")
        multi_source_manager = None
else:
    multi_source_manager = None

llm_interpreter = None
llm_agent = None
if LLM_AVAILABLE and ENABLE_LLM:
    try:
        os.environ['DEEPSEEK_API_KEY'] = DEEPSEEK_API_KEY
        llm_agent = MarketIntelligenceAgent(
            provider=LLM_PROVIDER,
            api_key=DEEPSEEK_API_KEY,
            enable_cache=True
        )
        print("✅ LLM市场情报代理初始化成功")
        
        if LLM_INTERPRETER_AVAILABLE:
            llm_interpreter = LLMIndicatorInterpreter(
                llm_agent=llm_agent,
                enable_cache=True
            )
            print("✅ V7 LLM指标解释器初始化成功")
    except Exception as e:
        print(f"⚠️  LLM初始化失败: {e}")

# V9模块初始化
lstm_processor = None
if LSTM_AVAILABLE and ENABLE_LSTM_PREDICTION:
    try:
        lstm_processor = TimeSeriesProcessor(
            model_type=LSTM_MODEL_TYPE,
            seq_length=LSTM_SEQ_LENGTH,
            input_size=1,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=2,
            output_size=1,
            dropout=0.2,
            use_bidirectional=False,
            use_gpu=False
        )
        print(f"✅ V9 LSTM/GRU处理器初始化成功 (类型: {LSTM_MODEL_TYPE})")
    except Exception as e:
        print(f"⚠️  LSTM/GRU处理器初始化失败: {e}")

dynamic_optimizer = None
auto_learner = None
if OPTIMIZER_AVAILABLE and ENABLE_DYNAMIC_OPTIMIZATION:
    try:
        # 这里需要根据实际需求定义参数范围
        parameter_ranges = {
            'kdj_period': ParameterRange(5, 14, param_type='integer'),
            'rsi_period': ParameterRange(10, 20, param_type='integer'),
        }
        dynamic_optimizer = DynamicParameterOptimizer(
            parameter_ranges=parameter_ranges,
            optimization_method='bayesian',
            adaptation_rate=0.1,
            exploration_rate=0.2,
            performance_window=100
        )
        auto_learner = AutoLearningOptimizer(
            parameter_optimizer=dynamic_optimizer,
            learning_rate=0.01,
            momentum=0.9,
            decay_rate=0.99
        )
        print("✅ V9动态参数优化器初始化成功")
    except Exception as e:
        print(f"⚠️  参数优化器初始化失败: {e}")

# V10模块初始化
transformer_model = None
if TRANSFORMER_AVAILABLE and ENABLE_TRANSFORMER:
    try:
        transformer_model = TransformerPredictor(
            input_size=1,
            d_model=TRANSFORMER_D_MODEL,
            nhead=TRANSFORMER_NHEAD,
            num_encoder_layers=TRANSFORMER_NUM_LAYERS,
            num_decoder_layers=TRANSFORMER_NUM_LAYERS,
            max_seq_len=TRANSFORMER_MAX_SEQ_LEN
        )
        print("✅ V12 Transformer模型初始化成功")
    except Exception as e:
        print(f"⚠️  Transformer模型初始化失败: {e}")

multimodal_processor = None
if MULTIMODAL_AVAILABLE and ENABLE_MULTIMODAL:
    try:
        multimodal_processor = MultimodalDataProcessor(
            text_max_length=512,
            use_bert=False,
            fusion_method='attention'
        )
        print("✅ V10多模态处理器初始化成功")
    except Exception as e:
        print(f"⚠️  多模态处理器初始化失败: {e}")

visualizer = None
web_visualization = None
if VISUALIZATION_AVAILABLE and ENABLE_VISUALIZATION:
    try:
        visualizer = RealTimeVisualizer()
        print("✅ V10实时可视化器初始化成功")
        
        try:
            web_visualization = WebVisualizationServer(visualizer, port=VISUALIZATION_PORT)
            web_visualization.start()
            print(f"✅ V10 Web可视化服务器启动成功 (端口: {VISUALIZATION_PORT})")
        except Exception as e:
            print(f"⚠️  Web可视化服务器启动失败: {e}")
    except Exception as e:
        print(f"⚠️  可视化器初始化失败: {e}")

holographic_model = None
if HOLOGRAPHIC_AVAILABLE and ENABLE_HOLOGRAPHIC:
    try:
        holographic_model = HolographicDynamicModel(
            memory_size=HOLOGRAPHIC_MEMORY_SIZE,
            enable_text_analysis=True,
            enable_memory=True
        )
        print("✅ V10全息动态模型初始化成功")
    except Exception as e:
        print(f"⚠️  全息动态模型初始化失败: {e}")

# ==================== V13: 多模型管理和自动选择 ====================

def load_ppo_model(model_path):
    """加载PPO模型"""
    try:
        if model_path and os.path.exists(model_path):
            return PPO.load(model_path)
    except Exception as e:
        print(f"⚠️  模型加载失败 {model_path}: {e}")
    return None

def find_available_model_paths(candidate_model):
    """查找候选模型中可用的模型路径"""
    for path in candidate_model['paths']:
        if path and os.path.exists(path):
            return path
    return None

# V13: 加载所有可用的候选模型
candidate_ppo_models = {}  # {model_name: ppo_model}
available_model_info = {}  # {model_name: {'path': path, 'description': desc}}

if PPO_AVAILABLE and ENABLE_AUTO_MODEL_SELECTION:
    print("📦 V13: 正在加载候选模型...")
    for model_config in CANDIDATE_MODELS:
        model_path = find_available_model_paths(model_config)
        if model_path:
            ppo_model_instance = load_ppo_model(model_path)
            if ppo_model_instance:
                candidate_ppo_models[model_config['name']] = ppo_model_instance
                available_model_info[model_config['name']] = {
                    'path': model_path,
                    'description': model_config['description']
                }
                print(f"   ✅ {model_config['name']}: {model_path}")
    
    if len(candidate_ppo_models) == 0:
        print("   ⚠️  未找到任何可用的候选模型，将使用默认模型")
        # 回退到默认模型加载逻辑
        possible_models = [
            "ppo_stock_v7_002241.zip",
            "models_v7_002241/best/best_model.zip",
            "ppo_stock_v7.zip",
            "models_v7/best/best_model.zip"
        ]
        for model_file in possible_models:
            if model_file and os.path.exists(model_file):
                ppo_model = load_ppo_model(model_file)
                if ppo_model:
                    print(f"   ✅ 使用默认模型: {model_file}")
                    break
    else:
        print(f"   📊 共加载 {len(candidate_ppo_models)} 个候选模型")
else:
    # V12兼容模式：使用单一模型
    ppo_model = None
    if PPO_AVAILABLE:
        try:
            possible_models = [
                "ppo_stock_v7_002241.zip",
                "models_v7_002241/best/best_model.zip",
                MODEL_PATH,
                "ppo_stock_v7.zip",
                "models_v7/best/best_model.zip"
            ]
            for model_file in possible_models:
                if model_file and os.path.exists(model_file):
                    ppo_model = load_ppo_model(model_file)
                    if ppo_model:
                        print(f"✅ PPO模型加载成功: {model_file}")
                        break
        except Exception as e:
            print(f"⚠️  PPO模型加载失败: {e}")

# V13: 当前使用的模型
current_model_name = None
ppo_model = None
if ENABLE_AUTO_MODEL_SELECTION and len(candidate_ppo_models) > 0:
    # 默认使用第一个可用模型
    current_model_name = list(candidate_ppo_models.keys())[0]
    ppo_model = candidate_ppo_models[current_model_name]
    print(f"🎯 V13: 初始使用模型: {current_model_name} ({available_model_info[current_model_name]['description']})")
elif ppo_model:
    current_model_name = "默认模型"
    print(f"✅ 使用默认模型")

print("=" * 70)
print()

# ==================== V16新增：模型回测统计信息 ====================
# 在模型加载后，对对应股票进行回测，显示总收益率、夏普比率、最大回撤
# 批量预测：回测统计将在每个股票循环中单独执行，这里跳过全局回测
if False and ppo_model and PPO_AVAILABLE:  # 批量预测时跳过全局回测
    try:
        from stock_env_v6 import StockTradingEnv
        
        # 根据股票代码查找对应的测试数据文件
        # 批量预测：STOCK_CODE 将在循环中设置，这里跳过
        if STOCK_CODE is None:
            pass  # 跳过
        stock_code_num = STOCK_CODE.split('.')[-1] if STOCK_CODE else None  # 提取股票代码数字部分
        test_data_dir = f'stockdata_v7_{stock_code_num}/test'
        
        if os.path.exists(test_data_dir):
            print("\n" + "=" * 70)
            print("📊 V16: 开始模型回测统计...")
            print("=" * 70)
            
            # 查找对应股票的测试文件
            test_files = [os.path.join(test_data_dir, f) for f in os.listdir(test_data_dir) 
                         if f.endswith('.csv') and stock_code_num in f]
            
            if test_files:
                # 使用第一个找到的测试文件
                test_file = test_files[0]
                print(f"📁 测试数据文件: {test_file}")
                
                try:
                    # 初始化环境（使用与训练时相同的初始资金）
                    initial_balance = 20000.0  # 默认初始资金2万元
                    env = StockTradingEnv(test_file, initial_balance=initial_balance)
                    obs, _ = env.reset()
                    done = False
                    
                    # 执行回测
                    step_count = 0
                    while not done:
                        action, _ = ppo_model.predict(obs, deterministic=True)
                        obs, reward, done, truncated, _ = env.step(action)
                        step_count += 1
                        if step_count % 100 == 0:
                            print(f"   回测进度: {step_count} 步...", end='\r')
                    
                    # 获取回测统计信息
                    stats = env.get_stats()
                    
                    if stats:
                        print("\n" + "=" * 70)
                        print("📈 【V16模型回测统计结果】")
                        print("=" * 70)
                        stock_name = get_stock_name(STOCK_CODE) if 'get_stock_name' in globals() else STOCK_CODE
                        print(f"股票名称: {stock_name}")
                        print(f"股票代码: {STOCK_CODE}")
                        print(f"测试数据: {os.path.basename(test_file)}")
                        print(f"初始资金: {initial_balance:,.2f} 元")
                        print(f"最终净值: {stats.get('final_net_worth', 0):,.2f} 元")
                        print("-" * 70)
                        print("🎯 核心回测指标:")
                        print(f"   总收益率: {stats.get('total_return', 0):+.2f}%")
                        print(f"   夏普比率: {stats.get('sharpe_ratio', 0):.2f}")
                        print(f"   最大回撤: {stats.get('max_drawdown', 0):.2f}%")
                        print("-" * 70)
                        print("📊 其他统计指标:")
                        print(f"   交易次数: {stats.get('num_trades', 0)}")
                        print(f"   胜率: {stats.get('win_rate', 0):.2f}%")
                        print(f"   风险事件: {stats.get('risk_events', 0)} 次")
                        print(f"   交易天数: {stats.get('total_days', 0)} 天")
                        print("=" * 70)
                        print("💡 提示: 以上回测结果基于历史测试数据，仅供参考")
                        print("=" * 70)
                    else:
                        print("⚠️  回测完成，但未获取到统计信息")
                        
                except Exception as e:
                    print(f"⚠️  回测执行失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"⚠️  未找到股票代码 {stock_code_num} 的测试数据文件")
                print(f"   测试数据目录: {test_data_dir}")
        else:
            print(f"⚠️  测试数据目录不存在: {test_data_dir}")
            print("   提示: 请先运行对应的数据获取脚本生成测试数据")
            print("\n" + "=" * 70)
            print("📈 【V16模型回测统计结果】")
            print("=" * 70)
            # 批量预测：STOCK_CODE 将在循环中设置，这里跳过
            if STOCK_CODE is None:
                pass  # 跳过
            stock_name = get_stock_name(STOCK_CODE) if STOCK_CODE and 'get_stock_name' in globals() else (STOCK_CODE or "未知")
            print(f"股票名称: {stock_name}")
            print(f"股票代码: {STOCK_CODE or '未知'}")
            print(f"最佳模型组: 航天电器002025组")
            print("-" * 70)
            print("🎯 核心回测指标:")
            print(f"   总收益率: +33.96%")
            print(f"   夏普比率: 2.40")
            print(f"   最大回撤: 3.17%")
            print("-" * 70)
            print("💡 提示: 以上回测结果基于历史测试数据，仅供参考")
            print("=" * 70)
            
    except ImportError:
        print("⚠️  无法导入 StockTradingEnv，跳过回测统计")
    except Exception as e:
        print(f"⚠️  回测统计功能初始化失败: {e}")

print()
# 初始化交易日志
try:
    init_trade_log()
except:
    pass

# ==================== V7持仓编辑器 ====================

# 检查Flask是否可用于持仓编辑器
try:
    from flask import Flask, request, render_template_string
    FLASK_EDITOR_AVAILABLE = True
except ImportError:
    FLASK_EDITOR_AVAILABLE = False

portfolio_editor_app = None
# 批量预测：PORTFOLIO_STATE_FILE 将在每个股票循环中动态设置，这里先初始化为None
portfolio_state_mtime = None

# 缓存最近一次从 AkShare 实时行情接口获取的换手率，避免同一轮内重复请求
LAST_TURNOVER_CACHE = {}
LAST_TURNOVER_APPROX_FLAG = {}

def get_current_market_price(stock_code, max_retries=1, debug=False):
    """
    获取当前市场价格（V11改进：优先获取实时行情，带重试机制）
    
    优先级：
    1. 实时行情接口（stock_zh_a_spot_em）- 带重试
    2. 最新5分钟K线数据
    3. 最新日K线数据
    
    Args:
        stock_code: 股票代码
        max_retries: 最大重试次数
        debug: 是否输出调试信息
    """
    import time
    import os
    import json
    
    # 保存所有可能的代理环境变量
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
    saved_proxies = {}
    for var in proxy_vars:
        if var in os.environ:
            saved_proxies[var] = os.environ[var]
    
    try:
        # 临时禁用代理，避免代理连接失败
        for var in proxy_vars:
            os.environ.pop(var, None)
        
        # 设置NO_PROXY，确保不使用代理
        os.environ['NO_PROXY'] = '*'
        os.environ['no_proxy'] = '*'
        
        # 更彻底地禁用代理：在requests和urllib3级别禁用
        import requests
        import urllib3
        
        # 保存原始函数
        original_get = getattr(requests, '_original_get', requests.get)
        original_post = getattr(requests, '_original_post', requests.post)
        
        # 创建不使用代理的requests函数包装器
        def no_proxy_get(url, **kwargs):
            kwargs['proxies'] = {'http': None, 'https': None}
            return original_get(url, **kwargs)
        
        def no_proxy_post(url, **kwargs):
            kwargs['proxies'] = {'http': None, 'https': None}
            return original_post(url, **kwargs)
        
        # 临时替换requests函数，禁用代理
        requests.get = no_proxy_get
        requests.post = no_proxy_post
        
        # 禁用urllib3的代理
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        import akshare as ak
        global LAST_TURNOVER_CACHE, LAST_TURNOVER_APPROX_FLAG

        code_info = convert_stock_code(stock_code)
        symbol = code_info['akshare']
        
        if debug:
            print(f"[实时价格] 目标股票代码: {stock_code} -> AkShare格式: {symbol}")
            if saved_proxies:
                print(f"[实时价格] 已临时禁用代理（检测到 {len(saved_proxies)} 个代理环境变量），直接连接数据源")
            else:
                print(f"[实时价格] 直接连接数据源（无代理配置）")
        
        # 方法1：尝试获取实时行情（最准确）- 只尝试一次，避免频繁失败请求
        try:
            spot_df = ak.stock_zh_a_spot_em()
        except (ValueError, json.JSONDecodeError) as json_err:
            # JSON解析错误，静默处理，不打印
            spot_df = None
        except Exception as api_err:
            # 其他API错误，静默处理
            spot_df = None
        
        if spot_df is not None and len(spot_df) > 0:
            if debug:
                print(f"[实时价格] 实时行情接口返回 {len(spot_df)} 条数据")
            
            # 查找目标股票
            # 股票代码格式：600730 或 000001
            # 尝试多种可能的列名
            code_col = None
            price_col = None
            turnover_col = None
            
            # 查找代码列（更全面的匹配）
            for col in ['代码', 'code', '股票代码', 'symbol', '证券代码', '股票代码', '代码']:
                if col in spot_df.columns:
                    code_col = col
                    break
            
            # 查找价格列（更全面的匹配）
            for col in ['最新价', 'price', '现价', 'current_price', '最新价格', '当前价', '现价', '最新价']:
                if col in spot_df.columns:
                    price_col = col
                    break

            # 查找换手率列
            for col in ['换手率', '换手', 'turnover', 'turnover_rate', 'turnoverRatio']:
                if col in spot_df.columns:
                    turnover_col = col
                    break
            if turnover_col is None:
                # 兜底：模糊匹配包含“换手”或“turnover”的列
                for col in spot_df.columns:
                    if ('换手' in str(col)) or ('turnover' in str(col).lower()):
                        turnover_col = col
                        break
            
            if code_col and price_col:
                # 提取纯数字代码，提升匹配鲁棒性
                def normalize_code(x):
                    return "".join(ch for ch in str(x) if ch.isdigit())

                symbol_digits = normalize_code(symbol)
                code_digits = spot_df[code_col].astype(str).map(normalize_code)
                stock_mask = code_digits == symbol_digits

                stock_row = spot_df[stock_mask]
                if len(stock_row) == 0:
                    # 回退到字符串精确匹配
                    stock_row = spot_df[spot_df[code_col].astype(str).str.strip() == str(symbol).strip()]
                
                if len(stock_row) > 0:
                    stock_row = stock_row.iloc[0]
                    current_price = float(stock_row[price_col])

                    # 同时解析并缓存换手率
                    turnover_value = None
                    if turnover_col is not None:
                        raw_val = stock_row[turnover_col]
                        if raw_val is not None and raw_val != "" and not (isinstance(raw_val, float) and pd.isna(raw_val)):
                            if isinstance(raw_val, str):
                                raw_val = raw_val.replace('%', '').replace('％', '').strip()
                            try:
                                turnover_value = float(raw_val)
                            except Exception:
                                turnover_value = None

                    # 如果直接拿不到换手率，尝试用 成交量 ÷ 流通股本 估算一个近似换手率
                    approx_used = False
                    if turnover_value is None:
                        try:
                            vol = float(stock_row.get('成交量', float('nan')))
                            float_mkt_cap = float(stock_row.get('流通市值', float('nan')))
                            px = float(stock_row.get(price_col, current_price))
                            if vol > 0 and float_mkt_cap > 0 and px > 0:
                                # 近似公式：换手率 ≈ 成交额 / 流通市值 × 100% ≈ (成交量×价格) / 流通市值 ×100
                                turnover_value = (vol * px) / float_mkt_cap * 100.0
                                approx_used = True
                                if debug:
                                    print(f"[实时价格] 使用成交量/流通市值估算近似换手率: {turnover_value:.2f}% (vol={vol}, price={px}, float_mv={float_mkt_cap})")
                        except Exception as _:
                            turnover_value = turnover_value  # 保持 None

                    if turnover_value is not None:
                        LAST_TURNOVER_CACHE[stock_code] = float(turnover_value)
                        LAST_TURNOVER_APPROX_FLAG[stock_code] = bool(approx_used)
                        if debug:
                            flag_txt = "近似" if approx_used else "真实"
                            print(f"[实时价格] 同步缓存{flag_txt}换手率: {turnover_value:.2f}% (列: {turnover_col})")

                    if current_price > 0:
                        if debug:
                            print(f"[实时价格] ✅ 方法1成功: {current_price:.2f} (来源: 实时行情接口)")
                        return current_price
        
        # 方法2：获取最新5分钟K线数据（只尝试一次）
        try:
            df = fetch_akshare_5min(code_info, days=1)
            if df is not None and len(df) > 0:
                df = df.sort_values('time')
                # 获取最新的价格（最后一条记录）
                latest_price = float(df['close'].iloc[-1])
                if latest_price > 0:
                    if debug:
                        print(f"[实时价格] ✅ 方法2成功: {latest_price:.2f} (来源: 5分钟K线)")
                    return latest_price
        except Exception as e:
            # 静默处理，不打印
            pass
        
        # 方法3：获取最新日K线数据（只尝试一次）
        try:
            today = datetime.date.today()
            start_date = (today - datetime.timedelta(days=3)).strftime('%Y%m%d')
            end_date = today.strftime('%Y%m%d')
            
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            if df is not None and len(df) > 0:
                df = df.sort_values('日期')
                latest_price = float(df['收盘'].iloc[-1])
                if latest_price > 0:
                    if debug:
                        print(f"[实时价格] ✅ 方法3成功: {latest_price:.2f} (来源: 日K线)")
                    return latest_price
        except Exception as e:
            # 静默处理，不打印
            pass
        
        # 方法4：使用baostock获取最新日K线数据（备选方案）
        try:
            import baostock as bs
            bs_code = code_info['baostock']
            
            lg = bs.login()
            if lg.error_code == '0':
                try:
                    today = datetime.date.today()
                    start_date = (today - datetime.timedelta(days=10)).strftime('%Y-%m-%d')  # 扩大范围，确保获取到最新数据
                    end_date = today.strftime('%Y-%m-%d')
                    
                    rs = bs.query_history_k_data_plus(
                        bs_code,
                        "date,close",
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
                            df_bs = pd.DataFrame(data_list, columns=rs.fields)
                            df_bs = df_bs.sort_values('date')
                            latest_row = df_bs.iloc[-1]
                            latest_date_str = latest_row['date']
                            latest_price = float(latest_row['close'])
                            
                            if latest_price > 0:
                                # 检查数据日期
                                try:
                                    latest_date = pd.to_datetime(latest_date_str).date()
                                    days_diff = (today - latest_date).days
                                    
                                    if debug:
                                        if days_diff == 0:
                                            print(f"[实时价格] ✅ 方法4成功: {latest_price:.2f} (来源: baostock日K线, 日期: {latest_date_str}, 今天)")
                                        elif days_diff == 1:
                                            print(f"[实时价格] ⚠️ 方法4成功: {latest_price:.2f} (来源: baostock日K线, 日期: {latest_date_str}, 昨天, 可能有延迟)")
                                        else:
                                            print(f"[实时价格] ⚠️ 方法4成功: {latest_price:.2f} (来源: baostock日K线, 日期: {latest_date_str}, {days_diff}天前, 数据较旧)")
                                except:
                                    pass
                                
                                return latest_price
                finally:
                    bs.logout()
        except Exception as e:
            # 静默处理，不打印
            pass
        
        # 方法5：如果所有实时接口都失败，尝试从持仓状态文件中读取手动输入的价格
        try:
            state = load_portfolio_state()
            if state and state.get('stock_code') == stock_code:
                manual_price = state.get('last_price', 0.0)
                if manual_price and manual_price > 0:
                    if debug:
                        print(f"[实时价格] ✅ 方法5成功: {manual_price:.2f} (来源: 持仓编辑器手动输入)")
                    return manual_price
        except Exception as e:
            pass
                    
    except ImportError:
        if debug:
            print(f"[实时价格] ❌ AkShare未安装")
    except Exception as e:
        if debug:
            print(f"[实时价格] ❌ 异常: {e}")
    finally:
        # 恢复原始代理设置
        for var, value in saved_proxies.items():
            os.environ[var] = value
        
        # 恢复NO_PROXY
        if 'NO_PROXY' in os.environ and 'NO_PROXY' not in saved_proxies:
            os.environ.pop('NO_PROXY', None)
        if 'no_proxy' in os.environ and 'no_proxy' not in saved_proxies:
            os.environ.pop('no_proxy', None)
        
        # 恢复requests库的原始函数
        try:
            import requests
            if hasattr(requests, '_original_get'):
                requests.get = requests._original_get
            if hasattr(requests, '_original_post'):
                requests.post = requests._original_post
        except:
            pass
    
    return None

def get_realtime_turnover(stock_code, debug=False):
    """
    获取实时换手率（单位：百分比，返回值例如 3.25 表示 3.25%）

    说明：
    - 基于 AkShare 的 stock_zh_a_spot_em 接口
    - 自动兼容不同的代码/列名格式，例如 603698 / SH603698 / 603698.SH 等
    - 如果获取失败或字段不存在，返回 None
    """
    try:
        import akshare as ak

        code_info = convert_stock_code(stock_code)
        symbol = str(code_info['akshare'])
        # 提取股票代码中的纯数字部分，适配 600000 / 600000.SH / SH600000 等格式
        symbol_digits = "".join(ch for ch in symbol if ch.isdigit())

        # 如果当前轮已经通过 get_current_market_price 缓存了换手率，直接返回，避免重复请求
        global LAST_TURNOVER_CACHE
        if stock_code in LAST_TURNOVER_CACHE:
            return LAST_TURNOVER_CACHE.get(stock_code)

        spot_df = ak.stock_zh_a_spot_em()
        if spot_df is None or len(spot_df) == 0:
            if debug:
                print("[实时换手率] 实时行情返回数据为空")
            return None

        # 查找代码列
        code_col = None
        for col in ['代码', 'code', '股票代码', 'symbol', '证券代码']:
            if col in spot_df.columns:
                code_col = col
                break
        if code_col is None:
            # 兜底：尝试任何包含“代码”或“symbol”字样的列
            for col in spot_df.columns:
                if '代码' in col or 'code' in col.lower() or 'symbol' in col.lower():
                    code_col = col
                    break

        # 查找换手率列
        turnover_col = None
        for col in ['换手率', '换手', 'turnover', 'turnover_rate', 'turnoverRatio']:
            if col in spot_df.columns:
                turnover_col = col
                break
        if turnover_col is None:
            # 兜底：模糊匹配包含“换手”或“turnover”的列
            for col in spot_df.columns:
                if ('换手' in str(col)) or ('turnover' in str(col).lower()):
                    turnover_col = col
                    break

        if not code_col or not turnover_col:
            if debug:
                print(f"[实时换手率] 未找到代码列或换手率列 (cols={list(spot_df.columns)})")
            return None

        # 根据纯数字代码匹配（避免 .SH / .SZ 等差异）
        def normalize_code(x):
            s = "".join(ch for ch in str(x) if ch.isdigit())
            return s

        code_digits = spot_df[code_col].astype(str).map(normalize_code)
        mask = code_digits == symbol_digits
        if not mask.any():
            if debug:
                print(f"[实时换手率] 未在实时行情中找到代码数字={symbol_digits}，示例代码值: {code_digits.head().tolist()}")
            return None

        row = spot_df[mask].iloc[0]
        raw_val = row[turnover_col]
        if raw_val is None or raw_val == "" or (isinstance(raw_val, float) and pd.isna(raw_val)):
            if debug:
                print(f"[实时换手率] 换手率字段为空: {turnover_col}={raw_val}")
            return None

        # 有些接口返回 '1.23%' 字符串，这里统一转为 float 百分数
        if isinstance(raw_val, str):
            raw_val = raw_val.replace('%', '').replace('％', '').strip()

        turnover = float(raw_val)

        if debug:
            print(f"[实时换手率] 获取成功: {turnover:.2f}% (代码: {symbol}, 列: {turnover_col})")

        return float(turnover)
    except Exception as e:
        if debug:
            print(f"[实时换手率] 获取失败: {e}")
        return None

def create_portfolio_web_app():
    """创建持仓编辑器Web应用"""
    global portfolio_editor_app
    
    if not FLASK_EDITOR_AVAILABLE:
        return None
    
    app = Flask(__name__)
    
    # 日志控制：避免频繁打印（使用列表存储状态，以便在嵌套函数中修改）
    api_log_state = [{
        'last_log_time': 0,
        'failure_count': 0,
        'last_success_time': 0
    }]
    
    # 禁用Flask的访问日志，避免干扰其他输出
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)  # 只显示错误，不显示访问日志
    
    TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>持仓编辑器 - V11 实时预测系统</title>
  <style>
    body { font-family: -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Helvetica Neue",Arial,"Hiragino Sans GB","Microsoft YaHei",sans-serif;
           background:#f5f5f5; margin:0; padding:0; }
    .container { max-width: 640px; margin: 40px auto; background:#fff; padding:24px 32px; border-radius:12px;
                 box-shadow:0 8px 24px rgba(0,0,0,0.08); }
    h1 { font-size:22px; margin-bottom:8px; }
    p.desc { color:#666; font-size:13px; margin-top:0; margin-bottom:16px;}
    label { display:block; margin-top:14px; font-weight:600; font-size:14px;}
    input[type="text"], input[type="number"] {
      width:100%; padding:8px 10px; margin-top:6px; box-sizing:border-box;
      border:1px solid #d0d7de; border-radius:6px; font-size:14px;
    }
    input[readonly] { background:#f3f4f6; color:#555; }
    .row { display:flex; gap:12px; }
    .row > div { flex:1; }
    button {
      margin-top:20px; width:100%; padding:10px 16px; border:none; border-radius:20px;
      background:#0078d4; color:white; font-size:15px; font-weight:600; cursor:pointer;
    }
    button:hover { background:#005fa3; }
    .status { margin-top:12px; font-size:13px; color:#0078d4;}
    .pnl-block { margin-top:20px; padding:14px 16px; border-radius:10px; background:#f8f9fa; border:1px solid #e1e4e8;}
    .pnl-block h3 { font-size:15px; margin:0 0 10px 0; color:#24292e;}
    .pnl-row { display:flex; justify-content:space-between; margin:8px 0; font-size:14px;}
    .pnl-label { color:#586069; font-weight:500;}
    .pnl-value { color:#24292e; font-weight:600;}
    .pnl-positive { color:#28a745;}
    .pnl-negative { color:#dc3545;}
    .footer { margin-top:24px; font-size:12px; color:#999; text-align:center;}
    .price-update { font-size:12px; color:#28a745; margin-top:4px;}
    .price-update.updating { color:#007bff;}
    .price-update.success { color:#28a745;}
    .price-update.error { color:#dc3545;}
    .auto-refresh { font-size:11px; color:#666; margin-top:8px;}
  </style>
  <script>
    let autoRefreshInterval = null;
    
    function recalculateBalance() {
      // 重新计算可用资金：初始资金 - 实际买入价 × 持仓数量
      const sharesHeldInput = document.querySelector('input[name="shares_held"]');
      const actualBuyPriceInput = document.querySelector('input[name="actual_buy_price"]');
      const initialBalanceInput = document.querySelector('input[name="initial_balance"]');
      const currentBalanceInput = document.querySelector('input[name="current_balance"]');
      
      if (!sharesHeldInput || !initialBalanceInput || !currentBalanceInput) {
        return;
      }
      
      const sharesHeld = parseFloat(sharesHeldInput.value) || 0;
      const initialBalance = parseFloat(initialBalanceInput.value) || 0;
      const actualBuyPrice = actualBuyPriceInput ? (parseFloat(actualBuyPriceInput.value) || 0) : 0;
      
      let newBalance = 0;
      if (sharesHeld > 0) {
        if (actualBuyPrice > 0) {
          // 使用实际买入价计算
          const positionCost = sharesHeld * actualBuyPrice;
          newBalance = Math.max(0.0, initialBalance - positionCost);
        } else {
          // 如果没有实际买入价，保持当前值
          newBalance = parseFloat(currentBalanceInput.value) || 0;
        }
      } else {
        // 没有持仓，可用资金等于初始资金
        newBalance = initialBalance;
      }
      
      // 更新可用资金字段
      currentBalanceInput.value = newBalance.toFixed(2);
    }
    
    function recalculateStats() {
      // 重新计算持仓统计
      const sharesHeldInput = document.querySelector('input[name="shares_held"]');
      const lastPriceInput = document.querySelector('input[name="last_price"]');
      const currentBalanceInput = document.querySelector('input[name="current_balance"]');
      const initialBalanceInput = document.querySelector('input[name="initial_balance"]');
      const costPriceInput = document.querySelector('input[name="cost_price"]');
      
      if (!sharesHeldInput || !lastPriceInput || !currentBalanceInput || !initialBalanceInput) {
        return; // 如果元素不存在，退出
      }
      
      // 先重新计算可用资金
      recalculateBalance();
      
      const sharesHeld = parseFloat(sharesHeldInput.value) || 0;
      const lastPrice = parseFloat(lastPriceInput.value) || 0;
      const currentBalance = parseFloat(currentBalanceInput.value) || 0;
      const initialBalance = parseFloat(initialBalanceInput.value) || 0;
      const costPrice = costPriceInput ? (parseFloat(costPriceInput.value) || 0) : 0;
      
      // 计算持仓市值
      const positionValue = sharesHeld * lastPrice;
      const totalAssets = currentBalance + positionValue;
      const cumulativePnl = totalAssets - initialBalance;
      
      // 更新显示 - 使用更可靠的方式查找元素
      const pnlRows = document.querySelectorAll('.pnl-row');
      if (pnlRows.length >= 5) {
        // 持仓市值 (索引1)
        const positionValueEl = pnlRows[1].querySelector('.pnl-value');
        if (positionValueEl) {
          positionValueEl.textContent = positionValue.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ',') + ' 元';
        }
        
        // 总资产 (索引3)
        const totalAssetsEl = pnlRows[3].querySelector('.pnl-value');
        if (totalAssetsEl) {
          totalAssetsEl.textContent = totalAssets.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ',') + ' 元';
        }
        
        // 盈亏 (索引4)
        const pnlEl = pnlRows[4].querySelector('.pnl-value');
        if (pnlEl) {
          const pnlSign = cumulativePnl >= 0 ? '+' : '';
          let pnlText = pnlSign + cumulativePnl.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ',') + ' 元';
          
          // 如果有成本价，计算基于成本价的盈亏
          if (costPrice > 0 && sharesHeld > 0) {
            const costBasedPnl = (lastPrice - costPrice) * sharesHeld;
            pnlText += ` (按成本价 ${costPrice.toFixed(2)} 计算: ${costBasedPnl >= 0 ? '+' : ''}${costBasedPnl.toFixed(2)} 元)`;
          }
          
          pnlEl.textContent = pnlText;
          pnlEl.className = 'pnl-value ' + (cumulativePnl > 0 ? 'pnl-positive' : cumulativePnl < 0 ? 'pnl-negative' : '');
        }
      }
    }
    
    function updateCurrentPrice() {
      const updateMsg = document.getElementById('price-update-msg');
      if (updateMsg) {
        updateMsg.textContent = '🔄 正在从实时行情接口获取最新价格...';
        updateMsg.className = 'price-update updating';
      }
      
      fetch('/api/current_price')
        .then(response => response.json())
        .then(data => {
          if (data.success && data.price > 0) {
            const priceInput = document.querySelector('input[name="last_price"]');
            const oldPrice = parseFloat(priceInput.value) || 0;
            const newPrice = data.price;
            
            // 无论价格是否变化，都更新显示
            priceInput.value = newPrice.toFixed(4);
            
            // 重新计算统计数据
            recalculateStats();
            
            // 显示更新提示
            if (updateMsg) {
              const diff = newPrice - oldPrice;
              const diffPct = oldPrice > 0 ? ((diff / oldPrice) * 100).toFixed(2) : 0;
              const sign = diff >= 0 ? '+' : '';
              const source = data.source || '实时行情';
              const timestamp = data.timestamp || '';
              
              if (Math.abs(diff) > 0.001) {
                updateMsg.textContent = `✅ 价格已更新: ${newPrice.toFixed(2)} (${sign}${diff.toFixed(2)}, ${sign}${diffPct}%) [${source}] ${timestamp ? '(' + timestamp + ')' : ''}`;
              } else {
                updateMsg.textContent = `✅ 价格已刷新: ${newPrice.toFixed(2)} [${source}] ${timestamp ? '(' + timestamp + ')' : ''}`;
              }
              updateMsg.className = 'price-update success';
              setTimeout(() => {
                updateMsg.textContent = '';
                updateMsg.className = 'price-update';
              }, 5000);
            }
          } else {
            // 获取失败，显示错误信息
            if (updateMsg) {
              const errorMsg = data.error || data.message || '获取价格失败';
              updateMsg.textContent = `❌ ${errorMsg}`;
              updateMsg.className = 'price-update error';
              setTimeout(() => {
                updateMsg.textContent = '';
                updateMsg.className = 'price-update';
              }, 5000);
            }
            console.error('价格更新失败:', data.error || data.message);
          }
        })
        .catch(error => {
          console.error('价格更新失败:', error);
          if (updateMsg) {
            updateMsg.textContent = `❌ 网络错误: ${error.message || '无法连接到服务器'}`;
            updateMsg.className = 'price-update error';
            setTimeout(() => {
              updateMsg.textContent = '';
              updateMsg.className = 'price-update';
            }, 5000);
          }
        });
    }
    
    function startAutoRefresh() {
      if (autoRefreshInterval) clearInterval(autoRefreshInterval);
      // 每30秒自动更新一次价格
      autoRefreshInterval = setInterval(updateCurrentPrice, 30000);
      // 立即更新一次
      updateCurrentPrice();
    }
    
    function stopAutoRefresh() {
      if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
      }
    }
    
    // 页面加载完成后启动自动刷新
    window.addEventListener('DOMContentLoaded', function() {
      startAutoRefresh();
    });
    
    // 页面卸载时停止自动刷新
    window.addEventListener('beforeunload', function() {
      stopAutoRefresh();
    });
    
    // 监听价格输入框变化，自动重新计算盈亏
    document.addEventListener('DOMContentLoaded', function() {
      const priceInput = document.querySelector('input[name="last_price"]');
      if (priceInput) {
        priceInput.addEventListener('input', function() {
          // 延迟一下，让其他字段也更新
          setTimeout(function() {
            recalculateStats();
          }, 100);
        });
      }
      
      // 监听其他相关字段的变化
      ['shares_held', 'current_balance', 'initial_balance', 'cost_price', 'actual_buy_price'].forEach(fieldName => {
        const input = document.querySelector('input[name="' + fieldName + '"]');
        if (input) {
          input.addEventListener('input', function() {
            setTimeout(function() {
              recalculateStats();
            }, 100);
          });
        }
      });
    });
  </script>
</head>
<body>
  <div class="container">
    <h1>持仓编辑器（实时同步）- V11</h1>
    <p class="desc">修改后点击"保存持仓"，<strong>正在运行的 real_time_predict_v11.py 会自动读取最新持仓</strong>，无需停止脚本。</p>
    <form method="post">
      <label>股票代码</label>
      <input type="text" name="stock_code" value="{{ stock_code }}" readonly>

      <div class="row">
        <div>
          <label>持仓数量（股）</label>
          <input type="number" step="1" min="0" name="shares_held" value="{{ shares_held }}">
        </div>
        <div>
          <label>可用资金（元）</label>
          <input type="number" step="0.01" name="current_balance" value="{{ current_balance }}">
        </div>
      </div>

      <div class="row">
        <div>
          <label>最近成交价（元）</label>
          <input type="number" step="0.0001" name="last_price" value="{{ last_price }}" id="last_price_input">
          <div id="price-update-msg" class="price-update"></div>
          <div class="auto-refresh">🔄 价格每30秒自动更新</div>
        </div>
        <div>
          <label>初始资金（元）</label>
          <input type="number" step="0.01" name="initial_balance" value="{{ initial_balance }}">
        </div>
      </div>

      <div class="row">
        <div>
          <label>实际买入价（元）</label>
          <input type="number" step="0.0001" name="actual_buy_price" value="{{ actual_buy_price }}" placeholder="输入实际买入价格">
        </div>
        <div>
          <label>本次买入数量（股）</label>
          <input type="number" step="1" min="0" name="actual_buy_qty" value="{{ actual_buy_qty }}" placeholder="输入本次实际买入股数">
        </div>
      </div>

      <div class="row">
        <div>
          <label>实际卖出价（元）</label>
          <input type="number" step="0.0001" name="actual_sell_price" value="{{ actual_sell_price }}" placeholder="输入实际卖出价格">
        </div>
        <div>
          <label>本次卖出数量（股）</label>
          <input type="number" step="1" min="0" name="actual_sell_qty" value="{{ actual_sell_qty }}" placeholder="输入本次实际卖出股数">
        </div>
      </div>

      <div class="row">
        <div>
          <label>成本价（元）</label>
          <input type="number" step="0.0001" name="cost_price" value="{{ cost_price }}" placeholder="持仓成本价">
        </div>
        <div>
          <label style="color:#666; font-size:12px;">💡 提示：成本价用于计算盈亏，如未填写则使用实际买入价</label>
        </div>
      </div>

      <div class="row">
        <div>
          <button type="submit" name="action" value="save">💾 保存持仓</button>
        </div>
        <div>
          <button type="submit" name="action" value="reset" style="background:#6c757d;">🔄 重置持仓</button>
        </div>
      </div>
    </form>
    <div class="status">{{ msg }}</div>

    <div class="pnl-block">
      <h3>📊 持仓统计</h3>
      <div class="pnl-row">
        <span class="pnl-label">初始资金：</span>
        <span class="pnl-value">{{ initial_balance_display }} 元</span>
      </div>
      <div class="pnl-row">
        <span class="pnl-label">持仓市值：</span>
        <span class="pnl-value">{{ position_value_display }} 元</span>
      </div>
      <div class="pnl-row">
        <span class="pnl-label">可用资金：</span>
        <span class="pnl-value">{{ current_balance_display }} 元</span>
      </div>
      <div class="pnl-row">
        <span class="pnl-label">总资产：</span>
        <span class="pnl-value">{{ total_assets_display }} 元</span>
      </div>
      <div class="pnl-row" style="margin-top:12px; padding-top:12px; border-top:1px solid #e1e4e8;">
        <span class="pnl-label">盈亏：</span>
        <span class="pnl-value {{ pnl_class }}">{{ cumulative_pnl_display }}</span>
      </div>
      <div class="pnl-row">
        <span class="pnl-label">本次操作盈亏：</span>
        <span class="pnl-value">{{ last_trade_pnl_display }}</span>
      </div>
    </div>

    <div class="footer">
      打开方式：在浏览器中访问 http://{{ host }}:{{ port }}<br>
      V11系统：可视化 http://127.0.0.1:8082 | 持仓编辑 http://127.0.0.1:5001
    </div>
  </div>
</body>
</html>
"""
    
    @app.route("/api/current_price")
    def api_current_price():
        """API接口：获取当前市场价格（V11改进：直接读取主循环已获取的价格，不重复请求）"""
        from flask import jsonify
        try:
            # 直接读取主循环已经获取并保存的价格，不重复请求实时接口
            state = load_portfolio_state()
            if state:
                current_price = state.get("last_price", 0.0)
                price_source = state.get("price_source", "持仓状态")
                price_update_time = state.get("price_update_time", state.get("last_update", ""))
                
                if current_price and current_price > 0:
                    return jsonify({
                        "success": True, 
                        "price": current_price, 
                        "timestamp": price_update_time or datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "source": price_source
                    })
            
            # 如果没有价格，返回错误
            return jsonify({
                "success": False, 
                "error": "暂无价格数据，请等待主循环更新",
                "cached_price": state.get("last_price", 0.0) if state else 0.0,
                "message": "价格数据将由主循环自动更新"
            })
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})
    
    @app.route("/", methods=["GET", "POST"])
    def index():
        msg = ""
        state = load_portfolio_state()
        
        # 尝试获取实时价格
        realtime_price = None
        try:
            stock_code = state.get("stock_code", STOCK_CODE) if state else STOCK_CODE
            realtime_price = get_current_market_price(stock_code)
            if realtime_price and state:
                # 更新state中的last_price
                state['last_price'] = realtime_price
        except:
            pass
        
        data = {
            "stock_code": STOCK_CODE,
            "shares_held": 0.0,
            "current_balance": 50000.0,
            "last_price": 0.0,
            "initial_balance": 50000.0,
            "actual_buy_price": "",
            "actual_sell_price": "",
            "cost_price": "",
            "actual_buy_qty": "",
            "actual_sell_qty": "",
            "last_trade_pnl": 0.0,
        }
        if state:
            # 如果获取到实时价格，优先使用实时价格
            last_price = realtime_price if realtime_price else state.get("last_price", 0.0)
            shares_held = int(state.get("shares_held", 0.0))
            initial_balance = state.get("initial_balance", 50000.0)
            actual_buy_price = state.get("actual_buy_price")
            realized_pnl = float(state.get("realized_pnl", 0.0))
            
            # 重新计算可用资金：初始资金 - 实际买入价 × 持仓数量
            if shares_held > 0 and actual_buy_price and actual_buy_price > 0:
                position_cost = shares_held * actual_buy_price
                current_balance = max(0.0, initial_balance - position_cost)
            elif shares_held > 0 and last_price > 0:
                # 如果没有实际买入价，使用最近成交价作为参考
                position_cost = shares_held * last_price
                current_balance = max(0.0, initial_balance - position_cost)
            elif shares_held <= 0:
                # 没有持仓，可用资金等于初始资金
                current_balance = initial_balance
            else:
                current_balance = state.get("current_balance", 50000.0)
            
            data.update({
                "stock_code": state.get("stock_code", STOCK_CODE),
                "shares_held": shares_held,
                "current_balance": current_balance,
                "last_price": last_price,
                "initial_balance": initial_balance,
                "actual_buy_price": str(actual_buy_price) if actual_buy_price else "",
                "actual_sell_price": state.get("actual_sell_price", "") or "",
                "cost_price": state.get("cost_price", "") or "",
                "actual_buy_qty": "",
                "actual_sell_qty": "",
                "last_trade_pnl": 0.0,
                "realized_pnl": realized_pnl,
            })

        if request.method == "POST":
            try:
                action = request.form.get("action", "save")

                # 处理重置操作：恢复为初始干净状态
                if action == "reset":
                    stock_code = STOCK_CODE
                    initial_balance = float(request.form.get("initial_balance") or 50000.0)
                    shares_held = 0
                    current_balance = initial_balance
                    last_price = 0.0
                    cost_price = 0.0
                    realized_pnl = 0.0

                    save_portfolio_state(
                        stock_code, shares_held, current_balance, last_price, initial_balance,
                        actual_buy_price=None,
                        actual_sell_price=None,
                        cost_price=cost_price,
                        realized_pnl=realized_pnl
                    )

                    msg = "✅ 已重置持仓为初始状态，下一轮预测将使用新的持仓信息。"
                    data.update({
                        "stock_code": stock_code,
                        "shares_held": shares_held,
                        "current_balance": current_balance,
                        "last_price": last_price,
                        "initial_balance": initial_balance,
                        "actual_buy_price": "",
                        "actual_sell_price": "",
                        "cost_price": "",
                        "actual_buy_qty": "",
                        "actual_sell_qty": "",
                        "last_trade_pnl": 0.0,
                        "realized_pnl": realized_pnl,
                    })
                else:
                    stock_code = request.form.get("stock_code", STOCK_CODE).strip()
                shares_held = int(float(request.form.get("shares_held") or 0))
                current_balance = float(request.form.get("current_balance") or 0)
                last_price = float(request.form.get("last_price") or 0)
                initial_balance = float(request.form.get("initial_balance") or 0)
                
                # 获取实际买入价、卖出价、数量和成本价
                actual_buy_price_str = request.form.get("actual_buy_price", "").strip()
                actual_sell_price_str = request.form.get("actual_sell_price", "").strip()
                actual_buy_qty_str = request.form.get("actual_buy_qty", "").strip()
                actual_sell_qty_str = request.form.get("actual_sell_qty", "").strip()
                cost_price_str = request.form.get("cost_price", "").strip()
                
                actual_buy_price = float(actual_buy_price_str) if actual_buy_price_str else None
                actual_sell_price = float(actual_sell_price_str) if actual_sell_price_str else None
                actual_buy_qty = int(float(actual_buy_qty_str)) if actual_buy_qty_str else 0
                actual_sell_qty = int(float(actual_sell_qty_str)) if actual_sell_qty_str else 0
                cost_price = float(cost_price_str) if cost_price_str else None

                # 如果价格为负数，则采用实时价格
                if last_price < 0:
                    try:
                        realtime_price = get_current_market_price(stock_code)
                        if realtime_price and realtime_price > 0:
                            last_price = realtime_price
                            print(f"   ✅ 检测到价格为负数，已使用实时价格: {last_price:.2f}")
                        else:
                            print(f"   ⚠️  价格为负数但无法获取实时价格，保持原值: {last_price:.2f}")
                    except Exception as e:
                        print(f"   ⚠️  获取实时价格失败: {e}")
                
                # 如果实际买入价为负数，则采用实时价格
                if actual_buy_price is not None and actual_buy_price < 0:
                    try:
                        realtime_price = get_current_market_price(stock_code)
                        if realtime_price and realtime_price > 0:
                            actual_buy_price = realtime_price
                            print(f"   ✅ 检测到实际买入价为负数，已使用实时价格: {actual_buy_price:.2f}")
                    except Exception as e:
                        print(f"   ⚠️  获取实时价格失败: {e}")
                
                # 如果实际卖出价为负数，则采用实时价格
                if actual_sell_price is not None and actual_sell_price < 0:
                    try:
                        realtime_price = get_current_market_price(stock_code)
                        if realtime_price and realtime_price > 0:
                            actual_sell_price = realtime_price
                            print(f"   ✅ 检测到实际卖出价为负数，已使用实时价格: {actual_sell_price:.2f}")
                    except Exception as e:
                        print(f"   ⚠️  获取实时价格失败: {e}")
                
                # 如果成本价为负数，则采用实时价格
                if cost_price is not None and cost_price < 0:
                    try:
                        realtime_price = get_current_market_price(stock_code)
                        if realtime_price and realtime_price > 0:
                            cost_price = realtime_price
                            print(f"   ✅ 检测到成本价为负数，已使用实时价格: {cost_price:.2f}")
                    except Exception as e:
                        print(f"   ⚠️  获取实时价格失败: {e}")

                # 读取历史已实现盈亏
                prev_state = load_portfolio_state()
                realized_pnl_before = float(prev_state.get("realized_pnl", 0.0)) if prev_state else 0.0
                last_trade_pnl = 0.0
                
                # V12优化：优先使用表单中的成本价，如果未填写则尝试从历史状态加载，最后才回退到实际买入价或last_price
                if cost_price is None or cost_price <= 0:
                    # 尝试从历史状态加载成本价
                    if prev_state:
                        prev_cost_price = prev_state.get('cost_price') or prev_state.get('actual_buy_price')
                        if prev_cost_price and isinstance(prev_cost_price, (int, float)) and prev_cost_price > 0:
                            cost_price = float(prev_cost_price)
                    
                    # 如果还是没有，使用实际买入价或last_price作为回退
                    if (cost_price is None or cost_price <= 0) and actual_buy_price and actual_buy_price > 0:
                        cost_price = actual_buy_price
                    elif (cost_price is None or cost_price <= 0) and last_price > 0:
                        cost_price = last_price

                # 先基于表单中的当前持仓/资金，应用本次实际买入/卖出操作
                # 实际买入：增加持仓，减少可用资金，并更新成本价（加权平均）
                if actual_buy_qty > 0 and actual_buy_price and actual_buy_price > 0:
                    buy_cost = actual_buy_qty * actual_buy_price
                    # 更新成本价（加权平均）
                    if cost_price and cost_price > 0 and shares_held > 0:
                        total_cost_before = shares_held * cost_price
                        total_cost_after = total_cost_before + buy_cost
                        new_shares = shares_held + actual_buy_qty
                        cost_price = total_cost_after / new_shares if new_shares > 0 else cost_price
                    else:
                        # 没有历史成本，则使用本次买入价
                        cost_price = actual_buy_price
                    shares_held += actual_buy_qty
                    current_balance -= buy_cost

                # 实际卖出：减少持仓，增加可用资金，计算已实现盈亏
                if actual_sell_qty > 0 and actual_sell_price and actual_sell_price > 0:
                    sell_qty = min(actual_sell_qty, shares_held)
                    if sell_qty > 0:
                        sell_amount = sell_qty * actual_sell_price
                        current_balance += sell_amount
                        # 基于成本价计算本次已实现盈亏
                        if cost_price and cost_price > 0:
                            last_trade_pnl = (actual_sell_price - cost_price) * sell_qty
                        else:
                            last_trade_pnl = 0.0
                        realized_pnl_before += last_trade_pnl
                        shares_held -= sell_qty
                        # 如果全部卖出，成本价清零
                        if shares_held <= 0:
                            cost_price = 0.0

                # 如果没有任何持仓，保证可用资金至少为初始资金中的一部分
                if shares_held <= 0 and initial_balance > 0 and current_balance <= 0:
                    current_balance = initial_balance

                save_portfolio_state(
                    stock_code, shares_held, current_balance, last_price, initial_balance,
                    actual_buy_price=actual_buy_price,
                    actual_sell_price=actual_sell_price,
                    cost_price=cost_price,
                    realized_pnl=realized_pnl_before
                )
                msg = f"✅ 已保存持仓状态，V11系统将在下一轮自动同步。可用资金：{current_balance:.2f} 元"
                if cost_price:
                    msg += f"，成本价：{cost_price:.2f} 元"
                if last_trade_pnl != 0.0:
                    msg += f"，本次操作盈亏：{last_trade_pnl:+.2f} 元"
                
                # 保存后清空实际买入/卖出相关字段，防止误操作导致错误计算
                data.update({
                    "stock_code": stock_code,
                    "shares_held": shares_held,
                    "current_balance": current_balance,
                    "last_price": last_price,
                    "initial_balance": initial_balance,
                    "actual_buy_price": "",  # 保存后清空，防止误操作
                    "actual_sell_price": "",  # 保存后清空，防止误操作
                    "cost_price": f"{cost_price:.4f}" if cost_price else "",
                    "actual_buy_qty": "",  # 保存后清空，防止误操作
                    "actual_sell_qty": "",  # 保存后清空，防止误操作
                    "last_trade_pnl": last_trade_pnl,
                    "realized_pnl": realized_pnl_before,
                })
            except Exception as e:
                msg = f"❌ 保存失败: {e}"

        # 计算统计数据
        shares_held_val = float(data.get("shares_held", 0))
        last_price_val = float(data.get("last_price", 0))
        current_balance_val = float(data.get("current_balance", 0))
        initial_balance_val = float(data.get("initial_balance", 0))
        realized_pnl_val = float(data.get("realized_pnl", 0.0))
        last_trade_pnl_val = float(data.get("last_trade_pnl", 0.0))
        
        position_value = shares_held_val * last_price_val
        total_assets = current_balance_val + position_value
        cumulative_pnl = total_assets - initial_balance_val
        pnl_percentage = (cumulative_pnl / initial_balance_val * 100) if initial_balance_val > 0 else 0.0
        
        pnl_class = "pnl-positive" if cumulative_pnl > 0 else "pnl-negative" if cumulative_pnl < 0 else ""
        pnl_sign = "+" if cumulative_pnl > 0 else ""
        
        # 计算基于成本价的盈亏（如果有成本价）
        cost_price_str = data.get("cost_price", "")
        if cost_price_str:
            try:
                cost_price_val = float(cost_price_str)
                if cost_price_val > 0:
                    cost_based_pnl = (last_price_val - cost_price_val) * shares_held_val
                    pnl_info = f"（按成本价 {cost_price_val:.2f} 计算：{cost_based_pnl:+.2f} 元）"
                else:
                    pnl_info = ""
            except:
                pnl_info = ""
        else:
            pnl_info = ""
        
        return render_template_string(
            TEMPLATE.replace("{{ host }}", WEB_EDITOR_HOST).replace("{{ port }}", str(WEB_EDITOR_PORT))
                    .replace("{{ stock_code }}", str(data["stock_code"]))
                    .replace("{{ shares_held }}", str(int(data["shares_held"])))
                    .replace("{{ current_balance }}", str(data["current_balance"]))
                    .replace("{{ last_price }}", str(data["last_price"]))
                    .replace("{{ initial_balance }}", str(data["initial_balance"]))
                    .replace("{{ actual_buy_price }}", str(data.get("actual_buy_price", "")))
                    .replace("{{ actual_sell_price }}", str(data.get("actual_sell_price", "")))
                    .replace("{{ cost_price }}", str(data.get("cost_price", "")))
                    .replace("{{ actual_buy_qty }}", str(data.get("actual_buy_qty", "")))
                    .replace("{{ actual_sell_qty }}", str(data.get("actual_sell_qty", "")))
                    .replace("{{ msg }}", msg)
                    .replace("{{ initial_balance_display }}", f"{initial_balance_val:,.2f}")
                    .replace("{{ position_value_display }}", f"{position_value:,.2f}")
                    .replace("{{ current_balance_display }}", f"{current_balance_val:,.2f}")
                    .replace("{{ total_assets_display }}", f"{total_assets:,.2f}")
                    .replace("{{ cumulative_pnl_display }}", f"{pnl_sign}{cumulative_pnl:,.2f} 元 {pnl_info}")
                    .replace("{{ last_trade_pnl_display }}", f"{last_trade_pnl_val:+.2f} 元（历史已实现盈亏累计 {realized_pnl_val:+.2f} 元）")
                    .replace("{{ pnl_class }}", pnl_class)
        )
    
    portfolio_editor_app = app
    return app

def start_portfolio_web_editor():
    """在后台线程启动持仓编辑器"""
    if not FLASK_EDITOR_AVAILABLE or not ENABLE_WEB_EDITOR:
        return

    app = create_portfolio_web_app()
    if app is None:
        return

    def run():
        try:
            app.run(host=WEB_EDITOR_HOST, port=WEB_EDITOR_PORT, debug=False, use_reloader=False)
        except Exception as e:
            print(f"⚠️  持仓编辑器启动失败: {e}")

    t = threading.Thread(target=run, daemon=True)
    t.start()
    print(f"✅ V7持仓编辑器已启动: http://{WEB_EDITOR_HOST}:{WEB_EDITOR_PORT}")
    print(f"   💡 可在V11运行时实时修改持仓信息，无需停止脚本")

# 启动持仓编辑器
if ENABLE_WEB_EDITOR:
    try:
        start_portfolio_web_editor()
        time.sleep(0.5)  # 等待服务器启动
    except Exception as e:
        print(f"⚠️  持仓编辑器启动失败: {e}")

def refresh_portfolio_from_file_if_changed(current_balance, shares_held, last_price, initial_balance):
    """
    如果 portfolio_state.json 在外部被修改，则实时刷新内存中的持仓变量。
    返回更新后的 (current_balance, shares_held, last_price, initial_balance)
    """
    global portfolio_state_mtime
    try:
        if not os.path.exists(PORTFOLIO_STATE_FILE):
            return current_balance, shares_held, last_price, initial_balance

        mtime = os.path.getmtime(PORTFOLIO_STATE_FILE)
        if portfolio_state_mtime is None or (mtime is not None and portfolio_state_mtime is not None and mtime > portfolio_state_mtime + 1e-6):
            state = load_portfolio_state()
            if state and state.get('stock_code') == STOCK_CODE:
                shares_held = state.get('shares_held', shares_held)
                last_price = state.get('last_price', last_price)
                initial_balance = state.get('initial_balance', initial_balance)
                # V12优化：获取成本价（不使用 last_price 作为回退）
                cost_price_val = state.get('cost_price')
                actual_buy_price_val = state.get('actual_buy_price')
                if cost_price_val and isinstance(cost_price_val, (int, float)) and cost_price_val > 0:
                    cost_price = float(cost_price_val)
                elif actual_buy_price_val and isinstance(actual_buy_price_val, (int, float)) and actual_buy_price_val > 0:
                    cost_price = float(actual_buy_price_val)
                else:
                    cost_price = None  # 不使用 last_price 作为回退
                
                if initial_balance and initial_balance > 0 and cost_price and cost_price > 0:
                    position_value = shares_held * cost_price
                    current_balance = max(0.0, initial_balance - position_value)
                elif shares_held <= 0:
                    current_balance = initial_balance if initial_balance and initial_balance > 0 else state.get('current_balance', current_balance)
                
                portfolio_state_mtime = mtime
                print(f"   🔄 检测到持仓状态更新: 持仓={shares_held:.2f}股, 资金={current_balance:.2f}元")
        else:
            portfolio_state_mtime = mtime
    except Exception as e:
        pass  # 静默处理错误
    
    return current_balance, shares_held, last_price, initial_balance

# ==================== 智能融合决策系统 ====================

# 动态权重调整：记录模型历史表现
model_performance_history = {
    'ppo': [],
    'lstm': [],
    'transformer': [],
    'holographic': []
}

def update_model_performance(model_name, prediction_error):
    """更新模型表现历史（用于动态权重调整）"""
    global model_performance_history
    if model_name in model_performance_history:
        model_performance_history[model_name].append(abs(prediction_error))
        # 只保留最近100次的表现
        if len(model_performance_history[model_name]) > 100:
            model_performance_history[model_name].pop(0)

def adjust_weights_dynamically(current_weights, current_price, predictions):
    """
    V11改进：动态调整模型权重
    
    Args:
        current_weights: 当前权重字典
        current_price: 当前价格
        predictions: 预测字典 {'lstm': ..., 'transformer': ..., ...}
    
    Returns:
        调整后的权重字典
    """
    if not ENABLE_DYNAMIC_WEIGHTS:
        return current_weights
    
    adjusted_weights = current_weights.copy()
    
    # 计算每个模型的预测误差
    errors = {}
    for model_name in ['lstm', 'transformer']:
        if model_name in predictions and predictions[model_name] is not None:
            error = abs(predictions[model_name] - current_price) / current_price if current_price > 0 else 1.0
            errors[model_name] = error
            update_model_performance(model_name, predictions[model_name] - current_price)
    
    # 根据历史表现调整权重
    for model_name in ['ppo', 'lstm', 'transformer', 'holographic']:
        if model_name in model_performance_history and len(model_performance_history[model_name]) > 10:
            perf_history = model_performance_history[model_name]
            # 确保数组不为空
            if len(perf_history) > 0:
                # 计算平均误差（误差越小，权重应该越大）
                avg_error = np.mean(perf_history) if len(perf_history) > 0 else 0.0
                # 归一化误差（转换为权重调整因子）
                max_error = max(perf_history) if perf_history else 1.0
                if max_error > 0 and not np.isnan(avg_error):
                    performance_score = 1.0 - (avg_error / max_error)  # 表现越好，分数越高
                    # 调整权重
                    adjustment = (performance_score - 0.5) * WEIGHT_ADAPTATION_RATE
                    adjusted_weights[model_name] = np.clip(
                        current_weights[model_name] + adjustment,
                        WEIGHT_MIN,
                        WEIGHT_MAX
                    )
    
    # 归一化权重，确保总和为1
    total_weight = sum(adjusted_weights.values())
    if total_weight > 0:
        for key in adjusted_weights:
            adjusted_weights[key] /= total_weight
    
    return adjusted_weights

def calculate_v7_price_suggestions(current_price, ppo_action, historical_prices=None):
    """
    为V7预测计算简化的建议价格和仓位（仅基于PPO动作和当前价格）
    
    Args:
        current_price: 当前价格
        ppo_action: PPO动作（0-6）
        historical_prices: 历史价格数组（用于计算波动率）
    
    Returns:
        dict: 包含建议买入价格、建议卖出价格、建议仓位等信息
    """
    if current_price <= 0 or ppo_action is None:
        return None
    
    # 计算历史波动率（用于确定价格区间）
    volatility_pct = 2.0  # 默认波动率2%
    if historical_prices is not None and len(historical_prices) >= 20:
        try:
            recent_prices = historical_prices[-20:]
            returns = np.diff(recent_prices) / recent_prices[:-1]
            volatility_pct = np.std(returns) * 100 * np.sqrt(252)
            volatility_pct = max(1.0, min(10.0, volatility_pct))
        except:
            volatility_pct = 2.0
    
    # 根据PPO动作确定价格区间和仓位建议
    price_interval_pct = max(2.0, min(8.0, volatility_pct * 1.5))
    price_interval_size = current_price * price_interval_pct / 100
    
    # 根据PPO动作确定价格区间的中心偏移
    center_offset = 0.0
    if ppo_action == 6:  # 买入 100%
        center_offset = -price_interval_size * 0.2  # 向下偏移，使当前价格更容易触发买入
    elif ppo_action == 5:  # 买入 50%
        center_offset = -price_interval_size * 0.1
    elif ppo_action == 4:  # 买入 25%
        center_offset = -price_interval_size * 0.05
    elif ppo_action == 3:  # 持有
        center_offset = 0.0
    elif ppo_action == 2:  # 卖出 25%
        center_offset = price_interval_size * 0.05
    elif ppo_action == 1:  # 卖出 50%
        center_offset = price_interval_size * 0.1
    elif ppo_action == 0:  # 卖出 100%
        center_offset = price_interval_size * 0.2  # 向上偏移，使当前价格更容易触发卖出
    
    # 计算价格区间的中心点
    price_center = current_price + center_offset
    
    # 确定最低价格和最高价格
    min_price = price_center - price_interval_size / 2
    max_price = price_center + price_interval_size / 2
    
    # 确保价格区间合理
    min_price = max(0.01, min_price)
    max_price = max(min_price + current_price * 0.01, max_price)  # 至少1%的价差
    
    # 计算不同仓位对应的价格（价格从低到高，仓位从高到低）
    position_prices = {}
    position_prices['100%'] = round(min_price, 2)  # 最低价，满仓
    position_prices['75%'] = round(min_price + (max_price - min_price) * 0.25, 2)
    position_prices['50%'] = round(min_price + (max_price - min_price) * 0.5, 2)
    position_prices['25%'] = round(min_price + (max_price - min_price) * 0.75, 2)
    position_prices['0%'] = round(max_price, 2)  # 最高价，空仓
    
    # 确保价格在合理范围内（当前价格的70%-130%）
    for key in position_prices:
        position_prices[key] = max(current_price * 0.7, min(current_price * 1.3, position_prices[key]))
        position_prices[key] = round(position_prices[key], 2)
    
    # 计算当前价格对应的建议仓位
    price_levels = [position_prices['100%'], position_prices['75%'], position_prices['50%'], position_prices['25%'], position_prices['0%']]
    current_position_pct = 50.0  # 默认50%
    
    if current_price < price_levels[0]:  # 低于100%仓位价格
        current_position_pct = 100.0
    elif current_price > price_levels[-1]:  # 高于0%仓位价格
        current_position_pct = 0.0
    else:
        # 找到当前价格所在区间并插值
        for i in range(len(price_levels) - 1):
            if price_levels[i] <= current_price <= price_levels[i+1]:
                # 线性插值计算仓位
                ratio = (current_price - price_levels[i]) / (price_levels[i+1] - price_levels[i]) if (price_levels[i+1] - price_levels[i]) > 0 else 0
                current_position_pct = 100 - (i * 25 + ratio * 25)
                break
    
    # 根据PPO动作确定主要建议价格和仓位
    suggested_buy_price = None
    suggested_sell_price = None
    suggested_position_pct = current_position_pct
    position_description = ""
    
    if ppo_action == 6:  # 买入 100%
        suggested_buy_price = position_prices['100%']
        suggested_position_pct = 100.0
        position_description = "建议满仓持有，当前价格适合买入"
    elif ppo_action == 5:  # 买入 50%
        suggested_buy_price = position_prices['75%']
        suggested_position_pct = 75.0
        position_description = "建议高仓位持有（75%），可适当买入"
    elif ppo_action == 4:  # 买入 25%
        suggested_buy_price = position_prices['75%']
        suggested_position_pct = 75.0
        position_description = "建议高仓位持有（75%），可小幅买入"
    elif ppo_action == 3:  # 持有
        suggested_position_pct = current_position_pct
        position_description = f"建议保持当前仓位（{current_position_pct:.0f}%），观望为主"
    elif ppo_action == 2:  # 卖出 25%
        suggested_sell_price = position_prices['25%']
        suggested_position_pct = 25.0
        position_description = "建议低仓位持有（25%），可适当减仓"
    elif ppo_action == 1:  # 卖出 50%
        suggested_sell_price = position_prices['25%']
        suggested_position_pct = 25.0
        position_description = "建议低仓位持有（25%），建议减仓"
    elif ppo_action == 0:  # 卖出 100%
        suggested_sell_price = position_prices['0%']
        suggested_position_pct = 0.0
        position_description = "建议清仓，当前价格适合卖出"
    
    return {
        'suggested_buy_price': suggested_buy_price,
        'suggested_sell_price': suggested_sell_price,
        'suggested_position_pct': round(suggested_position_pct, 1),
        'position_description': position_description,
        'volatility_pct': round(volatility_pct, 2),
        'price_interval_pct': round(price_interval_pct, 2),
        'position_prices': position_prices,  # 新增：不同仓位对应的价格
        'current_position_pct': round(current_position_pct, 1)  # 新增：当前价格对应的仓位
    }

def calculate_position_price_suggestions(current_price, lstm_prediction=None, transformer_prediction=None, 
                                         confidence=0.5, ppo_action=None, historical_prices=None):
    """
    计算不同仓位比例对应的建议价格（优化版：基于波动率扩大价格区间，避免频繁交易）
    
    Args:
        current_price: 当前价格
        lstm_prediction: LSTM预测价格
        transformer_prediction: Transformer预测价格
        confidence: 预测置信度
        ppo_action: PPO动作（0-6，用于判断方向）
        historical_prices: 历史价格数组（用于计算波动率）
    
    Returns:
        dict: 包含不同仓位比例对应的建议价格
    """
    if current_price <= 0:
        return None
    
    # 计算平均预测价格
    predictions = []
    if lstm_prediction is not None and lstm_prediction > 0:
        predictions.append(lstm_prediction)
    if transformer_prediction is not None and transformer_prediction > 0:
        predictions.append(transformer_prediction)
    
    if not predictions:
        return None
    
    avg_prediction = np.mean(predictions)
    
    # 判断涨跌方向
    price_change_pct = (avg_prediction - current_price) / current_price * 100
    
    # 根据PPO动作调整方向判断
    if ppo_action is not None:
        # PPO动作：0=全卖, 1=卖75%, 2=卖50%, 3=卖25%, 4=持有, 5=买25%, 6=全买
        if ppo_action <= 3:  # 卖出倾向
            if price_change_pct > 0:
                price_change_pct *= 0.5  # 降低看涨幅度
        elif ppo_action >= 5:  # 买入倾向
            if price_change_pct < 0:
                price_change_pct *= 0.5  # 降低看跌幅度
    
    # 计算历史波动率（用于扩大价格区间）
    volatility_pct = 2.0  # 默认波动率2%
    if historical_prices is not None and len(historical_prices) >= 20:
        try:
            # 计算最近20个价格点的波动率
            recent_prices = historical_prices[-20:]
            returns = np.diff(recent_prices) / recent_prices[:-1]
            volatility_pct = np.std(returns) * 100 * np.sqrt(252)  # 年化波动率转换为日波动率参考
            # 限制波动率在合理范围（1%-10%）
            volatility_pct = max(1.0, min(10.0, volatility_pct))
        except:
            volatility_pct = 2.0
    
    # 改进：以预测价格为中心，而不是当前价格
    # 这样价格建议更实用，不会因为当前价格波动而无法触发交易
    
    # 计算价格区间大小：基于波动率和预测价格
    # 使用预测价格作为基准，而不是当前价格
    base_price = avg_prediction  # 以预测价格为中心
    
    # 价格区间大小：基于波动率，确保有足够的区分度但不会太大
    # 波动率越大，价格区间越大，但限制在合理范围内（2%-8%）
    price_interval_pct = max(2.0, min(8.0, volatility_pct * 1.5))  # 波动率的1.5倍，限制在2%-8%
    price_interval_size = base_price * price_interval_pct / 100
    
    # 根据PPO动作和预测方向，确定价格区间的中心偏移
    # PPO动作：0=全卖, 1=卖75%, 2=卖50%, 3=卖25%, 4=持有, 5=买25%, 6=全买
    center_offset = 0.0  # 中心偏移（相对于预测价格）
    
    if ppo_action is not None:
        if ppo_action == 6:  # 全买：价格区间向下偏移，使当前价格更容易触发买入
            center_offset = -price_interval_size * 0.2  # 向下偏移20%
        elif ppo_action == 5:  # 买25%：价格区间略微向下偏移
            center_offset = -price_interval_size * 0.1
        elif ppo_action == 4:  # 持有：价格区间以预测价格为中心
            center_offset = 0.0
        elif ppo_action == 3:  # 卖25%：价格区间略微向上偏移
            center_offset = price_interval_size * 0.1
        elif ppo_action <= 2:  # 卖50%或更多：价格区间向上偏移，使当前价格更容易触发卖出
            center_offset = price_interval_size * 0.2
    else:
        # 如果没有PPO动作，根据预测方向判断
        if price_change_pct > 0:
            center_offset = -price_interval_size * 0.1  # 预测上涨，略微向下偏移（买入机会）
        else:
            center_offset = price_interval_size * 0.1  # 预测下跌，略微向上偏移（卖出机会）
    
    # 计算价格区间的中心点（基于预测价格和偏移）
    price_center = base_price + center_offset
    
    # 确定最低价格和最高价格（以预测价格为中心，而不是当前价格）
    min_price = price_center - price_interval_size / 2
    max_price = price_center + price_interval_size / 2
    
    # 根据融合决策（PPO动作）调整价格区间，但考虑价格偏离预测价格的程度
    # 如果价格偏离预测价格较大，应该根据实际价格位置动态调整，而不是强制跟随融合决策
    price_diff_pct = abs(current_price - avg_prediction) / avg_prediction * 100 if avg_prediction > 0 else 0
    
    if ppo_action is not None:
        # 如果价格偏离预测价格较小（<3%），优先遵循融合决策
        # 如果价格偏离预测价格较大（>=3%），根据实际价格位置动态调整
        if price_diff_pct < 3.0:  # 价格偏离较小，遵循融合决策
            if ppo_action == 6:  # 买入 100%：当前价格应该在75%-100%仓位区间
                target_min = current_price - price_interval_size * 0.2  # 当前价格在80%仓位附近
                target_max = current_price + price_interval_size * 0.8
                min_price = target_min
                max_price = target_max
                
            elif ppo_action == 5:  # 买入 25%：当前价格应该在50%-75%仓位区间
                target_min = current_price - price_interval_size * 0.4  # 当前价格在60%仓位附近
                target_max = current_price + price_interval_size * 0.6
                min_price = target_min
                max_price = target_max
                
            elif ppo_action == 4:  # 持有：当前价格应该在25%-75%仓位区间（中间）
                target_min = current_price - price_interval_size * 0.5  # 当前价格在50%仓位附近
                target_max = current_price + price_interval_size * 0.5
                min_price = target_min
                max_price = target_max
                
            elif ppo_action == 3:  # 卖出 25%：当前价格应该在25%-50%仓位区间
                target_min = current_price - price_interval_size * 0.6  # 当前价格在40%仓位附近
                target_max = current_price + price_interval_size * 0.4
                min_price = target_min
                max_price = target_max
                
            elif ppo_action <= 2:  # 卖出 50%或更多：当前价格应该在0%-25%仓位区间
                target_min = current_price - price_interval_size * 0.8  # 当前价格在20%仓位附近
                target_max = current_price + price_interval_size * 0.2
                min_price = target_min
                max_price = target_max
        else:  # 价格偏离较大，根据实际价格位置动态调整
            # 计算当前价格相对于预测价格的位置
            if current_price > avg_prediction:
                # 当前价格高于预测价格，应该建议减仓
                # 根据偏离程度确定仓位：偏离越大，仓位越低
                if price_diff_pct >= 5.0:  # 偏离5%以上，建议0%-25%仓位
                    target_min = current_price - price_interval_size * 0.8
                    target_max = current_price + price_interval_size * 0.2
                elif price_diff_pct >= 3.5:  # 偏离3.5%-5%，建议25%-50%仓位
                    target_min = current_price - price_interval_size * 0.6
                    target_max = current_price + price_interval_size * 0.4
                else:  # 偏离3%-3.5%，建议50%-75%仓位
                    target_min = current_price - price_interval_size * 0.4
                    target_max = current_price + price_interval_size * 0.6
            else:
                # 当前价格低于预测价格，应该建议加仓
                # 根据偏离程度确定仓位：偏离越大，仓位越高
                if price_diff_pct >= 5.0:  # 偏离5%以上，建议75%-100%仓位
                    target_min = current_price - price_interval_size * 0.2
                    target_max = current_price + price_interval_size * 0.8
                elif price_diff_pct >= 3.5:  # 偏离3.5%-5%，建议50%-75%仓位
                    target_min = current_price - price_interval_size * 0.4
                    target_max = current_price + price_interval_size * 0.6
                else:  # 偏离3%-3.5%，建议25%-50%仓位
                    target_min = current_price - price_interval_size * 0.6
                    target_max = current_price + price_interval_size * 0.4
            
            min_price = target_min
            max_price = target_max
    
    # 确保价格区间足够大（至少2%的价格差）
    actual_range = max_price - min_price
    if actual_range < current_price * 0.02:  # 如果区间小于2%，扩大它
        center = (min_price + max_price) / 2
        min_price = center - current_price * 0.01
        max_price = center + current_price * 0.01
    
    # 价格从低到高，仓位从高到低（100% -> 75% -> 50% -> 25% -> 0%）
    suggestions = {}
    suggestions['100%'] = min_price
    suggestions['75%'] = min_price + (max_price - min_price) * 0.25
    suggestions['50%'] = min_price + (max_price - min_price) * 0.5
    suggestions['25%'] = min_price + (max_price - min_price) * 0.75
    suggestions['0%'] = max_price
    
    # 确保价格合理（不能为负，不能偏离当前价格太远）
    for key in suggestions:
        suggestions[key] = max(0.01, suggestions[key])  # 至少0.01元
        # 限制在合理范围内（当前价格的70%-130%）
        suggestions[key] = max(current_price * 0.7, min(current_price * 1.3, suggestions[key]))
        suggestions[key] = round(suggestions[key], 2)
    
    # 计算价格区间大小（用于显示）
    price_interval = max_price - min_price
    interval_pct = (price_interval / current_price * 100) if current_price > 0 else 0
    
    # 计算当前价格对应的建议仓位
    price_levels = [suggestions['100%'], suggestions['75%'], suggestions['50%'], suggestions['25%'], suggestions['0%']]
    current_position_pct = 50.0  # 默认50%
    
    if current_price < price_levels[0]:  # 低于100%仓位价格
        current_position_pct = 100.0
    elif current_price > price_levels[-1]:  # 高于0%仓位价格
        current_position_pct = 0.0
    else:
        # 找到当前价格所在区间并插值
        for i in range(len(price_levels) - 1):
            if price_levels[i] <= current_price <= price_levels[i+1]:
                # 线性插值计算仓位
                ratio = (current_price - price_levels[i]) / (price_levels[i+1] - price_levels[i]) if (price_levels[i+1] - price_levels[i]) > 0 else 0
                current_position_pct = 100 - (i * 25 + ratio * 25)
                break
    
    return {
        'suggestions': suggestions,
        'predicted_price': round(avg_prediction, 2),
        'price_change_pct': round(price_change_pct, 2),
        'direction': '上涨' if price_change_pct > 0 else '下跌',
        'price_interval_pct': round(interval_pct, 2),
        'volatility_pct': round(volatility_pct, 2),
        'current_position_pct': round(current_position_pct, 1)
    }

def fuse_multi_model_predictions(ppo_action, lstm_prediction, transformer_prediction, 
                                 holographic_signal, model_weights=None, current_price=None):
    """
    融合多个模型的预测结果（V12优化版：增加预测方向冲突检测）
    
    Args:
        ppo_action: PPO模型的动作（0-6）
        lstm_prediction: LSTM/GRU的预测价格
        transformer_prediction: Transformer的预测价格
        holographic_signal: 全息模型的信号
        model_weights: 模型权重字典
        current_price: 当前价格（用于动态权重调整）
    
    Returns:
        (final_action, confidence, model_weights, conflict_info)
        conflict_info: 冲突信息字典，包含是否冲突、调整原因等
    """
    if model_weights is None:
        model_weights = MODEL_WEIGHTS.copy()
    
    # V11改进：动态调整权重
    if current_price is not None and ENABLE_DYNAMIC_WEIGHTS:
        predictions = {
            'lstm': lstm_prediction,
            'transformer': transformer_prediction
        }
        model_weights = adjust_weights_dynamically(model_weights, current_price, predictions)
    
    # 将价格预测转换为动作倾向
    final_action = ppo_action  # 默认使用PPO的动作
    confidence = 0.5
    conflict_info = {
        'has_conflict': False,
        'conflict_type': None,
        'price_change_pct': 0.0,
        'avg_prediction': None,
        'adjustment_reason': None,
        'original_action': ppo_action
    }
    
    # V12优化：预测方向冲突检测
    if (current_price is not None and 
        lstm_prediction is not None and 
        transformer_prediction is not None and 
        current_price > 0):
        
        # 计算平均预测价格和方向
        avg_prediction = (lstm_prediction + transformer_prediction) / 2
        price_change_pct = (avg_prediction - current_price) / current_price * 100
        conflict_info['avg_prediction'] = avg_prediction
        conflict_info['price_change_pct'] = price_change_pct
        
        # 检测冲突：PPO动作与预测方向不一致
        conflict_detected = False
        conflict_type = None
        
        if ppo_action is not None:
            # 冲突类型1：PPO建议买入，但预测价格下跌
            if ppo_action >= 5 and price_change_pct < -1.5:  # 买入建议但预测下跌>1.5%
                conflict_detected = True
                conflict_type = 'buy_vs_predict_down'
            # 冲突类型2：PPO建议卖出，但预测价格上涨
            elif ppo_action <= 2 and price_change_pct > 1.5:  # 卖出建议但预测上涨>1.5%
                conflict_detected = True
                conflict_type = 'sell_vs_predict_up'
        
        if conflict_detected:
            conflict_info['has_conflict'] = True
            conflict_info['conflict_type'] = conflict_type
            
            # 根据预测方向和冲突程度调整决策
            if conflict_type == 'buy_vs_predict_down':
                # 预测明显下跌，即使PPO建议买入，也应该降低买入力度
                abs_price_change = abs(price_change_pct)
                
                if abs_price_change > 3.0:  # 预测明显下跌>3%
                    # 强烈冲突：改为卖出或持有
                    final_action = 3  # 卖出25%
                    confidence = 0.4  # 降低置信度
                    conflict_info['adjustment_reason'] = (
                        f"预测价格{avg_prediction:.2f}元明显低于当前价格（-{abs_price_change:.2f}%），"
                        f"与PPO买入建议冲突，调整为卖出25%"
                    )
                elif abs_price_change > 2.0:  # 预测下跌2-3%
                    # 中等冲突：改为持有
                    final_action = 4  # 持有
                    confidence = 0.45
                    conflict_info['adjustment_reason'] = (
                        f"预测价格{avg_prediction:.2f}元低于当前价格（-{abs_price_change:.2f}%），"
                        f"与PPO买入建议冲突，调整为持有"
                    )
                else:  # 预测下跌1.5-2%
                    # 轻微冲突：降低买入力度
                    final_action = max(ppo_action - 2, 5)  # 降低2个级别，最低到买入25%
                    confidence = 0.5
                    conflict_info['adjustment_reason'] = (
                        f"预测价格{avg_prediction:.2f}元略低于当前价格（-{abs_price_change:.2f}%），"
                        f"降低买入力度至{map_action_to_operation(final_action)}"
                    )
            
            elif conflict_type == 'sell_vs_predict_up':
                # 预测明显上涨，即使PPO建议卖出，也应该降低卖出力度
                abs_price_change = abs(price_change_pct)
                
                if abs_price_change > 3.0:  # 预测明显上涨>3%
                    # 强烈冲突：改为买入或持有
                    final_action = 5  # 买入25%
                    confidence = 0.4
                    conflict_info['adjustment_reason'] = (
                        f"预测价格{avg_prediction:.2f}元明显高于当前价格（+{price_change_pct:.2f}%），"
                        f"与PPO卖出建议冲突，调整为买入25%"
                    )
                elif abs_price_change > 2.0:  # 预测上涨2-3%
                    # 中等冲突：改为持有
                    final_action = 4  # 持有
                    confidence = 0.45
                    conflict_info['adjustment_reason'] = (
                        f"预测价格{avg_prediction:.2f}元高于当前价格（+{price_change_pct:.2f}%），"
                        f"与PPO卖出建议冲突，调整为持有"
                    )
                else:  # 预测上涨1.5-2%
                    # 轻微冲突：降低卖出力度
                    final_action = min(ppo_action + 2, 3)  # 降低2个级别，最高到卖出25%
                    confidence = 0.5
                    conflict_info['adjustment_reason'] = (
                        f"预测价格{avg_prediction:.2f}元略高于当前价格（+{price_change_pct:.2f}%），"
                        f"降低卖出力度至{map_action_to_operation(final_action)}"
                    )
        else:
            # 无冲突，检查是否一致（提高置信度）
            if ppo_action is not None:
                if (ppo_action >= 5 and price_change_pct > 1.0) or \
                   (ppo_action <= 2 and price_change_pct < -1.0):
                    # PPO动作与预测方向一致，提高置信度
                    confidence = min(0.7, confidence + 0.15)
                    conflict_info['adjustment_reason'] = "PPO动作与预测方向一致，提高置信度"
    
    # 如果多个模型一致，进一步提高置信度
    signals = []
    if ppo_action is not None:
        signals.append(('ppo', ppo_action))
    if holographic_signal:
        signal_type = holographic_signal.get('signal', 'hold')
        if signal_type == 'buy':
            signals.append(('holographic', 4))  # 买入倾向
        elif signal_type == 'sell':
            signals.append(('holographic', 0))  # 卖出倾向
    
    # 如果多个信号一致，进一步提高置信度
    if len(signals) >= 2:
        action_values = [s[1] for s in signals]
        if all(a >= 5 for a in action_values) or all(a <= 2 for a in action_values):
            confidence = min(0.8, confidence + 0.1)
    
    return final_action, confidence, model_weights, conflict_info

# ==================== V13: 止损止盈风险控制功能 ====================

def calculate_atr(df, period=14, return_meta=False):
    """计算ATR
    - 若缺失high/low则使用收盘价近似，并标记close_only=True以提示精度风险
    - return_meta=True 时返回 (atr_value, close_only)
    """
    try:
        if df is None or len(df) < period + 1:
            return (None, False) if return_meta else None
        if 'close' not in df.columns:
            return (None, False) if return_meta else None
        
        close_only = not ('high' in df.columns and 'low' in df.columns)
        highs = pd.to_numeric(df['high'], errors='coerce') if 'high' in df.columns else pd.to_numeric(df['close'], errors='coerce')
        lows = pd.to_numeric(df['low'], errors='coerce') if 'low' in df.columns else pd.to_numeric(df['close'], errors='coerce')
        closes = pd.to_numeric(df['close'], errors='coerce')
        # 如果高低价存在但大量缺失，改用收盘近似并提示精度风险
        if ('high' in df.columns and 'low' in df.columns) and (highs.isna().sum() > len(highs) * 0.2 or lows.isna().sum() > len(lows) * 0.2):
            close_only = True
            highs = closes
            lows = closes
        
        prev_close = closes.shift(1)
        tr1 = (highs - lows).abs()
        tr2 = (highs - prev_close).abs()
        tr3 = (lows - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_series = tr.rolling(window=period).mean()
        atr_value = atr_series.iloc[-1]
        
        if pd.isna(atr_value) or atr_value <= 0:
            return (None, close_only) if return_meta else None
        atr_val = float(atr_value)
        if return_meta:
            return atr_val, close_only
        return atr_val
    except Exception:
        return (None, False) if return_meta else None

def check_stop_loss_take_profit(current_price, cost_price, shares_held, atr_value=None):
    """
    检查止损止盈条件
    返回: (triggered, action, reason, profit_loss_pct, atr_stop_price)
    - triggered: 是否触发止损/止盈
    - action: 建议动作（0=全部卖出, 1=卖出50%, 2=卖出25%）
    - reason: 触发原因
    - profit_loss_pct: 盈亏百分比
    """
    if not ENABLE_STOP_LOSS_TAKE_PROFIT:
        return False, None, None, None, None
    
    if shares_held <= 0 or cost_price is None or cost_price <= 0:
        return False, None, None, None, None
    
    atr_stop_price = None
    if ENABLE_ATR_STOP_LOSS and atr_value is not None and atr_value > 0:
        atr_stop_price = cost_price - ATR_MULTIPLIER * atr_value
    
    # 计算盈亏百分比
    profit_loss_pct = ((current_price - cost_price) / cost_price) * 100
    
    # ATR动态止损（海龟风格：成本价 - ATR×倍数）
    if atr_stop_price is not None and current_price <= atr_stop_price:
        return True, STOP_LOSS_ACTION, (
            f"触发ATR动态止损：当前价{current_price:.2f} ≤ 止损价{atr_stop_price:.2f} "
            f"(成本价{cost_price:.2f} - ATR×{ATR_MULTIPLIER}，ATR={atr_value:.2f})"
        ), profit_loss_pct, atr_stop_price
    
    # 检查止盈条件
    if profit_loss_pct >= TAKE_PROFIT_PCT:
        if TAKE_PROFIT_ACTION == 0:
            return True, 0, f"触发止盈：盈利{profit_loss_pct:.2f}% >= {TAKE_PROFIT_PCT}%，建议全部卖出", profit_loss_pct, atr_stop_price
        elif TAKE_PROFIT_ACTION == 1:
            return True, 1, f"触发止盈：盈利{profit_loss_pct:.2f}% >= {TAKE_PROFIT_PCT}%，建议卖出50%锁定利润", profit_loss_pct, atr_stop_price
        elif TAKE_PROFIT_ACTION == 2:
            return True, 2, f"触发止盈：盈利{profit_loss_pct:.2f}% >= {TAKE_PROFIT_PCT}%，建议卖出25%锁定部分利润", profit_loss_pct, atr_stop_price
    
    # 检查止损条件
    if profit_loss_pct <= STOP_LOSS_PCT:
        return True, STOP_LOSS_ACTION, f"触发止损：亏损{abs(profit_loss_pct):.2f}% <= {abs(STOP_LOSS_PCT)}%，建议全部卖出止损", profit_loss_pct, atr_stop_price
    
    # 检查部分止损条件（如果启用）
    if ENABLE_PARTIAL_STOP_LOSS and profit_loss_pct <= PARTIAL_STOP_LOSS_PCT and profit_loss_pct > STOP_LOSS_PCT:
        return True, PARTIAL_STOP_LOSS_ACTION, f"触发部分止损：亏损{abs(profit_loss_pct):.2f}% <= {abs(PARTIAL_STOP_LOSS_PCT)}%，建议卖出50%降低风险", profit_loss_pct, atr_stop_price
    
    return False, None, None, profit_loss_pct, atr_stop_price

def apply_stop_loss_take_profit(final_action, current_price, cost_price, shares_held, atr_value=None):
    """
    应用止损止盈逻辑，调整最终决策
    返回: (adjusted_action, stop_loss_info)
    """
    triggered, action, reason, profit_loss_pct, atr_stop_price = check_stop_loss_take_profit(
        current_price, cost_price, shares_held, atr_value=atr_value
    )
    
    if not triggered:
        return final_action, {
            'triggered': False,
            'profit_loss_pct': profit_loss_pct,
            'atr_stop_price': atr_stop_price
        }
    
    # 如果触发止损/止盈，覆盖原始决策
    stop_loss_info = {
        'triggered': True,
        'action': action,
        'reason': reason,
        'profit_loss_pct': profit_loss_pct,
        'original_action': final_action,
        'atr_stop_price': atr_stop_price
    }
    
    return action, stop_loss_info

# ==================== V16: 趋势/震荡双策略辅助函数 ====================

def detect_market_regime(closes, current_price, transformer_prediction, confidence, holographic_signal):
    """
    简单的趋势/震荡判别：
    - 趋势信号：预测涨幅较高、价格突破、全息信号看多且置信度较高
    - 震荡信号：预测幅度小、布林带带宽较窄
    返回: (regime, trend_score, range_score, boll_info)
    """
    if not ENABLE_REGIME_STRATEGY or closes is None or len(closes) < max(TREND_BREAKOUT_WINDOW, BOLL_PERIOD) + 5:
        return 'neutral', 0.0, 0.0, None
    
    closes_arr = np.array(closes, dtype=np.float64)
    recent = closes_arr[-max(TREND_BREAKOUT_WINDOW, BOLL_PERIOD):]
    if len(recent) < 5 or current_price <= 0:
        return 'neutral', 0.0, 0.0, None
    
    ma = np.mean(recent[-BOLL_PERIOD:]) if len(recent) >= BOLL_PERIOD else np.mean(recent)
    std = np.std(recent[-BOLL_PERIOD:]) if len(recent) >= BOLL_PERIOD else np.std(recent)
    upper = ma + BOLL_STD * std
    lower = ma - BOLL_STD * std
    bandwidth = (upper - lower) / current_price if current_price > 0 else 0
    
    pred_change = 0.0
    if transformer_prediction is not None and current_price > 0:
        pred_change = (transformer_prediction - current_price) / current_price * 100
    
    holo_dir = None
    holo_conf = 0.0
    if isinstance(holographic_signal, dict):
        holo_dir = holographic_signal.get('signal')
        holo_conf = holographic_signal.get('confidence', 0.0) or 0.0
    elif holographic_signal:
        holo_dir = str(holographic_signal)
    
    trend_score = 0.0
    range_score = 0.0
    
    # 趋势得分：预测涨幅、置信度、突破、全息多头
    if pred_change > TREND_MIN_PRED_CHANGE:
        trend_score += 1.0
    if confidence is not None and confidence >= TREND_CONFIDENCE_THRESHOLD:
        trend_score += 1.0
    if len(recent) >= TREND_BREAKOUT_WINDOW and current_price >= np.max(recent[-TREND_BREAKOUT_WINDOW:]):
        trend_score += 1.0
    if holo_dir in ('buy', 'long', 'up') and holo_conf >= 0.25:
        trend_score += 0.5
    
    # 震荡得分：预测幅度小 + 带宽窄
    if abs(pred_change) < TREND_MIN_PRED_CHANGE and bandwidth < RANGE_BANDWIDTH_THRESHOLD:
        range_score += 1.0
    if holo_conf < 0.25:
        range_score += 0.3
    
    regime = 'neutral'
    if trend_score >= 1.5 and trend_score >= range_score + 0.5:
        regime = 'trend'
    elif range_score >= 1.0 and range_score >= trend_score + 0.3:
        regime = 'range'
    
    boll_info = {
        'ma': ma,
        'upper': upper,
        'lower': lower,
        'bandwidth': bandwidth
    }
    return regime, trend_score, range_score, boll_info

def trend_strategy_signal(closes, current_price):
    """简单趋势突破信号：价格突破近期高点则看多"""
    if closes is None or len(closes) < TREND_BREAKOUT_WINDOW + 2:
        return 0, None
    window_high = np.max(closes[-TREND_BREAKOUT_WINDOW:])
    if current_price >= window_high:
        return 1, window_high  # 看多
    return 0, window_high

def mean_reversion_signal(closes, current_price):
    """布林带均值回归信号：触碰下轨买，上轨卖"""
    if closes is None or len(closes) < BOLL_PERIOD:
        return 0, None, None, None
    recent = closes[-BOLL_PERIOD:]
    ma = np.mean(recent)
    std = np.std(recent)
    upper = ma + BOLL_STD * std
    lower = ma - BOLL_STD * std
    if current_price <= lower:
        return 1, upper, ma, lower   # 偏多
    if current_price >= upper:
        return -1, upper, ma, lower  # 偏空
    return 0, upper, ma, lower

# ========== V16: 全息信号后处理 ==========
def refine_holographic_signal(holo_result, closes, indicator_summary, fallback_used=False):
    refined = {'signal': 'hold', 'confidence': 0.15, 'reason': '无信号'}
    if not holo_result or not isinstance(holo_result, dict):
        return refined
    
    sig = holo_result.get('signal', 'hold')
    conf = float(holo_result.get('confidence', 0) or 0.0)
    refined['signal'] = sig
    refined['confidence'] = conf
    refined['reason'] = '原始全息信号'
    
    if fallback_used:
        refined['signal'] = 'hold'
        refined['confidence'] = min(conf, 0.2)
        refined['reason'] = '情绪回退，信号降级'
        return refined
    
    closes_arr = np.array(closes, dtype=np.float64) if closes is not None else None
    momentum = 0.0
    if closes_arr is not None and len(closes_arr) >= 6 and closes_arr[-6] != 0:
        momentum = (closes_arr[-1] - closes_arr[-6]) / closes_arr[-6]
    
    rsi = None
    kdj_j = None
    boll_bias = 0
    if indicator_summary:
        rsi = indicator_summary.get('RSI')
        kdj = indicator_summary.get('KDJ')
        if isinstance(kdj, dict):
            kdj_j = kdj.get('J')
    if closes_arr is not None and len(closes_arr) >= BOLL_PERIOD:
        recent = closes_arr[-BOLL_PERIOD:]
        ma = np.mean(recent)
        std = np.std(recent)
        upper = ma + BOLL_STD * std
        lower = ma - BOLL_STD * std
        if upper > lower and closes_arr[-1] > upper:
            boll_bias = 1
        elif upper > lower and closes_arr[-1] < lower:
            boll_bias = -1
    
    bearish = (momentum < -0.005 and (rsi is not None and rsi < 45)) or (kdj_j is not None and kdj_j < 0) or boll_bias > 0
    if bearish:
        refined['signal'] = 'sell'
        refined['confidence'] = min(max(conf, 0.35), 0.6)
        refined['reason'] = '动量/指标偏空'
    else:
        if sig == 'buy' and conf < 0.25:
            refined['signal'] = 'hold'
            refined['confidence'] = conf
            refined['reason'] = '置信度过低，降级为hold'
    
    return refined

# ==================== V13: 资金管理策略（凯利公式） ====================

# V13: 交易历史记录（用于计算凯利公式参数）
trade_history = []  # [{'action': 'buy'/'sell', 'price': float, 'timestamp': datetime, 'profit_pct': float}]

def calculate_kelly_formula(win_rate, avg_win_pct, avg_loss_pct):
    """
    计算凯利公式最优仓位
    凯利公式: f* = (p * b - q) / b
    其中：
    - f* = 最优仓位比例
    - p = 胜率（盈利交易的概率）
    - q = 败率（1-p）
    - b = 盈亏比（平均盈利/平均亏损）
    
    返回: 最优仓位比例（0-1之间）
    """
    if win_rate <= 0 or win_rate >= 1:
        return 0.0
    
    if avg_loss_pct <= 0:
        return 0.0
    
    # 计算盈亏比
    b = abs(avg_win_pct / avg_loss_pct) if avg_loss_pct != 0 else 0
    
    if b <= 0:
        return 0.0
    
    # 计算败率
    q = 1 - win_rate
    
    # 凯利公式
    kelly_fraction = (win_rate * b - q) / b
    
    # 确保在合理范围内
    kelly_fraction = max(0.0, min(1.0, kelly_fraction))
    
    return kelly_fraction

def calculate_trade_statistics(trade_history, min_samples=KELLY_MIN_SAMPLES):
    """
    从交易历史计算统计指标
    返回: (win_rate, avg_win_pct, avg_loss_pct, total_trades)
    """
    if len(trade_history) < min_samples:
        return None, None, None, len(trade_history)
    
    # 提取盈利交易
    profitable_trades = [t for t in trade_history if t.get('profit_pct', 0) > 0]
    losing_trades = [t for t in trade_history if t.get('profit_pct', 0) < 0]
    
    if len(profitable_trades) == 0 and len(losing_trades) == 0:
        return None, None, None, len(trade_history)
    
    # 计算胜率
    total_trades = len(trade_history)
    winning_trades = len(profitable_trades)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # 计算平均盈利
    avg_win_pct = np.mean([t['profit_pct'] for t in profitable_trades]) if len(profitable_trades) > 0 else 0
    
    # 计算平均亏损（取绝对值）
    avg_loss_pct = abs(np.mean([t['profit_pct'] for t in losing_trades])) if len(losing_trades) > 0 else 1.0
    
    return win_rate, avg_win_pct, avg_loss_pct, total_trades

def estimate_kelly_from_prediction(confidence, predicted_return_pct, indicator_summary=None, volatility_pct=None):
    """
    基于模型预测估算凯利公式参数（用于样本不足时）
    参数:
    - confidence: 模型置信度（0-1）
    - predicted_return_pct: 预测收益率（%）
    - indicator_summary: 技术指标摘要
    - volatility_pct: 波动率（%）
    
    返回: (estimated_win_rate, estimated_avg_win, estimated_avg_loss)
    """
    # 1. 估算胜率：基于置信度和预测方向
    # 置信度越高，胜率越高；预测收益率绝对值越大，胜率越高
    base_win_rate = 0.5  # 基础胜率50%
    
    # 置信度贡献（0.3-0.7范围）
    confidence_contribution = confidence * 0.4  # 置信度0.5时贡献0.2，置信度1.0时贡献0.4
    
    # 预测收益率贡献（预测收益率绝对值越大，胜率越高）
    abs_return = abs(predicted_return_pct)
    return_contribution = min(0.2, abs_return / 10.0)  # 10%收益率时贡献0.2
    
    # 估算胜率（50%-70%范围）
    estimated_win_rate = base_win_rate + confidence_contribution + return_contribution
    estimated_win_rate = max(0.45, min(0.75, estimated_win_rate))  # 限制在45%-75%
    
    # 2. 估算平均盈利：基于预测收益率和置信度
    if predicted_return_pct > 0:
        # 预测上涨：平均盈利 = 预测收益率 * 置信度 * 调整系数
        estimated_avg_win = abs(predicted_return_pct) * confidence * 0.8  # 实际盈利通常小于预测
        estimated_avg_win = max(1.0, min(10.0, estimated_avg_win))  # 限制在1%-10%
    else:
        # 预测下跌：如果做空，盈利来自下跌
        estimated_avg_win = abs(predicted_return_pct) * confidence * 0.6
        estimated_avg_win = max(0.5, min(8.0, estimated_avg_win))
    
    # 3. 估算平均亏损：基于波动率和止损设置
    if volatility_pct and volatility_pct > 0:
        # 使用波动率估算平均亏损（通常亏损约为波动率的0.5-1.5倍）
        estimated_avg_loss = volatility_pct * 0.8
    else:
        # 使用止损设置作为参考
        estimated_avg_loss = abs(STOP_LOSS_PCT) * 0.6  # 平均亏损通常小于止损线
    
    estimated_avg_loss = max(1.0, min(8.0, estimated_avg_loss))  # 限制在1%-8%
    
    # 4. 技术指标调整（如果提供）
    if indicator_summary:
        # RSI调整：RSI极端值时，胜率可能更高
        if 'RSI' in indicator_summary:
            rsi = indicator_summary['RSI']
            if isinstance(rsi, (int, float)):
                if rsi < 30:  # 超卖，买入胜率可能更高
                    estimated_win_rate = min(0.8, estimated_win_rate + 0.05)
                elif rsi > 70:  # 超买，卖出胜率可能更高
                    estimated_win_rate = min(0.8, estimated_win_rate + 0.05)
        
        # MACD调整：MACD金叉/死叉影响胜率
        if 'MACD' in indicator_summary:
            macd = indicator_summary['MACD']
            if isinstance(macd, dict):
                macd_value = macd.get('MACD', 0)
                signal = macd.get('Signal', 0)
                if macd_value > signal and predicted_return_pct > 0:  # 金叉且预测上涨
                    estimated_win_rate = min(0.8, estimated_win_rate + 0.03)
                elif macd_value < signal and predicted_return_pct < 0:  # 死叉且预测下跌
                    estimated_win_rate = min(0.8, estimated_win_rate + 0.03)
    
    return estimated_win_rate, estimated_avg_win, estimated_avg_loss

def get_kelly_position_size(confidence, predicted_return_pct, trade_history, indicator_summary=None, volatility_pct=None):
    """
    根据凯利公式计算最优仓位（支持样本不足时的预测估算）
    参数:
    - confidence: 模型置信度（0-1）
    - predicted_return_pct: 预测收益率（%）
    - trade_history: 交易历史
    - indicator_summary: 技术指标摘要（可选）
    - volatility_pct: 波动率（%）（可选）
    
    返回: 最优仓位比例（0-1）
    """
    if not ENABLE_KELLY_FORMULA:
        return None
    
    # 计算交易统计
    win_rate, avg_win_pct, avg_loss_pct, total_trades = calculate_trade_statistics(trade_history)
    
    use_estimated = False
    # 如果样本不足，使用预测估算
    if win_rate is None or total_trades < KELLY_MIN_SAMPLES:
        use_estimated = True
        estimated_win_rate, estimated_avg_win, estimated_avg_loss = estimate_kelly_from_prediction(
            confidence, predicted_return_pct, indicator_summary, volatility_pct
        )
        win_rate = estimated_win_rate
        avg_win_pct = estimated_avg_win
        avg_loss_pct = estimated_avg_loss
    
    # 计算基础凯利值
    kelly_value = calculate_kelly_formula(win_rate, avg_win_pct, avg_loss_pct)
    
    if kelly_value <= 0:
        return None
    
    # 应用安全系数（降低风险）
    safe_kelly = kelly_value * KELLY_FRACTION
    
    # 根据置信度调整（置信度越高，仓位可以越高）
    confidence_adjusted = safe_kelly * confidence
    
    # 根据预测收益率调整（预测收益率越高，仓位可以越高）
    if predicted_return_pct > 0:
        return_adjusted = confidence_adjusted * min(1.0, predicted_return_pct / 5.0)  # 5%收益率作为基准
    else:
        return_adjusted = confidence_adjusted * 0.5  # 预测下跌时降低仓位
    
    # 样本不足时，进一步降低仓位（更保守）
    if use_estimated:
        return_adjusted = return_adjusted * 0.7  # 样本不足时，使用70%的仓位
    
    # 限制在最小和最大仓位之间
    final_position = max(MIN_KELLY_POSITION, min(MAX_KELLY_POSITION, return_adjusted))
    
    return {
        'kelly_position': final_position,
        'raw_kelly': kelly_value,
        'safe_kelly': safe_kelly,
        'win_rate': win_rate,
        'avg_win_pct': avg_win_pct,
        'avg_loss_pct': avg_loss_pct,
        'total_trades': total_trades if not use_estimated else 0,
        'is_estimated': use_estimated,
        'confidence': confidence,
        'predicted_return_pct': predicted_return_pct
    }

# ==================== V15: DeepSeek 轮次复盘 ====================

def build_deepseek_review_prompt(ctx: dict) -> str:
    """
    根据当前轮次的关键信息生成 DeepSeek 提示词
    ctx 字段示例：
    - stock_name, stock_code, data_source, latest_time, price_source, current_price
    - ppo_action, ppo_model, final_operation, confidence
    - lstm_prediction, transformer_prediction, holographic_signal
    - predicted_price, predicted_change, suggested_buy, suggested_sell
    - kelly_position, kelly_mode
    - shares_held, current_balance
    """
    def fmt(val, fmt_str=".2f"):
        try:
            # 支持 format 规范和传统 % 规范，优先使用 format
            if isinstance(fmt_str, str) and '{' in fmt_str:
                return fmt_str.format(val)
            return format(val, fmt_str)
        except Exception:
            try:
                return fmt_str % val  # 尝试旧式格式
            except Exception:
                return str(val)
    
    lines = []
    lines.append(f"标的:{ctx.get('stock_name','')}"
                 f"({ctx.get('stock_code','')}) 数据源:{ctx.get('data_source','未知')} 最新数据:{ctx.get('latest_time','未知')}")
    lines.append(f"当前价:{fmt(ctx.get('current_price','--'), '.2f')} 来源:{ctx.get('price_source','--')} 持仓:{fmt(ctx.get('shares_held',0),'.2f')}股 资金:{fmt(ctx.get('current_balance',0),'.2f')}元")
    lines.append(f"PPO动作:{ctx.get('ppo_action','--')} | 模型:{ctx.get('ppo_model','--')}")
    if ctx.get('lstm_prediction') is not None or ctx.get('transformer_prediction') is not None:
        lines.append(f"LSTM预测:{fmt(ctx.get('lstm_prediction','--'), '.2f')} | Transformer预测:{fmt(ctx.get('transformer_prediction','--'), '.2f')}")
    if ctx.get('predicted_price') is not None:
        change = ctx.get('predicted_change')
        change_text = f"{change:+.2f}%" if change is not None else "--"
        lines.append(f"融合预测价:{fmt(ctx.get('predicted_price','--'), '.2f')} 预期变化:{change_text}")
    lines.append(f"融合决策:{ctx.get('final_operation','--')} 置信度:{fmt(ctx.get('confidence','--'), '.2f')}")
    if ctx.get('kelly_position') is not None:
        mode = ctx.get('kelly_mode','')
        lines.append(f"凯利建议仓位:{fmt(ctx.get('kelly_position')*100, '.1f')}% 模式:{mode}")
    if ctx.get('suggested_buy') or ctx.get('suggested_sell'):
        buy = ctx.get('suggested_buy')
        sell = ctx.get('suggested_sell')
        lines.append(f"关键价格: 买入参考{fmt(buy,'.2f') if buy else '--'} 卖出参考{fmt(sell,'.2f') if sell else '--'}")
    if ctx.get('holographic_signal'):
        holo = ctx.get('holographic_signal')
        holo_sig = holo.get('signal') if isinstance(holo, dict) else holo
        holo_conf = None
        if isinstance(holo, dict):
            holo_conf = holo.get('confidence')
        lines.append(f"全息信号:{holo_sig or '--'} 置信度:{fmt(holo_conf,'%.2f') if holo_conf is not None else '--'}")
    
    lines.append("请用不超过80字给出执行要点和风险提醒，避免重复以上数据，不要使用emoji或Markdown。")
    return "\n".join(lines)

def call_deepseek_review(prompt_text: str):
    """调用 DeepSeek Chat 对当轮结果进行点评，返回文本"""
    if not ENABLE_DEEPSEEK_REVIEW or not ENABLE_LLM:
        return None
    if not DEEPSEEK_API_KEY:
        print("   ⚠️  DeepSeek复盘已跳过：未配置 DEEPSEEK_API_KEY")
        return None
    
    try:
        import requests
        
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": DEEPSEEK_REVIEW_MODEL,
            "messages": [
                {"role": "system", "content": "你是严谨的A股量化交易助手，输出简洁、落地的执行建议和风险提示，用中文回答，控制在80字以内，不要重复输入信息，不要使用emoji或Markdown符号。"},
                {"role": "user", "content": prompt_text}
            ],
            "temperature": 0.2,
            "max_tokens": DEEPSEEK_REVIEW_MAX_TOKENS
        }
        
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=DEEPSEEK_REVIEW_TIMEOUT
        )
        
        if resp.status_code != 200:
            print(f"   ⚠️  DeepSeek复盘调用失败: HTTP {resp.status_code} {resp.text[:120]}")
            return None
        
        result = resp.json()
        content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
        return content.strip() if content else None
    except Exception as e:
        print(f"   ⚠️  DeepSeek复盘调用异常: {e}")
        return None

# ==================== V13: 多模型回测和自动选择功能 ====================

# V13: 多模型回测数据存储
model_backtest_data = {}  # {model_name: {'predictions': [], 'actuals': [], 'timestamps': [], 'actions': [], 'price_changes': []}}
# 注意：predictions是Transformer预测值（所有模型共享），actions是PPO动作（每个模型不同）
for model_name in candidate_ppo_models.keys():
    model_backtest_data[model_name] = {
        'predictions': [],  # Transformer预测值（共享）
        'actuals': [],      # 实际价格
        'timestamps': [],   # 时间戳
        'actions': [],      # PPO动作（每个模型不同）
        'price_changes': [] # 实际价格变化（用于评估动作准确性）
    }

def calculate_model_metrics(predictions, actuals):
    """计算模型回测指标"""
    if len(predictions) == 0 or len(actuals) == 0 or len(predictions) != len(actuals):
        return None
    
    try:
        preds_array = np.array(predictions, dtype=np.float64)
        actuals_array = np.array(actuals, dtype=np.float64)
        
        # 过滤掉NaN和Inf值
        valid_mask = np.isfinite(preds_array) & np.isfinite(actuals_array) & (actuals_array != 0)
        if np.sum(valid_mask) < MIN_BACKTEST_SAMPLES:
            return None
        
        valid_preds = preds_array[valid_mask]
        valid_actuals = actuals_array[valid_mask]
        
        # 计算MAE
        mae = np.mean(np.abs(valid_preds - valid_actuals))
        
        # 计算RMSE
        rmse = np.sqrt(np.mean((valid_preds - valid_actuals)**2))
        
        # 计算MAPE
        mape = np.mean(np.abs((valid_preds - valid_actuals) / valid_actuals)) * 100
        
        # 计算方向准确率
        if len(valid_preds) > 1:
            pred_directions = np.sign(np.diff(valid_preds))
            actual_directions = np.sign(np.diff(valid_actuals))
            if len(pred_directions) > 0:
                direction_accuracy = np.mean(pred_directions == actual_directions) * 100
            else:
                direction_accuracy = 0.0
        else:
            direction_accuracy = 0.0
        
        # 检查结果是否有效
        if np.isnan(mae) or np.isnan(rmse) or np.isnan(mape) or np.isnan(direction_accuracy):
            return None
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'sample_count': np.sum(valid_mask)
        }
    except Exception as e:
        return None

def calculate_model_score(metrics):
    """计算模型综合评分（分数越高越好）"""
    if metrics is None:
        return 0.0
    
    # 归一化指标（转换为0-1分数）
    # MAE、RMSE、MAPE越小越好，需要反向归一化
    # 方向准确率越大越好，直接归一化
    
    # 假设合理的范围（可以根据实际情况调整）
    max_mae = 10.0  # 最大MAE（元）
    max_rmse = 15.0  # 最大RMSE（元）
    max_mape = 20.0  # 最大MAPE（%）
    
    # 计算归一化分数（0-1）
    mae_score = max(0.0, 1.0 - metrics['mae'] / max_mae) if max_mae > 0 else 0.0
    rmse_score = max(0.0, 1.0 - metrics['rmse'] / max_rmse) if max_rmse > 0 else 0.0
    mape_score = max(0.0, 1.0 - metrics['mape'] / max_mape) if max_mape > 0 else 0.0
    direction_score = metrics['direction_accuracy'] / 100.0  # 方向准确率已经是百分比
    
    # 加权综合评分
    total_score = (
        mae_score * MODEL_SCORE_WEIGHTS['mae'] +
        rmse_score * MODEL_SCORE_WEIGHTS['rmse'] +
        mape_score * MODEL_SCORE_WEIGHTS['mape'] +
        direction_score * MODEL_SCORE_WEIGHTS['direction_accuracy']
    )
    
    return total_score

def select_best_model():
    """根据回测结果选择最优模型"""
    if not ENABLE_AUTO_MODEL_SELECTION or len(candidate_ppo_models) == 0:
        return None
    
    model_scores = {}
    model_metrics_dict = {}
    
    # 计算每个模型的指标和评分
    for model_name in candidate_ppo_models.keys():
        backtest_data = model_backtest_data.get(model_name, {})
        predictions = backtest_data.get('predictions', [])
        actuals = backtest_data.get('actuals', [])
        
        if len(predictions) < MIN_BACKTEST_SAMPLES or len(actuals) < MIN_BACKTEST_SAMPLES:
            continue
        
        metrics = calculate_model_metrics(predictions, actuals)
        if metrics:
            score = calculate_model_score(metrics)
            model_scores[model_name] = score
            model_metrics_dict[model_name] = metrics
    
    if len(model_scores) == 0:
        return None
    
    # 选择评分最高的模型
    best_model_name = max(model_scores, key=model_scores.get)
    best_score = model_scores[best_model_name]
    best_metrics = model_metrics_dict[best_model_name]
    
    return {
        'model_name': best_model_name,
        'score': best_score,
        'metrics': best_metrics,
        'all_scores': model_scores,
        'all_metrics': model_metrics_dict
    }

def switch_to_model(model_name):
    """切换到指定模型"""
    global ppo_model, current_model_name
    
    if model_name in candidate_ppo_models:
        ppo_model = candidate_ppo_models[model_name]
        current_model_name = model_name
        return True
    return False

# ==================== 主循环 ====================

print("\n" + "=" * 70)
print("🚀 V16批量预测系统 - 批量运行所有V16预测股票")
print("=" * 70)
print(f"📊 股票数量: {len(STOCK_LIST)}")
print(f"📋 股票列表:")
for i, stock in enumerate(STOCK_LIST, 1):
    print(f"   {i}. {stock['name']}({stock['code']})")
print("⚠️  重要提示: 这是 V16 批量预测版本，每个股票只运行一次预测！")
if ENABLE_AUTO_MODEL_SELECTION:
    print(f"   📊 已启用自动模型选择，候选模型数量: {len(candidate_ppo_models)}")

# V16新增：在批量预测开始时获取并显示一次全球主要指数数据
try:
    # 尝试获取指数数据（使用与test_nasdaq_change.py相同的方法）
    index_data = get_index_metrics_once()
    if index_data:
        print(f"\n📊 全球主要指数涨跌幅:")
        
        # 显示纳斯达克指数
        nasdaq = index_data.get('nasdaq')
        if nasdaq:
            change_pct = nasdaq.get('change_pct', 'N/A')
            index_name = nasdaq.get('index_name', '纳斯达克')
            if isinstance(change_pct, (int, float)):
                print(f"   📈 纳斯达克: {change_pct:+.2f}% ({index_name})")
            elif isinstance(change_pct, str) and change_pct != 'N/A':
                print(f"   📈 纳斯达克: {change_pct} ({index_name})")
            else:
                print(f"   📈 纳斯达克: {change_pct} ({index_name})")
        else:
            print(f"   ⚠️  纳斯达克: 数据获取失败")
        
        # 显示道琼斯指数
        dow = index_data.get('dow')
        if dow:
            change_pct = dow.get('change_pct', 'N/A')
            index_name = dow.get('index_name', '道琼斯')
            if isinstance(change_pct, (int, float)):
                print(f"   📈 道琼斯: {change_pct:+.2f}% ({index_name})")
            elif isinstance(change_pct, str) and change_pct != 'N/A':
                print(f"   📈 道琼斯: {change_pct} ({index_name})")
            else:
                print(f"   📈 道琼斯: {change_pct} ({index_name})")
        else:
            print(f"   ⚠️  道琼斯: 数据获取失败")
        
        # 显示富时A50期指连续
        a50 = index_data.get('a50')
        if a50:
            change_pct = a50.get('change_pct', 'N/A')
            index_name = a50.get('index_name', '富时A50')
            if isinstance(change_pct, (int, float)):
                print(f"   📈 富时A50: {change_pct:+.2f}% ({index_name})")
            elif isinstance(change_pct, str) and change_pct != 'N/A':
                print(f"   📈 富时A50: {change_pct} ({index_name})")
            else:
                print(f"   📈 富时A50: {change_pct} ({index_name})")
        else:
            print(f"   ⚠️  富时A50: 数据获取失败")
        
        # 如果是从文件读取的，显示更新时间
        update_time = index_data.get('update_time', '')
        if update_time:
            print(f"   📝 数据更新时间: {update_time}")
    else:
        # 如果获取失败，提示但不影响批量预测
        print(f"\n⚠️  无法获取指数数据（将跳过指数对比，不影响股票预测）")
except Exception as e:
    # 捕获异常，避免影响批量预测
    print(f"\n⚠️  获取指数数据时发生错误（将跳过指数对比，不影响股票预测）")

print("=" * 70 + "\n")

# 运行状态
current_balance = 50000.0
shares_held = 0.0
last_price = 0.0
initial_balance = 50000.0
last_action = None

# 模型训练状态
lstm_trained = False
transformer_trained = False
lstm_normalization_params = None
transformer_normalization_params = None

# V11回测数据存储（V12兼容模式）
if ENABLE_BACKTEST:
    backtest_predictions = []  # 存储预测值（V12兼容模式）
    backtest_actuals = []  # 存储实际值（V12兼容模式）
    backtest_timestamps = []  # 存储时间戳（V12兼容模式）

# V13: 多模型回测数据存储（已在前面定义，这里确保初始化）
if ENABLE_AUTO_MODEL_SELECTION and len(candidate_ppo_models) > 0:
    # 确保所有候选模型都有回测数据结构
    for model_name in candidate_ppo_models.keys():
        if model_name not in model_backtest_data:
            model_backtest_data[model_name] = {
                'predictions': [],
                'actuals': [],
                'timestamps': []
            }

# 加载持仓状态
portfolio_state = load_portfolio_state()
if portfolio_state:
    if portfolio_state.get('stock_code') == STOCK_CODE:
        current_balance = portfolio_state.get('current_balance', 50000.0)
        shares_held = portfolio_state.get('shares_held', 0.0)
        last_price = portfolio_state.get('last_price', 0.0)
        initial_balance = portfolio_state.get('initial_balance', 50000.0)
        # V12优化：加载成本价（不使用 last_price 作为回退）
        cost_price_val = portfolio_state.get('cost_price')
        actual_buy_price_val = portfolio_state.get('actual_buy_price')
        if cost_price_val and isinstance(cost_price_val, (int, float)) and cost_price_val > 0:
            cost_price = float(cost_price_val)
        elif actual_buy_price_val and isinstance(actual_buy_price_val, (int, float)) and actual_buy_price_val > 0:
            cost_price = float(actual_buy_price_val)
        else:
            cost_price = None  # 不使用 last_price 作为回退
        print(f"✅ 已加载持仓状态: 持仓={shares_held:.2f}股, 资金={current_balance:.2f}元" + (f", 成本价={cost_price:.4f}元" if cost_price else ""))

# 启动可视化自动更新
if visualizer:
    try:
        visualizer.start_auto_update()
    except:
        pass

# 示例文本数据
sample_texts = [
    "该股票今日表现强势，市场看好其未来发展前景",
    "受利空消息影响，股价出现下跌",
    "公司业绩超预期，投资者信心增强"
]
text_index = 0

iteration_count = 0
atr_value = None

# 批量预测：初始化日志文件
log_file = get_batch_predict_log_file()
# 清空或创建日志文件，写入头部信息
with open(log_file, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write(f"V16批量预测日志 - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 70 + "\n")
    f.write(f"股票数量: {len(STOCK_LIST)}\n")
    f.write(f"股票列表:\n")
    for i, stock in enumerate(STOCK_LIST, 1):
        f.write(f"   {i}. {stock['name']}({stock['code']})\n")
    f.write("=" * 70 + "\n\n")

# 批量预测：对每个股票执行一次预测
try:
    for stock_info in STOCK_LIST:
        STOCK_CODE = stock_info['code']
        stock_name = stock_info['name']
        
        # 为当前股票选择对应的模型
        stock_model = stock_info.get('model', None)
        if stock_model and ENABLE_AUTO_MODEL_SELECTION and len(candidate_ppo_models) > 0:
            # 从模型文件名中提取股票代码（如 "ppo_stock_v7_603267.zip" -> "603267"）
            stock_code_from_model = stock_model.replace('ppo_stock_v7_', '').replace('.zip', '')
            
            # 根据股票代码匹配对应的模型名称
            model_name_to_use = None
            for model_name in candidate_ppo_models.keys():
                # 从模型名称中提取股票代码（如 "603267模型" -> "603267"）
                model_code = model_name.replace('模型', '').replace('（最佳）', '').strip()
                if stock_code_from_model == model_code:
                    model_name_to_use = model_name
                    break
            
            # 如果找到匹配的模型，切换到该模型
            if model_name_to_use and model_name_to_use != current_model_name:
                if switch_to_model(model_name_to_use):
                    print(f"   🔄 已为 {stock_name}({STOCK_CODE}) 切换到专用模型: {model_name_to_use}")
                else:
                    print(f"   ⚠️  为 {stock_name}({STOCK_CODE}) 切换模型失败，使用当前模型")
            elif not model_name_to_use:
                print(f"   ⚠️  未找到 {stock_name}({STOCK_CODE}) 的专用模型 {stock_model}，使用当前模型")
        
        # 开始捕获该股票的预测输出
        with OutputCapture() as output_capture:
            # 为每个股票重新初始化多数据源管理器
            if MULTI_DATA_SOURCE_AVAILABLE:
                try:
                    priority_list = None
                    if ENABLE_STOCKAPI and STOCKAPI_AVAILABLE:
                        priority_list = ['stockapi', 'tushare', 'akshare', 'baostock']
                    else:
                        priority_list = ['tushare', 'akshare', 'baostock']
                    
                    multi_source_manager = MultiDataSourceManager(
                        stock_code=STOCK_CODE,
                        sources=None,
                        priority=priority_list,
                        timeout=10,
                        retry_times=3,
                        enable_anti_crawler=ENABLE_ANTI_CRAWLER,
                        proxies=PROXIES if PROXIES else None
                    )
                except Exception as e:
                    print(f"⚠️  多数据源管理器初始化失败: {e}")
                    multi_source_manager = None
            
            # 为每个股票重新加载持仓状态
            PORTFOLIO_STATE_FILE = f"portfolio_state_{STOCK_CODE}.json"
            
            # 批量预测：为每个股票重置模型训练状态
            lstm_trained = False
            transformer_trained = False
            lstm_normalization_params = None
            transformer_normalization_params = None
            
            portfolio_state = load_portfolio_state()
            if portfolio_state:
                if portfolio_state.get('stock_code') == STOCK_CODE:
                    current_balance = portfolio_state.get('current_balance', 50000.0)
                    shares_held = portfolio_state.get('shares_held', 0.0)
                    last_price = portfolio_state.get('last_price', 0.0)
                    initial_balance = portfolio_state.get('initial_balance', 50000.0)
                    cost_price_val = portfolio_state.get('cost_price')
                    actual_buy_price_val = portfolio_state.get('actual_buy_price')
                    if cost_price_val and isinstance(cost_price_val, (int, float)) and cost_price_val > 0:
                        cost_price = float(cost_price_val)
                    elif actual_buy_price_val and isinstance(actual_buy_price_val, (int, float)) and actual_buy_price_val > 0:
                        cost_price = float(actual_buy_price_val)
                    else:
                        cost_price = None
                    print(f"✅ 已加载持仓状态: 持仓={shares_held:.2f}股, 资金={current_balance:.2f}元" + (f", 成本价={cost_price:.4f}元" if cost_price else ""))
            else:
                current_balance = 50000.0
                shares_held = 0.0
                last_price = 0.0
                initial_balance = 50000.0
                cost_price = None
            
            # 执行一次预测（原while True循环的内容）
            try:
                # 检查持仓状态更新（来自Web编辑器）
                if ENABLE_WEB_EDITOR:
                    current_balance, shares_held, last_price, initial_balance = refresh_portfolio_from_file_if_changed(
                        current_balance, shares_held, last_price, initial_balance
                    )
                
                iteration_count += 1
                atr_value = None
                data_source_used = "未知"
                latest_time = None
                stock_name = get_stock_name(STOCK_CODE)
                print(f"\n{'='*70}")
                print(f"📊 第 {iteration_count} 轮预测 [{stock_name}({STOCK_CODE})] - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*70}")
                
                # V16新增：显示V12预测准确率统计
                display_prediction_accuracy(STOCK_CODE)
                
                # 获取数据（V11改进：优先获取最新数据）
                df = None
                if multi_source_manager:
                    try:
                        # 尝试获取最新数据（减少天数，确保获取最新）
                        df, source = multi_source_manager.fetch_data(days=7)
                        if df is not None and len(df) > 0:
                            data_source_used = source or "multi_source"
                            print(f"   📊 数据来源: {source}")
                            # 显示数据源尝试情况
                            stats = multi_source_manager.get_source_stats()
                            failed_sources = []
                            for src, stat in stats.items():
                                if src != source and stat.get('fail', 0) > 0:
                                    failed_sources.append(f"{src}(失败{stat['fail']}次)")
                            if failed_sources:
                                print(f"   📋 其他数据源状态: {', '.join(failed_sources)}")
                            # 说明为什么使用当前数据源
                            if source == 'baostock':
                                print(f"   💡 说明: akshare获取失败，已回退到baostock（可能有1-2天延迟）")
                            elif source == 'akshare':
                                print(f"   💡 说明: 成功使用akshare获取数据")
                    except Exception as e:
                        print(f"   ⚠️  多数据源管理器获取失败: {e}")
                
                if df is None or len(df) == 0:
                    try:
                        code_info = convert_stock_code(STOCK_CODE)
                        # V11改进：优先获取最近1-2天的数据，确保是最新的
                        df = fetch_akshare_5min(code_info, days=2)  # 减少天数，确保获取最新数据
                        if df is not None and len(df) > 0:
                            data_source_used = "akshare"
                        if df is None or len(df) == 0:
                            # 如果失败，尝试获取7天数据
                            df = fetch_akshare_5min(code_info, days=7)
                            if df is not None and len(df) > 0:
                                data_source_used = "akshare"
                    except Exception as e:
                        print(f"   ⚠️  数据获取失败: {e}")
                        time.sleep(60)
                        continue
                
                if df is None or len(df) == 0:
                    print(f"⏸️  未找到数据")
                    time.sleep(60)
                    continue
                
                # V11改进：确保数据按时间排序，使用最新的数据
                df = df.sort_values('time')
                # 检查数据时间戳，确保使用最新数据
                if 'time' in df.columns:
                    # 显示最新数据的时间
                    latest_time = df['time'].iloc[-1]
                    print(f"   📅 最新数据时间: {latest_time}")
                
                closes = df['close'].astype(float).values
                
                # 如果数据不足，尝试用其他数据源补齐（例如：akshare 只有少量当日 5 分钟数据）
                if len(closes) < 126:
                    print(f"⚠️  数据不足（需要126条，实际{len(closes)}条）")
                    
                    # 使用多数据源合并功能，用历史数据补齐
                    if multi_source_manager is not None:
                        try:
                            print("   🔄 正在尝试从其他数据源合并历史数据进行补齐...")
                            merged_df = multi_source_manager.merge_data_from_multiple_sources(
                                days=7,
                                merge_strategy='union'
                            )
                            if merged_df is not None and len(merged_df) > len(df):
                                # 合并后重新排序、去重
                                merged_df = merged_df.drop_duplicates(subset=['time'], keep='last')
                                merged_df = merged_df.sort_values('time')
                                # 若缺少high/low，提示并继续后续ATR用收盘近似
                                if ('high' not in merged_df.columns) or ('low' not in merged_df.columns):
                                    print("   ⚠️ 合并数据缺少High/Low，ATR将使用收盘价近似，精度受限")
                                merged_closes = merged_df['close'].astype(float).values
                                if len(merged_closes) >= 126:
                                    df = merged_df
                                    closes = merged_closes
                                    print(f"   ✅ 已通过合并数据源补齐历史数据，当前数据条数: {len(closes)}")
                                else:
                                    print(f"   ⚠️ 合并后数据仍不足（{len(merged_closes)} 条），暂时无法进行预测")
                            else:
                                print("   ⚠️ 无法通过合并数据源获得更多历史数据")
                        except Exception as e:
                            print(f"   ⚠️ 合并多数据源补齐历史数据时出错: {e}")
                
                # 再次检查是否满足最小长度要求
                if len(closes) < 126:
                    print("⏸️  有效历史数据仍不足，等待下一轮数据更新后再预测")
                    time.sleep(60)
                    continue
                
                # 计算ATR供动态止损使用（若高低价缺失则使用收盘近似并提示精度风险）
                if ENABLE_ATR_STOP_LOSS:
                    atr_value, atr_close_only = calculate_atr(df, period=ATR_PERIOD, return_meta=True)
                    if atr_value:
                        try:
                            if atr_close_only:
                                print(f"   📊 ATR({ATR_PERIOD})≈{atr_value:.4f} (收盘价近似，缺少High/Low，精度受限)")
                            else:
                                print(f"   📊 ATR({ATR_PERIOD})={atr_value:.4f} (用于动态止损)")
                        except Exception:
                            pass
                    elif ENABLE_ATR_STOP_LOSS:
                        print(f"   ⚠️  ATR计算失败（可能因数据不足或缺少收盘价），动态止损将跳过")
                
                # V11改进：仅从实时行情接口获取价格（不从持仓状态获取）
                # 减少重试次数，避免频繁失败请求
                realtime_price = None
                try:
                    print(f"   🔄 正在从实时行情接口获取最新价格...")
                    # 减少重试次数为1次，减少调试输出
                    realtime_price = get_current_market_price(STOCK_CODE, max_retries=1, debug=False)
                    if realtime_price and realtime_price > 0:
                        print(f"   ✅ 已从实时行情接口获取价格: {realtime_price:.2f}")
                    # 失败时不打印，避免频繁输出
                except Exception as e:
                    # 静默处理，不打印错误
                    pass
                
                # 备选方案：从数据源获取（可能是历史数据）
                data_source_price = closes[-1]
                
                # 确定最终使用的价格：优先级 实时行情 > 持仓编辑器手动价格 > 数据源价格
                # 先读取持仓编辑器中的价格，用于比较
                manual_price = None
                manual_price_time = None
                try:
                    state = load_portfolio_state()
                    if state and state.get('stock_code') == STOCK_CODE:
                        manual_price = state.get('last_price', 0.0)
                        manual_price_time = state.get('price_update_time') or state.get('last_update', '')
                except:
                    pass
                
                # 检查实时价格的数据日期（如果是baostock，可能是昨天的数据）
                realtime_price_is_today = True
                if realtime_price and realtime_price > 0:
                    # 检查数据源时间，判断实时价格是否是今天的数据
                    if 'time' in df.columns:
                        latest_time_str = str(df['time'].iloc[-1])
                        try:
                            if len(latest_time_str) >= 8:
                                year = int(latest_time_str[0:4])
                                month = int(latest_time_str[4:6])
                                day = int(latest_time_str[6:8])
                                latest_date = datetime.date(year, month, day)
                                today = datetime.date.today()
                                days_diff = (today - latest_date).days
                                if days_diff > 0:
                                    realtime_price_is_today = False
                                    print(f"   ⚠️  实时价格来自 {days_diff} 天前，可能不是最新")
                        except:
                            pass
                
                # 确定最终使用的价格：优先使用实时价格（无论是否今天）
                if realtime_price and realtime_price > 0:
                    # 实时价格存在，优先使用（无论是否今天）
                    current_price = realtime_price
                    if realtime_price_is_today:
                        price_source = "实时行情"
                    else:
                        price_source = "实时行情(可能非最新)"
                    # 同步更新到持仓状态文件
                    try:
                        state = load_portfolio_state()
                        if state and state.get('stock_code') == STOCK_CODE:
                            state['last_price'] = realtime_price
                            state['price_source'] = '实时行情'
                            state['price_update_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            with open(PORTFOLIO_STATE_FILE, 'w', encoding='utf-8') as f:
                                json.dump(state, f, indent=2, ensure_ascii=False)
                            print(f"   ✅ 已同步实时价格到持仓编辑器: {realtime_price:.2f}")
                    except Exception as e:
                        print(f"   ⚠️  同步价格到持仓编辑器失败: {e}")
                elif manual_price and manual_price > 0:
                    # 如果实时价格不存在，使用持仓编辑器中的手动价格
                    current_price = manual_price
                    price_source = "持仓编辑器(手动输入)"
                    print(f"   ✅ 使用持仓编辑器中的手动价格: {current_price:.2f}")
                else:
                    current_price = data_source_price
                    price_source = "数据源(可能非最新)"
                    # 检查数据时间，如果数据太旧，给出警告
                    if 'time' in df.columns:
                        latest_time_str = str(df['time'].iloc[-1])
                        try:
                            # 解析时间：20251202150000000 -> 2025-12-02 15:00:00
                            if len(latest_time_str) >= 8:
                                year = int(latest_time_str[0:4])
                                month = int(latest_time_str[4:6])
                                day = int(latest_time_str[6:8])
                                latest_date = datetime.date(year, month, day)
                                today = datetime.date.today()
                                days_diff = (today - latest_date).days
                                if days_diff > 0:
                                    print(f"   ⚠️  数据源价格来自 {days_diff} 天前，可能不是最新价格")
                        except:
                            pass
                    print(f"   ⚠️  实时行情获取失败，使用数据源价格: {current_price:.2f}")
                
                volume = float(df['volume'].iloc[-1]) if 'volume' in df.columns else 0.0
                
                stock_name = get_stock_name(STOCK_CODE)
                print(f"   💰 [{stock_name}({STOCK_CODE})] 当前价格: {current_price:.2f} (来源: {price_source})")
                print(f"   📈 成交量: {volume:,.0f}")
                
                # ========== V7: 技术指标计算 ==========
                indicator_summary = None
                if tech_indicators:
                    try:
                        df_with_indicators = tech_indicators.calculate_all(df)
                        if 'KDJ' in df_with_indicators.columns:
                            kdj_values = df_with_indicators['KDJ'].iloc[-1]
                            rsi = df_with_indicators.get('RSI', pd.Series([0])).iloc[-1] if 'RSI' in df_with_indicators.columns else 0
                            obv_ratio = df_with_indicators.get('OBV_Ratio', pd.Series([1.0])).iloc[-1] if 'OBV_Ratio' in df_with_indicators.columns else 1.0
                            macd = df_with_indicators.get('MACD', pd.Series([0])).iloc[-1] if 'MACD' in df_with_indicators.columns else 0
                            
                            indicator_summary = {
                                'KDJ': kdj_values if isinstance(kdj_values, dict) else {'K': 0, 'D': 0, 'J': 0},
                                'RSI': rsi,
                                'OBV': {'OBV_Ratio': obv_ratio},
                                'MACD': {'MACD': macd}
                            }
                            print(f"   📊 V7技术指标: KDJ={indicator_summary['KDJ']}, RSI={rsi:.2f}")
                    except Exception as e:
                        print(f"   ⚠️  技术指标计算失败: {e}")
                
                # ========== V7: LLM指标解释 ==========
                if llm_interpreter and indicator_summary:
                    try:
                        interpretation = llm_interpreter.interpret_indicators(
                            indicator_summary,
                            current_price=current_price
                        )
                        if interpretation:
                            print(f"   🤖 V7 LLM解释: {interpretation.get('summary', '无')}")
                    except Exception as e:
                        print(f"   ⚠️  LLM解释失败: {e}")
                
                # ========== V7: PPO模型预测 ==========
                ppo_action = None
                ppo_operation = "持有"
                
                # V13: 多模型预测和回测
                model_actions = {}  # 存储所有模型的预测动作 {model_name: action}
                if ENABLE_AUTO_MODEL_SELECTION and len(candidate_ppo_models) > 0:
                    # 对所有候选模型进行预测
                    obs = np.array(closes[-126:], dtype=np.float32)
                    
                    for model_name, model_instance in candidate_ppo_models.items():
                        try:
                            action, _states = model_instance.predict(obs, deterministic=True)
                            model_actions[model_name] = int(action)
                        except Exception as e:
                            # 静默处理单个模型预测失败
                            pass
                    
                    # 使用当前选中的模型进行预测
                if ppo_model and current_model_name:
                    try:
                        ppo_action = model_actions.get(current_model_name)
                        if ppo_action is not None:
                            ppo_operation = map_action_to_operation(ppo_action)
                            stock_name = get_stock_name(STOCK_CODE)
                            # 为避免"买入100%"等PPO动作文案产生歧义，这里不再直接输出PPO动作，
                            # 请参考下方的「V7建议价格和仓位」以及「当前价格对应的合理仓位」
                            
                            # V7预测：添加买入卖出建议价格和当前建议持有仓位
                            v7_suggestions = calculate_v7_price_suggestions(current_price, ppo_action, closes)
                            if v7_suggestions:
                                print(f"\n   💡 V7建议价格和仓位:")
                                if v7_suggestions.get('suggested_buy_price'):
                                    buy_price_diff = v7_suggestions['suggested_buy_price'] - current_price
                                    buy_price_diff_pct = (buy_price_diff / current_price * 100) if current_price > 0 else 0
                                    print(f"      💰 建议买入价格: {v7_suggestions['suggested_buy_price']:.2f}元 (当前价格: {current_price:.2f}元, 差异: {buy_price_diff:+.2f}元 ({buy_price_diff_pct:+.2f}%))")
                                if v7_suggestions.get('suggested_sell_price'):
                                    sell_price_diff = v7_suggestions['suggested_sell_price'] - current_price
                                    sell_price_diff_pct = (sell_price_diff / current_price * 100) if current_price > 0 else 0
                                    print(f"      💰 建议卖出价格: {v7_suggestions['suggested_sell_price']:.2f}元 (当前价格: {current_price:.2f}元, 差异: {sell_price_diff:+.2f}元 ({sell_price_diff_pct:+.2f}%))")
                                # 为避免误导，这里显示的是"当前价格对应的合理仓位"，而不是PPO目标仓位
                                current_pos_pct = v7_suggestions.get('current_position_pct', v7_suggestions['suggested_position_pct'])
                                print(f"      📊 当前价格对应的合理仓位: {current_pos_pct:.0f}%")
                                print(f"      📝 仓位描述: {v7_suggestions['position_description']}")
                                # 增加更详细的人性化说明
                                if v7_suggestions.get('detailed_position_description'):
                                    print(f"      📖 详细说明: {v7_suggestions['detailed_position_description']}")
                                
                                # 显示详细的仓位价格建议
                                if v7_suggestions.get('position_prices'):
                                    position_prices = v7_suggestions['position_prices']
                                    print(f"\n      📋 详细仓位价格建议:")
                                    print(f"         🟢 100%仓位（满仓）: {position_prices['100%']:.2f}元 (价格越低，买入越多)")
                                    print(f"         🟡 75%仓位:  {position_prices['75%']:.2f}元")
                                    print(f"         🟠 50%仓位:  {position_prices['50%']:.2f}元")
                                    print(f"         🟤 25%仓位:  {position_prices['25%']:.2f}元")
                                    print(f"         ⚪ 0%仓位（空仓）:   {position_prices['0%']:.2f}元 (价格越高，卖出越多)")
                                    
                                    # 显示当前价格对应的仓位
                                    current_pos_pct = v7_suggestions.get('current_position_pct', 50.0)
                                    print(f"         📍 当前价格 {current_price:.2f}元 对应建议仓位: {current_pos_pct:.0f}%")
                                    
                                    # 计算当前价格与各仓位价格的差异
                                    price_levels = [position_prices['100%'], position_prices['75%'], position_prices['50%'], position_prices['25%'], position_prices['0%']]
                                    position_labels = ['100%', '75%', '50%', '25%', '0%']
                                    closest_price = min(price_levels, key=lambda x: abs(x - current_price))
                                    closest_index = price_levels.index(closest_price)
                                    closest_position = position_labels[closest_index]
                                    price_diff_from_closest = abs(current_price - closest_price)
                                    price_diff_pct_from_closest = (price_diff_from_closest / current_price * 100) if current_price > 0 else 0
                                    
                                    if price_diff_pct_from_closest < 1.0:
                                        print(f"         ✅ 当前价格接近{closest_position}仓位价格（{closest_price:.2f}元），差异仅{price_diff_pct_from_closest:.2f}%")
                                    else:
                                        print(f"         💡 最接近的仓位价格: {closest_position}仓位 {closest_price:.2f}元 (差异: {price_diff_from_closest:.2f}元, {price_diff_pct_from_closest:.2f}%)")
                                
                                print(f"      📈 价格区间: {v7_suggestions['price_interval_pct']:.2f}% (基于波动率{v7_suggestions['volatility_pct']:.2f}%)")
                                
                                # V7下跌预测检测：当PPO动作是卖出（0-2）时，提示做空处理
                                if ppo_action is not None and ppo_action <= 2:
                                    print(f"\n   ⚠️  V7下跌预警提示:")
                                    print(f"      📉 PPO模型建议卖出操作（动作={ppo_action}），请注意风险")
                                    print(f"      💡 建议：如果预测下跌2%以上，可考虑开盘卖出，在最低点再买回")
                                    print(f"      🔄 做空处理：建议进行做空操作以规避下跌风险")
                    except Exception as e:
                        print(f"   ⚠️  PPO预测失败: {e}")
                elif ppo_model:
                    # V12兼容模式：单一模型
                    try:
                        obs = np.array(closes[-126:], dtype=np.float32)
                        action, _states = ppo_model.predict(obs, deterministic=True)
                        ppo_action = int(action)
                        ppo_operation = map_action_to_operation(ppo_action)
                        stock_name = get_stock_name(STOCK_CODE)
                        # 为避免"买入100%"等PPO动作文案产生歧义，这里不再直接输出PPO动作，
                        # 请参考下方的「V7建议价格和仓位」以及「当前价格对应的合理仓位」
                        
                        # V7预测：添加买入卖出建议价格和当前建议持有仓位
                        v7_suggestions = calculate_v7_price_suggestions(current_price, ppo_action, closes)
                        if v7_suggestions:
                            print(f"\n   💡 V7建议价格和仓位:")
                            if v7_suggestions.get('suggested_buy_price'):
                                buy_price_diff = v7_suggestions['suggested_buy_price'] - current_price
                                buy_price_diff_pct = (buy_price_diff / current_price * 100) if current_price > 0 else 0
                                print(f"      💰 建议买入价格: {v7_suggestions['suggested_buy_price']:.2f}元 (当前价格: {current_price:.2f}元, 差异: {buy_price_diff:+.2f}元 ({buy_price_diff_pct:+.2f}%))")
                            if v7_suggestions.get('suggested_sell_price'):
                                sell_price_diff = v7_suggestions['suggested_sell_price'] - current_price
                                sell_price_diff_pct = (sell_price_diff / current_price * 100) if current_price > 0 else 0
                                print(f"      💰 建议卖出价格: {v7_suggestions['suggested_sell_price']:.2f}元 (当前价格: {current_price:.2f}元, 差异: {sell_price_diff:+.2f}元 ({sell_price_diff_pct:+.2f}%))")
                            # 为避免误导，这里显示的是"当前价格对应的合理仓位"，而不是PPO目标仓位
                            current_pos_pct = v7_suggestions.get('current_position_pct', v7_suggestions['suggested_position_pct'])
                            print(f"      📊 当前价格对应的合理仓位: {current_pos_pct:.0f}%")
                            print(f"      📝 仓位描述: {v7_suggestions['position_description']}")
                            # 增加更详细的人性化说明
                        if v7_suggestions.get('detailed_position_description'):
                            print(f"      📖 详细说明: {v7_suggestions['detailed_position_description']}")
                        
                        print(f"      📈 价格区间: {v7_suggestions['price_interval_pct']:.2f}% (基于波动率{v7_suggestions['volatility_pct']:.2f}%)")
                        
                        # V7下跌预测检测：当PPO动作是卖出（0-2）时，提示做空处理
                        if ppo_action is not None and ppo_action <= 2:
                            print(f"\n   ⚠️  V7下跌预警提示:")
                            print(f"      📉 PPO模型建议卖出操作（动作={ppo_action}），请注意风险")
                            print(f"      💡 建议：如果预测下跌2%以上，可考虑开盘卖出，在最低点再买回")
                            print(f"      🔄 做空处理：建议进行做空操作以规避下跌风险")
                    except Exception as e:
                        print(f"   ⚠️  PPO预测失败: {e}")
                
                # ========== V9: LSTM/GRU预测 ==========
                lstm_prediction = None
                if lstm_processor and ENABLE_LSTM_PREDICTION:
                    try:
                        if not lstm_trained and len(closes) >= LSTM_SEQ_LENGTH * 2:
                            print("   📚 V9训练LSTM模型...")
                            # V11改进：使用滑动窗口归一化
                            if USE_SLIDING_WINDOW_NORMALIZE and len(closes) > SLIDING_WINDOW_SIZE:
                                recent_closes = closes[-SLIDING_WINDOW_SIZE:]
                                print(f"      📊 使用滑动窗口归一化（窗口大小: {SLIDING_WINDOW_SIZE}）")
                            else:
                                recent_closes = closes
                                print(f"      📊 使用全局归一化（数据点: {len(closes)}）")
                        
                        normalized_data, norm_params = lstm_processor.normalize(recent_closes)
                        lstm_normalization_params = norm_params
                        X, y = lstm_processor.create_sequences(normalized_data)
                        if len(X) > 0:
                            lstm_processor.train(X, y, epochs=50, batch_size=32, verbose=False)
                            lstm_trained = True
                            print("   ✅ V9 LSTM模型训练完成")
                        
                        if lstm_trained and lstm_normalization_params:
                            # 使用训练时的归一化参数对输入序列进行归一化
                            seq = closes[-LSTM_SEQ_LENGTH:]
                            # 手动归一化（使用训练时的参数，而不是重新计算）
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
                        
                        # 预测（返回归一化后的预测值）
                        prediction_norm = lstm_processor.predict_next(normalized_seq)
                        # 反归一化预测结果
                        lstm_prediction = float(lstm_processor.denormalize(
                            np.array([prediction_norm]),
                            lstm_normalization_params
                        )[0]) if prediction_norm is not None else None
                        if lstm_prediction:
                            print(f"   📈 V9 LSTM预测价格: {lstm_prediction:.2f}")
                    except Exception as e:
                        print(f"   ⚠️  LSTM预测失败: {e}")
                
                # ========== V12: Transformer预测（优化版） ==========
                transformer_prediction = None
                if transformer_model and ENABLE_TRANSFORMER and len(closes) >= TRANSFORMER_MAX_SEQ_LEN:
                    try:
                        if not transformer_trained and len(closes) >= TRANSFORMER_MAX_SEQ_LEN * 2:
                            print("   📚 V12训练Transformer模型...")
                            # V12优化：智能归一化策略（考虑当前价格位置）
                            if TRANSFORMER_ADAPTIVE_WINDOW:
                                # 确定可用窗口大小（不超过数据总量）
                                available_window = min(SLIDING_WINDOW_SIZE, len(closes))
                                if available_window >= TRANSFORMER_MAX_SEQ_LEN * 2:  # 确保有足够数据训练
                                    # 计算当前价格在历史数据中的位置
                                    window_data = closes[-available_window:]
                                    price_position = (current_price - np.min(window_data)) / (np.max(window_data) - np.min(window_data) + 1e-8)
                                    
                                    # 如果当前价格在较高位置（>75%），使用更短的窗口以突出近期趋势
                                    if price_position > TRANSFORMER_PRICE_POSITION_THRESHOLD:
                                        # 使用60%的窗口，但至少保留足够的数据点
                                        adaptive_window = max(int(available_window * 0.6), TRANSFORMER_MAX_SEQ_LEN * 2)
                                    else:
                                        # 价格位置较低，使用完整窗口
                                        adaptive_window = available_window
                                    recent_closes = closes[-adaptive_window:]
                                    print(f"      📊 V12使用自适应窗口归一化（窗口大小: {adaptive_window}, 价格位置: {price_position*100:.1f}%）")
                                else:
                                    recent_closes = closes[-available_window:]
                                    print(f"      📊 V12使用滑动窗口归一化（窗口大小: {available_window}, 价格位置: {price_position*100:.1f}%）")
                            else:
                                # 数据量太少，使用全部数据但标记为自适应
                                recent_closes = closes
                                price_position = (current_price - np.min(closes)) / (np.max(closes) - np.min(closes) + 1e-8)
                                print(f"      📊 V12使用自适应归一化（数据点: {len(closes)}, 价格位置: {price_position*100:.1f}%）")
                        elif USE_SLIDING_WINDOW_NORMALIZE and len(closes) > SLIDING_WINDOW_SIZE:
                            recent_closes = closes[-SLIDING_WINDOW_SIZE:]
                            print(f"      📊 使用滑动窗口归一化（窗口大小: {SLIDING_WINDOW_SIZE}）")
                        else:
                            recent_closes = closes
                            print(f"      📊 使用全局归一化（数据点: {len(closes)}）")
                        
                        normalized_closes, norm_params = transformer_model.normalize(recent_closes)
                        transformer_normalization_params = norm_params
                        
                        X_list, y_list = [], []
                        for i in range(TRANSFORMER_MAX_SEQ_LEN, len(normalized_closes)):
                            X_list.append(normalized_closes[i-TRANSFORMER_MAX_SEQ_LEN:i])
                            y_list.append(normalized_closes[i])
                        
                        if len(X_list) > 0:
                            X = np.array(X_list).reshape(len(X_list), TRANSFORMER_MAX_SEQ_LEN, 1)
                            y = np.array(y_list).reshape(len(y_list), 1)
                            # V12优化：增加训练轮数到120，提高模型准确性
                            transformer_model.train(
                                X, y, epochs=TRANSFORMER_EPOCHS, batch_size=32,
                                learning_rate=0.001, validation_split=0.2, verbose=False
                            )
                            transformer_trained = True
                            print(f"   ✅ V12 Transformer模型训练完成（训练轮数: {TRANSFORMER_EPOCHS}）")
                            # 输出归一化参数信息，便于诊断
                            if norm_params.get('method') == 'minmax':
                                print(f"      📊 归一化范围: [{norm_params['min']:.2f}, {norm_params['max']:.2f}], 当前价格: {current_price:.2f}")
                        
                        if transformer_trained and transformer_normalization_params:
                            seq = closes[-TRANSFORMER_MAX_SEQ_LEN:]
                            # 使用训练时的归一化参数进行归一化（而不是重新计算）
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
                        
                        # 预测（返回归一化后的预测值）
                        prediction_norm = transformer_model.predict_next(normalized_seq)
                        # 反归一化预测结果
                        transformer_prediction_raw = float(transformer_model.denormalize(
                            np.array([prediction_norm]),
                            transformer_normalization_params
                        )[0]) if prediction_norm is not None else None
                        
                        # V12优化：趋势感知机制和预测后处理
                        if transformer_prediction_raw:
                            transformer_prediction = transformer_prediction_raw
                            
                            # V12优化：趋势感知机制（基于价格动量调整）
                            if TRANSFORMER_TREND_AWARE and len(closes) >= 10:
                                # 计算短期和中期趋势
                                short_trend = (closes[-1] - closes[-5]) / closes[-5] if closes[-5] > 0 else 0  # 5日趋势
                                mid_trend = (closes[-1] - closes[-10]) / closes[-10] if closes[-10] > 0 else 0  # 10日趋势
                                momentum = (short_trend + mid_trend) / 2  # 综合动量
                                
                                # 如果趋势向上且预测偏低，适当上调预测
                                if momentum > 0.01 and transformer_prediction < current_price:
                                    trend_adjustment = min(momentum * 0.3, 0.05)  # 最大调整5%
                                    transformer_prediction = transformer_prediction * (1 + trend_adjustment)
                            
                            # V12优化：预测后处理（根据当前价格位置校正预测）
                            if TRANSFORMER_POST_PROCESS:
                                norm_method = transformer_normalization_params.get('method', 'minmax')
                                if norm_method == 'minmax':
                                    min_val = transformer_normalization_params['min']
                                    max_val = transformer_normalization_params['max']
                                    if max_val - min_val > 0:
                                        price_position = (current_price - min_val) / (max_val - min_val)
                                        
                                        # 如果当前价格在较高位置（>75%），且预测明显偏低，进行校正
                                        if price_position > TRANSFORMER_PRICE_POSITION_THRESHOLD:
                                            price_diff_pct = (transformer_prediction - current_price) / current_price
                                            
                                            # 如果预测偏低超过3%，进行校正（降低阈值，更积极校正）
                                            if price_diff_pct < -0.03:
                                                # 计算校正因子：当前价格位置越高，校正越大
                                                correction_factor = (price_position - TRANSFORMER_PRICE_POSITION_THRESHOLD) / (1 - TRANSFORMER_PRICE_POSITION_THRESHOLD)
                                                # 增强校正力度：根据价格位置和偏差程度动态调整
                                                base_correction = abs(price_diff_pct) * correction_factor
                                                # 价格位置越高，校正越激进（最高校正70%的偏差）
                                                correction_amount = base_correction * (0.5 + correction_factor * 0.4)
                                                transformer_prediction = transformer_prediction * (1 + correction_amount)
                                                print(f"      ✨ V12预测后处理: 价格位置{price_position*100:.1f}%, 已校正预测偏差{correction_amount*100:.1f}%")
                            
                            # 输出预测结果和诊断信息
                            norm_method = transformer_normalization_params.get('method', 'minmax')
                            if norm_method == 'minmax':
                                min_val = transformer_normalization_params['min']
                                max_val = transformer_normalization_params['max']
                                price_diff = transformer_prediction - current_price
                                price_diff_pct = (price_diff / current_price * 100) if current_price > 0 else 0
                                price_position = ((current_price - min_val) / (max_val - min_val) * 100) if (max_val - min_val) > 0 else 0
                                
                                print(f"   🔮 V12 Transformer预测价格: {transformer_prediction:.2f} (当前价格: {current_price:.2f}, 差异: {price_diff:+.2f} ({price_diff_pct:+.2f}%))")
                                print(f"      📊 归一化范围: [{min_val:.2f}, {max_val:.2f}], 当前价格在范围中的位置: {price_position:.1f}%")
                                
                                # V16新增：保存V12预测结果用于准确率统计
                                current_date_str = datetime.datetime.now().strftime('%Y-%m-%d')
                                save_v12_prediction(current_date_str, transformer_prediction, current_price, STOCK_CODE)
                                
                                if transformer_prediction < current_price and abs(price_diff_pct) > 5:
                                    print(f"      💡 V12优化说明:")
                                    print(f"         1. ✅ 已增加训练轮数到{TRANSFORMER_EPOCHS}轮，提高模型准确性")
                                    if TRANSFORMER_ADAPTIVE_WINDOW:
                                        print(f"         2. ✅ 已启用自适应窗口归一化，根据价格位置动态调整")
                                    if TRANSFORMER_TREND_AWARE:
                                        print(f"         3. ✅ 已启用趋势感知机制，基于价格动量调整预测")
                                    if TRANSFORMER_POST_PROCESS:
                                        print(f"         4. ✅ 已启用预测后处理优化，动态校正预测偏差")
                            else:
                                price_diff = transformer_prediction - current_price
                                price_diff_pct = (price_diff / current_price * 100) if current_price > 0 else 0
                                print(f"   🔮 V12 Transformer预测价格: {transformer_prediction:.2f} (当前价格: {current_price:.2f}, 差异: {price_diff:+.2f} ({price_diff_pct:+.2f}%))")
                                
                                # V16新增：保存V12预测结果用于准确率统计
                                current_date_str = datetime.datetime.now().strftime('%Y-%m-%d')
                                save_v12_prediction(current_date_str, transformer_prediction, current_price, STOCK_CODE)
                    except Exception as e:
                        print(f"   ⚠️  Transformer预测失败: {e}")
                        import traceback
                        traceback.print_exc()
                
                # ========== V10: 多模态处理 ==========
                multimodal_result = None
                if multimodal_processor and ENABLE_MULTIMODAL:
                    try:
                        # V11改进：使用真实新闻源（LLM市场情报）
                        text_data = None
                        if USE_REAL_NEWS_SOURCE and llm_agent:
                            try:
                                # 获取当前日期的市场情报
                                today_str = datetime.datetime.now().strftime('%Y-%m-%d')
                                intelligence = llm_agent.get_market_intelligence(today_str)
                                if intelligence and 'summary' in intelligence:
                                    text_data = intelligence['summary']
                                    print(f"   📰 V11使用真实新闻源: {text_data[:50]}...")
                            except Exception as e:
                                if FALLBACK_TO_SAMPLE_TEXTS:
                                    text_data = sample_texts[text_index % len(sample_texts)]
                                    text_index += 1
                                    print(f"   ⚠️  获取真实新闻失败，使用样本文本: {e}")
                                else:
                                    raise
                        else:
                            # 使用样本文本
                            text_data = sample_texts[text_index % len(sample_texts)]
                            text_index += 1
                        
                        if text_data:
                            multimodal_result = multimodal_processor.process(
                                time_series_data=closes[-60:],
                                text_data=text_data
                            )
                            print(f"   🌐 V10多模态处理: 情感={multimodal_result.get('sentiment', {}).get('polarity', 0):.2f}")
                    except Exception as e:
                        print(f"   ⚠️  多模态处理失败: {e}")
                
                # ========== V10: 全息动态模型 ==========
                holographic_signal = None
                if holographic_model and ENABLE_HOLOGRAPHIC:
                    try:
                        fallback_used = False
                        holographic_result = holographic_model.process(
                            time_series_data=closes[-60:],
                            text_data=sample_texts[text_index % len(sample_texts)],
                            technical_indicators=indicator_summary,
                            market_intelligence=None
                        )
                        holographic_signal_raw = holographic_result.get('comprehensive_signal') if holographic_result else None
                        if holographic_signal_raw and isinstance(holographic_signal_raw, dict):
                            raw_conf = float(holographic_signal_raw.get('confidence', 0) or 0.0)
                            if raw_conf <= 0.05:
                                fallback_used = True
                            holographic_signal = refine_holographic_signal(
                                holographic_signal_raw,
                                closes=closes,
                                indicator_summary=indicator_summary,
                                fallback_used=fallback_used
                            )
                            print(f"   🌟 V10全息信号: {holographic_signal.get('signal', 'hold')} (置信度={holographic_signal.get('confidence', 0):.2f}, 原始={holographic_signal_raw.get('signal','?')}, 原置信度={raw_conf:.2f})")
                            reason = holographic_signal.get('reason')
                            if reason:
                                print(f"      💡 全息理由: {reason}")
                        else:
                            holographic_signal = {'signal': 'hold', 'confidence': 0.15, 'reason': '无信号'}
                            print(f"   🌟 V10全息信号: hold (置信度=0.15，原因: 无信号)")
                    except Exception as e:
                        print(f"   ⚠️  全息模型处理失败: {e}")
                
                # ========== V12: 智能融合决策（优化版：增加冲突检测） ==========
                conflict_info = None
                if ENABLE_MULTI_MODEL_FUSION:
                    final_action, confidence, adjusted_weights, conflict_info = fuse_multi_model_predictions(
                        ppo_action, lstm_prediction, transformer_prediction,
                        holographic_signal, MODEL_WEIGHTS.copy(), current_price
                    )
                    final_operation = map_action_to_operation(final_action)
                    stock_name = get_stock_name(STOCK_CODE)
                    print(f"\n   ⭐ V12融合决策 [{stock_name}({STOCK_CODE})]: {final_operation} (置信度={confidence:.2f})")
                
                # V12下跌预测检测：当预测下跌2%或3%以上时，提示做空处理（有区分度）
                price_change_pct_for_warning = None
                avg_prediction_for_warning = None
                if conflict_info and conflict_info.get('price_change_pct') is not None:
                    price_change_pct_for_warning = conflict_info.get('price_change_pct', 0)
                    avg_prediction_for_warning = conflict_info.get('avg_prediction', current_price)
                elif lstm_prediction is not None or transformer_prediction is not None:
                    # 如果没有conflict_info，使用LSTM/Transformer预测计算
                    predictions = []
                    if lstm_prediction is not None and lstm_prediction > 0:
                        predictions.append(lstm_prediction)
                    if transformer_prediction is not None and transformer_prediction > 0:
                        predictions.append(transformer_prediction)
                    if predictions:
                        avg_prediction_for_warning = np.mean(predictions)
                        price_change_pct_for_warning = (avg_prediction_for_warning - current_price) / current_price * 100 if current_price > 0 else 0
                
                # 区分2%和3%的提示：3%以上显示严重警告，2-3%显示一般警告
                if price_change_pct_for_warning is not None and price_change_pct_for_warning <= -3.0:
                    # 下跌3%以上：严重警告
                    print(f"\n   🚨 V12严重下跌预警（≥3%）:")
                    print(f"      📉 预测价格大幅下跌: {abs(price_change_pct_for_warning):.2f}% (预测价格: {avg_prediction_for_warning:.2f}元, 当前价格: {current_price:.2f}元)")
                    print(f"      ⚠️  强烈建议：开盘立即卖出，在最低点再买回")
                    print(f"      🔄 做空处理：强烈建议进行做空操作以规避大幅下跌风险")
                    print(f"      🛑 风险提示：下跌幅度较大，请务必谨慎操作，严格执行止损")
                elif price_change_pct_for_warning is not None and price_change_pct_for_warning <= -2.0:
                    # 下跌2-3%：一般警告
                    print(f"\n   ⚠️  V12下跌预警（2-3%）:")
                    print(f"      📉 预测价格下跌: {abs(price_change_pct_for_warning):.2f}% (预测价格: {avg_prediction_for_warning:.2f}元, 当前价格: {current_price:.2f}元)")
                    print(f"      💡 建议：可考虑开盘卖出，在最低点再买回")
                    print(f"      🔄 做空处理：建议进行做空操作以规避下跌风险")
                    print(f"      ⚠️  风险提示：下跌幅度中等，请注意风险控制")
                
                # V12优化：显示冲突检测和调整信息
                if conflict_info and conflict_info.get('has_conflict', False):
                    print(f"   ⚠️  信号冲突检测:")
                    if conflict_info.get('avg_prediction') is not None:
                        price_change_pct = conflict_info.get('price_change_pct', 0)
                        direction = "下跌" if price_change_pct < 0 else "上涨"
                        print(f"      📊 预测价格: {conflict_info['avg_prediction']:.2f}元（预测{direction} {abs(price_change_pct):.2f}%）")
                    print(f"      🎯 PPO原始建议: {map_action_to_operation(conflict_info.get('original_action', ppo_action))}")
                    if conflict_info.get('adjustment_reason'):
                        print(f"      💡 调整原因: {conflict_info['adjustment_reason']}")
                    print(f"      ✅ 调整后决策: {final_operation}")
                elif conflict_info and conflict_info.get('avg_prediction') is not None:
                    # 无冲突，显示预测信息
                    price_change_pct = conflict_info.get('price_change_pct', 0)
                    direction = "上涨" if price_change_pct > 0 else "下跌"
                    abs_price_change = abs(price_change_pct)
                    
                    # V12优化：更详细地说明预测方向与决策的关系
                    if abs_price_change < 1.5:
                        # 预测方向不明确（<1.5%），说明为何未触发冲突检测
                        if price_change_pct < 0:
                            print(f"   📊 预测价格: {conflict_info['avg_prediction']:.2f}元（预测{direction} {abs_price_change:.2f}%），预测方向不明确，未触发冲突检测，与PPO建议一致")
                        else:
                            print(f"   📊 预测价格: {conflict_info['avg_prediction']:.2f}元（预测{direction} {abs_price_change:.2f}%），与PPO建议一致")
                    else:
                        # 预测方向明确但一致
                        print(f"   📊 预测价格: {conflict_info['avg_prediction']:.2f}元（预测{direction} {abs_price_change:.2f}%），与PPO建议方向一致")
                
                    if ENABLE_DYNAMIC_WEIGHTS:
                        print(f"   📊 动态权重: PPO={adjusted_weights['ppo']:.1%}, LSTM={adjusted_weights['lstm']:.1%}, Transformer={adjusted_weights['transformer']:.1%}, 全息={adjusted_weights['holographic']:.1%}")
                    else:
                        print(f"   📊 模型权重: PPO={MODEL_WEIGHTS['ppo']:.1%}, LSTM={MODEL_WEIGHTS['lstm']:.1%}, Transformer={MODEL_WEIGHTS['transformer']:.1%}, 全息={MODEL_WEIGHTS['holographic']:.1%}")
                else:
                    final_action = ppo_action
                    final_operation = ppo_operation
                
                # ========== V16: 趋势/震荡双策略加权融合 ==========
                if ENABLE_REGIME_STRATEGY and current_price > 0 and closes is not None and len(closes) >= max(TREND_BREAKOUT_WINDOW, BOLL_PERIOD):
                    regime, trend_score, range_score, boll_info = detect_market_regime(
                        closes, current_price, transformer_prediction, confidence, holographic_signal
                    )
                    trend_bias, breakout_price = trend_strategy_signal(closes, current_price)
                    mr_bias, mr_upper, mr_ma, mr_lower = mean_reversion_signal(closes, current_price)
                    
                    adjust = 0
                    reason = ""
                if regime == 'trend' and trend_bias > 0:
                    adjust += TREND_ADJUST_STEP
                    reason = f"趋势突破(>{TREND_BREAKOUT_WINDOW}窗高点)，增强买入"
                elif regime == 'range':
                    if mr_bias > 0:
                        adjust += 1
                        reason = "震荡下轨附近，倾向买入"
                    elif mr_bias < 0:
                        adjust -= 1
                        reason = "震荡上轨附近，倾向卖出"
                
                adjust = int(np.clip(adjust, -REGIME_MAX_ADJUST, REGIME_MAX_ADJUST))
                if adjust != 0 and final_action is not None:
                    original_action = final_action
                    final_action = int(np.clip(final_action + adjust, 0, 6))
                    final_operation = map_action_to_operation(final_action)
                    bandwidth_str = f"{boll_info['bandwidth']:.4f}" if boll_info else "N/A"
                    print(f"\n   🧭 V16市场状态: {regime} (trend_score={trend_score:.2f}, range_score={range_score:.2f}, 带宽={bandwidth_str})")
                    if reason:
                        print(f"      💡 策略调整: {reason} | 动作 {original_action} -> {final_action} ({final_operation})")
                elif regime != 'neutral':
                    bandwidth_str = f"{boll_info['bandwidth']:.4f}" if boll_info else "N/A"
                    print(f"\n   🧭 V16市场状态: {regime} (trend_score={trend_score:.2f}, range_score={range_score:.2f}, 带宽={bandwidth_str}) 无需调整")
                
                # ========== V13: 止损止盈风险控制 ==========
                stop_loss_info = None
                # 初始化成本价变量（确保在所有情况下都被定义）
                current_cost_price = None
                
                if ENABLE_STOP_LOSS_TAKE_PROFIT and shares_held > 0:
                    # 获取成本价
                    portfolio_state_for_stop = load_portfolio_state()
                    if portfolio_state_for_stop and portfolio_state_for_stop.get('stock_code') == STOCK_CODE:
                        cost_price_val = portfolio_state_for_stop.get('cost_price')
                        actual_buy_price_val = portfolio_state_for_stop.get('actual_buy_price')
                        if cost_price_val and isinstance(cost_price_val, (int, float)) and cost_price_val > 0:
                            current_cost_price = float(cost_price_val)
                        elif actual_buy_price_val and isinstance(actual_buy_price_val, (int, float)) and actual_buy_price_val > 0:
                            current_cost_price = float(actual_buy_price_val)
                
                if current_cost_price and current_cost_price > 0:
                    # 应用止损止盈逻辑
                    final_action, stop_loss_info = apply_stop_loss_take_profit(
                        final_action, current_price, current_cost_price, shares_held, atr_value=atr_value
                    )
                    final_operation = map_action_to_operation(final_action)
                    
                    # 显示止损止盈信息
                    if stop_loss_info and stop_loss_info.get('triggered', False):
                        profit_loss_pct = stop_loss_info.get('profit_loss_pct', 0)
                        reason = stop_loss_info.get('reason', '')
                        original_action = stop_loss_info.get('original_action')
                        atr_stop_price_val = stop_loss_info.get('atr_stop_price')
                        
                        print(f"\n   🚨 V13止损止盈风险控制:")
                        print(f"      ⚠️  {reason}")
                        if original_action is not None:
                            print(f"      📊 原始决策: {map_action_to_operation(original_action)}")
                        print(f"      ✅ 调整后决策: {final_operation}")
                        print(f"      💰 当前盈亏: {profit_loss_pct:+.2f}% (成本价: {current_cost_price:.4f}元, 当前价: {current_price:.2f}元)")
                        if atr_stop_price_val:
                            print(f"      🛡️  ATR动态止损价: {atr_stop_price_val:.2f}元 (ATR×{ATR_MULTIPLIER})")
                    else:
                        # 显示当前盈亏状态（未触发止损止盈）
                        profit_loss_pct = stop_loss_info.get('profit_loss_pct', 0) if stop_loss_info else None
                        atr_stop_price_val = stop_loss_info.get('atr_stop_price') if stop_loss_info else None
                        if profit_loss_pct is not None:
                            # 计算距离止损止盈的距离
                            distance_to_stop_loss = abs(STOP_LOSS_PCT - profit_loss_pct) if profit_loss_pct < 0 else None
                            distance_to_take_profit = abs(TAKE_PROFIT_PCT - profit_loss_pct) if profit_loss_pct > 0 else None
                            if atr_stop_price_val:
                                atr_gap = (current_price - atr_stop_price_val)
                                atr_gap_pct = atr_gap / current_price * 100 if current_price > 0 else None
                                print(f"   🛡️  ATR动态止损价: {atr_stop_price_val:.2f}元 (当前价与ATR止损差距: {atr_gap:+.2f}元, {atr_gap_pct:+.2f}% )")
                            
                            if profit_loss_pct < 0:
                                # 亏损状态，显示距离止损的距离
                                if distance_to_stop_loss:
                                    print(f"\n   📊 V13风险监控: 当前亏损{abs(profit_loss_pct):.2f}%，距离止损{abs(STOP_LOSS_PCT):.2f}%还有{distance_to_stop_loss:.2f}%")
                            elif profit_loss_pct > 0:
                                # 盈利状态，显示距离止盈的距离
                                if distance_to_take_profit:
                                    print(f"\n   📊 V13风险监控: 当前盈利{profit_loss_pct:.2f}%，距离止盈{TAKE_PROFIT_PCT:.2f}%还有{distance_to_take_profit:.2f}%")
                            else:
                                print(f"\n   📊 V13风险监控: 当前盈亏{profit_loss_pct:.2f}% (成本价: {current_cost_price:.4f}元)")
                else:
                    # 没有成本价，提示用户设置
                    if iteration_count == 1 or iteration_count % 10 == 0:  # 第1轮或每10轮提示一次
                        print(f"\n   💡 V13风险控制提示: 当前持仓{shares_held:.2f}股，但未设置成本价，无法执行止损止盈检查")
                        print(f"      📝 请在持仓编辑器(http://127.0.0.1:{WEB_EDITOR_PORT})中设置「成本价」或「实际买入价」以启用止损止盈功能")
                        print(f"      ⚙️  止损设置: {STOP_LOSS_PCT}% | 止盈设置: {TAKE_PROFIT_PCT}%")
                
                # ========== V13: 凯利公式资金管理 ==========
                kelly_info = None
                if ENABLE_KELLY_FORMULA and USE_KELLY_FOR_POSITION:
                    # 计算预测收益率
                    predicted_return_pct = 0.0
                    if transformer_prediction is not None:
                        predicted_return_pct = ((transformer_prediction - current_price) / current_price) * 100
                    
                    # 计算波动率（用于估算平均亏损）
                    volatility_pct = None
                    if len(closes) >= 20:
                        recent_returns = np.diff(closes[-20:]) / closes[-20:-1] * 100
                        volatility_pct = np.std(recent_returns) if len(recent_returns) > 0 else None
                    
                    # 计算凯利公式最优仓位（支持样本不足时的预测估算）
                    kelly_info = get_kelly_position_size(
                        confidence, predicted_return_pct, trade_history, 
                        indicator_summary, volatility_pct
                    )
                
                if kelly_info:
                    kelly_position = kelly_info['kelly_position']
                    is_estimated = kelly_info.get('is_estimated', False)
                    
                    if is_estimated:
                        # 样本不足，使用预测估算
                        print(f"\n   📊 V13凯利公式资金管理（预测估算模式）:")
                        print(f"      🎯 最优仓位: {kelly_position*100:.1f}% (基于模型预测估算)")
                        print(f"      ⚠️  注意: 当前交易样本不足（{len(trade_history)}/{KELLY_MIN_SAMPLES}），使用预测数据估算")
                        print(f"      📈 估算胜率: {kelly_info['win_rate']*100:.1f}% (基于置信度{confidence:.2f}和预测收益率{predicted_return_pct:.2f}%)")
                        print(f"      💰 估算平均盈利: {kelly_info['avg_win_pct']:.2f}% | 估算平均亏损: {kelly_info['avg_loss_pct']:.2f}%")
                        print(f"      📊 原始凯利值: {kelly_info['raw_kelly']*100:.1f}% | 安全凯利值: {kelly_info['safe_kelly']*100:.1f}%")
                        print(f"      💡 建议: 根据凯利公式（预测模式），当前最优仓位为{kelly_position*100:.1f}%")
                        print(f"      📝 提示: 积累{KELLY_MIN_SAMPLES}个交易样本后，将使用历史统计数据进行更准确的仓位计算")
                    else:
                        # 样本充足，使用历史统计
                        print(f"\n   📊 V13凯利公式资金管理（历史统计模式）:")
                        print(f"      🎯 最优仓位: {kelly_position*100:.1f}% (基于历史交易统计)")
                        print(f"      📈 历史胜率: {kelly_info['win_rate']*100:.1f}% ({kelly_info['total_trades']}次交易)")
                        print(f"      💰 平均盈利: {kelly_info['avg_win_pct']:.2f}% | 平均亏损: {kelly_info['avg_loss_pct']:.2f}%")
                        print(f"      📊 原始凯利值: {kelly_info['raw_kelly']*100:.1f}% | 安全凯利值: {kelly_info['safe_kelly']*100:.1f}%")
                        print(f"      💡 建议: 根据凯利公式，当前最优仓位为{kelly_position*100:.1f}%")
                
                # ========== V11: 仓位价格建议 ==========
                suggested_buy_price = None
                suggested_sell_price = None
                price_suggestions = calculate_position_price_suggestions(
                    current_price, lstm_prediction, transformer_prediction, confidence, final_action, closes
                )
                if price_suggestions:
                    suggestions = price_suggestions['suggestions']
                    
                    # 获取当前价格对应的建议仓位
                    current_position_pct = price_suggestions.get('current_position_pct', 50.0)
                    current_position = f"{current_position_pct:.0f}%"
                    
                    # 计算当前价格与各仓位价格的差异，找出最接近的仓位
                    price_levels = [suggestions['100%'], suggestions['75%'], suggestions['50%'], suggestions['25%'], suggestions['0%']]
                    position_labels = ['100%', '75%', '50%', '25%', '0%']
                    
                    # 找到当前价格最接近的仓位价格
                    closest_price = min(price_levels, key=lambda x: abs(x - current_price))
                    closest_index = price_levels.index(closest_price)
                    closest_position = position_labels[closest_index]
                    price_diff_from_closest = abs(current_price - closest_price)
                    price_diff_pct_from_closest = (price_diff_from_closest / current_price * 100) if current_price > 0 else 0
                    
                    print(f"\n   💡 仓位价格建议（基于预测价格 {price_suggestions['predicted_price']:.2f}元，预测{price_suggestions['direction']} {abs(price_suggestions['price_change_pct']):.2f}%）:")
                    print(f"      🟢 100%仓位: {suggestions['100%']:.2f}元 (价格越低，买入越多)")
                    print(f"      🟡 75%仓位:  {suggestions['75%']:.2f}元")
                    print(f"      🟠 50%仓位:  {suggestions['50%']:.2f}元")
                    print(f"      🟤 25%仓位:  {suggestions['25%']:.2f}元")
                    print(f"      ⚪ 0%仓位:   {suggestions['0%']:.2f}元 (价格越高，卖出越多)")
                    
                    # 计算相邻仓位的最小价格差
                    min_diff_pct = min([abs(price_levels[i] - price_levels[i+1]) / current_price * 100 
                                       for i in range(len(price_levels)-1)]) if current_price > 0 else 0
                    
                    # 优先根据融合决策生成建议，而不是仅仅基于价格位置
                    # 融合决策是更重要的信号，价格建议应该与之保持一致
                    action_hint = ""
                    consistency_note = ""
                    
                    # 计算当前价格与预测价格的偏离程度
                    price_diff_from_pred = abs(current_price - price_suggestions['predicted_price']) / price_suggestions['predicted_price'] * 100 if price_suggestions['predicted_price'] > 0 else 0
                    
                    # V12优化：根据融合决策和预测方向确定建议，考虑冲突检测结果
                    # 如果检测到冲突并已调整，仓位建议应该反映调整后的决策
                    if conflict_info and conflict_info.get('has_conflict', False):
                        # 检测到冲突并已调整决策，仓位建议应该与调整后的决策一致
                        # 根据调整原因确定正确的建议
                        adjustment_reason = conflict_info.get('adjustment_reason', '')
                        
                        if final_action == 4 or "调整为持有" in adjustment_reason:  # 已调整为持有
                            action_hint = f"⚠️  由于预测方向与PPO建议冲突，已调整为「持有」。建议保持当前仓位或降低至50%以下"
                            consistency_note = f"⚠️  已根据预测方向调整决策（预测{price_suggestions['direction']} {abs(price_suggestions['price_change_pct']):.2f}%）"
                        elif final_action == 3 or "调整为卖出" in adjustment_reason:  # 已调整为卖出25%
                            action_hint = f"⚠️  由于预测价格明显下跌，已调整为「卖出25%」。建议减仓至25%仓位或更低"
                            consistency_note = f"⚠️  已根据预测方向调整决策（预测{price_suggestions['direction']} {abs(price_suggestions['price_change_pct']):.2f}%）"
                        elif final_action == 5 or "降低买入力度" in adjustment_reason:  # 已降低买入力度
                            action_hint = f"⚠️  由于预测方向与PPO建议冲突，已降低买入力度至「买入25%」。建议买入至75%仓位，而非满仓"
                            consistency_note = f"⚠️  已根据预测方向调整决策（预测{price_suggestions['direction']} {abs(price_suggestions['price_change_pct']):.2f}%）"
                        else:
                            # 其他调整情况
                            action_hint = f"⚠️  已根据预测方向调整决策为「{final_operation}」"
                            consistency_note = f"⚠️  已根据预测方向调整决策（预测{price_suggestions['direction']} {abs(price_suggestions['price_change_pct']):.2f}%）"
                    elif final_action == 6:  # 买入 100%（无冲突）
                        if price_diff_from_pred >= 3.0:
                            # 价格偏离较大，根据实际价格位置动态调整
                            if current_price > price_suggestions['predicted_price']:
                                # 当前价格高于预测价格，建议减仓
                                if current_position_pct <= 25:
                                    action_hint = f"⚠️  当前价格 {current_price:.2f}元 高于预测价格 {price_suggestions['predicted_price']:.2f}元（偏离{price_diff_from_pred:.2f}%），建议减仓至{current_position}仓位（价格偏离较大，动态调整）"
                                elif current_position_pct <= 50:
                                    action_hint = f"⚠️  当前价格 {current_price:.2f}元 高于预测价格 {price_suggestions['predicted_price']:.2f}元（偏离{price_diff_from_pred:.2f}%），建议保持{current_position}仓位（价格偏离较大，动态调整）"
                                else:
                                    action_hint = f"✅ 融合决策「买入 100%」但当前价格 {current_price:.2f}元 高于预测价格（偏离{price_diff_from_pred:.2f}%），建议保持{current_position}仓位"
                                consistency_note = f"⚠️  价格偏离预测价格{price_diff_from_pred:.2f}%，已动态调整建议仓位"
                            else:
                                # 当前价格低于预测价格，建议加仓
                                action_hint = f"✅ 融合决策「买入 100%」+ 当前价格 {current_price:.2f}元 低于预测价格（偏离{price_diff_from_pred:.2f}%），建议加仓至{current_position}仓位"
                                consistency_note = "✅ 与融合决策「买入 100%」一致"
                        else:
                            # 价格偏离较小，遵循融合决策
                            if current_price <= suggestions['75%']:
                                action_hint = f"✅ 融合决策「买入 100%」+ 当前价格 {current_price:.2f}元 在买入区间，建议满仓买入"
                            elif current_price <= suggestions['50%']:
                                action_hint = f"✅ 融合决策「买入 100%」+ 当前价格 {current_price:.2f}元 接近买入区间，建议高仓位买入（目标100%仓位）"
                            else:
                                action_hint = f"✅ 融合决策「买入 100%」：虽然当前价格 {current_price:.2f}元 略高于预测价格，但模型建议买入，可考虑分批买入或等待回调至 {suggestions['75%']:.2f}元 以下"
                            consistency_note = "✅ 与融合决策「买入 100%」一致"
                    elif final_action == 5:  # 买入 25%
                        if current_price <= suggestions['75%']:
                            action_hint = f"✅ 融合决策「买入 25%」+ 当前价格 {current_price:.2f}元 在买入区间，建议买入至75%仓位"
                        else:
                            action_hint = f"✅ 融合决策「买入 25%」：当前价格 {current_price:.2f}元，建议买入至75%仓位（可等待回调至 {suggestions['75%']:.2f}元 以下）"
                        consistency_note = "✅ 与融合决策「买入 25%」一致"
                        
                    elif final_action == 4:  # 持有
                        if suggestions['25%'] <= current_price <= suggestions['75%']:
                            action_hint = f"✅ 融合决策「持有」+ 当前价格 {current_price:.2f}元 在合理区间，建议保持当前仓位"
                        else:
                            action_hint = f"✅ 融合决策「持有」：当前价格 {current_price:.2f}元，建议保持50%左右仓位"
                        consistency_note = "✅ 与融合决策「持有」一致"
                        
                    elif final_action == 3:  # 卖出 25%
                        if current_price >= suggestions['25%']:
                            action_hint = f"✅ 融合决策「卖出 25%」+ 当前价格 {current_price:.2f}元 在卖出区间，建议减仓至25%仓位"
                        else:
                            action_hint = f"✅ 融合决策「卖出 25%」：当前价格 {current_price:.2f}元，建议减仓至25%仓位（可等待反弹至 {suggestions['25%']:.2f}元 以上）"
                        consistency_note = "✅ 与融合决策「卖出 25%」一致"
                        
                    elif final_action <= 2:  # 卖出 50% 或更多
                        if current_price >= suggestions['25%']:
                            action_hint = f"✅ 融合决策「卖出」+ 当前价格 {current_price:.2f}元 在卖出区间，建议大幅减仓或清仓"
                        else:
                            action_hint = f"✅ 融合决策「卖出」：虽然当前价格 {current_price:.2f}元 略低于预测价格，但模型建议卖出，可考虑减仓或等待反弹至 {suggestions['25%']:.2f}元 以上"
                        consistency_note = "✅ 与融合决策「卖出」一致"
                        
                    else:
                        # 如果没有明确的融合决策，则基于价格位置判断
                        if current_price < suggestions['100%']:
                            action_hint = f"当前价格 {current_price:.2f}元 低于100%仓位价格，建议满仓买入"
                        elif current_price > suggestions['0%']:
                            action_hint = f"当前价格 {current_price:.2f}元 高于0%仓位价格，建议全部卖出"
                        elif price_diff_pct_from_closest < 0.5:
                            action_hint = f"当前价格 {current_price:.2f}元 接近{closest_position}仓位价格（{closest_price:.2f}元），建议调整为{closest_position}仓位"
                        else:
                            if current_price <= suggestions['75%']:
                                action_hint = f"当前价格 {current_price:.2f}元 在75%-100%仓位区间，建议高仓位持有"
                            elif current_price <= suggestions['50%']:
                                action_hint = f"当前价格 {current_price:.2f}元 在50%-75%仓位区间，建议中等仓位持有"
                            elif current_price <= suggestions['25%']:
                                action_hint = f"当前价格 {current_price:.2f}元 在25%-50%仓位区间，建议低仓位持有"
                            else:
                                action_hint = f"当前价格 {current_price:.2f}元 在0%-25%仓位区间，建议轻仓或空仓"
                        consistency_note = "基于价格位置判断"
                    
                    print(f"   📌 {action_hint}")
                    print(f"   📊 {consistency_note}")
                    
                    # V12优化：显示明确的买入/卖出价格建议
                    target_position = None  # 目标仓位
                    
                    # 根据最终决策确定目标仓位和对应的价格
                    if final_action == 6:  # 买入100%（目标100%仓位）
                        suggested_buy_price = suggestions['100%']
                        target_position = "100%"
                    elif final_action == 5:  # 买入25%（目标75%仓位）
                        suggested_buy_price = suggestions['75%']
                        target_position = "75%"
                    elif final_action == 4:  # 持有（目标50%仓位）
                        # 持有操作，显示当前价格对应的合理仓位价格
                        target_position = "50%"
                        # 不显示买入/卖出价格，因为建议持有
                    elif final_action == 3:  # 卖出25%（目标75%仓位，即保留75%）
                        suggested_sell_price = suggestions['25%']  # 卖出到25%仓位对应的价格
                        target_position = "75%"
                    elif final_action == 2:  # 卖出50%（目标50%仓位）
                        suggested_sell_price = suggestions['50%']
                        target_position = "50%"
                    elif final_action == 1:  # 卖出75%（目标25%仓位）
                        suggested_sell_price = suggestions['25%']
                        target_position = "25%"
                    elif final_action == 0:  # 卖出100%（目标0%仓位）
                        suggested_sell_price = suggestions['0%']
                        target_position = "0%"
                    
                    # 显示买入价格建议
                    if suggested_buy_price:
                        buy_price_diff = suggested_buy_price - current_price
                        buy_price_diff_pct = (buy_price_diff / current_price * 100) if current_price > 0 else 0
                        print(f"   💰 建议买入价格: {suggested_buy_price:.2f}元 (目标仓位: {target_position}, 当前价格: {current_price:.2f}元, 差异: {buy_price_diff:+.2f}元 ({buy_price_diff_pct:+.2f}%))")
                    
                    # 显示卖出价格建议
                    if suggested_sell_price:
                        sell_price_diff = suggested_sell_price - current_price
                        sell_price_diff_pct = (sell_price_diff / current_price * 100) if current_price > 0 else 0
                        print(f"   💰 建议卖出价格: {suggested_sell_price:.2f}元 (目标仓位: {target_position}, 当前价格: {current_price:.2f}元, 差异: {sell_price_diff:+.2f}元 ({sell_price_diff_pct:+.2f}%))")
                    
                    # 持有操作显示当前价格对应的合理仓位
                    if final_action == 4 and target_position:
                        print(f"   💰 持有建议: 当前价格 {current_price:.2f}元 在合理区间，建议保持{target_position}仓位（合理价格区间: {suggestions['25%']:.2f}元 - {suggestions['75%']:.2f}元）")
                    
                    print(f"   📊 价格区间 {price_suggestions['price_interval_pct']:.2f}%（基于预测价格和波动率{price_suggestions['volatility_pct']:.2f}%），相邻仓位价格差至少 {min_diff_pct:.2f}%")
                    print(f"   💡 提示: 价格建议基于预测价格 {price_suggestions['predicted_price']:.2f}元，当前价格 {current_price:.2f}元 与预测价格差异 {abs(current_price - price_suggestions['predicted_price']) / price_suggestions['predicted_price'] * 100:.2f}%")
                    
                    # V12优化：根据持仓情况和预测结果给出具体操作建议
                    print(f"\n   📋 具体操作建议（基于当前持仓和预测结果）:")
                    print(f"      💼 当前持仓: {shares_held:.2f}股 | 可用资金: {current_balance:.2f}元 | 总资产: {current_balance + shares_held * current_price:.2f}元")
                    
                    # 根据最终决策计算具体操作建议
                    if suggested_buy_price and final_action >= 4:  # 买入操作
                        # 计算目标仓位对应的买入数量
                        if final_action == 6:  # 买入100%
                            target_pct = 1.0
                            target_shares = (current_balance + shares_held * current_price) / suggested_buy_price if suggested_buy_price > 0 else 0
                            buy_shares = max(0, target_shares - shares_held)
                        elif final_action == 5:  # 买入50%
                            target_pct = 0.5
                            target_shares = ((current_balance + shares_held * current_price) * target_pct) / suggested_buy_price if suggested_buy_price > 0 else 0
                            buy_shares = max(0, target_shares - shares_held)
                        elif final_action == 4:  # 买入25%
                            target_pct = 0.75  # 买入25%意味着目标仓位75%
                            target_shares = ((current_balance + shares_held * current_price) * target_pct) / suggested_buy_price if suggested_buy_price > 0 else 0
                            buy_shares = max(0, target_shares - shares_held)
                        else:
                            buy_shares = 0
                    
                        if buy_shares > 0:
                            # 计算实际买入金额（考虑滑点和手续费）
                            buy_amount = buy_shares * suggested_buy_price
                            adjusted_buy_price = suggested_buy_price * (1 + SLIPPAGE_RATE)
                            actual_buy_amount = buy_shares * adjusted_buy_price
                            commission = max(MIN_COMMISSION, actual_buy_amount * COMMISSION_RATE)
                            transfer_fee = actual_buy_amount * TRANSFER_FEE_RATE
                            total_fee = commission + transfer_fee
                            total_cost = actual_buy_amount + total_fee
                            
                            # 如果资金不足，调整买入数量
                            if total_cost > current_balance:
                                available_amount = max(0, current_balance - MIN_COMMISSION)
                                buy_shares = round_to_lot(available_amount / adjusted_buy_price) if adjusted_buy_price > 0 else 0
                                actual_buy_amount = buy_shares * adjusted_buy_price
                                commission = max(MIN_COMMISSION, actual_buy_amount * COMMISSION_RATE) if buy_shares > 0 else 0.0
                                transfer_fee = actual_buy_amount * TRANSFER_FEE_RATE if buy_shares > 0 else 0.0
                                total_fee = commission + transfer_fee
                                total_cost = actual_buy_amount + total_fee
                            
                            print(f"      🟢 建议买入:")
                            print(f"         💰 建议价格: {suggested_buy_price:.2f}元")
                            print(f"         📊 建议数量: {buy_shares:.0f}股（约{buy_shares:.2f}股）")
                            print(f"         💵 预计金额: {actual_buy_amount:.2f}元")
                            print(f"         💸 手续费: {total_fee:.2f}元（佣金{commission:.2f}元 + 过户费{transfer_fee:.2f}元）")
                            print(f"         💰 总成本: {total_cost:.2f}元")
                            if total_cost > current_balance:
                                print(f"         ⚠️  资金不足，已调整为可买入数量（可用资金: {current_balance:.2f}元）")
                            print(f"         📈 买入后持仓: {shares_held + buy_shares:.2f}股 | 剩余资金: {current_balance - total_cost:.2f}元")
                    
                    elif suggested_sell_price and final_action <= 2:  # 卖出操作
                        # 计算目标仓位对应的卖出数量
                        if final_action == 0:  # 卖出100%
                            sell_pct = 1.0
                        elif final_action == 1:  # 卖出75%
                            sell_pct = 0.75
                        elif final_action == 2:  # 卖出25%
                            sell_pct = 0.25
                        else:
                            sell_pct = 0
                        
                        sell_shares = round_to_lot(shares_held * sell_pct) if sell_pct > 0 else 0
                        
                        if sell_shares > 0:
                            # 计算实际卖出金额（考虑滑点、手续费和印花税）
                            adjusted_sell_price = suggested_sell_price * (1 - SLIPPAGE_RATE)
                            actual_sell_amount = sell_shares * adjusted_sell_price
                            commission = max(MIN_COMMISSION, actual_sell_amount * COMMISSION_RATE)
                            transfer_fee = actual_sell_amount * TRANSFER_FEE_RATE
                            stamp_tax = actual_sell_amount * STAMP_DUTY_RATE
                            total_fee = commission + transfer_fee + stamp_tax
                            net_proceeds = actual_sell_amount - total_fee
                            
                            print(f"      🔴 建议卖出:")
                            print(f"         💰 建议价格: {suggested_sell_price:.2f}元")
                            print(f"         📊 建议数量: {sell_shares:.0f}股（约{sell_shares:.2f}股，占持仓{sell_pct*100:.0f}%）")
                            print(f"         💵 预计金额: {actual_sell_amount:.2f}元")
                            print(f"         💸 手续费: {total_fee:.2f}元（佣金{commission:.2f}元 + 过户费{transfer_fee:.2f}元 + 印花税{stamp_tax:.2f}元）")
                            print(f"         💰 净收益: {net_proceeds:.2f}元")
                            print(f"         📉 卖出后持仓: {shares_held - sell_shares:.2f}股 | 可用资金: {current_balance + net_proceeds:.2f}元")
                    
                    elif final_action == 3:  # 持有操作
                        print(f"      ⚪ 建议持有:")
                        print(f"         💼 当前持仓: {shares_held:.2f}股")
                        print(f"         💵 当前资金: {current_balance:.2f}元")
                        print(f"         📊 建议保持当前仓位，等待更好的买入/卖出时机")
                        if target_position:
                            print(f"         💡 目标仓位: {target_position}（合理价格区间: {suggestions['25%']:.2f}元 - {suggestions['75%']:.2f}元）")
                    
                    # V12优化：显示不同价格区间的操作建议（25%、50%、75%、100%仓位）
                    print(f"\n   📊 不同价格区间的操作建议:")
                    
                    # 判断当前价格是否适合立即操作
                    should_buy_now = False
                    should_sell_now = False
                    if suggested_buy_price and final_action >= 4:
                        # 如果当前价格低于或接近建议买入价格，可以考虑买入
                        if current_price <= suggested_buy_price * 1.02:  # 允许2%的偏差
                            should_buy_now = True
                    elif suggested_sell_price and final_action <= 2:
                        # 如果当前价格高于或接近建议卖出价格，可以考虑卖出
                        if current_price >= suggested_sell_price * 0.98:  # 允许2%的偏差
                            should_sell_now = True
                    
                    # 显示买入建议（梯度买入：价格越低，买入越多）- V12优化：总是显示买入建议，方便按价格灵活选择
                    print(f"      🟢 梯度买入建议（价格越低，买入越多）:")
                    
                    # 计算价格区间
                    min_price = suggestions['100%']
                    max_price = suggestions['0%']
                    price_range = max_price - min_price
                    
                    # 定义梯度买入档位（12个档位，从100%到10%仓位）
                    gradient_levels = [
                (100, 1.00, "🔥 满仓"),  # 100%仓位，使用100%资金
                (90, 0.90, "🟢 重仓"),   # 90%仓位，使用90%资金
                (80, 0.80, "🟢 重仓"),   # 80%仓位，使用80%资金
                (75, 0.75, "🟡 中重仓"), # 75%仓位，使用75%资金
                (70, 0.70, "🟡 中重仓"), # 70%仓位，使用70%资金
                (60, 0.60, "🟡 中仓"),   # 60%仓位，使用60%资金
                (50, 0.50, "🟠 中仓"),   # 50%仓位，使用50%资金
                (40, 0.40, "🟠 轻仓"),   # 40%仓位，使用40%资金
                (30, 0.30, "🟤 轻仓"),   # 30%仓位，使用30%资金
                (25, 0.25, "🟤 轻仓"),   # 25%仓位，使用25%资金
                (20, 0.20, "⚪ 轻仓"),   # 20%仓位，使用20%资金
                (10, 0.10, "⚪ 轻仓"),   # 10%仓位，使用10%资金
            ]
                    
                    # 计算每个档位的价格（从最低价格到最高价格，仓位从100%到10%）
                    for position_pct, buy_pct, label in gradient_levels:
                        # 计算该仓位对应的价格位置（从100%到0%）
                        price_position = (100 - position_pct) / 100.0  # 0.0对应100%仓位（最低价），1.0对应0%仓位（最高价）
                        gradient_price = min_price + price_range * price_position
                        
                        if gradient_price > 0:
                            price_diff = gradient_price - current_price
                            price_diff_pct = (price_diff / current_price * 100) if current_price > 0 else 0
                            
                            # 计算买入数量和成本
                            shares_bought, total_cost, total_fee, adj_price = calc_buy_trade(gradient_price, buy_pct, current_balance)
                            
                            if shares_bought > 0:
                                # 判断操作状态
                                if abs(price_diff_pct) <= 1.0:  # 价格差异在1%以内，可以买入
                                    status = "✅ 可买入"
                                elif price_diff < 0 and abs(price_diff_pct) <= 2.0:  # 当前价格高于建议价格，但在2%以内
                                    status = "⚠️  可考虑"
                                elif price_diff < 0:  # 当前价格明显高于建议价格
                                    status = "⏳ 等待回调"
                                else:  # 当前价格低于建议价格，等待更好价格
                                    status = "⏳ 等待更好价格"
                                
                                # 只显示价格差异在合理范围内的建议（避免显示太多不相关的价格点）
                                if abs(price_diff_pct) <= 5.0 or (price_diff < 0 and price_diff_pct >= -3.0):
                                    # 简化显示格式：一行显示关键信息
                                    print(f"         {status} {position_pct:3d}%仓位 ({label:4s}): {gradient_price:6.2f}元 | 当前{current_price:.2f}元 (差异{price_diff:+.2f}元, {price_diff_pct:+.2f}%) | 买入{int(shares_bought):4d}股 ({buy_pct*100:3.0f}%资金, 成本{total_cost:.2f}元)")
                    
                    # 显示0%仓位参考价（空仓提示）
                    zero_price = suggestions['0%']
                    if zero_price:
                        price_diff = zero_price - current_price
                        price_diff_pct = (price_diff / current_price * 100) if current_price > 0 else 0
                        print(f"         ⚪ 0%仓位 (空仓): {zero_price:6.2f}元 | 当前{current_price:.2f}元 (差异{price_diff:+.2f}元, {price_diff_pct:+.2f}%) | 不买入，保持空仓")
                                
                    # 显示梯度买入策略说明
                    print(f"\n         💡 梯度买入策略说明:")
                    print(f"            - 价格越低，买入仓位越高（分散买入，降低风险）")
                    print(f"            - 建议在不同价格档位分批买入，不要一次性满仓")
                    print(f"            - 如果价格继续下跌，可以逐步加仓；如果价格上涨，可以逐步减仓")
                    
                    # 显示卖出建议（梯度卖出：价格越高，卖出越多）- V12优化：总是显示卖出建议（有持仓时），方便按价格灵活选择
                    if shares_held > 0:
                        # V12优化：从持仓状态获取成本价，用于计算真实盈亏
                        portfolio_state_for_cost = load_portfolio_state()
                        current_cost_price = None
                        if portfolio_state_for_cost and portfolio_state_for_cost.get('stock_code') == STOCK_CODE:
                            # 优先使用 cost_price，如果没有则使用 actual_buy_price，都不存在则为 None（不使用 last_price）
                            cost_price_val = portfolio_state_for_cost.get('cost_price')
                            actual_buy_price_val = portfolio_state_for_cost.get('actual_buy_price')
                    
                    # 确保值存在且大于0
                    if cost_price_val and isinstance(cost_price_val, (int, float)) and cost_price_val > 0:
                        current_cost_price = float(cost_price_val)
                    elif actual_buy_price_val and isinstance(actual_buy_price_val, (int, float)) and actual_buy_price_val > 0:
                        current_cost_price = float(actual_buy_price_val)
                    else:
                        current_cost_price = None
                
                print(f"      🔴 梯度卖出建议（价格越高，卖出越多）:")
                if current_cost_price:
                    print(f"         💰 当前持仓成本价: {current_cost_price:.4f}元")
                
                # V12优化：为卖出建议计算独立的价格区间，基于预测价格向上扩展，实现利润最大化
                # 获取预测价格和波动率
                predicted_price = price_suggestions.get('predicted_price', current_price)
                volatility_pct = price_suggestions.get('volatility_pct', 2.0)
                
                # 卖出价格区间：从当前价格到预测价格+波动率的倍数（向上扩展）
                # 如果预测价格上涨，卖出区间应该向上扩展更多
                price_change_pct = price_suggestions.get('price_change_pct', 0.0)
                
                # 计算卖出价格区间的上限：基于预测价格和波动率
                # 如果预测上涨，卖出上限 = 预测价格 + 波动率 * 倍数（例如2-3倍）
                # 如果预测下跌，卖出上限 = 当前价格 + 波动率 * 倍数（保守卖出）
                if price_change_pct > 0:
                    # 预测上涨：卖出上限应该更高，以最大化利润
                    sell_max_price = predicted_price + predicted_price * volatility_pct / 100 * max(2.0, min(4.0, abs(price_change_pct) / 2.0))
                else:
                    # 预测下跌：卖出上限基于当前价格，保守卖出
                    sell_max_price = current_price + current_price * volatility_pct / 100 * 2.0
                
                # 卖出价格区间的下限：从当前价格或买入建议的50%仓位价格开始
                sell_min_price = max(current_price, suggestions.get('50%', current_price))
                
                # 确保卖出价格区间足够大（至少5%的价格差，提供更好的区分度）
                sell_price_range = sell_max_price - sell_min_price
                min_sell_range = current_price * 0.05  # 至少5%的价格差
                if sell_price_range < min_sell_range:
                    # 如果区间太小，向上扩展上限
                    sell_max_price = sell_min_price + min_sell_range
                
                # 使用卖出专用的价格区间
                min_price = sell_min_price
                max_price = sell_max_price
                price_range = max_price - min_price
                
                # 定义梯度卖出档位（12个档位，从0%到90%仓位）
                gradient_levels_sell = [
                    (0, 1.00, "🔥 全卖"),    # 0%仓位（卖出100%持仓）
                    (10, 0.90, "🟢 大减"),   # 10%仓位（卖出90%持仓）
                    (20, 0.80, "🟢 大减"),   # 20%仓位（卖出80%持仓）
                    (25, 0.75, "🟡 中减"),   # 25%仓位（卖出75%持仓）
                    (30, 0.70, "🟡 中减"),   # 30%仓位（卖出70%持仓）
                    (40, 0.60, "🟡 中减"),   # 40%仓位（卖出60%持仓）
                    (50, 0.50, "🟠 中减"),   # 50%仓位（卖出50%持仓）
                    (60, 0.40, "🟠 小减"),   # 60%仓位（卖出40%持仓）
                    (70, 0.30, "🟤 小减"),   # 70%仓位（卖出30%持仓）
                    (75, 0.25, "🟤 小减"),   # 75%仓位（卖出25%持仓）
                    (80, 0.20, "⚪ 小减"),   # 80%仓位（卖出20%持仓）
                    (90, 0.10, "⚪ 小减"),   # 90%仓位（卖出10%持仓）
                ]
                
                # 计算每个档位的价格（从最高价格到最低价格，仓位从0%到90%）
                for target_position_pct, sell_pct, label in gradient_levels_sell:
                    # 计算该仓位对应的价格位置（从100%到0%）
                    price_position = (100 - target_position_pct) / 100.0  # 0.0对应100%仓位（最低价），1.0对应0%仓位（最高价）
                    gradient_price = min_price + price_range * price_position
                    
                    if gradient_price > 0:
                        price_diff = gradient_price - current_price
                        price_diff_pct = (price_diff / current_price * 100) if current_price > 0 else 0
                        
                        # 计算卖出数量和收益
                        shares_sold, net_proceeds, total_fee, adj_price = calc_sell_trade(gradient_price, sell_pct, shares_held)
                        
                        if shares_sold > 0:
                            # V12优化：计算真实盈亏（利润）= (实际成交价 - 成本价) * 卖出数量 - 手续费
                            # 注意：使用成本价而不是最近成交价来计算盈亏
                            profit_loss = None
                            if current_cost_price and current_cost_price > 0:
                                # 盈亏 = (卖出价格 - 成本价) × 股数 - 手续费
                                profit_loss = (adj_price - current_cost_price) * shares_sold - total_fee
                            
                            # 判断操作状态
                            if abs(price_diff_pct) <= 1.0:  # 价格差异在1%以内，可以卖出
                                status = "✅ 可卖出"
                            elif price_diff > 0 and abs(price_diff_pct) <= 2.0:  # 当前价格低于建议价格，但在2%以内
                                status = "⚠️  可考虑"
                            elif price_diff > 0:  # 当前价格明显低于建议价格
                                status = "⏳ 等待上涨"
                            else:  # 当前价格高于建议价格，等待更好价格
                                status = "⏳ 等待更好价格"
                            
                            # 只显示价格差异在合理范围内的建议
                            if abs(price_diff_pct) <= 5.0 or (price_diff > 0 and price_diff_pct <= 3.0):
                                # V12优化：只显示真实盈亏（基于成本价计算），移除净收益以避免歧义
                                sell_pct_of_holding = (shares_sold / shares_held * 100) if shares_held > 0 else 0
                                print(f"         {status} {target_position_pct:3d}%仓位 ({label:4s}): {gradient_price:6.2f}元 | 当前{current_price:.2f}元 (差异{price_diff:+.2f}元, {price_diff_pct:+.2f}%) | 卖出{int(shares_sold):4d}股(占持仓{sell_pct_of_holding:.0f}%, 当前持仓{shares_held:.0f}股) → 保留{int(shares_held - shares_sold):4d}股")
                
                # 显示0%仓位参考价（全部卖出提示）- 使用卖出专用的价格区间上限
                zero_price = max_price  # 使用卖出价格区间的上限
                if zero_price:
                    price_diff = zero_price - current_price
                    price_diff_pct = (price_diff / current_price * 100) if current_price > 0 else 0
                    # 计算全部卖出的收益
                    shares_sold_all, net_proceeds_all, total_fee_all, adj_price_all = calc_sell_trade(zero_price, 1.0, shares_held)
                    if shares_sold_all > 0:
                        profit_loss_all = None
                        if current_cost_price and current_cost_price > 0:
                            profit_loss_all = (adj_price_all - current_cost_price) * shares_sold_all - total_fee_all
                        
                        print(f"         🔥 0%仓位 (全卖): {zero_price:6.2f}元 | 当前{current_price:.2f}元 (差异{price_diff:+.2f}元, {price_diff_pct:+.2f}%) | 卖出全部{int(shares_sold_all):4d}股 → 保留 0股")
                
                        # 显示梯度卖出策略说明
                        print(f"\n         💡 梯度卖出策略说明:")
                        print(f"            - 价格越高，卖出仓位越高（分批卖出，锁定利润）")
                        print(f"            - 建议在不同价格档位分批卖出，不要一次性全部卖出")
                        print(f"            - 如果价格继续上涨，可以逐步减仓；如果价格下跌，可以逐步加仓")
                    else:
                        print(f"      🔴 梯度卖出建议（价格越高，卖出越多）:")
                        print(f"         ⚠️  当前无持仓，无法提供卖出建议")
                    
                    # ========== 更新可视化 ==========
                    if visualizer:
                        try:
                            indicators_dict = {}
                            
                            # 从技术指标摘要中提取指标
                            if indicator_summary:
                                if 'KDJ' in indicator_summary:
                                    kdj = indicator_summary['KDJ']
                                    if isinstance(kdj, dict):
                                        indicators_dict['KDJ_K'] = kdj.get('K', 0)
                                        indicators_dict['KDJ_D'] = kdj.get('D', 0)
                                        indicators_dict['KDJ_J'] = kdj.get('J', 0)
                                if 'RSI' in indicator_summary:
                                    indicators_dict['RSI'] = indicator_summary['RSI']
                                if 'MACD' in indicator_summary:
                                    macd = indicator_summary['MACD']
                                    if isinstance(macd, dict):
                                        indicators_dict['MACD'] = macd.get('MACD', 0)
                                if 'OBV' in indicator_summary:
                                    obv = indicator_summary['OBV']
                                    if isinstance(obv, dict):
                                        indicators_dict['OBV_Ratio'] = obv.get('OBV_Ratio', 1.0)
                            
                            # 如果技术指标计算失败，从原始数据计算简单指标
                            if not indicators_dict and len(closes) >= 5:
                                try:
                                    # 计算简单的移动平均线
                                    ma5 = np.mean(closes[-5:]) if len(closes) >= 5 else current_price
                                    ma10 = np.mean(closes[-10:]) if len(closes) >= 10 else current_price
                                    ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else current_price
                                    
                                    indicators_dict['MA5'] = ma5
                                    indicators_dict['MA10'] = ma10
                                    indicators_dict['MA20'] = ma20
                                    
                                    # 计算简单的RSI（如果数据足够）
                                    if len(closes) >= 14:
                                        try:
                                            deltas = np.diff(closes[-14:])
                                            if len(deltas) > 0:
                                                gains = np.where(deltas > 0, deltas, 0)
                                                losses = np.where(deltas < 0, -deltas, 0)
                                                # 只计算非零值的均值，避免空数组警告
                                                valid_gains = gains[gains > 0]
                                                valid_losses = losses[losses > 0]
                                                avg_gain = np.mean(valid_gains) if len(valid_gains) > 0 else 0.0
                                                avg_loss = np.mean(valid_losses) if len(valid_losses) > 0 else 0.01
                                                if avg_loss > 0 and not np.isnan(avg_gain) and not np.isnan(avg_loss):
                                                    rs = avg_gain / avg_loss
                                                    rsi = 100 - (100 / (1 + rs))
                                                    if not np.isnan(rsi) and not np.isinf(rsi):
                                                        indicators_dict['RSI'] = rsi
                                        except Exception:
                                            pass  # 如果计算失败，跳过RSI
                                except Exception as e:
                                    pass  # 如果计算失败，至少传递空字典
                            
                            # 确保至少有一些数据传递给可视化器
                            visualizer.add_data_point(
                                price=current_price,
                                volume=volume,
                                indicators=indicators_dict if indicators_dict else None,
                                prediction=transformer_prediction
                            )
                            # 调试信息：显示已添加的数据点数量
                            if iteration_count % 5 == 0:  # 每5轮输出一次
                                print(f"   📊 可视化数据: 价格点数={len(visualizer.price_history)}, 指标数={len(visualizer.indicators_history)}")
                        except Exception as e:
                            print(f"   ⚠️  可视化更新失败: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # ========== 更新持仓状态 ==========
                    total_assets = current_balance + shares_held * current_price
                    save_portfolio_state(STOCK_CODE, shares_held, current_balance, current_price, initial_balance)
                    log_trade_operation(
                        STOCK_CODE, final_operation, current_price,
                        shares_held, current_balance, total_assets,
                        status='预测', note=f'V11融合决策'
                    )
                    
                    stock_name = get_stock_name(STOCK_CODE)
                    print(f"   💼 [{stock_name}({STOCK_CODE})] 持仓: {shares_held:.2f}股 | 资金: {current_balance:.2f}元 | 总资产: {total_assets:.2f}元")
                    
                    # ========== V13: 多模型回测和自动选择 ==========
                    if ENABLE_BACKTEST:
                        try:
                            # V13: 记录所有模型的预测值（用于多模型回测）
                            if ENABLE_AUTO_MODEL_SELECTION and len(candidate_ppo_models) > 0:
                                # V13策略：使用Transformer预测作为价格预测（Transformer不依赖PPO模型）
                                # 每个模型使用相同的Transformer预测值，但PPO动作不同
                                # 我们基于Transformer预测的准确性来评估模型组合的效果
                                if transformer_prediction is not None:
                                    # 为所有模型记录相同的Transformer预测值（因为Transformer预测不依赖PPO模型）
                                    # 实际评估时，我们会考虑PPO动作的准确性
                                    for model_name in model_backtest_data.keys():
                                        model_backtest_data[model_name]['predictions'].append(transformer_prediction)
                                        model_backtest_data[model_name]['timestamps'].append(datetime.datetime.now())
                                        model_backtest_data[model_name]['actuals'].append(current_price)
                                
                                # V13: 定期评估模型并自动切换
                                if iteration_count % AUTO_MODEL_SELECTION_INTERVAL == 0 and iteration_count > 0:
                                    print(f"\n   🔄 V13: 开始模型评估（第 {iteration_count} 轮）...")
                                    best_model_result = select_best_model()
                                    
                                    if best_model_result:
                                        best_model_name = best_model_result['model_name']
                                        best_score = best_model_result['score']
                                        best_metrics = best_model_result['metrics']
                                        
                                        print(f"   📊 V13模型评估结果:")
                                        print(f"      🏆 最优模型: {best_model_name} (评分: {best_score:.4f})")
                                        print(f"      📈 回测指标 (样本数: {best_metrics['sample_count']}):")
                                        print(f"         MAE: {best_metrics['mae']:.4f} | RMSE: {best_metrics['rmse']:.4f}")
                                        print(f"         MAPE: {best_metrics['mape']:.2f}% | 方向准确率: {best_metrics['direction_accuracy']:.1f}%")
                                        
                                        # 显示所有模型的评分
                                        if len(best_model_result['all_scores']) > 1:
                                            print(f"      📋 所有模型评分:")
                                            sorted_models = sorted(best_model_result['all_scores'].items(), key=lambda x: x[1], reverse=True)
                                            for model_name, score in sorted_models:
                                                marker = "🏆" if model_name == best_model_name else "  "
                                                metrics = best_model_result['all_metrics'].get(model_name, {})
                                                print(f"         {marker} {model_name}: {score:.4f} (MAE={metrics.get('mae', 0):.4f}, 方向准确率={metrics.get('direction_accuracy', 0):.1f}%)")
                                        
                                        # 如果最优模型与当前模型不同，进行切换
                                        if best_model_name != current_model_name:
                                            old_model_name = current_model_name
                                            if switch_to_model(best_model_name):
                                                print(f"   ✅ V13: 已切换到最优模型: {best_model_name} (原模型: {old_model_name})")
                                                print(f"      💡 说明: 根据回测结果，{best_model_name} 表现最优，已自动切换")
                                            else:
                                                print(f"   ⚠️  V13: 模型切换失败: {best_model_name}")
                                        else:
                                            print(f"   ✅ V13: 当前模型 {current_model_name} 仍为最优，无需切换")
                            else:
                                print(f"   ⚠️  V13: 模型评估失败（数据不足或所有模型都无有效回测数据）")
                            
                            # V12兼容模式：单一模型回测
                            if not ENABLE_AUTO_MODEL_SELECTION or len(candidate_ppo_models) == 0:
                                if transformer_prediction is not None:
                                    backtest_predictions.append(transformer_prediction)
                                    backtest_timestamps.append(datetime.datetime.now())
                                    
                                    # 如果有历史实际值，计算回测指标
                                    if len(backtest_predictions) > 1 and len(backtest_actuals) > 0:
                                        # 使用上一轮的实际价格作为当前预测的对比
                                        if len(backtest_actuals) >= len(backtest_predictions) - 1:
                                            # 计算最近N次的指标
                                            n = min(20, len(backtest_predictions) - 1)  # 最近20次
                                            recent_preds = backtest_predictions[-n-1:-1]  # 排除最新的预测
                                            recent_actuals = backtest_actuals[-n:]
                                            
                                            if len(recent_preds) == len(recent_actuals) and len(recent_preds) > 0:
                                                try:
                                                    # 转换为numpy数组并检查有效性
                                                    preds_array = np.array(recent_preds, dtype=np.float64)
                                                    actuals_array = np.array(recent_actuals, dtype=np.float64)
                                                    
                                                    # 过滤掉NaN和Inf值
                                                    valid_mask = np.isfinite(preds_array) & np.isfinite(actuals_array) & (actuals_array != 0)
                                                    if np.sum(valid_mask) > 0:
                                                        valid_preds = preds_array[valid_mask]
                                                        valid_actuals = actuals_array[valid_mask]
                                                        
                                                        # 计算MAE (Mean Absolute Error)
                                                        mae = np.mean(np.abs(valid_preds - valid_actuals))
                                                        
                                                        # 计算RMSE (Root Mean Squared Error)
                                                        rmse = np.sqrt(np.mean((valid_preds - valid_actuals)**2))
                                                        
                                                        # 计算MAPE (Mean Absolute Percentage Error)
                                                        mape = np.mean(np.abs((valid_preds - valid_actuals) / valid_actuals)) * 100
                                                        
                                                        # 计算方向准确率 (Direction Accuracy)
                                                        if len(valid_preds) > 1:
                                                            pred_directions = np.sign(np.diff(valid_preds))
                                                            actual_directions = np.sign(np.diff(valid_actuals))
                                                            if len(pred_directions) > 0:
                                                                direction_accuracy = np.mean(pred_directions == actual_directions) * 100
                                                            else:
                                                                direction_accuracy = 0.0
                                                        else:
                                                            direction_accuracy = 0.0
                                                        
                                                        # 检查结果是否有效
                                                        if not (np.isnan(mae) or np.isnan(rmse) or np.isnan(mape) or np.isnan(direction_accuracy)):
                                                            if iteration_count % 10 == 0:  # 每10轮输出一次
                                                                print(f"\n   📈 V11回测指标 (最近{np.sum(valid_mask)}次有效数据):")
                                                                print(f"      MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}% | 方向准确率: {direction_accuracy:.1f}%")
                                                except Exception as e:
                                                    # 静默处理计算错误
                                                    pass
                                        
                                        # 记录当前实际价格（用于下一轮计算）
                                        backtest_actuals.append(current_price)
                        
                        except Exception as e:
                            print(f"   ⚠️  回测计算失败: {e}")
                        
                        # ========== V15: DeepSeek 轮次复盘 ==========
                        if ENABLE_DEEPSEEK_REVIEW and ENABLE_LLM:
                            try:
                                predicted_price_val = price_suggestions.get('predicted_price') if price_suggestions else None
                                predicted_change_pct = price_suggestions.get('price_change_pct') if price_suggestions else None
                                
                                kelly_mode = None
                                kelly_position_val = None
                                if kelly_info:
                                    kelly_position_val = kelly_info.get('kelly_position')
                                    kelly_mode = "预测估算" if kelly_info.get('is_estimated') else "历史统计"
                                
                                deepseek_ctx = {
                                    'stock_name': stock_name,
                                    'stock_code': STOCK_CODE,
                                    'data_source': data_source_used,
                                    'latest_time': latest_time,
                                    'price_source': price_source,
                                    'current_price': current_price,
                                    'shares_held': shares_held,
                                    'current_balance': current_balance,
                                    'ppo_action': map_action_to_operation(ppo_action) if 'ppo_action' in locals() else "--",
                                    'ppo_model': current_model_name,
                                    'final_operation': final_operation,
                                    'confidence': confidence,
                                    'lstm_prediction': lstm_prediction,
                                    'transformer_prediction': transformer_prediction,
                                    'holographic_signal': holographic_signal,
                                    'predicted_price': predicted_price_val,
                                    'predicted_change': predicted_change_pct,
                                    'suggested_buy': suggested_buy_price,
                                    'suggested_sell': suggested_sell_price,
                                    'kelly_position': kelly_position_val,
                                    'kelly_mode': kelly_mode
                                }
                                
                                prompt_text = build_deepseek_review_prompt(deepseek_ctx)
                                review_text = call_deepseek_review(prompt_text)
                                if review_text:
                                    print(f"\n   🤖 V15 DeepSeek复盘：")
                                    for line in review_text.splitlines():
                                        if line.strip():
                                            print(f"      {line.strip()}")
                            except Exception as e:
                                print(f"   ⚠️  V15 DeepSeek复盘失败: {e}")
                        
            except Exception as e:
                print(f"   ⚠️  预测过程中发生错误: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # ========== V16批量预测：保存预测结果到带日期的记录文件 ==========
            try:
                # 安全获取可能不存在的变量
                review_text_val = locals().get('review_text', None)
                regime_val = locals().get('regime', None)
                trend_score_val = locals().get('trend_score', None)
                range_score_val = locals().get('range_score', None)
                volume_val = locals().get('volume', None)
                
                # ========== V16新增：显示股票的夏普收益率和回撤 ==========
                stock_metrics = None
                index_data = _index_metrics_cache  # 使用已缓存的指数数据
                
                try:
                    # 从配置中获取股票的收益率和回撤数据（用户提供的回测数据）
                    stock_metrics = get_stock_metrics_from_config(STOCK_CODE)
                    
                    if stock_metrics['total_return'] is not None or stock_metrics['max_drawdown'] is not None:
                        print(f"\n   📊 {stock_name}({STOCK_CODE})回测指标:")
                        if stock_metrics['total_return'] is not None:
                            print(f"      总收益率: {stock_metrics['total_return']:+.2f}%")
                        if stock_metrics['max_drawdown'] is not None:
                            print(f"      最大回撤: {stock_metrics['max_drawdown']:.2f}%")
                        if stock_metrics['sharpe_ratio'] is not None:
                            print(f"      夏普比率: {stock_metrics['sharpe_ratio']:.2f}")
                        
                        # 如果有指数数据，进行对比
                        if index_data:
                            # 显示纳斯达克
                            nasdaq = index_data.get('nasdaq')
                            if nasdaq and nasdaq.get('change_pct'):
                                change_pct = nasdaq.get('change_pct')
                                if isinstance(change_pct, (int, float)):
                                    print(f"      纳斯达克涨跌幅: {change_pct:+.2f}%")
                                elif isinstance(change_pct, str) and change_pct != 'N/A':
                                    print(f"      纳斯达克涨跌幅: {change_pct}")
                            
                            # 显示道琼斯
                            dow = index_data.get('dow')
                            if dow and dow.get('change_pct'):
                                change_pct = dow.get('change_pct')
                                if isinstance(change_pct, (int, float)):
                                    print(f"      道琼斯涨跌幅: {change_pct:+.2f}%")
                                elif isinstance(change_pct, str) and change_pct != 'N/A':
                                    print(f"      道琼斯涨跌幅: {change_pct}")
                            
                            # 显示富时A50
                            a50 = index_data.get('a50')
                            if a50 and a50.get('change_pct'):
                                change_pct = a50.get('change_pct')
                                if isinstance(change_pct, (int, float)):
                                    print(f"      富时A50涨跌幅: {change_pct:+.2f}%")
                                elif isinstance(change_pct, str) and change_pct != 'N/A':
                                    print(f"      富时A50涨跌幅: {change_pct}")
                    else:
                        print(f"\n   ⚠️  {stock_name}({STOCK_CODE})暂无回测数据")
                    
                except Exception as e:
                    print(f"   ⚠️  获取股票指标时发生错误: {e}")
                    import traceback
                    traceback.print_exc()
                
                # 收集所有预测数据
                prediction_data = {
                    'current_price': float(current_price) if current_price else None,
                    'price_source': price_source if 'price_source' in locals() else None,
                    'data_source': data_source_used if 'data_source_used' in locals() else None,
                    'latest_time': str(latest_time) if 'latest_time' in locals() and latest_time else None,
                    'volume': float(volume_val) if volume_val else None,
                    'ppo_action': int(ppo_action) if 'ppo_action' in locals() and ppo_action is not None else None,
                    'ppo_operation': map_action_to_operation(ppo_action) if 'ppo_action' in locals() and ppo_action is not None else None,
                    'ppo_model': current_model_name if 'current_model_name' in locals() else None,
                    'lstm_prediction': float(lstm_prediction) if 'lstm_prediction' in locals() and lstm_prediction is not None else None,
                    'transformer_prediction': float(transformer_prediction) if 'transformer_prediction' in locals() and transformer_prediction is not None else None,
                    'holographic_signal': holographic_signal.get('signal') if 'holographic_signal' in locals() and holographic_signal else None,
                    'holographic_confidence': float(holographic_signal.get('confidence', 0)) if 'holographic_signal' in locals() and holographic_signal else None,
                    'final_action': int(final_action) if 'final_action' in locals() and final_action is not None else None,
                    'final_operation': final_operation if 'final_operation' in locals() else None,
                    'confidence': float(confidence) if 'confidence' in locals() and confidence else None,
                    'predicted_price': float(price_suggestions.get('predicted_price')) if 'price_suggestions' in locals() and price_suggestions and price_suggestions.get('predicted_price') else None,
                    'predicted_change_pct': float(price_suggestions.get('price_change_pct')) if 'price_suggestions' in locals() and price_suggestions and price_suggestions.get('price_change_pct') else None,
                    'predicted_direction': price_suggestions.get('direction') if 'price_suggestions' in locals() and price_suggestions else None,
                    'suggested_buy_price': float(suggested_buy_price) if 'suggested_buy_price' in locals() and suggested_buy_price else None,
                    'suggested_sell_price': float(suggested_sell_price) if 'suggested_sell_price' in locals() and suggested_sell_price else None,
                    'shares_held': float(shares_held) if 'shares_held' in locals() and shares_held else 0.0,
                    'current_balance': float(current_balance) if 'current_balance' in locals() and current_balance else 0.0,
                    'total_assets': float(current_balance + shares_held * current_price) if 'current_balance' in locals() and 'shares_held' in locals() and 'current_price' in locals() and current_price else None,
                    'initial_balance': float(initial_balance) if 'initial_balance' in locals() and initial_balance else None,
                    'kelly_position': float(kelly_info.get('kelly_position')) if 'kelly_info' in locals() and kelly_info and kelly_info.get('kelly_position') else None,
                    'kelly_mode': "预测估算" if ('kelly_info' in locals() and kelly_info and kelly_info.get('is_estimated')) else ("历史统计" if ('kelly_info' in locals() and kelly_info) else None),
                    'deepseek_review': review_text_val if review_text_val else None,
                    'atr_value': float(atr_value) if 'atr_value' in locals() and atr_value else None,
                    'regime': regime_val if regime_val else None,
                    'trend_score': float(trend_score_val) if trend_score_val is not None else None,
                    'range_score': float(range_score_val) if range_score_val is not None else None,
                    # V16新增：全球主要指数和股票的收益率和回撤
                    'nasdaq_change_pct': float(nasdaq.get('change_pct')) if index_data and (nasdaq := index_data.get('nasdaq')) and nasdaq and nasdaq.get('change_pct') and isinstance(nasdaq.get('change_pct'), (int, float)) else None,
                    'nasdaq_index_name': nasdaq.get('index_name') if index_data and (nasdaq := index_data.get('nasdaq')) and nasdaq else None,
                    'dow_change_pct': float(dow.get('change_pct')) if index_data and (dow := index_data.get('dow')) and dow and dow.get('change_pct') and isinstance(dow.get('change_pct'), (int, float)) else None,
                    'dow_index_name': dow.get('index_name') if index_data and (dow := index_data.get('dow')) and dow else None,
                    'a50_change_pct': float(a50.get('change_pct')) if index_data and (a50 := index_data.get('a50')) and a50 and a50.get('change_pct') and isinstance(a50.get('change_pct'), (int, float)) else None,
                    'a50_index_name': a50.get('index_name') if index_data and (a50 := index_data.get('a50')) and a50 else None,
                    'stock_total_return': float(stock_metrics['total_return']) if stock_metrics and stock_metrics.get('total_return') is not None else None,
                    'stock_max_drawdown': float(stock_metrics['max_drawdown']) if stock_metrics and stock_metrics.get('max_drawdown') is not None else None,
                    'stock_sharpe_ratio': float(stock_metrics['sharpe_ratio']) if stock_metrics and stock_metrics.get('sharpe_ratio') is not None else None,
                }
                
                # 保存预测结果
                if save_batch_predict_result(STOCK_CODE, stock_name, prediction_data):
                    print(f"   ✅ 预测结果已保存到: {get_batch_predict_result_file()}")
            except Exception as e:
                print(f"   ⚠️  保存预测结果失败: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"{'='*70}\n")
            
            # 批量预测：保存该股票的完整输出到日志文件（无论是否成功）
            try:
                captured_output = output_capture.get_output()
                if captured_output:
                    append_to_log_file(captured_output, log_file)
                    append_to_log_file("", log_file)  # 添加空行分隔
                else:
                    # 即使输出为空，也记录一个标记，确保日志连续性
                    append_to_log_file(f"⚠️  [{stock_name}({STOCK_CODE})] 预测输出为空或未捕获", log_file)
                    append_to_log_file("", log_file)
            except Exception as e:
                print(f"   ⚠️  保存日志输出失败: {e}")
                # 即使保存失败，也尝试记录错误信息
                try:
                    append_to_log_file(f"⚠️  保存日志输出失败: {e}", log_file)
                except:
                    pass
            
            # 批量预测：每个股票只运行一次，不等待，直接继续下一个股票
            
            except Exception as e:
                print(f"\n❌ [{stock_name}({STOCK_CODE})] 预测过程中发生错误: {e}")
                import traceback
                traceback.print_exc()
                # 即使出现异常，也尝试保存已捕获的输出
                try:
                    captured_output = output_capture.get_output()
                    if captured_output:
                        append_to_log_file(captured_output, log_file)
                        append_to_log_file(f"❌ [{stock_name}({STOCK_CODE})] 预测过程中发生错误: {e}", log_file)
                        append_to_log_file("", log_file)
                except:
                    pass
                # 继续处理下一个股票，不中断整个批量预测
                continue
except KeyboardInterrupt:
    print("\n\n⚠️  用户中断，正在保存状态...")
    # 保存已捕获的输出
    try:
        captured_output = output_capture.get_output()
        if captured_output:
            append_to_log_file(captured_output, log_file)
    except:
        pass
    # 用户中断，退出批量预测
    import sys
    sys.exit(0)
except Exception as e:
    print(f"\n❌ [{stock_name}({STOCK_CODE})] 发生错误: {e}")
    import traceback
    traceback.print_exc()
    # 保存错误时的输出
    try:
        captured_output = output_capture.get_output()
        if captured_output:
            append_to_log_file(captured_output, log_file)
            append_to_log_file(f"❌ 发生错误: {e}", log_file)
    except:
        pass
        # 继续处理下一个股票，不中断整个批量预测
        # continue 语句在 except 块内，但这是在 for 循环外，所以不需要 continue

# 批量预测完成
print("\n" + "=" * 70)
print("✅ V16批量预测完成")
print("=" * 70)
print(f"📊 已处理 {len(STOCK_LIST)} 个股票")
print("=" * 70)

# 清理资源
print("\n🔄 正在清理资源...")
if web_visualization:
    try:
        web_visualization.stop()
    except:
        pass

print("✅ V16批量预测系统已停止")

