# -*- coding: utf-8 -*-
"""
判断中国卫星(600118)多周期均线多头排列
多头排列定义：
- 30分钟均线多头排列：MA5 > MA10 > MA20 > MA30
- 60分钟均线多头排列：MA5 > MA10 > MA20 > MA30
- 120分钟均线多头排列：MA5 > MA10 > MA20 > MA30
- 日均线多头排列：5日均线 > 10日均线 > 20日均线 > 60日均线
- 周均线多头排列：5周均线 > 10周均线 > 20周均线
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
from contextlib import contextmanager
from io import StringIO
warnings.filterwarnings('ignore')

# 尝试导入数据源
TUSHARE_AVAILABLE = False
AKSHARE_AVAILABLE = False
BAOSTOCK_AVAILABLE = False

try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    pass

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    pass

try:
    import baostock as bs
    BAOSTOCK_AVAILABLE = True
except ImportError:
    pass

# ==================== 配置参数 ====================
TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN', '')  # Tushare token（从环境变量或这里设置）

# 股票代码
STOCK_CODE = '600118'
STOCK_NAME = '中国卫星'
STOCK_MARKET = 'sh'  # 上海

# 均线参数
MINUTE_MA_PERIODS = [5, 10, 20, 30]  # 分钟级均线周期（30分钟、60分钟、120分钟）
DAILY_MA_PERIODS = [5, 10, 20, 60]   # 日均线周期
WEEKLY_MA_PERIODS = [5, 10, 20]      # 周均线周期

# 反爬虫配置
ENABLE_ANTI_CRAWLER = os.getenv('ENABLE_ANTI_CRAWLER', 'true').lower() == 'true'

# ==================== 辅助函数 ====================
@contextmanager
def suppress_stdout():
    """临时抑制标准输出"""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout

# ==================== 初始化数据源 ====================
print("=" * 70)
print(f"判断 {STOCK_NAME}({STOCK_CODE}) 多周期均线多头排列")
print("=" * 70)

DATA_SOURCE = None
pro = None

# 优先尝试 Tushare
if TUSHARE_AVAILABLE and TUSHARE_TOKEN:
    try:
        ts.set_token(TUSHARE_TOKEN)
        pro = ts.pro_api()
        DATA_SOURCE = "tushare"
        print("✅ 数据源: Tushare")
    except Exception as e:
        print(f"⚠️  Tushare 初始化失败: {e}")
        TUSHARE_AVAILABLE = False

# 如果 Tushare 不可用，尝试 AkShare
if DATA_SOURCE is None and AKSHARE_AVAILABLE:
    try:
        DATA_SOURCE = "akshare"
        print("✅ 数据源: AkShare")
    except Exception as e:
        print(f"⚠️  AkShare 初始化失败: {e}")
        AKSHARE_AVAILABLE = False

# 如果前两者都不可用，尝试 baostock
if DATA_SOURCE is None and BAOSTOCK_AVAILABLE:
    try:
        with suppress_stdout():
            bs.login()
        DATA_SOURCE = "baostock"
        print("✅ 数据源: baostock")
    except Exception as e:
        print(f"⚠️  baostock 初始化失败: {e}")
        BAOSTOCK_AVAILABLE = False

# 如果所有数据源都不可用，报错
if DATA_SOURCE is None:
    raise Exception("未找到可用的数据源！请安装 tushare、akshare 或 baostock")

print("=" * 70)
print()

# ==================== 反爬虫支持 ====================
def setup_akshare_environment():
    """设置akshare环境，支持反爬虫"""
    if not AKSHARE_AVAILABLE:
        return
    
    try:
        from anti_crawler_pool import AntiCrawlerPool, setup_akshare_environment as setup_akshare
        
        if ENABLE_ANTI_CRAWLER:
            anti_crawler_pool = AntiCrawlerPool()
            setup_akshare(anti_crawler_pool)
            return anti_crawler_pool
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️  反爬虫设置失败: {e}")
    
    return None

# ==================== 辅助函数 ====================

def get_kline_data(code, market, period='daily', days=250):
    """获取K线数据（日K/周K）"""
    global DATA_SOURCE, TUSHARE_AVAILABLE, AKSHARE_AVAILABLE, BAOSTOCK_AVAILABLE, pro
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
    # baostock需要YYYY-MM-DD格式
    end_date_bs = datetime.now().strftime('%Y-%m-%d')
    start_date_bs = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    df = None
    
    if DATA_SOURCE == "tushare" and TUSHARE_AVAILABLE:
        try:
            ts_code = f"{code}.{market.upper()}" if market == 'sh' else f"{code}.SZ"
            if period == 'daily':
                df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            elif period == 'weekly':
                df = pro.weekly(ts_code=ts_code, start_date=start_date, end_date=end_date)
            
            if df is not None and len(df) > 0:
                df = df.rename(columns={'trade_date': 'date', 'close': 'close', 'open': 'open', 
                                       'high': 'high', 'low': 'low', 'vol': 'volume'})
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
                df = df.sort_values('date')
        except Exception as e:
            print(f"⚠️  Tushare获取数据失败: {e}")
            pass
    
    elif DATA_SOURCE == "akshare" and AKSHARE_AVAILABLE:
        try:
            # 设置反爬虫环境
            if ENABLE_ANTI_CRAWLER:
                setup_akshare_environment()
                time.sleep(1)  # 延迟避免请求过快
            
            if period == 'daily':
                df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, 
                                        end_date=end_date, adjust="qfq")
            elif period == 'weekly':
                df = ak.stock_zh_a_hist(symbol=code, period="weekly", start_date=start_date, 
                                        end_date=end_date, adjust="qfq")
            
            if df is not None and len(df) > 0:
                df = df.rename(columns={'日期': 'date', '收盘': 'close', '开盘': 'open',
                                       '最高': 'high', '最低': 'low', '成交量': 'volume'})
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
        except Exception as e:
            print(f"⚠️  AkShare获取数据失败: {e}")
            # 如果akshare失败，尝试使用baostock
            if BAOSTOCK_AVAILABLE:
                print(f"   尝试使用baostock作为备用数据源...")
                try:
                    # 确保baostock已登录
                    with suppress_stdout():
                        bs.login()
                    bs_code = f"{market}.{code}"
                    freq_map = {'daily': 'd', 'weekly': 'w'}
                    rs = bs.query_history_k_data_plus(bs_code, 
                        "date,open,high,low,close,volume",
                        start_date=start_date_bs, end_date=end_date_bs,
                        frequency=freq_map[period], adjustflag="3")
                    
                    if rs and rs.error_code == '0':
                        data_list = []
                        while rs.next():
                            data_list.append(rs.get_row_data())
                        if data_list:
                            df = pd.DataFrame(data_list, columns=rs.fields)
                            df['date'] = pd.to_datetime(df['date'])
                            for col in ['open', 'high', 'low', 'close', 'volume']:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                            df = df.sort_values('date')
                            print(f"   ✅ 使用baostock成功获取数据")
                except Exception as e2:
                    print(f"   ⚠️  baostock也失败: {e2}")
            pass
    
    elif DATA_SOURCE == "baostock" and BAOSTOCK_AVAILABLE:
        try:
            bs_code = f"{market}.{code}"
            freq_map = {'daily': 'd', 'weekly': 'w'}
            rs = bs.query_history_k_data_plus(bs_code, 
                "date,open,high,low,close,volume",
                start_date=start_date_bs, end_date=end_date_bs,
                frequency=freq_map[period], adjustflag="3")
            
            if rs and rs.error_code == '0':
                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())
                if data_list:
                    df = pd.DataFrame(data_list, columns=rs.fields)
                    df['date'] = pd.to_datetime(df['date'])
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df = df.sort_values('date')
        except Exception as e:
            print(f"⚠️  baostock获取数据失败: {e}")
            pass
    
    return df

def get_minute_kline_data(code, period_minutes=30, days=30):
    """获取分钟级K线数据（30分钟、60分钟、120分钟）"""
    global DATA_SOURCE, AKSHARE_AVAILABLE
    
    if not AKSHARE_AVAILABLE:
        return None
    
    try:
        # 设置反爬虫环境
        if ENABLE_ANTI_CRAWLER:
            setup_akshare_environment()
            time.sleep(1)  # 延迟避免请求过快
        
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
        
        # 对于120分钟，使用60分钟数据聚合
        if period_minutes == 120:
            # 先获取60分钟数据
            df_60min = ak.stock_zh_a_hist_min_em(
                symbol=code, 
                period="60", 
                adjust="qfq", 
                start_date=start_date, 
                end_date=end_date
            )
            
            if df_60min is None or len(df_60min) == 0:
                return None
            
            # 转换列名
            if '收盘' in df_60min.columns:
                df_60min = df_60min.rename(columns={
                    '收盘': 'close', 
                    '开盘': 'open', 
                    '最高': 'high', 
                    '最低': 'low', 
                    '成交量': 'volume'
                })
            
            # 确保必要的列存在
            if 'close' not in df_60min.columns:
                return None
            
            # 添加缺失的列
            if 'high' not in df_60min.columns:
                df_60min['high'] = df_60min['close']
            if 'low' not in df_60min.columns:
                df_60min['low'] = df_60min['close']
            if 'open' not in df_60min.columns:
                df_60min['open'] = df_60min['close']
            if 'volume' not in df_60min.columns:
                df_60min['volume'] = 0
            
            # 排序
            sort_col = '时间' if '时间' in df_60min.columns else ('date' if 'date' in df_60min.columns else df_60min.columns[0])
            df_60min = df_60min.sort_values(sort_col)
            
            # 转换为数值类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df_60min.columns:
                    df_60min[col] = pd.to_numeric(df_60min[col], errors='coerce')
            
            df_60min = df_60min.dropna(subset=['close'])
            
            # 将60分钟数据聚合为120分钟数据（每2条合并为1条）
            if len(df_60min) >= 2:
                df_120min = []
                for i in range(0, len(df_60min), 2):
                    if i + 1 < len(df_60min):
                        row1 = df_60min.iloc[i]
                        row2 = df_60min.iloc[i + 1]
                        new_row = {
                            'open': row1['open'],
                            'high': max(row1['high'], row2['high']),
                            'low': min(row1['low'], row2['low']),
                            'close': row2['close'],
                            'volume': row1['volume'] + row2['volume']
                        }
                        if 'date' in df_60min.columns:
                            new_row['date'] = row2['date']
                        elif '时间' in df_60min.columns:
                            new_row['时间'] = row2['时间']
                        df_120min.append(new_row)
                
                if df_120min:
                    df_result = pd.DataFrame(df_120min)
                    return df_result
            
            return df_60min
        else:
            # 30分钟或60分钟
            period_str = str(period_minutes)
            df = ak.stock_zh_a_hist_min_em(
                symbol=code, 
                period=period_str, 
                adjust="qfq", 
                start_date=start_date, 
                end_date=end_date
            )
            
            if df is None or len(df) == 0:
                return None
            
            # 转换列名
            if '收盘' in df.columns:
                df = df.rename(columns={
                    '收盘': 'close', 
                    '开盘': 'open', 
                    '最高': 'high', 
                    '最低': 'low', 
                    '成交量': 'volume'
                })
            
            # 确保必要的列存在
            if 'close' not in df.columns:
                return None
            
            # 添加缺失的列
            if 'high' not in df.columns:
                df['high'] = df['close']
            if 'low' not in df.columns:
                df['low'] = df['close']
            if 'open' not in df.columns:
                df['open'] = df['close']
            if 'volume' not in df.columns:
                df['volume'] = 0
            
            # 排序
            sort_col = '时间' if '时间' in df.columns else ('date' if 'date' in df.columns else df.columns[0])
            df = df.sort_values(sort_col)
            
            # 转换为数值类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=['close'])
            return df
            
    except Exception as e:
        print(f"⚠️  获取{period_minutes}分钟K线数据失败: {e}")
        return None

def calculate_ma(df, periods):
    """计算多条均线"""
    df = df.copy()
    for period in periods:
        df[f'MA{period}'] = df['close'].rolling(window=period).mean()
    return df

def check_ma_alignment(df, periods):
    """
    检查均线多头排列
    多头排列：MA5 > MA10 > MA20 > MA60（根据periods参数）
    同时检查均线是否呈上升趋势（可选）
    """
    if df is None or len(df) < max(periods):
        return False, {}
    
    # 获取最新数据
    latest = df.iloc[-1]
    
    # 检查均线值是否存在
    ma_values = {}
    for period in periods:
        ma_col = f'MA{period}'
        if ma_col not in df.columns:
            return False, {}
        ma_values[period] = latest[ma_col]
    
    # 检查是否为空值
    if any(pd.isna(ma_values[p]) for p in periods):
        return False, {}
    
    # 检查多头排列：短期均线 > 长期均线
    is_aligned = True
    for i in range(len(periods) - 1):
        if ma_values[periods[i]] <= ma_values[periods[i+1]]:
            is_aligned = False
            break
    
    # 检查均线上升趋势（当前值大于前一个值）
    is_rising = {}
    if len(df) >= 2:
        prev = df.iloc[-2]
        for period in periods:
            ma_col = f'MA{period}'
            if not pd.isna(prev[ma_col]) and not pd.isna(latest[ma_col]):
                is_rising[period] = latest[ma_col] > prev[ma_col]
            else:
                is_rising[period] = False
    
    return is_aligned, {
        'ma_values': ma_values,
        'is_rising': is_rising,
        'current_price': latest['close']
    }

def check_daily_ma_alignment(code, market):
    """检查日均线多头排列"""
    print(f"\n📊 获取 {STOCK_NAME}({code}) 日K线数据...")
    daily_df = get_kline_data(code, market, period='daily', days=250)
    
    if daily_df is None or len(daily_df) < max(DAILY_MA_PERIODS):
        print(f"❌ 日K线数据不足，无法计算均线")
        return False, None, None
    
    print(f"✅ 获取到 {len(daily_df)} 条日K线数据")
    print(f"   最新日期: {daily_df.iloc[-1]['date'].strftime('%Y-%m-%d')}")
    
    # 计算均线
    daily_df = calculate_ma(daily_df, DAILY_MA_PERIODS)
    
    # 检查多头排列
    is_aligned, details = check_ma_alignment(daily_df, DAILY_MA_PERIODS)
    
    return is_aligned, daily_df, details

def check_weekly_ma_alignment(code, market):
    """检查周均线多头排列"""
    print(f"\n📊 获取 {STOCK_NAME}({code}) 周K线数据...")
    weekly_df = get_kline_data(code, market, period='weekly', days=500)
    
    if weekly_df is None or len(weekly_df) < max(WEEKLY_MA_PERIODS):
        print(f"❌ 周K线数据不足，无法计算均线")
        return False, None, None
    
    print(f"✅ 获取到 {len(weekly_df)} 条周K线数据")
    print(f"   最新日期: {weekly_df.iloc[-1]['date'].strftime('%Y-%m-%d')}")
    
    # 计算均线
    weekly_df = calculate_ma(weekly_df, WEEKLY_MA_PERIODS)
    
    # 检查多头排列
    is_aligned, details = check_ma_alignment(weekly_df, WEEKLY_MA_PERIODS)
    
    return is_aligned, weekly_df, details

def check_minute_ma_alignment(code, period_minutes):
    """检查分钟级均线多头排列（30分钟、60分钟、120分钟）"""
    period_name = f"{period_minutes}分钟"
    print(f"\n📊 获取 {STOCK_NAME}({code}) {period_name}K线数据...")
    
    days = 30 if period_minutes == 30 else (60 if period_minutes == 60 else 120)
    minute_df = get_minute_kline_data(code, period_minutes=period_minutes, days=days)
    
    if minute_df is None or len(minute_df) < max(MINUTE_MA_PERIODS):
        print(f"❌ {period_name}K线数据不足，无法计算均线")
        return False, None, None
    
    print(f"✅ 获取到 {len(minute_df)} 条{period_name}K线数据")
    
    # 获取最新时间
    if '时间' in minute_df.columns:
        latest_time = minute_df.iloc[-1]['时间']
        if isinstance(latest_time, str):
            print(f"   最新时间: {latest_time}")
    elif 'date' in minute_df.columns:
        latest_date = minute_df.iloc[-1]['date']
        if hasattr(latest_date, 'strftime'):
            print(f"   最新日期: {latest_date.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"   最新日期: {latest_date}")
    
    # 计算均线
    minute_df = calculate_ma(minute_df, MINUTE_MA_PERIODS)
    
    # 检查多头排列
    is_aligned, details = check_ma_alignment(minute_df, MINUTE_MA_PERIODS)
    
    return is_aligned, minute_df, details

def format_ma_info(periods, details):
    """格式化均线信息"""
    if details is None:
        return "无数据"
    
    ma_values = details.get('ma_values', {})
    is_rising = details.get('is_rising', {})
    current_price = details.get('current_price', 0)
    
    info_lines = []
    info_lines.append(f"   当前价格: {current_price:.2f} 元")
    info_lines.append(f"   均线值:")
    for period in periods:
        ma_val = ma_values.get(period, 0)
        rising_symbol = "📈" if is_rising.get(period, False) else "📉"
        info_lines.append(f"     MA{period}: {ma_val:.2f} 元 {rising_symbol}")
    
    return "\n".join(info_lines)

# ==================== 主程序 ====================

def main():
    print(f"\n{'='*70}")
    print(f"开始判断 {STOCK_NAME}({STOCK_CODE}) 多周期均线多头排列")
    print(f"{'='*70}\n")
    
    # 检查30分钟均线多头排列
    min30_aligned, min30_df, min30_details = check_minute_ma_alignment(STOCK_CODE, 30)
    
    print(f"\n{'='*70}")
    print("📈 30分钟均线多头排列判断结果")
    print(f"{'='*70}")
    if min30_details:
        print(format_ma_info(MINUTE_MA_PERIODS, min30_details))
        print(f"\n   多头排列状态: {'✅ 满足' if min30_aligned else '❌ 不满足'}")
        if min30_aligned:
            print(f"   ✅ 30分钟均线多头排列：MA{MINUTE_MA_PERIODS[0]} > MA{MINUTE_MA_PERIODS[1]} > MA{MINUTE_MA_PERIODS[2]} > MA{MINUTE_MA_PERIODS[3]}")
        else:
            print(f"   ❌ 30分钟均线未形成多头排列")
    else:
        print("   ❌ 无法获取30分钟均线数据")
    
    # 检查60分钟均线多头排列
    min60_aligned, min60_df, min60_details = check_minute_ma_alignment(STOCK_CODE, 60)
    
    print(f"\n{'='*70}")
    print("📈 60分钟均线多头排列判断结果")
    print(f"{'='*70}")
    if min60_details:
        print(format_ma_info(MINUTE_MA_PERIODS, min60_details))
        print(f"\n   多头排列状态: {'✅ 满足' if min60_aligned else '❌ 不满足'}")
        if min60_aligned:
            print(f"   ✅ 60分钟均线多头排列：MA{MINUTE_MA_PERIODS[0]} > MA{MINUTE_MA_PERIODS[1]} > MA{MINUTE_MA_PERIODS[2]} > MA{MINUTE_MA_PERIODS[3]}")
        else:
            print(f"   ❌ 60分钟均线未形成多头排列")
    else:
        print("   ❌ 无法获取60分钟均线数据")
    
    # 检查120分钟均线多头排列
    min120_aligned, min120_df, min120_details = check_minute_ma_alignment(STOCK_CODE, 120)
    
    print(f"\n{'='*70}")
    print("📈 120分钟均线多头排列判断结果")
    print(f"{'='*70}")
    if min120_details:
        print(format_ma_info(MINUTE_MA_PERIODS, min120_details))
        print(f"\n   多头排列状态: {'✅ 满足' if min120_aligned else '❌ 不满足'}")
        if min120_aligned:
            print(f"   ✅ 120分钟均线多头排列：MA{MINUTE_MA_PERIODS[0]} > MA{MINUTE_MA_PERIODS[1]} > MA{MINUTE_MA_PERIODS[2]} > MA{MINUTE_MA_PERIODS[3]}")
        else:
            print(f"   ❌ 120分钟均线未形成多头排列")
    else:
        print("   ❌ 无法获取120分钟均线数据")
    
    # 检查日均线多头排列
    daily_aligned, daily_df, daily_details = check_daily_ma_alignment(STOCK_CODE, STOCK_MARKET)
    
    print(f"\n{'='*70}")
    print("📈 日均线多头排列判断结果")
    print(f"{'='*70}")
    if daily_details:
        print(format_ma_info(DAILY_MA_PERIODS, daily_details))
        print(f"\n   多头排列状态: {'✅ 满足' if daily_aligned else '❌ 不满足'}")
        if daily_aligned:
            print(f"   ✅ 日均线多头排列：MA{DAILY_MA_PERIODS[0]} > MA{DAILY_MA_PERIODS[1]} > MA{DAILY_MA_PERIODS[2]} > MA{DAILY_MA_PERIODS[3]}")
        else:
            print(f"   ❌ 日均线未形成多头排列")
    else:
        print("   ❌ 无法获取日均线数据")
    
    # 检查周均线多头排列
    weekly_aligned, weekly_df, weekly_details = check_weekly_ma_alignment(STOCK_CODE, STOCK_MARKET)
    
    print(f"\n{'='*70}")
    print("📈 周均线多头排列判断结果")
    print(f"{'='*70}")
    if weekly_details:
        print(format_ma_info(WEEKLY_MA_PERIODS, weekly_details))
        print(f"\n   多头排列状态: {'✅ 满足' if weekly_aligned else '❌ 不满足'}")
        if weekly_aligned:
            print(f"   ✅ 周均线多头排列：MA{WEEKLY_MA_PERIODS[0]} > MA{WEEKLY_MA_PERIODS[1]} > MA{WEEKLY_MA_PERIODS[2]}")
        else:
            print(f"   ❌ 周均线未形成多头排列")
    else:
        print("   ❌ 无法获取周均线数据")
    
    # 综合判断
    print(f"\n{'='*70}")
    print("📊 综合判断结果")
    print(f"{'='*70}")
    print(f"   30分钟均线多头排列: {'✅ 满足' if min30_aligned else '❌ 不满足'}")
    print(f"   60分钟均线多头排列: {'✅ 满足' if min60_aligned else '❌ 不满足'}")
    print(f"   120分钟均线多头排列: {'✅ 满足' if min120_aligned else '❌ 不满足'}")
    print(f"   日均线多头排列: {'✅ 满足' if daily_aligned else '❌ 不满足'}")
    print(f"   周均线多头排列: {'✅ 满足' if weekly_aligned else '❌ 不满足'}")
    
    # 统计满足条件的周期数
    satisfied_count = sum([
        min30_aligned if min30_details else False,
        min60_aligned if min60_details else False,
        min120_aligned if min120_details else False,
        daily_aligned if daily_details else False,
        weekly_aligned if weekly_details else False
    ])
    
    total_count = sum([
        1 if min30_details else 0,
        1 if min60_details else 0,
        1 if min120_details else 0,
        1 if daily_details else 0,
        1 if weekly_details else 0
    ])
    
    print(f"\n   满足条件: {satisfied_count}/{total_count} 个周期")
    
    if satisfied_count == total_count and total_count > 0:
        print(f"\n   🎉 {STOCK_NAME}({STOCK_CODE}) 所有周期均线均满足多头排列！")
    elif satisfied_count > 0:
        print(f"\n   ⚠️  {STOCK_NAME}({STOCK_CODE}) 部分周期满足多头排列")
    else:
        print(f"\n   ❌ {STOCK_NAME}({STOCK_CODE}) 所有周期均线均不满足多头排列")
    
    print(f"\n{'='*70}\n")
    
    # 关闭baostock连接
    if DATA_SOURCE == "baostock" and BAOSTOCK_AVAILABLE:
        try:
            with suppress_stdout():
                bs.logout()
        except:
            pass

if __name__ == "__main__":
    main()

