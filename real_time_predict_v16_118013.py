"""
V16 实时预测系统 - 全功能集成版（道通转债118013专用，日线「准实时」）
说明：
- 使用 V16 全套多模型 / 风险控制 / DeepSeek 复盘框架
- 但行情数据部分对 118013 做特殊处理：使用 akshare 可转债日线 (bond_zh_hs_daily)
- 每轮预测基于最近一段日线数据，属于按日更新的「准实时」预测
"""

from real_time_predict_v16_603698 import *  # 复用完整V16实现
import datetime as _dt

# ==================== 道通转债118013 专用配置覆盖 ====================

MODEL_PATH = "ppo_stock_v7_118013.zip"  # 使用道通转债专用PPO模型
STOCK_CODE = 'sh.118013'                # 可转债代码

def get_stock_name(code):
    """根据股票代码获取股票名称（在原有映射基础上增加118013）"""
    base_name = {
        'sh.118013': '道通转债',
    }
    if code in base_name:
        return base_name[code]
    # 回退到原脚本里的映射（如果有）
    try:
        from real_time_predict_v16_603698 import get_stock_name as _orig_get_stock_name
        return _orig_get_stock_name(code)
    except Exception:
        return code

def fetch_bond_daily_quasi_realtime(stock_code, days=365):
    """
    使用 AkShare 获取可转债日线数据，作为「准实时」行情：
    - 每次循环重新获取最近 N 天日线
    - 返回的 DataFrame 至少包含: date, time, close, volume
    """
    try:
        import akshare as ak
        # 将 sh.118013 转成 akshare 可转债代码，如 sh118013
        symbol = stock_code.replace('.', '')
        df = ak.bond_zh_hs_daily(symbol=symbol)
        if df is None or len(df) == 0:
            return None

        # 标准化列名
        col_map = {
            '日期': 'date',
            '收盘': 'close',
            '成交量': 'volume'
        }
        for old, new in col_map.items():
            if old in df.columns:
                df = df.rename(columns={old: new})

        if 'date' not in df.columns or 'close' not in df.columns:
            return None

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # 只保留最近 N 天
        end_date = _dt.date.today()
        start_date = end_date - _dt.timedelta(days=days)
        df = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]
        if len(df) == 0:
            return None

        # 构造 time 列（用收盘日期 + 固定时间表示）
        df['time'] = df['date'].dt.strftime('%Y%m%d') + '150000'

        # 确保 volume 存在
        if 'volume' not in df.columns:
            df['volume'] = 0.0

        return df[['date', 'time', 'close', 'volume']]
    except Exception:
        return None

def fetch_akshare_5min(code_info, days=7):
    """
    覆盖原来的5分钟K数据获取：
    - 对于道通转债118013，直接走日线bond接口
    - 其他标的（如果以后复用本脚本）回退到原始实现
    """
    if code_info.get('baostock') == 'sh.118013' or code_info.get('tushare') == '118013.SH':
        df = fetch_bond_daily_quasi_realtime(STOCK_CODE, days=365)
        if df is not None and len(df) > 0:
            latest_date = df['date'].iloc[-1]
            print(f"   📊 数据来源: akshare 可转债日线 (bond_zh_hs_daily)")
            print(f"   📅 最新日线日期: {latest_date.strftime('%Y-%m-%d')}")
            print("   💡 说明: 可转债不支持5分钟级别实时数据，当前为按日线更新的『准实时』预测")
        return df

    # 回退到原始5分钟逻辑（用于其他标的复用时）
    from real_time_predict_v16_603698 import fetch_akshare_5min as _orig_fetch_akshare_5min
    return _orig_fetch_akshare_5min(code_info, days=days)


