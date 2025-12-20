# -*- coding: utf-8 -*-
"""
测试获取纳斯达克指数涨跌幅功能
"""

import sys
from datetime import datetime

# 尝试导入akshare
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
    print("✅ akshare 库已导入")
except ImportError:
    AKSHARE_AVAILABLE = False
    print("❌ akshare 库未安装，请运行: pip install akshare")

def get_nasdaq_change():
    """
    获取纳斯达克指数最新涨跌幅
    
    Returns:
        dict: 包含纳指信息的字典，如果失败返回None
    """
    try:
        print("\n正在获取纳斯达克指数数据...")
        
        # 方法1: 优先使用yfinance获取纳指指数 (^IXIC)
        try:
            import yfinance as yf
            print("  尝试使用 yfinance 获取纳指数据...")
            nasdaq = yf.Ticker("^IXIC")
            hist = nasdaq.history(period="2d")
            if len(hist) > 0:
                latest = hist.iloc[-1]
                prev = hist.iloc[-2] if len(hist) > 1 else latest
                change_pct = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
                change = latest['Close'] - prev['Close']
                
                # 获取实时信息
                info = nasdaq.info
                current_price = info.get('regularMarketPrice', latest['Close'])
                
                return {
                    'index_name': '纳斯达克综合指数 (IXIC)',
                    'latest_price': round(current_price, 2),
                    'change_pct': round(change_pct, 2),
                    'change': round(change, 2),
                    'date': latest.name.strftime('%Y-%m-%d'),
                    'time': latest.name.strftime('%H:%M:%S') if hasattr(latest.name, 'strftime') else 'N/A',
                    'source': 'yfinance',
                    'prev_close': round(prev['Close'], 2),
                    'high': round(latest['High'], 2),
                    'low': round(latest['Low'], 2)
                }
        except ImportError:
            print("  yfinance 未安装，尝试其他方法...")
            print("  提示: 可以运行 'pip install yfinance' 安装")
        except Exception as e2:
            print(f"  yfinance 方法失败: {e2}")
        
        # 方法2: 使用akshare获取美股指数数据
        if AKSHARE_AVAILABLE:
            try:
                print("  尝试使用 akshare 获取美股指数数据...")
                # 获取美股实时行情，查找纳指相关ETF或指数
                us_spot = ak.stock_us_spot_em()
                
                # 查找纳指相关标的
                nasdaq_keywords = ['纳指', 'NASDAQ', 'QQQ', 'IXIC']
                nasdaq_stocks = us_spot[
                    us_spot['名称'].str.contains('|'.join(nasdaq_keywords), case=False, na=False)
                ]
                
                if len(nasdaq_stocks) > 0:
                    # 优先选择QQQ（纳指100 ETF）或直接包含"纳指"的标的
                    qqq = nasdaq_stocks[nasdaq_stocks['名称'].str.contains('QQQ|纳指100', case=False, na=False)]
                    if len(qqq) > 0:
                        latest = qqq.iloc[0]
                    else:
                        latest = nasdaq_stocks.iloc[0]
                    
                    return {
                        'index_name': latest.get('名称', '纳斯达克相关标的'),
                        'latest_price': latest.get('最新价', 'N/A'),
                        'change_pct': latest.get('涨跌幅', 'N/A'),
                        'change': latest.get('涨跌额', 'N/A'),
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'source': 'akshare_us_spot',
                        'note': '这是ETF或相关标的，非指数本身'
                    }
            except Exception as e3:
                print(f"  akshare 方法失败: {e3}")
        
        print("❌ 所有方法都失败了")
        print("  建议: 安装 yfinance 库以获得更准确的纳指数据")
        print("  命令: pip install yfinance")
        return None
        
    except Exception as e:
        print(f"❌ 获取纳指数据时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    print("=" * 70)
    print("纳斯达克指数涨跌幅获取测试")
    print("=" * 70)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    result = get_nasdaq_change()
    
    if result:
        print("\n" + "=" * 70)
        print("✅ 获取成功！")
        print("=" * 70)
        print(f"指数名称: {result.get('index_name', 'N/A')}")
        print(f"最新价格: {result.get('latest_price', 'N/A')}")
        print(f"涨跌幅: {result.get('change_pct', 'N/A')}%")
        print(f"涨跌额: {result.get('change', 'N/A')}")
        print(f"日期: {result.get('date', 'N/A')}")
        if 'time' in result:
            print(f"时间: {result.get('time', 'N/A')}")
        print(f"数据源: {result.get('source', 'N/A')}")
        
        # 显示额外信息（如果有）
        if 'prev_close' in result:
            print(f"前收盘价: {result.get('prev_close', 'N/A')}")
        if 'high' in result:
            print(f"最高价: {result.get('high', 'N/A')}")
        if 'low' in result:
            print(f"最低价: {result.get('low', 'N/A')}")
        if 'note' in result:
            print(f"备注: {result.get('note', '')}")
        print("=" * 70)
        
        # 格式化输出涨跌幅
        change_pct = result.get('change_pct', 0)
        try:
            change_pct_float = float(change_pct)
            if change_pct_float > 0:
                print(f"\n📈 纳指今日上涨 {abs(change_pct_float):.2f}%")
            elif change_pct_float < 0:
                print(f"\n📉 纳指今日下跌 {abs(change_pct_float):.2f}%")
            else:
                print(f"\n➡️ 纳指今日持平")
        except (ValueError, TypeError):
            print(f"\n涨跌幅数据: {change_pct}")
        
        return True
    else:
        print("\n" + "=" * 70)
        print("❌ 获取失败，请检查网络连接或数据源")
        print("=" * 70)
        print("\n建议:")
        print("1. 检查网络连接")
        print("2. 安装 yfinance: pip install yfinance")
        print("3. 确保 akshare 已正确安装: pip install akshare")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

