# -*- coding: utf-8 -*-
"""
获取全球主要指数涨跌幅功能
支持：纳斯达克、道琼斯、富时A50期指连续
"""

import sys
import json
import os
from datetime import datetime

# 尝试导入akshare
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
    print("✅ akshare 库已导入")
except ImportError:
    AKSHARE_AVAILABLE = False
    print("❌ akshare 库未安装，请运行: pip install akshare")

def save_index_data_to_file(data):
    """
    将指数数据保存到JSON文件
    
    Args:
        data: 指数数据字典（包含多个指数）
    """
    try:
        # 获取脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'index_data.json')
        
        # 保存数据到文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        # 保存失败不影响主流程
        pass


def validate_index_data(data, index_type):
    """
    验证指数数据是否合理
    
    Args:
        data: 指数数据字典
        index_type: 指数类型 ('nasdaq', 'dow', 'a50')
    
    Returns:
        bool: 数据是否合理
    """
    if not data:
        return False
    
    change_pct = data.get('change_pct')
    if change_pct is None or change_pct == 'N/A':
        return False
    
    # 尝试转换为数字
    try:
        if isinstance(change_pct, str):
            # 移除百分号
            change_pct = float(change_pct.replace('%', ''))
        else:
            change_pct = float(change_pct)
    except (ValueError, TypeError):
        return False
    
    # 指数涨跌幅通常在-15%到+15%之间，超过这个范围可能是异常数据
    if abs(change_pct) > 15:
        return False
    
    return True


def get_single_index_data(index_type, keywords, yfinance_symbol=None):
    """
    获取单个指数数据
    
    Args:
        index_type: 指数类型 ('nasdaq', 'dow', 'a50')
        keywords: 搜索关键词列表
        yfinance_symbol: yfinance符号（如 '^IXIC', '^DJI'）
    
    Returns:
        dict: 指数数据字典，如果失败返回None
    """
    result = None
    
    # 方法1: 优先使用yfinance获取指数数据（当日涨跌幅，非累计）
    if yfinance_symbol:
        try:
            import yfinance as yf
            ticker = yf.Ticker(yfinance_symbol)
            
            # 优先使用info获取当日实时涨跌幅（非累计）
            try:
                info = ticker.info
                # 获取当日涨跌幅（regularMarketChangePercent是当日涨跌幅百分比）
                change_pct = info.get('regularMarketChangePercent')
                # 获取当日涨跌额
                change = info.get('regularMarketChange')
                # 获取最新价格
                current_price = info.get('regularMarketPrice')
                # 获取前收盘价
                prev_close = info.get('previousClose')
                # 获取当日最高价
                high = info.get('regularMarketDayHigh')
                # 获取当日最低价
                low = info.get('regularMarketDayLow')
                
                # 如果info中有当日涨跌幅数据，优先使用
                if change_pct is not None and current_price is not None:
                    index_names = {
                        'nasdaq': '纳斯达克综合指数 (IXIC)',
                        'dow': '道琼斯工业平均指数 (DJI)',
                        'a50': '富时A50期指连续'
                    }
                    
                    result = {
                        'index_name': index_names.get(index_type, '指数'),
                        'latest_price': round(current_price, 2) if current_price else 'N/A',
                        'change_pct': round(change_pct, 2) if change_pct is not None else 'N/A',
                        'change': round(change, 2) if change is not None else 'N/A',
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'source': 'yfinance_info',
                        'prev_close': round(prev_close, 2) if prev_close else 'N/A',
                        'high': round(high, 2) if high else 'N/A',
                        'low': round(low, 2) if low else 'N/A',
                        'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    # 验证数据合理性
                    if validate_index_data(result, index_type):
                        return result
            except:
                pass
            
            # 如果info获取失败，使用history计算当日涨跌幅
            hist = ticker.history(period="2d")
            if len(hist) > 0:
                latest = hist.iloc[-1]
                prev = hist.iloc[-2] if len(hist) > 1 else latest
                # 计算当日涨跌幅（最新收盘价相对于前收盘价）
                change_pct = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
                change = latest['Close'] - prev['Close']
                
                # 获取实时信息
                try:
                    info = ticker.info
                    current_price = info.get('regularMarketPrice', latest['Close'])
                except:
                    current_price = latest['Close']
                
                index_names = {
                    'nasdaq': '纳斯达克综合指数 (IXIC)',
                    'dow': '道琼斯工业平均指数 (DJI)',
                    'a50': '富时A50期指连续'
                }
                
                result = {
                    'index_name': index_names.get(index_type, '指数'),
                    'latest_price': round(current_price, 2),
                    'change_pct': round(change_pct, 2),
                    'change': round(change, 2),
                    'date': latest.name.strftime('%Y-%m-%d') if hasattr(latest.name, 'strftime') else datetime.now().strftime('%Y-%m-%d'),
                    'time': latest.name.strftime('%H:%M:%S') if hasattr(latest.name, 'strftime') else 'N/A',
                    'source': 'yfinance_hist',
                    'prev_close': round(prev['Close'], 2),
                    'high': round(latest['High'], 2),
                    'low': round(latest['Low'], 2),
                    'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                # 验证数据合理性
                if validate_index_data(result, index_type):
                    return result
                else:
                    # 数据异常，继续尝试其他方法
                    pass
        except ImportError:
            pass
        except Exception as e:
            pass
    
    # 方法2: 使用akshare获取指数数据
    if AKSHARE_AVAILABLE:
        try:
            # 获取美股实时行情
            us_spot = ak.stock_us_spot_em()
            
            if us_spot is not None and len(us_spot) > 0:
                # 查找相关标的
                index_stocks = us_spot[
                    us_spot['名称'].str.contains('|'.join(keywords), case=False, na=False)
                ]
                
                if len(index_stocks) > 0:
                    # 优先选择ETF（如QQQ、DIA），而不是个股
                    # 对于纳斯达克，优先选择QQQ
                    if index_type == 'nasdaq':
                        qqq_stocks = index_stocks[index_stocks['名称'].str.contains('QQQ|纳指100', case=False, na=False)]
                        if len(qqq_stocks) > 0:
                            index_stocks = qqq_stocks
                    # 对于道琼斯，优先选择DIA
                    elif index_type == 'dow':
                        dia_stocks = index_stocks[index_stocks['名称'].str.contains('DIA|道指ETF', case=False, na=False)]
                        if len(dia_stocks) > 0:
                            index_stocks = dia_stocks
                    
                    # 遍历所有匹配的标的，找到第一个数据合理的
                    for idx in range(len(index_stocks)):
                        latest = index_stocks.iloc[idx]
                        
                        index_names = {
                            'nasdaq': '纳斯达克相关标的',
                            'dow': '道琼斯相关标的',
                            'a50': '富时A50相关标的'
                        }
                        
                        # akshare的涨跌幅字段通常是当日涨跌幅（非累计）
                        change_pct = latest.get('涨跌幅', 'N/A')
                        # 确保是数字类型，如果是字符串，尝试转换
                        if isinstance(change_pct, str):
                            try:
                                # 移除百分号并转换为浮点数
                                change_pct = float(change_pct.replace('%', ''))
                            except:
                                pass
                        
                        result = {
                            'index_name': latest.get('名称', index_names.get(index_type, '指数相关标的')),
                            'latest_price': latest.get('最新价', 'N/A'),
                            'change_pct': change_pct,  # 当日涨跌幅（非累计）
                            'change': latest.get('涨跌额', 'N/A'),
                            'date': datetime.now().strftime('%Y-%m-%d'),
                            'source': 'akshare_us_spot',
                            'note': '这是ETF或相关标的，非指数本身，涨跌幅为当日数据',
                            'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        
                        # 验证数据合理性
                        if validate_index_data(result, index_type):
                            return result
                    
                    # 如果所有匹配的标的数据都不合理，返回None
                    return None
        except Exception as e:
            pass
    
    # 方法3: 对于A50，尝试使用akshare的期货数据
    if index_type == 'a50' and AKSHARE_AVAILABLE:
        try:
            # 尝试获取富时A50期指数据
            futures_data = ak.futures_main_sina()
            if futures_data is not None and len(futures_data) > 0:
                # 查找A50相关合约
                a50_data = futures_data[futures_data['symbol'].str.contains('A50|富时', case=False, na=False)]
                if len(a50_data) > 0:
                    latest = a50_data.iloc[0]
                    result = {
                        'index_name': latest.get('symbol', '富时A50期指连续'),
                        'latest_price': latest.get('price', 'N/A'),
                        'change_pct': latest.get('changepercent', 'N/A'),
                        'change': latest.get('change', 'N/A'),
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'source': 'akshare_futures',
                        'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    return result
        except Exception as e:
            pass
    
    return None


def get_index_data():
    """
    获取全球主要指数最新涨跌幅
    包括：纳斯达克、道琼斯、富时A50期指连续
    
    Returns:
        dict: 包含所有指数信息的字典，格式为：
        {
            'nasdaq': {...},  # 纳斯达克
            'dow': {...},     # 道琼斯
            'a50': {...}      # 富时A50
        }
    """
    result = {
        'nasdaq': None,
        'dow': None,
        'a50': None,
        'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    try:
        print("\n正在获取全球主要指数数据...")
        
        # 获取纳斯达克指数
        print("  1. 获取纳斯达克指数...")
        nasdaq_data = get_single_index_data(
            'nasdaq',
            ['纳指', 'NASDAQ', 'QQQ', 'IXIC'],
            '^IXIC'
        )
        if nasdaq_data:
            result['nasdaq'] = nasdaq_data
            print(f"     ✅ 纳斯达克: {nasdaq_data.get('change_pct', 'N/A')}%")
        else:
            print(f"     ❌ 纳斯达克获取失败")
        
        # 获取道琼斯指数
        print("  2. 获取道琼斯指数...")
        dow_data = get_single_index_data(
            'dow',
            ['道指', '道琼斯', 'DOW', 'DJI', 'DIA'],
            '^DJI'
        )
        if dow_data:
            result['dow'] = dow_data
            print(f"     ✅ 道琼斯: {dow_data.get('change_pct', 'N/A')}%")
        else:
            print(f"     ❌ 道琼斯获取失败")
        
        # 获取富时A50期指连续
        print("  3. 获取富时A50期指连续...")
        a50_data = get_single_index_data(
            'a50',
            ['A50', '富时', 'FTSE'],
            None  # yfinance可能不支持A50
        )
        if a50_data:
            result['a50'] = a50_data
            print(f"     ✅ 富时A50: {a50_data.get('change_pct', 'N/A')}%")
        else:
            print(f"     ❌ 富时A50获取失败")
        
        # 保存结果到文件
        save_index_data_to_file(result)
        
        # 如果至少有一个指数获取成功，返回结果
        if result['nasdaq'] or result['dow'] or result['a50']:
            return result
        else:
            return None
        
    except Exception as e:
        print(f"❌ 获取指数数据时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None


# 保持向后兼容的函数名
def get_nasdaq_change():
    """
    获取纳斯达克指数最新涨跌幅（向后兼容函数）
    
    Returns:
        dict: 包含纳指信息的字典，如果失败返回None
    """
    index_data = get_index_data()
    if index_data and index_data.get('nasdaq'):
        return index_data['nasdaq']
    return None


def main():
    """主函数"""
    print("=" * 70)
    print("全球主要指数涨跌幅获取测试")
    print("=" * 70)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    result = get_index_data()
    
    if result:
        print("\n" + "=" * 70)
        print("✅ 获取成功！")
        print("=" * 70)
        
        # 显示纳斯达克
        if result.get('nasdaq'):
            nasdaq = result['nasdaq']
            print(f"\n📈 纳斯达克指数:")
            print(f"   指数名称: {nasdaq.get('index_name', 'N/A')}")
            print(f"   最新价格: {nasdaq.get('latest_price', 'N/A')}")
            print(f"   涨跌幅: {nasdaq.get('change_pct', 'N/A')}%")
            print(f"   涨跌额: {nasdaq.get('change', 'N/A')}")
            print(f"   日期: {nasdaq.get('date', 'N/A')}")
            if nasdaq.get('time'):
                print(f"   时间: {nasdaq.get('time', 'N/A')}")
            print(f"   数据源: {nasdaq.get('source', 'N/A')}")
        
        # 显示道琼斯
        if result.get('dow'):
            dow = result['dow']
            print(f"\n📈 道琼斯指数:")
            print(f"   指数名称: {dow.get('index_name', 'N/A')}")
            print(f"   最新价格: {dow.get('latest_price', 'N/A')}")
            print(f"   涨跌幅: {dow.get('change_pct', 'N/A')}%")
            print(f"   涨跌额: {dow.get('change', 'N/A')}")
            print(f"   日期: {dow.get('date', 'N/A')}")
            if dow.get('time'):
                print(f"   时间: {dow.get('time', 'N/A')}")
            print(f"   数据源: {dow.get('source', 'N/A')}")
        
        # 显示富时A50
        if result.get('a50'):
            a50 = result['a50']
            print(f"\n📈 富时A50期指连续:")
            print(f"   指数名称: {a50.get('index_name', 'N/A')}")
            print(f"   最新价格: {a50.get('latest_price', 'N/A')}")
            print(f"   涨跌幅: {a50.get('change_pct', 'N/A')}%")
            print(f"   涨跌额: {a50.get('change', 'N/A')}")
            print(f"   日期: {a50.get('date', 'N/A')}")
            print(f"   数据源: {a50.get('source', 'N/A')}")
        
        print("\n" + "=" * 70)
        
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
