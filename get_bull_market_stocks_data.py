"""
获取历史牛市牛股数据脚本
用于V20模型训练，提高模型在牛市中的表现

历史牛市牛股列表：
- 2007年牛市：云南铜业、中国船舶、中国神华、中国平安、招商银行等
- 2015年牛市：东方财富、同花顺、中国中车、中国中铁等
- 2019-2021年牛市：宁德时代、比亚迪、贵州茅台、五粮液等
"""

import baostock as bs
import pandas as pd
import os
from datetime import datetime

# 登录baostock
lg = bs.login(user_id="anonymous", password="123456")
print("登录响应:", lg.error_code, lg.error_msg)

# 历史牛市牛股列表（包含代码、名称、上市日期、牛市时期）
BULL_MARKET_STOCKS = [
    # 2007年牛市牛股
    {"code": "sz.000878", "name": "云南铜业", "start_date": "1998-06-02", "bull_market": "2007"},
    {"code": "sh.600150", "name": "中国船舶", "start_date": "1998-05-20", "bull_market": "2007"},
    {"code": "sh.601088", "name": "中国神华", "start_date": "2007-10-09", "bull_market": "2007"},
    {"code": "sh.601318", "name": "中国平安", "start_date": "2007-03-01", "bull_market": "2007"},
    {"code": "sh.600036", "name": "招商银行", "start_date": "2002-04-09", "bull_market": "2007"},
    {"code": "sh.600519", "name": "贵州茅台", "start_date": "2001-08-27", "bull_market": "2007"},
    {"code": "sh.600000", "name": "浦发银行", "start_date": "1999-11-10", "bull_market": "2007"},
    
    # 2015年牛市牛股
    {"code": "sz.300059", "name": "东方财富", "start_date": "2010-03-19", "bull_market": "2015"},
    {"code": "sz.300033", "name": "同花顺", "start_date": "2009-12-25", "bull_market": "2015"},
    {"code": "sh.601766", "name": "中国中车", "start_date": "2008-08-18", "bull_market": "2015"},
    {"code": "sh.601390", "name": "中国中铁", "start_date": "2007-12-03", "bull_market": "2015"},
    {"code": "sz.000002", "name": "万科A", "start_date": "1991-01-29", "bull_market": "2015"},
    {"code": "sh.600104", "name": "上汽集团", "start_date": "1997-11-25", "bull_market": "2015"},
    
    # 2019-2021年牛市牛股
    {"code": "sz.300750", "name": "宁德时代", "start_date": "2018-06-11", "bull_market": "2019-2021"},
    {"code": "sz.002594", "name": "比亚迪", "start_date": "2011-06-30", "bull_market": "2019-2021"},
    {"code": "sh.600519", "name": "贵州茅台", "start_date": "2001-08-27", "bull_market": "2019-2021"},
    {"code": "sz.000858", "name": "五粮液", "start_date": "1998-04-27", "bull_market": "2019-2021"},
    {"code": "sh.600276", "name": "恒瑞医药", "start_date": "2000-10-18", "bull_market": "2019-2021"},
    {"code": "sz.000001", "name": "平安银行", "start_date": "1991-04-03", "bull_market": "2019-2021"},
    {"code": "sh.600036", "name": "招商银行", "start_date": "2002-04-09", "bull_market": "2019-2021"},
    {"code": "sh.600887", "name": "伊利股份", "start_date": "1996-03-12", "bull_market": "2019-2021"},
    {"code": "sh.600009", "name": "上海机场", "start_date": "1998-02-18", "bull_market": "2019-2021"},
    {"code": "sh.600585", "name": "海螺水泥", "start_date": "2002-02-07", "bull_market": "2019-2021"},
]

# 创建目录
os.makedirs('stockdata_v20_bull_market/train', exist_ok=True)
os.makedirs('stockdata_v20_bull_market/test', exist_ok=True)

success_count = 0
fail_count = 0

print(f"\n开始获取 {len(BULL_MARKET_STOCKS)} 只牛市牛股数据...")
print("=" * 70)

for stock in BULL_MARKET_STOCKS:
    code = stock["code"]
    name = stock["name"]
    start_date = stock["start_date"]
    bull_market = stock["bull_market"]
    
    print(f"\n[{bull_market}年牛市] 查询 {code} ({name}), 起始: {start_date}")
    
    try:
        rs = bs.query_history_k_data_plus(
            code,
            "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST",
            start_date=start_date, 
            end_date=datetime.now().strftime('%Y-%m-%d'),
            frequency="d", 
            adjustflag="3"
        )
        
        if rs.error_code != '0':
            print(f"  [失败] 查询错误: {rs.error_msg}")
            fail_count += 1
            continue
        
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        
        if len(data_list) == 0:
            print(f"  [警告] 无数据，跳过")
            fail_count += 1
            continue
        
        result = pd.DataFrame(data_list, columns=rs.fields)
        print(f"  [成功] 获取 {len(result)} 条数据")
        
        # 数据预处理
        result['date'] = pd.to_datetime(result['date'])
        result = result.sort_values('date')
        
        # 分割训练集和测试集
        train_data = result[result['date'] <= '2024-12-31']
        test_data = result[result['date'] > '2024-12-31']
        
        if len(train_data) < 100:
            print(f"  [警告] 训练数据不足100条，跳过")
            fail_count += 1
            continue
        
        # 保存
        train_file = f'stockdata_v20_bull_market/train/{code}.{name}.csv'
        test_file = f'stockdata_v20_bull_market/test/{code}.{name}.csv'
        
        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)
        
        print(f"  [保存] 训练: {len(train_data)} | 测试: {len(test_data)}")
        success_count += 1
        
    except Exception as e:
        print(f"  [失败] 异常: {e}")
        fail_count += 1
        continue

print("\n" + "=" * 70)
print(f"✅ 完成！成功: {success_count}, 失败: {fail_count}")
print("=" * 70)

bs.logout()







