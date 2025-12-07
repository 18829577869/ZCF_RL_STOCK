# get_stock_data_v11_commercial_space.py - V11商业航天板块专用版
# -*- coding: utf-8 -*-
"""
V11 商业航天板块专用版：
1. 专门针对商业航天板块进行数据获取和训练
2. 包含商业航天板块主要股票
3. 初始资金2万元（匹配实盘操作）
4. 数据用于V11全功能集成版训练和评估
"""
import baostock as bs
import pandas as pd
import os
from datetime import datetime

# 登录
lg = bs.login(user_id="anonymous", password="123456")
print("登录响应:", lg.error_code, lg.error_msg)

# V11 商业航天板块股票列表
# 包含商业航天板块主要股票
stocks = [
    # === 商业航天核心标的 ===
    {"code": "sh.600118", "name": "中国卫星", "start_date": "1997-09-08", 
     "category": "商业航天", "volatility": "中", "style": "平衡", "priority": "核心"},
    {"code": "sz.002025", "name": "航天电器", "start_date": "2004-07-26", 
     "category": "商业航天", "volatility": "中", "style": "平衡", "priority": "核心"},
    {"code": "sh.688066", "name": "航天宏图", "start_date": "2019-07-22", 
     "category": "商业航天", "volatility": "高", "style": "激进", "priority": "核心"},
    
    # === 商业航天相关标的 ===
    {"code": "sz.001208", "name": "华菱线缆", "start_date": "2021-06-24", 
     "category": "商业航天", "volatility": "中", "style": "平衡", "priority": "相关"},
    {"code": "sh.688102", "name": "斯瑞新材", "start_date": "2021-03-08", 
     "category": "商业航天", "volatility": "高", "style": "激进", "priority": "相关"},
    {"code": "sz.300424", "name": "航新科技", "start_date": "2015-04-22", 
     "category": "商业航天", "volatility": "中", "style": "平衡", "priority": "相关"},
    {"code": "sh.600879", "name": "航天电子", "start_date": "1995-11-15", 
     "category": "商业航天", "volatility": "中", "style": "平衡", "priority": "相关"},
    {"code": "sh.600855", "name": "航天长峰", "start_date": "1994-04-25", 
     "category": "商业航天", "volatility": "中", "style": "平衡", "priority": "相关"},
    {"code": "sz.300456", "name": "赛微电子", "start_date": "2015-05-14", 
     "category": "商业航天", "volatility": "高", "style": "激进", "priority": "相关"},
    {"code": "sz.002151", "name": "北斗星通", "start_date": "2007-08-13", 
     "category": "商业航天", "volatility": "中", "style": "平衡", "priority": "相关"},
    {"code": "sh.603678", "name": "火炬电子", "start_date": "2015-01-26", 
     "category": "商业航天", "volatility": "中", "style": "平衡", "priority": "相关"},
    {"code": "sh.603267", "name": "鸿远电子", "start_date": "2019-05-15", 
     "category": "商业航天", "volatility": "中", "style": "平衡", "priority": "相关"},
    
    # === 航空航天产业链相关 ===
    {"code": "sh.600893", "name": "航发动力", "start_date": "1996-04-08", 
     "category": "航空航天", "volatility": "中", "style": "平衡", "priority": "产业链"},
    {"code": "sz.000768", "name": "中航西飞", "start_date": "1997-06-26", 
     "category": "航空航天", "volatility": "中", "style": "平衡", "priority": "产业链"},
    {"code": "sz.002013", "name": "中航机电", "start_date": "2004-07-05", 
     "category": "航空航天", "volatility": "中", "style": "平衡", "priority": "产业链"},
]

print("="*70)
print("V11 商业航天板块专用版 - 数据获取")
print("="*70)
print(f"总共 {len(stocks)} 只标的")
print(f"  核心标的: {len([s for s in stocks if s.get('priority') == '核心'])}只")
print(f"  相关股票: {len([s for s in stocks if s.get('priority') == '相关'])}只")
print(f"  产业链: {len([s for s in stocks if s.get('priority') == '产业链'])}只")

# 按分类统计
from collections import Counter
category_count = Counter([s['category'] for s in stocks])
print(f"\n按类别分布:")
for cat, count in category_count.items():
    print(f"  - {cat}: {count}只")

# 按风格统计
style_count = Counter([s['style'] for s in stocks])
print(f"\n按风格分布:")
for style, count in style_count.items():
    print(f"  - {style}: {count}只")

print("\n" + "="*70)
print("开始下载数据...")
print("="*70 + "\n")

success_count = 0
fail_count = 0
failed_stocks = []

for stock in stocks:
    code = stock["code"]
    name = stock["name"]
    start_date = stock["start_date"]
    category = stock["category"]
    priority = stock.get("priority", "相关")
    
    print(f"[{category}|{priority}] 查询 {code} ({name}), 起始: {start_date}")
    
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
            failed_stocks.append(f"{code} {name}")
            continue
        
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        
        if len(data_list) == 0:
            print(f"  [警告] 无数据，跳过")
            fail_count += 1
            failed_stocks.append(f"{code} {name}")
            continue
        
        result = pd.DataFrame(data_list, columns=rs.fields)
        print(f"  [成功] 获取 {len(result)} 条数据")
        
        # 数据预处理
        result['date'] = pd.to_datetime(result['date'])
        result = result.sort_values('date')
        
        # 分割训练集和测试集（以2024-12-31为分界）
        train_data = result[result['date'] <= '2024-12-31']
        test_data = result[result['date'] > '2024-12-31']
        
        if len(train_data) < 100:
            print(f"  [警告] 训练数据不足100条，跳过")
            fail_count += 1
            failed_stocks.append(f"{code} {name}")
            continue
        
        # 保存到专用目录（V11可以使用V7的数据格式）
        os.makedirs('stockdata_v7_commercial_space/train', exist_ok=True)
        os.makedirs('stockdata_v7_commercial_space/test', exist_ok=True)
        
        train_file = f'stockdata_v7_commercial_space/train/{code}.{name}.csv'
        test_file = f'stockdata_v7_commercial_space/test/{code}.{name}.csv'
        
        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)
        
        print(f"  [保存] 训练: {len(train_data)} | 测试: {len(test_data)}")
        success_count += 1
        
    except Exception as e:
        print(f"  [错误] {e}")
        fail_count += 1
        failed_stocks.append(f"{code} {name}")
    
    print()

print("="*70)
print("下载完成")
print("="*70)
print(f"成功: {success_count} 只")
print(f"失败: {fail_count} 只")

if failed_stocks:
    print(f"\n失败的股票:")
    for stock in failed_stocks:
        print(f"  - {stock}")

if success_count >= 8:
    print(f"\n[优秀] 成功{success_count}只，足够训练！")
elif success_count >= 5:
    print(f"\n[良好] 成功{success_count}只，可以训练")
else:
    print(f"\n[警告] 仅成功{success_count}只，建议至少5只")

if success_count > 0:
    # 保存元数据
    metadata_df = pd.DataFrame(stocks)
    metadata_df.to_csv('stockdata_v7_commercial_space/metadata_v7_commercial_space.csv', index=False, encoding='utf-8-sig')
    print(f"\n元数据已保存: stockdata_v7_commercial_space/metadata_v7_commercial_space.csv")
    
    print("\n[完成] 可以开始评估：")
    print("  python evaluate_commercial_space_v11.py")
    print("\n[说明] 本版本专门针对商业航天板块优化，初始资金2万元")
    print("[说明] 可以使用V11模型进行评估，筛选出收益率和夏普比率高的股票")
else:
    print("\n[失败] 没有成功下载任何数据")

bs.logout()

