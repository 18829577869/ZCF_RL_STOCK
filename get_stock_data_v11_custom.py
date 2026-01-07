# get_stock_data_v11_custom.py - V11自定义股票数据获取
# -*- coding: utf-8 -*-
"""
V11 自定义股票数据获取：
1. 获取用户指定的10只股票数据
2. 包含有色金属、航天军工、基础化工、机械设备等多个领域
3. 初始资金2万元（匹配实盘操作）
4. 数据用于V11全功能集成版训练
"""
import baostock as bs
import pandas as pd
import os
from datetime import datetime

# 登录
lg = bs.login(user_id="anonymous", password="123456")
print("登录响应:", lg.error_code, lg.error_msg)

# V11 自定义股票列表（用户提供的10只股票）
stocks = [
    # === 有色金属板块 ===
    {"code": "sz.000408", "name": "藏格矿业", "start_date": "1996-06-28", 
     "category": "有色金属", "volatility": "中", "style": "平衡", "priority": "核心"},
    {"code": "sz.000893", "name": "亚钾国际", "start_date": "1999-01-19", 
     "category": "基础化工", "volatility": "中", "style": "平衡", "priority": "核心"},
    {"code": "sz.000933", "name": "神火股份", "start_date": "1999-08-31", 
     "category": "有色金属", "volatility": "中", "style": "平衡", "priority": "核心"},
    {"code": "sz.000807", "name": "云铝股份", "start_date": "1998-04-08", 
     "category": "有色金属", "volatility": "中", "style": "平衡", "priority": "核心"},
    {"code": "sz.000603", "name": "盛达资源", "start_date": "1996-08-23", 
     "category": "有色金属", "volatility": "中", "style": "平衡", "priority": "核心"},
    {"code": "sh.600497", "name": "驰宏锌锗", "start_date": "2004-04-20", 
     "category": "有色金属", "volatility": "中", "style": "平衡", "priority": "核心"},
    {"code": "sh.600219", "name": "南山铝业", "start_date": "1999-12-23", 
     "category": "有色金属", "volatility": "中", "style": "平衡", "priority": "核心"},
    
    # === 航天军工板块 ===
    {"code": "sh.600118", "name": "中国卫星", "start_date": "1997-09-08", 
     "category": "航天军工", "volatility": "中", "style": "平衡", "priority": "核心"},
    {"code": "sh.600879", "name": "航天电子", "start_date": "1995-11-15", 
     "category": "航天军工", "volatility": "中", "style": "平衡", "priority": "核心"},
    
    # === 机械设备板块 ===
    {"code": "sh.600577", "name": "精达股份", "start_date": "2002-09-11", 
     "category": "机械设备", "volatility": "中", "style": "平衡", "priority": "核心"},
]

print("="*70)
print("V11 自定义股票数据获取 - 数据获取")
print("="*70)
print(f"总共 {len(stocks)} 只标的")
print(f"  核心标的: {len([s for s in stocks if s.get('priority') == '核心'])}只")

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
    priority = stock.get("priority", "核心")
    
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
        
        # 保存到专用目录
        os.makedirs('stockdata_v11_custom/train', exist_ok=True)
        os.makedirs('stockdata_v11_custom/test', exist_ok=True)
        
        train_file = f'stockdata_v11_custom/train/{code}.{name}.csv'
        test_file = f'stockdata_v11_custom/test/{code}.{name}.csv'
        
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
    metadata_df.to_csv('stockdata_v11_custom/metadata_v11_custom.csv', index=False, encoding='utf-8-sig')
    print(f"\n元数据已保存: stockdata_v11_custom/metadata_v11_custom.csv")
    
    print("\n[完成] 可以开始训练：")
    print("  python train_v11_custom.py")
    print("\n[说明] 本版本包含10只自定义股票，初始资金2万元")
    print("[说明] 训练后的模型可用于V11全功能集成版实时预测")
else:
    print("\n[失败] 没有成功下载任何数据")

bs.logout()

