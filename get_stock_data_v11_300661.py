# get_stock_data_v11_300661.py - V11圣邦股份专用版
# -*- coding: utf-8 -*-
"""
V11 圣邦股份专用版：
1. 专门针对圣邦股份(sz.300661)进行数据获取和训练
2. 包含相关股票，确保更好的针对性
3. 初始资金10万元
4. 数据用于V11全功能集成版训练
"""
import baostock as bs
import pandas as pd
import os
from datetime import datetime

# 登录
lg = bs.login(user_id="anonymous", password="123456")
print("登录响应:", lg.error_code, lg.error_msg)

# V11 圣邦股份专用股票列表
stocks = [
    # === 核心标的：圣邦股份300661 ===
    {"code": "sz.300661", "name": "圣邦股份", "start_date": "2017-06-06", 
     "category": "半导体", "volatility": "中", "style": "成长", "priority": "核心"},
    
    # === 半导体/芯片板块（相关股票）===
    {"code": "sz.002049", "name": "紫光国微", "start_date": "2005-06-06", 
     "category": "半导体", "volatility": "中", "style": "成长", "priority": "相关"},
    {"code": "sz.002371", "name": "北方华创", "start_date": "2010-03-16", 
     "category": "半导体设备", "volatility": "中", "style": "成长", "priority": "相关"},
    {"code": "sz.300782", "name": "卓胜微", "start_date": "2019-06-18", 
     "category": "射频芯片", "volatility": "中", "style": "成长", "priority": "相关"},
    {"code": "sh.688981", "name": "中芯国际", "start_date": "2020-07-16", 
     "category": "半导体制造", "volatility": "中", "style": "成长", "priority": "相关"},
    
    # === 电子元件/消费电子板块（相关股票）===
    {"code": "sz.002241", "name": "歌尔股份", "start_date": "2008-05-22", 
     "category": "消费电子", "volatility": "中", "style": "成长", "priority": "相关"},
    {"code": "sz.002475", "name": "立讯精密", "start_date": "2010-09-15", 
     "category": "消费电子", "volatility": "中", "style": "成长", "priority": "相关"},
    {"code": "sz.300726", "name": "宏达电子", "start_date": "2017-11-21", 
     "category": "电子元件", "volatility": "中", "style": "成长", "priority": "相关"},
    
    # === 科技板块（配置股票）===
    {"code": "sz.002837", "name": "英维克", "start_date": "2016-12-29", 
     "category": "数据中心", "volatility": "中", "style": "成长", "priority": "配置"},
    {"code": "sz.002851", "name": "麦格米特", "start_date": "2017-03-06", 
     "category": "电力设备", "volatility": "中", "style": "成长", "priority": "配置"},
    
    # === 金融板块（稳健配置）===
    {"code": "sh.600036", "name": "招商银行", "start_date": "2002-04-09", 
     "category": "金融", "volatility": "低", "style": "稳健", "priority": "配置"},
]

print("="*70)
print("V11 圣邦股份专用版 - 数据获取")
print("="*70)
print(f"总共 {len(stocks)} 只标的")
print(f"  核心标的: 圣邦股份(sz.300661)")
print(f"  相关股票: {len([s for s in stocks if s.get('priority') == '相关'])}只")
print(f"  配置股票: {len([s for s in stocks if s.get('priority') == '配置'])}只")

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
shengbang_success = False

for stock in stocks:
    code = stock["code"]
    name = stock["name"]
    start_date = stock["start_date"]
    category = stock["category"]
    priority = stock.get("priority", "配置")
    
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
            if code == "sz.300661":
                print(f"  [严重] 圣邦股份300661数据获取失败！")
            continue
        
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        
        if len(data_list) == 0:
            print(f"  [警告] 无数据，跳过")
            fail_count += 1
            if code == "sz.300661":
                print(f"  [严重] 圣邦股份300661无数据！")
            continue
        
        result = pd.DataFrame(data_list, columns=rs.fields)
        print(f"  [成功] 获取 {len(result)} 条数据")
        
        # 数据预处理
        result['date'] = pd.to_datetime(result['date'])
        result = result.sort_values('date')
        
        # 过滤停牌日
        result['tradestatus'] = pd.to_numeric(result['tradestatus'], errors='coerce')
        result = result[result['tradestatus'] == 1]
        
        # 分割训练集和测试集（以2024-12-31为分界）
        train_data = result[result['date'] <= '2024-12-31']
        test_data = result[result['date'] > '2024-12-31']
        
        if len(train_data) < 100:
            print(f"  [警告] 训练数据不足100条，跳过")
            fail_count += 1
            if code == "sz.300661":
                print(f"  [严重] 圣邦股份300661训练数据不足！")
            continue
        
        # 保存到专用目录（V11可以使用V7的数据格式）
        os.makedirs('stockdata_v7_300661/train', exist_ok=True)
        os.makedirs('stockdata_v7_300661/test', exist_ok=True)
        
        train_file = f'stockdata_v7_300661/train/{code}.{name}.csv'
        test_file = f'stockdata_v7_300661/test/{code}.{name}.csv'
        
        train_data.to_csv(train_file, index=False, encoding='utf-8-sig')
        test_data.to_csv(test_file, index=False, encoding='utf-8-sig')
        
        print(f"  [保存] 训练: {len(train_data)} | 测试: {len(test_data)}")
        success_count += 1
        
        if code == "sz.300661":
            shengbang_success = True
            print(f"  [核心] ✅ 圣邦股份300661数据获取成功！")
            print(f"  [核心] 训练数据: {len(train_data)}条，测试数据: {len(test_data)}条")
        
    except Exception as e:
        print(f"  [错误] {e}")
        fail_count += 1
        if code == "sz.300661":
            print(f"  [严重] 圣邦股份300661数据获取异常！")
    
    print()

print("="*70)
print("下载完成")
print("="*70)
print(f"成功: {success_count} 只")
print(f"失败: {fail_count} 只")

if shengbang_success:
    print(f"\n[核心] ✅ 圣邦股份300661数据获取成功！")
else:
    print(f"\n[严重] ❌ 圣邦股份300661数据获取失败！请检查！")

if success_count >= 8:
    print(f"\n[优秀] 成功{success_count}只，足够训练！")
elif success_count >= 5:
    print(f"\n[良好] 成功{success_count}只，可以训练")
else:
    print(f"\n[警告] 仅成功{success_count}只，建议至少5只")

if success_count > 0:
    # 保存元数据
    metadata_df = pd.DataFrame(stocks)
    metadata_df.to_csv('stockdata_v7_300661/metadata_v7_300661.csv', index=False, encoding='utf-8-sig')
    print(f"\n元数据已保存: stockdata_v7_300661/metadata_v7_300661.csv")
    
    print("\n[完成] 可以开始训练：")
    print("  python train_v11_300661.py")
    print("\n[说明] 本版本专门针对圣邦股份300661优化，初始资金10万元")
    print("[说明] 训练后的模型可用于V11全功能集成版实时预测")
else:
    print("\n[失败] 没有成功下载任何数据")

bs.logout()

