# get_stock_data_v11_603019.py - V11中科曙光603019专用版
# -*- coding: utf-8 -*-
"""
V11 中科曙光603019专用版：
1. 专门针对中科曙光603019进行数据获取和训练
2. 包含中科曙光603019及其相关股票，确保更好的针对性
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

# V11 中科曙光603019专用股票列表
# 包含中科曙光603019及其相关股票，确保更好的针对性
stocks = [
    # === 核心标的：中科曙光603019 ===
    {"code": "sh.603019", "name": "中科曙光", "start_date": "2014-11-06", 
     "category": "服务器/超算", "volatility": "中", "style": "平衡", "priority": "核心"},
    
    # === 服务器/超算板块（相关股票）===
    {"code": "sz.000977", "name": "浪潮信息", "start_date": "2000-06-08", 
     "category": "服务器/超算", "volatility": "高", "style": "激进", "priority": "相关"},
    {"code": "sz.000938", "name": "紫光股份", "start_date": "1999-11-04", 
     "category": "服务器/超算", "volatility": "中", "style": "平衡", "priority": "相关"},
    {"code": "sz.000063", "name": "中兴通讯", "start_date": "1997-11-18", 
     "category": "服务器/超算", "volatility": "中", "style": "平衡", "priority": "相关"},
    
    # === AI/芯片板块（相关股票）===
    {"code": "sh.688041", "name": "海光信息", "start_date": "2022-08-12", 
     "category": "AI/芯片", "volatility": "高", "style": "激进", "priority": "相关"},
    {"code": "sh.688256", "name": "寒武纪", "start_date": "2020-07-20", 
     "category": "AI/芯片", "volatility": "高", "style": "激进", "priority": "相关"},
    {"code": "sz.002230", "name": "科大讯飞", "start_date": "2008-05-12", 
     "category": "AI/芯片", "volatility": "高", "style": "激进", "priority": "相关"},
    
    # === 科技板块（成长配置）===
    {"code": "sz.002415", "name": "海康威视", "start_date": "2010-05-28", 
     "category": "科技", "volatility": "中", "style": "平衡", "priority": "配置"},
    {"code": "sh.600276", "name": "恒瑞医药", "start_date": "2000-10-18", 
     "category": "医药", "volatility": "中", "style": "平衡", "priority": "配置"},
    
    # === 消费板块（平衡配置）===
    {"code": "sz.000333", "name": "美的集团", "start_date": "2013-09-18", 
     "category": "消费", "volatility": "中", "style": "平衡", "priority": "配置"},
    
    # === 金融板块（稳健配置）===
    {"code": "sh.600036", "name": "招商银行", "start_date": "2002-04-09", 
     "category": "金融", "volatility": "低", "style": "稳健", "priority": "配置"},
]

print("="*70)
print("V11 中科曙光603019专用版 - 数据获取")
print("="*70)
print(f"总共 {len(stocks)} 只标的")
print(f"  核心标的: 中科曙光603019")
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
zhongke_shuguang_success = False

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
            if code == "sh.603019":
                print(f"  [严重] 中科曙光603019数据获取失败！")
            continue
        
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        
        if len(data_list) == 0:
            print(f"  [警告] 无数据，跳过")
            fail_count += 1
            if code == "sh.603019":
                print(f"  [严重] 中科曙光603019无数据！")
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
            if code == "sh.603019":
                print(f"  [严重] 中科曙光603019训练数据不足！")
            continue
        
        # 保存到专用目录（V11可以使用V7的数据格式）
        os.makedirs('stockdata_v7_603019/train', exist_ok=True)
        os.makedirs('stockdata_v7_603019/test', exist_ok=True)
        
        train_file = f'stockdata_v7_603019/train/{code}.{name}.csv'
        test_file = f'stockdata_v7_603019/test/{code}.{name}.csv'
        
        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)
        
        print(f"  [保存] 训练: {len(train_data)} | 测试: {len(test_data)}")
        success_count += 1
        
        if code == "sh.603019":
            zhongke_shuguang_success = True
            print(f"  [核心] ✅ 中科曙光603019数据获取成功！")
            print(f"  [核心] 训练数据: {len(train_data)}条，测试数据: {len(test_data)}条")
        
    except Exception as e:
        print(f"  [错误] {e}")
        fail_count += 1
        if code == "sh.603019":
            print(f"  [严重] 中科曙光603019数据获取异常！")
    
    print()

print("="*70)
print("下载完成")
print("="*70)
print(f"成功: {success_count} 只")
print(f"失败: {fail_count} 只")

if zhongke_shuguang_success:
    print(f"\n[核心] ✅ 中科曙光603019数据获取成功！")
else:
    print(f"\n[严重] ❌ 中科曙光603019数据获取失败！请检查！")

if success_count >= 8:
    print(f"\n[优秀] 成功{success_count}只，足够训练！")
elif success_count >= 5:
    print(f"\n[良好] 成功{success_count}只，可以训练")
else:
    print(f"\n[警告] 仅成功{success_count}只，建议至少5只")

if success_count > 0:
    # 保存元数据
    metadata_df = pd.DataFrame(stocks)
    metadata_df.to_csv('stockdata_v7_603019/metadata_v7_603019.csv', index=False, encoding='utf-8-sig')
    print(f"\n元数据已保存: stockdata_v7_603019/metadata_v7_603019.csv")
    
    print("\n[完成] 可以开始训练：")
    print("  python train_v11_603019.py")
    print("\n[说明] 本版本专门针对中科曙光603019优化，初始资金2万元")
    print("[说明] 训练后的模型可用于V11全功能集成版实时预测")
else:
    print("\n[失败] 没有成功下载任何数据")

bs.logout()

