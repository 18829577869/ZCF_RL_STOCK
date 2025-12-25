# get_stock_data_v11_301389_600110.py - V11隆扬电子和诺德股份专用版
# -*- coding: utf-8 -*-
"""
V11 隆扬电子和诺德股份专用版：
1. 专门针对隆扬电子(sz.301389)和诺德股份(sh.600110)进行数据获取和训练
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

# V11 隆扬电子和诺德股份专用股票列表
stocks = [
    # === 核心标的：隆扬电子301389 ===
    {"code": "sz.301389", "name": "隆扬电子", "start_date": "2022-10-31", 
     "category": "电子元件", "volatility": "中", "style": "成长", "priority": "核心"},
    
    # === 核心标的：诺德股份600110 ===
    {"code": "sh.600110", "name": "诺德股份", "start_date": "1997-10-07", 
     "category": "新材料", "volatility": "中", "style": "成长", "priority": "核心"},
    
    # === 电子元件板块（相关股票）===
    {"code": "sz.002241", "name": "歌尔股份", "start_date": "2008-05-22", 
     "category": "消费电子", "volatility": "中", "style": "成长", "priority": "相关"},
    {"code": "sz.002475", "name": "立讯精密", "start_date": "2010-09-15", 
     "category": "消费电子", "volatility": "中", "style": "成长", "priority": "相关"},
    {"code": "sz.300726", "name": "宏达电子", "start_date": "2017-11-21", 
     "category": "电子元件", "volatility": "中", "style": "成长", "priority": "相关"},
    {"code": "sz.301005", "name": "超捷股份", "start_date": "2021-06-01", 
     "category": "汽车零部件", "volatility": "高", "style": "成长", "priority": "相关"},
    
    # === 新材料/新能源板块（相关股票）===
    {"code": "sz.300750", "name": "宁德时代", "start_date": "2018-06-11", 
     "category": "新能源", "volatility": "中", "style": "成长", "priority": "相关"},
    {"code": "sz.002594", "name": "比亚迪", "start_date": "2011-06-30", 
     "category": "新能源", "volatility": "中", "style": "成长", "priority": "相关"},
    {"code": "sz.300274", "name": "阳光电源", "start_date": "2011-11-02", 
     "category": "新能源", "volatility": "中", "style": "成长", "priority": "相关"},
    
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
print("V11 隆扬电子和诺德股份专用版 - 数据获取")
print("="*70)
print(f"总共 {len(stocks)} 只标的")
print(f"  核心标的: 隆扬电子(sz.301389)、诺德股份(sh.600110)")
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
longyang_success = False
nuode_success = False

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
            if code == "sz.301389":
                print(f"  [严重] 隆扬电子301389数据获取失败！")
            elif code == "sh.600110":
                print(f"  [严重] 诺德股份600110数据获取失败！")
            continue
        
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        
        if len(data_list) == 0:
            print(f"  [警告] 无数据，跳过")
            fail_count += 1
            if code == "sz.301389":
                print(f"  [严重] 隆扬电子301389无数据！")
            elif code == "sh.600110":
                print(f"  [严重] 诺德股份600110无数据！")
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
            if code == "sz.301389":
                print(f"  [严重] 隆扬电子301389训练数据不足！")
            elif code == "sh.600110":
                print(f"  [严重] 诺德股份600110训练数据不足！")
            continue
        
        # 保存到专用目录（V11可以使用V7的数据格式）
        os.makedirs('stockdata_v7_301389_600110/train', exist_ok=True)
        os.makedirs('stockdata_v7_301389_600110/test', exist_ok=True)
        
        train_file = f'stockdata_v7_301389_600110/train/{code}.{name}.csv'
        test_file = f'stockdata_v7_301389_600110/test/{code}.{name}.csv'
        
        train_data.to_csv(train_file, index=False, encoding='utf-8-sig')
        test_data.to_csv(test_file, index=False, encoding='utf-8-sig')
        
        print(f"  [保存] 训练: {len(train_data)} | 测试: {len(test_data)}")
        success_count += 1
        
        if code == "sz.301389":
            longyang_success = True
            print(f"  [核心] ✅ 隆扬电子301389数据获取成功！")
            print(f"  [核心] 训练数据: {len(train_data)}条，测试数据: {len(test_data)}条")
        elif code == "sh.600110":
            nuode_success = True
            print(f"  [核心] ✅ 诺德股份600110数据获取成功！")
            print(f"  [核心] 训练数据: {len(train_data)}条，测试数据: {len(test_data)}条")
        
    except Exception as e:
        print(f"  [错误] {e}")
        fail_count += 1
        if code == "sz.301389":
            print(f"  [严重] 隆扬电子301389数据获取异常！")
        elif code == "sh.600110":
            print(f"  [严重] 诺德股份600110数据获取异常！")
    
    print()

print("="*70)
print("下载完成")
print("="*70)
print(f"成功: {success_count} 只")
print(f"失败: {fail_count} 只")

if longyang_success:
    print(f"\n[核心] ✅ 隆扬电子301389数据获取成功！")
else:
    print(f"\n[严重] ❌ 隆扬电子301389数据获取失败！请检查！")

if nuode_success:
    print(f"[核心] ✅ 诺德股份600110数据获取成功！")
else:
    print(f"[严重] ❌ 诺德股份600110数据获取失败！请检查！")

if success_count >= 8:
    print(f"\n[优秀] 成功{success_count}只，足够训练！")
elif success_count >= 5:
    print(f"\n[良好] 成功{success_count}只，可以训练")
else:
    print(f"\n[警告] 仅成功{success_count}只，建议至少5只")

if success_count > 0:
    # 保存元数据
    metadata_df = pd.DataFrame(stocks)
    metadata_df.to_csv('stockdata_v7_301389_600110/metadata_v7_301389_600110.csv', index=False, encoding='utf-8-sig')
    print(f"\n元数据已保存: stockdata_v7_301389_600110/metadata_v7_301389_600110.csv")
    
    print("\n[完成] 可以开始训练：")
    print("  python train_v11_301389_600110.py")
    print("\n[说明] 本版本专门针对隆扬电子301389和诺德股份600110优化，初始资金10万元")
    print("[说明] 训练后的模型可用于V11全功能集成版实时预测")
else:
    print("\n[失败] 没有成功下载任何数据")

bs.logout()

