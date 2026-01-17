# get_stock_data_v11_1A0001.py - V11上证指数1A0001专用版
# -*- coding: utf-8 -*-
"""
V11 上证指数1A0001专用版：
1. 专门针对上证指数1A0001（sh.000001）进行数据获取和训练
2. 数据用于V11全功能集成版训练
3. 上证指数起始日期：1990-12-19（上海证券交易所成立）
"""
import baostock as bs
import pandas as pd
import os
from datetime import datetime

# 登录
lg = bs.login(user_id="anonymous", password="123456")
print("登录响应:", lg.error_code, lg.error_msg)

# V11 上证指数1A0001专用股票列表
# 包含上证指数及其相关指数，确保更好的针对性
stocks = [
    # === 核心标的：上证指数 ===
    {"code": "sh.000001", "name": "上证指数", "start_date": "1990-12-19", 
     "category": "指数", "volatility": "中", "style": "平衡", "priority": "核心"},
    
    # === 相关指数（用于训练参考）===
    {"code": "sh.000016", "name": "上证50", "start_date": "2004-01-02", 
     "category": "指数", "volatility": "中", "style": "平衡", "priority": "相关"},
    {"code": "sh.000300", "name": "沪深300", "start_date": "2005-04-08", 
     "category": "指数", "volatility": "中", "style": "平衡", "priority": "相关"},
    {"code": "sz.399001", "name": "深证成指", "start_date": "1991-04-03", 
     "category": "指数", "volatility": "中", "style": "平衡", "priority": "相关"},
    {"code": "sz.399006", "name": "创业板指", "start_date": "2010-06-01", 
     "category": "指数", "volatility": "高", "style": "激进", "priority": "相关"},
]

print(f"\n总共 {len(stocks)} 只标的")
print(f"  - 核心: 上证指数1A0001 (sh.000001)")
print(f"  - 相关: {len(stocks) - 1}只指数")

# 按分类统计
from collections import Counter
category_count = Counter([s['category'] for s in stocks])
print(f"\n按类别分布:")
for cat, count in category_count.items():
    print(f"  - {cat}: {count}只")

print("\n" + "="*70)
print("开始下载数据...")
print("="*70 + "\n")

success_count = 0
fail_count = 0
shangzheng_success = False

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
            adjustflag="3"  # 前复权
        )
        
        if rs.error_code != '0':
            print(f"  [失败] 查询错误: {rs.error_msg}")
            fail_count += 1
            if code == "sh.000001":
                print(f"  [严重] 上证指数1A0001数据获取失败！")
            continue
        
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        
        if len(data_list) == 0:
            print(f"  [警告] 无数据，跳过")
            fail_count += 1
            if code == "sh.000001":
                print(f"  [严重] 上证指数1A0001无数据！")
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
            if code == "sh.000001":
                print(f"  [严重] 上证指数1A0001训练数据不足！")
            continue
        
        # 保存
        os.makedirs('stockdata_v7_1A0001/train', exist_ok=True)
        os.makedirs('stockdata_v7_1A0001/test', exist_ok=True)
        
        train_file = f'stockdata_v7_1A0001/train/{code}.{name}.csv'
        test_file = f'stockdata_v7_1A0001/test/{code}.{name}.csv'
        
        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)
        
        print(f"  [保存] 训练: {len(train_data)} | 测试: {len(test_data)}")
        success_count += 1
        
        if code == "sh.000001":
            shangzheng_success = True
            print(f"  [✅] 上证指数1A0001数据获取成功！")
        
    except Exception as e:
        print(f"  [异常] {str(e)}")
        fail_count += 1
        if code == "sh.000001":
            print(f"  [严重] 上证指数1A0001数据获取异常！")
        import traceback
        traceback.print_exc()
    
    print()

bs.logout()

print("="*70)
print("数据获取完成")
print("="*70)
print(f"成功: {success_count} | 失败: {fail_count}")

if shangzheng_success:
    print("\n✅ 上证指数1A0001数据获取成功！可以开始训练V11模型。")
else:
    print("\n❌ 上证指数1A0001数据获取失败！请检查网络连接和代码。")































