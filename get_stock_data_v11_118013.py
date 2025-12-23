# get_stock_data_v11_118013.py - V11道通转债118013专用版
# -*- coding: utf-8 -*-
"""
V11 道通转债118013专用版：
1. 专门针对道通转债118013进行数据获取和训练
2. 包含道通转债118013及其相关可转债和正股，确保更好的针对性
3. 初始资金2万元（匹配实盘操作）
4. 数据用于V11全功能集成版训练
"""
import baostock as bs
import pandas as pd
import os
from datetime import datetime

# 尝试导入akshare（用于可转债数据）
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
    print("✅ akshare可用，将用于可转债数据获取")
except ImportError:
    AKSHARE_AVAILABLE = False
    print("⚠️  akshare不可用，可转债数据可能无法获取")

# 登录baostock
lg = bs.login(user_id="anonymous", password="123456")
print("登录响应:", lg.error_code, lg.error_msg)

# V11 道通转债118013专用股票列表
# 包含道通转债118013及其相关可转债和正股，确保更好的针对性
stocks = [
    # === 核心标的：道通转债118013 ===
    {"code": "sh.118013", "name": "道通转债", "start_date": "2022-07-28", 
     "category": "可转债", "volatility": "中", "style": "平衡", "priority": "核心"},
    
    # === 正股：道通科技 ===
    {"code": "sh.688208", "name": "道通科技", "start_date": "2020-02-13", 
     "category": "汽车电子", "volatility": "中", "style": "成长", "priority": "相关"},
    
    # === 其他可转债（相关标的）===
    {"code": "sh.118001", "name": "南银转债", "start_date": "2021-06-18", 
     "category": "可转债", "volatility": "低", "style": "稳健", "priority": "相关"},
    {"code": "sh.118002", "name": "康泰转2", "start_date": "2021-08-09", 
     "category": "可转债", "volatility": "中", "style": "平衡", "priority": "相关"},
    {"code": "sz.123050", "name": "聚飞转债", "start_date": "2020-03-17", 
     "category": "可转债", "volatility": "中", "style": "平衡", "priority": "相关"},
    {"code": "sz.123051", "name": "今天转债", "start_date": "2020-03-17", 
     "category": "可转债", "volatility": "中", "style": "平衡", "priority": "相关"},
    
    # === 汽车电子板块（成长配置）===
    {"code": "sz.002920", "name": "德赛西威", "start_date": "2017-12-26", 
     "category": "汽车电子", "volatility": "中", "style": "成长", "priority": "配置"},
    {"code": "sz.300496", "name": "中科创达", "start_date": "2015-12-10", 
     "category": "汽车电子", "volatility": "高", "style": "成长", "priority": "配置"},
    
    # === 消费板块（平衡配置）===
    {"code": "sz.000333", "name": "美的集团", "start_date": "2013-09-18", 
     "category": "消费", "volatility": "中", "style": "平衡", "priority": "配置"},
    
    # === 金融板块（稳健配置）===
    {"code": "sh.600036", "name": "招商银行", "start_date": "2002-04-09", 
     "category": "金融", "volatility": "低", "style": "稳健", "priority": "配置"},
]

print("="*70)
print("V11 道通转债118013专用版 - 数据获取")
print("="*70)
print(f"总共 {len(stocks)} 只标的")
print(f"  核心标的: 道通转债118013")
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
daotong_convertible_success = False

for stock in stocks:
    code = stock["code"]
    name = stock["name"]
    start_date = stock["start_date"]
    category = stock["category"]
    priority = stock.get("priority", "配置")
    
    print(f"[{category}|{priority}] 查询 {code} ({name}), 起始: {start_date}")
    
    result = None
    
    # 如果是可转债，优先使用akshare
    if category == "可转债" and AKSHARE_AVAILABLE:
        try:
            print(f"  [尝试] 使用akshare获取可转债数据...")
            # 提取可转债代码（去掉市场前缀）
            bond_code = code.split('.')[-1] if '.' in code else code
            
            # 尝试使用akshare获取可转债数据
            # akshare的可转债接口可能有多种，尝试不同的方法
            df = None
            try:
                # 方法1: bond_zh_hs_daily (需要完整代码，如"sh118013"或"sz123050")
                full_code = code.replace('.', '')  # sh.118013 -> sh118013
                df = ak.bond_zh_hs_daily(symbol=full_code)
                if df is not None and len(df) > 0:
                    print(f"  [成功] 使用方法1获取数据")
            except:
                try:
                    # 方法2: 使用bond_code直接获取
                    df = ak.bond_zh_hs_daily(symbol=bond_code)
                    if df is not None and len(df) > 0:
                        print(f"  [成功] 使用方法2获取数据")
                except:
                    try:
                        # 方法3: 尝试使用股票接口（有些可转债可以用股票接口）
                        df = ak.stock_zh_a_hist(symbol=bond_code, period="daily", 
                                               start_date=start_date.replace('-', ''),
                                               end_date=datetime.now().strftime('%Y%m%d'),
                                               adjust="qfq")
                        if df is not None and len(df) > 0:
                            df = df.rename(columns={'日期': 'date', '开盘': 'open', '最高': 'high',
                                                   '最低': 'low', '收盘': 'close', '成交量': 'volume'})
                            print(f"  [成功] 使用方法3获取数据")
                    except Exception as e:
                        print(f"  [警告] 所有akshare方法都失败: {e}")
            
            if df is not None and len(df) > 0:
                # 确保有date列
                if 'date' not in df.columns and '日期' in df.columns:
                    df = df.rename(columns={'日期': 'date'})
                
                # 确保日期格式正确
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    # 过滤日期范围
                    df = df[df['date'] >= start_date]
                    df = df[df['date'] <= datetime.now().strftime('%Y-%m-%d')]
                    
                    if len(df) > 0:
                        # 确保必要的列存在
                        if 'open' not in df.columns:
                            df['open'] = df.get('开盘', df.get('open', df['close']))
                        if 'high' not in df.columns:
                            df['high'] = df.get('最高', df.get('high', df['close']))
                        if 'low' not in df.columns:
                            df['low'] = df.get('最低', df.get('low', df['close']))
                        if 'close' not in df.columns:
                            df['close'] = df.get('收盘', df.get('close', 0))
                        if 'volume' not in df.columns:
                            df['volume'] = df.get('成交量', df.get('volume', 0))
                        
                        # 计算前收盘价和涨跌幅
                        df['preclose'] = df['close'].shift(1).fillna(df['close'])
                        df['pctChg'] = ((df['close'] - df['preclose']) / df['preclose'] * 100).fillna(0)
                        
                        # 构建与baostock相同格式的DataFrame
                        result = pd.DataFrame({
                            'date': df['date'].dt.strftime('%Y-%m-%d'),
                            'code': code,
                            'open': df['open'].astype(float).round(2).astype(str),
                            'high': df['high'].astype(float).round(2).astype(str),
                            'low': df['low'].astype(float).round(2).astype(str),
                            'close': df['close'].astype(float).round(2).astype(str),
                            'preclose': df['preclose'].astype(float).round(2).astype(str),
                            'volume': df['volume'].astype(float).astype(str),
                            'amount': (df['close'].astype(float) * df['volume'].astype(float)).astype(str),
                            'adjustflag': '3',
                            'turn': '0',
                            'tradestatus': '1',
                            'pctChg': df['pctChg'].astype(float).round(2).astype(str),
                            'peTTM': '',
                            'psTTM': '',
                            'pcfNcfTTM': '',
                            'pbMRQ': '',
                            'isST': '0'
                        })
                        result = result.fillna('0')
                        print(f"  [成功] 使用akshare获取 {len(result)} 条数据")
        except Exception as e:
            print(f"  [警告] akshare获取异常: {e}")
    
    # 如果不是可转债，或者akshare获取失败，使用baostock
    if result is None or len(result) == 0:
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
                print(f"  [失败] baostock查询错误: {rs.error_msg}")
                if result is None:
                    fail_count += 1
                    if code == "sh.118013":
                        print(f"  [严重] 道通转债118013数据获取失败！")
                    continue
            
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())
            
            if len(data_list) == 0:
                if result is None:
                    print(f"  [警告] 无数据，跳过")
                    fail_count += 1
                    if code == "sh.118013":
                        print(f"  [严重] 道通转债118013无数据！")
                    continue
            
            if result is None:
                result = pd.DataFrame(data_list, columns=rs.fields)
                print(f"  [成功] 使用baostock获取 {len(result)} 条数据")
        except Exception as e:
            if result is None:
                print(f"  [错误] baostock获取异常: {e}")
                fail_count += 1
                if code == "sh.118013":
                    print(f"  [严重] 道通转债118013数据获取异常！")
                continue
    
    if result is None or len(result) == 0:
        print(f"  [警告] 最终无数据，跳过")
        fail_count += 1
        if code == "sh.118013":
            print(f"  [严重] 道通转债118013无数据！")
        continue
    
    try:
        # 数据预处理
        result['date'] = pd.to_datetime(result['date'])
        result = result.sort_values('date')
        
        # 分割训练集和测试集（以2024-12-31为分界）
        train_data = result[result['date'] <= '2024-12-31']
        test_data = result[result['date'] > '2024-12-31']
        
        if len(train_data) < 100:
            print(f"  [警告] 训练数据不足100条，跳过")
            fail_count += 1
            if code == "sh.118013":
                print(f"  [严重] 道通转债118013训练数据不足！")
            continue
        
        # 保存到专用目录（V11可以使用V7的数据格式）
        os.makedirs('stockdata_v7_118013/train', exist_ok=True)
        os.makedirs('stockdata_v7_118013/test', exist_ok=True)
        
        train_file = f'stockdata_v7_118013/train/{code}.{name}.csv'
        test_file = f'stockdata_v7_118013/test/{code}.{name}.csv'
        
        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)
        
        print(f"  [保存] 训练: {len(train_data)} | 测试: {len(test_data)}")
        success_count += 1
        
        if code == "sh.118013":
            daotong_convertible_success = True
            print(f"  [核心] ✅ 道通转债118013数据获取成功！")
            print(f"  [核心] 训练数据: {len(train_data)}条，测试数据: {len(test_data)}条")
        
    except Exception as e:
        print(f"  [错误] {e}")
        fail_count += 1
        if code == "sh.118013":
            print(f"  [严重] 道通转债118013数据获取异常！")
    
    print()

print("="*70)
print("下载完成")
print("="*70)
print(f"成功: {success_count} 只")
print(f"失败: {fail_count} 只")

if daotong_convertible_success:
    print(f"\n[核心] ✅ 道通转债118013数据获取成功！")
else:
    print(f"\n[严重] ❌ 道通转债118013数据获取失败！请检查！")

if success_count >= 8:
    print(f"\n[优秀] 成功{success_count}只，足够训练！")
elif success_count >= 5:
    print(f"\n[良好] 成功{success_count}只，可以训练")
else:
    print(f"\n[警告] 仅成功{success_count}只，建议至少5只")

if success_count > 0:
    # 保存元数据
    metadata_df = pd.DataFrame(stocks)
    metadata_df.to_csv('stockdata_v7_118013/metadata_v7_118013.csv', index=False, encoding='utf-8-sig')
    print(f"\n元数据已保存: stockdata_v7_118013/metadata_v7_118013.csv")
    
    print("\n[完成] 可以开始训练：")
    print("  python train_v11_118013.py")
    print("\n[说明] 本版本专门针对道通转债118013优化，初始资金5万元")
    print("[说明] 训练后的模型可用于V11全功能集成版实时预测")
else:
    print("\n[失败] 没有成功下载任何数据")

bs.logout()

