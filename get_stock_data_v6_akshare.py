# get_stock_data_v6_akshare.py - 使用AkShare获取V6数据
# -*- coding: utf-8 -*-
"""
V6 使用AkShare获取数据（支持ETF）
需要安装: pip install akshare
"""
import akshare as ak
import pandas as pd
import os
from datetime import datetime
# V6 股票列表
stocks = [
    # === 原有股票 ===
    {"code": "600000", "name": "浦发银行", "market": "sh",
     "category": "金融", "volatility": "低", "style": "稳健"},
    {"code": "600036", "name": "招商银行", "market": "sh",
     "category": "金融", "volatility": "低", "style": "稳健"},
    {"code": "002083", "name": "孚日股份", "market": "sz",
     "category": "周期", "volatility": "高", "style": "激进"},
    {"code": "001389", "name": "广合科技", "market": "sz",
     "category": "科技", "volatility": "高", "style": "激进"},
    {"code": "600418", "name": "江淮汽车", "market": "sh",
     "category": "周期", "volatility": "中", "style": "平衡"},
    
    # === 原有ETF ===
    {"code": "159966", "name": "创蓝筹", "market": "sz",
     "category": "宽基", "volatility": "中", "style": "平衡"},
    {"code": "159876", "name": "有色基金", "market": "sz",
     "category": "周期", "volatility": "高", "style": "激进"},
    {"code": "159928", "name": "消费ETF", "market": "sz",
     "category": "消费", "volatility": "中", "style": "平衡"},
    
    # === V6 新增ETF ===
    {"code": "513130", "name": "恒生科技ETF", "market": "sh",
     "category": "科技", "volatility": "高", "style": "激进"},
    {"code": "513180", "name": "恒生科技", "market": "sh",
     "category": "科技", "volatility": "高", "style": "激进"},
    {"code": "512480", "name": "半导体ETF", "market": "sh",
     "category": "科技", "volatility": "高", "style": "激进"},
    {"code": "513090", "name": "港股非银", "market": "sh",
     "category": "金融", "volatility": "中", "style": "平衡"},
    {"code": "159992", "name": "创新药", "market": "sz",
     "category": "医药", "volatility": "高", "style": "激进"},
    {"code": "159770", "name": "机器人", "market": "sz",
     "category": "科技", "volatility": "高", "style": "激进"},
    {"code": "588000", "name": "科创50", "market": "sh",
     "category": "科技", "volatility": "高", "style": "激进"},
    {"code": "515070", "name": "AI创新", "market": "sh",
     "category": "科技", "volatility": "高", "style": "激进"},
    {"code": "588030", "name": "科创智能", "market": "sh",
     "category": "科技", "volatility": "高", "style": "激进"},
    {"code": "516160", "name": "新能源", "market": "sh",
     "category": "新能源", "volatility": "高", "style": "激进"},
]

print("="*70)
print("V6 数据获取（使用AkShare）")
print("="*70)
print(f"\n总共 {len(stocks)} 只标的")
print("  - 原有: 8只")
print(f"  - 新增: {len(stocks) - 8}只\n")

# 按分类统计
from collections import Counter
category_count = Counter([s['category'] for s in stocks])
print("按类别分布:")
for cat, count in category_count.items():
    print(f"  - {cat}: {count}只")

print("\n" + "="*70)
print("开始下载数据...")
print("="*70 + "\n")

success_count = 0
fail_count = 0

os.makedirs('stockdata/train', exist_ok=True)
os.makedirs('stockdata/test', exist_ok=True)

for stock in stocks:
    code = stock["code"]
    name = stock["name"]
    market = stock["market"]
    category = stock["category"]
    
    # AkShare使用不同格式
    symbol = code if market == "sz" else code
    
    print(f"[{category}] 下载 {market}.{code} ({name})")
    
    try:
        # 获取股票/ETF数据
        # 对于ETF使用fund接口，对于股票使用stock接口
        if len(code) == 6 and code.startswith(('1', '5')):  # ETF
            # 尝试ETF接口
            try:
                df = ak.fund_etf_hist_em(symbol=symbol, period="daily", 
                                         start_date="20100101", 
                                         end_date="20251122", 
                                         adjust="qfq")
            except:
                # 如果失败，尝试股票接口
                df = ak.stock_zh_a_hist(symbol=code, period="daily", 
                                       start_date="20100101", 
                                       end_date="20251122", 
                                       adjust="qfq")
        else:  # 个股
            df = ak.stock_zh_a_hist(symbol=code, period="daily", 
                                   start_date="19990101", 
                                   end_date="20251122", 
                                   adjust="qfq")
        
        if df is None or len(df) == 0:
            print(f"  [警告] 无数据，跳过\n")
            fail_count += 1
            continue
        
        print(f"  [成功] 获取 {len(df)} 条数据")
        
        # 重命名列以匹配原格式
        df = df.rename(columns={
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'pctChg',
            '涨跌额': 'change',
            '换手率': 'turn'
        })
        
        # 添加缺失列（使用合理默认值）
        if 'preclose' not in df.columns:
            df['preclose'] = df['close'].shift(1)
        
        # 添加估值指标（ETF可能没有，使用0填充）
        for col in ['peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ']:
            if col not in df.columns:
                df[col] = 0
        
        df['tradestatus'] = 1  # 假设都是交易日
        df['code'] = f"{market}.{code}"
        
        # 确保日期格式正确
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # 选择需要的列
        columns = ['date', 'code', 'open', 'high', 'low', 'close', 'preclose', 
                   'volume', 'amount', 'turn', 'pctChg', 'peTTM', 'psTTM', 
                   'pcfNcfTTM', 'pbMRQ', 'tradestatus']
        
        # 只保留存在的列
        available_cols = [col for col in columns if col in df.columns]
        df = df[available_cols]
        
        # 补充缺失列
        for col in columns:
            if col not in df.columns:
                if col in ['peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ', 'turn']:
                    df[col] = 0
                elif col == 'tradestatus':
                    df[col] = 1
                elif col == 'preclose':
                    df[col] = df['close'].shift(1)
        
        # 调整列顺序
        df = df[columns]
        
        # 分割训练集和测试集
        train_data = df[df['date'] <= '2024-12-31']
        test_data = df[df['date'] > '2024-12-31']
        
        if len(train_data) < 100:
            print(f"  [警告] 训练数据不足100条，跳过\n")
            fail_count += 1
            continue
        
        # 保存
        train_file = f'stockdata/train/{market}.{code}.{name}.csv'
        test_file = f'stockdata/test/{market}.{code}.{name}.csv'
        
        train_data.to_csv(train_file, index=False, encoding='utf-8-sig')
        test_data.to_csv(test_file, index=False, encoding='utf-8-sig')
        
        print(f"  [保存] 训练: {len(train_data)} | 测试: {len(test_data)}")
        success_count += 1
        
    except Exception as e:
        print(f"  [错误] {e}")
        fail_count += 1
    
    print()

print("="*70)
print("下载完成")
print("="*70)
print(f"成功: {success_count} 只")
print(f"失败: {fail_count} 只")

if success_count > 0:
    # 保存元数据
    metadata_df = pd.DataFrame(stocks)
    metadata_df['code'] = metadata_df['market'] + '.' + metadata_df['code']
    metadata_df['start_date'] = '1999-01-01'  # 占位
    metadata_df = metadata_df[['code', 'name', 'category', 'volatility', 'style', 'start_date']]
    metadata_df.to_csv('stockdata/metadata_v6.csv', index=False, encoding='utf-8-sig')
    print(f"\n元数据已保存: stockdata/metadata_v6.csv")
    
    print("\n[完成] 可以开始训练：")
    print("  python train_v6.py")
else:
    print("\n[失败] 没有成功下载任何数据")
    print("请检查网络或尝试其他数据源")



