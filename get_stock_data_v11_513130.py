# get_stock_data_v11_513130.py - V11 恒生科技ETF 513130 专用版（含 Baostock 获取 + AkShare 兜底）
# -*- coding: utf-8 -*-
"""
V11 恒生科技ETF 513130 专用版：
1. 核心标的：恒生科技ETF 513130
2. 先用 Baostock 获取日线；若失败或无数据，自动用 AkShare fund_etf_hist_em 兜底
3. 数据用于 V11 全功能集成版训练
"""
import os
from datetime import datetime

import baostock as bs
import pandas as pd

try:
    import akshare as ak
    AK_AVAILABLE = True
except Exception:
    AK_AVAILABLE = False

# 登录 Baostock
lg = bs.login(user_id="anonymous", password="123456")
print("登录响应:", lg.error_code, lg.error_msg)

# 核心标的列表（可按需扩展）
stocks = [
    {
        "code": "sh.513130",
        "name": "恒生科技ETF",
        "start_date": "2020-11-09",  # 上市日附近
        "category": "科技ETF",
        "volatility": "高",
        "style": "激进",
        "priority": "核心",
    },
]

print("=" * 70)
print("V11 恒生科技ETF 513130 专用版 - 数据获取")
print("=" * 70)
print(f"总共 {len(stocks)} 只标的")
print(f"  核心标的: 恒生科技ETF 513130")
print(f"  相关/配置股票: 0只（可按需扩展）")

# 按分类统计
from collections import Counter

category_count = Counter([s["category"] for s in stocks])
print(f"\n按类别分布:")
for cat, count in category_count.items():
    print(f"  - {cat}: {count}只")

print("\n" + "=" * 70)
print("开始下载数据...")
print("=" * 70 + "\n")

success_count = 0
fail_count = 0
core_success = False

os.makedirs("stockdata_v7_513130/train", exist_ok=True)
os.makedirs("stockdata_v7_513130/test", exist_ok=True)

def fetch_with_baostock(code: str, start_date: str):
    """使用 Baostock 获取日线数据"""
    rs = bs.query_history_k_data_plus(
        code,
        "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST",
        start_date=start_date,
        end_date=datetime.now().strftime("%Y-%m-%d"),
        frequency="d",
        adjustflag="3",
    )
    if rs.error_code != "0":
        raise RuntimeError(f"Baostock error: {rs.error_msg}")
    data_list = []
    while rs.next():
        data_list.append(rs.get_row_data())
    if len(data_list) == 0:
        raise RuntimeError("Baostock no data")
    return pd.DataFrame(data_list, columns=rs.fields)


def fetch_with_akshare(code: str):
    """使用 AkShare 获取ETF日线数据（前复权）"""
    if not AK_AVAILABLE:
        raise RuntimeError("AkShare not installed")
    symbol = code.split(".")[1] if "." in code else code
    df = ak.fund_etf_hist_em(
        symbol=symbol, period="daily", start_date="20100101", end_date=datetime.now().strftime("%Y%m%d"), adjust="qfq"
    )
    if df is None or len(df) == 0:
        raise RuntimeError("AkShare no data")
    df = df.rename(
        columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "涨跌幅": "pctChg",
            "涨跌额": "change",
            "换手率": "turn",
        }
    )
    if "preclose" not in df.columns:
        df["preclose"] = df["close"].shift(1)
    df["code"] = code
    df["tradestatus"] = 1
    for col in ["peTTM", "psTTM", "pcfNcfTTM", "pbMRQ", "isST"]:
        if col not in df.columns:
            df[col] = 0
    return df[
        [
            "date",
            "code",
            "open",
            "high",
            "low",
            "close",
            "preclose",
            "volume",
            "amount",
            "turn",
            "pctChg",
            "peTTM",
            "psTTM",
            "pcfNcfTTM",
            "pbMRQ",
            "tradestatus",
            "isST",
        ]
    ]


for stock in stocks:
    code = stock["code"]
    name = stock["name"]
    start_date = stock["start_date"]
    category = stock["category"]
    priority = stock.get("priority", "配置")

    print(f"[{category}|{priority}] 查询 {code} ({name}), 起始: {start_date}")

    try:
        try:
            result = fetch_with_baostock(code, start_date)
            source = "baostock"
        except Exception as e_bs:
            print(f"  [警告] Baostock失败: {e_bs}")
            result = fetch_with_akshare(code)
            source = "akshare"

        print(f"  [成功] 获取 {len(result)} 条数据 (来源: {source})")

        # 数据预处理
        result["date"] = pd.to_datetime(result["date"])
        result = result.sort_values("date")

        # 分割训练集和测试集（以2024-12-31为分界）
        train_data = result[result["date"] <= "2024-12-31"]
        test_data = result[result["date"] > "2024-12-31"]

        if len(train_data) < 100:
            print(f"  [警告] 训练数据不足100条，跳过")
            fail_count += 1
            if code == "sh.513130":
                print(f"  [严重] 恒生科技ETF 513130 训练数据不足！")
            continue

        # 保存到专用目录
        train_file = f"stockdata_v7_513130/train/{code}.{name}.csv"
        test_file = f"stockdata_v7_513130/test/{code}.{name}.csv"

        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)

        print(f"  [保存] 训练: {len(train_data)} | 测试: {len(test_data)}")
        success_count += 1

        if code == "sh.513130":
            core_success = True
            print(f"  [核心] ✅ 恒生科技ETF 513130 数据获取成功！")
            print(f"  [核心] 训练数据: {len(train_data)}条，测试数据: {len(test_data)}条")

    except Exception as e:
        print(f"  [错误] {e}")
        fail_count += 1
        if code == "sh.513130":
            print(f"  [严重] 恒生科技ETF 513130 数据获取异常！")

    print()

print("=" * 70)
print("下载完成")
print("=" * 70)
print(f"成功: {success_count} 只")
print(f"失败: {fail_count} 只")

if core_success:
    print(f"\n[核心] ✅ 恒生科技ETF 513130 数据获取成功！")
else:
    print(f"\n[严重] ❌ 恒生科技ETF 513130 数据获取失败！请检查！")

if success_count >= 1:
    # 保存元数据
    metadata_df = pd.DataFrame(stocks)
    metadata_df.to_csv("stockdata_v7_513130/metadata_v7_513130.csv", index=False, encoding="utf-8-sig")
    print(f"\n元数据已保存: stockdata_v7_513130/metadata_v7_513130.csv")
    print("\n[完成] 可以开始训练：")
    print("  python train_v11_513130.py")
else:
    print("\n[失败] 没有成功下载任何数据")

bs.logout()


