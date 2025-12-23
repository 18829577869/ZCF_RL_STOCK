"""
使用英维克(002837)和麦格米特(002851)两个 V7 PPO 模型
分别对科士达(002518)和中恒电气(002364)做一次回测对比。

流程：
1. 用 baostock 获取 002518 / 002364 的日线数据，保存到临时目录 backtest_tmp/
2. 用 stock_env_v6.StockTradingEnv 加载数据
3. 分别加载 ppo_stock_v7_002837.zip / ppo_stock_v7_002851.zip 做完整回测
4. 打印每个组合的核心指标：最终净值、总收益率、最大回撤、夏普比率、交易次数、胜率
"""

import os
import datetime
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stock_env_v6 import StockTradingEnv

DATA_DIR = "backtest_tmp_002518_002364"
INITIAL_BALANCE = 50000.0

TARGET_STOCKS = [
    {"code": "sz.002518", "name": "科士达"},
    {"code": "sz.002364", "name": "中恒电气"},
]

MODELS = [
    {"path": "ppo_stock_v7_002837.zip", "label": "英维克模型(002837)"},
    {"path": "ppo_stock_v7_002851.zip", "label": "麦格米特模型(002851)"},
]


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def fetch_history_bs(stock_code: str, stock_name: str, start_date: str = "2015-01-01"):
    """用 baostock 获取日线数据，保存到临时 CSV，返回文件路径"""
    try:
        import baostock as bs
    except ImportError:
        print("❌ 未安装 baostock，无法获取数据，请先 pip install baostock")
        return None

    ensure_data_dir()

    if "." in stock_code:
        market, num = stock_code.split(".")
        bs_code = f"{market}.{num}"
    else:
        # 简单推断
        if stock_code.startswith("6"):
            bs_code = f"sh.{stock_code}"
        else:
            bs_code = f"sz.{stock_code}"

    today = datetime.date.today().strftime("%Y-%m-%d")
    print(f"📥 获取 {stock_name}({bs_code}) 日线数据: {start_date} ~ {today}")

    lg = bs.login()
    print("  登录响应:", lg.error_code, lg.error_msg)
    try:
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,code,open,high,low,close,preclose,volume,amount,adjustflag,"
            "turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST",
            start_date=start_date,
            end_date=today,
            frequency="d",
            adjustflag="3",
        )
        if rs.error_code != "0":
            print(f"  ❌ 查询错误: {rs.error_msg}")
            return None

        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            print("  ❌ 无数据")
            return None

        df = pd.DataFrame(data_list, columns=rs.fields)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        print(f"  ✅ 共 {len(df)} 条记录")

        out_file = os.path.join(DATA_DIR, f"{bs_code.replace('.', '_')}_{stock_name}.csv")
        df.to_csv(out_file, index=False, encoding="utf-8-sig")
        print(f"  💾 已保存到: {out_file}")
        return out_file
    finally:
        bs.logout()


def run_backtest(data_file: str, model_path: str, stock_label: str, model_label: str):
    """用给定模型对指定数据文件做一次完整回测"""
    if not os.path.exists(model_path):
        print(f"  ❌ 模型文件不存在: {model_path}")
        return

    try:
        env = StockTradingEnv(data_file, initial_balance=INITIAL_BALANCE)
    except Exception as e:
        print(f"  ❌ 创建环境失败: {e}")
        return

    print(f"\n  📌 使用模型: {model_label} ({model_path})")
    model = PPO.load(model_path)

    obs, _ = env.reset()
    done = False
    step_count = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        step_count += 1

    stats = env.get_stats()
    if not stats:
        print("  ⚠️ 无回测统计信息")
        return

    print(f"  🔚 步数: {step_count}")
    print(f"  💼 最终净值: {stats.get('final_net_worth', 0):,.2f} 元")
    print(f"  📈 总收益率: {stats.get('total_return', 0):+.2f}%")
    print(f"  📉 最大回撤: {stats.get('max_drawdown', 0):.2f}%")
    print(f"  📊 夏普比率: {stats.get('sharpe_ratio', 0):.2f}")
    print(f"  🔁 交易次数: {stats.get('num_trades', 0)}")
    print(f"  ✅ 胜率: {stats.get('win_rate', 0):.2f}%")
    print(f"  ⚠️ 风险事件: {stats.get('risk_events', 0)} 次")


def main():
    print("=" * 70)
    print("V7 模型迁移回测 - 使用英维克(002837)/麦格米特(002851)模型")
    print("目标标的：科士达(002518)、中恒电气(002364)")
    print("=" * 70)

    ensure_data_dir()

    # 1. 准备数据
    data_files = {}
    for t in TARGET_STOCKS:
        f = fetch_history_bs(t["code"], t["name"])
        if f:
            data_files[t["code"]] = f

    if not data_files:
        print("❌ 目标标的数据全部获取失败，无法回测")
        return

    # 2. 依次跑回测
    for t in TARGET_STOCKS:
        code = t["code"]
        name = t["name"]
        data_file = data_files.get(code)
        if not data_file:
            continue

        print("\n" + "=" * 70)
        print(f"🎯 目标标的: {name}({code})")
        print(f"📁 数据文件: {data_file}")
        print("=" * 70)

        for m in MODELS:
            run_backtest(data_file, m["path"], f"{name}({code})", m["label"])

    print("\n" + "=" * 70)
    print("✅ 回测完成（英维克/麦格米特模型 → 科士达 / 中恒电气）")
    print("=" * 70)


if __name__ == "__main__":
    main()




