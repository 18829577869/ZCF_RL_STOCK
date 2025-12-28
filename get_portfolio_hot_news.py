# -*- coding: utf-8 -*-
"""
获取投资组合股票的最新热点新闻

从指定的股票列表中获取新闻，合并后选择最新最热的5条
"""

from get_latest_news_akshare import get_latest_news_akshare, format_news_output, print_news
import akshare as ak
import pandas as pd
from typing import List, Optional


def get_portfolio_hot_news(stock_codes: List[str], count: int = 5) -> Optional[pd.DataFrame]:
    """
    获取投资组合股票的最新热点新闻
    
    Args:
        stock_codes: 股票代码列表（如 ["002706", "300811", ...]）
        count: 要获取的新闻数量，默认5条
    
    Returns:
        DataFrame，包含新闻数据，失败返回None
    """
    if not stock_codes:
        print("❌ 股票代码列表为空")
        return None
    
    print(f"📰 正在从 {len(stock_codes)} 只股票获取最新热点新闻...")
    print(f"股票列表: {', '.join(stock_codes[:10])}{'...' if len(stock_codes) > 10 else ''}\n")
    
    all_news = []
    success_count = 0
    
    for idx, code in enumerate(stock_codes, 1):
        try:
            # 跳过转债代码（通常以11、12开头，且akshare的stock_news_em不支持转债）
            if code.startswith('11') or code.startswith('12'):
                print(f"  [{idx}/{len(stock_codes)}] 跳过 {code} (转债代码，暂不支持)...")
                continue
                
            print(f"  [{idx}/{len(stock_codes)}] 获取 {code} 的新闻...", end=" ")
            temp_news = ak.stock_news_em(symbol=code)
            
            if temp_news is not None and len(temp_news) > 0:
                # 添加股票代码列以便识别来源
                temp_news['来源股票'] = code
                all_news.append(temp_news)
                success_count += 1
                print(f"✅ {len(temp_news)}条")
            else:
                print("⚠️ 无数据")
        except Exception as e:
            error_msg = str(e)[:50]
            # 如果是转债相关的错误，给出更友好的提示
            if 'code' in error_msg.lower() or '转债' in error_msg:
                print(f"⚠️ 跳过 (不支持转债)")
            else:
                print(f"❌ 失败: {error_msg}")
            continue
    
    if not all_news:
        print(f"\n❌ 未能从任何股票获取新闻")
        return None
    
    # 合并所有新闻
    news_df = pd.concat(all_news, ignore_index=True)
    
    print(f"\n✅ 成功从 {success_count}/{len(stock_codes)} 只股票获取新闻")
    print(f"📊 合并后共 {len(news_df)} 条新闻")
    
    # 按标题去重，保留第一条（通常是最新的）
    if '新闻标题' in news_df.columns:
        # 去重
        before_dedup = len(news_df)
        news_df = news_df.drop_duplicates(subset=['新闻标题'], keep='first')
        after_dedup = len(news_df)
        if before_dedup != after_dedup:
            print(f"🔍 去重后剩余 {after_dedup} 条新闻（去除了 {before_dedup - after_dedup} 条重复）")
        
        # 按发布时间降序排序（最新的在前）
        if '发布时间' in news_df.columns:
            news_df = news_df.sort_values('发布时间', ascending=False, na_position='last')
            news_df = news_df.reset_index(drop=True)
        
        # 移除临时添加的来源股票列
        if '来源股票' in news_df.columns:
            news_df = news_df.drop('来源股票', axis=1)
    
    # 获取最新的指定数量
    actual_count = min(count, len(news_df))
    latest_news = news_df.head(actual_count)
    
    print(f"📋 最终选择最新的 {actual_count} 条新闻\n")
    
    return latest_news


def main():
    """主函数"""
    # 投资组合股票代码列表（从用户提供的列表提取）
    portfolio_stocks = [
        "002706",  # 良信股份
        "300811",  # 铂科新材
        "688208",  # 道通科技
        "002241",  # 歌尔股份
        "002475",  # 立讯精密
        "300274",  # 阳光电源
        "300499",  # 高澜股份
        "603267",  # 鸿远电子
        "002335",  # 科华数据
        "002851",  # 麦格米特
        "118013",  # 道通转债
        "300153",  # 科泰电源
        "301389",  # 隆扬电子
        "300762",  # 上海瀚讯
        "002025",  # 航天电器
        "002837",  # 英维克
        "300726",  # 宏达电子
        "002364",  # 中恒电气
        "601012",  # 隆基绿能
    ]
    
    print("\n" + "="*80)
    print(" " * 25 + "📰 投资组合最新热点新闻")
    print("="*80)
    print(f"\n📊 投资组合包含 {len(portfolio_stocks)} 只股票")
    print(f"🎯 目标：获取最新最热的 5 条新闻\n")
    
    # 获取最新最热的5条新闻
    news_df = get_portfolio_hot_news(portfolio_stocks, count=5)
    
    if news_df is None:
        print("❌ 获取新闻失败")
        return
    
    # 格式化输出
    news_list = format_news_output(news_df)
    
    # 打印新闻
    print("\n" + "="*80)
    print(" " * 30 + "🔥 最新最热新闻")
    print("="*80)
    print_news(news_list)
    
    print("✅ 新闻获取完成\n")


if __name__ == "__main__":
    main()

