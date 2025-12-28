# -*- coding: utf-8 -*-
"""
获取市场最热新闻 - 不限制板块和股票

从各个板块的代表性热门股票中获取新闻，汇总后选择最新最热的5条
"""

import akshare as ak
import pandas as pd
from datetime import datetime
from typing import Optional
from get_latest_news_akshare import format_news_output, print_news


def get_market_hot_news(count: int = 5) -> Optional[pd.DataFrame]:
    """
    获取市场最热新闻（从各个板块的代表性热门股票汇总）
    
    Args:
        count: 要获取的新闻数量，默认5条
    
    Returns:
        DataFrame，包含新闻数据，失败返回None
    """
    # 选择各个板块的代表性热门股票，确保覆盖面广
    # 包括：金融、地产、消费、科技、医药、新能源、军工、周期等板块
    market_representative_stocks = [
        # 金融板块
        "000001",  # 平安银行
        "600036",  # 招商银行
        "600519",  # 贵州茅台（消费+金融属性）
        
        # 科技板块
        "002415",  # 海康威视
        "000063",  # 中兴通讯
        "300059",  # 东方财富
        
        # 新能源板块
        "300274",  # 阳光电源
        "601012",  # 隆基绿能
        
        # 消费板块
        "000858",  # 五粮液
        "002304",  # 洋河股份
        "000002",  # 万科A（地产）
        
        # 医药板块
        "000538",  # 云南白药
        "002007",  # 华兰生物
        
        # 军工板块
        "002025",  # 航天电器
        "600893",  # 航发动力
        
        # 周期板块
        "600028",  # 中国石化
        "601088",  # 中国神华
        
        # 电子/半导体
        "002241",  # 歌尔股份
        "002475",  # 立讯精密
        
        # 通信
        "600050",  # 中国联通
        "000776",  # 广发证券（券商）
    ]
    
    print(f"📰 正在从市场各板块代表性股票获取最新热点新闻...")
    print(f"📊 覆盖股票数量: {len(market_representative_stocks)} 只")
    print(f"🎯 目标：获取最新最热的 {count} 条新闻\n")
    
    all_news = []
    success_count = 0
    
    for idx, code in enumerate(market_representative_stocks, 1):
        try:
            print(f"  [{idx}/{len(market_representative_stocks)}] 获取 {code} 的新闻...", end=" ")
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
            error_msg = str(e)[:30]
            print(f"❌ 失败: {error_msg}")
            continue
    
    if not all_news:
        print(f"\n❌ 未能获取任何新闻")
        return None
    
    # 合并所有新闻
    news_df = pd.concat(all_news, ignore_index=True)
    
    print(f"\n✅ 成功从 {success_count}/{len(market_representative_stocks)} 只股票获取新闻")
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
    
    print(f"📋 最终选择最新的 {actual_count} 条市场最热新闻\n")
    
    return latest_news


def main():
    """主函数"""
    print("\n" + "="*80)
    print(" " * 30 + "🔥 市场最热新闻")
    print("="*80)
    print("\n功能说明:")
    print("  - 从市场各板块代表性股票中汇总新闻")
    print("  - 涵盖金融、科技、新能源、消费、医药、军工、周期等板块")
    print("  - 自动去重并按时间排序")
    print("  - 返回最新最热的5条新闻\n")
    
    # 获取最新最热的5条新闻
    news_df = get_market_hot_news(count=5)
    
    if news_df is None:
        print("❌ 获取新闻失败，请检查网络连接或稍后重试")
        return
    
    # 格式化输出
    news_list = format_news_output(news_df)
    
    # 打印新闻
    print("\n" + "="*80)
    print(" " * 30 + "🔥 市场最热新闻 Top 5")
    print("="*80)
    print_news(news_list)
    
    print("✅ 新闻获取完成\n")


if __name__ == "__main__":
    main()

