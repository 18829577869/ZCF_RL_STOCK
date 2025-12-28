# -*- coding: utf-8 -*-
"""
获取最新新闻热点 - 简化版

功能：
- 快速获取最新的5条新闻热点
- 支持股票新闻和CCTV新闻
- 自动格式化输出
"""

import akshare as ak
import pandas as pd
from datetime import datetime
from typing import Optional


def get_latest_news(count: int = 5, stock_code: str = None, use_cctv: bool = False):
    """
    获取最新新闻
    
    Args:
        count: 要获取的新闻数量，默认5条
        stock_code: 股票代码（如"300762"），如果为None则使用"000001"获取市场新闻
        use_cctv: 是否使用CCTV新闻，默认False（使用股票新闻）
    
    Returns:
        DataFrame，包含新闻数据
    """
    try:
        if use_cctv:
            print(f"📺 正在获取CCTV新闻...")
            news_df = ak.news_cctv()
            source_name = "CCTV新闻"
        else:
            if stock_code is None:
                stock_code = "000001"
                stock_name = "市场热点"
            else:
                stock_name = f"股票{stock_code}"
            
            print(f"📰 正在获取{stock_name}的股票新闻...")
            news_df = ak.stock_news_em(symbol=stock_code)
            source_name = f"{stock_name}新闻"
        
        if news_df is None or len(news_df) == 0:
            print("❌ 未获取到新闻数据")
            return None
        
        print(f"✅ 成功获取 {len(news_df)} 条{source_name}")
        
        # 获取最新的指定数量
        actual_count = min(count, len(news_df))
        latest_news = news_df.head(actual_count)
        
        return latest_news
        
    except Exception as e:
        print(f"❌ 获取新闻失败: {e}")
        return None


def format_date(date_str: str) -> str:
    """格式化日期字符串"""
    if not date_str or pd.isna(date_str):
        return ""
    
    date_str = str(date_str).strip()
    if "-" in date_str:
        return date_str
    
    # 格式化 YYYYMMDD -> YYYY-MM-DD
    if len(date_str) == 8 and date_str.isdigit():
        try:
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        except:
            return date_str
    
    return date_str


def print_news_simple(news_df: pd.DataFrame):
    """简单格式化打印新闻"""
    if news_df is None or len(news_df) == 0:
        print("❌ 没有新闻数据")
        return
    
    print("\n" + "="*80)
    print(" " * 30 + "🔥 最新新闻热点")
    print("="*80)
    print(f"获取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"新闻数量: {len(news_df)} 条\n")
    
    # 根据列名自动识别字段
    title_col = None
    content_col = None
    date_col = None
    source_col = None
    
    for col in news_df.columns:
        col_lower = col.lower()
        if "标题" in col or "title" in col_lower:
            title_col = col
        elif "内容" in col or "content" in col_lower:
            content_col = col
        elif "时间" in col or "日期" in col or "date" in col_lower or "time" in col_lower:
            date_col = col
        elif "来源" in col or "source" in col_lower or "文章来源" in col:
            source_col = col
    
    for idx, row in news_df.iterrows():
        print("-" * 80)
        print(f"【新闻 {idx + 1}】")
        
        if title_col:
            print(f"标题: {row.get(title_col, '未知')}")
        
        if date_col:
            date_val = row.get(date_col, '')
            if pd.notna(date_val):
                print(f"时间: {format_date(str(date_val))}")
        
        if source_col:
            source_val = row.get(source_col, '')
            if pd.notna(source_val):
                print(f"来源: {source_val}")
        
        if content_col:
            content = str(row.get(content_col, ''))
            if content and len(content) > 200:
                content = content[:200] + "..."
            if content:
                print(f"内容: {content}")
        
        print()
    
    print("="*80 + "\n")


def main():
    """主函数 - 获取最新5条新闻"""
    print("\n" + "="*80)
    print(" " * 30 + "📰 最新新闻热点")
    print("="*80 + "\n")
    
    # 获取最新5条股票新闻（市场热点）
    news_df = get_latest_news(count=5, stock_code=None, use_cctv=False)
    
    if news_df is not None:
        print_news_simple(news_df)
    else:
        print("❌ 获取新闻失败")


if __name__ == "__main__":
    """
    使用示例：
    
    方法1: 直接运行（获取市场热点新闻）
        python get_latest_news.py
    
    方法2: 在代码中使用
        from get_latest_news import get_latest_news, print_news_simple
        
        # 获取市场热点新闻
        news_df = get_latest_news(count=5)
        print_news_simple(news_df)
        
        # 获取指定股票的新闻（如上海瀚讯300762）
        news_df = get_latest_news(count=5, stock_code="300762")
        print_news_simple(news_df)
        
        # 获取CCTV新闻
        news_df = get_latest_news(count=5, use_cctv=True)
        print_news_simple(news_df)
    """
    
    # 默认获取市场热点新闻
    main()

