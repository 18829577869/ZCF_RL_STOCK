# -*- coding: utf-8 -*-
"""
获取上海瀚讯(300762)的最新新闻
示例脚本
"""

from get_latest_news_akshare import get_stock_news_example

if __name__ == "__main__":
    # 获取上海瀚讯的最新5条新闻
    get_stock_news_example(stock_code="300762", stock_name="上海瀚讯", count=5)

