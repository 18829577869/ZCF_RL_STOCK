# -*- coding: utf-8 -*-
"""
使用 AkShare 获取最新的五个新闻热点

功能：
1. 获取最新的财经新闻资讯（最近4小时内的新闻）
2. 筛选出最新的五个新闻热点
3. 格式化输出并可选保存到文件
"""

import akshare as ak
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import json


def get_latest_news_akshare(count: int = 5, source: str = "auto", stock_code: str = None) -> Optional[pd.DataFrame]:
    """
    使用 AkShare 获取最新的新闻热点
    
    Args:
        count: 要获取的新闻数量，默认5条
        source: 数据源，可选 "stock"（股票新闻）、"cctv"（CCTV新闻）、"auto"（自动选择）、"hot"（热门股票汇总）
        stock_code: 股票代码（如 "300762"、"000001"），如果指定则获取该股票的新闻；如果为None且source为"stock"或"auto"，则获取热门股票汇总新闻
    
    Returns:
        DataFrame，包含新闻数据，失败返回None
    """
    news_df = None
    
    # 如果指定了股票代码，直接获取该股票的新闻
    if stock_code is not None:
        try:
            stock_name = f"股票代码{stock_code}"
            print(f"📰 正在从 AkShare 获取{stock_name}的股票新闻...")
            news_df = ak.stock_news_em(symbol=stock_code)
            
            if news_df is not None and len(news_df) > 0:
                print(f"✅ 成功获取 {len(news_df)} 条股票新闻")
                print(f"📋 数据列: {list(news_df.columns)}")
            else:
                return None
        except Exception as e:
            print(f"❌ 获取股票新闻失败: {e}")
            return None
    
    # 优先尝试获取热门股票汇总新闻（不限定单个股票）
    elif source == "auto" or source == "hot" or source == "stock":
        try:
            print(f"📰 正在从 AkShare 获取最新市场热点新闻（汇总多个热门股票）...")
            
            # 选择不同板块的代表性热门股票代码，获取它们的新闻并汇总
            # 使用多个不同板块的股票代码，这样可以获取更广泛的市场热点
            hot_stocks = [
                "000001",  # 平安银行（金融）
                "000002",  # 万科A（地产）
                "600519",  # 贵州茅台（消费）
                "000858",  # 五粮液（消费）
                "002415",  # 海康威视（科技）
                "300059",  # 东方财富（科技）
            ]
            
            all_news = []
            success_count = 0
            
            for code in hot_stocks:
                try:
                    temp_news = ak.stock_news_em(symbol=code)
                    if temp_news is not None and len(temp_news) > 0:
                        # 添加股票代码列以便识别
                        temp_news['来源股票'] = code
                        all_news.append(temp_news)
                        success_count += 1
                except:
                    continue
            
            if all_news:
                # 合并所有新闻
                news_df = pd.concat(all_news, ignore_index=True)
                
                # 按发布时间排序，去重（基于标题）
                if '新闻标题' in news_df.columns:
                    # 去重，保留第一条
                    news_df = news_df.drop_duplicates(subset=['新闻标题'], keep='first')
                    # 按发布时间降序排序（最新的在前）
                    if '发布时间' in news_df.columns:
                        news_df = news_df.sort_values('发布时间', ascending=False, na_position='last')
                    news_df = news_df.reset_index(drop=True)
                    # 移除临时添加的列
                    if '来源股票' in news_df.columns:
                        news_df = news_df.drop('来源股票', axis=1)
                
                print(f"✅ 成功从 {success_count} 个热门股票获取 {len(news_df)} 条市场热点新闻")
                print(f"📋 数据列: {list(news_df.columns)}")
            else:
                print(f"⚠️  未能从热门股票获取新闻，尝试使用单个股票...")
                # 如果汇总失败，回退到使用000001
                try:
                    news_df = ak.stock_news_em(symbol="000001")
                    if news_df is not None and len(news_df) > 0:
                        print(f"✅ 成功获取 {len(news_df)} 条股票新闻")
                except:
                    pass
                    
        except Exception as e:
            if source == "stock" or source == "hot":
                print(f"❌ 获取股票新闻失败: {e}")
                return None
            else:
                print(f"⚠️  股票新闻接口失败，尝试其他接口: {e}")
    
    # 如果股票新闻失败且是自动模式，尝试CCTV新闻
    if (news_df is None or len(news_df) == 0) and (source == "auto" or source == "cctv"):
        try:
            print(f"📰 正在从 AkShare 获取CCTV新闻...")
            news_df = ak.news_cctv()
            
            if news_df is not None and len(news_df) > 0:
                print(f"✅ 成功获取 {len(news_df)} 条CCTV新闻")
                print(f"📋 数据列: {list(news_df.columns)}")
        except Exception as e:
            if source == "cctv":
                print(f"❌ 获取CCTV新闻失败: {e}")
                return None
            else:
                print(f"⚠️  CCTV新闻接口失败: {e}")
    
    if news_df is None or len(news_df) == 0:
        print("❌ 所有新闻接口都无法获取数据")
        return None
    
    # 获取最新的指定数量新闻（通常已经是按时间排序的）
    actual_count = min(count, len(news_df))
    latest_news = news_df.head(actual_count)
    
    if actual_count < count:
        print(f"⚠️  实际获取 {actual_count} 条新闻（少于请求的 {count} 条）")
    
    return latest_news


def format_date(date_str: str) -> str:
    """
    格式化日期字符串，将"20240424"格式转换为"2024-04-24"
    
    Args:
        date_str: 日期字符串
    
    Returns:
        格式化后的日期字符串
    """
    if not date_str or pd.isna(date_str):
        return ""
    
    date_str = str(date_str).strip()
    
    # 如果已经是标准格式（包含"-"），直接返回
    if "-" in date_str:
        return date_str
    
    # 尝试格式化 YYYYMMDD 格式（如 20240424 -> 2024-04-24）
    if len(date_str) == 8 and date_str.isdigit():
        try:
            year = date_str[:4]
            month = date_str[4:6]
            day = date_str[6:8]
            return f"{year}-{month}-{day}"
        except:
            return date_str
    
    return date_str


def format_news_output(news_df: pd.DataFrame) -> List[Dict]:
    """
    格式化新闻数据为字典列表
    
    Args:
        news_df: 新闻DataFrame
    
    Returns:
        格式化后的新闻列表
    """
    news_list = []
    
    # 列名映射表（处理中英文列名）
    column_mapping = {
        "标题": ["标题", "title", "Title", "TITLE", "新闻标题"],
        "内容": ["内容", "content", "Content", "CONTENT", "正文", "body", "新闻内容"],
        "时间": ["时间", "time", "Time", "TIME", "日期", "date", "Date", "DATE", "发布时间"],
        "来源": ["来源", "source", "Source", "SOURCE", "出处", "文章来源"]
    }
    
    # 构建实际列名到标准列名的映射
    col_mapping_dict = {}
    for standard_col, possible_names in column_mapping.items():
        for possible_name in possible_names:
            if possible_name in news_df.columns:
                col_mapping_dict[possible_name] = standard_col
                break
    
    # 获取所有已映射的标准列
    mapped_cols = set(col_mapping_dict.values())
    
    for idx, row in news_df.iterrows():
        news_item = {"序号": idx + 1}
        
        # 提取标准列的数据
        for standard_col in ["标题", "内容", "时间", "来源"]:
            value = ""
            # 查找匹配的实际列名
            for actual_col, mapped_standard in col_mapping_dict.items():
                if mapped_standard == standard_col:
                    val = row.get(actual_col, "")
                    if pd.notna(val) and str(val).strip():
                        value = str(val).strip()
                        # 如果是时间字段，格式化日期
                        if standard_col == "时间":
                            value = format_date(value)
                        break
            news_item[standard_col] = value
        
        # 添加所有其他未映射的列（保留原始列名）
        for col in news_df.columns:
            if col not in col_mapping_dict:
                val = row.get(col, "")
                if pd.notna(val):
                    news_item[col] = str(val).strip()
        
        news_list.append(news_item)
    
    return news_list


def print_news(news_list: List[Dict]):
    """
    打印新闻列表
    
    Args:
        news_list: 新闻列表
    """
    print("\n" + "="*80)
    print(" " * 30 + "🔥 最新新闻热点")
    print("="*80)
    print(f"获取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"新闻数量: {len(news_list)} 条\n")
    
    for news in news_list:
        print("-" * 80)
        print(f"【新闻 {news.get('序号', '')}】")
        print(f"标题: {news.get('标题', '未知')}")
        if news.get('时间'):
            print(f"时间: {news.get('时间')}")
        if news.get('来源'):
            print(f"来源: {news.get('来源')}")
        if news.get('内容'):
            content = news.get('内容', '')
            # 如果内容太长，截取前200字符
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"内容: {content}")
        print()
    
    print("="*80 + "\n")


def save_news_to_file(news_list: List[Dict], filename: Optional[str] = None, format_type: str = "json"):
    """
    保存新闻到文件
    
    Args:
        news_list: 新闻列表
        filename: 文件名，如果为None则自动生成
        format_type: 保存格式，支持 "json" 和 "csv"
    """
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if format_type == "json":
            filename = f"latest_news_{timestamp}.json"
        else:
            filename = f"latest_news_{timestamp}.csv"
    
    try:
        if format_type == "json":
            output_data = {
                "获取时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "新闻数量": len(news_list),
                "新闻列表": news_list
            }
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
        else:  # csv
            df = pd.DataFrame(news_list)
            df.to_csv(filename, index=False, encoding='utf-8-sig')
        
        print(f"💾 新闻已保存到: {filename}")
        
    except Exception as e:
        print(f"⚠️  保存文件失败: {e}")


def get_stock_news_example(stock_code: str, stock_name: str = None, count: int = 5):
    """
    获取指定股票的新闻示例函数
    
    Args:
        stock_code: 股票代码（如 "300762"、"000001"）
        stock_name: 股票名称（可选，用于显示）
        count: 要获取的新闻数量，默认5条
    """
    if stock_name is None:
        stock_name = f"股票代码{stock_code}"
    
    print("\n" + "="*80)
    print(f" " * 25 + f"📰 {stock_name} 最新新闻")
    print("="*80 + "\n")
    
    # 获取指定股票的新闻
    news_df = get_latest_news_akshare(count=count, source="stock", stock_code=stock_code)
    
    if news_df is None:
        print(f"❌ 获取{stock_name}的新闻失败，请检查股票代码或网络连接")
        return
    
    # 格式化输出
    news_list = format_news_output(news_df)
    
    # 打印新闻
    print(f"\n【{stock_name}】最新 {len(news_list)} 条新闻：")
    print_news(news_list)
    
    return news_list


def main():
    """主函数"""
    print("\n" + "="*80)
    print(" " * 25 + "📰 AkShare 最新新闻热点获取工具")
    print("="*80)
    print("\n功能说明:")
    print("  - 获取最新的财经新闻资讯（汇总多个热门股票，涵盖不同板块）")
    print("  - 显示最新的5条新闻热点（按时间排序，去重）")
    print("  - 支持保存为 JSON 或 CSV 格式")
    print("  - 数据源：汇总多个热门股票新闻（金融、地产、消费、科技等），失败时使用CCTV新闻")
    print("  - 支持指定股票代码获取特定股票的新闻\n")
    
    # 获取最新的5条新闻（默认使用000001）
    news_df = get_latest_news_akshare(count=5, source="auto")
    
    if news_df is None:
        print("❌ 获取新闻失败，请检查网络连接或稍后重试")
        return
    
    # 格式化输出
    news_list = format_news_output(news_df)
    
    # 打印新闻
    print_news(news_list)
    
    # 询问是否保存
    try:
        save_choice = input("是否保存新闻到文件？(y/n，默认n): ").strip().lower()
        if save_choice == 'y':
            format_choice = input("选择保存格式 (json/csv，默认json): ").strip().lower()
            if format_choice not in ['json', 'csv']:
                format_choice = 'json'
            save_news_to_file(news_list, format_type=format_choice)
    except KeyboardInterrupt:
        print("\n\n用户取消操作")
    except Exception as e:
        print(f"\n⚠️  操作异常: {e}")
    
    print("\n✅ 程序执行完成\n")


if __name__ == "__main__":
    import sys
    
    # 如果通过命令行参数指定了股票代码，直接获取该股票的新闻
    if len(sys.argv) > 1:
        stock_code = sys.argv[1].strip()
        get_stock_news_example(stock_code=stock_code, count=5)
    else:
        main()


# ==================== 使用示例 ====================
# 
# 示例1: 获取上海瀚讯(300762)的新闻
#   get_stock_news_example(stock_code="300762", stock_name="上海瀚讯", count=5)
#
# 示例2: 获取平安银行(000001)的新闻
#   get_stock_news_example(stock_code="000001", stock_name="平安银行", count=5)
#
# 示例3: 命令行方式
#   python get_latest_news_akshare.py 300762
#
# ==================== 使用示例 ====================

