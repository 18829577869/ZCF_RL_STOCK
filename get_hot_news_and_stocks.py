"""
获取上一个交易日至今的最热三个新闻，并识别相关度最大的三个股票
使用 DeepSeek API 进行分析

⚠️ 重要警告：
本工具生成的内容是基于 DeepSeek 模型的训练数据进行的推理和模拟，并非真实的市场新闻和股票关联数据。

局限性说明：
1. 缺乏实时数据访问：模型无法访问最新的新闻和市场数据，只能基于训练截止日期前的数据
2. 可能产生"幻觉"：模型可能生成看似合理但非真实的新闻和股票关联
3. 分析逻辑不同：基于文本关联推理，而非真实的市场资金流向和投资者情绪

因此，本工具生成的结果：
- ❌ 不能作为真实的投资参考依据
- ❌ 不能替代真实的财经新闻和数据
- ✅ 仅作为大语言模型能力的演示
- ✅ 可用于了解模型的分析思路和逻辑

如需真实的市场信息，请使用：
- 证券时报、澎湃新闻等权威财经媒体
- 同花顺、东方财富等专业财经平台
- Tushare、AkShare 等数据API获取真实交易数据
"""

import os
import json
import requests
import time
import pandas as pd
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional

# 尝试导入数据源
AKSHARE_AVAILABLE = False
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    print("⚠️  警告: AkShare 未安装，无法获取真实市场数据")
    print("💡 提示: 请运行 pip install akshare 安装")

# ==================== 配置参数 ====================
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'sk-167914945f7945d498e09a7f186c101d')
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"


# ==================== 辅助函数 ====================
def is_trading_day(date_obj: date = None) -> bool:
    """检查指定日期是否是交易日（周一到周五）"""
    if date_obj is None:
        date_obj = date.today()
    return date_obj.weekday() < 5  # 0-4 表示周一到周五


def get_last_trading_date() -> date:
    """获取上一个交易日"""
    current_date = date.today()
    # 从昨天开始往前找，最多找7天
    for i in range(1, 8):
        check_date = current_date - timedelta(days=i)
        if is_trading_day(check_date):
            return check_date
    # 如果找不到，返回昨天（作为备用）
    return current_date - timedelta(days=1)


def call_deepseek_api(prompt: str, max_tokens: int = 2000, temperature: float = 0.7) -> Optional[str]:
    """
    调用 DeepSeek API
    
    Args:
        prompt: 提示词
        max_tokens: 最大token数
        temperature: 温度参数
    
    Returns:
        API返回的文本内容，失败返回None
    """
    if not DEEPSEEK_API_KEY:
        print("⚠️  错误: 未设置 DEEPSEEK_API_KEY")
        return None
    
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": "你是一位专业的金融市场分析师和新闻分析师，擅长分析市场热点新闻和股票关联性。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(
            f"{DEEPSEEK_API_BASE}/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        
        if response.status_code != 200:
            print(f"⚠️  API 调用失败: HTTP {response.status_code}")
            print(f"响应内容: {response.text[:500]}")
            return None
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        return content.strip()
        
    except requests.exceptions.RequestException as e:
        print(f"⚠️  网络错误: {e}")
        return None
    except Exception as e:
        print(f"⚠️  调用异常: {e}")
        return None


def get_real_market_data(target_date: date) -> Dict:
    """
    获取真实的市场数据（涨停板股票、板块涨跌幅等）
    
    Args:
        target_date: 目标日期
    
    Returns:
        包含涨停板股票和板块数据的字典
    """
    if not AKSHARE_AVAILABLE:
        return {"error": "AkShare 未安装"}
    
    try:
        # 获取涨停板股票
        print(f"📊 正在获取 {target_date.strftime('%Y-%m-%d')} 的真实市场数据...")
        
        limit_up_stocks = []
        sector_data = []
        
        # 获取涨停板股票列表
        try:
            # 获取实时涨停板数据
            df_limit = ak.stock_zt_pool_em(date=target_date.strftime('%Y%m%d'))
            if df_limit is not None and len(df_limit) > 0:
                # 取前10只涨停股票
                for idx, row in df_limit.head(10).iterrows():
                    limit_up_stocks.append({
                        "code": str(row.get('代码', '')).strip(),
                        "name": str(row.get('名称', '')).strip(),
                        "latest_price": float(row.get('最新价', 0)),
                        "change_pct": float(row.get('涨跌幅', 0)),
                        "turnover_rate": float(row.get('换手率', 0)),
                        "volume_ratio": float(row.get('量比', 0))
                    })
                print(f"✅ 获取到 {len(limit_up_stocks)} 只涨停板股票")
        except Exception as e:
            print(f"⚠️  获取涨停板数据失败: {e}")
        
        # 获取板块涨跌幅排行
        try:
            df_sector = ak.stock_board_industry_name_em()
            if df_sector is not None and len(df_sector) > 0:
                # 按涨跌幅排序，取前5
                df_sector_sorted = df_sector.nlargest(5, '涨跌幅')
                for idx, row in df_sector_sorted.iterrows():
                    sector_data.append({
                        "name": str(row.get('板块名称', '')).strip(),
                        "change_pct": float(row.get('涨跌幅', 0)),
                        "stock_count": int(row.get('上涨家数', 0)),
                        "leader_stock": str(row.get('领涨股票', '')).strip()
                    })
                print(f"✅ 获取到 {len(sector_data)} 个热门板块")
        except Exception as e:
            print(f"⚠️  获取板块数据失败: {e}")
        
        return {
            "limit_up_stocks": limit_up_stocks,
            "hot_sectors": sector_data,
            "data_date": target_date.strftime("%Y-%m-%d"),
            "is_real_data": True
        }
        
    except Exception as e:
        print(f"⚠️  获取真实市场数据失败: {e}")
        return {"error": str(e)}


def analyze_real_market_data(market_data: Dict) -> List[Dict]:
    """
    基于真实市场数据，使用 DeepSeek 分析可能的新闻背景
    
    Args:
        market_data: 真实市场数据
    
    Returns:
        分析后的新闻列表
    """
    if "error" in market_data:
        return []
    
    limit_up_stocks = market_data.get("limit_up_stocks", [])
    hot_sectors = market_data.get("hot_sectors", [])
    data_date = market_data.get("data_date", "")
    
    if not limit_up_stocks and not hot_sectors:
        return []
    
    # 构建市场数据摘要
    market_summary = f"日期: {data_date}\n\n"
    
    if hot_sectors:
        market_summary += "热门板块（涨幅前5）:\n"
        for i, sector in enumerate(hot_sectors[:5], 1):
            market_summary += f"{i}. {sector['name']}: 涨幅{sector['change_pct']:.2f}%, "
            market_summary += f"上涨{sector['stock_count']}家, 领涨股票: {sector['leader_stock']}\n"
        market_summary += "\n"
    
    if limit_up_stocks:
        market_summary += "涨停板股票（前10只）:\n"
        for i, stock in enumerate(limit_up_stocks[:10], 1):
            market_summary += f"{i}. {stock['name']}({stock['code']}): "
            market_summary += f"涨幅{stock['change_pct']:.2f}%, 换手率{stock['turnover_rate']:.2f}%\n"
    
    prompt = f"""基于以下真实A股市场数据，分析可能导致这些股票和板块表现突出的最可能的三条新闻或事件。

真实市场数据：
{market_summary}

要求：
1. 基于真实的市场表现（涨停板股票、热门板块），推断最可能的原因
2. 分析这些股票和板块的共性，找出可能的新闻事件
3. 每个新闻需要包含：
   - 标题（title）- 明确标注这是"基于真实市场数据的分析"
   - 简要内容（content，100-200字）- 说明为什么这条新闻可能导致上述市场表现
   - 新闻日期（date，YYYY-MM-DD格式）- 使用数据日期
   - 热度说明（why_hot，为什么这个新闻最热，关联哪些股票/板块）
   - is_real_analysis（true，标记为基于真实数据的分析）
   - related_stocks（数组，关联的股票代码）

请以 JSON 格式返回，格式如下：
{{
  "news": [
    {{
      "title": "【基于真实数据】新闻标题1",
      "content": "新闻内容1，解释为什么导致上述股票/板块表现...",
      "date": "{data_date}",
      "why_hot": "为什么这个新闻最热...",
      "is_real_analysis": "true",
      "related_stocks": ["代码1", "代码2"]
    }},
    {{
      "title": "【基于真实数据】新闻标题2",
      "content": "新闻内容2...",
      "date": "{data_date}",
      "why_hot": "为什么这个新闻最热...",
      "is_real_analysis": "true",
      "related_stocks": ["代码3"]
    }},
    {{
      "title": "【基于真实数据】新闻标题3",
      "content": "新闻内容3...",
      "date": "{data_date}",
      "why_hot": "为什么这个新闻最热...",
      "is_real_analysis": "true",
      "related_stocks": ["代码4", "代码5"]
    }}
  ]
}}

只返回 JSON，不要其他文字。"""
    
    print(f"🔍 正在分析真实市场数据，推断可能的新闻背景...")
    response = call_deepseek_api(prompt, max_tokens=2500)
    
    if not response:
        return []
    
    # 提取 JSON
    content = response.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)
        if content.strip().startswith("json"):
            content = content[4:].strip()
    
    try:
        data = json.loads(content)
        news_list = data.get("news", [])
        if len(news_list) > 3:
            news_list = news_list[:3]
        return news_list
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON 解析失败: {e}")
        print(f"原始响应: {response[:500]}")
        return []


def get_hot_news(last_trading_date: date, current_date: date) -> List[Dict]:
    """
    获取上一个交易日至今的最热三个新闻
    优先使用真实市场数据，失败时回退到模型推理
    
    Args:
        last_trading_date: 上一个交易日
        current_date: 当前日期
    
    Returns:
        新闻列表，每个新闻包含 title, content, date 等字段
    """
    # 优先尝试获取真实市场数据
    if AKSHARE_AVAILABLE:
        market_data = get_real_market_data(last_trading_date)
        if "error" not in market_data and (market_data.get("limit_up_stocks") or market_data.get("hot_sectors")):
            print("✅ 成功获取真实市场数据，基于真实数据进行分析\n")
            news_list = analyze_real_market_data(market_data)
            if news_list:
                return news_list
            else:
                print("⚠️  基于真实数据未能生成新闻，回退到模拟模式\n")
        else:
            print("⚠️  无法获取真实市场数据，使用模拟模式\n")
    
    # 回退到模型推理模式
    last_date_str = last_trading_date.strftime("%Y-%m-%d")
    current_date_str = current_date.strftime("%Y-%m-%d")
    
    prompt = f"""⚠️ 重要说明：由于无法获取实时新闻数据，以下内容是基于模型训练数据和对市场常见热点模式的推理生成的模拟案例，并非真实的市场新闻。请明确告知用户这是模型推理结果，不能作为真实投资参考。

请基于常见的A股市场热点模式（如宏观经济政策、行业政策、重大事件等），模拟生成从 {last_date_str} 至今（{current_date_str}）可能引起市场关注的三个热点话题。

要求：
1. 明确标注这是"基于模型推理的模拟案例"，不是真实新闻
2. 选择对股市影响较大、典型的市场热点类型
3. 每个话题需要包含：
   - 标题（title）- 标注为"【模拟案例】"
   - 简要内容（content，100-200字）
   - 推测日期（date，YYYY-MM-DD格式）
   - 热度说明（why_hot，为什么这类话题可能引起关注）
   - is_simulated（true，标记为模拟数据）

请以 JSON 格式返回，格式如下：
{{
  "news": [
    {{
      "title": "【模拟案例】新闻标题1",
      "content": "新闻内容1...（注：这是基于模型训练的推理，非真实新闻）",
      "date": "2025-12-25",
      "why_hot": "为什么这类话题可能引起关注...",
      "is_simulated": true
    }},
    {{
      "title": "新闻标题2",
      "content": "新闻内容2...",
      "date": "2025-12-26",
      "why_hot": "为什么这个新闻最热..."
    }},
    {{
      "title": "新闻标题3",
      "content": "新闻内容3...",
      "date": "2025-12-27",
      "why_hot": "为什么这个新闻最热..."
    }}
  ]
}}

只返回 JSON，不要其他文字。"""
    
    print(f"📰 正在获取 {last_date_str} 至今的最热新闻...")
    response = call_deepseek_api(prompt, max_tokens=2000)
    
    if not response:
        return []
    
    # 提取 JSON（处理可能的 markdown 代码块）
    content = response.strip()
    if content.startswith("```"):
        # 移除代码块标记
        lines = content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)
        # 移除可能的 json 标记
        if content.strip().startswith("json"):
            content = content[4:].strip()
    
    try:
        data = json.loads(content)
        news_list = data.get("news", [])
        if len(news_list) > 3:
            news_list = news_list[:3]
        return news_list
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON 解析失败: {e}")
        print(f"原始响应: {response[:500]}")
        return []


def analyze_stock_relevance(news_list: List[Dict], market_data: Dict = None) -> List[Dict]:
    """
    分析新闻与股票的相关度，找出相关度最大的三个股票
    如果提供了真实市场数据，优先使用真实股票
    
    Args:
        news_list: 新闻列表
        market_data: 真实市场数据（可选）
    
    Returns:
        股票列表，每个股票包含 code, name, relevance_score, related_news 等字段
    """
    if not news_list:
        return []
    
    # 检查是否是真实数据分析
    is_real_analysis = any(news.get('is_real_analysis', False) for news in news_list)
    
    # 构建新闻摘要
    news_summary = "\n".join([
        f"话题{i+1}: {news.get('title', '')} - {news.get('content', '')[:100]}"
        for i, news in enumerate(news_list)
    ])
    
    # 如果有真实市场数据，添加股票信息
    real_stocks_info = ""
    if market_data and is_real_analysis:
        limit_up_stocks = market_data.get("limit_up_stocks", [])
        if limit_up_stocks:
            real_stocks_info = "\n真实涨停板股票:\n" + "\n".join([
                f"- {s['name']}({s['code']}): 涨幅{s['change_pct']:.2f}%"
                for s in limit_up_stocks[:10]
            ])
    
    if is_real_analysis:
        prompt = f"""基于以下基于真实市场数据分析出的新闻，请找出相关度最高的三个A股股票。

分析出的新闻：
{news_summary}
{real_stocks_info}

要求：
1. 优先考虑新闻中提到的股票代码（related_stocks字段）
2. 如果没有明确股票代码，根据新闻内容关联真实涨停板股票
3. 每个股票需要包含：
   - 股票代码（code）
   - 股票名称（name）
   - 相关度评分（relevance_score，0-100）
   - 相关新闻索引（related_news，数组）
   - 相关性说明（relevance_reason）
   - is_real_analysis（true）

请以 JSON 格式返回，格式如下：
{{
  "stocks": [
    {{
      "code": "代码1",
      "name": "股票名称1",
      "relevance_score": 95,
      "related_news": [0, 1],
      "relevance_reason": "相关性说明...",
      "is_real_analysis": true
    }}
  ]
}}

只返回 JSON，不要其他文字。"""
    
    else:
        prompt = f"""⚠️ 重要说明：以下话题是基于模型推理的模拟案例，并非真实新闻。请基于这些话题类型，分析可能受影响的A股股票。

基于以下三个市场热点话题（模拟案例），请分析并找出可能相关度较高的三个A股股票。

注意：
1. 这是基于模型推理的分析，不是真实的市场关联
2. 模型会倾向于选择与宏观话题相关的大型龙头股
3. 真实市场热点通常会直接催生特定板块的中小盘概念股走强

话题内容：
{news_summary}

要求：
1. 分析每个新闻可能影响的股票
2. 综合考虑三个新闻，找出相关度最大的三个股票
3. 每个股票需要包含：
   - 股票代码（code，格式如 600036、000001、300661）
   - 股票名称（name）
   - 相关度评分（relevance_score，0-100，越高表示相关度越大）
   - 相关新闻索引（related_news，数组，如 [0, 2] 表示与新闻1和新闻3相关）
   - 相关性说明（relevance_reason）
   - is_real_analysis（{str(is_real_analysis).lower()}，是否基于真实数据分析）

请以 JSON 格式返回，格式如下：
{{
  "stocks": [
    {{
      "code": "600036",
      "name": "招商银行",
      "relevance_score": 95,
      "related_news": [0, 1],
      "relevance_reason": "该股票可能与话题1和话题2相关，因为...",
      "is_real_analysis": {str(is_real_analysis).lower()}
    }},
    {{
      "code": "000001",
      "name": "平安银行",
      "relevance_score": 88,
      "related_news": [0],
      "relevance_reason": "该股票与新闻1相关，因为..."
    }},
    {{
      "code": "300661",
      "name": "圣邦股份",
      "relevance_score": 82,
      "related_news": [2],
      "relevance_reason": "该股票与新闻3相关，因为..."
    }}
  ]
}}

只返回 JSON，不要其他文字。"""
    
    print("🔍 正在分析股票相关度...")
    response = call_deepseek_api(prompt, max_tokens=2000)
    
    if not response:
        return []
    
    # 提取 JSON（处理可能的 markdown 代码块）
    content = response.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)
        if content.strip().startswith("json"):
            content = content[4:].strip()
    
    try:
        data = json.loads(content)
        stocks = data.get("stocks", [])
        # 按相关度评分排序
        stocks.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        if len(stocks) > 3:
            stocks = stocks[:3]
        return stocks
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON 解析失败: {e}")
        print(f"原始响应: {response[:500]}")
        return []


def format_output(news_list: List[Dict], stocks: List[Dict], last_trading_date: date, current_date: date, market_data: Dict = None):
    """
    格式化输出结果
    
    Args:
        news_list: 新闻列表
        stocks: 股票列表
        last_trading_date: 上一个交易日
        current_date: 当前日期
        market_data: 真实市场数据（可选）
    """
    # 检查是否使用真实数据
    is_real_data = any(news.get('is_real_analysis', False) for news in news_list)
    
    print("\n" + "="*80)
    if is_real_data:
        print(" " * 20 + "📰 基于真实市场数据的新闻分析报告")
        print("="*80)
        print("\n✅ 本报告基于真实的市场数据（涨停板股票、板块涨跌幅）进行分析")
        print("⚠️  新闻背景是基于市场表现推断的可能原因，仅供参考")
        print("💡 建议：结合权威财经媒体确认具体新闻事件")
    else:
        print(" " * 20 + "📰 市场热点话题与股票关联分析报告（模拟模式）")
        print("="*80)
        print("\n" + "⚠️" * 40)
        print(" " * 15 + "【重要警告：模型推理结果，非真实市场数据】")
        print("⚠️" * 40)
        print("\n本报告基于 DeepSeek 模型的训练数据推理生成，存在以下局限性：")
        print("1. ❌ 无法访问实时新闻和市场数据")
        print("2. ❌ 可能生成看似合理但非真实的内容（'幻觉'）")
        print("3. ❌ 分析逻辑与真实市场资金流向不同")
        print("\n⚠️  因此，本报告：")
        print("   - 不能作为真实的投资参考依据")
        print("   - 不能替代真实的财经新闻和数据")
        print("   - 仅作为大语言模型能力的演示")
    print("\n" + "-"*80 + "\n")
    print(f"📅 分析时间范围: {last_trading_date.strftime('%Y-%m-%d')} 至 {current_date.strftime('%Y-%m-%d')}")
    print(f"📅 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 输出新闻
    print("="*80)
    if is_real_data:
        print("🔥 基于真实市场数据推断的新闻（前3条）")
    else:
        print("🔥 模拟的市场热点话题（基于模型推理）")
    print("="*80)
    for i, news in enumerate(news_list, 1):
        print(f"\n【新闻 {i}】")
        print(f"标题: {news.get('title', '未知')}")
        print(f"日期: {news.get('date', '未知')}")
        print(f"内容: {news.get('content', '未知')}")
        print(f"热度说明: {news.get('why_hot', '未知')}")
    
    # 输出股票
    print("\n" + "="*80)
    if is_real_data:
        print("📈 相关度最高的股票（基于真实数据分析）")
    else:
        print("📈 可能相关的股票（基于模型推理，非真实市场关联）")
    print("="*80)
    for i, stock in enumerate(stocks, 1):
        print(f"\n【股票 {i}】")
        print(f"代码: {stock.get('code', '未知')}")
        print(f"名称: {stock.get('name', '未知')}")
        print(f"相关度评分: {stock.get('relevance_score', 0)}/100")
        related_news_indices = stock.get('related_news', [])
        related_news_str = "、".join([f"新闻{j+1}" for j in related_news_indices])
        print(f"相关新闻: {related_news_str if related_news_str else '无'}")
        print(f"相关性说明: {stock.get('relevance_reason', '未知')}")
    
    # 如果有真实市场数据，显示原始数据
    if market_data and "error" not in market_data:
        print("\n" + "="*80)
        print("📊 原始市场数据（参考）")
        print("="*80)
        hot_sectors = market_data.get("hot_sectors", [])
        if hot_sectors:
            print("\n热门板块:")
            for sector in hot_sectors[:5]:
                print(f"  - {sector['name']}: 涨幅{sector['change_pct']:.2f}% (上涨{sector['stock_count']}家)")
        limit_up_stocks = market_data.get("limit_up_stocks", [])
        if limit_up_stocks:
            print("\n涨停板股票:")
            for stock in limit_up_stocks[:5]:
                print(f"  - {stock['name']}({stock['code']}): 涨幅{stock['change_pct']:.2f}%")
    
    print("\n" + "="*80)
    is_real_data = any(news.get('is_real_analysis', False) for news in news_list)
    if is_real_data:
        print("✅ 基于真实数据的分析完成")
    else:
        print("✅ 模型推理分析完成")
    print("="*80)
    print("\n💡 提示：如需真实市场信息，请访问：")
    print("   - 证券时报、澎湃新闻等权威财经媒体")
    print("   - 同花顺、东方财富等专业财经平台")
    print("   - Tushare、AkShare 等数据API")
    print("="*80 + "\n")


def save_to_json(news_list: List[Dict], stocks: List[Dict], last_trading_date: date, current_date: date):
    """
    保存结果到 JSON 文件（包含警告信息）
    
    Args:
        news_list: 新闻列表（模拟数据）
        stocks: 股票列表（模拟数据）
        last_trading_date: 上一个交易日
        current_date: 当前日期
    """
    output_data = {
        "warning": {
            "is_simulated": True,
            "warning_text": "本数据基于 DeepSeek 模型推理生成，并非真实的市场新闻和股票关联数据。不能作为真实的投资参考依据。",
            "limitations": [
                "缺乏实时数据访问能力",
                "可能产生'幻觉'内容",
                "分析逻辑与真实市场不同"
            ],
            "recommended_sources": [
                "证券时报、澎湃新闻等权威财经媒体",
                "同花顺、东方财富等专业财经平台",
                "Tushare、AkShare 等数据API"
            ]
        },
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "date_range": {
            "start": last_trading_date.strftime("%Y-%m-%d"),
            "end": current_date.strftime("%Y-%m-%d")
        },
        "news": news_list,
        "stocks": stocks
    }
    
    filename = f"hot_news_stocks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"💾 结果已保存到: {filename}")
    except Exception as e:
        print(f"⚠️  保存文件失败: {e}")


# ==================== 主函数 ====================
def main():
    """主函数"""
    print("\n" + "="*80)
    print(" " * 15 + "📰 市场热点话题与股票关联分析（模型推理版）")
    print("="*80)
    print("\n⚠️  重要提示：本工具生成的是基于模型推理的模拟案例，")
    print("   并非真实的市场新闻和股票关联数据，不能作为投资参考。")
    print("="*80 + "\n")
    
    # 检查 API Key
    if not DEEPSEEK_API_KEY:
        print("⚠️  错误: 未设置 DEEPSEEK_API_KEY 环境变量")
        print("💡 提示: 请设置环境变量 DEEPSEEK_API_KEY 或在代码中配置")
        return
    
    # 获取上一个交易日
    last_trading_date = get_last_trading_date()
    current_date = date.today()
    
    print(f"📅 上一个交易日: {last_trading_date.strftime('%Y-%m-%d')}")
    print(f"📅 当前日期: {current_date.strftime('%Y-%m-%d')}\n")
    
    # 获取真实市场数据（如果需要）
    market_data = None
    if AKSHARE_AVAILABLE:
        market_data = get_real_market_data(last_trading_date)
    
    # 获取热门新闻（优先使用真实数据）
    news_list = get_hot_news(last_trading_date, current_date)
    
    if not news_list:
        print("⚠️  未能获取到新闻，请检查网络连接和 API 配置")
        return
    
    is_real_data = any(news.get('is_real_analysis', False) for news in news_list)
    if is_real_data:
        print(f"✅ 基于真实市场数据生成了 {len(news_list)} 条新闻分析\n")
    else:
        print(f"⚠️  使用模拟模式生成了 {len(news_list)} 条热点话题\n")
    
    # 分析股票相关度
    stocks = analyze_stock_relevance(news_list, market_data)
    
    if not stocks:
        print("⚠️  未能分析出相关股票，请检查网络连接和 API 配置")
        return
    
    print(f"✅ 模型已分析出 {len(stocks)} 只可能相关的股票\n")
    
    # 格式化输出
    format_output(news_list, stocks, last_trading_date, current_date, market_data)
    
    # 保存到文件
    save_to_json(news_list, stocks, last_trading_date, current_date)


if __name__ == "__main__":
    main()

