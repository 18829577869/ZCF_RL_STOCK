"""
V8 - LLM 市场情报增强模块
支持 DeepSeek 和 Grok，提供宏观经济、新闻舆情、市场情绪等关键信息
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Optional
import requests
import pandas as pd


class MarketIntelligenceAgent:
    """
    市场情报分析代理，使用 LLM 分析市场环境
    
    支持的信息维度：
    1. 宏观经济指标（GDP、CPI、利率）
    2. 新闻和舆情分析
    3. 市场情绪（恐慌指数 VIX）
    4. 资金流向
    5. 政策变化
    6. 国际市场联动
    7. 突发事件
    """
    
    def __init__(
        self,
        provider: str = "deepseek",  # "deepseek" 或 "grok"
        api_key: Optional[str] = None,
        cache_dir: str = "./market_intelligence_cache/",
        enable_cache: bool = True
    ):
        """
        初始化市场情报代理
        
        Args:
            provider: LLM 提供商，支持 "deepseek" 或 "grok"
            api_key: API 密钥（如果为 None，会从环境变量读取）
            cache_dir: 缓存目录
            enable_cache: 是否启用缓存
        """
        self.provider = provider.lower()
        self.cache_dir = cache_dir
        self.enable_cache = enable_cache
        
        # 创建缓存目录
        if self.enable_cache:
            os.makedirs(cache_dir, exist_ok=True)
            # 创建 provider 子目录（如 deepseek/ 或 grok/）
            provider_dir = os.path.join(cache_dir, self.provider)
            os.makedirs(provider_dir, exist_ok=True)
        
        # 设置 API 配置
        if self.provider == "deepseek":
            self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            self.api_base = "https://api.deepseek.com/v1"
            self.model = "deepseek-chat"
        elif self.provider == "grok":
            self.api_key = api_key or os.getenv("GROK_API_KEY")
            self.api_base = "https://api.x.ai/v1"
            self.model = "grok-beta"
        else:
            raise ValueError(f"不支持的提供商: {provider}，请使用 'deepseek' 或 'grok'")
        
        if not self.api_key:
            print(f"[警告] 未设置 API 密钥，请设置环境变量 {self.provider.upper()}_API_KEY")
            print(f"[提示] 将使用模拟数据进行训练")
            self.mock_mode = True
        else:
            self.mock_mode = False
        
        print(f"[初始化] 市场情报代理: {self.provider.upper()}")
        print(f"[缓存] {'启用' if self.enable_cache else '禁用'}: {cache_dir}")
    
    def _get_cache_path(self, date: str) -> str:
        """获取缓存文件路径"""
        # 文件保存在 provider 子目录下（如 market_intelligence_cache/deepseek/）
        provider_dir = os.path.join(self.cache_dir, self.provider)
        return os.path.join(provider_dir, f"{date}_{self.provider}.json")
    
    def _load_from_cache(self, date: str) -> Optional[Dict]:
        """从缓存加载数据"""
        if not self.enable_cache:
            return None
        
        cache_path = self._get_cache_path(date)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[警告] 缓存加载失败: {e}")
                return None
        return None
    
    def _save_to_cache(self, date: str, data: Dict):
        """保存到缓存"""
        if not self.enable_cache:
            return
        
        cache_path = self._get_cache_path(date)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[警告] 缓存保存失败: {e}")
    
    def _generate_mock_intelligence(self, date: str) -> Dict:
        """生成模拟市场情报（用于训练和测试）"""
        import random
        import hashlib
        
        # 使用日期作为种子，确保同一天的数据一致
        seed = int(hashlib.md5(date.encode()).hexdigest()[:8], 16)
        random.seed(seed)
        
        # 添加一些趋势性（周期性波动）
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        day_of_year = date_obj.timetuple().tm_yday
        trend = 0.3 * (day_of_year % 60 - 30) / 30  # -0.3 到 0.3 的周期
        
        return {
            "date": date,
            "macro_economic_score": round(random.uniform(-0.3, 0.5) + trend * 0.5, 3),  # -1 到 1
            "market_sentiment_score": round(random.uniform(-0.4, 0.6) + trend, 3),  # -1 到 1
            "risk_level": round(random.uniform(0.2, 0.7), 3),  # 0 到 1
            "policy_impact_score": round(random.uniform(-0.2, 0.3), 3),  # -1 到 1
            "emergency_impact_score": round(random.uniform(-0.1, 0.1), 3),  # -1 到 1
            "capital_flow_score": round(random.uniform(-0.3, 0.4) + trend * 0.8, 3),  # -1 到 1
            "international_correlation": round(random.uniform(0.3, 0.8), 3),  # 0 到 1
            "vix_level": round(random.uniform(12, 25), 2),  # 实际 VIX 范围
            "source": "mock_data",
            "timestamp": datetime.now().isoformat()
        }
    
    def _call_llm_api(self, date: str, market_context: str = "", company_financial_info: str = "") -> Dict:
        """
        调用 LLM API 分析市场情报
        
        Args:
            date: 日期字符串 (YYYY-MM-DD)
            market_context: 额外的市场背景信息
            company_financial_info: 公司财务信息（年报、财务指标等）
        
        Returns:
            结构化的市场情报字典
        """
        prompt = f"""你是一位资深的金融市场分析师。请分析 {date} 的中国A股市场环境，提供以下维度的评分和分析：

1. **宏观经济评分** (macro_economic_score): GDP增长、CPI通胀、利率政策的综合影响 [-1 到 1，-1极度悲观，0中性，1极度乐观]
2. **市场情绪评分** (market_sentiment_score): 投资者情绪、恐慌指数VIX、市场人气 [-1 到 1]
3. **风险等级** (risk_level): 当前市场的整体风险水平 [0 到 1，0极低风险，1极高风险]
4. **政策影响评分** (policy_impact_score): 货币政策、财政政策、监管政策的影响 [-1 到 1]
5. **突发事件影响** (emergency_impact_score): 地缘政治、疫情、自然灾害等 [-1 到 1]
6. **资金流向评分** (capital_flow_score): 外资流入流出、北向资金、融资融券 [-1 到 1]
7. **国际联动系数** (international_correlation): 与美股、港股的联动程度 [0 到 1]
8. **VIX水平** (vix_level): 恐慌指数的具体数值 [10-40]

{market_context}

{company_financial_info if company_financial_info else ""}

请以 JSON 格式返回，只返回 JSON，不要其他文字：
{{
  "macro_economic_score": 0.2,
  "market_sentiment_score": 0.3,
  "risk_level": 0.4,
  "policy_impact_score": 0.1,
  "emergency_impact_score": 0.0,
  "capital_flow_score": 0.2,
  "international_correlation": 0.6,
  "vix_level": 18.5,
  "reasoning": "简短的分析理由（1-2句话）"
}}"""

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "你是一位专业的金融市场分析师，擅长量化分析和风险评估。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,  # 降低温度以获得更稳定的输出
                "max_tokens": 500
            }
            
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"[错误] API 调用失败: HTTP {response.status_code}")
                print(f"[响应] {response.text[:500]}")  # 只显示前500字符
                error_info = f"API调用失败 (HTTP {response.status_code})"
                if response.status_code == 401:
                    error_info += " - API 密钥可能无效或已过期"
                elif response.status_code == 429:
                    error_info += " - API 调用频率超限"
                elif response.status_code >= 500:
                    error_info += " - 服务器错误，请稍后重试"
                print(f"[错误详情] {error_info}")
                print(f"[回退] 使用模拟数据")
                return self._generate_mock_intelligence(date)
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # 提取 JSON（处理可能的 markdown 代码块）
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            intelligence = json.loads(content)
            intelligence["date"] = date
            intelligence["source"] = self.provider
            intelligence["timestamp"] = datetime.now().isoformat()
            
            return intelligence
            
        except requests.exceptions.RequestException as e:
            print(f"[网络错误] LLM API 调用失败: {e}")
            print(f"[错误类型] 网络连接问题")
            print(f"[回退] 使用模拟数据")
            return self._generate_mock_intelligence(date)
        except json.JSONDecodeError as e:
            print(f"[解析错误] API 返回的 JSON 格式无效: {e}")
            print(f"[回退] 使用模拟数据")
            return self._generate_mock_intelligence(date)
        except Exception as e:
            print(f"[未知错误] LLM API 调用异常: {e}")
            import traceback
            print(f"[详细错误] {traceback.format_exc()}")
            print(f"[回退] 使用模拟数据")
            return self._generate_mock_intelligence(date)
    
    def get_market_intelligence(
        self,
        date: str,
        market_context: str = "",
        company_financial_info: str = "",
        force_refresh: bool = False
    ) -> Dict:
        """
        获取指定日期的市场情报
        
        Args:
            date: 日期字符串 (YYYY-MM-DD)
            market_context: 额外的市场背景信息
            company_financial_info: 公司财务信息（年报、财务指标等）
            force_refresh: 是否强制刷新（忽略缓存）
        
        Returns:
            市场情报字典
        """
        # 1. 检查缓存（如果有财务信息，不缓存，因为财务信息会变化）
        if not force_refresh and not company_financial_info:
            cached = self._load_from_cache(date)
            if cached is not None:
                return cached
        
        # 2. 调用 LLM 或使用模拟数据
        if self.mock_mode:
            intelligence = self._generate_mock_intelligence(date)
        else:
            intelligence = self._call_llm_api(date, market_context, company_financial_info)
            time.sleep(0.5)  # 避免频繁调用
        
        # 3. 保存到缓存（如果有财务信息，不缓存）
        if not company_financial_info:
            self._save_to_cache(date, intelligence)
        
        return intelligence
    
    def batch_generate_intelligence(
        self,
        start_date: str,
        end_date: str,
        use_mock: bool = True
    ):
        """
        批量生成历史日期的市场情报（用于训练数据准备）
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            use_mock: 是否使用模拟数据（推荐，避免大量 API 调用）
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        print(f"\n{'='*70}")
        print(f"批量生成市场情报: {start_date} 至 {end_date}")
        print(f"总计: {len(date_range)} 天")
        print(f"模式: {'模拟数据' if use_mock else 'LLM API'}")
        print(f"{'='*70}\n")
        
        original_mock_mode = self.mock_mode
        if use_mock:
            self.mock_mode = True
        
        success_count = 0
        for i, date in enumerate(date_range):
            date_str = date.strftime("%Y-%m-%d")
            
            # 跳过已缓存的
            if self._load_from_cache(date_str) is not None:
                continue
            
            try:
                self.get_market_intelligence(date_str)
                success_count += 1
                
                if (i + 1) % 100 == 0:
                    print(f"[进度] {i+1}/{len(date_range)} 完成")
            
            except Exception as e:
                print(f"[错误] {date_str} 生成失败: {e}")
        
        self.mock_mode = original_mock_mode
        
        print(f"\n[完成] 成功生成 {success_count} 天的市场情报")
        print(f"[缓存] {self.cache_dir}\n")
    
    def get_feature_vector(self, intelligence: Dict) -> list:
        """
        将市场情报转换为特征向量（用于观察空间）
        
        Returns:
            8维特征向量
        """
        return [
            intelligence.get("macro_economic_score", 0.0),
            intelligence.get("market_sentiment_score", 0.0),
            intelligence.get("risk_level", 0.5),
            intelligence.get("policy_impact_score", 0.0),
            intelligence.get("emergency_impact_score", 0.0),
            intelligence.get("capital_flow_score", 0.0),
            intelligence.get("international_correlation", 0.5),
            intelligence.get("vix_level", 20.0) / 40.0  # 归一化到 0-1
        ]


# 测试代码
if __name__ == "__main__":
    print("=== 市场情报代理测试 ===\n")
    
    # 1. 测试单日获取
    agent = MarketIntelligenceAgent(provider="deepseek", enable_cache=True)
    
    test_date = "2024-12-01"
    intelligence = agent.get_market_intelligence(test_date)
    
    print(f"日期: {intelligence['date']}")
    print(f"宏观经济评分: {intelligence['macro_economic_score']:+.3f}")
    print(f"市场情绪评分: {intelligence['market_sentiment_score']:+.3f}")
    print(f"风险等级: {intelligence['risk_level']:.3f}")
    print(f"政策影响评分: {intelligence['policy_impact_score']:+.3f}")
    print(f"突发事件影响: {intelligence['emergency_impact_score']:+.3f}")
    print(f"资金流向评分: {intelligence['capital_flow_score']:+.3f}")
    print(f"国际联动系数: {intelligence['international_correlation']:.3f}")
    print(f"VIX水平: {intelligence['vix_level']:.2f}")
    print(f"数据来源: {intelligence['source']}")
    
    # 2. 测试特征向量
    print(f"\n特征向量: {agent.get_feature_vector(intelligence)}")
    
    # 3. 测试批量生成（模拟模式）
    print("\n=== 测试批量生成 ===")
    agent.batch_generate_intelligence(
        start_date="2023-01-01",
        end_date="2023-01-10",
        use_mock=True
    )
    
    print("\n测试完成！")



