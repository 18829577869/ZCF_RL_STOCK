"""
多数据源组合管理模块
支持 Tushare、AkShare、baostock 等多个数据源的组合使用
实现数据源优先级、容错切换、数据合并等功能
"""

import pandas as pd
import numpy as np
import datetime
import time
from typing import Dict, List, Optional, Tuple
import warnings
import os

# 导入反爬虫工具
try:
    from .anti_crawler_pool import get_global_pool, setup_akshare_environment, monkey_patch_requests
    ANTI_CRAWLER_AVAILABLE = True
except ImportError:
    try:
        from anti_crawler_pool import get_global_pool, setup_akshare_environment, monkey_patch_requests
        ANTI_CRAWLER_AVAILABLE = True
    except ImportError:
        ANTI_CRAWLER_AVAILABLE = False
        warnings.warn("反爬虫工具模块未找到，将使用默认请求方式")


class MultiDataSourceManager:
    """多数据源组合管理器"""
    
    def __init__(self, 
                 stock_code: str,
                 sources: Optional[List[str]] = None,
                 priority: Optional[List[str]] = None,
                 timeout: int = 10,
                 retry_times: int = 3,
                 enable_anti_crawler: bool = True,
                 proxies: Optional[List[str]] = None):
        """
        初始化多数据源管理器
        
        参数:
            stock_code: 股票代码（如 'sh.600036' 或 '600036'）
            sources: 可用数据源列表，None则自动检测
            priority: 数据源优先级列表，None则使用默认优先级
            timeout: 请求超时时间（秒）
            retry_times: 重试次数
            enable_anti_crawler: 是否启用反爬虫功能（Cookie/UA/代理池）
            proxies: 代理列表，格式如 ['http://user:pass@host:port', ...]
        """
        self.stock_code = stock_code
        self.timeout = timeout
        self.retry_times = retry_times
        self.enable_anti_crawler = enable_anti_crawler and ANTI_CRAWLER_AVAILABLE
        
        # 初始化反爬虫池
        if self.enable_anti_crawler:
            self.anti_crawler_pool = get_global_pool()
            if proxies:
                self.anti_crawler_pool.add_proxies(proxies)
            # 对requests进行monkey patch
            monkey_patch_requests(self.anti_crawler_pool)
            print(f"🛡️  反爬虫功能已启用 (UA池: {len(self.anti_crawler_pool.user_agents)}个, 代理池: {len(self.anti_crawler_pool.proxies_pool)}个)")
        else:
            self.anti_crawler_pool = None
        
        # 检测可用数据源
        self.available_sources = self._detect_available_sources()
        
        # 设置数据源列表
        if sources is None:
            self.sources = self.available_sources
        else:
            self.sources = [s for s in sources if s in self.available_sources]
        
        # 设置优先级（默认：stockapi > tushare > akshare > baostock）
        if priority is None:
            # V14: 如果StockAPI可用，设置为最高优先级
            if 'stockapi' in self.available_sources:
                self.priority = ['stockapi', 'tushare', 'akshare', 'baostock']
            else:
                self.priority = ['tushare', 'akshare', 'baostock']
        else:
            self.priority = [p for p in priority if p in self.available_sources]
        
        # 数据源状态统计
        self.source_stats = {source: {'success': 0, 'fail': 0, 'last_success': None} 
                            for source in self.sources}
        
        # 当前使用的数据源
        self.current_source = None
        
        print(f"✅ 多数据源管理器初始化完成")
        print(f"   可用数据源: {', '.join(self.available_sources)}")
        print(f"   优先级顺序: {' > '.join(self.priority)}")
    
    def _detect_available_sources(self) -> List[str]:
        """检测可用的数据源"""
        available = []
        
        # V14: 检测 StockAPI
        try:
            import requests
            # 检查是否有StockAPI配置
            stockapi_key = os.getenv('STOCKAPI_API_KEY', '')
            if stockapi_key:
                available.append('stockapi')
        except ImportError:
            pass
        
        # 检测 Tushare
        try:
            import tushare as ts
            # 尝试获取token（如果设置了）
            token = self._get_tushare_token()
            if token:
                ts.set_token(token)
                available.append('tushare')
        except ImportError:
            pass
        
        # 检测 AkShare
        try:
            import akshare as ak
            available.append('akshare')
        except ImportError:
            pass
        
        # 检测 baostock
        try:
            import baostock as bs
            available.append('baostock')
        except ImportError:
            pass
        
        return available
    
    def _get_tushare_token(self) -> Optional[str]:
        """获取 Tushare token（从环境变量或配置文件）"""
        import os
        return os.getenv('TUSHARE_TOKEN', '')
    
    def _convert_stock_code(self, code: str, target_source: str) -> str:
        """
        转换股票代码格式以适应不同数据源
        
        参数:
            code: 原始股票代码（如 'sh.600036' 或 '600036'）
            target_source: 目标数据源名称
        
        返回:
            转换后的股票代码
        """
        # 解析代码
        if '.' in code:
            market, num = code.split('.')
        else:
            # 根据代码判断市场
            if code.startswith('6'):
                market = 'sh'
                num = code
            else:
                market = 'sz'
                num = code
        
        # 转换为目标格式
        if target_source == 'stockapi':
            # V14: StockAPI格式，如SH600036或SZ002241
            return f"{market.upper()}{num}"
        elif target_source == 'tushare':
            return f"{num}.{market.upper()}"
        elif target_source == 'akshare':
            return num
        elif target_source == 'baostock':
            return f"{market}.{num}"
        else:
            return code
    
    def _fetch_from_tushare(self, days: int = 7) -> Optional[pd.DataFrame]:
        """从 Tushare 获取数据"""
        try:
            import tushare as ts
            token = self._get_tushare_token()
            if not token:
                return None
            
            ts.set_token(token)
            pro = ts.pro_api()
            
            code = self._convert_stock_code(self.stock_code, 'tushare')
            today = datetime.date.today()
            start_date = (today - datetime.timedelta(days=days)).strftime('%Y%m%d')
            end_date = today.strftime('%Y%m%d')
            
            # 尝试获取5分钟K线
            try:
                df = pro.stk_mins(
                    ts_code=code,
                    freq='5min',
                    start_date=start_date + '0930',
                    end_date=end_date + '1500'
                )
                if df is not None and len(df) > 0:
                    # 转换格式
                    if 'trade_time' in df.columns:
                        df['time'] = pd.to_datetime(df['trade_time']).dt.strftime('%Y%m%d%H%M%S')
                        df['date'] = pd.to_datetime(df['trade_time']).dt.strftime('%Y-%m-%d')
                    df = df.rename(columns={'close': 'close', 'vol': 'volume'})
                    return df[['date', 'time', 'close', 'volume']]
            except:
                # 如果5分钟数据失败，尝试日线数据
                df = pro.daily(
                    ts_code=code,
                    start_date=start_date,
                    end_date=end_date
                )
                if df is not None and len(df) > 0:
                    df = df.rename(columns={'trade_date': 'date', 'close': 'close', 'vol': 'volume'})
                    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
                    df['time'] = df['date'] + '15000000'
                    return df[['date', 'time', 'close', 'volume']]
            
            return None
        except Exception as e:
            warnings.warn(f"Tushare 获取数据失败: {e}")
            return None
    
    def _fetch_from_akshare(self, days: int = 7) -> Optional[pd.DataFrame]:
        """从 AkShare 获取数据（带反爬虫保护）"""
        # 设置反爬虫环境
        if self.enable_anti_crawler and self.anti_crawler_pool:
            setup_akshare_environment(self.anti_crawler_pool)
            # 添加随机延迟
            self.anti_crawler_pool.random_delay(0.5, 1.5)
        
        max_retries = self.retry_times
        for attempt in range(max_retries):
            try:
                import akshare as ak
                
                code = self._convert_stock_code(self.stock_code, 'akshare')
                today = datetime.date.today()
                start_date = (today - datetime.timedelta(days=days)).strftime('%Y%m%d')
                end_date = today.strftime('%Y%m%d')
                
                # 获取5分钟K线
                try:
                    df = ak.stock_zh_a_hist_min_em(
                        symbol=code,
                        period="5",
                        adjust="qfq",
                        start_date=start_date,
                        end_date=end_date
                    )
                    if df is not None and len(df) > 0:
                        # 转换列名
                        column_mapping = {
                            '时间': 'time',
                            '收盘': 'close',
                            '成交量': 'volume',
                            '日期': 'date'
                        }
                        for old_col, new_col in column_mapping.items():
                            if old_col in df.columns:
                                df = df.rename(columns={old_col: new_col})
                        
                        if 'time' in df.columns:
                            df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y%m%d%H%M%S')
                            df['date'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d')
                        elif 'date' in df.columns:
                            df['time'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d%H%M%S')
                            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                        
                        return df[['date', 'time', 'close', 'volume']]
                except Exception as e1:
                    # 如果5分钟数据失败，尝试日线数据
                    if attempt < max_retries - 1:
                        # 重试前切换代理/UA
                        if self.enable_anti_crawler and self.anti_crawler_pool:
                            setup_akshare_environment(self.anti_crawler_pool)
                            self.anti_crawler_pool.random_delay(1.0, 2.0)
                        continue
                    
                    try:
                        df = ak.stock_zh_a_hist(
                            symbol=code,
                            period="daily",
                            start_date=start_date,
                            end_date=end_date,
                            adjust="qfq"
                        )
                        if df is not None and len(df) > 0:
                            df = df.rename(columns={'日期': 'date', '收盘': 'close', '成交量': 'volume'})
                            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                            df['time'] = df['date'] + '15000000'
                            return df[['date', 'time', 'close', 'volume']]
                    except Exception as e2:
                        if attempt < max_retries - 1:
                            # 重试前切换代理/UA
                            if self.enable_anti_crawler and self.anti_crawler_pool:
                                setup_akshare_environment(self.anti_crawler_pool)
                                self.anti_crawler_pool.random_delay(1.0, 2.0)
                            continue
                        raise e2
                
                return None
            except (ConnectionError, TimeoutError, OSError) as e:
                # 网络连接错误，尝试重试
                if attempt < max_retries - 1:
                    # 切换代理/UA后重试
                    if self.enable_anti_crawler and self.anti_crawler_pool:
                        setup_akshare_environment(self.anti_crawler_pool)
                        self.anti_crawler_pool.random_delay(2.0, 4.0)
                    continue
                # 最后一次重试失败，静默处理
                return None
            except Exception as e:
                # 其他错误，不重试
                if attempt < max_retries - 1:
                    if self.enable_anti_crawler and self.anti_crawler_pool:
                        setup_akshare_environment(self.anti_crawler_pool)
                        self.anti_crawler_pool.random_delay(1.0, 2.0)
                    continue
                # 静默处理网络连接错误，避免过多警告
                return None
        
        return None
    
    def _fetch_from_baostock(self, days: int = 7) -> Optional[pd.DataFrame]:
        """从 baostock 获取数据"""
        try:
            import baostock as bs
            
            code = self._convert_stock_code(self.stock_code, 'baostock')
            today = datetime.date.today()
            start_date = (today - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
            end_date = today.strftime('%Y-%m-%d')
            
            bs.login()
            try:
                rs = bs.query_history_k_data_plus(
                    code,
                    "date,time,close,volume",
                    start_date=start_date,
                    end_date=end_date,
                    frequency='5',
                    adjustflag='3'
                )
                
                if rs.error_code != '0':
                    return None
                
                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())
                
                if not data_list:
                    return None
                
                df = pd.DataFrame(data_list, columns=rs.fields)
                
                # 添加日期列
                # 添加日期列
                if 'time' in df.columns:
                    try:
                        # 尝试解析时间格式
                        if df['time'].dtype == 'object':
                            # 如果是字符串格式，尝试多种格式
                            # 先尝试指定格式，避免警告
                            try:
                                time_parsed = pd.to_datetime(df['time'], format='%Y%m%d%H%M%S', errors='coerce')
                                if time_parsed.isna().all():
                                    # 如果解析失败，尝试其他常见格式
                                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%Y-%m-%d']:
                                        time_parsed = pd.to_datetime(df['time'], format=fmt, errors='coerce')
                                        if not time_parsed.isna().all():
                                            break
                                    # 如果所有格式都失败，使用dateutil（会产生警告，但至少能解析）
                                    if time_parsed.isna().all():
                                        with warnings.catch_warnings():
                                            warnings.simplefilter('ignore', UserWarning)
                                            time_parsed = pd.to_datetime(df['time'], errors='coerce')
                            except Exception:
                                # 如果出错，使用dateutil（会产生警告）
                                with warnings.catch_warnings():
                                    warnings.simplefilter('ignore', UserWarning)
                                    time_parsed = pd.to_datetime(df['time'], errors='coerce')
                            df['date'] = time_parsed.dt.strftime('%Y-%m-%d')
                        else:
                            df['date'] = pd.to_datetime(df['time'], errors='coerce').dt.strftime('%Y-%m-%d')
                    except:
                        # 如果解析失败，使用当前日期
                        df['date'] = datetime.date.today().strftime('%Y-%m-%d')
                
                return df
            finally:
                bs.logout()
                
        except Exception as e:
            warnings.warn(f"baostock 获取数据失败: {e}")
            return None
    
    def _fetch_from_stockapi(self, days: int = 7) -> Optional[pd.DataFrame]:
        """V14: 从 StockAPI 获取数据"""
        try:
            import requests
            
            # 获取StockAPI配置
            api_key = os.getenv('STOCKAPI_API_KEY', '')
            base_url = os.getenv('STOCKAPI_BASE_URL', 'https://api.stockapi.com')
            
            if not api_key:
                return None
            
            # 转换股票代码
            stock_code = self._convert_stock_code(self.stock_code, 'stockapi')
            
            # 计算日期范围
            today = datetime.date.today()
            start_date = (today - datetime.timedelta(days=days)).strftime('%Y%m%d')
            end_date = today.strftime('%Y%m%d')
            
            # 构建请求URL（根据StockAPI的实际API格式调整）
            # 这里使用通用的REST API格式
            url = f"{base_url}/api/v1/stock/kline"
            params = {
                'code': stock_code,
                'start_date': start_date,
                'end_date': end_date,
                'period': '5min',  # 5分钟K线
                'api_key': api_key
            }
            
            # 发送请求
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                
                # 解析响应数据（根据StockAPI的实际响应格式调整）
                if 'data' in data and len(data['data']) > 0:
                    # 假设返回格式为列表，每个元素包含时间、收盘价、成交量等
                    records = data['data']
                    
                    # 转换为DataFrame
                    df_data = []
                    for record in records:
                        # 根据实际API响应格式解析字段
                        # 这里使用通用字段名，需要根据实际API调整
                        time_str = record.get('time') or record.get('datetime') or record.get('date')
                        close = float(record.get('close') or record.get('price') or 0)
                        volume = float(record.get('volume') or record.get('vol') or 0)
                        
                        if time_str and close > 0:
                            df_data.append({
                                'time': time_str,
                                'close': close,
                                'volume': volume
                            })
                    
                    if len(df_data) > 0:
                        df = pd.DataFrame(df_data)
                        
                        # 标准化时间格式
                        if 'time' in df.columns:
                            try:
                                df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y%m%d%H%M%S')
                                df['date'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d')
                            except:
                                df['date'] = datetime.date.today().strftime('%Y-%m-%d')
                                df['time'] = df['date'].str.replace('-', '') + '15000000'
                        
                        # 确保有必要的列
                        if 'close' not in df.columns:
                            return None
                        if 'volume' not in df.columns:
                            df['volume'] = 0
                        
                        # 按时间排序
                        df = df.sort_values('time').reset_index(drop=True)
                        
                        return df[['date', 'time', 'close', 'volume']]
            
            return None
            
        except Exception as e:
            warnings.warn(f"StockAPI 获取数据失败: {e}")
            return None
    
    def fetch_data(self, days: int = 7, use_cache: bool = True) -> Tuple[Optional[pd.DataFrame], str]:
        """
        从多个数据源获取数据（按优先级尝试）
        
        参数:
            days: 获取最近N天的数据
            use_cache: 是否使用缓存（未来实现）
        
        返回:
            (数据DataFrame, 数据源名称) 的元组
        """
        # 按优先级尝试各个数据源
        for source in self.priority:
            if source not in self.sources:
                continue
            
            try:
                df = None
                if source == 'stockapi':
                    df = self._fetch_from_stockapi(days)
                elif source == 'tushare':
                    df = self._fetch_from_tushare(days)
                elif source == 'akshare':
                    df = self._fetch_from_akshare(days)
                elif source == 'baostock':
                    df = self._fetch_from_baostock(days)
                
                if df is not None and len(df) > 0:
                    # 更新统计
                    self.source_stats[source]['success'] += 1
                    self.source_stats[source]['last_success'] = datetime.datetime.now()
                    self.current_source = source
                    return df, source
                else:
                    self.source_stats[source]['fail'] += 1
            except Exception as e:
                self.source_stats[source]['fail'] += 1
                warnings.warn(f"数据源 {source} 获取失败: {e}")
        
        # 所有数据源都失败
        return None, "none"
    
    def merge_data_from_multiple_sources(self, 
                                         days: int = 7,
                                         merge_strategy: str = 'priority') -> Optional[pd.DataFrame]:
        """
        从多个数据源合并数据
        
        参数:
            days: 获取最近N天的数据
            merge_strategy: 合并策略 ('priority': 按优先级, 'union': 取并集, 'intersect': 取交集)
        
        返回:
            合并后的 DataFrame
        """
        all_data = {}
        
        # 从所有可用数据源获取数据
        for source in self.sources:
            try:
                df = None
                if source == 'stockapi':
                    df = self._fetch_from_stockapi(days)
                elif source == 'tushare':
                    df = self._fetch_from_tushare(days)
                elif source == 'akshare':
                    df = self._fetch_from_akshare(days)
                elif source == 'baostock':
                    df = self._fetch_from_baostock(days)
                
                if df is not None and len(df) > 0:
                    all_data[source] = df
            except:
                pass
        
        if not all_data:
            return None
        
        # 根据策略合并
        if merge_strategy == 'priority':
            # 按优先级选择第一个有效数据源
            for source in self.priority:
                if source in all_data:
                    return all_data[source]
        
        elif merge_strategy == 'union':
            # 取并集（合并所有数据，去重）
            merged = pd.concat(all_data.values(), ignore_index=True)
            merged = merged.drop_duplicates(subset=['time'], keep='last')
            merged = merged.sort_values('time')
            return merged
        
        elif merge_strategy == 'intersect':
            # 取交集（只保留所有数据源都有的时间点）
            # 这里简化处理，返回优先级最高的数据源
            for source in self.priority:
                if source in all_data:
                    return all_data[source]
        
        return None
    
    def get_source_stats(self) -> Dict:
        """获取数据源统计信息"""
        return self.source_stats.copy()
    
    def get_current_source(self) -> Optional[str]:
        """获取当前使用的数据源"""
        return self.current_source

