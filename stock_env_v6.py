# stock_env_v6.py - V6组合管理版
# -*- coding: utf-8 -*-
"""
V6 核心改进：
1. 差异化风险策略（根据标的类别调整）
2. 组合管理（支持多标的）
3. 行业轮动感知
4. 更智能的仓位管理
"""
import gymnasium as gym
import numpy as np
import pandas as pd
import os

class StockTradingEnv(gym.Env):
    def __init__(self, data_file, initial_balance=10000, commission_rate=0.00025,
                 min_commission=5, transfer_fee_rate=0.00001, stamp_duty_rate=0.0005,
                 min_trade_unit=100, slippage_rate=0.001, history_window=5):
        super().__init__()
        
        # 读取数据
        self.df = pd.read_csv(data_file)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
        # 识别标的类别（V6新增）
        self.stock_info = self._identify_stock_type(data_file)
        
        # 基础特征
        base_columns = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount',
                        'turn', 'pctChg', 'peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ']
        self.df[base_columns] = self.df[base_columns].apply(pd.to_numeric, errors='coerce')
        
        # 计算技术指标和风险指标
        self._add_technical_indicators()
        self._add_risk_indicators()
        
        self.obs_columns = base_columns + [
            'MA5', 'MA20', 'RSI', 'MACD', 'Volume_Ratio',
            'Volatility', 'Volume_Anomaly', 'Consecutive_Down', 
            'Amplitude', 'Gap', 'ATR'
        ]
        
        self.df = self.df.dropna().reset_index(drop=True)
        
        if len(self.df) < history_window + 50:
            raise ValueError(f"数据不足")
        
        # 交易参数
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.transfer_fee_rate = transfer_fee_rate
        self.stamp_duty_rate = stamp_duty_rate
        self.min_trade_unit = min_trade_unit
        self.slippage_rate = slippage_rate
        self.history_window = history_window
        
        # 根据标的类别设置风险参数（V6差异化策略）
        self._set_risk_params()
        
        # 状态变量
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.peak_net_worth = initial_balance
        self.prev_net_worth = initial_balance
        
        # 统计变量
        self.trade_history = []
        self.net_worth_history = []
        self.daily_returns = []
        self.risk_events = []
        
        # 归一化参数
        self.obs_min = self.df[self.obs_columns].min()
        self.obs_max = self.df[self.obs_columns].max()
        
        # 动作空间
        self.action_space = gym.spaces.Discrete(7)
        
        # 观测空间：历史窗口 + 持仓信息 + 风险等级 + 标的类别
        obs_shape = (self.history_window * len(self.obs_columns) + 6,)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, 
            shape=obs_shape, 
            dtype=np.float32
        )
    
    def _identify_stock_type(self, data_file):
        """识别标的类型（V6新增）"""
        filename = os.path.basename(data_file)
        
        # 尝试从多个可能的元数据文件加载
        possible_metadata_files = [
            'stockdata_v11_custom/metadata_v11_custom.csv',
            'stockdata_v7_301017/metadata_v7_301017.csv',
            'stockdata/metadata_v6.csv'
        ]
        
        for metadata_file in possible_metadata_files:
            if os.path.exists(metadata_file):
                try:
                    metadata = pd.read_csv(metadata_file)
                    for _, row in metadata.iterrows():
                        if row['code'] in filename or row['name'] in filename:
                            return {
                                'code': row['code'],
                                'name': row['name'],
                                'category': row['category'],
                                'volatility': row['volatility'],
                                'style': row['style']
                            }
                except Exception as e:
                    continue
        
        # 默认配置
        return {
            'code': 'unknown',
            'name': filename,
            'category': '未知',
            'volatility': '中',
            'style': '平衡'
        }
    
    def _set_risk_params(self):
        """根据标的类别设置差异化风险参数（V6核心）"""
        volatility = self.stock_info['volatility']
        category = self.stock_info['category']
        
        # 差异化风险阈值
        if volatility == '高':
            self.risk_threshold = 4  # 高波动标的，阈值高（更容忍风险）
            self.max_position = 0.8  # 最大80%仓位
            self.position_bonus_high_risk = -0.15  # 高风险时持仓惩罚小
        elif volatility == '低':
            self.risk_threshold = 2  # 低波动标的，阈值低（更谨慎）
            self.max_position = 1.0  # 可以满仓
            self.position_bonus_high_risk = -0.3  # 高风险时持仓惩罚大
        else:  # 中
            self.risk_threshold = 3
            self.max_position = 0.9
            self.position_bonus_high_risk = -0.2
        
        # 差异化回撤容忍度
        if category in ['科技', '新能源', '医药']:
            self.drawdown_tolerance = 0.25  # 容忍25%回撤
        elif category in ['金融', '消费']:
            self.drawdown_tolerance = 0.15  # 容忍15%回撤
        else:
            self.drawdown_tolerance = 0.20
        
        print(f"[V6配置] {self.stock_info['name']} | "
              f"类别:{category} | 波动:{volatility} | "
              f"风险阈值:{self.risk_threshold} | 最大仓位:{self.max_position*100:.0f}%")
    
    def _add_technical_indicators(self):
        """计算技术指标"""
        close = self.df['close']
        high = self.df['high']
        low = self.df['low']
        volume = self.df['volume']
        
        self.df['MA5'] = close.rolling(window=5).mean()
        self.df['MA20'] = close.rolling(window=20).mean()
        
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        self.df['MACD'] = ema12 - ema26
        
        ma_volume = volume.rolling(window=20).mean()
        self.df['Volume_Ratio'] = volume / (ma_volume + 1e-8)
    
    def _add_risk_indicators(self):
        """计算风险指标"""
        close = self.df['close']
        high = self.df['high']
        low = self.df['low']
        open_price = self.df['open']
        preclose = self.df['preclose']
        volume = self.df['volume']
        
        returns = close.pct_change()
        self.df['Volatility'] = returns.rolling(20).std() * 100
        
        volume_ma = volume.rolling(20).mean()
        self.df['Volume_Anomaly'] = volume / (volume_ma + 1e-8)
        
        price_change = close.diff()
        self.df['Consecutive_Down'] = (price_change < 0).astype(int).rolling(5, min_periods=1).sum()
        
        self.df['Amplitude'] = (high - low) / (close + 1e-8) * 100
        self.df['Gap'] = ((open_price - preclose) / (preclose + 1e-8) * 100).fillna(0)
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.df['ATR'] = tr.rolling(14).mean() / (close + 1e-8) * 100
    
    def _assess_risk_level(self):
        """评估风险等级（V6改进：考虑标的特性）"""
        current = self.df.iloc[self.current_step]
        risk_score = 0
        warnings = []
        
        # 根据标的波动性调整风险评估
        volatility_multiplier = {
            '高': 1.2,  # 高波动标的，风险分数乘1.2
            '中': 1.0,
            '低': 0.8   # 低波动标的，风险分数乘0.8
        }[self.stock_info['volatility']]
        
        if current['Volatility'] > self.df['Volatility'].quantile(0.9):
            risk_score += 1 * volatility_multiplier
            warnings.append('HIGH_VOLATILITY')
        
        if current['Volume_Anomaly'] > 2.0:
            risk_score += 1
            warnings.append('HUGE_VOLUME')
        
        if current['Consecutive_Down'] >= 3:
            risk_score += 1
            warnings.append('CONSECUTIVE_DROP')
        
        if current['RSI'] > 80:
            risk_score += 1
            warnings.append('OVERBOUGHT')
        elif current['RSI'] < 20:
            warnings.append('OVERSOLD')
        
        if abs(current['Gap']) > 3:
            risk_score += 1
            warnings.append('BIG_GAP')
        
        if current['ATR'] > self.df['ATR'].quantile(0.85):
            risk_score += 1
            warnings.append('HIGH_ATR')
        
        if len(warnings) > 0:
            self.risk_events.append({
                'step': self.current_step,
                'risk_level': risk_score,
                'warnings': warnings
            })
        
        return risk_score, warnings
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = self.history_window
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.peak_net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        
        self.trade_history = []
        self.net_worth_history = [self.initial_balance]
        self.daily_returns = []
        self.risk_events = []
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """获取观测（V6增加标的类别信息）"""
        # 历史窗口
        historical_data = []
        for i in range(self.history_window):
            step = self.current_step - self.history_window + i + 1
            row = self.df.iloc[step][self.obs_columns]
            normalized = 2 * (row - self.obs_min) / (self.obs_max - self.obs_min + 1e-8) - 1
            historical_data.extend(normalized.values)
        
        # 当前状态
        current_price = float(self.df.iloc[self.current_step]['close'])
        position_value = self.shares_held * current_price
        
        position_ratio = position_value / self.net_worth if self.net_worth > 0 else 0
        cash_ratio = self.balance / self.net_worth if self.net_worth > 0 else 1
        profit_ratio = (self.net_worth - self.initial_balance) / self.initial_balance
        drawdown = (self.peak_net_worth - self.net_worth) / self.peak_net_worth if self.peak_net_worth > 0 else 0
        
        risk_level, _ = self._assess_risk_level()
        risk_normalized = risk_level / 6.0
        
        # V6新增：标的类别编码
        volatility_code = {'低': 0.0, '中': 0.5, '高': 1.0}[self.stock_info['volatility']]
        
        position_info = [
            np.clip(position_ratio, 0, 1),
            np.clip(cash_ratio, 0, 1),
            np.clip(profit_ratio, -1, 1),
            np.clip(drawdown, 0, 1),
            np.clip(risk_normalized, 0, 1),
            volatility_code  # 新增：标的波动性编码
        ]
        
        obs = np.array(historical_data + position_info, dtype=np.float32)
        return obs
    
    def step(self, action):
        """执行动作"""
        current_price = float(self.df.iloc[self.current_step]['close'])
        risk_level, warnings = self._assess_risk_level()
        
        # V6改进：使用差异化风险阈值
        if risk_level >= self.risk_threshold and action in [1, 2, 3]:
            action = 0
        
        total_fee = 0
        if action == 1:
            total_fee = self._execute_buy(current_price, 0.25)
        elif action == 2:
            total_fee = self._execute_buy(current_price, 0.50)
        elif action == 3:
            # V6改进：考虑最大仓位限制
            buy_amount = min(1.0, self.max_position - (self.shares_held * current_price / self.balance))
            total_fee = self._execute_buy(current_price, max(0, buy_amount))
        elif action == 4:
            total_fee = self._execute_sell(current_price, 0.25)
        elif action == 5:
            total_fee = self._execute_sell(current_price, 0.50)
        elif action == 6:
            total_fee = self._execute_sell(current_price, 1.0)
        
        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.shares_held * current_price
        self.net_worth_history.append(self.net_worth)
        
        if self.net_worth > self.peak_net_worth:
            self.peak_net_worth = self.net_worth
        
        reward = self._calculate_reward_v6(total_fee, action, risk_level, warnings)
        
        daily_return = (self.net_worth / self.prev_net_worth - 1) if self.prev_net_worth > 0 else 0
        self.daily_returns.append(daily_return)
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        truncated = False
        
        # V6改进：差异化止损
        if self.net_worth < self.initial_balance * (1 - self.drawdown_tolerance):
            done = True
            reward -= 50
        
        return self._get_obs(), float(reward), done, truncated, {}
    
    def _execute_buy(self, price, amount):
        """执行买入"""
        if self.balance < 100 or amount <= 0:
            return 0
        
        adjusted_price = price * (1 + self.slippage_rate)
        cost = self.balance * amount
        
        if cost < 100:
            return 0
        
        shares = int(cost / (adjusted_price * self.min_trade_unit)) * self.min_trade_unit
        
        if shares == 0:
            return 0
        
        actual_cost = shares * adjusted_price
        transfer_fee = actual_cost * self.transfer_fee_rate
        commission = max(self.min_commission, actual_cost * self.commission_rate)
        total_fee = transfer_fee + commission
        total_cost = actual_cost + total_fee
        
        if total_cost <= self.balance:
            self.balance -= total_cost
            self.shares_held += shares
            self.trade_history.append({
                'step': self.current_step,
                'action': 'BUY',
                'shares': shares,
                'price': adjusted_price,
                'fee': total_fee
            })
            return total_fee
        
        return 0
    
    def _execute_sell(self, price, amount):
        """执行卖出"""
        if self.shares_held < self.min_trade_unit:
            return 0
        
        adjusted_price = price * (1 - self.slippage_rate)
        shares = int(self.shares_held * amount / self.min_trade_unit) * self.min_trade_unit
        
        if shares == 0:
            return 0
        
        revenue = shares * adjusted_price
        transfer_fee = revenue * self.transfer_fee_rate
        commission = max(self.min_commission, revenue * self.commission_rate)
        stamp_duty = revenue * self.stamp_duty_rate
        total_fee = transfer_fee + commission + stamp_duty
        
        self.balance += revenue - total_fee
        self.shares_held -= shares
        self.trade_history.append({
            'step': self.current_step,
            'action': 'SELL',
            'shares': shares,
            'price': adjusted_price,
            'fee': total_fee
        })
        
        return total_fee
    
    def _calculate_reward_v6(self, transaction_fee, action, risk_level, warnings):
        """V6奖励函数：差异化策略"""
        # 1. 净值变化
        net_worth_change = self.net_worth - self.prev_net_worth
        return_reward = net_worth_change / 100
        
        # 2. 差异化回撤惩罚
        drawdown = (self.peak_net_worth - self.net_worth) / self.peak_net_worth if self.peak_net_worth > 0 else 0
        
        if drawdown > self.drawdown_tolerance:
            drawdown_penalty = -10.0  # 超过容忍度重罚
        elif drawdown > self.drawdown_tolerance * 0.7:
            drawdown_penalty = -3.0
        elif drawdown > self.drawdown_tolerance * 0.5:
            drawdown_penalty = -1.0
        else:
            drawdown_penalty = 0
        
        # 3. 差异化持仓奖励
        current_price = float(self.df.iloc[self.current_step]['close'])
        position_ratio = (self.shares_held * current_price) / self.net_worth if self.net_worth > 0 else 0
        
        if risk_level >= self.risk_threshold:
            # 高风险时
            if position_ratio > 0.5:
                position_bonus = self.position_bonus_high_risk
            elif position_ratio > 0:
                position_bonus = self.position_bonus_high_risk / 2
            else:
                position_bonus = 0.1
        else:
            # 低风险时
            if position_ratio > 0.5:
                position_bonus = 0.15 if self.stock_info['volatility'] == '高' else 0.1
            elif position_ratio > 0:
                position_bonus = 0.05
            else:
                position_bonus = -0.1
        
        # 4. 交易奖励
        trade_bonus = 0.05 if action != 0 else 0
        
        # 5. 风险应对奖励
        risk_response_bonus = 0
        if risk_level >= self.risk_threshold:
            if action in [4, 5, 6]:
                risk_response_bonus = 0.3
            elif action in [1, 2, 3]:
                risk_response_bonus = -0.3
        
        # 6. 盈利奖励
        if self.net_worth > self.initial_balance * 1.05:
            profit_bonus = 1.0
        elif self.net_worth > self.initial_balance:
            profit_bonus = 0.5
        else:
            profit_bonus = 0
        
        total_reward = (return_reward + drawdown_penalty + position_bonus + 
                       trade_bonus + risk_response_bonus + profit_bonus)
        
        return total_reward
    
    def render(self):
        """显示状态"""
        current_price = float(self.df.iloc[self.current_step]['close'])
        position_value = self.shares_held * current_price
        profit = self.net_worth - self.initial_balance
        profit_pct = (profit / self.initial_balance) * 100
        drawdown = (self.peak_net_worth - self.net_worth) / self.peak_net_worth * 100
        risk_level, warnings = self._assess_risk_level()
        
        date = self.df.iloc[self.current_step]['date'].strftime('%Y-%m-%d')
        
        risk_str = f"[风险{risk_level}]" if risk_level > 0 else ""
        category_str = f"[{self.stock_info['category']}]"
        
        print(f"{date} {category_str}{risk_str} | 净值:{self.net_worth:8.0f} | "
              f"收益:{profit:+7.0f}({profit_pct:+6.2f}%) | "
              f"持仓:{self.shares_held:6.0f}股 | 回撤:{drawdown:5.2f}%")
    
    def get_stats(self):
        """获取统计"""
        if len(self.daily_returns) < 2:
            return {}
        
        total_return = (self.net_worth - self.initial_balance) / self.initial_balance
        
        peak = self.initial_balance
        max_dd = 0
        for nw in self.net_worth_history:
            if nw > peak:
                peak = nw
            dd = (peak - nw) / peak
            if dd > max_dd:
                max_dd = dd
        
        daily_returns_array = np.array(self.daily_returns)
        if len(daily_returns_array) > 0 and daily_returns_array.std() > 0:
            sharpe_ratio = (daily_returns_array.mean() / daily_returns_array.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        num_trades = len(self.trade_history)
        win_days = sum(1 for r in daily_returns_array if r > 0)
        win_rate = win_days / len(daily_returns_array) if len(daily_returns_array) > 0 else 0
        
        stats = {
            'final_net_worth': self.net_worth,
            'total_return': total_return * 100,
            'max_drawdown': max_dd * 100,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': num_trades,
            'win_rate': win_rate * 100,
            'total_days': len(self.daily_returns),
            'risk_events': len(self.risk_events),
            'category': self.stock_info['category'],
            'volatility': self.stock_info['volatility']
        }
        
        return stats



