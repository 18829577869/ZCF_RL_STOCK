"""验证回测结果的计算是否正确"""
import json
import os

# 读取最新的回测结果
result_file = 'backtest_drop3percent_results_20251218_161659.json'

if not os.path.exists(result_file):
    print(f"文件不存在: {result_file}")
    exit(1)

with open(result_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

for stock in data:
    print(f"\n{'='*70}")
    print(f"验证: {stock['stock_name']} ({stock['stock_code']})")
    print(f"{'='*70}")
    
    initial_balance = stock['initial_balance']
    final_value = stock['final_value']
    total_return = stock['total_return']
    buy_hold_return = stock['buy_hold_return']
    
    # 验证总收益率
    calculated_return = (final_value - initial_balance) / initial_balance * 100
    print(f"\n1. 总收益率验证:")
    print(f"   初始资金: {initial_balance:,.2f} 元")
    print(f"   最终资产: {final_value:,.2f} 元")
    print(f"   计算收益率: {calculated_return:.2f}%")
    print(f"   报告收益率: {total_return:.2f}%")
    print(f"   差异: {abs(calculated_return - total_return):.4f}%")
    
    # 验证做T+0收益
    trades = stock['trades']
    t0_trades = [t for t in trades if isinstance(t, dict) and t.get('action') == 'buy_back_t0']
    
    print(f"\n2. 做T+0收益验证:")
    print(f"   做T+0次数: {len(t0_trades)}")
    
    # 找到对应的卖出交易
    total_t0_profit_calculated = 0.0
    t0_profits = []
    
    for i, t0_trade in enumerate(t0_trades):
        # 找到对应的卖出交易（同一天的sell_all_drop3pct）
        sell_trade = None
        for trade in trades:
            if (trade.get('date') == t0_trade.get('date') and 
                trade.get('action') == 'sell_all_drop3pct'):
                sell_trade = trade
                break
        
        if sell_trade:
            sell_price = sell_trade['price']
            sell_shares = sell_trade['shares']
            buy_price = t0_trade['price']
            buy_shares = t0_trade['shares']
            
            # 计算T+0收益：应该是卖出价-买入价，乘以实际交易的股数
            # 实际交易股数应该是min(卖出股数, 买入股数)
            actual_shares = min(sell_shares, buy_shares)
            t0_profit = (sell_price - buy_price) * actual_shares
            
            t0_profits.append(t0_profit)
            total_t0_profit_calculated += t0_profit
            
            if i < 5:  # 只显示前5次
                print(f"\n   第{i+1}次T+0 ({t0_trade.get('date')}):")
                print(f"     卖出: {sell_shares}股 @ {sell_price:.2f}元 = {sell_shares * sell_price:,.2f}元")
                print(f"     买入: {buy_shares}股 @ {buy_price:.2f}元 = {buy_shares * buy_price:,.2f}元")
                print(f"     实际交易股数: {actual_shares}股")
                print(f"     每股收益: {sell_price - buy_price:.2f}元")
                print(f"     T+0收益: {t0_profit:,.2f}元")
                print(f"     报告收益: {t0_trade.get('t0_profit', 0):,.2f}元")
                print(f"     差异: {abs(t0_profit - t0_trade.get('t0_profit', 0)):,.2f}元")
    
    print(f"\n   累计T+0收益:")
    print(f"     计算总和: {total_t0_profit_calculated:,.2f} 元")
    print(f"     报告总和: {stock.get('total_t0_profit', 0):,.2f} 元")
    print(f"     差异: {abs(total_t0_profit_calculated - stock.get('total_t0_profit', 0)):,.2f} 元")
    
    # 验证最终资产
    print(f"\n3. 最终资产验证:")
    print(f"   最终持仓: {stock.get('final_shares_held', 0):,.0f}股")
    print(f"   最终资金: {stock.get('final_balance', 0):,.2f}元")
    
    # 需要知道最终价格来计算
    if len(trades) > 0:
        last_trade = trades[-1]
        last_date = last_trade.get('date')
        print(f"   最后交易日期: {last_date}")
    
    print(f"\n4. 关键指标:")
    print(f"   买入持有收益率: {buy_hold_return:.2f}%")
    print(f"   策略总收益率: {total_return:.2f}%")
    print(f"   策略优势: {total_return - buy_hold_return:.2f}个百分点")
    print(f"   最大回撤: {stock.get('max_drawdown', 0):.2f}%")

