"""
从批量预测日志中生成简洁版报告

使用方法:
    python generate_simple_report.py
"""

import re
import os
from datetime import datetime

def parse_log_file(log_file_path):
    """解析日志文件，提取关键信息"""
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取所有股票的信息
    stocks = []
    
    # 匹配每个股票的预测部分
    # 格式：📊 第 X 轮预测 [股票名(代码)]
    stock_pattern = r'📊 第 \d+ 轮预测 \[([^(]+)\(([^)]+)\)\]'
    
    # 分割每个股票的部分
    stock_sections = re.split(r'📊 第 \d+ 轮预测', content)
    
    for section in stock_sections[1:]:  # 跳过第一部分（文件头）
        stock_info = {}
        
        # 提取股票名称和代码
        match = re.search(r'\[([^(]+)\(([^)]+)\)\]', section)
        if match:
            stock_info['name'] = match.group(1).strip()
            stock_info['code'] = match.group(2).strip()
        else:
            continue
        
        # 提取当前价格
        price_match = re.search(r'💰 \[.*?\] 当前价格: ([\d.]+)', section)
        if price_match:
            stock_info['current_price'] = float(price_match.group(1))
        else:
            continue
        
        # 提取V12预测价格（使用融合后的预测价格）
        predict_match = re.search(r'📊 预测价格: ([\d.]+)元', section)
        if predict_match:
            stock_info['v12_predict_price'] = float(predict_match.group(1))
        else:
            # 如果没有融合价格，使用Transformer预测价格
            transformer_match = re.search(r'🔮 V12 Transformer预测价格: ([\d.]+)', section)
            if transformer_match:
                stock_info['v12_predict_price'] = float(transformer_match.group(1))
            else:
                stock_info['v12_predict_price'] = None
        
        # 提取V7建议买入价格（100%仓位价格）
        v7_buy_match = re.search(r'🟢 100%仓位（满仓）: ([\d.]+)元', section)
        if v7_buy_match:
            stock_info['v7_buy_price'] = float(v7_buy_match.group(1))
        else:
            stock_info['v7_buy_price'] = None
        
        # 提取V7建议卖出价格（0%仓位价格）
        v7_sell_match = re.search(r'⚪ 0%仓位（空仓）:?\s+([\d.]+)元', section)
        if v7_sell_match:
            stock_info['v7_sell_price'] = float(v7_sell_match.group(1))
        else:
            stock_info['v7_sell_price'] = None
        
        # 提取V7当前价格对应的合理仓位
        v7_position_match = re.search(r'📊 当前价格对应的合理仓位: ([\d.]+)%', section)
        if v7_position_match:
            stock_info['v7_position'] = float(v7_position_match.group(1))
        else:
            stock_info['v7_position'] = None
        
        # 提取V12建议买入价格（100%仓位价格）
        v12_buy_match = re.search(r'💡 仓位价格建议.*?🟢 100%仓位: ([\d.]+)元', section, re.DOTALL)
        if v12_buy_match:
            stock_info['v12_buy_price'] = float(v12_buy_match.group(1))
        else:
            stock_info['v12_buy_price'] = None
        
        # 提取V12建议卖出价格（0%仓位价格）
        v12_sell_match = re.search(r'💡 仓位价格建议.*?⚪ 0%仓位:?\s+([\d.]+)元', section, re.DOTALL)
        if v12_sell_match:
            stock_info['v12_sell_price'] = float(v12_sell_match.group(1))
        else:
            stock_info['v12_sell_price'] = None
        
        # 提取V12融合决策建议仓位
        v12_decision_match = re.search(r'⭐ V12融合决策 \[.*?\]: (买入|卖出|持有)(?:\s+([\d.]+)%)?', section)
        if v12_decision_match:
            action = v12_decision_match.group(1)
            position_str = v12_decision_match.group(2)
            if action == '买入' and position_str:
                stock_info['v12_position'] = float(position_str)
            elif action == '卖出' and position_str:
                stock_info['v12_position'] = -float(position_str)  # 负数表示卖出
            else:  # 持有 或 没有百分比
                stock_info['v12_position'] = 0
        else:
            stock_info['v12_position'] = None
        
        stocks.append(stock_info)
    
    return stocks

def get_display_width(s):
    """计算字符串的显示宽度（中文字符占2个宽度，英文字符占1个宽度）"""
    width = 0
    for char in s:
        if ord(char) > 127:  # 中文字符
            width += 2
        else:
            width += 1
    return width

def pad_string(s, width, align='<'):
    """填充字符串到指定显示宽度，返回固定字符长度的字符串"""
    display_width = get_display_width(s)
    if display_width >= width:
        # 如果超出宽度，需要截断（对于中文字符需要特殊处理）
        if display_width > width:
            # 简单截断，实际应用中可能需要更智能的处理
            result = s
            while get_display_width(result) > width and len(result) > 0:
                result = result[:-1]
            s = result
            display_width = get_display_width(s)
    
    # 计算需要填充的字符数（不是显示宽度）
    # 为了确保对齐，我们需要考虑中文字符的实际显示宽度
    # 但为了简化，我们使用显示宽度来计算填充
    padding = width - display_width
    if align == '<':
        return s + ' ' * padding
    elif align == '>':
        return ' ' * padding + s
    else:  # '^'
        left_pad = padding // 2
        right_pad = padding - left_pad
        return ' ' * left_pad + s + ' ' * right_pad

def generate_simple_report(stocks, output_file):
    """生成简洁版报告"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # 定义列宽（确保能容纳表头和数据，考虑中文字符显示宽度）
        col_widths = {
            'idx': 6,
            'name': 16,
            'code': 14,
            'current_price': 12,
            'predict_price': 12,
            'v7_buy': 12,
            'v7_sell': 12,
            'v7_pos': 10,
            'v12_buy': 12,
            'v12_sell': 13,  # 增加宽度以容纳"100.50"这样的数字
            'v12_pos': 13  # 增加宽度以容纳"买入100%"这样的文本
        }
        
        total_width = sum(col_widths.values()) + len(col_widths) - 1
        
        # 写入表头
        f.write("=" * total_width + "\n")
        f.write("简洁版预测报告 - " + datetime.now().strftime("%Y-%m-%d") + "\n")
        f.write("=" * total_width + "\n\n")
        
        # 写入表头行，使用pad_string函数确保对齐
        header_idx = pad_string('序号', col_widths['idx'], '<')
        header_name = pad_string('股票名称', col_widths['name'], '<')
        header_code = pad_string('股票代码', col_widths['code'], '<')
        header_current = pad_string('当日价格', col_widths['current_price'], '>')
        header_predict = pad_string('预测价格', col_widths['predict_price'], '>')
        header_v7_buy = pad_string('V7买入价', col_widths['v7_buy'], '>')
        header_v7_sell = pad_string('V7卖出价', col_widths['v7_sell'], '>')
        header_v7_pos = pad_string('V7仓位', col_widths['v7_pos'], '>')
        header_v12_buy = pad_string('V12买入价', col_widths['v12_buy'], '>')
        header_v12_sell = pad_string('V12卖出价', col_widths['v12_sell'], '>')
        header_v12_pos = pad_string('V12仓位', col_widths['v12_pos'], '<')
        # 确保表头每列之间有空格分隔
        header = (f"{header_idx} {header_name} {header_code} {header_current} "
                 f"{header_predict} {header_v7_buy} {header_v7_sell} "
                 f"{header_v7_pos} {header_v12_buy} {header_v12_sell} "
                 f"{header_v12_pos}\n")
        f.write(header)
        f.write("-" * total_width + "\n")
        
        # 写入每个股票的信息
        for idx, stock in enumerate(stocks, 1):
            name = stock.get('name', '')
            code = stock.get('code', '')
            current_price = stock.get('current_price', 0)
            predict_price = stock.get('v12_predict_price', 0)
            v7_buy = stock.get('v7_buy_price', 0)
            v7_sell = stock.get('v7_sell_price', 0)
            v7_pos = stock.get('v7_position', 0)
            v12_buy = stock.get('v12_buy_price', 0)
            v12_sell = stock.get('v12_sell_price', 0)
            v12_pos = stock.get('v12_position', 0)
            
            # 格式化V12仓位（如果是负数显示为卖出）
            if v12_pos is not None:
                if v12_pos < 0:
                    v12_pos_str = f"卖出{abs(v12_pos):.0f}%"
                elif v12_pos == 0:
                    v12_pos_str = "持有"
                else:
                    v12_pos_str = f"买入{v12_pos:.0f}%"
            else:
                v12_pos_str = "N/A"
            
            # 格式化V7仓位
            v7_pos_str = f"{v7_pos:.0f}%" if v7_pos is not None else "N/A"
            
            # 格式化价格，如果为0或None则显示N/A
            current_price_str = f"{current_price:.2f}" if current_price else "N/A"
            predict_price_str = f"{predict_price:.2f}" if predict_price else "N/A"
            v7_buy_str = f"{v7_buy:.2f}" if v7_buy else "N/A"
            v7_sell_str = f"{v7_sell:.2f}" if v7_sell else "N/A"
            v12_buy_str = f"{v12_buy:.2f}" if v12_buy else "N/A"
            v12_sell_str = f"{v12_sell:.2f}" if v12_sell else "N/A"
            
            # 使用相同的格式化函数确保对齐
            idx_str = pad_string(str(idx), col_widths['idx'], '<')
            # 限制名称长度，避免超出列宽
            name_limited = name
            while get_display_width(name_limited) > col_widths['name'] and len(name_limited) > 0:
                name_limited = name_limited[:-1]
            name_str = pad_string(name_limited, col_widths['name'], '<')
            
            code_limited = code
            while get_display_width(code_limited) > col_widths['code'] and len(code_limited) > 0:
                code_limited = code_limited[:-1]
            code_str = pad_string(code_limited, col_widths['code'], '<')
            
            current_price_str_fmt = pad_string(current_price_str, col_widths['current_price'], '>')
            predict_price_str_fmt = pad_string(predict_price_str, col_widths['predict_price'], '>')
            v7_buy_str_fmt = pad_string(v7_buy_str, col_widths['v7_buy'], '>')
            v7_sell_str_fmt = pad_string(v7_sell_str, col_widths['v7_sell'], '>')
            v7_pos_str_fmt = pad_string(v7_pos_str, col_widths['v7_pos'], '>')
            v12_buy_str_fmt = pad_string(v12_buy_str, col_widths['v12_buy'], '>')
            v12_sell_str_fmt = pad_string(v12_sell_str, col_widths['v12_sell'], '>')
            
            v12_pos_limited = v12_pos_str
            while get_display_width(v12_pos_limited) > col_widths['v12_pos'] and len(v12_pos_limited) > 0:
                v12_pos_limited = v12_pos_limited[:-1]
            v12_pos_str_fmt = pad_string(v12_pos_limited, col_widths['v12_pos'], '<')
            
            line = f"{idx_str} {name_str} {code_str} {current_price_str_fmt} {predict_price_str_fmt} {v7_buy_str_fmt} {v7_sell_str_fmt} {v7_pos_str_fmt} {v12_buy_str_fmt} {v12_sell_str_fmt} {v12_pos_str_fmt}\n"
            f.write(line)
        
        f.write("\n" + "=" * total_width + "\n")
        f.write(f"共 {len(stocks)} 只股票\n")
    
    print(f"✅ 简洁版报告已生成: {output_file}")

def main():
    import sys
    
    # 支持命令行参数指定输入文件
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = "batch_predict_log_2025-12-20.txt"
    
    if not os.path.exists(log_file):
        print(f"❌ 日志文件不存在: {log_file}")
        return
    
    print(f"📖 正在解析日志文件: {log_file}")
    stocks = parse_log_file(log_file)
    
    if not stocks:
        print("❌ 未能解析出任何股票信息")
        return
    
    print(f"✅ 成功解析 {len(stocks)} 只股票")
    
    # 生成输出文件名
    base_name = os.path.splitext(log_file)[0]
    output_file = f"{base_name}_simple.txt"
    
    generate_simple_report(stocks, output_file)
    print(f"📊 报告已保存到: {output_file}")

if __name__ == "__main__":
    main()
