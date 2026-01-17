"""
V20中国卫星(600118)预测脚本
使用V20牛市牛股训练模型对中国卫星进行预测

使用方法：
1. 首先运行：python get_bull_market_stocks_data.py （获取牛市牛股数据）
2. 然后运行：python train_v20_bull_market.py （训练V20模型，可选）
3. 最后运行：python batch_predict_v20_600118_china_satellite.py （预测中国卫星）
"""

# 修改STOCK_LIST，只包含中国卫星
# 注意：这个文件会在batch_predict_v20_ma_alignment.py之前执行，修改STOCK_LIST

# 只预测中国卫星，使用V20牛市牛股训练模型
STOCK_LIST = [
    {'code': 'sh.600118', 'name': '中国卫星', 'model': 'models_v20_bull_market/best/best_model.zip', 'rank': 1, 'sharpe': None, 'return': None, 'drawdown': None, 'strategy': '🟢 均衡型'},
]

# 如果V20模型不存在，使用备用模型
import os
if not os.path.exists('models_v20_bull_market/best/best_model.zip'):
    if os.path.exists('models_v20_bull_market/ppo_stock_v20_bull_market_final.zip'):
        STOCK_LIST[0]['model'] = 'models_v20_bull_market/ppo_stock_v20_bull_market_final.zip'
    elif os.path.exists('ppo_stock_v7_000547.zip'):
        # 如果V20模型不存在，使用原来的航天发展模型
        STOCK_LIST[0]['model'] = 'ppo_stock_v7_000547.zip'
        print("⚠️  V20模型不存在，使用备用模型：ppo_stock_v7_000547.zip")
    else:
        print("⚠️  未找到可用模型，将使用默认模型路径")

print("="*70)
print("V20中国卫星(600118)预测")
print("="*70)
print("使用V20牛市牛股训练模型进行预测")
print(f"模型路径: {STOCK_LIST[0]['model']}")
print("="*70)
print()

# 导入V20主预测模块
# 由于V20模块在导入时会立即执行主逻辑，我们需要在执行前替换STOCK_LIST
# 采用读取文件、替换STOCK_LIST、然后exec执行的方式

print("\n开始执行V20预测...\n")

# 读取V20主文件
v20_file_path = 'batch_predict_v20_ma_alignment.py'
with open(v20_file_path, 'r', encoding='utf-8') as f:
    v20_code = f.read()

# 查找并替换STOCK_LIST定义
import re

# 查找注释 "# 批量预测：股票列表" 后的STOCK_LIST定义
insert_pos = v20_code.find('# 批量预测：股票列表')
if insert_pos > 0:
    list_start = v20_code.find('STOCK_LIST = [', insert_pos)
    if list_start > 0:
        # 找到对应的结束位置（计算括号匹配）
        bracket_count = 0
        list_end = list_start
        in_string = False
        string_char = None
        escape_next = False
        
        for i in range(list_start + len('STOCK_LIST = ['), len(v20_code)):
            char = v20_code[i]
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if not in_string:
                if char in ['"', "'"]:
                    in_string = True
                    string_char = char
                elif char == '[':
                    bracket_count += 1
                elif char == ']':
                    if bracket_count == 0:
                        list_end = i + 1
                        break
                    bracket_count -= 1
            else:
                if char == string_char:
                    in_string = False
                    string_char = None
        
        if list_end > list_start:
            new_stock_list_str = 'STOCK_LIST = [\n'
            for stock in STOCK_LIST:
                new_stock_list_str += f"    {repr(stock)},\n"
            new_stock_list_str += ']'
            v20_code = v20_code[:list_start] + new_stock_list_str + v20_code[list_end:]
            print(f"✅ 已替换STOCK_LIST定义，包含 {len(STOCK_LIST)} 只股票")
        else:
            # 尝试正则表达式
            pattern = r'STOCK_LIST\s*=\s*\[.*?\]'
            match = re.search(pattern, v20_code[insert_pos:insert_pos+5000], re.DOTALL)
            if match:
                actual_start = insert_pos + match.start()
                actual_end = insert_pos + match.end()
                new_stock_list_str = 'STOCK_LIST = ' + repr(STOCK_LIST)
                v20_code = v20_code[:actual_start] + new_stock_list_str + v20_code[actual_end:]
                print(f"✅ 已替换STOCK_LIST定义（使用正则表达式），包含 {len(STOCK_LIST)} 只股票")
            else:
                print("❌ 无法找到STOCK_LIST定义，将使用原始定义")
    else:
        print("⚠️  无法找到STOCK_LIST = [的位置")
else:
    print("⚠️  无法找到注释 '# 批量预测：股票列表'")

# 执行修改后的代码
print("🚀 开始执行V20预测...\n")
exec(v20_code, {'__file__': v20_file_path, '__name__': '__main__'})

