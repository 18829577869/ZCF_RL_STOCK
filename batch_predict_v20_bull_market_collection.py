"""
V20集合预测 - 牛市模型预测航天板块股票
使用V20牛市牛股训练模型预测以下5只股票：
1. 航天动力 (sh.600343)
2. 航天电子 (sh.600879)
3. 中国卫星 (sh.600118)
4. 大业股份 (sh.603278)
5. 航天工程 (sh.603698)

使用方法：
python batch_predict_v20_bull_market_collection.py
"""

import sys
import os

# 导入V20预测模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# V20集合预测：只预测这5只航天板块股票，使用V20牛市牛股训练模型
STOCK_LIST = [
    {'code': 'sh.600343', 'name': '航天动力', 'model': 'models_v20_bull_market/best/best_model.zip', 'rank': 1, 'sharpe': 2.82, 'return': 190.73, 'drawdown': 19.70, 'strategy': '🟡 进取型'},
    {'code': 'sh.600879', 'name': '航天电子', 'model': 'models_v20_bull_market/best/best_model.zip', 'rank': 2, 'sharpe': 2.05, 'return': 87.15, 'drawdown': 16.85, 'strategy': '🟢 均衡型'},
    {'code': 'sh.600118', 'name': '中国卫星', 'model': 'models_v20_bull_market/best/best_model.zip', 'rank': 3, 'sharpe': 2.04, 'return': 77.94, 'drawdown': 16.34, 'strategy': '🟢 均衡型'},
    {'code': 'sh.603278', 'name': '大业股份', 'model': 'models_v20_bull_market/best/best_model.zip', 'rank': 4, 'sharpe': 1.47, 'return': 24.28, 'drawdown': 7.55, 'strategy': '🟢 均衡型'},
    {'code': 'sh.603698', 'name': '航天工程', 'model': 'models_v20_bull_market/best/best_model.zip', 'rank': 5, 'sharpe': 2.76, 'return': 51.53, 'drawdown': 6.69, 'strategy': '🔵 稳健型'},
]

# 检查V20模型是否存在，如果不存在则使用备用模型
v20_model_path = 'models_v20_bull_market/best/best_model.zip'
v20_final_model_path = 'models_v20_bull_market/ppo_stock_v20_bull_market_final.zip'

if not os.path.exists(v20_model_path):
    if os.path.exists(v20_final_model_path):
        print(f"⚠️  V20最佳模型不存在，使用最终模型: {v20_final_model_path}")
        for stock in STOCK_LIST:
            stock['model'] = v20_final_model_path
    else:
        print("⚠️  V20模型不存在，使用备用模型")
        # 使用原来的模型作为备用
        backup_models = {
            'sh.600343': 'ppo_stock_v11_custom.zip',  # 航天动力 - V11自定义模型
            'sh.600879': 'ppo_stock_v7_000547.zip',  # 航天电子 - 航天发展模型
            'sh.600118': 'ppo_stock_v7_000547.zip',  # 中国卫星 - 航天发展模型
            'sh.603278': 'ppo_stock_v7_300007.zip',  # 大业股份 - 汉威科技模型
            'sh.603698': 'ppo_stock_v7_603698.zip',  # 航天工程 - 自身模型
        }
        for stock in STOCK_LIST:
            if stock['code'] in backup_models:
                stock['model'] = backup_models[stock['code']]
        print("   使用备用模型配置完成")
else:
    print(f"✅ 使用V20牛市牛股训练模型: {v20_model_path}")

print("="*70)
print("V20集合预测 - 牛市模型预测航天板块股票")
print("="*70)
print("预测股票列表：")
for idx, stock in enumerate(STOCK_LIST, 1):
    print(f"  {idx}. {stock['name']}({stock['code']}) - 模型: {stock['model']}")
print("="*70)
print()

# 导入V20主预测模块
# 由于V20模块在导入时会立即执行主逻辑，我们需要在执行前替换STOCK_LIST
# 采用读取文件、替换STOCK_LIST、然后exec执行的方式

print("\n开始执行V20集合预测...\n")

# 读取V20主文件
v20_file_path = 'batch_predict_v20_ma_alignment.py'
with open(v20_file_path, 'r', encoding='utf-8') as f:
    v20_code = f.read()

# 查找并替换STOCK_LIST定义
# 查找 "STOCK_LIST = [" 的位置（在第1832行附近）
import re

# 方法1：查找注释 "# 批量预测：股票列表" 后的STOCK_LIST定义
insert_pos = v20_code.find('# 批量预测：股票列表')
if insert_pos > 0:
    # 找到下一个STOCK_LIST = [的位置
    list_start = v20_code.find('STOCK_LIST = [', insert_pos)
    if list_start > 0:
        # 找到对应的结束位置（计算括号匹配）
        bracket_count = 0
        list_end = list_start
        in_string = False
        string_char = None
        escape_next = False
        
        # 从STOCK_LIST = [之后开始查找
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
            # 构建新的STOCK_LIST定义字符串（格式化，便于阅读）
            new_stock_list_str = 'STOCK_LIST = [\n'
            for stock in STOCK_LIST:
                new_stock_list_str += f"    {repr(stock)},\n"
            new_stock_list_str += ']'
            
            # 替换
            v20_code = v20_code[:list_start] + new_stock_list_str + v20_code[list_end:]
            print(f"✅ 已替换STOCK_LIST定义，包含 {len(STOCK_LIST)} 只股票")
        else:
            print("⚠️  无法找到STOCK_LIST定义的结束位置，尝试其他方法...")
            # 尝试使用正则表达式
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
    print("⚠️  无法找到注释 '# 批量预测：股票列表'，尝试直接查找STOCK_LIST...")
    # 直接查找STOCK_LIST = [
    list_start = v20_code.find('STOCK_LIST = [')
    if list_start > 0:
        # 使用相同的方法查找结束位置
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
            new_stock_list_str = 'STOCK_LIST = ' + repr(STOCK_LIST)
            v20_code = v20_code[:list_start] + new_stock_list_str + v20_code[list_end:]
            print(f"✅ 已替换STOCK_LIST定义，包含 {len(STOCK_LIST)} 只股票")
        else:
            print("❌ 无法找到STOCK_LIST定义的结束位置")
    else:
        print("❌ 无法找到STOCK_LIST定义")

# 执行修改后的代码
print("🚀 开始执行V20集合预测...\n")
exec(v20_code, {'__file__': v20_file_path, '__name__': '__main__'})

