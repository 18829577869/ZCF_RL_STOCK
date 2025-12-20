"""从V16预测文件中提取股票代码和对应模型"""
import re
import os
import glob

def extract_stock_info_from_v16_file(file_path):
    """从V16预测文件中提取股票代码和模型路径"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取股票代码
        stock_code_match = re.search(r"STOCK_CODE\s*=\s*['\"]([^'\"]+)['\"]", content)
        stock_code = stock_code_match.group(1) if stock_code_match else None
        
        # 提取模型路径
        model_path_match = re.search(r"MODEL_PATH\s*=\s*['\"]([^'\"]+\.zip)['\"]", content)
        model_path = model_path_match.group(1) if model_path_match else None
        
        # 提取股票名称（从文件名或文件内容）
        filename = os.path.basename(file_path)
        # 从文件名提取：real_time_predict_v16_002706.py -> 002706
        code_from_filename = re.search(r'v16_(\d+)\.py', filename)
        if code_from_filename:
            code_digits = code_from_filename.group(1)
            # 尝试从文件内容中提取股票名称
            name_match = re.search(r"专用.*?（([^）]+)）", content)
            if name_match:
                stock_name = name_match.group(1)
            else:
                stock_name = f"股票{code_digits}"
        else:
            stock_name = None
        
        return {
            'stock_code': stock_code,
            'model_path': model_path,
            'stock_name': stock_name,
            'file': file_path
        }
    except Exception as e:
        print(f"解析文件 {file_path} 失败: {e}")
        return None

# 查找所有V16预测文件
v16_files = glob.glob('real_time_predict_v16_*.py')

stocks_info = []
for file_path in v16_files:
    info = extract_stock_info_from_v16_file(file_path)
    if info and info['stock_code']:
        stocks_info.append(info)

# 输出结果
print("="*70)
print("V16预测文件中的股票和模型信息")
print("="*70)
for info in stocks_info:
    print(f"{info['stock_name']} ({info['stock_code']}): {info['model_path']}")

print(f"\n共找到 {len(stocks_info)} 只股票")

