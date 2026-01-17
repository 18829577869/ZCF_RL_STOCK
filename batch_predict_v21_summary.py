"""
V21批量预测系统 - 汇总预测（V18、V19、V20三种方式）
整合 V7-V20 的所有功能，对指定股票使用V18、V19、V20三种方式进行预测并汇总展示

V21新增功能：
- 汇总预测：对每个股票分别使用V18、V19、V20三种方式进行预测
- V18方式：基础版本，包含日K线均线、RSRS指标等
- V19方式：V18 + 两融余额监测
- V20方式：V19 + 牛市牛股训练数据优先级
- 结果汇总：将三种方式的预测结果并列显示，便于对比

目标股票：
- sh.600118: 中国卫星
- sh.600879: 航天电子
- sh.603698: 航天工程

设计理念：多方式预测对比，全面分析，决策支持
"""

import os
import sys
import json
import datetime
import importlib.util
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

# 目标股票列表
TARGET_STOCKS = [
    {'code': 'sh.600118', 'name': '中国卫星'},
    {'code': 'sh.600879', 'name': '航天电子'},
    {'code': 'sh.603698', 'name': '航天工程'},
]

# V18/V19/V20预测方式
PREDICTION_METHODS = {
    'V18': {
        'name': 'V18方式',
        'description': '基础版本：日K线均线、RSRS指标、筛选条件检测',
        'script': 'batch_predict_v18_ma_alignment.py',
    },
    'V19': {
        'name': 'V19方式',
        'description': 'V18 + 两融余额监测、数据查找优先级优化',
        'script': 'batch_predict_v19_ma_alignment.py',
    },
    'V20': {
        'name': 'V20方式',
        'description': 'V19 + 牛市牛股训练数据优先级',
        'script': 'batch_predict_v20_ma_alignment.py',
    },
}

def run_prediction(method_key, stock_codes):
    """
    运行指定股票的预测（通过读取并临时修改V18/V19/V20脚本文件，然后执行）
    
    Args:
        method_key: 预测方式（'V18', 'V19', 'V20'）
        stock_codes: 股票代码列表，格式如 [{'code': 'sh.600118', 'name': '中国卫星'}, ...]
    
    Returns:
        dict: 预测结果，如果失败返回None
    """
    method_info = PREDICTION_METHODS[method_key]
    script_file = method_info['script']
    
    # 检查脚本文件是否存在
    if not os.path.exists(script_file):
        print(f"   ⚠️  脚本文件不存在: {script_file}")
        return None
    
    print(f"\n   🔄 开始执行 {method_info['name']} 预测...")
    print(f"      脚本: {script_file}")
    # 避免在f-string中嵌套f-string，先构建字符串
    stock_names_str = ', '.join([f"{s['name']}({s['code']})" for s in stock_codes])
    print(f"      目标股票: {stock_names_str}")
    print(f"      开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 读取原始脚本文件内容
        with open(script_file, 'r', encoding='utf-8') as f:
            script_content = f.read()
        
        # 查找STOCK_LIST的定义位置（通常在文件中间部分）
        # 我们需要找到类似 "STOCK_LIST = [" 的位置
        
        # 先读取文件，找到STOCK_LIST定义的位置
        lines = script_content.split('\n')
        
        # 查找STOCK_LIST排序的位置（在排序之前插入筛选代码，这样排序时就已经是筛选后的列表了）
        stock_list_sorted_line = None
        for i, line in enumerate(lines):
            if 'STOCK_LIST = sorted' in line or 'STOCK_LIST=sorted' in line:
                stock_list_sorted_line = i
                break
        
        # 查找保存结果函数和prediction_data定义位置，用于添加方法版本标识
        save_result_func_line = None
        prediction_data_line = None
        for i, line in enumerate(lines):
            if 'def save_batch_predict_result' in line:
                save_result_func_line = i
            elif 'prediction_data = {' in line and i > 8000:  # 在主循环中的prediction_data定义
                prediction_data_line = i
                break
        
        if stock_list_sorted_line is None:
            print(f"   ⚠️  无法找到STOCK_LIST排序位置，将在主循环前插入筛选代码")
        
        # 构建新的STOCK_LIST（从原始STOCK_LIST中筛选出目标股票）
        # 方法：先导入模块获取原始STOCK_LIST，然后筛选
        # 但由于导入会立即执行，我们需要一个更巧妙的方法
        
        # 最简单的方法：直接执行脚本，但通过环境变量或命令行参数传递股票列表
        # 但V18/V19/V20不支持这个功能，所以我们采用另一种方法：
        # 在脚本执行前，临时修改STOCK_LIST定义
        
        # 方案：读取脚本内容，找到STOCK_LIST定义，替换为只包含目标股票的版本
        # 但这需要解析Python代码，比较复杂
        
        # 更实用的方案：直接导入并执行，然后通过修改模块属性来限制STOCK_LIST
        # 但模块导入时会立即执行主循环，所以我们需要在导入前修改代码
        
        # 最简单可行的方案：通过exec执行修改后的代码
        # 我们在STOCK_LIST定义之后添加代码，临时替换STOCK_LIST
        
        # 优先在STOCK_LIST排序之前插入筛选代码（这样排序时就已经是筛选后的列表了）
        # 如果找不到排序位置，则在主循环前插入
        insert_position = None
        
        if stock_list_sorted_line is not None:
            # 在STOCK_LIST排序之前插入筛选代码
            insert_position = stock_list_sorted_line
            print(f"   📍 将在STOCK_LIST排序前（第{insert_position}行）插入筛选代码")
        else:
            # 查找主循环开始的位置作为备选
            try_start = None
            main_loop_start = None
            
            # 先找到try语句
            for i, line in enumerate(lines):
                if '批量预测' in line and '对每个股票执行一次预测' in line:
                    # 找到注释行，下一行应该是try:
                    if i + 1 < len(lines) and 'try:' in lines[i + 1]:
                        try_start = i + 1
                        break
            
            # 在try块内找到for循环
            if try_start is not None:
                for i in range(try_start + 1, min(try_start + 10, len(lines))):
                    line = lines[i]
                    if 'for stock' in line and 'STOCK_LIST' in line and '#' not in line.strip()[:3]:
                        main_loop_start = i
                        break
            
            if main_loop_start is not None:
                insert_position = main_loop_start
                print(f"   📍 将在主循环前（第{insert_position}行）插入筛选代码")
        
        if insert_position is not None:
            # 构建筛选代码（需要在try块内，所以需要缩进）
            target_codes_list = [s['code'] for s in stock_codes]
            # 构建目标股票字典的字符串表示（使用repr避免引号问题）
            target_stocks_items = []
            for s in stock_codes:
                # 使用repr确保字符串正确转义
                code_repr = repr(s['code'])
                name_repr = repr(s['name'])
                target_stocks_items.append(f"{{'code': {code_repr}, 'name': {name_repr}}}")
            target_stocks_str = '[' + ', '.join(target_stocks_items) + ']'
            
            # 获取插入位置的缩进
            # 如果是在排序行之前插入，应该没有缩进（顶层代码）
            # 如果是在循环内插入，需要与循环行相同的缩进
            insert_line = lines[insert_position] if insert_position < len(lines) else lines[-1]
            base_indent = len(insert_line) - len(insert_line.lstrip())
            
            # 如果插入位置是注释行，查看下一行
            if insert_line.strip().startswith('#'):
                # 查找下一行非注释非空行来确定缩进
                for j in range(insert_position + 1, min(insert_position + 5, len(lines))):
                    next_line = lines[j]
                    if next_line.strip() and not next_line.strip().startswith('#'):
                        base_indent = len(next_line) - len(next_line.lstrip())
                        break
                # 如果还是注释，使用0缩进（顶层代码）
                if base_indent == len(insert_line) - len(insert_line.lstrip()) and insert_line.strip().startswith('#'):
                    base_indent = 0
            
            # 构建筛选代码，使用适当的缩进
            # 注意：筛选代码必须在STOCK_LIST排序之前执行，所以应该没有缩进（顶层代码）
            filter_code_lines = [
                "# V21自动添加：筛选目标股票（仅预测指定股票）",
                f"_V21_TARGET_CODES = {target_codes_list}",
                f"_V21_TARGET_STOCKS = {target_stocks_str}",
                "_V21_FILTERED_LIST = []",
                "for _stock in STOCK_LIST:",
                "    if _stock.get('code') in _V21_TARGET_CODES:",
                "        # 确保name正确",
                "        for _target in _V21_TARGET_STOCKS:",
                "            if _target['code'] == _stock.get('code'):",
                "                _stock['name'] = _target['name']",
                "                break",
                "        _V21_FILTERED_LIST.append(_stock)",
                "STOCK_LIST = _V21_FILTERED_LIST",
                "# V21筛选完成，STOCK_LIST现在只包含目标股票",
                ""  # 空行分隔
            ]
            
            # 在指定位置插入筛选代码
            for i, code_line in enumerate(filter_code_lines):
                lines.insert(insert_position + i, code_line)
            
            # 修改save_batch_predict_result函数，添加版本标识，避免覆盖
            # 查找save_batch_predict_result函数定义
            save_func_start = None
            for i, line in enumerate(lines):
                if 'def save_batch_predict_result' in line:
                    save_func_start = i
                    break
            
            if save_func_start is not None:
                # 查找current_result字典的构建位置
                current_result_line = None
                for i in range(save_func_start, min(save_func_start + 50, len(lines))):
                    if "current_result = {" in lines[i]:
                        current_result_line = i
                        break
                
                if current_result_line is not None:
                    # 在current_result字典中添加method_version字段
                    # 找到**prediction_data所在行
                    prediction_data_line = None
                    for i in range(current_result_line, min(current_result_line + 10, len(lines))):
                        if '**prediction_data' in lines[i]:
                            prediction_data_line = i
                            break
                    
                    if prediction_data_line is not None:
                        # 在**prediction_data之前添加method_version字段
                        # 检查是否已经存在method_version字段（避免重复插入）
                        already_has_method_version = False
                        for i in range(current_result_line, prediction_data_line):
                            if 'method_version' in lines[i]:
                                already_has_method_version = True
                                break
                        
                        if not already_has_method_version:
                            method_version_line = f"            'method_version': '{method_key}',  # V21添加：标识预测方式\n"
                            lines.insert(prediction_data_line, method_version_line)
                            print(f"   ✅ 已在第{prediction_data_line}行之前插入method_version字段")
                        else:
                            print(f"   ⚠️  method_version字段已存在，跳过插入")
                    else:
                        print(f"   ⚠️  未找到**prediction_data行，无法插入method_version字段")
                else:
                    print(f"   ⚠️  未找到current_result字典定义，无法插入method_version字段")
                
                # 修改检查逻辑：同时检查stock_code、date和method_version
                # 查找检查已存在记录的逻辑
                found_check_line = None
                for i in range(save_func_start, min(save_func_start + 100, len(lines))):
                    if "if (result.get('stock_code') == stock_code and" in lines[i]:
                        found_check_line = i
                        break
                
                if found_check_line is not None:
                    # 修改检查条件，添加method_version检查
                    # 检查下一行是否有date检查
                    next_line_idx = found_check_line + 1
                    if next_line_idx < len(lines) and "result.get('date') == current_result['date']" in lines[next_line_idx]:
                        # 两行格式：在下一行添加method_version检查
                        old_check_next = lines[next_line_idx]
                        # 去掉末尾的冒号和可能的右括号，添加method_version检查
                        if old_check_next.strip().endswith('):'):
                            # 去掉末尾的 ):
                            base_check = old_check_next.strip()[:-2]
                            new_check_next = f"                {base_check} and result.get('method_version') == current_result.get('method_version')):"
                        elif old_check_next.strip().endswith(')'):
                            # 去掉末尾的 )
                            base_check = old_check_next.strip()[:-1]
                            new_check_next = f"                {base_check} and result.get('method_version') == current_result.get('method_version')):"
                        else:
                            # 没有右括号，直接添加
                            new_check_next = old_check_next.rstrip() + " and result.get('method_version') == current_result.get('method_version')):"
                        lines[next_line_idx] = new_check_next
                    else:
                        # 单行格式：直接替换
                        old_check = lines[found_check_line]
                        new_check = old_check.replace(
                            "result.get('date') == current_result['date']):",
                            "result.get('date') == current_result['date'] and result.get('method_version') == current_result.get('method_version')):"
                        )
                        lines[found_check_line] = new_check
            
            # 修改日志文件写入方式：改为追加模式，避免覆盖
            # 查找日志文件初始化位置并替换为追加模式
            # 注意：只修改打开模式，不插入额外代码，避免语法错误
            for i, line in enumerate(lines):
                if 'with open(log_file' in line and ("'w'" in line or '"w"' in line) and 'encoding=' in line:
                    # 替换为追加模式（保持原有格式，只替换模式字符）
                    # 注意：保持原有的行结束符（\n），不要添加额外的换行
                    original_line = line
                    if "'w'" in line:
                        new_line = original_line.replace("'w'", "'a'")
                    elif '"w"' in line:
                        new_line = original_line.replace('"w"', '"a"')
                    else:
                        new_line = original_line
                    # 只在行尾添加注释（如果行尾没有注释），保持原有换行符
                    if '#' not in new_line.rstrip():
                        # 保持原有的换行符
                        line_without_newline = new_line.rstrip('\n\r')
                        new_line = line_without_newline + "  # V21修改：追加模式" + ('\n' if original_line.endswith('\n') else '')
                    lines[i] = new_line
                    break
            
            script_content = '\n'.join(lines)
        
        # 执行修改后的脚本
        # 创建新的命名空间
        namespace = {
            '__name__': f'__predict_{method_key.lower()}__',
            '__file__': script_file,
        }
        
        # 在执行脚本前，手动写入分隔标记到日志文件
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        log_file = f"batch_predict_log_{today}.txt"
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write('\n\n' + '=' * 120 + '\n')
                f.write(f'📊 {method_info["name"]}批量预测开始 - {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
                f.write('=' * 120 + '\n\n')
        except Exception as e:
            print(f"   ⚠️  写入日志分隔标记失败: {e}")
        
        # 执行脚本（使用修改后的代码，包含method_version字段）
        print(f"   📝 正在编译并执行脚本...")
        try:
            # 先检查语法，如果出错显示更多信息
            compiled_code = compile(script_content, script_file, 'exec')
            print(f"   ✅ 脚本编译成功，开始执行...")
            exec(compiled_code, namespace)
            print(f"   ✅ 脚本执行完成")
        except SyntaxError as e:
            print(f"   ❌ 语法错误: {e}")
            print(f"      错误位置: 第{e.lineno}行")
            # 显示错误行附近的内容
            script_lines = script_content.split('\n')
            error_line_idx = e.lineno - 1
            print(f"      错误行内容: {repr(script_lines[error_line_idx] if error_line_idx < len(script_lines) else 'N/A')}")
            if error_line_idx > 0:
                print(f"      前一行内容: {repr(script_lines[error_line_idx - 1])}")
            if error_line_idx + 1 < len(script_lines):
                print(f"      后一行内容: {repr(script_lines[error_line_idx + 1])}")
            raise
        except Exception as e:
            print(f"   ❌ 执行错误: {e}")
            print(f"      错误类型: {type(e).__name__}")
            raise
        
        # 验证结果是否已保存
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        result_file = f"batch_predict_results_{today}.json"
        saved_results = []
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    saved_results = json.load(f)
                # 检查是否有该方法的预测结果
                method_results = [r for r in saved_results if r.get('method_version') == method_key]
                if method_results:
                    print(f"   ✅ 已找到 {len(method_results)} 条 {method_info['name']} 的预测结果")
                    for r in method_results:
                        stock_code = r.get('stock_code', '未知')
                        stock_name = r.get('stock_name', '未知')
                        print(f"      - {stock_name}({stock_code})")
                else:
                    print(f"   ⚠️  未找到 {method_info['name']} 的预测结果（可能未保存或method_version字段未正确添加）")
            except Exception as e:
                print(f"   ⚠️  读取结果文件失败: {e}")
        
        # 返回成功状态
        end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n   ✅ {method_info['name']} 预测执行完成")
        print(f"      完成时间: {end_time}")
        return {
            'method': method_key,
            'stock_codes': stock_codes,
            'status': 'success',
            'message': f'{method_info["name"]}预测已完成',
            'timestamp': end_time,
        }
        
    except Exception as e:
        print(f"   ⚠️  执行预测时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return {
            'method': method_key,
            'stock_codes': stock_codes,
            'status': 'error',
            'message': f'执行失败: {str(e)}',
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

def collect_prediction_results(stock_codes):
    """
    收集指定股票的三种方式预测结果
    
    Args:
        stock_codes: 股票代码列表，格式如 [{'code': 'sh.600118', 'name': '中国卫星'}, ...]
    
    Returns:
        dict: 包含三种方式预测结果的字典
    """
    results = {
        'stock_codes': stock_codes,
        'methods': {},
    }
    
    # 对每种方式执行一次预测（一次预测所有目标股票）
    for method_key in ['V18', 'V19', 'V20']:
        print(f"\n{'='*120}")
        print(f"🔄 正在执行 {PREDICTION_METHODS[method_key]['name']} 预测...")
        print(f"{'='*120}")
        result = run_prediction(method_key, stock_codes)
        if result:
            results['methods'][method_key] = result
            status = result.get('status', 'unknown')
            if status == 'success':
                print(f"   ✅ {PREDICTION_METHODS[method_key]['name']} 执行成功")
            elif status == 'error':
                print(f"   ❌ {PREDICTION_METHODS[method_key]['name']} 执行失败: {result.get('message', '未知错误')}")
            else:
                print(f"   ⚠️  {PREDICTION_METHODS[method_key]['name']} 执行状态: {status}")
        else:
            print(f"   ❌ {PREDICTION_METHODS[method_key]['name']} 执行失败，返回None")
            results['methods'][method_key] = {
                'method': method_key,
                'status': 'error',
                'message': '执行失败，返回None',
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
    
    return results

def load_prediction_results_from_file():
    """
    从JSON结果文件中读取所有预测结果
    
    Returns:
        list: 预测结果列表
    """
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    result_file = f"batch_predict_results_{today}.json"
    
    if not os.path.exists(result_file):
        return []
    
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        return results
    except Exception as e:
        print(f"   ⚠️  读取结果文件失败: {e}")
        return []

def format_prediction_result_text(result):
    """
    将单个预测结果格式化为文本
    
    Args:
        result: 单个预测结果字典
    
    Returns:
        str: 格式化的文本
    """
    stock_name = result.get('stock_name', '未知')
    stock_code = result.get('stock_code', '未知')
    method_version = result.get('method_version', '未知')
    current_price = result.get('current_price')
    predicted_price = result.get('predicted_price')
    predicted_change_pct = result.get('predicted_change_pct')
    ppo_operation = result.get('ppo_operation', '未知')
    confidence = result.get('confidence')
    suggested_buy_price = result.get('v17_suggested_buy_price') or result.get('suggested_buy_price')
    suggested_sell_price = result.get('v17_suggested_sell_price_next_day') or result.get('suggested_sell_price')
    
    text = f"   📊 {method_version}方式预测结果:\n"
    
    if current_price:
        text += f"      当前价格: {current_price:.2f}元\n"
    
    if predicted_price:
        text += f"      预测价格: {predicted_price:.2f}元\n"
    
    if predicted_change_pct is not None:
        direction = "📈" if predicted_change_pct > 0 else "📉" if predicted_change_pct < 0 else "➡️"
        abs_change = abs(predicted_change_pct)
        
        # V21异常预测检测：涨跌幅>10%时标记为异常
        if abs_change > 10.0:
            warning_icon = "🚨" if abs_change > 20.0 else "⚠️"
            text += f"      预测涨跌幅: {direction} {predicted_change_pct:+.2f}% {warning_icon}【异常预测：涨跌幅>{10.0 if abs_change <= 20.0 else 20.0}%】\n"
            if abs_change > 20.0:
                text += f"      ⚠️  严重异常：预测涨跌幅{abs_change:.2f}%超过20%，可能存在模型异常或数据问题，建议谨慎参考\n"
            else:
                text += f"      ⚠️  异常预测：预测涨跌幅{abs_change:.2f}%超过10%，建议结合其他指标综合判断\n"
        else:
            text += f"      预测涨跌幅: {direction} {predicted_change_pct:+.2f}%\n"
    
    # V12融合决策信息
    conflict_info = result.get('conflict_info')
    if conflict_info and isinstance(conflict_info, dict):
        if conflict_info.get('has_conflict', False):
            text += f"      ⚠️  V12信号冲突: {conflict_info.get('adjustment_reason', '检测到信号冲突')}\n"
    
    if ppo_operation:
        text += f"      PPO操作建议: {ppo_operation}\n"
    
    if confidence is not None:
        text += f"      置信度: {confidence:.2%}\n"
        # 如果置信度较低且预测异常，添加提示
        if confidence < 0.5 and predicted_change_pct is not None and abs(predicted_change_pct) > 10.0:
            text += f"      ⚠️  低置信度警告：置信度{confidence:.2%}较低，且预测涨跌幅异常，建议谨慎参考\n"
    
    if suggested_buy_price:
        text += f"      建议买入价: {suggested_buy_price:.2f}元\n"
    
    if suggested_sell_price:
        text += f"      建议卖出价: {suggested_sell_price:.2f}元\n"
    
    return text

def display_summary_results(result):
    """
    汇总显示所有股票的三种方式预测结果（文本格式）
    
    Args:
        result: 包含所有股票三种方式预测结果的字典
    """
    print("\n" + "=" * 120)
    print(" " * 35 + "📊 V21汇总预测结果（文本格式）")
    print("=" * 120)
    
    stock_codes = result.get('stock_codes', [])
    methods = result.get('methods', {})
    
    # 显示目标股票列表
    print(f"\n📈 目标股票（共{len(stock_codes)}只）:")
    for stock in stock_codes:
        print(f"  - {stock['name']}({stock['code']})")
    
    # 从JSON文件读取详细的预测结果
    all_results = load_prediction_results_from_file()
    
    if all_results:
        # 先筛选异常预测（涨跌幅>10%）
        abnormal_results = []
        for r in all_results:
            predicted_change_pct = r.get('predicted_change_pct')
            if predicted_change_pct is not None and abs(predicted_change_pct) > 10.0:
                abnormal_results.append(r)
        
        # 显示异常预测汇总
        if abnormal_results:
            print(f"\n{'='*120}")
            print("🚨 异常预测汇总（涨跌幅>10%）:")
            print(f"{'='*120}")
            print(f"{'股票名称':<20} | {'股票代码':<15} | {'预测方式':<10} | {'当前价格':<12} | {'预测价格':<12} | {'预测涨跌幅':<15} | {'PPO操作':<12}")
            print(f"{'-'*120}")
            for r in abnormal_results:
                stock_name = r.get('stock_name', '未知')
                stock_code = r.get('stock_code', '未知')
                method_version = r.get('method_version', '未知')
                current_price = r.get('current_price', 0)
                predicted_price = r.get('predicted_price', 0)
                predicted_change_pct = r.get('predicted_change_pct', 0)
                ppo_operation = r.get('ppo_operation', '未知')
                warning = "🚨严重异常" if abs(predicted_change_pct) > 20.0 else "⚠️异常"
                print(f"{stock_name:<20} | {stock_code:<15} | {method_version:<10} | {current_price:>12.2f} | {predicted_price:>12.2f} | {predicted_change_pct:>+14.2f}% {warning} | {ppo_operation:<12}")
            print(f"{'-'*120}")
            print("💡 提示：异常预测（涨跌幅>10%）可能存在模型异常或数据问题，建议结合其他指标综合判断")
        
        print(f"\n{'='*120}")
        print("详细预测结果对比:")
        print(f"{'='*120}")
        
        # 按股票分组显示
        for stock in stock_codes:
            stock_code = stock['code']
            stock_name = stock['name']
            
            # 筛选出该股票的所有预测结果
            stock_results = [r for r in all_results if r.get('stock_code') == stock_code]
            
            if stock_results:
                print(f"\n{'='*120}")
                print(f"📈 {stock_name}({stock_code})")
                print(f"{'='*120}")
                
                # 按方式分组显示
                for method_key in ['V18', 'V19', 'V20']:
                    method_result = [r for r in stock_results if r.get('method_version') == method_key]
                    if method_result:
                        result = method_result[0]  # 取最新的一个
                        print(format_prediction_result_text(result))
                    else:
                        method_info = PREDICTION_METHODS[method_key]
                        print(f"   ⚠️  {method_info['name']}: 暂无预测结果")
                
                print(f"{'-'*120}")
            else:
                print(f"\n⚠️  {stock_name}({stock_code}): 暂无预测结果")
    else:
        print(f"\n⚠️  未找到预测结果文件，请确保V18/V19/V20已成功执行并保存结果")
    
    # 显示三种方式的执行状态
    print(f"\n{'='*120}")
    print("预测方式执行状态:")
    print(f"{'='*120}")
    
    for method_key in ['V18', 'V19', 'V20']:
        method_info = PREDICTION_METHODS[method_key]
        print(f"\n   {method_info['name']}:")
        print(f"      {method_info['description']}")
        
        if method_key in methods:
            method_result = methods[method_key]
            print(f"      状态: {method_result.get('status', 'unknown')}")
            print(f"      消息: {method_result.get('message', '无')}")
            print(f"      时间: {method_result.get('timestamp', '无')}")
        else:
            print(f"      ⚠️  预测失败或未执行")
    
    print(f"\n{'-'*120}")
    print("💡 提示：详细预测结果已从JSON文件读取并以文本格式显示。")

def main():
    """主函数"""
    print("\n" + "=" * 120)
    print(" " * 30 + "🚀 V21批量预测系统 - 汇总预测（V18、V19、V20三种方式）")
    print("=" * 120)
    
    print(f"\n📊 目标股票数量: {len(TARGET_STOCKS)}")
    print("目标股票列表:")
    for i, stock in enumerate(TARGET_STOCKS, 1):
        print(f"  {i}. {stock['name']}({stock['code']})")
    
    print(f"\n📋 预测方式: {len(PREDICTION_METHODS)} 种")
    for method_key, method_info in PREDICTION_METHODS.items():
        print(f"  - {method_info['name']}: {method_info['description']}")
    
    print("\n" + "=" * 120)
    print("开始预测...")
    print("=" * 120)
    print("💡 说明：将依次执行V18、V19、V20三种方式的预测，每种方式都会预测所有目标股票。")
    print("=" * 120)
    
    # 收集所有股票的三种方式预测结果
    # 对每种方式执行一次，预测所有目标股票
    result = collect_prediction_results(TARGET_STOCKS)
    
    # 汇总显示结果
    display_summary_results(result)
    
    print("\n" + "=" * 120)
    print("✅ V21汇总预测完成")
    print("=" * 120)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断，退出...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
