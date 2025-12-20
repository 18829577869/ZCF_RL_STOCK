#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量更新所有V16文件，添加预测准确率统计功能
"""

import os
import glob
import re

# 需要添加的功能代码（在配置参数之前）
FUNCTION_CODE = '''# ==================== V16预测准确率统计功能 ====================

def get_prediction_log_file(stock_code):
    """获取预测日志文件路径"""
    return f"v12_prediction_log_{stock_code.replace('.', '_')}.json"

def save_v12_prediction(date_str, transformer_prediction, current_price, stock_code):
    """
    保存V12 Transformer预测结果
    
    Args:
        date_str: 日期字符串（YYYY-MM-DD）
        transformer_prediction: Transformer预测价格
        current_price: 当前价格
        stock_code: 股票代码
    """
    try:
        prediction_log_file = get_prediction_log_file(stock_code)
        predictions = []
        if os.path.exists(prediction_log_file):
            with open(prediction_log_file, 'r', encoding='utf-8') as f:
                predictions = json.load(f)
        
        # 检查是否已存在该日期的预测，如果存在则更新
        found = False
        for pred in predictions:
            if pred.get('date') == date_str:
                pred['predicted_price'] = transformer_prediction
                pred['current_price'] = current_price
                pred['timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                found = True
                break
        
        if not found:
            predictions.append({
                'date': date_str,
                'predicted_price': transformer_prediction,
                'current_price': current_price,
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # 保存到文件
        with open(prediction_log_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"   ⚠️  保存V12预测失败: {e}")
        return False

def get_actual_close_price(stock_code, date_str):
    """
    获取指定日期的实际收盘价
    
    Args:
        stock_code: 股票代码
        date_str: 日期字符串（YYYY-MM-DD）
    
    Returns:
        float: 收盘价，如果获取失败返回None
    """
    try:
        import baostock as bs
        bs.login()
        
        # 转换股票代码格式
        if stock_code.startswith('sh.'):
            bs_code = f"sh.{stock_code.split('.')[1]}"
        elif stock_code.startswith('sz.'):
            bs_code = f"sz.{stock_code.split('.')[1]}"
        else:
            bs_code = stock_code
        
        # 查询指定日期的K线数据
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,close",
            start_date=date_str,
            end_date=date_str,
            frequency="d",
            adjustflag="3"
        )
        
        if rs.error_code == '0':
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())
            
            bs.logout()
            
            if len(data_list) > 0:
                return float(data_list[0][1])  # 返回收盘价
        
        bs.logout()
        return None
    except Exception as e:
        print(f"   ⚠️  获取实际收盘价失败: {e}")
        return None

def calculate_prediction_accuracy(stock_code):
    """
    计算V12预测准确率统计
    
    Args:
        stock_code: 股票代码
    
    Returns:
        dict: 统计结果
    """
    try:
        prediction_log_file = get_prediction_log_file(stock_code)
        if not os.path.exists(prediction_log_file):
            return None
        
        with open(prediction_log_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        if len(predictions) == 0:
            return None
        
        # 按日期排序
        predictions.sort(key=lambda x: x.get('date', ''))
        
        accuracy_stats = {
            'total_predictions': 0,
            'valid_comparisons': 0,
            'total_error': 0.0,
            'total_abs_error': 0.0,
            'total_error_pct': 0.0,
            'total_abs_error_pct': 0.0,
            'details': []
        }
        
        today = datetime.datetime.now().date()
        
        for i, pred in enumerate(predictions):
            pred_date_str = pred.get('date')
            if not pred_date_str:
                continue
            
            try:
                pred_date = datetime.datetime.strptime(pred_date_str, '%Y-%m-%d').date()
            except:
                continue
            
            # 只统计昨天的预测和今天的实际收盘价
            if pred_date >= today:
                continue  # 跳过今天及未来的预测
            
            predicted_price = pred.get('predicted_price')
            if predicted_price is None or predicted_price <= 0:
                continue
            
            accuracy_stats['total_predictions'] += 1
            
            # 获取预测日期后一天的实际收盘价
            next_date = pred_date + datetime.timedelta(days=1)
            
            # 跳过周末
            while next_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
                next_date += datetime.timedelta(days=1)
            
            # 如果下一天是今天或未来，跳过
            if next_date >= today:
                continue
            
            next_date_str = next_date.strftime('%Y-%m-%d')
            actual_close = get_actual_close_price(stock_code, next_date_str)
            
            if actual_close is None or actual_close <= 0:
                continue
            
            # 计算误差
            error = predicted_price - actual_close
            abs_error = abs(error)
            error_pct = (error / actual_close * 100) if actual_close > 0 else 0
            abs_error_pct = abs(error_pct)
            
            accuracy_stats['valid_comparisons'] += 1
            accuracy_stats['total_error'] += error
            accuracy_stats['total_abs_error'] += abs_error
            accuracy_stats['total_error_pct'] += error_pct
            accuracy_stats['total_abs_error_pct'] += abs_error_pct
            
            accuracy_stats['details'].append({
                'prediction_date': pred_date_str,
                'actual_date': next_date_str,
                'predicted_price': predicted_price,
                'actual_close': actual_close,
                'error': error,
                'abs_error': abs_error,
                'error_pct': error_pct,
                'abs_error_pct': abs_error_pct
            })
        
        # 计算平均值
        if accuracy_stats['valid_comparisons'] > 0:
            accuracy_stats['avg_error'] = accuracy_stats['total_error'] / accuracy_stats['valid_comparisons']
            accuracy_stats['avg_abs_error'] = accuracy_stats['total_abs_error'] / accuracy_stats['valid_comparisons']
            accuracy_stats['avg_error_pct'] = accuracy_stats['total_error_pct'] / accuracy_stats['valid_comparisons']
            accuracy_stats['avg_abs_error_pct'] = accuracy_stats['total_abs_error_pct'] / accuracy_stats['valid_comparisons']
        else:
            accuracy_stats['avg_error'] = 0.0
            accuracy_stats['avg_abs_error'] = 0.0
            accuracy_stats['avg_error_pct'] = 0.0
            accuracy_stats['avg_abs_error_pct'] = 0.0
        
        return accuracy_stats
    except Exception as e:
        print(f"   ⚠️  计算预测准确率失败: {e}")
        return None

def display_prediction_accuracy(stock_code):
    """显示V12预测准确率统计"""
    try:
        stats = calculate_prediction_accuracy(stock_code)
        if stats is None or stats['valid_comparisons'] == 0:
            print(f"\\n   📊 V12预测准确率统计: 暂无有效数据")
            return
        
        print(f"\\n   📊 V12预测准确率统计:")
        print(f"      ✅ 总预测次数: {stats['total_predictions']} 次")
        print(f"      ✅ 有效对比次数: {stats['valid_comparisons']} 次")
        print(f"      📈 平均误差: {stats['avg_error']:.2f} 元 ({stats['avg_error_pct']:+.2f}%)")
        print(f"      📊 平均绝对误差: {stats['avg_abs_error']:.2f} 元 ({stats['avg_abs_error_pct']:.2f}%)")
        
        # 显示最近5次预测的详细情况
        if len(stats['details']) > 0:
            print(f"\\n      📋 最近5次预测详情:")
            recent_details = stats['details'][-5:]
            for detail in recent_details:
                print(f"         {detail['prediction_date']} 预测 {detail['predicted_price']:.2f}元 → "
                      f"{detail['actual_date']} 实际 {detail['actual_close']:.2f}元 "
                      f"(误差: {detail['error']:+.2f}元, {detail['error_pct']:+.2f}%)")
    except Exception as e:
        print(f"   ⚠️  显示预测准确率失败: {e}")

'''

DISPLAY_CALL = '''        # V16新增：显示V12预测准确率统计
        display_prediction_accuracy(STOCK_CODE)
        
'''

SAVE_CALL = '''                            # V16新增：保存V12预测结果用于准确率统计
                            current_date_str = datetime.datetime.now().strftime('%Y-%m-%d')
                            save_v12_prediction(current_date_str, transformer_prediction, current_price, STOCK_CODE)
'''

def update_file(filepath):
    """更新单个文件"""
    print(f"处理: {os.path.basename(filepath)}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        modified = False
        
        # 1. 检查并添加功能代码
        if 'V16预测准确率统计功能' not in content:
            # 在"配置参数"之前插入
            pattern = r'(# ==================== 配置参数 ====================)'
            if re.search(pattern, content):
                content = re.sub(pattern, FUNCTION_CODE + r'\\1', content, count=1)
                modified = True
                print("  ✅ 已添加功能代码")
            else:
                print("  ⚠️  未找到配置参数位置")
        else:
            print("  ✓ 功能代码已存在")
        
        # 2. 检查并添加display调用
        if 'display_prediction_accuracy(STOCK_CODE)' not in content:
            # 在主循环开始处添加
            pattern = r'(print\(f"📊 第 \{iteration_count\} 轮预测.*?\)\s*print\(f"\{.*?\}.*?"\)\s*)'
            if re.search(pattern, content):
                content = re.sub(pattern, r'\\1' + DISPLAY_CALL, content, count=1)
                modified = True
                print("  ✅ 已添加display调用")
            else:
                print("  ⚠️  未找到主循环开始位置")
        else:
            print("  ✓ display调用已存在")
        
        # 3. 检查并添加save调用（两个位置）
        save_count = content.count('save_v12_prediction(current_date_str, transformer_prediction, current_price, STOCK_CODE)')
        if save_count < 2:
            # 位置1：有归一化范围输出的
            pattern1 = r'(print\(f"   🔮 V12 Transformer预测价格: \{transformer_prediction:.2f\}.*?\)\s*print\(f"      📊 归一化范围:.*?"\)\s*)'
            if re.search(pattern1, content) and 'save_v12_prediction' not in re.search(pattern1, content).group(0):
                content = re.sub(pattern1, r'\\1' + SAVE_CALL, content, count=1)
                modified = True
                print("  ✅ 已添加save调用（位置1）")
            
            # 位置2：else分支中的
            pattern2 = r'(print\(f"   🔮 V12 Transformer预测价格: \{transformer_prediction:.2f\}.*?\)\s*)(?=            except Exception)'
            matches = list(re.finditer(pattern2, content))
            if len(matches) >= 2:
                second_match = matches[1]
                pos = second_match.end()
                if 'save_v12_prediction' not in content[pos-100:pos+100]:
                    content = content[:pos] + SAVE_CALL + content[pos:]
                    modified = True
                    print("  ✅ 已添加save调用（位置2）")
        else:
            print("  ✓ save调用已存在")
        
        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print("  ✅ 文件已更新\\n")
            return True
        else:
            print("  ✓ 文件无需更新\\n")
            return False
            
    except Exception as e:
        print(f"  ❌ 错误: {e}\\n")
        return False

def main():
    v16_files = glob.glob('real_time_predict_v16_*.py')
    # 排除已处理的
    v16_files = [f for f in v16_files if f not in ['real_time_predict_v16_000625.py']]
    
    print(f"找到 {len(v16_files)} 个文件需要处理\\n")
    
    success = 0
    for f in sorted(v16_files):
        if update_file(f):
            success += 1
    
    print(f"\\n完成: {success}/{len(v16_files)} 个文件已更新")

if __name__ == '__main__':
    main()

