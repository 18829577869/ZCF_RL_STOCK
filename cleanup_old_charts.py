# cleanup_old_charts.py - 清理旧的图表文件
# -*- coding: utf-8 -*-
"""
清理 visualization_output 目录中的旧图表文件
只保留：
- price_chart_current.png（实时图表）
- dashboard.png（仪表板）
- dashboard.html（HTML仪表板）
- data.csv（数据文件）
- price_chart_hourly_*.png（每小时历史图表，保留最新的24个）
- 删除所有旧的 price_chart_YYYYMMDD_HHMMSS.png 文件
"""
import os
import glob

VISUALIZATION_OUTPUT_DIR = "visualization_output"

def cleanup_old_charts():
    """清理旧的图表文件"""
    if not os.path.exists(VISUALIZATION_OUTPUT_DIR):
        print(f"目录不存在: {VISUALIZATION_OUTPUT_DIR}")
        return
    
    # 查找所有旧的 price_chart_YYYYMMDD_HHMMSS.png 文件
    pattern = os.path.join(VISUALIZATION_OUTPUT_DIR, 'price_chart_*.png')
    all_chart_files = glob.glob(pattern)
    
    # 排除要保留的文件
    files_to_keep = [
        'price_chart_current.png',
        'dashboard.png'
    ]
    
    # 查找历史图表文件（price_chart_hourly_*.png）
    hourly_pattern = os.path.join(VISUALIZATION_OUTPUT_DIR, 'price_chart_hourly_*.png')
    hourly_files = glob.glob(hourly_pattern)
    
    # 保留最新的24个历史图表文件
    if len(hourly_files) > 24:
        hourly_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        hourly_files_to_keep = hourly_files[:24]
    else:
        hourly_files_to_keep = hourly_files
    
    # 计算要删除的文件
    files_to_delete = []
    for file_path in all_chart_files:
        filename = os.path.basename(file_path)
        
        # 跳过要保留的文件
        if filename in files_to_keep:
            continue
        
        # 跳过要保留的历史图表文件
        if file_path in hourly_files_to_keep:
            continue
        
        # 如果文件名格式是 price_chart_YYYYMMDD_HHMMSS.png（旧格式）
        if filename.startswith('price_chart_') and filename.endswith('.png'):
            # 检查是否是旧格式（包含秒数，即长度为30以上：price_chart_YYYYMMDD_HHMMSS.png）
            if len(filename) > 30:  # price_chart_ + 15位时间戳 + .png = 30+字符
                files_to_delete.append(file_path)
            elif filename.startswith('price_chart_hourly_'):
                # 如果不在保留列表中，也要删除
                if file_path not in hourly_files_to_keep:
                    files_to_delete.append(file_path)
    
    # 删除文件
    deleted_count = 0
    deleted_size = 0
    for file_path in files_to_delete:
        try:
            file_size = os.path.getsize(file_path)
            os.remove(file_path)
            deleted_count += 1
            deleted_size += file_size
        except Exception as e:
            print(f"  删除失败: {os.path.basename(file_path)} - {e}")
    
    print(f"="*70)
    print(f"清理完成")
    print(f"="*70)
    print(f"删除文件数: {deleted_count}")
    print(f"释放空间: {deleted_size / 1024 / 1024:.2f} MB")
    print(f"\n保留的文件:")
    print(f"  - price_chart_current.png（实时图表）")
    print(f"  - dashboard.png（仪表板）")
    print(f"  - 历史图表: {len(hourly_files_to_keep)} 个（每小时一个）")

if __name__ == "__main__":
    print("="*70)
    print("清理旧的图表文件")
    print("="*70)
    cleanup_old_charts()

