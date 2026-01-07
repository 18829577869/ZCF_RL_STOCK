# run_custom_stocks.py - 一键运行自定义股票数据获取、训练和回测
# -*- coding: utf-8 -*-
"""
一键运行脚本：
1. 获取10只自定义股票数据
2. 训练模型
3. 回测评估
"""
import subprocess
import sys
import os

def run_step(step_name, script_name):
    """运行单个步骤"""
    print("\n" + "="*70)
    print(f"步骤: {step_name}")
    print("="*70)
    print(f"运行: {script_name}")
    print("="*70 + "\n")
    
    result = subprocess.run([sys.executable, script_name], 
                          capture_output=False, 
                          text=True)
    
    if result.returncode != 0:
        print(f"\n[错误] {step_name} 失败，退出码: {result.returncode}")
        return False
    
    print(f"\n[成功] {step_name} 完成")
    return True

if __name__ == "__main__":
    print("="*70)
    print("V11 自定义股票 - 一键运行")
    print("="*70)
    print("包含10只股票：")
    print("  - 藏格矿业 (000408)")
    print("  - 中国卫星 (600118)")
    print("  - 亚钾国际 (000893)")
    print("  - 神火股份 (000933)")
    print("  - 云铝股份 (000807)")
    print("  - 盛达资源 (000603)")
    print("  - 航天电子 (600879)")
    print("  - 精达股份 (600577)")
    print("  - 驰宏锌锗 (600497)")
    print("  - 南山铝业 (600219)")
    print("="*70)
    
    # 步骤1: 获取数据
    if not run_step("数据获取", "get_stock_data_v11_custom.py"):
        print("\n[终止] 数据获取失败，请检查网络连接和股票代码")
        sys.exit(1)
    
    # 检查数据是否足够
    train_dir = 'stockdata_v11_custom/train'
    if os.path.exists(train_dir):
        train_files = [f for f in os.listdir(train_dir) if f.endswith('.csv')]
        print(f"\n[检查] 已获取 {len(train_files)} 只股票的训练数据")
        if len(train_files) < 5:
            print(f"[警告] 数据不足5只，建议至少5只才能训练")
            response = input("是否继续训练？(y/n): ")
            if response.lower() != 'y':
                print("[终止] 用户取消训练")
                sys.exit(0)
    
    # 步骤2: 训练模型
    if not run_step("模型训练", "train_v11_custom.py"):
        print("\n[错误] 训练失败")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("[完成] 所有步骤完成！")
    print("="*70)
    print("模型文件: ppo_stock_v11_custom.zip")
    print("训练日志: ./logs_v11_custom/")
    print("模型检查点: ./models_v11_custom/")
    print("\n可以使用以下命令查看训练曲线：")
    print("  tensorboard --logdir=./logs_v11_custom/")

