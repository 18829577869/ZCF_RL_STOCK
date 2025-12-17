# -*- coding: utf-8 -*-
"""
批量更新所有V16预测文件，添加回测统计信息功能
"""
import os
import re
import glob

# 回测统计代码片段
BACKTEST_CODE = '''# ==================== V16新增：模型回测统计信息 ====================
# 在模型加载后，对对应股票进行回测，显示总收益率、夏普比率、最大回撤
if ppo_model and PPO_AVAILABLE:
    try:
        from stock_env_v6 import StockTradingEnv
        
        # 根据股票代码查找对应的测试数据文件
        stock_code_num = STOCK_CODE.split('.')[-1]  # 提取股票代码数字部分
        test_data_dir = f'stockdata_v7_{stock_code_num}/test'
        
        if os.path.exists(test_data_dir):
            print("\\n" + "=" * 70)
            print("📊 V16: 开始模型回测统计...")
            print("=" * 70)
            
            # 查找对应股票的测试文件
            test_files = [os.path.join(test_data_dir, f) for f in os.listdir(test_data_dir) 
                         if f.endswith('.csv') and stock_code_num in f]
            
            if test_files:
                # 使用第一个找到的测试文件
                test_file = test_files[0]
                print(f"📁 测试数据文件: {test_file}")
                
                try:
                    # 初始化环境（使用与训练时相同的初始资金）
                    initial_balance = 20000.0  # 默认初始资金2万元
                    env = StockTradingEnv(test_file, initial_balance=initial_balance)
                    obs, _ = env.reset()
                    done = False
                    
                    # 执行回测
                    step_count = 0
                    while not done:
                        action, _ = ppo_model.predict(obs, deterministic=True)
                        obs, reward, done, truncated, _ = env.step(action)
                        step_count += 1
                        if step_count % 100 == 0:
                            print(f"   回测进度: {step_count} 步...", end='\\r')
                    
                    # 获取回测统计信息
                    stats = env.get_stats()
                    
                    if stats:
                        print("\\n" + "=" * 70)
                        print("📈 模型回测统计结果")
                        print("=" * 70)
                        print(f"股票代码: {STOCK_CODE}")
                        print(f"测试数据: {os.path.basename(test_file)}")
                        print(f"初始资金: {initial_balance:,.2f} 元")
                        print(f"最终净值: {stats.get('final_net_worth', 0):,.2f} 元")
                        print(f"总收益率: {stats.get('total_return', 0):+.2f}%")
                        print(f"夏普比率: {stats.get('sharpe_ratio', 0):.2f}")
                        print(f"最大回撤: {stats.get('max_drawdown', 0):.2f}%")
                        print(f"交易次数: {stats.get('num_trades', 0)}")
                        print(f"胜率: {stats.get('win_rate', 0):.2f}%")
                        print(f"风险事件: {stats.get('risk_events', 0)} 次")
                        print("=" * 70)
                    else:
                        print("⚠️  回测完成，但未获取到统计信息")
                        
                except Exception as e:
                    print(f"⚠️  回测执行失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"⚠️  未找到股票代码 {stock_code_num} 的测试数据文件")
                print(f"   测试数据目录: {test_data_dir}")
        else:
            print(f"⚠️  测试数据目录不存在: {test_data_dir}")
            print("   提示: 请先运行对应的数据获取脚本生成测试数据")
            
    except ImportError:
        print("⚠️  无法导入 StockTradingEnv，跳过回测统计")
    except Exception as e:
        print(f"⚠️  回测统计功能初始化失败: {e}")

print()
'''

# 查找所有V16文件
v16_files = glob.glob('real_time_predict_v16_*.py')
v16_files.sort()

print(f"找到 {len(v16_files)} 个V16文件")
print("=" * 70)

updated_count = 0
skipped_count = 0

for file_path in v16_files:
    print(f"处理: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否已经包含回测统计代码
        if 'V16新增：模型回测统计信息' in content:
            print(f"  ⏭️  已包含回测统计代码，跳过")
            skipped_count += 1
            continue
        
        # 查找插入位置：在 "print('=' * 70)" 和 "初始化交易日志" 之间
        pattern = r'(print\("=" \* 70\)\s+print\(\)\s+)(# 初始化交易日志)'
        match = re.search(pattern, content)
        
        if match:
            # 替换：在 print("=" * 70) 和 print() 之后，初始化交易日志之前插入回测代码
            replacement = match.group(1) + BACKTEST_CODE + match.group(2)
            new_content = content[:match.start()] + replacement + content[match.end():]
            
            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"  ✅ 已添加回测统计代码")
            updated_count += 1
        else:
            # 尝试另一种模式
            pattern2 = r'(print\("=" \* 70\)\s+print\(\)\s+)(\n# 初始化交易日志)'
            match2 = re.search(pattern2, content)
            
            if match2:
                replacement = match2.group(1) + BACKTEST_CODE + match2.group(2)
                new_content = content[:match2.start()] + replacement + content[match2.end():]
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"  ✅ 已添加回测统计代码（模式2）")
                updated_count += 1
            else:
                print(f"  ⚠️  未找到插入位置，跳过")
                skipped_count += 1
                
    except Exception as e:
        print(f"  ❌ 处理失败: {e}")
        skipped_count += 1

print("=" * 70)
print(f"完成！")
print(f"  更新: {updated_count} 个文件")
print(f"  跳过: {skipped_count} 个文件")
print(f"  总计: {len(v16_files)} 个文件")

