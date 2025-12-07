"""
实时数据可视化模块
支持实时图表、指标计算可视化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import os
import threading
import time
import io

# 抑制matplotlib的字体警告
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    MATPLOTLIB_AVAILABLE = True
    
    # 配置中文字体支持（优先使用Microsoft YaHei）
    try:
        import platform
        import matplotlib.font_manager as fm
        
        # Windows系统：优先使用Microsoft YaHei
        if platform.system() == 'Windows':
            # 尝试找到Microsoft YaHei字体
            font_found = False
            try:
                font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
                # 优先查找Microsoft YaHei（微软雅黑）
                msyh_fonts = [f for f in font_list if 'msyh' in f.lower() or 'microsoft yahei' in f.lower()]
                if msyh_fonts:
                    font_path = msyh_fonts[0]
                    font_prop = fm.FontProperties(fname=font_path)
                    plt.rcParams['font.family'] = font_prop.get_name()
                    font_found = True
                elif not font_found:
                    # 其次尝试SimHei（黑体）
                    simhei_fonts = [f for f in font_list if 'simhei' in f.lower() or 'simhei' in f.lower()]
                    if simhei_fonts:
                        font_path = simhei_fonts[0]
                        font_prop = fm.FontProperties(fname=font_path)
                        plt.rcParams['font.family'] = font_prop.get_name()
                        font_found = True
            except Exception:
                pass
            
            # 如果找不到字体文件，使用字体名称列表（matplotlib会自动选择）
            if not font_found:
                plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
        else:
            # Linux/Mac系统字体设置
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'WenQuanYi Micro Hei', 'sans-serif']
        
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
    except Exception as e:
        # 如果所有字体配置都失败，使用默认设置
        try:
            plt.rcParams['font.sans-serif'] = ['sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception:
            pass
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib未安装，可视化功能将受限")

try:
    from flask import Flask, render_template_string, Response
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("⚠️  Flask未安装，Web可视化将受限")


class RealTimeVisualizer:
    """实时数据可视化器"""
    
    def __init__(self,
                 data_window_size: int = 100,
                 update_interval: float = 1.0,
                 output_dir: str = "visualization_output",
                 max_history_files: int = 24):  # V12优化：最大保留历史图表文件数（24小时）
        """
        初始化可视化器
        
        参数:
            data_window_size: 数据窗口大小（显示最近N个数据点）
            update_interval: 更新间隔（秒）
            output_dir: 输出目录
            max_history_files: 最大保留历史图表文件数（默认24个，即最近24小时）
        """
        self.data_window_size = data_window_size
        self.update_interval = update_interval
        self.output_dir = output_dir
        self.max_history_files = max_history_files  # V12优化：限制历史文件数量
        os.makedirs(output_dir, exist_ok=True)
        
        # V12优化：记录上次保存历史图表的小时
        self.last_hourly_save = None
        
        # 数据存储
        self.price_history = []
        self.volume_history = []
        self.indicators_history = {}
        self.predictions_history = []
        self.timestamps = []
        
        # 图表配置
        self.fig = None
        self.axes = None
        
        # 线程控制
        self.running = False
        self.update_thread = None
        
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  matplotlib不可用，只能使用数据导出功能")
    
    def add_data_point(self,
                      price: float,
                      volume: float = 0,
                      indicators: Optional[Dict] = None,
                      prediction: Optional[float] = None,
                      timestamp: Optional[datetime] = None):
        """
        添加数据点
        
        参数:
            price: 价格
            volume: 成交量
            indicators: 技术指标字典
            prediction: 预测值
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.predictions_history.append(prediction)
        self.timestamps.append(timestamp)
        
        if indicators:
            for key, value in indicators.items():
                if key not in self.indicators_history:
                    self.indicators_history[key] = []
                self.indicators_history[key].append(value)
        
        # 保持窗口大小
        if len(self.price_history) > self.data_window_size:
            self.price_history.pop(0)
            self.volume_history.pop(0)
            self.predictions_history.pop(0)
            self.timestamps.pop(0)
            for key in self.indicators_history:
                if len(self.indicators_history[key]) > 0:
                    self.indicators_history[key].pop(0)
    
    def plot_price_chart(self, save_path: Optional[str] = None) -> Optional[bytes]:
        """
        绘制价格图表
        
        参数:
            save_path: 保存路径（如果为None，返回图像字节）
        
        返回:
            图像字节（如果save_path为None）
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        # 确保字体配置正确（静默失败，避免错误信息）
        # 注意：字体警告已在模块级别被抑制
        try:
            import platform
            if platform.system() == 'Windows':
                plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
            else:
                plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'WenQuanYi Micro Hei', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception:
            pass  # 如果配置失败，使用默认设置
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 如果没有数据，显示提示
        if len(self.price_history) == 0:
            ax.text(0.5, 0.5, '等待数据中...\nWaiting for data...', 
                   ha='center', va='center', fontsize=16, transform=ax.transAxes)
            ax.axis('off')
            # 使用warnings来抑制tight_layout的警告
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    plt.tight_layout()
            except Exception:
                pass
        else:
            # 准备数据
            prices = self.price_history[-self.data_window_size:]
            timestamps = self.timestamps[-self.data_window_size:] if len(self.timestamps) > 0 else None
            
            x = range(len(prices)) if timestamps is None else timestamps
            
            # 绘制价格
            ax.plot(x, prices, label='价格', color='blue', linewidth=2)
            
            # 绘制预测值（如果有）
            if len(self.predictions_history) > 0:
                predictions = [p for p in self.predictions_history[-self.data_window_size:] if p is not None]
                if len(predictions) > 0:
                    pred_indices = [i for i, p in enumerate(self.predictions_history[-self.data_window_size:]) if p is not None]
                    if timestamps is None:
                        pred_x = [i for i in pred_indices]
                    else:
                        pred_x = [timestamps[-self.data_window_size:][i] for i in pred_indices]
                    ax.scatter(pred_x, predictions, label='预测', color='red', marker='x', s=50)
            
            ax.set_xlabel('时间' if timestamps else '索引')
            ax.set_ylabel('价格')
            ax.set_title('实时价格走势')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 使用warnings来抑制tight_layout的警告
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    plt.tight_layout()
            except Exception:
                # 如果tight_layout失败，跳过
                pass
        
        # 保存或返回图片
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return None
        else:
            # 转换为字节
            try:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                img_bytes = buf.read()
                buf.close()
                plt.close(fig)
                return img_bytes
            except Exception as e:
                plt.close(fig)
                return None
    
    def plot_indicators(self, save_path: Optional[str] = None) -> Optional[bytes]:
        """
        绘制技术指标图表
        
        参数:
            save_path: 保存路径
        
        返回:
            图像字节
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        # 确保字体配置正确
        try:
            import platform
            if platform.system() == 'Windows':
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
            else:
                plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'WenQuanYi Micro Hei', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception:
            pass
        
        # 如果没有指标数据，显示提示
        if len(self.indicators_history) == 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, '等待指标数据...\nWaiting for indicator data...', 
                   ha='center', va='center', fontsize=16, transform=ax.transAxes)
            ax.axis('off')
        else:
            num_indicators = len(self.indicators_history)
            fig, axes = plt.subplots(num_indicators, 1, figsize=(12, 4 * num_indicators))
            
            if num_indicators == 1:
                axes = [axes]
            
            for idx, (indicator_name, values) in enumerate(self.indicators_history.items()):
                if len(values) == 0:
                    continue
                
                ax = axes[idx]
                indicator_values = values[-self.data_window_size:]
                x = range(len(indicator_values))
                
                ax.plot(x, indicator_values, label=indicator_name, linewidth=2)
                ax.set_ylabel(indicator_name)
                ax.set_title(f'{indicator_name} 指标')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.xlabel('时间')
            # 使用warnings来抑制tight_layout的警告
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    plt.tight_layout()
            except Exception:
                # 如果tight_layout失败，跳过
                pass
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return None
        else:
            try:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                img_bytes = buf.read()
                buf.close()
                plt.close(fig)
                return img_bytes
            except Exception as e:
                plt.close(fig)
                return None
    
    def plot_comprehensive_dashboard(self, save_path: Optional[str] = None) -> Optional[bytes]:
        """
        绘制综合仪表板
        
        参数:
            save_path: 保存路径
        
        返回:
            图像字节
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        # 确保字体配置正确
        try:
            import platform
            if platform.system() == 'Windows':
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
            else:
                plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'WenQuanYi Micro Hei', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception:
            pass
        
        fig = plt.figure(figsize=(16, 10))
        
        # 如果没有数据，显示提示
        if len(self.price_history) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, '等待仪表板数据...\nWaiting for dashboard data...', 
                   ha='center', va='center', fontsize=16, transform=ax.transAxes)
            ax.axis('off')
        else:
            # 改进布局：增加间距，避免重叠
            gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.35, 
                                 left=0.08, right=0.95, top=0.95, bottom=0.08)
            
            # 1. 价格走势（主图）
            ax1 = fig.add_subplot(gs[0:2, :])
            prices = self.price_history[-self.data_window_size:]
            if len(prices) > 0:
                x = list(range(len(prices)))
                ax1.plot(x, prices, label='价格', color='blue', linewidth=2, marker='o', markersize=4)
                
                if len(self.predictions_history) > 0:
                    predictions = [p for p in self.predictions_history[-self.data_window_size:] if p is not None]
                    pred_indices = [i for i, p in enumerate(self.predictions_history[-self.data_window_size:]) if p is not None]
                    if predictions and len(pred_indices) > 0:
                        ax1.scatter([x[i] for i in pred_indices if i < len(x)], 
                                   [predictions[j] for j, i in enumerate(pred_indices) if i < len(x)], 
                                   label='预测', color='red', marker='x', s=50)
                
                # 设置X轴范围，避免显示异常
                if len(x) > 1:
                    ax1.set_xlim(-0.5, len(x) - 0.5)
                else:
                    ax1.set_xlim(-0.5, 0.5)
                ax1.set_xlabel('时间')
                ax1.set_ylabel('价格')
                ax1.set_title('实时价格走势')
                ax1.legend(loc='upper left')
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, '等待价格数据...\nWaiting for price data...', 
                        ha='center', va='center', fontsize=14, transform=ax1.transAxes)
                ax1.set_xlabel('时间')
                ax1.set_ylabel('价格')
                ax1.set_title('实时价格走势')
            
            # 2. 成交量（如果有）
            if len(self.volume_history) > 0 and any(v > 0 for v in self.volume_history):
                ax2 = fig.add_subplot(gs[2, 0])
                volumes = self.volume_history[-self.data_window_size:]
                if len(volumes) > 0:
                    x_vol = list(range(len(volumes)))
                    ax2.bar(x_vol, volumes, color='green', alpha=0.6, width=0.8)
                    # 设置X轴范围，避免显示异常
                    if len(x_vol) > 1:
                        ax2.set_xlim(-0.5, len(x_vol) - 0.5)
                    else:
                        ax2.set_xlim(-0.5, 0.5)
                    ax2.set_xlabel('时间')
                    ax2.set_ylabel('成交量')
                    ax2.set_title('成交量')
                    ax2.grid(True, alpha=0.3)
                else:
                    ax2.text(0.5, 0.5, '等待成交量数据...', 
                            ha='center', va='center', fontsize=12, transform=ax2.transAxes)
                    ax2.set_xlabel('时间')
                    ax2.set_ylabel('成交量')
                    ax2.set_title('成交量')
            
            # 3-6. 技术指标（最多4个）
            indicator_items = list(self.indicators_history.items())[:4]
            positions = [(2, 1), (3, 0), (3, 1)]
            
            for idx, (indicator_name, values) in enumerate(indicator_items):
                if idx >= len(positions):
                    break
                
                if len(values) == 0:
                    continue
                
                row, col = positions[idx]
                ax = fig.add_subplot(gs[row, col])
                indicator_values = values[-self.data_window_size:]
                if len(indicator_values) > 0:
                    x_indices = list(range(len(indicator_values)))
                    ax.plot(x_indices, indicator_values, 
                           label=indicator_name, linewidth=1.5, marker='o', markersize=3)
                    # 设置X轴范围，避免显示异常
                    if len(x_indices) > 1:
                        ax.set_xlim(-0.5, len(x_indices) - 0.5)
                    else:
                        ax.set_xlim(-0.5, 0.5)
                    ax.set_xlabel('时间')
                    ax.set_ylabel(indicator_name)
                    ax.set_title(indicator_name)
                    ax.legend(loc='upper right', fontsize=8)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, f'等待{indicator_name}数据...', 
                           ha='center', va='center', fontsize=10, transform=ax.transAxes)
                    ax.set_xlabel('时间')
                    ax.set_ylabel(indicator_name)
                    ax.set_title(indicator_name)
            
            plt.suptitle('实时交易仪表板', fontsize=16, y=0.98)
            
            # 使用tight_layout确保布局不重叠
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    plt.tight_layout(rect=[0, 0, 1, 0.98])  # 为suptitle留出空间
            except Exception:
                pass
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return None
        else:
            try:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                img_bytes = buf.read()
                buf.close()
                plt.close(fig)
                return img_bytes
            except Exception as e:
                plt.close(fig)
                return None
    
    def export_data(self, filepath: str):
        """
        导出数据到CSV
        
        参数:
            filepath: 文件路径
        """
        data = {
            'timestamp': [ts.isoformat() if isinstance(ts, datetime) else str(ts) 
                         for ts in self.timestamps[-self.data_window_size:]],
            'price': self.price_history[-self.data_window_size:],
            'volume': self.volume_history[-self.data_window_size:],
            'prediction': [p if p is not None else np.nan 
                          for p in self.predictions_history[-self.data_window_size:]]
        }
        
        # 添加技术指标
        for key, values in self.indicators_history.items():
            data[key] = values[-self.data_window_size:]
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
    
    def generate_html_dashboard(self, filepath: str):
        """
        生成HTML仪表板
        
        参数:
            filepath: 输出文件路径
        """
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>实时交易仪表板</title>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="5">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .dashboard { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .stats { display: flex; gap: 20px; margin-bottom: 20px; }
        .stat-card { flex: 1; background: #f0f0f0; padding: 15px; border-radius: 6px; }
        .stat-card h3 { margin: 0 0 10px 0; color: #333; }
        .stat-value { font-size: 24px; font-weight: bold; color: #007bff; }
        img { max-width: 100%; height: auto; margin: 10px 0; }
        .update-time { text-align: right; color: #666; font-size: 12px; }
    </style>
</head>
<body>
    <div class="dashboard">
        <h1>实时交易仪表板</h1>
        <div class="update-time">更新时间: {update_time}</div>
        
        <div class="stats">
            <div class="stat-card">
                <h3>当前价格</h3>
                <div class="stat-value">{current_price:.2f}</div>
            </div>
            <div class="stat-card">
                <h3>数据点数</h3>
                <div class="stat-value">{data_count}</div>
            </div>
            <div class="stat-card">
                <h3>最新预测</h3>
                <div class="stat-value">{latest_prediction}</div>
            </div>
        </div>
        
        <h2>价格走势</h2>
        <img src="price_chart.png" alt="价格走势图">
        
        <h2>技术指标</h2>
        <img src="indicators_chart.png" alt="技术指标图">
    </div>
</body>
</html>
"""
        
        current_price = self.price_history[-1] if self.price_history else 0
        latest_prediction = self.predictions_history[-1] if self.predictions_history else None
        latest_prediction = f"{latest_prediction:.2f}" if latest_prediction else "N/A"
        
        html_content = html_template.format(
            update_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            current_price=current_price,
            data_count=len(self.price_history),
            latest_prediction=latest_prediction
        )
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def cleanup_old_history_files(self):
        """V12优化：清理旧的历史图表文件，只保留最新的N个"""
        try:
            import glob
            # 查找所有历史图表文件（price_chart_hourly_*.png）
            pattern = os.path.join(self.output_dir, 'price_chart_hourly_*.png')
            history_files = glob.glob(pattern)
            
            if len(history_files) > self.max_history_files:
                # 按修改时间排序（最新的在前）
                history_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                # 删除超出数量的旧文件
                for old_file in history_files[self.max_history_files:]:
                    try:
                        os.remove(old_file)
                    except:
                        pass
        except Exception as e:
            # 静默处理清理错误
            pass
    
    def auto_update_charts(self):
        """自动更新图表（后台线程）- V12优化：减少文件数量"""
        while self.running:
            try:
                now = datetime.now()
                current_hour = now.strftime('%Y%m%d_%H')  # 只精确到小时
                
                # V12优化：实时图表覆盖同一个文件（用于Web界面实时显示）
                try:
                    # 实时图表：覆盖同一个文件，不生成新文件
                    price_current_path = os.path.join(self.output_dir, 'price_chart_current.png')
                    self.plot_price_chart(save_path=price_current_path)
                except Exception as e:
                    # 静默处理价格图错误
                    pass
                
                # V12优化：每小时保存一个历史图表（用于历史记录）
                should_save_hourly = (self.last_hourly_save != current_hour)
                if should_save_hourly:
                    try:
                        price_hourly_path = os.path.join(self.output_dir, f'price_chart_hourly_{current_hour}.png')
                        self.plot_price_chart(save_path=price_hourly_path)
                        self.last_hourly_save = current_hour
                        
                        # 清理旧的历史文件
                        self.cleanup_old_history_files()
                    except Exception as e:
                        # 静默处理历史图表保存错误
                        pass
                
                # 综合仪表板（覆盖同一个文件）
                try:
                    dashboard_path = os.path.join(self.output_dir, 'dashboard.png')
                    self.plot_comprehensive_dashboard(save_path=dashboard_path)
                except Exception as e:
                    # 静默处理仪表板错误
                    pass
                
                # HTML仪表板（覆盖同一个文件）
                try:
                    html_path = os.path.join(self.output_dir, 'dashboard.html')
                    self.generate_html_dashboard(html_path)
                except Exception as e:
                    # 静默处理HTML错误
                    pass
                
                # 导出数据（覆盖同一个文件）
                try:
                    csv_path = os.path.join(self.output_dir, 'data.csv')
                    self.export_data(csv_path)
                except Exception as e:
                    # 静默处理数据导出错误
                    pass
                
                time.sleep(self.update_interval)
            except Exception as e:
                # 只在关键错误时输出信息
                if 'font' not in str(e).lower():
                    print(f"⚠️  图表更新失败: {e}")
                time.sleep(self.update_interval)
    
    def start_auto_update(self):
        """启动自动更新"""
        if self.running:
            return
        
        self.running = True
        self.update_thread = threading.Thread(target=self.auto_update_charts, daemon=True)
        self.update_thread.start()
        print(f"✅ 实时可视化自动更新已启动（输出目录: {self.output_dir}）")
    
    def stop_auto_update(self):
        """停止自动更新"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=2)


class WebVisualizationServer:
    """Web可视化服务器（基于Flask）"""
    
    def __init__(self, visualizer: RealTimeVisualizer, port: int = 8080):
        """
        初始化Web服务器
        
        参数:
            visualizer: 可视化器实例
            port: 端口号
        """
        if not FLASK_AVAILABLE:
            raise ImportError("Flask未安装，无法启动Web服务器")
        
        self.visualizer = visualizer
        self.port = port
        self.app = Flask(__name__)
        
        # 禁用Flask的访问日志，避免干扰其他输出
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)  # 只显示错误，不显示访问日志
        
        self.setup_routes()
        self.running = False
    
    def setup_routes(self):
        """设置路由"""
        
        @self.app.route('/')
        def index():
            return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>实时交易可视化</title>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="5">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        img { max-width: 100%; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>实时交易可视化</h1>
    <img src="/plot/price" alt="价格图">
    <img src="/plot/indicators" alt="指标图">
    <img src="/plot/dashboard" alt="仪表板">
</body>
</html>
""")
        
        @self.app.route('/plot/price')
        def plot_price():
            try:
                img_bytes = self.visualizer.plot_price_chart()
                if img_bytes is None or len(img_bytes) == 0:
                    # 返回一个简单的占位图（使用matplotlib生成）
                    if MATPLOTLIB_AVAILABLE:
                        try:
                            fig, ax = plt.subplots(figsize=(8, 4))
                            ax.text(0.5, 0.5, '暂无数据\nWaiting for data...', 
                                   ha='center', va='center', fontsize=16)
                            ax.axis('off')
                            canvas = FigureCanvasAgg(fig)
                            canvas.draw()
                            buf = io.BytesIO()
                            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                            buf.seek(0)
                            plt.close(fig)
                            return Response(buf.read(), mimetype='image/png')
                        except Exception as e:
                            pass
                    # 如果生成占位图失败，返回最小的1x1透明PNG
                    import base64
                    minimal_png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==')
                    return Response(minimal_png, mimetype='image/png')
                return Response(img_bytes, mimetype='image/png')
            except Exception as e:
                # 发生错误时返回占位图
                if MATPLOTLIB_AVAILABLE:
                    try:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.text(0.5, 0.5, f'图表加载错误\nError loading chart', 
                               ha='center', va='center', fontsize=16)
                        ax.axis('off')
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                        buf.seek(0)
                        plt.close(fig)
                        return Response(buf.read(), mimetype='image/png')
                    except:
                        pass
                # 返回最小的透明PNG
                import base64
                minimal_png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==')
                return Response(minimal_png, mimetype='image/png')
        
        @self.app.route('/plot/indicators')
        def plot_indicators():
            try:
                img_bytes = self.visualizer.plot_indicators()
                if img_bytes is None or len(img_bytes) == 0:
                    # 返回一个简单的占位图
                    if MATPLOTLIB_AVAILABLE:
                        try:
                            fig, ax = plt.subplots(figsize=(8, 4))
                            ax.text(0.5, 0.5, '暂无指标数据\nWaiting for indicator data...', 
                                   ha='center', va='center', fontsize=16)
                            ax.axis('off')
                            buf = io.BytesIO()
                            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                            buf.seek(0)
                            plt.close(fig)
                            return Response(buf.read(), mimetype='image/png')
                        except Exception as e:
                            pass
                    # 返回最小的透明PNG
                    import base64
                    minimal_png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==')
                    return Response(minimal_png, mimetype='image/png')
                return Response(img_bytes, mimetype='image/png')
            except Exception as e:
                # 发生错误时返回占位图
                if MATPLOTLIB_AVAILABLE:
                    try:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.text(0.5, 0.5, f'图表加载错误\nError loading chart', 
                               ha='center', va='center', fontsize=16)
                        ax.axis('off')
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                        buf.seek(0)
                        plt.close(fig)
                        return Response(buf.read(), mimetype='image/png')
                    except:
                        pass
                # 返回最小的透明PNG
                import base64
                minimal_png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==')
                return Response(minimal_png, mimetype='image/png')
        
        @self.app.route('/plot/dashboard')
        def plot_dashboard():
            try:
                img_bytes = self.visualizer.plot_comprehensive_dashboard()
                if img_bytes is None or len(img_bytes) == 0:
                    # 返回一个简单的占位图
                    if MATPLOTLIB_AVAILABLE:
                        try:
                            fig, ax = plt.subplots(figsize=(12, 6))
                            ax.text(0.5, 0.5, '暂无仪表板数据\nWaiting for dashboard data...', 
                                   ha='center', va='center', fontsize=16)
                            ax.axis('off')
                            buf = io.BytesIO()
                            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                            buf.seek(0)
                            plt.close(fig)
                            return Response(buf.read(), mimetype='image/png')
                        except Exception as e:
                            pass
                    # 返回最小的透明PNG
                    import base64
                    minimal_png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==')
                    return Response(minimal_png, mimetype='image/png')
                return Response(img_bytes, mimetype='image/png')
            except Exception as e:
                # 发生错误时返回占位图
                if MATPLOTLIB_AVAILABLE:
                    try:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.text(0.5, 0.5, f'图表加载错误\nError loading chart', 
                               ha='center', va='center', fontsize=16)
                        ax.axis('off')
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                        buf.seek(0)
                        plt.close(fig)
                        return Response(buf.read(), mimetype='image/png')
                    except:
                        pass
                # 返回最小的透明PNG
                import base64
                minimal_png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==')
                return Response(minimal_png, mimetype='image/png')
        
        @self.app.route('/api/data')
        def api_data():
            data = {
                'prices': self.visualizer.price_history[-self.visualizer.data_window_size:],
                'volumes': self.visualizer.volume_history[-self.visualizer.data_window_size:],
                'predictions': [p if p is not None else None 
                               for p in self.visualizer.predictions_history[-self.visualizer.data_window_size:]],
                'indicators': {k: v[-self.visualizer.data_window_size:] 
                              for k, v in self.visualizer.indicators_history.items()}
            }
            return json.dumps(data, ensure_ascii=False)
    
    def start(self, host: str = '127.0.0.1', debug: bool = False):
        """启动服务器"""
        if self.running:
            print(f"⚠️  Web服务器已在运行")
            return
        
        def run():
            try:
                self.app.run(host=host, port=self.port, debug=debug, use_reloader=False, threaded=True)
            except Exception as e:
                print(f"⚠️  Web服务器运行错误: {e}")
                self.running = False
        
        self.running = True
        server_thread = threading.Thread(target=run, daemon=True)
        server_thread.start()
        
        # 等待一小段时间确保服务器启动
        time.sleep(1)
        
        if self.running:
            print(f"✅ Web可视化服务器已启动: http://{host}:{self.port}")
        else:
            print(f"⚠️  Web服务器启动可能失败，请检查端口 {self.port} 是否被占用")

