import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import importlib

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入回测引擎和策略
from strategies.range_trading_strategy import RangeTradingStrategy
from backtest_engine import BacktestEngine, load_data

def main():
    """主函数"""
    # 数据文件路径
    data_file = 'data/BTC_USDT_15m.csv'  # 确保这是你的数据文件路径
    
    # 加载数据
    data = load_data(data_file)
    if data is None:
        return
    
    print(f"加载了 {len(data)} 条数据，时间范围: {data.index[0]} 到 {data.index[-1]}")
    
    # 定义时间段
    time_periods = {
        "2022-01-01_to_2022-11-30": ("2022-01-01", "2022-11-30"),  # 熊市
        "2022-12-01_to_2023-06-30": ("2022-12-01", "2023-06-30"),  # 底部盘整期
        "2023-07-01_to_2024-03-31": ("2023-07-01", "2024-03-31"),  # 复苏期
        "2024-04-01_to_2024-09-30": ("2024-04-01", "2024-09-30"),  # 减半前盘整
        "2024-10-01_to_2025-03-31": ("2024-10-01", "2025-03-31") # 牛市主升浪
        # "2025-04-01_to_2025-03-31": ("2025-04-01", "2025-03-31")   # 牛市末期
    }
    
    # 回测参数
    initial_capital = 10000      # 初始资金
    commission = 0.001           # 手续费率 (0.1%)
    
    # 遍历每个时间段进行回测
    for period_name, (start_date, end_date) in time_periods.items():
        print(f"正在回测时间段: {period_name} ({start_date} 到 {end_date})")
        
        # 过滤数据
        filtered_data = data[(data.index >= start_date) & (data.index <= end_date)]
        
        # 创建策略实例
        strategy = RangeTradingStrategy(
            lookback_period=50,    # 支撑/阻力回溯周期
            obv_length=14,         # OBV均线周期
            atr_length=14,         # ATR周期
            sl_multiplier=1.0,     # 降低止损ATR倍数
            tp_multiplier=2.0,     # 降低止盈ATR倍数
            max_position_size=0.1  # 最大仓位大小
        )
        
        # 创建回测引擎
        engine = BacktestEngine(
            data=filtered_data,
            strategy=strategy,
            initial_capital=initial_capital,
            commission=commission
        )
        
        # 运行回测
        results, trades_df, metrics = engine.run()
        
        # 打印性能指标
        engine.print_metrics()
        
        # 保存结果
        results_path = os.path.join("results", f"{period_name}_results.csv")
        trades_path = os.path.join("results", f"{period_name}_trades.csv")
        engine.save_results(results_path, trades_path)
        
        # 绘制结果
        plot_path = os.path.join("results", f"{period_name}_plot.png")
        engine.plot_results(save_path=plot_path)
        print(f"图表已保存到 {plot_path}")

if __name__ == "__main__":
    main() 