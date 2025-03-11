import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import importlib

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# 导入回测引擎和策略
from backtest_engine import BacktestEngine, load_data
from strategies.slingshot_strategy import SlingshotStrategy

def main():
    """主函数"""
    # 数据文件路径
    data_file = 'data/BTC_USDT_15m.csv'  # 确保这是你的数据文件路径
    
    # 定义时间段
    time_periods = {
        "2022-01-01_to_2022-11-30": ("2022-01-01", "2022-11-30"),  # 熊市
        "2022-12-01_to_2023-06-30": ("2022-12-01", "2023-06-30"),  # 底部盘整期
        "2023-07-01_to_2024-03-31": ("2023-07-01", "2024-03-31"),  # 复苏期
        "2024-04-01_to_2024-09-30": ("2024-04-01", "2024-09-30"),  # 减半前盘整
        "2024-10-01_to_2025-03-31": ("2024-10-01", "2025-03-31") # 牛市主升浪
        # "2025-04-01_to_2025-03-31": ("2025-04-01", "2025-03-31")   # 牛市末期
    }
    
    # 创建策略实例
    strategy = SlingshotStrategy(
        ema_slow_length=25,       # 慢EMA周期
        ema_fast_length=7,        # 快EMA周期
        rsi_length=14,            # RSI周期
        dmi_length=14,            # DMI周期
        atr_length=14,            # ATR周期
        fib_lookback=50,          # 斐波那契回溯周期
        bb_length=20,             # 布林带周期
        bb_dev=2.0,               # 布林带标准差
        volatility_threshold=0.002, # 波动率阈值
        sl_multiplier=1.5,        # 止损ATR倍数
        tp_multiplier=3.5,        # 止盈ATR倍数
        trail_sl=True              # 启用追踪止损
    )
    
    # 回测参数
    initial_capital = 10000      # 初始资金
    commission = 0.001           # 手续费率 (0.1%)
    
    # 输出选项
    output_prefix = 'results'    # 输出文件前缀
    show_plot = True             # 显示图表
    save_plot = True             # 保存图表
    
    # 加载数据
    data = load_data(data_file)
    if data is None:
        return
    
    print(f"加载了 {len(data)} 条数据，时间范围: {data.index[0]} 到 {data.index[-1]}")
    
    print(f"使用策略: {strategy.name}")
    
    # 遍历每个时间段进行回测
    for period_name, (start_date, end_date) in time_periods.items():
        print(f"正在回测时间段: {period_name} ({start_date} 到 {end_date})")
        
        # 过滤数据
        filtered_data = data[(data.index >= start_date) & (data.index <= end_date)]
        
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
        results_path = os.path.join("results", f"{output_prefix}_SlingshotStrategy_{period_name}_results.csv")
        trades_path = os.path.join("results", f"{output_prefix}_SlingshotStrategy_{period_name}_trades.csv")
        engine.save_results(results_path, trades_path)
        
        # 绘制结果
        if show_plot:
            engine.plot_results()
        
        if save_plot:
            plot_path = os.path.join("results", f"{output_prefix}_SlingshotStrategy_{period_name}_plot.png")
            engine.plot_results(save_path=plot_path)
            print(f"图表已保存到 {plot_path}")

if __name__ == "__main__":
    main() 