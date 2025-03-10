import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import importlib

# 导入回测引擎和策略
from backtest_engine import BacktestEngine, load_data
from strategies.slingshot_strategy import SlingshotStrategy

def main():
    """主函数"""
    # 数据文件路径
    data_file = 'data/BTC_USDT_15m.csv'  # 确保这是你的数据文件路径
    
    # 创建策略实例
    strategy = SlingshotStrategy(
        ema_slow_length=25,       # 慢EMA周期
        ema_fast_length=7,       # 快EMA周期
        rsi_length=14,            # RSI周期
        dmi_length=14,            # DMI周期
        atr_length=14,            # ATR周期
        fib_lookback=50,          # 斐波那契回溯周期
        bb_length=20,             # 布林带周期
        bb_dev=2.0,               # 布林带标准差
        volatility_threshold=0.002, # 波动率阈值
        sl_multiplier=1.5,        # 止损ATR倍数
        tp_multiplier=3.5,        # 止盈ATR倍数
        trail_sl=True             # 启用追踪止损
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
    
    # 创建回测引擎
    engine = BacktestEngine(
        data=data,
        strategy=strategy,
        initial_capital=initial_capital,
        commission=commission
    )
    
    # 运行回测
    results, trades_df, metrics = engine.run()
    
    # 打印性能指标
    engine.print_metrics()
    
    # 保存结果


    results_path = os.path.join("results",f"{output_prefix}_SlingshotStrategy_results.csv")
    trades_path = os.path.join("results",f"{output_prefix}_SlingshotStrategy_trades.csv")
    engine.save_results(results_path, trades_path)
    
    # 绘制结果
    if show_plot:
        engine.plot_results()
    
    if save_plot:
        plot_path = os.path.join("results",f"{output_prefix}_SlingshotStrategy_plot.png")
        engine.plot_results(save_path=plot_path)
        print(f"图表已保存到 {plot_path}")

if __name__ == "__main__":
    main() 