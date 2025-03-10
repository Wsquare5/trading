import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# 假设你的策略类在 strategies 文件夹中
from strategies.ma_cross_strategy import MACrossStrategy

def load_data(file_path):
    """加载CSV数据文件"""
    df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
    return df

def backtest_strategy(data, strategy, initial_capital=10000, commission=0.001):
    """
    回测策略
    
    参数:
    data: DataFrame, 包含OHLCV数据
    strategy: 策略对象
    initial_capital: 初始资金
    commission: 手续费率
    
    返回:
    DataFrame, 包含回测结果
    """
    # 生成信号
    signals = strategy.generate_signals(data)
    
    # 创建回测结果DataFrame
    results = signals.copy()
    
    # 初始化列
    results['capital'] = initial_capital
    results['holdings'] = 0
    results['cash'] = initial_capital
    results['total'] = initial_capital
    results['returns'] = 0
    results['trade'] = 0
    results['trade_price'] = 0
    results['commission'] = 0
    
    # 当前持仓状态
    position = 0
    entry_price = 0
    
    # 交易记录
    trades = []
    
    for i in range(len(results)):
        # 获取当前行
        current = results.iloc[i]
        
        # 如果不是第一行，复制上一行的资金状态
        if i > 0:
            prev = results.iloc[i-1]
            results.loc[results.index[i], 'capital'] = prev['capital']
            results.loc[results.index[i], 'holdings'] = prev['holdings']
            results.loc[results.index[i], 'cash'] = prev['cash']
            results.loc[results.index[i], 'total'] = prev['total']
        
        # 处理交易信号
        signal = current['signal']
        price = current['close']
        
        # 买入信号
        if signal == 1 and position == 0:
            # 计算可买入数量
            available_cash = results.loc[results.index[i], 'cash']
            commission_cost = available_cash * commission
            buy_amount = (available_cash - commission_cost) / price
            
            # 更新持仓
            results.loc[results.index[i], 'holdings'] = buy_amount
            results.loc[results.index[i], 'cash'] = 0
            results.loc[results.index[i], 'commission'] = commission_cost
            results.loc[results.index[i], 'trade'] = 1
            results.loc[results.index[i], 'trade_price'] = price
            
            # 更新状态
            position = 1
            entry_price = price
            
            # 记录交易
            trades.append({
                'type': 'buy',
                'date': results.index[i],
                'price': price,
                'amount': buy_amount,
                'commission': commission_cost,
                'value': buy_amount * price
            })
        
        # 卖出信号
        elif signal == -1 and position == 1:
            # 计算卖出金额
            holdings = results.loc[results.index[i], 'holdings']
            sell_value = holdings * price
            commission_cost = sell_value * commission
            
            # 更新持仓
            results.loc[results.index[i], 'holdings'] = 0
            results.loc[results.index[i], 'cash'] = sell_value - commission_cost
            results.loc[results.index[i], 'commission'] += commission_cost
            results.loc[results.index[i], 'trade'] = -1
            results.loc[results.index[i], 'trade_price'] = price
            
            # 更新状态
            position = 0
            
            # 记录交易
            trades.append({
                'type': 'sell',
                'date': results.index[i],
                'price': price,
                'amount': holdings,
                'commission': commission_cost,
                'value': sell_value,
                'profit': (price - entry_price) * holdings - commission_cost
            })
        
        # 更新总资产
        results.loc[results.index[i], 'total'] = results.loc[results.index[i], 'cash'] + \
                                                results.loc[results.index[i], 'holdings'] * price
        
        # 计算收益率
        if i > 0:
            prev_total = results.iloc[i-1]['total']
            current_total = results.loc[results.index[i], 'total']
            results.loc[results.index[i], 'returns'] = (current_total / prev_total) - 1
    
    # 转换交易记录为DataFrame
    trades_df = pd.DataFrame(trades)
    
    return results, trades_df

def calculate_metrics(results, trades_df):
    """
    计算回测性能指标
    
    参数:
    results: DataFrame, 回测结果
    trades_df: DataFrame, 交易记录
    
    返回:
    dict, 性能指标
    """
    metrics = {}
    
    # 总收益率
    initial_capital = results.iloc[0]['total']
    final_capital = results.iloc[-1]['total']
    metrics['total_return'] = (final_capital / initial_capital - 1) * 100
    
    # 年化收益率
    days = (results.index[-1] - results.index[0]).days
    metrics['annual_return'] = (((final_capital / initial_capital) ** (365 / days)) - 1) * 100
    
    # 日收益率
    daily_returns = results['returns'].resample('D').sum()
    
    # 夏普比率 (假设无风险利率为0)
    metrics['sharpe_ratio'] = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    
    # 最大回撤
    cumulative_returns = (1 + results['returns']).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max - 1) * 100
    metrics['max_drawdown'] = drawdown.min()
    
    # 交易次数
    metrics['total_trades'] = len(trades_df)
    
    # 盈利交易
    if 'profit' in trades_df.columns and len(trades_df) > 0:
        profitable_trades = trades_df[trades_df['profit'] > 0]
        metrics['profitable_trades'] = len(profitable_trades)
        metrics['win_rate'] = len(profitable_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        
        # 平均盈利/亏损
        if len(profitable_trades) > 0:
            metrics['avg_profit'] = profitable_trades['profit'].mean()
        
        losing_trades = trades_df[trades_df['profit'] <= 0]
        if len(losing_trades) > 0:
            metrics['avg_loss'] = losing_trades['profit'].mean()
        
        # 盈亏比
        if len(losing_trades) > 0 and len(profitable_trades) > 0 and metrics['avg_loss'] != 0:
            metrics['profit_loss_ratio'] = abs(metrics['avg_profit'] / metrics['avg_loss'])
        else:
            metrics['profit_loss_ratio'] = 0
    
    return metrics

def plot_results(results, trades_df, strategy_name):
    """
    绘制回测结果
    
    参数:
    results: DataFrame, 回测结果
    trades_df: DataFrame, 交易记录
    strategy_name: str, 策略名称
    """
    plt.figure(figsize=(15, 12))
    
    # 绘制价格和移动平均线
    plt.subplot(3, 1, 1)
    plt.plot(results.index, results['close'], label='Price')
    if 'SMA_short' in results.columns and 'SMA_long' in results.columns:
        plt.plot(results.index, results['SMA_short'], label='Short MA')
        plt.plot(results.index, results['SMA_long'], label='Long MA')
    
    # 标记买入点和卖出点
    buy_signals = results[results['trade'] == 1]
    sell_signals = results[results['trade'] == -1]
    
    plt.scatter(buy_signals.index, buy_signals['trade_price'], 
                marker='^', color='g', s=100, label='Buy')
    plt.scatter(sell_signals.index, sell_signals['trade_price'], 
                marker='v', color='r', s=100, label='Sell')
    
    plt.title(f'{strategy_name} - Price and Signals')
    plt.legend()
    
    # 绘制资产价值
    plt.subplot(3, 1, 2)
    plt.plot(results.index, results['total'], label='Portfolio Value')
    plt.title('Portfolio Value')
    plt.legend()
    
    # 绘制回撤
    plt.subplot(3, 1, 3)
    cumulative_returns = (1 + results['returns']).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max - 1) * 100
    
    plt.fill_between(results.index, drawdown, 0, color='red', alpha=0.3)
    plt.title('Drawdown (%)')
    
    plt.tight_layout()
    plt.show()

def main():
    # 加载数据
    file_path = 'data/BTC_USDT_15m.csv'  # 替换为你的数据文件路径
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return
    
    data = load_data(file_path)
    print(f"加载了 {len(data)} 条数据，时间范围: {data.index[0]} 到 {data.index[-1]}")
    
    # 创建策略实例
    strategy = MACrossStrategy(
        short_window=10,  # 短期均线周期
        long_window=30,   # 长期均线周期
        stop_loss=0.02,   # 止损比例
        take_profit=0.05  # 止盈比例
    )
    
    # 回测策略
    results, trades_df = backtest_strategy(
        data=data,
        strategy=strategy,
        initial_capital=10000,  # 初始资金
        commission=0.001        # 手续费率 (0.1%)
    )
    
    # 计算性能指标
    metrics = calculate_metrics(results, trades_df)
    
    # 打印性能指标
    print("\n===== 策略性能指标 =====")
    print(f"总收益率: {metrics['total_return']:.2f}%")
    print(f"年化收益率: {metrics['annual_return']:.2f}%")
    print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
    print(f"最大回撤: {metrics['max_drawdown']:.2f}%")
    print(f"总交易次数: {metrics['total_trades']}")
    
    if 'profitable_trades' in metrics:
        print(f"盈利交易次数: {metrics['profitable_trades']}")
        print(f"胜率: {metrics['win_rate']:.2f}%")
    
    if 'avg_profit' in metrics:
        print(f"平均盈利: {metrics['avg_profit']:.2f}")
    
    if 'avg_loss' in metrics:
        print(f"平均亏损: {metrics['avg_loss']:.2f}")
    
    if 'profit_loss_ratio' in metrics:
        print(f"盈亏比: {metrics['profit_loss_ratio']:.2f}")
    
    # 绘制结果
    plot_results(results, trades_df, strategy.name)
    
    # 保存结果
    results.to_csv('macross_backtest_results.csv')
    trades_df.to_csv('macross_trades.csv')
    print("\n结果已保存到 macross_backtest_results.csv 和 macross_trades.csv")

if __name__ == "__main__":
    main() 