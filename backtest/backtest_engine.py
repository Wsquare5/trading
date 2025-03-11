import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import importlib
import re

class BacktestEngine:
    """通用回测引擎"""
    
    def __init__(self, data, strategy, initial_capital=10000, commission=0.001):
        """
        初始化回测引擎
        
        参数:
        data: DataFrame, 包含OHLCV数据
        strategy: 策略对象
        initial_capital: 初始资金
        commission: 手续费率
        """
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.results = None
        self.trades_df = None
        self.metrics = None
    
    def run(self):
        """运行回测"""
        # 生成信号
        signals = self.strategy.generate_signals(self.data)
        
        # 创建回测结果DataFrame
        self.results = signals.copy()
        
        # 初始化列
        self.results['capital'] = self.initial_capital
        self.results['holdings'] = 0
        self.results['cash'] = self.initial_capital
        self.results['total'] = self.initial_capital
        self.results['returns'] = 0
        self.results['trade'] = 0
        self.results['trade_price'] = 0
        self.results['commission'] = 0
        
        # 当前持仓状态
        position = 0
        entry_price = 0
        
        # 交易记录
        trades = []
        
        for i in range(len(self.results)):
            # 获取当前行
            current = self.results.iloc[i]
            
            # 如果不是第一行，复制上一行的资金状态
            if i > 0:
                prev = self.results.iloc[i-1]
                self.results.loc[self.results.index[i], 'capital'] = prev['capital']
                self.results.loc[self.results.index[i], 'holdings'] = prev['holdings']
                self.results.loc[self.results.index[i], 'cash'] = prev['cash']
                self.results.loc[self.results.index[i], 'total'] = prev['total']
            
            # 处理交易信号
            signal = current['signal']
            price = current['close']
            
            # 买入信号
            if signal == 1 and position == 0:
                # 计算可买入数量
                available_cash = self.results.loc[self.results.index[i], 'cash']
                commission_cost = available_cash * self.commission
                buy_amount = (available_cash - commission_cost) / price
                
                # 更新持仓
                self.results.loc[self.results.index[i], 'holdings'] = buy_amount
                self.results.loc[self.results.index[i], 'cash'] = 0
                self.results.loc[self.results.index[i], 'commission'] = commission_cost
                self.results.loc[self.results.index[i], 'trade'] = 1
                self.results.loc[self.results.index[i], 'trade_price'] = price
                
                # 更新状态
                position = 1
                entry_price = price
                
                # 记录交易
                trades.append({
                    'type': 'buy',
                    'date': self.results.index[i],
                    'price': price,
                    'amount': buy_amount,
                    'commission': commission_cost,
                    'value': buy_amount * price
                })
            
            # 卖出信号
            elif signal == -1 and position == 1:
                # 计算卖出金额
                holdings = self.results.loc[self.results.index[i], 'holdings']
                sell_value = holdings * price
                commission_cost = sell_value * self.commission
                
                # 更新持仓
                self.results.loc[self.results.index[i], 'holdings'] = 0
                self.results.loc[self.results.index[i], 'cash'] = sell_value - commission_cost
                self.results.loc[self.results.index[i], 'commission'] += commission_cost
                self.results.loc[self.results.index[i], 'trade'] = -1
                self.results.loc[self.results.index[i], 'trade_price'] = price
                
                # 更新状态
                position = 0
                
                # 记录交易
                trades.append({
                    'type': 'sell',
                    'date': self.results.index[i],
                    'price': price,
                    'amount': holdings,
                    'commission': commission_cost,
                    'value': sell_value,
                    'profit': (price - entry_price) * holdings - commission_cost
                })
            
            # 更新总资产
            self.results.loc[self.results.index[i], 'total'] = self.results.loc[self.results.index[i], 'cash'] + \
                                                    self.results.loc[self.results.index[i], 'holdings'] * price
            
            # 计算收益率
            if i > 0:
                prev_total = self.results.iloc[i-1]['total']
                current_total = self.results.loc[self.results.index[i], 'total']
                self.results.loc[self.results.index[i], 'returns'] = (current_total / prev_total) - 1
        
        # 转换交易记录为DataFrame
        self.trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        # 计算性能指标
        self.calculate_metrics()
        
        return self.results, self.trades_df, self.metrics
    
    def calculate_metrics(self):
        """计算回测性能指标"""
        self.metrics = {}
        
        if len(self.results) == 0:
            return self.metrics
        
        # 总收益率
        initial_capital = self.results.iloc[0]['total']
        final_capital = self.results.iloc[-1]['total']
        self.metrics['total_return'] = (final_capital / initial_capital - 1) * 100
        
        # 年化收益率
        days = (self.results.index[-1] - self.results.index[0]).days
        if days > 0:
            self.metrics['annual_return'] = (((final_capital / initial_capital) ** (365 / days)) - 1) * 100
        else:
            self.metrics['annual_return'] = 0
        
        # 日收益率
        daily_returns = self.results['returns'].resample('D').sum()
        
        # 夏普比率 (假设无风险利率为0)
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            self.metrics['sharpe_ratio'] = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        else:
            self.metrics['sharpe_ratio'] = 0
        
        # 最大回撤
        cumulative_returns = (1 + self.results['returns']).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max - 1) * 100
        self.metrics['max_drawdown'] = drawdown.min()
        
        # 交易次数
        self.metrics['total_trades'] = len(self.trades_df)
        
        # 盈利交易
        if not self.trades_df.empty and 'profit' in self.trades_df.columns:
            profitable_trades = self.trades_df[self.trades_df['profit'] > 0]
            self.metrics['profitable_trades'] = len(profitable_trades)
            self.metrics['win_rate'] = len(profitable_trades) / len(self.trades_df) * 100 if len(self.trades_df) > 0 else 0
            
            # 平均盈利/亏损
            if len(profitable_trades) > 0:
                self.metrics['avg_profit'] = profitable_trades['profit'].mean()
            
            losing_trades = self.trades_df[self.trades_df['profit'] <= 0]
            if len(losing_trades) > 0:
                self.metrics['avg_loss'] = losing_trades['profit'].mean()
            
            # 盈亏比
            if len(losing_trades) > 0 and len(profitable_trades) > 0 and 'avg_loss' in self.metrics and self.metrics['avg_loss'] != 0:
                self.metrics['profit_loss_ratio'] = abs(self.metrics['avg_profit'] / self.metrics['avg_loss'])
            else:
                self.metrics['profit_loss_ratio'] = 0
        
        return self.metrics
    
    def plot_results(self, save_path=None):
        """
        绘制回测结果
        
        参数:
        save_path: 保存图表的路径，如果为None则不保存
        """
        if self.results is None:
            print("请先运行回测")
            return
        
        strategy_name = self.strategy.name
        
        plt.figure(figsize=(15, 12))
        
        # 绘制价格和信号
        ax1 = plt.subplot(3, 1, 1)
        
        # 获取价格数据的范围，设置合适的y轴范围
        price_min = self.results['close'].min()
        price_max = self.results['close'].max()
        price_range = price_max - price_min
        y_min = max(0, price_min - price_range * 0.1)  # 下限不低于0
        y_max = price_max + price_range * 0.1
        
        # 绘制价格
        ax1.plot(self.results.index, self.results['close'], label='Price', color='blue')
        
        # 绘制策略特定的指标
        if hasattr(self.strategy, 'short_window') and hasattr(self.strategy, 'long_window'):
            if 'SMA_short' in self.results.columns and 'SMA_long' in self.results.columns:
                ax1.plot(self.results.index, self.results['SMA_short'], 
                        label=f'SMA {self.strategy.short_window}', color='orange')
                ax1.plot(self.results.index, self.results['SMA_long'], 
                        label=f'SMA {self.strategy.long_window}', color='green')
        
        if 'upper_band' in self.results.columns and 'lower_band' in self.results.columns:
            ax1.plot(self.results.index, self.results['upper_band'], 
                    label='Upper Band', color='red', linestyle='--')
            ax1.plot(self.results.index, self.results['lower_band'], 
                    label='Lower Band', color='green', linestyle='--')
            if 'SMA' in self.results.columns:
                ax1.plot(self.results.index, self.results['SMA'], 
                        label='SMA', color='purple')
        
        # 标记买入点和卖出点
        buy_signals = self.results[self.results['trade'] == 1]
        sell_signals = self.results[self.results['trade'] == -1]
        
        ax1.scatter(buy_signals.index, buy_signals['trade_price'], 
                    marker='^', color='g', s=100, label='Buy')
        ax1.scatter(sell_signals.index, sell_signals['trade_price'], 
                    marker='v', color='r', s=100, label='Sell')
        
        # 设置y轴范围
        ax1.set_ylim(y_min, y_max)
        
        # 添加网格线，使图表更易读
        ax1.grid(True, alpha=0.3)
        
        ax1.set_title(f'{strategy_name} - Price and Signals')
        ax1.legend(loc='upper left')
        
        # 如果是RSI策略，添加RSI子图
        if 'RSI' in self.results.columns:
            # 在主图下方添加RSI子图
            ax_rsi = plt.subplot(3, 1, 2)
            ax_rsi.plot(self.results.index, self.results['RSI'], label='RSI', color='purple')
            ax_rsi.axhline(y=30, color='g', linestyle='--')
            ax_rsi.axhline(y=70, color='r', linestyle='--')
            ax_rsi.set_ylim(0, 100)
            ax_rsi.grid(True, alpha=0.3)
            ax_rsi.set_title('RSI Indicator')
            ax_rsi.legend(loc='upper left')
            
            # 资产价值图移到第三个位置
            ax2 = plt.subplot(3, 1, 3)
        else:
            # 绘制资产价值
            ax2 = plt.subplot(3, 1, 2)
        
        # 绘制资产价值
        ax2.plot(self.results.index, self.results['total'], label='Portfolio Value', color='blue')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Portfolio Value')
        ax2.legend(loc='upper left')
        
        # 绘制回撤
        if 'RSI' in self.results.columns:
            ax3 = plt.subplot(3, 1, 3)
        else:
            ax3 = plt.subplot(3, 1, 3)
        
        cumulative_returns = (1 + self.results['returns']).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max - 1) * 100
        
        ax3.fill_between(self.results.index, drawdown, 0, color='red', alpha=0.3)
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Drawdown (%)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def print_metrics(self):
        """打印性能指标"""
        if self.metrics is None:
            print("请先运行回测")
            return
        
        print("\n===== 策略性能指标 =====")
        print(f"策略名称: {self.strategy.name}")
        print(f"回测周期: {self.results.index[0]} 到 {self.results.index[-1]}")
        print(f"总收益率: {self.metrics['total_return']:.2f}%")
        print(f"年化收益率: {self.metrics['annual_return']:.2f}%")
        print(f"夏普比率: {self.metrics['sharpe_ratio']:.2f}")
        print(f"最大回撤: {self.metrics['max_drawdown']:.2f}%")
        print(f"总交易次数: {self.metrics['total_trades']}")
        
        if 'profitable_trades' in self.metrics:
            print(f"盈利交易次数: {self.metrics['profitable_trades']}")
            print(f"胜率: {self.metrics['win_rate']:.2f}%")
        
        if 'avg_profit' in self.metrics:
            print(f"平均盈利: {self.metrics['avg_profit']:.2f}")
        
        if 'avg_loss' in self.metrics:
            print(f"平均亏损: {self.metrics['avg_loss']:.2f}")
        
        if 'profit_loss_ratio' in self.metrics:
            print(f"盈亏比: {self.metrics['profit_loss_ratio']:.2f}")
    
    def save_results(self, results_path, trades_path):
        """
        保存回测结果
        
        参数:
        results_path: 保存回测结果的路径
        trades_path: 保存交易记录的路径
        """
        if self.results is None or self.trades_df is None:
            print("请先运行回测")
            return
        
        self.results.to_csv(results_path)
        if not self.trades_df.empty:
            self.trades_df.to_csv(trades_path)
        
        print(f"结果已保存到 {results_path} 和 {trades_path}")


def load_data(file_path):
    """加载CSV数据文件"""
    try:
        df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
        return df
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        return None


def load_strategy(strategy_name, params=None):
    """
    动态加载策略
    
    参数:
    strategy_name: 策略类名，例如 'MACrossStrategy'
    params: 策略参数字典
    
    返回:
    策略实例
    """
    try:
        # 特殊处理 "MA" 缩写
        file_name = strategy_name.replace('MACross', 'ma_cross')
        file_name = strategy_name.replace('RSI', 'rsi')
        file_name = ''.join(['_' + c.lower() if c.isupper() else c for c in file_name]).lstrip('_')
        
        # 导入模块
        module_path = f"strategies.{file_name}"
        module = importlib.import_module(module_path)
        
        # 获取策略类 (使用原始的策略类名)
        strategy_class = getattr(module, strategy_name)
        
        # 创建策略实例
        if params:
            strategy = strategy_class(**params)
        else:
            strategy = strategy_class()
        
        return strategy
    except Exception as e:
        print(f"加载策略时出错: {str(e)}")
        print(f"尝试加载的模块路径: strategies.{file_name}")  # 添加调试信息
        return None


def main():
    """主函数"""
    # 直接在脚本中设置参数，而不是通过命令行参数
    
    # 数据文件路径
    data_file = 'data/BTC_USDT_15m.csv'
    
    # 策略名称和参数
    strategy_name = 'RSIStrategy'
    strategy_params = {
        'rsi_period': 14,        # RSI计算周期
        'oversold': 30,          # 超卖阈值
        'overbought': 70,        # 超买阈值
        'stop_loss': 0.05,       # 止损比例 (2%)
        'take_profit': 0.05      # 止盈比例 (5%)
    }
    
    # 回测参数
    initial_capital = 10000
    commission = 0.001
    
    # 输出选项
    output_prefix = 'results'    # 输出文件前缀
    show_plot = True            # 显示图表
    save_plot = True            # 保存图表
    
    # 加载数据
    data = load_data(data_file)
    if data is None:
        return
    
    print(f"加载了 {len(data)} 条数据，时间范围: {data.index[0]} 到 {data.index[-1]}")
    
    # 加载策略
    strategy = load_strategy(strategy_name, strategy_params)
    if strategy is None:
        return
    
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
    results_path = os.path.join("results", f"{output_prefix}_{strategy_name}_results.csv")
    trades_path = os.path.join("results", f"{output_prefix}_{strategy_name}_trades.csv")
    engine.save_results(results_path, trades_path)
    
    # 绘制结果
    if show_plot:
        engine.plot_results()
    
    if save_plot:
        plot_path = f"{output_prefix}_{strategy_name}_plot.png"
        engine.plot_results(save_path=plot_path)
        print(f"图表已保存到 {plot_path}")


if __name__ == "__main__":
    main() 