import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy

class MACrossStrategy(BaseStrategy):
    """
    移动平均线交叉策略
    当短期均线上穿长期均线时买入
    当短期均线下穿长期均线时卖出
    """
    def __init__(self, short_window=20, long_window=50, stop_loss=0.05, take_profit=0.1):
        super().__init__(name="MA Cross Strategy")
        self.short_window = short_window
        self.long_window = long_window
        self.stop_loss = stop_loss
        self.take_profit = take_profit
    
    def generate_signals(self, data):
        """
        生成交易信号
        :param data: DataFrame，包含市场数据
        :return: DataFrame，包含交易信号
        """
        # 复制数据，避免修改原始数据
        df = data.copy()
        
        # 计算移动平均线
        df['SMA_short'] = df['close'].rolling(window=self.short_window).mean()
        df['SMA_long'] = df['close'].rolling(window=self.long_window).mean()
        
        # 生成信号
        df['signal'] = 0
        
        # 买入信号：短期均线上穿长期均线
        df.loc[(df['SMA_short'] > df['SMA_long']) & 
               (df['SMA_short'].shift(1) <= df['SMA_long'].shift(1)), 'signal'] = 1
        
        # 卖出信号：短期均线下穿长期均线
        df.loc[(df['SMA_short'] < df['SMA_long']) & 
               (df['SMA_short'].shift(1) >= df['SMA_long'].shift(1)), 'signal'] = -1
        
        # 应用止损和止盈
        if self.stop_loss > 0:
            df = self.set_stop_loss(df, self.stop_loss)
        
        if self.take_profit > 0:
            df = self.set_take_profit(df, self.take_profit)
        
        # 计算持仓
        df = self.calculate_positions(df)
        
        return df 