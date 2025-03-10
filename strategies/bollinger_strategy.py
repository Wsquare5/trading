import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy

class BollingerStrategy(BaseStrategy):
    """
    布林带策略
    当价格触及下轨时买入
    当价格触及上轨时卖出
    """
    def __init__(self, window=20, num_std=2, stop_loss=0.05, take_profit=0.1):
        super().__init__(name="Bollinger Bands Strategy")
        self.window = window
        self.num_std = num_std
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
        
        # 计算布林带
        df['SMA'] = df['close'].rolling(window=self.window).mean()
        df['STD'] = df['close'].rolling(window=self.window).std()
        df['upper_band'] = df['SMA'] + (df['STD'] * self.num_std)
        df['lower_band'] = df['SMA'] - (df['STD'] * self.num_std)
        
        # 生成信号
        df['signal'] = 0
        
        # 买入信号：价格触及下轨
        df.loc[df['close'] <= df['lower_band'], 'signal'] = 1
        
        # 卖出信号：价格触及上轨
        df.loc[df['close'] >= df['upper_band'], 'signal'] = -1
        
        # 应用止损和止盈
        if self.stop_loss > 0:
            df = self.set_stop_loss(df, self.stop_loss)
        
        if self.take_profit > 0:
            df = self.set_take_profit(df, self.take_profit)
        
        # 计算持仓
        df = self.calculate_positions(df)
        
        return df 