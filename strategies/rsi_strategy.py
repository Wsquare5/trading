import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy

class RSIStrategy(BaseStrategy):
    """
    RSI策略
    当RSI低于超卖水平时买入
    当RSI高于超买水平时卖出
    """
    def __init__(self, rsi_period=14, oversold=30, overbought=70, stop_loss=0.05, take_profit=0.1):
        super().__init__(name="RSI Strategy")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.stop_loss = stop_loss
        self.take_profit = take_profit
    
    def calculate_rsi(self, prices, period=14):
        """
        计算RSI指标
        :param prices: Series，价格序列
        :param period: RSI周期
        :return: Series，RSI值
        """
        # 计算价格变化
        delta = prices.diff()
        
        # 分离上涨和下跌
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 计算平均上涨和下跌
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # 计算相对强度
        rs = avg_gain / avg_loss
        
        # 计算RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, data):
        """
        生成交易信号
        :param data: DataFrame，包含市场数据
        :return: DataFrame，包含交易信号
        """
        # 复制数据，避免修改原始数据
        df = data.copy()
        
        # 计算RSI
        df['RSI'] = self.calculate_rsi(df['close'], self.rsi_period)
        
        # 生成信号
        df['signal'] = 0
        
        # 买入信号：RSI低于超卖水平
        df.loc[(df['RSI'] < self.oversold) & (df['RSI'].shift(1) >= self.oversold), 'signal'] = 1
        
        # 卖出信号：RSI高于超买水平
        df.loc[(df['RSI'] > self.overbought) & (df['RSI'].shift(1) <= self.overbought), 'signal'] = -1
        
        # 应用止损和止盈
        if self.stop_loss > 0:
            df = self.set_stop_loss(df, self.stop_loss)
        
        if self.take_profit > 0:
            df = self.set_take_profit(df, self.take_profit)
        
        # 计算持仓
        df = self.calculate_positions(df)
        
        return df 