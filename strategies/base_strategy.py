import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """
    策略基类，所有具体策略都应该继承这个类
    """
    def __init__(self, name="BaseStrategy"):
        self.name = name
        self.positions = None
        self.signals = None
    
    @abstractmethod
    def generate_signals(self, data):
        """
        生成交易信号
        :param data: DataFrame，包含市场数据
        :return: DataFrame，包含交易信号
        """
        pass
    
    def set_stop_loss(self, data, stop_loss_pct):
        """
        设置止损
        :param data: DataFrame，包含市场数据和信号
        :param stop_loss_pct: 止损百分比
        :return: DataFrame，更新后的信号
        """
        # 复制数据，避免修改原始数据
        result = data.copy()
        
        # 初始化止损价格列
        result['stop_loss'] = np.nan
        
        # 当前持仓状态
        position = 0
        entry_price = 0
        
        for i in range(len(result)):
            # 获取当前信号
            current_signal = result['signal'].iloc[i]
            current_price = result['close'].iloc[i]
            
            # 如果有新的入场信号
            if current_signal == 1 and position == 0:  # 买入信号
                position = 1
                entry_price = current_price
                # 设置止损价格
                result.loc[result.index[i], 'stop_loss'] = entry_price * (1 - stop_loss_pct)
            
            elif current_signal == -1 and position == 0:  # 卖出信号（做空）
                position = -1
                entry_price = current_price
                # 设置止损价格
                result.loc[result.index[i], 'stop_loss'] = entry_price * (1 + stop_loss_pct)
            
            # 如果已经有持仓，检查是否触发止损
            elif position != 0:
                if position == 1:  # 多头持仓
                    # 更新止损价格
                    result.loc[result.index[i], 'stop_loss'] = entry_price * (1 - stop_loss_pct)
                    # 检查是否触发止损
                    if current_price <= result['stop_loss'].iloc[i]:
                        result.loc[result.index[i], 'signal'] = -1  # 触发止损，平仓
                        position = 0
                
                elif position == -1:  # 空头持仓
                    # 更新止损价格
                    result.loc[result.index[i], 'stop_loss'] = entry_price * (1 + stop_loss_pct)
                    # 检查是否触发止损
                    if current_price >= result['stop_loss'].iloc[i]:
                        result.loc[result.index[i], 'signal'] = 1  # 触发止损，平仓
                        position = 0
                
                # 如果有平仓信号
                if (position == 1 and current_signal == -1) or (position == -1 and current_signal == 1):
                    position = 0
        
        return result
    
    def set_take_profit(self, data, take_profit_pct):
        """
        设置止盈
        :param data: DataFrame，包含市场数据和信号
        :param take_profit_pct: 止盈百分比
        :return: DataFrame，更新后的信号
        """
        # 复制数据，避免修改原始数据
        result = data.copy()
        
        # 初始化止盈价格列
        result['take_profit'] = np.nan
        
        # 当前持仓状态
        position = 0
        entry_price = 0
        
        for i in range(len(result)):
            # 获取当前信号
            current_signal = result['signal'].iloc[i]
            current_price = result['close'].iloc[i]
            
            # 如果有新的入场信号
            if current_signal == 1 and position == 0:  # 买入信号
                position = 1
                entry_price = current_price
                # 设置止盈价格
                result.loc[result.index[i], 'take_profit'] = entry_price * (1 + take_profit_pct)
            
            elif current_signal == -1 and position == 0:  # 卖出信号（做空）
                position = -1
                entry_price = current_price
                # 设置止盈价格
                result.loc[result.index[i], 'take_profit'] = entry_price * (1 - take_profit_pct)
            
            # 如果已经有持仓，检查是否触发止盈
            elif position != 0:
                if position == 1:  # 多头持仓
                    # 更新止盈价格
                    result.loc[result.index[i], 'take_profit'] = entry_price * (1 + take_profit_pct)
                    # 检查是否触发止盈
                    if current_price >= result['take_profit'].iloc[i]:
                        result.loc[result.index[i], 'signal'] = -1  # 触发止盈，平仓
                        position = 0
                
                elif position == -1:  # 空头持仓
                    # 更新止盈价格
                    result.loc[result.index[i], 'take_profit'] = entry_price * (1 - take_profit_pct)
                    # 检查是否触发止盈
                    if current_price <= result['take_profit'].iloc[i]:
                        result.loc[result.index[i], 'signal'] = 1  # 触发止盈，平仓
                        position = 0
                
                # 如果有平仓信号
                if (position == 1 and current_signal == -1) or (position == -1 and current_signal == 1):
                    position = 0
        
        return result
    
    def calculate_positions(self, data):
        """
        根据信号计算持仓
        :param data: DataFrame，包含市场数据和信号
        :return: DataFrame，包含持仓信息
        """
        # 复制数据，避免修改原始数据
        result = data.copy()
        
        # 初始化持仓列
        result['position'] = 0
        
        # 当前持仓状态
        position = 0
        
        for i in range(len(result)):
            # 获取当前信号
            current_signal = result['signal'].iloc[i]
            
            # 更新持仓状态
            if current_signal == 1:  # 买入信号
                position = 1
            elif current_signal == -1:  # 卖出信号
                position = 0
            
            # 记录当前持仓
            result.loc[result.index[i], 'position'] = position
        
        return result 