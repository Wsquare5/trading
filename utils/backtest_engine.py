import pandas as pd
import numpy as np
from datetime import datetime

class BacktestEngine:
    def __init__(self, data, initial_balance=10000):
        self.data = data
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.position = 0
        self.trades = []
        self.trade_history = pd.DataFrame()
        
    def buy(self, price, amount, timestamp):
        """执行买入操作"""
        if self.current_balance >= amount * price:
            cost = amount * price
            self.current_balance -= cost
            self.position += amount
            self.trades.append({
                'timestamp': timestamp,
                'type': 'BUY',
                'price': price,
                'amount': amount,
                'cost': cost,
                'balance': self.current_balance
            })
            return True
        return False
    
    def sell(self, price, amount, timestamp):
        """执行卖出操作"""
        if self.position >= amount:
            revenue = amount * price
            self.current_balance += revenue
            self.position -= amount
            self.trades.append({
                'timestamp': timestamp,
                'type': 'SELL',
                'price': price,
                'amount': amount,
                'revenue': revenue,
                'balance': self.current_balance
            })
            return True
        return False
    
    def calculate_metrics(self):
        """计算回测指标"""
        if not self.trades:
            return {}
            
        trade_df = pd.DataFrame(self.trades)
        
        # 计算收益率
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance
        
        # 计算最大回撤
        balance_series = trade_df['balance']
        rolling_max = balance_series.expanding().max()
        drawdowns = (rolling_max - balance_series) / rolling_max
        max_drawdown = drawdowns.max()
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'final_balance': self.current_balance
        } 