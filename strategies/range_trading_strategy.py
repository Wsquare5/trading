import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy

class RangeTradingStrategy(BaseStrategy):
    """
    区间交易策略
    使用支撑/阻力位和OBV指标进行交易
    """
    def __init__(self, 
                 lookback_period=50, 
                 obv_length=14, 
                 atr_length=14,
                 sl_multiplier=1.0,  # 降低止损倍数
                 tp_multiplier=2.0,  # 降低止盈倍数
                 max_position_size=0.1):  # 添加仓位大小控制
        """
        初始化区间交易策略
        
        参数:
        lookback_period: 支撑/阻力回溯周期
        obv_length: OBV均线周期
        atr_length: ATR周期
        sl_multiplier: 止损ATR倍数
        tp_multiplier: 止盈ATR倍数
        max_position_size: 最大仓位大小（占总资金比例）
        """
        super().__init__(name="Range Trading Strategy")
        self.lookback_period = lookback_period
        self.obv_length = obv_length
        self.atr_length = atr_length
        self.sl_multiplier = sl_multiplier
        self.tp_multiplier = tp_multiplier
        self.max_position_size = max_position_size
    
    def calculate_obv(self, close, volume):
        """计算OBV (On-Balance Volume)"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def calculate_atr(self, high, low, close, length):
        """计算ATR (Average True Range)"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=length).mean()
        return atr
    
    def generate_signals(self, data):
        """
        生成交易信号
        
        参数:
        data: 包含OHLCV数据的DataFrame
        
        返回:
        带有信号的DataFrame
        """
        # 复制数据，避免修改原始数据
        df = data.copy()
        
        # 确保数据包含必要的列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"数据缺少必要的列: {col}")
        
        # 计算支撑和阻力
        df['support'] = df['low'].rolling(window=self.lookback_period).min()
        df['resistance'] = df['high'].rolling(window=self.lookback_period).max()
        
        # 计算OBV和OBV均线
        df['OBV'] = self.calculate_obv(df['close'], df['volume'])
        df['OBV_SMA'] = df['OBV'].rolling(window=self.obv_length).mean()
        
        # 计算ATR
        df['ATR'] = self.calculate_atr(df['high'], df['low'], df['close'], self.atr_length)
        
        # 添加趋势过滤器
        df['price_sma'] = df['close'].rolling(window=50).mean()  # 50周期均线
        df['uptrend'] = df['close'] > df['price_sma']
        df['downtrend'] = df['close'] < df['price_sma']
        
        # 计算OBV交叉信号
        df['OBV_cross_over'] = (df['OBV'] > df['OBV_SMA']) & (df['OBV'].shift() <= df['OBV_SMA'].shift())
        df['OBV_cross_under'] = (df['OBV'] < df['OBV_SMA']) & (df['OBV'].shift() >= df['OBV_SMA'].shift())
        
        # 进场条件 - 添加趋势过滤
        df['long_condition'] = (df['close'] > df['support']) & df['OBV_cross_over'] & df['uptrend']
        df['short_condition'] = (df['close'] < df['resistance']) & df['OBV_cross_under'] & df['downtrend']
        
        # 初始化信号列
        df['signal'] = 0
        df['position_size'] = self.max_position_size  # 设置固定仓位大小
        
        # 初始化持仓状态列
        df['long_active'] = False
        df['short_active'] = False
        df['long_entry_price'] = np.nan
        df['short_entry_price'] = np.nan
        df['long_stop_loss'] = np.nan
        df['long_take_profit'] = np.nan
        df['short_stop_loss'] = np.nan
        df['short_take_profit'] = np.nan
        
        # 模拟交易过程
        long_active = False
        short_active = False
        long_entry_price = 0
        short_entry_price = 0
        long_stop_loss = 0
        long_take_profit = 0
        short_stop_loss = 0
        short_take_profit = 0
        
        # 添加连续亏损计数器和风险控制
        consecutive_losses = 0
        max_consecutive_losses = 3  # 最大连续亏损次数
        
        for i in range(len(df)):
            current = df.iloc[i]
            
            # 更新当前状态
            df.loc[df.index[i], 'long_active'] = long_active
            df.loc[df.index[i], 'short_active'] = short_active
            
            if long_active:
                df.loc[df.index[i], 'long_entry_price'] = long_entry_price
                df.loc[df.index[i], 'long_stop_loss'] = long_stop_loss
                df.loc[df.index[i], 'long_take_profit'] = long_take_profit
            
            if short_active:
                df.loc[df.index[i], 'short_entry_price'] = short_entry_price
                df.loc[df.index[i], 'short_stop_loss'] = short_stop_loss
                df.loc[df.index[i], 'short_take_profit'] = short_take_profit
            
            # 检查出场条件
            if long_active:
                # 检查是否有相反方向的入场信号
                if current['short_condition']:
                    # 相反方向入场信号作为出场信号
                    df.loc[df.index[i], 'signal'] = -1  # 平多仓
                    
                    # 判断是盈利还是亏损
                    if current['close'] > long_entry_price:
                        consecutive_losses = 0  # 盈利，重置连续亏损计数
                    else:
                        consecutive_losses += 1  # 亏损，增加连续亏损计数
                    
                    long_active = False
                    # 不执行这个空头信号
                    continue
                # 检查止损和止盈
                elif current['close'] <= long_stop_loss:
                    df.loc[df.index[i], 'signal'] = -1  # 平多仓 (止损)
                    consecutive_losses += 1  # 止损，增加连续亏损计数
                    long_active = False
                elif current['close'] >= long_take_profit:
                    df.loc[df.index[i], 'signal'] = -1  # 平多仓 (止盈)
                    consecutive_losses = 0  # 止盈，重置连续亏损计数
                    long_active = False
            
            if short_active:
                # 检查是否有相反方向的入场信号
                if current['long_condition']:
                    # 相反方向入场信号作为出场信号
                    df.loc[df.index[i], 'signal'] = 1  # 平空仓
                    
                    # 判断是盈利还是亏损
                    if current['close'] < short_entry_price:
                        consecutive_losses = 0  # 盈利，重置连续亏损计数
                    else:
                        consecutive_losses += 1  # 亏损，增加连续亏损计数
                    
                    short_active = False
                    # 不执行这个多头信号
                    continue
                # 检查止损和止盈
                elif current['close'] >= short_stop_loss:
                    df.loc[df.index[i], 'signal'] = 1  # 平空仓 (止损)
                    consecutive_losses += 1  # 止损，增加连续亏损计数
                    short_active = False
                elif current['close'] <= short_take_profit:
                    df.loc[df.index[i], 'signal'] = 1  # 平空仓 (止盈)
                    consecutive_losses = 0  # 止盈，重置连续亏损计数
                    short_active = False
            
            # 检查入场条件 - 添加连续亏损限制
            if not long_active and not short_active and consecutive_losses < max_consecutive_losses:
                # 根据连续亏损调整仓位大小
                position_size = self.max_position_size * (1 - consecutive_losses * 0.2)  # 每次亏损减少20%仓位
                df.loc[df.index[i], 'position_size'] = position_size
                
                if current['long_condition']:
                    df.loc[df.index[i], 'signal'] = 1  # 开多仓
                    long_active = True
                    long_entry_price = current['close']
                    long_stop_loss = long_entry_price - current['ATR'] * self.sl_multiplier
                    long_take_profit = long_entry_price + current['ATR'] * self.tp_multiplier
                elif current['short_condition']:
                    df.loc[df.index[i], 'signal'] = -1  # 开空仓
                    short_active = True
                    short_entry_price = current['close']
                    short_stop_loss = short_entry_price + current['ATR'] * self.sl_multiplier
                    short_take_profit = short_entry_price - current['ATR'] * self.tp_multiplier
        
        # 计算持仓
        df = self.calculate_positions(df)
        
        return df 