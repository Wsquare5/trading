import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy

class SlingshotStrategy(BaseStrategy):
    """
    CM SlingShot Ultimate v3 策略
    结合EMA、RSI、DMI、ATR、布林带、VWAP和斐波那契回撤位的综合策略
    """
    def __init__(self, 
                 ema_slow_length=50, 
                 ema_fast_length=20, 
                 rsi_length=14, 
                 dmi_length=14, 
                 atr_length=14,
                 fib_lookback=50, 
                 bb_length=20, 
                 bb_dev=2.0, 
                 volatility_threshold=0.005,
                 sl_multiplier=2.0, 
                 tp_multiplier=3.0, 
                 trail_sl=True):
        """
        初始化SlingshotStrategy
        
        参数:
        ema_slow_length: 慢EMA周期
        ema_fast_length: 快EMA周期
        rsi_length: RSI周期
        dmi_length: DMI周期
        atr_length: ATR周期
        fib_lookback: 斐波那契回溯周期
        bb_length: 布林带周期
        bb_dev: 布林带标准差
        volatility_threshold: 波动率阈值(%)
        sl_multiplier: 止损ATR倍数
        tp_multiplier: 止盈ATR倍数
        trail_sl: 是否启用追踪止损
        """
        super().__init__(name="CM SlingShot Ultimate v3")
        self.ema_slow_length = ema_slow_length
        self.ema_fast_length = ema_fast_length
        self.rsi_length = rsi_length
        self.dmi_length = dmi_length
        self.atr_length = atr_length
        self.fib_lookback = fib_lookback
        self.bb_length = bb_length
        self.bb_dev = bb_dev
        self.volatility_threshold = volatility_threshold
        self.sl_multiplier = sl_multiplier
        self.tp_multiplier = tp_multiplier
        self.trail_sl = trail_sl
    
    # 自定义指标计算函数
    def calculate_ema(self, series, length):
        """计算EMA"""
        return series.ewm(span=length, adjust=False).mean()
    
    def calculate_rsi(self, series, length):
        """计算RSI"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=length).mean()
        avg_loss = loss.rolling(window=length).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, high, low, close, length):
        """计算ATR"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=length).mean()
        return atr
    
    def calculate_bbands(self, series, length, dev):
        """计算布林带"""
        middle = series.rolling(window=length).mean()
        std = series.rolling(window=length).std()
        upper = middle + dev * std
        lower = middle - dev * std
        return upper, middle, lower
    
    def calculate_dmi(self, high, low, close, length):
        """计算DMI"""
        # 计算方向变动
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # 计算真实波幅
        tr = self.calculate_atr(high, low, close, 1)
        
        # 计算方向指标
        plus_di = 100 * pd.Series(plus_dm).rolling(window=length).mean() / tr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=length).mean() / tr
        
        # 计算方向变动指数
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=length).mean()
        
        return plus_di, minus_di, adx
    
    def generate_signals(self, data):
        """
        生成交易信号
        
        参数:
        data: DataFrame，包含市场数据
        
        返回:
        DataFrame，包含交易信号
        """
        # 复制数据，避免修改原始数据
        df = data.copy()
        
        # 确保数据包含必要的列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"数据缺少必要的列: {col}")
        
        # ===== 指标计算 =====
        # 移动平均线
        df['EMA_slow'] = self.calculate_ema(df['close'], self.ema_slow_length)
        df['EMA_fast'] = self.calculate_ema(df['close'], self.ema_fast_length)
        
        # 布林带波动率过滤
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = self.calculate_bbands(
            df['close'], self.bb_length, self.bb_dev)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']  # 标准化波动率
        
        # VWAP 计算 (简化版，实际VWAP应该是日内计算)
        df['VWAP'] = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()
        
        # 斐波那契回撤位
        df['swing_high'] = df['high'].rolling(window=self.fib_lookback).max()
        df['swing_low'] = df['low'].rolling(window=self.fib_lookback).min()
        df['price_range'] = df['swing_high'] - df['swing_low']
        df['fib_382'] = df['swing_high'] - df['price_range'] * 0.382
        df['fib_500'] = df['swing_high'] - df['price_range'] * 0.5
        df['fib_618'] = df['swing_high'] - df['price_range'] * 0.618
        
        # 动量指标
        df['RSI'] = self.calculate_rsi(df['close'], self.rsi_length)
        
        # DMI计算
        df['plus_di'], df['minus_di'], df['adx'] = self.calculate_dmi(
            df['high'], df['low'], df['close'], self.dmi_length)
        
        # ATR计算
        df['ATR'] = self.calculate_atr(df['high'], df['low'], df['close'], self.atr_length)
        
        # ===== 信号生成 =====
        # 趋势确认条件
        df['uptrend'] = (df['EMA_fast'] > df['EMA_slow']) & (df['close'] > df['VWAP'])
        df['downtrend'] = (df['EMA_fast'] < df['EMA_slow']) & (df['close'] < df['VWAP'])
        
        # 斐波那契区域过滤
        df['near_fib_382'] = (df['close'] - df['fib_382']).abs() < df['ATR'] * 1.0
        df['near_fib_500'] = (df['close'] - df['fib_500']).abs() < df['ATR'] * 1.0
        df['near_fib_618'] = (df['close'] - df['fib_618']).abs() < df['ATR'] * 1.0
        df['near_fib_level'] = df['near_fib_382'] | df['near_fib_500'] | df['near_fib_618']
        
        # 成交量过滤
        df['volume_sma'] = df['volume'].rolling(window=10).mean()
        df['volume_filter'] = df['volume'] > df['volume_sma'] * 1.05
        
        # 增强入场条件
        df['long_condition'] = (
            df['uptrend'] & 
            (df['RSI'] < 50) & 
            # (df['plus_di'] > df['minus_di']) & 
            (df['BB_width'] > self.volatility_threshold * 0.5) & 
            (df['close'] > df['BB_middle']) & 
            df['volume_filter']
        )
        
        df['short_condition'] = (
            df['downtrend'] & 
            (df['RSI'] > 50) & 
            # (df['minus_di'] > df['plus_di']) & 
            (df['BB_width'] > self.volatility_threshold * 0.5) & 
            (df['close'] < df['BB_middle']) & 
            df['volume_filter']
        )
        
        # 初始化信号列
        df['signal'] = 0
        
        # 生成信号
        df.loc[df['long_condition'], 'signal'] = 1
        df.loc[df['short_condition'], 'signal'] = -1
        
        # ===== 风险管理 =====
        # 初始化列
        df['entry_price'] = np.nan
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        df['trailing_level'] = np.nan
        df['position_active'] = False
        df['is_long'] = False
        
        # 模拟交易过程
        position_active = False
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        trailing_level = 0
        is_long = False
        
        for i in range(len(df)):
            # 获取当前行
            current = df.iloc[i]
            
            # 如果没有持仓且有信号，则入场
            if not position_active and (current['long_condition'] or current['short_condition']):
                position_active = True
                entry_price = current['close']
                is_long = current['long_condition']
                
                # 设置止损和止盈
                if is_long:
                    trailing_level = entry_price - current['ATR'] * self.sl_multiplier
                    stop_loss = trailing_level
                    take_profit = entry_price + current['ATR'] * self.tp_multiplier
                else:
                    trailing_level = entry_price + current['ATR'] * self.sl_multiplier
                    stop_loss = trailing_level
                    take_profit = entry_price - current['ATR'] * self.tp_multiplier
                
                # 更新DataFrame
                df.loc[df.index[i], 'entry_price'] = entry_price
                df.loc[df.index[i], 'stop_loss'] = stop_loss
                df.loc[df.index[i], 'take_profit'] = take_profit
                df.loc[df.index[i], 'trailing_level'] = trailing_level
                df.loc[df.index[i], 'position_active'] = position_active
                df.loc[df.index[i], 'is_long'] = is_long
            
            # 如果有持仓，更新追踪止损
            elif position_active:
                # 更新DataFrame
                df.loc[df.index[i], 'entry_price'] = entry_price
                df.loc[df.index[i], 'position_active'] = position_active
                df.loc[df.index[i], 'is_long'] = is_long
                
                # 追踪止损逻辑
                if self.trail_sl:
                    if is_long:
                        new_trailing = max(trailing_level, current['close'] - current['ATR'] * 1.5)
                        if new_trailing > trailing_level:
                            trailing_level = new_trailing
                            stop_loss = trailing_level
                    else:
                        new_trailing = min(trailing_level, current['close'] + current['ATR'] * 1.5)
                        if new_trailing < trailing_level:
                            trailing_level = new_trailing
                            stop_loss = trailing_level
                
                df.loc[df.index[i], 'stop_loss'] = stop_loss
                df.loc[df.index[i], 'take_profit'] = take_profit
                df.loc[df.index[i], 'trailing_level'] = trailing_level
                
                # 平仓逻辑
                close_price = current['close']
                
                # 止损或止盈触发
                if (is_long and close_price < stop_loss) or (not is_long and close_price > stop_loss) or \
                   (is_long and close_price > take_profit) or (not is_long and close_price < take_profit):
                    # 生成平仓信号
                    df.loc[df.index[i], 'signal'] = -1 if is_long else 1
                    
                    # 重置持仓状态
                    position_active = False
                    entry_price = 0
                    stop_loss = 0
                    take_profit = 0
                    trailing_level = 0
        
        # 计算持仓
        df = self.calculate_positions(df)
        
        return df 