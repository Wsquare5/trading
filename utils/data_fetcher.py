import ccxt
import pandas as pd
from datetime import datetime
import time

class CryptoDataFetcher:
    def __init__(self, exchange_name='binance'):
        """
        初始化数据获取器
        :param exchange_name: 交易所名称，默认使用binance
        """
        # 获取交易所实例
        self.exchange = getattr(ccxt, exchange_name)()
        
    def fetch_ohlcv(self, symbol, timeframe, start_time=None, end_time=None, limit=None):
        """
        获取加密货币的OHLCV数据
        
        :param symbol: 交易对，例如 'BTC/USDT'
        :param timeframe: K线周期，例如 '1m', '5m', '1h', '1d'
        :param start_time: 开始时间，可以是时间戳或者datetime对象
        :param end_time: 结束时间，可以是时间戳或者datetime对象
        :param limit: 需要获取的K线总数
        :return: DataFrame，包含OHLCV数据
        """
        try:
            # 检查时间周期是否有效
            if timeframe not in self.exchange.timeframes:
                raise ValueError(f"无效的时间周期: {timeframe}. 可用的时间周期: {self.exchange.timeframes}")
            
            # 转换时间格式
            if isinstance(start_time, datetime):
                start_time = int(start_time.timestamp() * 1000)
            if isinstance(end_time, datetime):
                end_time = int(end_time.timestamp() * 1000)

            all_ohlcv = []
            current_start = start_time
            
            while True:
                # 获取数据
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_start,
                    limit=1000  # binance的每次请求限制
                )
                
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                
                # 更新下一次请求的起始时间
                current_start = ohlcv[-1][0] + 1
                
                # 如果达到了结束时间或者获取的数据量达到了限制，就停止
                if (end_time and current_start >= end_time) or \
                   (limit and len(all_ohlcv) >= limit):
                    break
                    
                # 添加延时以避免触发频率限制
                time.sleep(0.2)
                
                print(f"已获取 {len(all_ohlcv)} 条数据...")

            # 如果设置了limit，截取指定数量的数据
            if limit and len(all_ohlcv) > limit:
                all_ohlcv = all_ohlcv[:limit]

            # 转换为DataFrame
            if not all_ohlcv:
                print(f"没有获取到数据，请检查参数是否正确")
                return None
                
            df = pd.DataFrame(
                all_ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # 转换时间戳为datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            print(f"总共获取了 {len(df)} 条数据")
            return df
            
        except Exception as e:
            print(f"获取数据时发生错误: {str(e)}")
            return None
    
    def get_available_symbols(self):
        """
        获取可用的交易对列表
        :return: 交易对列表
        """
        try:
            markets = self.exchange.load_markets()
            return list(markets.keys())
        except Exception as e:
            print(f"获取交易对列表时发生错误: {str(e)}")
            return []
    
    def get_available_timeframes(self):
        """
        获取可用的时间周期
        :return: 时间周期字典
        """
        return self.exchange.timeframes

if __name__ == "__main__":
    # 使用示例
    fetcher = CryptoDataFetcher()
    
    # 获取可用的交易对
    symbols = fetcher.get_available_symbols()
    print("\n可用的交易对示例（前5个）:")
    print(symbols[:5])
    
    # 获取可用的时间周期
    timeframes = fetcher.get_available_timeframes()
    print("\n可用的时间周期:")
    print(timeframes)
    
    # 获取比特币数据示例
    symbol = 'BTC/USDT'  # 确保使用正确的交易对格式
    timeframe = '15m'    # 确保使用正确的时间周期格式
    
    print(f"\n尝试获取 {symbol} 的 {timeframe} 数据...")
    
    start_time = datetime(2022, 1, 1)
    
    btc_data = fetcher.fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        start_time=start_time,
        limit=10000000
    )
    
    if btc_data is not None:
        print("\n获取的数据示例（前5行）:")
        print(btc_data.head())
        
        # 保存数据示例
        output_file = f"data/{symbol.replace('/', '_')}_{timeframe}.csv"
        btc_data.to_csv(output_file)
        print(f"\n数据已保存到 {output_file}") 