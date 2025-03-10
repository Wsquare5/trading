from dotenv import load_dotenv
import os

load_dotenv()

# 测试网配置
USE_TESTNET = True
TESTNET_API_KEY = os.getenv('BINANCE_TESTNET_API_KEY')
TESTNET_API_SECRET = os.getenv('BINANCE_TESTNET_API_SECRET')

# 交易对和时间框架配置
SYMBOL = 'BTCUSDT'
TIMEFRAME = '1h'

# 回测配置
START_DATE = '2023-01-01'
END_DATE = '2023-12-31'

# 策略参数
INITIAL_BALANCE = 10000  # 初始资金（USDT） 