from binance.client import Client
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

def test_testnet_connection():
    try:
        # 初始化测试网客户端
        client = Client(
            os.getenv('BINANCE_TESTNET_API_KEY'),
            os.getenv('BINANCE_TESTNET_API_SECRET'),
            testnet=True  # 启用测试网
        )
        
        # 测试连接
        print("=== 测试网连接信息 ===")
        
        # 获取账户信息
        account = client.get_account()
        print("\n账户余额:")
        for asset in account['balances']:
            if float(asset['free']) > 0 or float(asset['locked']) > 0:
                print(f"{asset['asset']}: 可用 {asset['free']}, 锁定 {asset['locked']}")
        
        # 获取BTC当前价格
        btc_price = client.get_symbol_ticker(symbol="BTCUSDT")
        print(f"\nBTC当前价格: {btc_price['price']}")
        
        # 测试下单（市价单）
        test_order = client.create_test_order(
            symbol='BTCUSDT',
            side=Client.SIDE_BUY,
            type=Client.ORDER_TYPE_MARKET,
            quantity=0.001
        )
        print("\n测试下单成功！")
        
        return True
        
    except Exception as e:
        print(f"测试网连接失败: {str(e)}")
        return False

if __name__ == "__main__":
    test_testnet_connection() 