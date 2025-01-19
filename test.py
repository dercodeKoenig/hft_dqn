import MetaTrader5 as mt5
from datetime import datetime, timedelta
import time

from candle_manager import CandleManager
from mt5_tools import *

# Get tick data from MetaTrader 5 for a given symbol and time period
def fetch_tick_data(symbol, from_time, to_time):
    ticks = mt5.copy_ticks_range(symbol, from_time, to_time, mt5.COPY_TICKS_TRADE )
    if ticks is None or len(ticks) == 0:
        print("No tick data available for the specified range.")
        return []

    # Convert ticks to a more usable format (list of dictionaries)
    formatted_ticks = []
    for tick in ticks:
        # Convert the 'time' field from Unix timestamp to a datetime object
        tick_time = datetime.fromtimestamp(tick['time'])

        formatted_ticks.append({
            "time": tick_time,
            "bid": tick['bid']
        })
    return formatted_ticks



# Initialize MetaTrader 5
def initialize_mt5():
    if not mt5.initialize():
        print("MT5 initialization failed")
        return False
    return True



initialize_mt5()
# Set time period for backtesting (last 10 minutes as an example)
to_time = datetime.now()
from_time = to_time - timedelta(minutes=10000)

# Fetch tick data from MetaTrader 5
ticks = fetch_tick_data("@ENQ" , from_time, to_time)


    # Initialize the CandleManager
manager = CandleManager()
manager.set_tick_data(ticks)
candle = manager.get_next_15_candle()
print(candle.o,candle.h,candle.l,candle.c,candle.t)
candle = manager.get_next_15_candle()
print(candle.o,candle.h,candle.l,candle.c,candle.t)