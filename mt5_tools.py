import time
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from Candle import Candle

def get_prices(symbol, tf, lookback):
    t = int(time.time()) + 60 * 60 * 24 * 7

    prices = mt5.copy_rates_from(symbol, tf, t, lookback)
    # print(prices)
    candles = []
    for t, o, h, l, c, _, _, _ in prices:
        t = datetime.fromtimestamp(int(t))
        x = Candle()
        x.h = h
        x.l = l
        x.o = o
        x.c = c
        x.t = t
        candles.append(x)

    return candles


# size = mt5.account_info().balance / 100000
def open_position(pair, order_type, price, size, tp, sl, implement_spread=False):
    symbol_info = mt5.symbol_info(pair)
    if symbol_info is None:
        print(pair, "not found")
        return

    if not symbol_info.visible:
        print(pair, "is not visible, trying to switch on")
        if not mt5.symbol_select(pair, True):
            print("symbol_select({}}) failed, exit", pair)
            return
    # print(pair, "found!")

    symbol = pair
    symbol_info = mt5.symbol_info(symbol)
    lot = size
    point = mt5.symbol_info(symbol).point
    deviation = 200

    if order_type == "SELL":
        otype = mt5.ORDER_TYPE_SELL_LIMIT
        if implement_spread == True:
            spread = mt5.symbol_info(pair).spread * mt5.symbol_info(pair).point
            tp += spread
    elif order_type == "BUY":
        otype = mt5.ORDER_TYPE_BUY_LIMIT
        if implement_spread == True:
            spread = mt5.symbol_info(pair).spread * mt5.symbol_info(pair).point
            price += spread
    else:
        print("ERROR")

    request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": lot,
        "type": otype,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": deviation,
        "magic": 234000,
        "comment": "dqn_sw_agent",
        "type_time": mt5.ORDER_TIME_DAY,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    # send a trading request
    # print(request)
    return mt5.order_send(request)


def delete_pending(ticket):
    close_request = {
        "action": mt5.TRADE_ACTION_REMOVE,
        "order": ticket,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(close_request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        result_dict = result._asdict()
        print(result_dict)
    # else:
    # print('Delete complete...')


def close_position(position):
    order_type = position.type
    symbol = position.symbol
    volume = position.volume
    deal_id = position.ticket

    if (order_type == mt5.ORDER_TYPE_BUY):
        order_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask

    close_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": order_type,
        "position": deal_id,
        "price": price,
        "magic": 234000,
        "comment": "Close trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(close_request)


def open_position_market(pair, order_type, size):
    symbol_info = mt5.symbol_info(pair)
    if symbol_info is None:
        print(pair, "not found")
        return

    if not symbol_info.visible:
        print(pair, "is not visible, trying to switch on")
        if not mt5.symbol_select(pair, True):
            print("symbol_select({}}) failed, exit", pair)
            return
    # print(pair, "found!")

    symbol = pair
    symbol_info = mt5.symbol_info(symbol)
    lot = size

    if order_type == "SELL":
        otype = mt5.ORDER_TYPE_SELL
    elif order_type == "BUY":
        otype = mt5.ORDER_TYPE_BUY
    else:
        print("ERROR")

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": otype,
        "magic": 234000,
        "comment": "dqn_sw_agent",
        "type_time": mt5.ORDER_TIME_DAY,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    # send a trading request
    # print(request)
    return mt5.order_send(request)