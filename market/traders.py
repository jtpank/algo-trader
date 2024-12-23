from data import utils
import os
import pandas as pd
import numpy as np
from loguru import logger as log

ALLOWED_POSITIVE_ACTIONS = ["Buy", "Sell"]
ALLOWED_NEGATIVE_ACTIONS = ["Buy to Cover", "Sell Short"]
ALLOWED_ACTIONS = ALLOWED_POSITIVE_ACTIONS + ALLOWED_NEGATIVE_ACTIONS
MARKET_DAYS = pd.read_csv('./market/info.csv')["market_dates"].to_list()
MARKET_UNIX_S = ((pd.to_datetime(MARKET_DAYS) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")).to_numpy()

def fmt(num):
    return "{:.2f}".format(num)

class SimulatedTrader(object):
    """
    Simulates market trades on historical markets
    """
    def __init__(self, current_day="2023-01-01", trade_on="Open"):
        self.current_day = current_day
        self.positions = dict()
        self.trade_on = trade_on
        data_folder = os.path.join(".", "data", "historical")
        self.dataFetcher = utils.DataFetcher(data_folder, interval="1d", start=current_day, end=None)
    
    def trade(self, symbol, action, quantity):
        assert action in ALLOWED_ACTIONS
        if symbol not in self.positions:
            self.positions[symbol] = 0

        curr_pos = self.positions[symbol]
        if curr_pos < 0:
            assert action in ALLOWED_NEGATIVE_ACTIONS
        elif curr_pos > 0:
            assert action in ALLOWED_POSITIVE_ACTIONS
        
        price = self.get_price(symbol)
        cost = price*quantity

        if "Buy" in action:
            self.positions[symbol] += quantity
        elif "Sell" in action:
            self.positions[symbol] -= quantity
        
        log.info(f"{action} {fmt(quantity)} shares of {symbol} performed at a cost of {fmt(cost)}")
        log.info(f"Position now at {fmt(self.positions[symbol])} shares on {self.current_day}")
        return cost
    
    def go_next_trading_day(self):
        unix_off = pd.to_datetime([self.current_day]) - pd.Timestamp("1970-01-01")
        unix_off_s = (unix_off // pd.Timedelta("1s")).to_numpy()
        [ind] = np.searchsorted(MARKET_UNIX_S, unix_off_s, side='right')
        if ind >= len(MARKET_DAYS):
            log.warning(f"No market days marked for next day following {self.current_day}") 
            self.current_day = None
            return
        self.current_day = MARKET_DAYS[ind]
        # log.info(f"Went to next trading day {self.current_day}")

    def get_price(self, symbol):
        df = self.dataFetcher.get_bars(symbol)
        if df is None:
            log.error(f"Failed to get price of {symbol}: No data")
            return None
        
        if not self.current_day in df.index.to_list():
            log.warning(f"{self.current_day} not in bars of {symbol}")
            return None
        
        bar = df.loc[self.current_day]
        return bar[self.trade_on]