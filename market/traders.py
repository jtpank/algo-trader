from data import utils
import os
import pandas as pd
import numpy as np
from loguru import logger as log
import yfinance as yf

ALLOWED_POSITIVE_ACTIONS = ["Buy", "Sell"]
ALLOWED_NEGATIVE_ACTIONS = ["Buy to Cover", "Sell Short"]
ALLOWED_ACTIONS = ALLOWED_POSITIVE_ACTIONS + ALLOWED_NEGATIVE_ACTIONS
MARKET_DATETIMES = pd.read_csv('./market/info.csv')["market_datetimes"].to_list()
MARKET_UNIX_S = ((pd.to_datetime(MARKET_DATETIMES) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")).to_numpy()

def fmt(num):
    return "{:.2f}".format(num)

class SimulatedTrader(object):
    """
    Simulates market trades on historical markets
    """
    def __init__(self, fetcher: utils.DataFetcher, current_datetime="2023-01-03 09:30:00", trade_on="Open"):
        self.current_datetime = current_datetime
        self.positions = dict()
        self.trade_on = trade_on
        self.fetcher = fetcher
        self.interval = self.fetcher.interval
    
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
        log.info(f"Position now at {fmt(self.positions[symbol])} shares at {self.current_datetime}")
        return cost

    def go_next_trading_hour(self):
        unix_off = pd.to_datetime([self.current_datetime]) - pd.Timestamp("1970-01-01")
        unix_off_s = (unix_off // pd.Timedelta("1s")).to_numpy()
        [ind] = np.searchsorted(MARKET_UNIX_S, unix_off_s, side='right')
        if ind >= len(MARKET_DATETIMES):
            log.warning(f"No market bars marked for next timestep following {self.current_datetime}") 
            self.current_datetime = None
            return
        self.current_datetime = MARKET_DATETIMES[ind]
    
    def go_next_trading_day(self):
        unix_off = pd.to_datetime([self.current_datetime]) - pd.Timestamp("1970-01-01")
        unix_off_s = (unix_off // pd.Timedelta("1s")).to_numpy() + 60*60*7 # 7hrs, next trading day at least
        [ind] = np.searchsorted(MARKET_UNIX_S, unix_off_s, side='right')
        if ind >= len(MARKET_DATETIMES):
            log.warning(f"No market hours marked for next day following {self.current_datetime}") 
            self.current_datetime = None
            return
        self.current_datetime = MARKET_DATETIMES[ind]

    def get_price(self, symbol):
        df = self.fetcher.get_bars(symbol)
        if df is None:
            log.error(f"Failed to get price of {symbol}: No data")
            return None
        
        if not self.current_datetime in df.index.to_list():
            log.warning(f"{self.current_datetime} not in bars of {symbol}")
            return None
        
        bar = df.loc[self.current_datetime]
        return bar[self.trade_on]
    
def get_market_info():
    fetcher = utils.DataFetcher(os.path.join(".", "data", "historical"), "1h", "2023-01-01", None)
    top_companies = yf.Sector("basic-materials").top_companies.index.to_list()
    # fetcher.bulk_download(top_companies)
    total = set()
    for symbol in top_companies:
        df = fetcher.get_bars(symbol)
        new_dates = set(df.index.to_list())
        total.update(new_dates)
    
    sorted_datetimes = list(total)
    sorted_datetimes.sort()
    df = pd.DataFrame(sorted_datetimes, columns=["market_datetimes"])
    df.to_csv(os.path.join(".", "market", "info.csv"))
