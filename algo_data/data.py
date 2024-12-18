import datetime
import os
import os.path
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger as log

from algo_event.event import MarketEvent


class DataHandler(object):
    """
    DataHandler is an abstract base class providing an interface for
    all subsequent (inherited) data handlers (both live and historic).
    The goal of a (derived) DataHandler object is to output a generated
    set of bars (OHLCVI) for each symbol requested.
    This will replicate how a live strategy would function as current
    market data would be sent "down the pipe". Thus a historic and live
    system will be treated identically by the rest of the backtesting suite.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_latest_bar(self, symbol):
        """
        Returns the last bar updated.
        """
        raise NotImplementedError("Should implement get_latest_bar()")

    @abstractmethod
    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N bars updated.
        """
        raise NotImplementedError("Should implement get_latest_bars()")

    @abstractmethod
    def get_latest_bar_datetime(self, symbol):
        """
        Returns a Python datetime object for the last bar.
        """
        raise NotImplementedError("Should implement get_latest_bar_datetime()")

    @abstractmethod
    def get_latest_bar_value(self, symbol, val_type):
        """
        Returns one of the Open, High, Low, Close, Volume or OI
        from the last bar.
        """
        raise NotImplementedError("Should implement get_latest_bar_value()")

    @abstractmethod
    def get_latest_bars_values(self, symbol, val_type, N=1):
        """
        Returns the last N bar values from the
        latest_symbol list, or N-k if less available.
        """
        raise NotImplementedError("Should implement get_latest_bars_values()")

    @abstractmethod
    def update_bars(self):
        """
        Pushes the latest bars to the bars_queue for each symbol
        in a tuple OHLCVI format: (datetime, open, high, low,
        close, volume, open interest).
        """
        raise NotImplementedError("Should implement update_bars()")


class DataFetcher(object):
    """
    DataFetcher is a concrete class that abstracts the retrieval of
    ticker information. If the fetcher has the data on disk it retrieves
    it from there, otherwise it interfaces with yfinance to get it.
    """

    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.key_metrics = ["Open", "High", "Low", "Close", "Volume"]
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

    def _get_from_local(self, symbol, start, end, interval):
        log.trace(f"Getting {symbol}")
        file_path = os.path.join(self.data_folder, f"{symbol}.csv")
        data = pd.read_csv(file_path, index_col="Date")
        return self._filter_history(data)

    def _get_from_api(self, symbol, start, end, interval):
        log.trace(
            f"Getting {symbol} between ${start}, ${end} at ${interval} increments"
        )
        yf_ticker = yf.Ticker(symbol)
        history = yf_ticker.history(interval=interval, start=start, end=end)

        # Retry once since api is flakey
        if history.empty:
            log.info(f"{symbol} not found. Retrying once...")
            history = yf_ticker.history(interval=interval, start=start, end=end)
            if history.empty:
                log.warning(f"Could not find {symbol}")
                return None

        missing_metrics = [
            a for a in self.key_metrics if a not in history.columns.tolist()
        ]
        if len(missing_metrics) > 0:
            log.warning(
                f"Missing key metrics {missing_metrics} for {symbol}...Ignoring"
            )
            return None

        history.index = history.index.tz_localize(None)
        return self._filter_history(history)

    def _save_history(self, symbol: str, data: pd.DataFrame):
        log.trace(f"Saving {symbol} to {self.data_folder}")
        save_path = os.path.join(self.data_folder, f"{symbol}.csv")
        data.to_csv(save_path)

    def _filter_history(self, data: pd.DataFrame):
        filtered = data[self.key_metrics]
        return filtered

    def _has_symbol(self, symbol):
        files = os.listdir(self.data_folder)
        csv_files = [f for f in files if f.endswith(".csv")]
        csv_symbols = [f[:-4] for f in csv_files]
        return symbol in csv_symbols

    def get_bars(self, symbol, start, end=None, interval="1d"):
        """
        Gets the data for the given symbol and timeframe
        """
        if self._has_symbol(symbol):
            history = self._get_from_local(symbol, start, end, interval)
        else:
            history = self._get_from_api(symbol, start, end, interval)
            if history is None:
                return None
            self._save_history(symbol, history)

        return history
