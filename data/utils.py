import os

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger as log


class DataFetcher(object):
    """
    DataFetcher is a concrete class that abstracts the retrieval of
    ticker information. If the fetcher has the data on disk it retrieves
    it from there, otherwise it interfaces with yfinance to get it.
    """

    def __init__(self, data_folder, interval, start, end):
        self.key_metrics = ["Open", "High", "Low", "Close", "Volume"]
        self.interval = interval
        self.start = start
        self.end = end
        marked_dir = f"{interval}${start}${end}"
        self.data_folder = os.path.join(data_folder, marked_dir)
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

    def _get_from_local(self, symbol) -> pd.DataFrame:
        log.trace(f"Getting {symbol} from local")
        file_path = os.path.join(self.data_folder, f"{symbol}.csv")

        data = pd.read_csv(file_path, index_col="Datetime")
        return self._filter_history(data)

    def _get_from_api(self, symbol) -> pd.DataFrame | None:
        log.trace(f"Getting {symbol} from api")
        yf_ticker = yf.Ticker(symbol)
        history = yf_ticker.history(
            interval=self.interval, start=self.start, end=self.end
        )

        # Retry once since api is flakey
        if history.empty:
            log.info(f"{symbol} not found. Retrying once...")
            history = yf_ticker.history(
                interval=self.interval, start=self.start, end=self.end
            )
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
        if history.index.name == "Datetime":
            history.reset_index(inplace=True)
            history.rename(columns={'Date': 'Datetime'}, inplace=True)
            history.set_index("Datetime", inplace=True)
        return self._filter_history(history)

    def _save_history(self, symbol: str, data: pd.DataFrame):
        log.trace(f"Saving {symbol} to {self.data_folder}")
        save_path = os.path.join(self.data_folder, f"{symbol}.csv")
        data.to_csv(save_path)

    def _filter_history(self, data: pd.DataFrame) -> pd.DataFrame:
        filtered = data[self.key_metrics]
        return filtered

    def _has_symbol(self, symbol):
        files = os.listdir(self.data_folder)
        csv_files = [f for f in files if f.endswith(".csv")]
        csv_symbols = [f[:-4] for f in csv_files]
        return symbol in csv_symbols

    def get_bars(self, symbol) -> pd.DataFrame | None:
        """
        Gets the data for the given symbol and timeframe
        """
        if self._has_symbol(symbol):
            history = self._get_from_local(symbol)
        else:
            history = self._get_from_api(symbol)
            if history is None:
                return None
            self._save_history(symbol, history)

        return history

    def bulk_download(self, symbols: list[str], _retry=False) -> list[str]:
        """
        Quickly download a bunch of symbols. This is a QOL if you already know what tickers you want to deal with and want to
        speed up accessing them, rather than fetching one by one. Note this also consumes far fewer API calls.
        This function does NOT check if the symbols already exist and WILL overwrite them if they do.

        Returns the list of symbols successfully downloaded
        """
        log.trace(f"Downloading {symbols}")
        histories = yf.download(
            symbols,
            start=self.start,
            end=self.end,
            interval=self.interval,
            ignore_tz=True,
            group_by="ticker",
        )
        downloaded_symbols = list(histories.columns.unique(level="Ticker"))
        failed_symbols = []
        for symbol in downloaded_symbols:
            history = histories[symbol]
            if history.empty:
                failed_symbols.append(symbol)
                continue

            self._save_history(symbol, history)

        saved_symbols = [s for s in symbols if s not in failed_symbols]
        log.info(f"Successfully downloaded {saved_symbols}")

        # Retry once
        if len(failed_symbols) > 0:
            if _retry:
                log.warning(f"Failed to download {failed_symbols}")
            else:
                log.info(f"Retrying download for {failed_symbols}")
                return saved_symbols + self.bulk_download(failed_symbols, True)

        return saved_symbols


def to_np(data: pd.DataFrame) -> np.ndarray:
    """
    Converts a DataFrame into a Numpy array.
    The array is float64 to preserve efficiency with operations. The unix timestamp will not have an off-by-one
    floating point precision problem until it is scaled by approximately 2^22.\n
    The columns correspond to [Date, Open, High, Low, Close, Volume]\n
    Date is in UNIX seconds, not milliseconds
    """
    copy = data.reset_index()
    copy["Datetime"] = (
        pd.to_datetime(copy["Datetime"]) - pd.Timestamp("1970-01-01")
    ) // pd.Timedelta("1s")
    # Guarantee ordering
    copy = copy[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
    data_np = copy.to_numpy(dtype=np.float64)
    return data_np
