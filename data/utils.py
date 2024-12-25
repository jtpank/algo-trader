import os

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger as log
from datetime import datetime

def _convert_to_datetime_unix(datetimes):
    timestamp_start = "1970-01-01"
    # Dirty check for "%Y-%m-%d %H:%M:%S"
    if len(datetimes[0]) == 17:
        timestamp_start += " 00:00:00"
    return ((pd.to_datetime(datetimes) - pd.Timestamp(timestamp_start)) // pd.Timedelta("1s")).to_numpy()

def find_nearest_indices(datetimes_to_insert, datetimes):
    insert_unix_s = _convert_to_datetime_unix(datetimes_to_insert)
    dt_unix_s = _convert_to_datetime_unix(datetimes)
    inds = np.searchsorted(dt_unix_s, insert_unix_s, side='left')
    return inds
    

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

        data = pd.read_csv(file_path)
        data = data.rename(columns={data.columns[0]: 'Datetime'}) 
        data.set_index("Datetime", inplace=True)
        return self._filter_history(data)
    
    def _get_from_local_adj(self, symbol, folder) -> pd.DataFrame:
        log.trace(f"Getting {symbol} from local adjacent folder {folder}")
        file_path = os.path.join(folder, f"{symbol}.csv")

        data = pd.read_csv(file_path)
        data = data.rename(columns={data.columns[0]: 'Datetime'}) 
        data.set_index("Datetime", inplace=True)

        [ind_start, ind_end] = find_nearest_indices([self.start, self.end], data.index)
        data = data.iloc[ind_start:ind_end]
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
    
    def _adjacent_folder_with_symbol(self, symbol):
        parent_data_folder = os.path.dirname(self.data_folder)
        all_adjacent_folders = os.listdir(parent_data_folder)
        
        # Can look in folders if they have the same interval and broader (or equal) timeframe
        compatible_adj_folders = []
        my_start = datetime.strptime(self.start, "%Y-%m-%d")
        my_end = datetime.strptime(self.end, "%Y-%m-%d")
        for folder in all_adjacent_folders:
            assert len(folder.split("$")) == 3
            [interval, start, end] = folder.split("$")
            assert start != "None"
            if end == "None": continue
            elif interval != self.interval: continue
            start_time = datetime.strptime(start, "%Y-%m-%d")
            end_time = datetime.strptime(end, "%Y-%m-%d")
            if start_time <= my_start and end_time >= my_end:
                compatible_adj_folders.append(folder)
        
        for folder in compatible_adj_folders:
            folder_path = os.path.join(parent_data_folder, folder)
            files = os.listdir(folder_path)
            csv_files = [f for f in files if f.endswith(".csv")]
            csv_symbols = [f[:-4] for f in csv_files]
            if symbol in csv_symbols:
                return folder_path
        
        return None
        

    def get_bars(self, symbol) -> pd.DataFrame | None:
        """
        Gets the data for the given symbol and timeframe
        """
        adj_folder = self._adjacent_folder_with_symbol(symbol)
        if self._has_symbol(symbol):
            history = self._get_from_local(symbol)
        elif adj_folder is not None:
            history = self._get_from_local_adj(symbol, adj_folder)
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
        # Only download files you don't already have
        all_files = os.listdir(self.data_folder)
        existing_symbols = [f.split(".csv")[0] for f in all_files if f.endswith(".csv")]
        symbols = [f for f in symbols if f not in existing_symbols]
        if len(symbols) == 0:
            log.trace("Already have all symbols...no download")
            return
        
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
