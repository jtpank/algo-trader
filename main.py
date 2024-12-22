import os
import sys

from loguru import logger as log

from algo_event import event
from algo_model import strategy
from data import utils
from pipelines import pairs_trading_pipeline as ptp


def main():
    log.remove()
    log.add(sys.stderr, level="TRACE")
    # me = event.MarketEvent()
    log.info("In main")

    # 1. Collect time series data
    data_folder = os.path.join(".", "data", "historical")
    interval = "1d"
    start = "2023-01-01"
    end = "2025-01-01"
    # tickers = ["NVDA", "TSLA"]
    #note that the below has an example of confounding variables
    # these are tickers from solar sector
    tickers = ['FSLR', 'ENPH', 'NXT', 'RUN', 'ARRY', 'SEDG', 'CSIQ', 'MAXN']
    #tickers = ['ENPH', 'CSIQ']
    data_set_dict = {}
    for ticker in tickers:
        data_fetcher = utils.DataFetcher(data_folder, interval, start, end)
        df = data_fetcher.get_bars(ticker)
        arr = utils.to_np(df)
        data_set_dict[ticker] = arr

    # 2. Create and run a pipeline with stock set
    log.info("Starting pairs pipeline:")
    pipeline = ptp.PairsTradingPipeline(input_data_set=data_set_dict)
    pipeline.run()


if __name__ == "__main__":
    main()
