import os
import sys
from loguru import logger as log

from algo_event import event
from model import strategy
from data import utils
from pipelines import pairs_trading_pipeline as ptp
from model.strategy import PairsStrategy
from market.drivers import simulate_pairs, SimulatedTrader
import matplotlib.pyplot as plt
import yfinance as yf


def run_ptp():
    # 1. Collect time series data
    data_folder = os.path.join(".", "data", "historical")
    interval = "1d"
    start = "2023-01-01"
    end = "2025-01-01"
    # tickers = ["NVDA", "TSLA"]
    #note that the below has an example of confounding variables
    # these are tickers from solar sector
    # tickers = ['FSLR', 'ENPH', 'NXT', 'RUN', 'ARRY', 'SEDG', 'CSIQ', 'MAXN']
    tickers = yf.Industry("specialty-chemicals").top_companies.index.to_list()
    print(tickers)
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

def main():
    # log.remove()
    # log.add(sys.stderr, level="TRACE")
    log.info("In main")
    # run_ptp()

    # HWKN,ECVT;AVNT,PRM;EMN,BCPC;WLK,FUL
    symbolx = "AVNT"
    symboly = "PRM"

    fig, axs = plt.subplots(1, 3, figsize=(10,5))

    for iden, buying_power in enumerate([1000, 2000, 4000]):
        trader = SimulatedTrader()
        strategy = PairsStrategy(symbolx, symboly, trader, buying_power, z_enter=1, z_exit=0.5)
        simulate_pairs(symbolx, symboly, strategy, trader)
        [dates, capital] = list(zip(*strategy.record))
        # plt.plot(dates, capital)
        axs[iden].plot(dates, capital)
        axs[iden].set_title(f"capital: {buying_power}")

    plt.show()



if __name__ == "__main__":
    # print(yf.Sector("basic-materials").top_companies.index)
    # run_ptp()
    main()
