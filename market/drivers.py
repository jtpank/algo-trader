from market.traders import SimulatedTrader
from model.strategy import PairsStrategy
from data.utils import DataFetcher
from pipelines.PairsTrader import PairsTrader
import numpy as np
import os
import sys
from loguru import logger as log

def simulate_pairs(symbolx, symboly):
    trader = SimulatedTrader()
    strategy = PairsStrategy(symbolx, symboly, trader, 1000)
    data_folder = os.path.join(".", "data", "historical")
    fetcher = DataFetcher(data_folder, "1d", "2023-01-01", None)

    df1 = fetcher.get_bars(symbolx)
    df2 = fetcher.get_bars(symboly)
    assert df1 is not None and df2 is not None
    assert df1.index.equals(df2.index)

    # Quick and dirty way to get the first year, since 252 days
    df1_2023 = df1.iloc[:252]
    df2_2023 = df2.iloc[:252]
    pairs_analyzer = PairsTrader(df1_2023, df2_2023)
    
    while True:
        trader.go_next_trading_day()
        day = trader.current_day
        beta = pairs_analyzer.get_zscore(day)
        if not np.isnan(beta):
            break

    while True:
        beta = pairs_analyzer.get_beta(day)
        z_score = pairs_analyzer.get_zscore(day)
        if np.isnan(beta):
            break
        strategy.update(z_score, -beta, 1)
        trader.go_next_trading_day()
        day = trader.current_day

    ind = 252
    while True:
        if ind > len(df1.index): break
        df1_next_day = df1.iloc[[ind]]
        df2_next_day = df2.iloc[[ind]]
        pairs_analyzer.update(df1_next_day, df2_next_day)

        beta = pairs_analyzer.get_beta(day)
        z_score = pairs_analyzer.get_zscore(day)
        if np.isnan(beta): break
        strategy.update(z_score, -beta, 1)
        trader.go_next_trading_day()
        day = trader.current_day
        ind += 1

    
# log.remove()
# log.add(sys.stderr, level="TRACE")
simulate_pairs("ENPH", "CSIQ")