from market.traders import SimulatedTrader
from model.strategy import PairsStrategy
from data.utils import DataFetcher
from pipelines.PairsTrader import PairsTrader
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from loguru import logger as log

def simulate_pairs(symbolx, symboly, strategy: PairsStrategy, trader: SimulatedTrader, fetcher: DataFetcher, fail=False):
    df1 = fetcher.get_bars(symbolx)
    df2 = fetcher.get_bars(symboly)
    assert df1 is not None and df2 is not None
    # print(set(df1.index.to_list()).symmetric_difference(set(df2.index.to_list())))
    assert df1.index.equals(df2.index)

    # Quick and dirty way to get the first year, since ~1744 datetimes
    df1_2023 = df1.iloc[:1744]
    df2_2023 = df2.iloc[:1744]
    pairs_analyzer = PairsTrader(df1_2023, df2_2023)
    
    while True:
        trader.go_next_trading_hour()
        datetime = trader.current_datetime
        beta = pairs_analyzer.get_zscore(datetime)
        if not np.isnan(beta):
            break

    while True:
        beta = pairs_analyzer.get_beta(datetime)
        z_score = pairs_analyzer.get_zscore(datetime)
        if np.isnan(beta):
            break

        if fail:
            strategy.update(z_score, beta, -1)
        else:
            strategy.update(z_score, -beta, 1)
        # strategy.update(z_score, -beta, 1)
        trader.go_next_trading_hour()
        datetime = trader.current_datetime

    ind = 1744

    while True:
        if ind > len(df1.index) or datetime is None: break
        df1_next_datetime = df1.iloc[[ind]]
        df2_next_datetime = df2.iloc[[ind]]
        pairs_analyzer.update(df1_next_datetime, df2_next_datetime)
        if pairs_analyzer.is_cointegrated_on_date(df1_next_datetime.index[0]):
            beta = pairs_analyzer.get_beta(datetime)
            z_score = pairs_analyzer.get_zscore(datetime)
            if np.isnan(beta): break
            if fail:
                strategy.update(z_score, beta, -1)
            else:
                strategy.update(z_score, -beta, 1)
        # strategy.update(z_score, -beta, 1)
        trader.go_next_trading_hour()
        datetime = trader.current_datetime
        ind += 1
