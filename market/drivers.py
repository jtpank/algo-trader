from market.traders import SimulatedTrader
from model.strategy import PairsStrategy
from data.utils import DataFetcher
from pipelines.PairsTrader import PairsTraderStatic
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from loguru import logger as log

MIN_DRAWDOWN_PCT = 1
MAX_DRAWDOWN_PCT = 50

class PairsDriver(object):
    def __init__(self, symbolx, symboly, start_datetime, strategy: PairsStrategy, trader: SimulatedTrader, fetcher: DataFetcher):
        self.symbolx = symbolx
        self.symboly = symboly
        self.start_datetime = start_datetime
        self.strategy = strategy
        self.trader = trader
        self.fetcher = fetcher
        self.sharpe_ratio = float('-inf')
    
    def simulate(self):
        df1 = self.fetcher.get_bars(self.symbolx)
        df2 = self.fetcher.get_bars(self.symboly)
        assert df1 is not None and df2 is not None
        # print(set(df1.index.to_list()).symmetric_difference(set(df2.index.to_list())))
        if not df1.index.equals(df2.index):
            log.error(f"Mismatched indices for {self.symbolx} {self.symboly}")
            return True

        # Only introduce the 'test' section of the data
        df1_known = df1.loc[:self.start_datetime]
        df2_known = df2.loc[:self.start_datetime]
        pairs_analyzer = PairsTraderStatic(df1_known, df2_known)

        sharpe_ratio, max_dd_pct = self.backtest(pairs_analyzer)
        self.sharpe_ratio = sharpe_ratio
        # test to make sure

        self.strategy.max_dd_pct = max_dd_pct
        self.livetest(pairs_analyzer)


    def backtest(self, pairs_analyzer: PairsTraderStatic):
        # Find the first z_score to start backtesting
        while True:
            self.trader.go_next_trading_hour()
            datetime = self.trader.current_datetime
            z_score = pairs_analyzer.get_zscore(datetime)
            if not np.isnan(z_score):
                break

        # Backtest
        while True:
            beta = pairs_analyzer.get_beta(datetime)
            z_score = pairs_analyzer.get_zscore(datetime)
            if np.isnan(beta) or datetime == self.start_datetime:
                break

            self.strategy.update(z_score, -beta, 1)
            self.trader.go_next_trading_hour()
            datetime = self.trader.current_datetime
        
        [dates, capital] = list(zip(*self.strategy.record))
        capital = np.array(capital[::2])
        # for i in range(len(capital)): print(f"capital: {capital[i]}")
        acc_max = np.maximum.accumulate(capital)
        # for i in range(len(acc_max)): print(f"accmax: {acc_max[i]}")
        drawdowns = capital - acc_max
        ind = np.argmin(drawdowns)
        max_dd = drawdowns[ind]
        # print(f"max_dd: {max_dd}")
        # max_dd_date = dates[ind]
        log.error(f"accmax: {acc_max[ind]}, maxdd: {max_dd}")
        max_dd_pct = 100.0 * -max_dd/self.strategy.capital_per_trade
        log.info(f"true backtest max_dd_pct: {max_dd_pct}")
        max_dd_pct = min(MAX_DRAWDOWN_PCT, max(max_dd_pct, MIN_DRAWDOWN_PCT))
        if len(capital) <= 2:
            log.info(f"Not enough trades to create sharpe_ratio")
            return 'NA', max_dd_pct
        # print(capital)
        total_returns = (capital[-1] - capital[0])/capital[0]
        risk_free_ror = 0.06
        returns = (capital[1:] - capital[:-1]) / self.strategy.capital_per_trade
        returns_std = np.std(returns)
        # print(returns)
        log.error(f"totaled_returns: {np.sum(returns)}")
        log.error(f"total_returns: {total_returns}, rfror: {risk_free_ror}, ret_std: {returns_std}")
        sharpe_ratio = (total_returns - risk_free_ror)/returns_std
        log.error(f"sharpe_ratio: {sharpe_ratio}")

        return sharpe_ratio, max_dd_pct


    def livetest(self, pairs_analyzer: PairsTraderStatic):
        df1 = self.fetcher.get_bars(self.symbolx)
        df2 = self.fetcher.get_bars(self.symboly)
        datetime = self.start_datetime
        while True:
            if datetime is None: break
            df1_next_datetime = df1.loc[[datetime]]
            df2_next_datetime = df2.loc[[datetime]]
            pairs_analyzer.update(df1_next_datetime, df2_next_datetime)
            beta = pairs_analyzer.get_beta(datetime)
            # log.error(f"Beta: {beta}")
            z_score = pairs_analyzer.get_zscore(datetime)
            if np.isnan(beta): break
            self.strategy.update(z_score, -beta, 1)
            # strategy.update(z_score, -beta, 1)
            self.trader.go_next_trading_hour()
            datetime = self.trader.current_datetime
