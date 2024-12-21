import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from loguru import logger as log
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.rolling import RollingOLS
import itertools

#NOTE: Implemented from this lecture: https://www.youtube.com/watch?v=JTucMRYMOyY

@dataclass
class StationarityStruct:
    p_value: float
    is_stationary: bool
    integrator_order: int


class PairsTradingPipeline(object):
    """
    Pipeline to: check stationarity, cointigration, and integrate with a function
    """

    def __init__(
        self,
        input_data_set: Dict[str, np.ndarray],
        field: str = "Open",
        integrator_function_key: str = "pct_change_integrator",
        adf_cutoff: float = 0.01,
        cointegration_cutoff: float = 0.01,
    ):
        # for now only accept 1 time series
        self.input_data_set = input_data_set
        self.field = field
        self.integrator_function_key = integrator_function_key
        self.cleaned_data_set = {}
        self.stationarity_set = {}
        self.cointegrated_pairs_set = set()
        self.adf_cutoff = adf_cutoff
        self.integrator = IntegratorTypes()
        self.save_plots = False
        self.integrator_cutoff = 1  # max number of attempts to integrate
        self.cointegration_cutoff = cointegration_cutoff
        log.info(
            f"Generated PairsTradingPipeline Object with {len(input_data_set)} time series."
        )

    def run(self):
        assert self.input_data_set
        self._clean_data()

        assert self.cleaned_data_set
        self._generate_stationary_set()

        # Now lets begin to perform cointegration
        # 1. We need a set S of integrator of order n = 1 time series: S = (X0, X1, ... Xn) for Xi is order n
        # 2. If there exists a linear combination of those series in the set S such that the new series is stationary (order n = 0)
        #  ==> then the series are cointegrated!
        # NOTE: That the imported function: coint can be used! but it only comes with augmented Engle-Granger method see:
        # https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html#statsmodels.tsa.stattools.coint

        #TODO need to look for confounding! ( x coint y) and ( z coint y) therefore x is really coint with z
        assert len(self.stationarity_set) > 0
        self._find_cointegrated_pairs()
        assert len(self.cointegrated_pairs_set) > 0
        self._remove_confounding_pairs()

        first_pair = list(self.cointegrated_pairs_set)[0]
        data_y = self.cleaned_data_set[first_pair[0]]
        data_x = self.cleaned_data_set[first_pair[1]]
        self._run_moving_average_regression(data_y, data_x)

    def _clean_data(self):
        # TODO: make this better (this is hacky right now)... maybe clean before creating object so
        #  we are not lugging around all the data all the time...
        index = 0
        if self.field == "Open":
            index = 1
        for ticker, all_data in self.input_data_set.items():
            self.cleaned_data_set[ticker] = all_data[:, index]
        log.info("Cleaned data")

    def _check_stationarity(
        self, ticker: str, data: np.ndarray, int_order: int
    ) -> StationarityStruct:
        log.info(f"\t\tChecking stationarity for time series: {ticker}")
        adf_result = adfuller(data)
        p_value = adf_result[1]
        validate = True if p_value < self.adf_cutoff else False
        integrator_order = int_order
        log.info(
            f"\t\ttime series: {ticker} p_value: {p_value} is_stationary: {validate}"
        )
        return StationarityStruct(
            p_value=p_value, is_stationary=validate, integrator_order=integrator_order
        )

    def _generate_stationary_series(
        self, data: np.ndarray, integrator_type: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        log.info(f"Integrating data")
        return integrator_type(data)

    def _find_cointegrated_pairs(self):
        #create an array of tickers of order 1 integrated time series
        order_1_series = []
        for ticker, stationarity_obj in self.stationarity_set.items():
            if stationarity_obj.integrator_order == 1:
                order_1_series.append(ticker)
        result = [(a, b) for a, b in itertools.combinations(order_1_series, 2)]
        for pair in result:
            y0 = self.cleaned_data_set[pair[0]]
            y1 = self.cleaned_data_set[pair[1]]
            # assert( len(y0) == len(y1))
            if len(y0) == len(y1):
                coint_output = coint(y0, y1, trend='c', method='aeg', maxlag=None, autolag='aic', return_results=None)
                coint_pvalue = coint_output[1]
                if coint_pvalue < self.cointegration_cutoff:
                    log.info(f"Found cointegrated pair! {pair}, pvalue: {coint_pvalue}")
                    self.cointegrated_pairs_set.add(pair)
            else:
                log.error(f'Incompatible lengths of time series: {pair[0]} : {len(y0)} vs {pair[1]} : {len(y1)}')

    def _generate_stationary_set(self):
        for ticker, time_series in self.cleaned_data_set.items():
            count = 0
            temp_time_series = np.copy(time_series)
            stationarity_obj = self._check_stationarity(ticker, temp_time_series, count)
            #Note: What we really want to do is see for which IntegratorTypes do we get an
            # integrator of order 1 time series. 
            # Then we match all integrator of order 1 series for that specific IntegratorType
            # Then we can perform cointegration
            #TODO: FIX! see above
            while not stationarity_obj.is_stationary or (
                count > self.integrator_cutoff
            ):
                count += 1
                temp_time_series = self._generate_stationary_series(
                    temp_time_series, self.integrator.integrator_dict[self.integrator_function_key]
                )
                stationarity_obj = self._check_stationarity(ticker, temp_time_series, count)
            self.stationarity_set[ticker] = stationarity_obj
            if self.save_plots:
                # TODO make this a function
                plt.plot(list(range(0, len(temp_time_series))), temp_time_series)
                plt.ylabel("Pct Change Returns - time delta is 1d")
                plt.legend([ticker])
                data_folder = os.path.join(".", "pipelines", "pairs_pipeline_images")
                output_img_path = os.path.join(data_folder, f"{ticker}-stationary.png")
                plt.savefig(output_img_path)
                plt.close()
                log.info(f"Saving image to {output_img_path}")
    
    def _remove_confounding_pairs(self):
        #TODO check and remove confounded pairs
        pass

    def _run_moving_average_regression(self, data_y, data_x, window_size: int = 30):
        #This is currently a moving average (simple, and prevents look ahead bias)
        #seed for reproducibility
        #TODO this needs to be understood more...
        np.random.seed(69)
        rolling_model = RollingOLS(endog=data_y, exog=data_x, window=window_size)
        rolling_results = rolling_model.fit(params_only=True)
        spread = data_x - rolling_results.params * data_y

        # spread mov1  = pd.rolling_mean(spread, window=1)
        #also need a rolling mean + rolling std dev to accompany this
        # print(rolling_results.params)

        

class IntegratorTypes(object):
    """
    A collection of custom (or not) integrator functions for data
    Defines a lookup dictionary so you can pass in a string to select your desired integrator function
    """

    def __init__(self):
        log.info(f"Constructed IntegratorTypes Object")
        self.integrator_dict = {
            "difference_integrator": self.diff_integrator,
            "pct_change_integrator": self.pct_change_integrator
        }
        log.info(f"\t Integrator keys: {self.integrator_dict.keys()}")

    def diff_integrator(self, data: np.ndarray) -> np.ndarray:
        assert len(data) > 2
        # this is very simple, but you get the idea
        # we can implement custom integrator functions here
        return np.diff(data, n=1)

    def pct_change_integrator(self, data: np.ndarray) -> np.ndarray:
        # also very simple, but yea its another good one
        assert len(data) > 2
        pct_change = np.diff(data) / data[:-1] * 100
        return pct_change
