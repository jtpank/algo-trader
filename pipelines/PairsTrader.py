import pandas as pd
import numpy as np
import sys
import os
from loguru import logger
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import adfuller, coint
import matplotlib.pyplot as plt
from datetime import datetime
from data.utils import to_np

class PairsTrader(object):
    def __init__(self, df_series_x: pd.DataFrame, df_series_y: pd.DataFrame):
        self.key = "Open"
        self.window_size = 210
        self.do_plots = False
        self.is_initialized = False
        self.cointegration_cutoff: float = 0.05,
        self.rolling_coint = pd.DataFrame(columns=["Datetime", "Cointegrated"]).set_index("Datetime")
        self.df_x = pd.DataFrame()
        self.df_y = pd.DataFrame()
        self.series_x = pd.Series()
        self.series_y = pd.Series()
        self.spread = pd.Series()
        self.zscore_30_1 = pd.Series()
        self.p_value_series_chart = []
        self.series_x_chart = []
        self.series_y_chart = []
        self.rolling_zscore_chart = []

        self.rolling_zscore = pd.DataFrame()
        self.rolling_beta = pd.DataFrame()
        logger.info("Initializing PairsTrader object:")
        self._run_initialization(df_series_x, df_series_y)
        logger.info("Successfully initialized PairsTrader object.")

    #comput the rolling zscore from the cointegrated pair
    def _run_initialization(self, df_series_x: pd.DataFrame, df_series_y: pd.DataFrame):
        self.update(df_series_x, df_series_y)
        if self.do_plots:
            plt.plot(self.zscore_30_1.index, self.zscore_30_1)  # Use the index for x-values and the values for y
            plt.ylabel("zscore")
            data_folder = os.path.join(".", "pairs_zscores")
            output_img_path = os.path.join(data_folder, f"zscore.png")
            plt.savefig(output_img_path)
            plt.close()
            logger.info("Plotted rolling_zscore.")
    
    def get_zscore(self, date_str: str):
        """
        Asserts the datetime format is correct.
        Returns the float value of the zscore at that date.
        Returns nan if the date is not found OR the value is nan
        """
        try:
            datetime.fromisoformat(date_str)
        except ValueError:
            raise AssertionError(f"{date_str}")
    
        selected_row = self.rolling_zscore[self.rolling_zscore["Datetime"] == date_str]
        
        # selected_row = self.rolling_zscore.loc[date_str]
        if selected_row.empty:
            logger.error(f"Selected row for {date_str} is not found!")
            return np.nan
        val = selected_row["Zscore"].iloc[0]
        return val

    def get_beta(self, date_str: str):
        """
        Asserts the datetime format is correct.
        Returns the float value of the beta at that date.
        Returns nan if the date is not found OR the value is nan
        """
        try:
            datetime.fromisoformat(date_str)
        except ValueError:
            raise AssertionError(f"{date_str}")
        
        selected_row = self.rolling_beta[self.rolling_beta["Datetime"] == date_str]
        if selected_row.empty:
            logger.error(f"Selected row for {date_str} is not found!")
            return np.nan
        val = selected_row["Beta"].iloc[0]
        return val
    
    def update(self, data_row_x: pd.DataFrame, data_row_y: pd.DataFrame):
        #append the data point and update the zscore
        if not self.df_x.empty and not self.df_y.empty:
            self.df_y = pd.concat([self.df_y, data_row_y])
            self.df_x = pd.concat([self.df_x, data_row_x])
            logger.trace("Updated df_y and df_x with new data.")
        else:
            self.df_y = data_row_y
            self.df_x = data_row_x
            logger.trace("Initialized df_y and df_x.")
        self.series_x = self.df_x[self.key]
        self.series_y = self.df_y[self.key]
        assert(len(self.series_y) == len(self.series_x))

        date_start = self.df_x.index[-self.window_size]
        date_end = self.df_x.index[-1]
        # print(f"datestart: end : {date_start} : {date_end}")
        # self._check_cointegration_over_window(date_start, date_end)
        # self.is_cointegrated_on_date(date_end)
        # TODO: optimization only rolling OLS on the end
        series_x_const = sm.add_constant(self.series_x)
        roll_ols_model = RollingOLS(self.series_y,  series_x_const , window=self.window_size)
        rolling_results = roll_ols_model.fit(params_only=True)
        # import IPython
        # IPython.embed()

        self.rolling_beta = pd.DataFrame()
        self.rolling_beta["Datetime"] = self.df_y.index.values
        self.rolling_beta["Beta"] = rolling_results.params[self.key].reset_index(drop=True)
        spread = self.series_y - rolling_results.params['const'] - rolling_results.params[self.key] *  self.series_x 
        adf_result = adfuller(spread.dropna())
        p_value = adf_result[1]
        self.p_value_series_chart.append(p_value)
        self.series_x_chart.append(self.series_x.iloc[-1])
        self.series_y_chart.append(self.series_y.iloc[-1])

        spread_mavg1 = spread.rolling(window=1).mean()
        spread_mavg30 = spread.rolling(self.window_size).mean()
        std_30 = spread.rolling(window=self.window_size).std()
        self.zscore_30_1 = (spread_mavg1 - spread_mavg30)/std_30

        #inefficient
        self.rolling_zscore = pd.DataFrame()
        self.rolling_zscore["Datetime"] = self.df_y.index.values
        self.rolling_zscore["Zscore"] = self.zscore_30_1.values
        logger.trace("Updated rolling_zscore.")
        self.rolling_zscore_chart.append(self.rolling_zscore["Zscore"].iloc[-1])

    def is_cointegrated_on_date(self, date: str):
        coint_data_row = self.rolling_coint[self.rolling_coint.index == date]
        if not coint_data_row.empty and coint_data_row['Cointegrated'].iloc[0]:
            return True
        return False


    def _check_cointegration_over_window(self, date_start: str, date_end: str):
        df_slice_x = self.df_x.loc[date_start:date_end]
        df_slice_y = self.df_y.loc[date_start:date_end]

        # logger.info("Inside cointegration check")
        series_x = df_slice_x[self.key].to_numpy()
        series_y = df_slice_y[self.key].to_numpy()
        coint_output = coint(series_x, series_y, trend='c', method='aeg', maxlag=None, autolag='aic', return_results=None)
        col_date = df_slice_x.index[-1]
        bool_val = np.float64(coint_output[1]) < self.cointegration_cutoff
        self.rolling_coint.loc[col_date] = {'Cointegrated' : bool_val}
        # coint_data_row = pd.Series({'Datetime': col_date, 'Cointegrated': bool_val})
        # print(f"before: {self.rolling_coint}")
        # self.rolling_coint = pd.concat([self.rolling_coint, coint_data_row])c
        # print(f"after {self.rolling_coint}")
        # return np.float64(coint_output[1]) < self.cointegration_cutoff

    def output_primary_charts(self):
        plt.figure()
        fig, ax = plt.subplots(4, figsize=(30, 15))
        x_range = list(range(0, len(self.p_value_series_chart)))
        ax[0].plot(x_range, self.p_value_series_chart)
        ax[0].set_title('P-Value')
        ax[1].plot(x_range, self.rolling_zscore_chart)
        ax[1].set_title('rolling z scores')
        ax[2].plot(x_range, self.series_x_chart)
        ax[2].set_title('X series')
        ax[3].plot(x_range, self.series_y_chart)
        ax[3].set_title('Y series')
        data_folder = os.path.join(".", "pipelines", "pairs_pipeline_images")
        output_img_path = os.path.join(data_folder, f"combined-data.png")
        plt.savefig(output_img_path)
        plt.close()


if __name__=="__main__":
    base_path = '/Users/justin/algo/algo-trader/data/historical/1h$2023-01-01$2024-01-03'
    ticker_1 = "AVNT"
    ticker_2 = "PRM"
    stock_1_path = os.path.join(base_path,ticker_1 + ".csv" )
    stock_2_path = os.path.join(base_path,ticker_2 + ".csv" )
    stock_df_1 = pd.read_csv(stock_1_path) 
    stock_df_2 = pd.read_csv(stock_2_path)
    pt = PairsTrader(stock_df_1, stock_df_2)



