import pandas as pd
import numpy as np
import os
from loguru import logger
from statsmodels.regression.rolling import RollingOLS
import matplotlib.pyplot as plt
from datetime import datetime

class PairsTrader(object):
    def __init__(self, df_series_x: pd.DataFrame, df_series_y: pd.DataFrame):
        self.key = "Open"
        self.window_size = 30
        self.do_plots = False
        self.df_x = pd.DataFrame()
        self.df_y = pd.DataFrame()
        self.series_x = pd.Series()
        self.series_y = pd.Series()
        self.spread = pd.Series()
        self.zscore_30_1 = pd.Series()
        self.rolling_zscore = pd.DataFrame()
        self.rolling_beta = pd.DataFrame()
        logger.info("Initializing PairsTrader object:")
        self._run_initialization(df_series_x, df_series_y)
        logger.info("Successfully initialized ParisTrader object.")

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
        
        selected_row = self.rolling_zscore[self.rolling_zscore["Date"] == date_str]
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
        
        selected_row = self.rolling_beta[self.rolling_beta["Date"] == date_str]
        if selected_row.empty:
            logger.error(f"Selected row for {date_str} is not found!")
            return np.nan
        val = selected_row["Beta"].iloc[0]
        return val
    
    def update(self, data_row_x: pd.DataFrame, data_row_y: pd.DataFrame):
        #append the data point and update the zscore
        if not self.df_x.empty and not self.df_y.empty:
            self.df_y = pd.concat([self.df_y, data_row_y], ignore_index=True)
            self.df_x = pd.concat([self.df_x, data_row_x], ignore_index=True)
            logger.info("Updated df_y and df_x with new data.")
        else:
            self.df_y = data_row_y
            self.df_x = data_row_x
            logger.info("Initialized df_y and df_x.")
        self.series_x = self.df_x[self.key]
        self.series_y = self.df_y[self.key]
        assert(len(self.series_y) == len(self.series_x))

        roll_ols_model = RollingOLS(self.series_y,  self.series_x , window=self.window_size)
        rolling_results = roll_ols_model.fit(params_only=True)

        self.rolling_beta = pd.DataFrame()
        self.rolling_beta["Date"] = self.df_y["Date"].values
        self.rolling_beta["Beta"] = rolling_results.params[self.key]
        spread = self.series_y - rolling_results.params[self.key] *  self.series_x 
        spread_mavg1 = spread.rolling(window=1).mean()
        spread_mavg30 = spread.rolling(self.window_size).mean()
        std_30 = spread.rolling(window=self.window_size).std()
        self.zscore_30_1 = (spread_mavg1 - spread_mavg30)/std_30

        #inefficient
        self.rolling_zscore = pd.DataFrame()
        self.rolling_zscore["Date"] = self.df_y["Date"].values
        self.rolling_zscore["Zscore"] = self.zscore_30_1.values
        logger.info("Updated rolling_zscore.")


if __name__=="__main__":
    base_path = '/Users/justin/algo/algo-trader/data/historical/1d$2023-01-01$2025-01-01'
    ticker_1 = "ENPH"
    ticker_2 = "CSIQ"
    stock_1_path = os.path.join(base_path,ticker_1 + ".csv" )
    stock_2_path = os.path.join(base_path,ticker_2 + ".csv" )
    stock_df_1 = pd.read_csv(stock_1_path) 
    stock_df_2 = pd.read_csv(stock_2_path)
    pt = PairsTrader(stock_df_1, stock_df_2)
    # val = pt.get_zscore("2024-12-21")
    # beta = pt.get_beta("2024-12-21")
    # print(f"Val before: {val}, beta before: {beta}")
    # data_x = {
    #     'Date': ["2024-12-21"],
    #     "Open": 33.10,
    #     "High": 35.3,
    #     "Low": 31.3,
    #     "Close": 32.55,
    #     "Volume": 140302
    # }
    # data_y = {
    #     'Date': ["2024-12-21"],
    #     "Open": 133.10,
    #     "High": 135.3,
    #     "Low": 131.3,
    #     "Close": 132.55,
    #     "Volume": 230302
    # }
    # df_x_row = pd.DataFrame(data_x)
    # df_y_row = pd.DataFrame(data_y)
    # pt.update(df_x_row, df_y_row)
    # val = pt.get_zscore("2024-12-21")
    # beta = pt.get_beta("2024-12-21")
    # print(f"Val after: {val}, beta after: {beta}")


