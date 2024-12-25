import os
import sys
from loguru import logger as log

from data import utils
from market.portfolio import Portfolio
from pipelines import pairs_trading_pipeline as ptp
from model.strategy import PairsStrategy
from market.drivers import PairsDriver
from market.traders import SimulatedTrader
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import numpy as np
from market.portfolio import Portfolio

def fmt_float(num):
    return "{:2f}".format(num)

def run_ptp():
    # 1. Collect time series data
    data_folder = os.path.join(".", "data", "historical")
    interval = "1h"
    start = "2023-01-01"
    end = "2024-01-03"
    # tickers = ["NVDA", "TSLA"]
    #note that the below has an example of confounding variables
    # these are tickers from solar sector
    # tickers = ['FSLR', 'ENPH', 'NXT', 'RUN', 'ARRY', 'SEDG', 'CSIQ', 'MAXN']
    tickers = yf.Industry("specialty-chemicals").top_companies.index.to_list()
    print(tickers)
    # tickers = ['ENPH', 'CSIQ']
    tickers = ['AVNT', 'PRM']
    data_set_dict = {}
    for ticker in tickers:
        data_fetcher = utils.DataFetcher(data_folder, interval, start, end)
        df = data_fetcher.get_bars(ticker)
        arr = utils.to_np(df)
        print(f"len: {len(df)}")
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
    # portfolio = Portfolio()
    # portfolio.find_pairs()
    


    fetcher = utils.DataFetcher(os.path.join(".", "data", "historical"), "1h", "2024-06-03", "2024-11-01")
    pairs = [('T', 'FYBR'), ('YUM', 'SHAK'), ('EAT', 'CAKE'), ('PAG', 'CARG'), ('CVNA', 'AN'), ('PAG', 'SAH'), ('HGV', 'PLYA'), ('CZR', 'GDEN'), ('COLM', 'KTB'), ('ROL', 'SCI'), ('MKC', 'CAG'), ('GIS', 'LWAY'), ('BRBR', 'UTZ'), ('MKC', 'CPB'), ('DINO', 'DK'), ('NDAQ', 'TRU'), ('PEN', 'FNA'), ('QDEL', 'AHCO'), ('INMD', 'FNA'), ('CI', 'HUM'), ('ELV', 'OSCR'), ('IMVT', 'ACAD'), ('RPRX', 'BHVN'), ('RVMD', 'RNA'), ('SMMT', 'PCVX'), ('SMMT', 'ADMA'), ('RVMD', 'CRNX'), ('NUVL', 'ACLX'), ('CORT', 'DNLI'), ('SRPT', 'MRUS'), ('UTHR', 'HALO'), ('LH', 'SHC'), ('ICLR', 'MEDP'), ('ISRG', 'BLFS'), ('RMD', 'BLFS'), ('GD', 'ACHR'), ('GE', 'GD'), ('TDG', 'GD'), ('AOS', 'FELE'), ('DOV', 'FLS'), ('AOS', 'MIR'), ('RRX', 'FELE'), ('DCI', 'GTES'), ('IR', 'FELE'), ('CXT', 'TNC'), ('CSL', 'APOG'), ('CMPR', 'SPIR'), ('CTAS', 'ARMK'), ('GATX', 'CAR'), ('AEIS', 'ENS'), ('HAYW', 'POWL'), ('WERN', 'MRTN'), ('NNN', 'FCPT'), ('ADC', 'KRG'), ('O', 'ROIC'), ('KIM', 'ROIC'), ('AVB', 'CPT'), ('EQR', 'CPT'), ('AMH', 'IRT'), ('AMH', 'UMH'), ('AMH', 'VRE'), ('INVH', 'AIV'), ('AMH', 'CPT'), ('WELL', 'CTRE'), ('EPRT', 'GOOD'), ('RHP', 'SHO'), ('MSFT', 'ALTR'), ('IT', 'BR'), ('MSI', 'BDC'), ('CAMT', 'VECO'), ('ANET', 'STX'), ('DELL', 'WDC'), ('JBL', 'DAKT'), ('FE', 'EVRG'), ('PPL', 'EVRG'), ('PCG', 'NWE'), ('EIX', 'CMS'), ('D', 'PPL'), ('EIX', 'EVRG'), ('FE', 'CMS'), ('EIX', 'FE'), ('PPL', 'CMS'), ('CMS', 'ETR'), ('NWN', 'SPH'), ('NJR', 'BKH'), ('NI', 'NFE'), ('NJR', 'SR')]
    # pairs = [('YUM', 'SHAK'),('EAT', 'CAKE'), ('INMD', 'FNA')]
    # pairs = [('T', 'FYBR'), ('INMD', 'FNA')]
    zs = [
        [0, 0],
        [0, 0],
    ]
    all_recalls = []
    for pair in pairs:
        symbolx = pair[0]
        symboly = pair[1]
        fig, axs = plt.subplots(1, 2, figsize=(10,5))
        for iden, z_enter in enumerate([1,2]):
            # if iden == 1:
            #     break
            trader = SimulatedTrader(fetcher, "2024-06-03 09:30:00")
            strategy = PairsStrategy(symbolx, symboly, trader, buying_power=1000, z_enter=z_enter, z_exit=0.5)
            pairs_driver = PairsDriver(symbolx, symboly,  "2024-10-01 09:30:00", strategy, trader, fetcher)
            error = pairs_driver.simulate()
            if error:
                continue

            bound_date = datetime.strptime("2024-10-01 09:30:00", '%Y-%m-%d %H:%M:%S')
            # selected_date = datetime.strptime(date[i], '%Y-%m-%d %H:%M:%S')
            [all_dates, all_capital] = list(zip(*strategy.record))
            recs = [d for d in strategy.record if datetime.strptime(d[0], '%Y-%m-%d %H:%M:%S') > bound_date]
            recall = {
                "symbolx": symbolx,
                "symboly": symboly,
                "sharpe_ratio": fmt_float(pairs_driver.sharpe_ratio) if pairs_driver.sharpe_ratio != 'NA' else 'NA',
                "max_dd_pct": f"{fmt_float(strategy.max_dd_pct)}%",
                "z_enter#z_exit": f"{z_enter}#{0.5}",
                "profit": "NA",
            }
            if len(recs) == 0:
                all_recalls.append(recall)
                continue
            [_, capital] = list(zip(*recs))
            recall["profit"] = fmt_float(capital[-1] - capital[0])
            all_recalls.append(recall)
            zs[iden][0] += capital[-1] - capital[0]
            zs[iden][1] += 1
            # plt.plot(dates, capital)
            starting_livetest_capital = capital[0]
            axs[iden].axhline(starting_livetest_capital, color='red', linestyle='--')
            axs[iden].plot(all_dates, all_capital)
            axs[iden].set_title(f"z_enter: {z_enter}, gained {capital[-1] - capital[0]}")
        output_img_path = f"./sim-results/x-{symbolx}--y-{symboly}-210-window.png"
        plt.savefig(output_img_path)
        plt.close()
        log.info(f"Saving image to {output_img_path}")
        log.error(f"zs[0] total gained {zs[0][0]} on {zs[0][1]} pairs")
        log.error(f"zs[1] total gained {zs[1][0]} on {zs[1][1]} pairs")
    
    with open('recalls.txt', 'w+') as f:
        for item in all_recalls:
            f.write(str(item) + '\n')

if __name__ == "__main__":
    # Portfolio().find_pairs()
    # print(yf.Sector("basic-materials").top_companies.index)
    # run_ptp()
    with open('recalls.txt', 'r') as f:
        recalls = []
        for line in f:
            recall = eval(line.strip())
            recalls.append(recall)
        
        z_enter = 1
        z_exit = 0.5
        recalls = [r for r in recalls if r["profit"] != "NA" and abs(float(r["profit"])) > 1e-9]
        recalls = [r for r in recalls if r["z_enter#z_exit"] == f"{z_enter}#{z_exit}"]
        recalls = [r for r in recalls if float(r["max_dd_pct"][:-1]) > 1.0 and float(r["max_dd_pct"][:-1]) < 10.0 ]
        # recalls = [r for r in recalls if float(r["sharpe_ratio"]) > 1.0]
        
        all_profits = []
        for r in recalls:
            # r["profit"] = float(r["profit"])
            print(r)
            all_profits.append(float(r["profit"]))
        
        profits_np = np.array(all_profits)
        initial_investment = 1000*len(profits_np)
        returns = profits_np / 1000.0
        total_profit = profits_np.sum()
        returns_std = np.std(returns)
        total_return = total_profit / initial_investment
        sharpe = (total_return - 0.06/4.0)/returns_std
        print(total_profit)
        print(sharpe)
        
        
        # recalls = sorted(recalls, key=lambda x: x["profit"])
        # recalls.reverse()
        # for r in recalls: print(r)
        # print(len(recalls))
        # print(total_profit)

    # main()
