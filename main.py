import os
import sys
from loguru import logger as log

from data import utils
from market.portfolio import Portfolio, get_industries_tickers
from pipelines import pairs_trading_pipeline as ptp
from model.strategy import PairsStrategy, DSStrategy
from market.drivers import PairsDriver, DSDriver
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

def run_pairs_driver():
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

def process_recalls(recalls_file):
    with open(recalls_file, 'r') as f:
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

def run_dss_driver():
    fetcher = utils.DataFetcher(os.path.join(".", "data", "historical"), "1h", "2024-01-01", "2024-11-01")
    industries = get_industries_tickers()
    initial_capital = 1000
    end_results = []
    count = 1
    running_gains = 0
    total = len(industries)
    num_tickers = 0
    for industry_name, tickers in industries.items():
        log.critical(f"On industry {industry_name}, {count} of {total}")
        log.critical(f"Running gains of {running_gains} on {num_tickers} stocks")
        for ticker in tickers:
            trader = SimulatedTrader(fetcher, "2024-01-02 09:30:00")
            strategy = DSStrategy(ticker, trader, initial_capital, update_interval=16, num_intervals=50)
            ds_driver = DSDriver(ticker, strategy, trader, fetcher)
            
            error = ds_driver.simulate()
            if error: continue
            final_amount = strategy.record[1][1]
            running_gains += final_amount - initial_capital
            end_results.append(final_amount)
            num_tickers += 1
        count += 1
        print(f"running_results: {end_results}")

def sharpe_ratio(arr: np.ndarray):
    end_capitals = np.array([np.float64(1052.3576189747612), np.float64(1176.8386961827725), np.float64(1114.7226694895528), np.float64(1349.6616717787392), np.float64(1144.3370177319348), np.float64(899.9008240395574), np.float64(820.6114949626235), np.float64(1181.726268647786), np.float64(1213.4967770978496), np.float64(840.2209839764671), np.float64(1168.1431933621354), np.float64(796.5768202812783), np.float64(1209.3511001444012), np.float64(1183.3483664423338), np.float64(1129.3428020484369), np.float64(1305.2347243939444), np.float64(1165.6488251644523), np.float64(884.3147951638878), np.float64(878.3325366346444), np.float64(1117.7371412034151), np.float64(1113.6497661310598), np.float64(832.4324529219641), np.float64(567.1983429897978), np.float64(1464.4089752867553), np.float64(721.4442691355925), np.float64(1008.2344135135979), np.float64(2480.3637834431315), np.float64(772.3684466714378), np.float64(967.0247917757251), np.float64(1069.1981753537411), np.float64(478.4785515132037), np.float64(607.6342318138811), np.float64(1173.3113651862652), np.float64(1360.3701788932576), np.float64(1303.1272711193092), np.float64(1833.3880567055053), np.float64(1412.8688010053825), np.float64(1604.65255386935), np.float64(1566.7384222267124), np.float64(1344.8016085759177), np.float64(1187.294030167473), np.float64(1144.4606855897396), np.float64(1422.8146218100796), np.float64(1299.3022797887302), np.float64(1555.227198525251), np.float64(1008.5636778050882), np.float64(1554.4178057011413), np.float64(1161.7536051711834), np.float64(998.6222371664253), np.float64(1248.6536299299069), np.float64(1248.0891987885238), np.float64(1097.3838219336362), np.float64(856.9265954719533), np.float64(1156.5885903635358), np.float64(1508.9057890684066), np.float64(1249.715687311273), np.float64(744.3355424142612), np.float64(1069.7630187877194), np.float64(953.8483111219016), np.float64(965.1986725346892), np.float64(1001.8942937475704), np.float64(644.5569481806649), np.float64(1347.9842719937255), np.float64(808.1374287684796), np.float64(2067.092566420597), np.float64(466.44377946606164), np.float64(825.9490017579938), np.float64(784.2443728831522), np.float64(701.3295236834933), np.float64(885.1971036401435), np.float64(638.4784652318181), np.float64(1137.7925883753453), np.float64(925.7023790857932), np.float64(506.64462361595497), np.float64(1008.6657585390756), np.float64(1940.90094008487), np.float64(1326.0768913292063), np.float64(791.7218987235626), np.float64(857.8436435806507), np.float64(1227.6472741124585), np.float64(984.1271122241953), np.float64(1060.8402270605998), np.float64(1327.9941942781147), np.float64(1470.9052444884733), np.float64(287.78612931212024), np.float64(942.1498196577456), np.float64(1531.2312145765754), np.float64(2670.3403589004934), np.float64(1113.252248560844), np.float64(1051.5541773339978), np.float64(501.08053446566237), np.float64(937.9719976254328), np.float64(1097.735713003809), np.float64(510.9411801272295), np.float64(1613.742308330478), np.float64(379.9430094899667), np.float64(652.2106400888515), np.float64(1133.238309814216), np.float64(1149.2097094032983), np.float64(1318.1683749956687), np.float64(1611.6870488742516), np.float64(1541.1366121851038), np.float64(807.6237710085888), np.float64(992.9514143991103), np.float64(1321.173692948133), np.float64(1100.499974032013), np.float64(1490.2483583372632), np.float64(1062.7566895014527), np.float64(1419.782691075593), np.float64(1140.4889554103752), np.float64(1112.637400273503), np.float64(1603.6896109279646), np.float64(1642.5581309495556), np.float64(5747.6868384335985), np.float64(1653.1638675319027), np.float64(1165.48221404688), np.float64(2043.9047497936522), np.float64(1356.7137439868388), np.float64(1028.942292202208), np.float64(1385.7168639330775), np.float64(1459.4070330992695), np.float64(573.7738342828579), np.float64(1393.0466285918706), np.float64(940.7681593848172), np.float64(1395.8213743063807), np.float64(972.3687523961246), np.float64(1265.132156121197), np.float64(1419.2299709398017), np.float64(1083.0245268713109), np.float64(1009.7202683818937), np.float64(1304.711908670292), np.float64(909.412728770672), np.float64(1216.4623575183396), np.float64(681.6219575059174), np.float64(916.2293653232682), np.float64(1853.260439888284), np.float64(1279.4621936255176), np.float64(1972.8030685953654), np.float64(953.4866357829158), np.float64(729.867629956507), np.float64(1087.3174690334433), np.float64(1740.007075679755), np.float64(725.9850885998117), np.float64(1168.9583197393601), np.float64(1578.9809312938582), np.float64(1117.2429350436464), np.float64(1076.7017575077364), np.float64(1151.443555261772), np.float64(900.8089321523564), np.float64(750.181152330867), np.float64(1171.331845947962), np.float64(918.8209916221772), np.float64(1157.4123367033956), np.float64(1499.992751334276), np.float64(1791.944014473905), np.float64(831.2186805289518), np.float64(1267.768789947033), np.float64(1458.3482361926704), np.float64(1821.1030901844601), np.float64(511.6073037895155), np.float64(1122.2553755028096), np.float64(1246.1022504611046), np.float64(1475.8822184532896), np.float64(1088.0480262731603), np.float64(1350.9714905804653), np.float64(1522.1598782005885), np.float64(1166.7069530024578), np.float64(1511.9083883904113), np.float64(1621.9379837880047), np.float64(1149.9237942538573), np.float64(686.3872201480917), np.float64(622.9421654220159), np.float64(1471.001816922847), np.float64(1309.6646466787215), np.float64(766.1653188507514), np.float64(687.931087525681), np.float64(1098.6417171535934), np.float64(1231.4653716372814), np.float64(1004.1794670466741), np.float64(965.248289395917), np.float64(1058.7005341167032), np.float64(1369.5137542047505), np.float64(852.3045514926785), np.float64(2494.163672976793), np.float64(1412.74400824958), np.float64(651.9086683760947), np.float64(1035.9974826443201), np.float64(2481.163934558659), np.float64(1558.5352727792156), np.float64(2247.050796835434), np.float64(1112.5975840053213), np.float64(1464.728845561035), np.float64(751.437316085736), np.float64(934.2431758300158), np.float64(1015.1086638747097), np.float64(860.4600359918957), np.float64(838.1568167174603), np.float64(1396.6708954189398), np.float64(606.1887585461584), np.float64(1189.8347652239338), np.float64(766.7786148566081), np.float64(1170.6367502983665), np.float64(1207.5745987205405), np.float64(897.4654150274741), np.float64(339.0128882481765), np.float64(1459.5607002359616), np.float64(916.8587592413753), np.float64(1771.2667439025245), np.float64(1616.314036207154), np.float64(1323.6148208471736), np.float64(1707.6168342507503), np.float64(1707.3764853340208), np.float64(1172.0946664716562), np.float64(728.5785297624298), np.float64(1486.3471375812928), np.float64(1148.5509731311586), np.float64(1145.4347142377003), np.float64(1033.1989668246663), np.float64(1003.928148196391), np.float64(791.0706716783325), np.float64(1181.9486427816487), np.float64(1017.1389641626647), np.float64(1332.5268039431294), np.float64(1502.426557390474), np.float64(1214.8202246207925), np.float64(514.8167194500359), np.float64(1310.3032731743697), np.float64(650.6061055769671), np.float64(1039.1415633088143), np.float64(778.3282968688388), np.float64(295.85823407940416), np.float64(613.8266544998082), np.float64(1195.39036183578), np.float64(489.0859305943641), np.float64(1141.321130053407), np.float64(599.2579100145329), np.float64(883.8741851609443), np.float64(1764.0576365508646), np.float64(814.9337334804582), np.float64(1155.76922563589), np.float64(963.8484621522377), np.float64(828.9448472662391), np.float64(1222.9656280627864), np.float64(983.3695529368422), np.float64(803.3484532615537), np.float64(1263.9659690880226), np.float64(1787.523737314399), np.float64(838.4144441470282), np.float64(1155.0710091967326), np.float64(814.6228983241244), np.float64(611.4615214616174), np.float64(979.1599703637369), np.float64(730.4735400104848), np.float64(527.3056823969721), np.float64(872.7189796353994), np.float64(728.0539118599995), np.float64(1142.1971984962377), np.float64(749.161788434541), np.float64(1175.2005927994876), np.float64(1545.9997003944063), np.float64(857.2352621172074), np.float64(927.6551509707112), np.float64(704.7904508094391), np.float64(1085.8563296389054), np.float64(1615.177314746023), np.float64(367.0919543613229), np.float64(1125.0641820358244), np.float64(1091.1643122801909), np.float64(1265.6001411850789), np.float64(1197.5243483925042), np.float64(1361.853800992331), np.float64(1296.7193991899912), np.float64(1090.9536678890393), np.float64(1225.772077282676), np.float64(975.9235607011044), np.float64(1215.0911008316016), np.float64(1234.0515041742656), np.float64(1188.3618753286444), np.float64(1094.2782650950578), np.float64(1232.2159531757588), np.float64(884.2870056936933), np.float64(924.0870271206113), np.float64(1049.892893423736), np.float64(1141.017216742055)])
    initial_investment = 1000 * len(end_capitals)
    profits = end_capitals - 1000
    returns = profits / 1000
    total_profit = profits.sum()
    returns_std = np.std(returns)
    total_return = total_profit / initial_investment
    sharpe = (total_return - 0.06)/returns_std
    print(sharpe)
    print(len(end_capitals))
    print(total_return)
    print(returns_std)


def main():
    # Portfolio().find_pairs()
    # print(yf.Sector("basic-materials").top_companies.index)
    # process_recalls('recalls.txt')
    # run_ptp()
    # main()
    # run_dss_driver()
    # for i in range(len(profits)): print(profits[i])
    spy = yf.Ticker("SPY")
    print(spy.history_metadata)
    print("###")
    print(spy.get_history_metadata())
    return
    option_chain = spy.option_chain(date="2024-10-01")
    # print(option_chain)
    calls = option_chain.calls
    filtered_calls = calls[["contractSymbol", "strike", "lastPrice", "bid", "ask", "change", "percentChange", "volume", "openInterest", "impliedVolatility"]]
    print(filtered_calls[filtered_calls["strike"] == 600]) #and options["strike"] >= 590])
    # print(option_chain._fields)

    
    # spy.option(interval="1m", start="2024-10-01", end="2024-11-01")



if __name__ == "__main__":
    log.remove()
    log.add(sys.stderr, level="WARNING")
    main()
