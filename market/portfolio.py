
from pipelines import pairs_trading_pipeline as ptp
from data import utils
from loguru import logger as log
import os
import sys
import yfinance as yf
import pickle
from typing import Dict

ALL_SECTORS = [
    "basic-materials",
    "communication-services",
    "consumer-cyclical",
    "consumer-defensive",
    "energy",
    "financial-services",
    "healthcare",
    "industrials",
    "real-estate",
    "technology",
    "utilities",
]
MIN_MARKET_WEIGHT = 0.001

class Portfolio(object):
    
    def __init__(self):
        folder = os.path.join(".", "data", "historical")
        self.fetcher = utils.DataFetcher(folder, "1h", "2023-10-01", None)
        self.industries = dict()
        self.cointegrated_pairs = []
    
    def _retrieve_api_top_companies(self):
        industries = dict()
        for sector_name in ALL_SECTORS:
            sector = yf.Sector(sector_name)
            industry_names = sector.industries.index.to_list()
            for industry_name in industry_names:
                log.trace(f"Getting tickers for industry {industry_name}")
                industry = yf.Industry(industry_name)
                tc_df = industry.top_companies
                if tc_df is None: continue
                tc = tc_df.loc[tc_df['market weight'] > MIN_MARKET_WEIGHT]
                top_tickers = tc.index.to_list()
                industries[industry_name] = top_tickers
        
        return industries

    def _retrieve_local_top_companies(self) -> Dict[str, list[str]]:
        with open(os.path.join(".", "market", "industries.pkl"), "rb") as f:
            d = pickle.load(f)
        return d
    
    def _download_top_companies(self):
        for symbol, top_tickers in self.industries.items():
            log.trace(f"Downloading industry {symbol}")
            self.fetcher.bulk_download(top_tickers)

    def find_pairs(self):
        self.industries = self._retrieve_local_top_companies()
        self._download_top_companies()
        data_dict = dict()

        all_pairs = []
        for industry, top_tickers in self.industries.items():
            log.trace(f"Industry: {industry}")
            data_dict = dict()
            for symbol in top_tickers:
                log.info(f"{symbol}")
                df = self.fetcher.get_bars(symbol)
                if len(df[df.isna().any(axis=1)]):
                    log.warning(f"Symbol {symbol} has na...skipping")
                    continue
                elif "2024-10-01 09:30:00" not in df.index.to_list():
                    log.warning(f"Not right cutoff for {symbol}...skipping")
                    continue
                df = df.loc[:"2024-10-01 09:30:00"]
                data_dict[symbol] = df.to_numpy()
            
            if len(data_dict) == 0: continue
                
            my_ptp = ptp.PairsTradingPipeline(data_dict)
            pairs = my_ptp.run()
            if len(pairs) == 0:
                continue
            
            all_pairs += pairs
            print(all_pairs)
        
        print(all_pairs)
        return all_pairs

# pairs = [('T', 'FYBR'), ('YUM', 'SHAK'), ('EAT', 'CAKE'), ('PAG', 'CARG'), ('CVNA', 'AN'), ('PAG', 'SAH'), ('HGV', 'PLYA'), ('CZR', 'GDEN'), ('COLM', 'KTB'), ('ROL', 'SCI'), ('MKC', 'CAG'), ('GIS', 'LWAY'), ('BRBR', 'UTZ'), ('MKC', 'CPB'), ('DINO', 'DK'), ('NDAQ', 'TRU'), ('PEN', 'FNA'), ('QDEL', 'AHCO'), ('INMD', 'FNA'), ('CI', 'HUM'), ('ELV', 'OSCR'), ('IMVT', 'ACAD'), ('RPRX', 'BHVN'), ('RVMD', 'RNA'), ('SMMT', 'PCVX'), ('SMMT', 'ADMA'), ('RVMD', 'CRNX'), ('NUVL', 'ACLX'), ('CORT', 'DNLI'), ('SRPT', 'MRUS'), ('UTHR', 'HALO'), ('LH', 'SHC'), ('ICLR', 'MEDP'), ('ISRG', 'BLFS'), ('RMD', 'BLFS'), ('GD', 'ACHR'), ('GE', 'GD'), ('TDG', 'GD'), ('AOS', 'FELE'), ('DOV', 'FLS'), ('AOS', 'MIR'), ('RRX', 'FELE'), ('DCI', 'GTES'), ('IR', 'FELE'), ('CXT', 'TNC'), ('CSL', 'APOG'), ('CMPR', 'SPIR'), ('CTAS', 'ARMK'), ('GATX', 'CAR'), ('AEIS', 'ENS'), ('HAYW', 'POWL'), ('WERN', 'MRTN'), ('NNN', 'FCPT'), ('ADC', 'KRG'), ('O', 'ROIC'), ('KIM', 'ROIC'), ('AVB', 'CPT'), ('EQR', 'CPT'), ('AMH', 'IRT'), ('AMH', 'UMH'), ('AMH', 'VRE'), ('INVH', 'AIV'), ('AMH', 'CPT'), ('WELL', 'CTRE'), ('EPRT', 'GOOD'), ('RHP', 'SHO'), ('MSFT', 'ALTR'), ('IT', 'BR'), ('MSI', 'BDC'), ('CAMT', 'VECO'), ('ANET', 'STX'), ('DELL', 'WDC'), ('JBL', 'DAKT'), ('FE', 'EVRG'), ('PPL', 'EVRG'), ('PCG', 'NWE'), ('EIX', 'CMS'), ('D', 'PPL'), ('EIX', 'EVRG'), ('FE', 'CMS'), ('EIX', 'FE'), ('PPL', 'CMS'), ('CMS', 'ETR'), ('NWN', 'SPH'), ('NJR', 'BKH'), ('NI', 'NFE'), ('NJR', 'SR')]