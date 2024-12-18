import os
import sys

from loguru import logger as log

from algo_data import data
from algo_event import event
from algo_model import strategy


def main():
    log.remove()
    log.add(sys.stderr, level="TRACE")
    # me = event.MarketEvent()
    log.info("In main")

    data_folder = os.path.join(".", "algo_data", "historical")
    interval = "1d"
    start = "2023-01-01"
    end = "2025-01-01"
    df = data.DataFetcher(data_folder, interval, start, end)
    df.get_bars("NVDA")


if __name__ == "__main__":
    main()
