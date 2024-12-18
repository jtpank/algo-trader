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
    df = data.DataFetcher(os.path.join(".", "algo_data", "historical"))
    df.get_bars("NVDA", start="2024-01-01")


if __name__ == "__main__":
    main()
