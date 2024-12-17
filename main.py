from loguru import logger
from algo_data import data
from algo_event import event


def main():
    me = event.MarketEvent()
    logger.info("In main")

if __name__=="__main__":
    main()
