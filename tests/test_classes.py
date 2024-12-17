from algo_event.event import MarketEvent


class TestMarketClass:
    def test_instantiation(self):
        market_event = MarketEvent()
        assert market_event.event_type == "MARKET"
