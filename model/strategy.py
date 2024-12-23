from loguru import logger as log
from market.traders import SimulatedTrader

def fmt(num):
    return "{:.2f}".format(num)

class PairsStrategy(object):
    """
    The strategy to employ
    """
    
    def __init__(self, symbol1, symbol2, trader: SimulatedTrader, buying_power, z_enter=2, z_exit=0.5):
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.trader = trader
        self.buying_power = buying_power
        self.short_bank = {
            symbol1: 0,
            symbol2: 0
        }
        self.capital_per_trade = buying_power
        self.z_enter = z_enter
        self.z_exit = z_exit
        self.has_position = False
        self.record = []

    def _enter(self, coeff_stock1, coeff_stock2):
        assert not self.has_position
        price1 = self.trader.get_price(self.symbol1)
        price2 = self.trader.get_price(self.symbol2)
        multiple = abs(coeff_stock1) * price1 + abs(coeff_stock2) * price2
        num_multiples = self.capital_per_trade / multiple
        quantity_stock1 = num_multiples * coeff_stock1
        quantity_stock2 = num_multiples * coeff_stock2

        action1 = "Buy" if quantity_stock1 > 0 else "Sell Short"
        action2 = "Buy" if quantity_stock2 > 0 else "Sell Short"

        log.trace(f"Entering {self.symbol1}")
        cost1 = self.trader.trade(self.symbol1, action1, abs(quantity_stock1))
        log.trace(f"Entering {self.symbol2}")
        cost2 = self.trader.trade(self.symbol2, action2, abs(quantity_stock2))
        self.buying_power -= cost1 + cost2

        self.short_bank[self.symbol1] += cost1 if action1 == "Sell Short" else 0
        self.short_bank[self.symbol2] += cost2 if action2 == "Sell Short" else 0
        log.info(f"Buying power remaining: {fmt(self.buying_power)}")
        self.has_position = True

    def _exit(self):
        assert self.has_position
        for symbol in [self.symbol1, self.symbol2]:
            pos = self.trader.positions.get(symbol, 0)
            assert abs(pos) > 0
            action = "Buy to Cover" if pos < 0 else "Sell"
            log.trace(f"Exiting {symbol}")
            cost = self.trader.trade(symbol, action, abs(pos))
            power = cost

            if action == "Buy to Cover":
                power = 2 * self.short_bank[symbol] - cost
                self.short_bank[symbol] = 0

            self.buying_power += power

        log.info(f"Buying power: {fmt(self.buying_power)}")
        self.has_position = False


    def update(self, z_score, coeff_stock1, coeff_stock2):
        if self.has_position:
            if abs(z_score) < self.z_exit:
                self._exit()
                self.record.append([self.trader.current_day, self.buying_power])
        else:
            if z_score > self.z_enter:
                self.record.append([self.trader.current_day, self.buying_power])
                self._enter(-coeff_stock1, -coeff_stock2)
            elif z_score < -self.z_enter:
                self.record.append([self.trader.current_day, self.buying_power])
                self._enter(coeff_stock1, coeff_stock2)
        
