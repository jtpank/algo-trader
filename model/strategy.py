from loguru import logger as log
from market.traders import SimulatedTrader

class PairsStrategy(object):
    """
    The strategy to employ
    """
    
    def __init__(self, symbol1, symbol2, trader: SimulatedTrader, allotted_capital, z_enter=2, z_exit=0.5):
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.trader = trader
        self.allotted_capital = allotted_capital
        self.z_enter = z_enter
        self.z_exit = z_exit
        self.has_position = False

    def _enter(self, coeff_stock1, coeff_stock2):
        assert not self.has_position
        price1 = self.trader.get_price(self.symbol1)
        price2 = self.trader.get_price(self.symbol2)
        multiple = abs(coeff_stock1) * price1 + abs(coeff_stock2) * price2
        num_multiples = self.allotted_capital / multiple
        quantity_stock1 = num_multiples * coeff_stock1
        quantity_stock2 = num_multiples * coeff_stock2

        action1 = "Buy" if quantity_stock1 > 0 else "Sell Short"
        action2 = "Buy" if quantity_stock2 > 0 else "Sell Short"
        cost1 = self.trader.trade(self.symbol1, action1, abs(quantity_stock1))
        cost2 = self.trader.trade(self.symbol2, action2, abs(quantity_stock2))
        self.allotted_capital -= cost1 + cost2

        log.trace(f"Entered {quantity_stock1} of {self.symbol1} costing {cost1}")
        log.trace(f"Entered {quantity_stock2} of {self.symbol2} costing {cost2}")
        log.trace(f"Capital remaining: {self.allotted_capital}")

    def _exit(self):
        assert self.has_position
        for symbol in [self.symbol1, self.symbol2]:
            pos = self.trader.positions.get(symbol, 0)
            assert pos > 0
            action = "Buy to Cover" if pos < 0 else "Sell"
            cost = self.trader.trade(symbol, action, abs(pos))
            self.allotted_capital += cost
            log.trace(f"Exited {pos} of {symbol} costing {cost}")

        log.trace(f"Capital total: {self.allotted_capital}")


    def update(self, z_score, coeff_stock1, coeff_stock2):
        if self.has_position:
            if abs(z_score) < self.z_exit:
                self._exit()
        else:
            if z_score > self.z_enter:
                self._enter(-coeff_stock1, -coeff_stock2)
            elif z_score < -self.z_enter:
                self._enter(coeff_stock1, coeff_stock2)
        
