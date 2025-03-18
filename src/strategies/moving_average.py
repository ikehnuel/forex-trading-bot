class MovingAverageStrategy:
    def __init__(self, short_window, long_window):
        self.short_window = short_window
        self.long_window = long_window
        self.prices = []

    def add_price(self, price):
        self.prices.append(price)
        if len(self.prices) > self.long_window:
            self.prices.pop(0)

    def calculate_short_moving_average(self):
        if len(self.prices) < self.short_window:
            return None
        return sum(self.prices[-self.short_window:]) / self.short_window

    def calculate_long_moving_average(self):
        if len(self.prices) < self.long_window:
            return None
        return sum(self.prices[-self.long_window:]) / self.long_window

    def generate_signal(self):
        short_ma = self.calculate_short_moving_average()
        long_ma = self.calculate_long_moving_average()

        if short_ma is None or long_ma is None:
            return None
        if short_ma > long_ma:
            return "buy"
        elif short_ma < long_ma:
            return "sell"
        else:
            return "hold"