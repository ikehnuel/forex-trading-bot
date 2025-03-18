class Trade:
    def __init__(self, entry_price, exit_price, quantity):
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.quantity = quantity
        self.profit_loss = self.calculate_profit_loss()

    def calculate_profit_loss(self):
        return (self.exit_price - self.entry_price) * self.quantity

    def __str__(self):
        return f"Trade(entry_price={self.entry_price}, exit_price={self.exit_price}, quantity={self.quantity}, profit_loss={self.profit_loss})"