def calculate_position_size(account_balance, risk_percentage, entry_price, stop_loss_price):
    """
    Calculate the position size based on account balance, risk percentage, entry price, and stop loss price.
    
    :param account_balance: Total account balance
    :param risk_percentage: Percentage of account balance to risk on a trade
    :param entry_price: Entry price of the trade
    :param stop_loss_price: Stop loss price of the trade
    :return: Position size
    """
    risk_amount = account_balance * (risk_percentage / 100)
    risk_per_share = abs(entry_price - stop_loss_price)
    position_size = risk_amount / risk_per_share
    return position_size


def calculate_stop_loss(entry_price, atr, multiplier):
    """
    Calculate the stop loss price based on entry price, Average True Range (ATR), and a multiplier.
    
    :param entry_price: Entry price of the trade
    :param atr: Average True Range value
    :param multiplier: Multiplier for the ATR to set the stop loss
    :return: Stop loss price
    """
    stop_loss_price = entry_price - (atr * multiplier)
    return stop_loss_price


def calculate_take_profit(entry_price, atr, multiplier):
    """
    Calculate the take profit price based on entry price, Average True Range (ATR), and a multiplier.
    
    :param entry_price: Entry price of the trade
    :param atr: Average True Range value
    :param multiplier: Multiplier for the ATR to set the take profit
    :return: Take profit price
    """
    take_profit_price = entry_price + (atr * multiplier)
    return take_profit_price