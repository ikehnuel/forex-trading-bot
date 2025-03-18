def calculate_sma(prices, period):
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period

def calculate_ema(prices, period):
    if len(prices) < period:
        return None
    multiplier = 2 / (period + 1)
    ema = prices[0]
    for price in prices[1:]:
        ema = (price - ema) * multiplier + ema
    return ema

def calculate_rsi(prices, period):
    if len(prices) < period:
        return None
    gains = []
    losses = []
    for i in range(1, period + 1):
        change = prices[i] - prices[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            losses.append(-change)
            gains.append(0)
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, short_period=12, long_period=26, signal_period=9):
    if len(prices) < long_period:
        return None
    ema_short = calculate_ema(prices, short_period)
    ema_long = calculate_ema(prices, long_period)
    macd = ema_short - ema_long
    signal_line = calculate_ema([macd] * signal_period, signal_period)
    return macd, signal_line

def calculate_bollinger_bands(prices, period, num_std_dev=2):
    if len(prices) < period:
        return None, None
    sma = calculate_sma(prices, period)
    variance = sum((x - sma) ** 2 for x in prices[-period:]) / period
    std_dev = variance ** 0.5
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return upper_band, lower_band