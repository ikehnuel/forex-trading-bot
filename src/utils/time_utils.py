from datetime import datetime, timedelta
import pytz

def get_current_forex_session() -> str:
    """
    Get the current forex trading session
    
    Returns:
        str: Current session (Sydney, Tokyo, London, New York)
    """
    # Get UTC time
    utc_now = datetime.now(pytz.UTC)
    
    # Define sessions in UTC
    sydney_open = 20  # 8 PM UTC
    tokyo_open = 23   # 11 PM UTC
    london_open = 7   # 7 AM UTC
    newyork_open = 12 # 12 PM UTC
    
    # Get current hour
    hour = utc_now.hour
    
    # Determine current session
    if sydney_open <= hour < tokyo_open:
        return "Sydney"
    elif tokyo_open <= hour < london_open or hour < 3:  # Tokyo spans midnight
        return "Tokyo"
    elif london_open <= hour < newyork_open:
        return "London"
    elif newyork_open <= hour < sydney_open:
        return "New York"
    else:
        return "Overlap"

def is_weekend() -> bool:
    """
    Check if current time is during the forex weekend
    
    Returns:
        bool: True if it's the weekend
    """
    utc_now = datetime.now(pytz.UTC)
    
    # Weekend is from Friday 22:00 UTC to Sunday 22:00 UTC
    if utc_now.weekday() == 4 and utc_now.hour >= 22:  # Friday after 22:00
        return True
    elif utc_now.weekday() == 5:  # Saturday
        return True
    elif utc_now.weekday() == 6 and utc_now.hour < 22:  # Sunday before 22:00
        return True
    
    return False

def next_market_open() -> datetime:
    """
    Get the next market open time
    
    Returns:
        datetime: Next market open time
    """
    utc_now = datetime.now(pytz.UTC)
    
    if is_weekend():
        # If it's weekend, next open is Sunday 22:00 UTC
        days_to_sunday = 6 - utc_now.weekday() if utc_now.weekday() < 6 else 0
        next_open = utc_now.replace(hour=22, minute=0, second=0, microsecond=0) + timedelta(days=days_to_sunday)
        
        # If we're already past Sunday 22:00, add a week
        if next_open < utc_now:
            next_open += timedelta(days=7)
    else:
        # If it's during the week, market is already open
        next_open = utc_now
        
    return next_open

def format_elapsed_time(seconds: float) -> str:
    """
    Format seconds into a human-readable elapsed time string
    
    Args:
        seconds: Number of seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"