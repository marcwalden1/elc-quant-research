# Standard library imports
import datetime as dt
import logging

# Third-party library imports
import numpy as np
import pandas as pd
import pytz


# --- Global Vars ---
NANOSECONDS_1M = 60 * 1_000_000_000
NANOSECONDS_15M = 15 * 60 * 1_000_000_000
NANOSECONDS_1H = 60 * 60 * 1_000_000_000



def nanoseconds_to_datetime64(ns_int):
    """
    Converts a nanosecond integer to a NumPy datetime64 object.

    Args:
        ns_int: An integer representing nanoseconds since the Unix epoch (1970-01-01T00:00:00).

    Returns:
        A NumPy datetime64 object representing the corresponding timestamp.
    """
    # Create a datetime64 object with nanosecond precision
    dt64 = np.datetime64(0, 'ns') + ns_int
    return dt64


def nanoseconds_to_new_york_datetime64(ns_int):
    """
    Converts a nanosecond integer (UTC) to a NumPy datetime64 object in New York time,
    optimized for speed.

    Args:
        ns_int: An integer representing nanoseconds since the Unix epoch (1970-01-01T00:00:00) in UTC.

    Returns:
        A NumPy datetime64 object representing the corresponding timestamp in New York time.
    """

    # 1. Create a datetime64 object with nanosecond precision (UTC)
    dt64_utc = np.datetime64(0, 'ns') + ns_int

    # 2. Convert to pandas Timestamp for timezone conversion
    ts_utc = pd.Timestamp(dt64_utc, tz='UTC')  # Directly create a UTC Timestamp

    # 3. Convert to New York time
    ts_ny = ts_utc.tz_convert('America/New_York')

    # 4. Convert back to NumPy datetime64
    dt64_ny = np.datetime64(ts_ny)

    return dt64_ny


def convert_nanoseconds_to_ny_datetime(nanoseconds: int):
    """
    Converts a nanosecond integer to a datetime object in New York time.

    Args:
        nanoseconds: An integer representing nanoseconds since the Unix epoch.

    Returns:
        A datetime object representing the corresponding New York time.
    """
    # Convert nanoseconds to seconds
    seconds = nanoseconds / 1e9

    # Create a timezone-aware datetime object from the timestamp (UTC)
    utc_datetime = dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc)

    # Define the New York timezone
    ny_timezone = pytz.timezone("America/New_York")

    # Convert to New York time
    ny_datetime = utc_datetime.astimezone(ny_timezone)

    return ny_datetime

def floor_to_hour_ns(ns_int):
    """
    Floors a nanosecond integer to the start of the minute, maintaining the nanosecond integer format.

    Args:
        ns_int: An integer representing nanoseconds since the Unix epoch (1970-01-01T00:00:00).

    Returns:
        An integer representing the nanoseconds at the start of the hour.
    """
    # Number of nanoseconds in an hour
    ns_per_hour = 60 * 60 * 1_000_000_000
    return (ns_int // ns_per_hour) * ns_per_hour

def floor_to_15minute_ns(ns_int):
    """
    Floors a nanosecond integer to the start of the minute, maintaining the nanosecond integer format.

    Args:
        ns_int: An integer representing nanoseconds since the Unix epoch (1970-01-01T00:00:00).

    Returns:
        An integer representing the nanoseconds at the start of the hour.
    """
    # Number of nanoseconds in an hour
    ns_per_15m = 15 * 60 * 1_000_000_000
    return (ns_int // ns_per_15m) * ns_per_15m


def floor_to_minute_ns(ns_int):
    """
    Floors a nanosecond integer to the start of the minute, maintaining the nanosecond integer format.

    Args:
        ns_int: An integer representing nanoseconds since the Unix epoch (1970-01-01T00:00:00).

    Returns:
        An integer representing the nanoseconds at the start of the minute.
    """
    # Number of nanoseconds in a minute
    ns_per_minute = 60 * 1_000_000_000
    return (ns_int // ns_per_minute) * ns_per_minute

def contract_size(stop_loss_size, risk=1000, asset = 'ES'):
    if asset == 'ES':
        number_of_contracts = round(risk/(50 * stop_loss_size))
    elif asset == 'NQ':
        number_of_contracts = round(risk/(20 * stop_loss_size))
    return number_of_contracts


def update_olhc_data(ohlc_data, trades, last_candle_time_ns, timeframe):
    
    """Append a new row to the olhc data."""
    # Step 1: Get all trades
    # trades = processor.get_raw_data()
    
    if last_candle_time_ns is None: 
        trades_in_current_interval = trades
    elif timeframe == '1h':
        start_time_ns = floor_to_hour_ns(last_candle_time_ns)
        trades_in_current_interval = [trade for trade in trades if (trade['ts_event'] >= start_time_ns and trade['ts_event'] < (start_time_ns + NANOSECONDS_1H))]
    elif timeframe == '15m':
        start_time_ns = floor_to_15minute_ns(last_candle_time_ns)
        trades_in_current_interval = [trade for trade in trades if (trade['ts_event'] >= start_time_ns and trade['ts_event'] < (start_time_ns + NANOSECONDS_15M))]
    elif timeframe == '1m':
        start_time_ns = floor_to_minute_ns(last_candle_time_ns)
        trades_in_current_interval = [trade for trade in trades if (trade['ts_event'] >= start_time_ns and trade['ts_event'] < (start_time_ns + NANOSECONDS_1M))]
        
    else:
        raise ValueError(f"Unknown timeframe: {timeframe}")

    # Step 3: Return if no trades
    if len(trades_in_current_interval) == 0:
        return ohlc_data


    # Step 4: Compute OHLC values
    open_price = trades_in_current_interval[0]['price']
    high_price = max(trade['price'] for trade in trades_in_current_interval)
    low_price = min(trade['price'] for trade in trades_in_current_interval)
    close_price = trades_in_current_interval[-1]['price']
    total_volume = sum(trade['size'] for trade in trades_in_current_interval)
    # print(total_volume)

    # Step 5: Construct and append new OHLC row
    ohlc_row = {
        'time(ns)': last_candle_time_ns,
        'time': pd.to_datetime(last_candle_time_ns).tz_localize('UTC').tz_convert('America/New_York'),  # keep as datetime, not string
        'open': open_price,
        'high': high_price,
        'low': low_price,
        'close': close_price,
        'volume': total_volume
    }

    ohlc_data = pd.concat([ohlc_data, pd.DataFrame([ohlc_row])], ignore_index=True)
    return ohlc_data, ohlc_row


def find_HTF_POI(ohlc_data, inactive_pois, timeframe):
    # Get the last 3 rows (candles) from the ohlc_data_1h DataFrame
            if timeframe == '1h':
                if len(ohlc_data) >= 3:
                    candle_1 = ohlc_data.iloc[-3]  # Third last row
                    candle_2 = ohlc_data.iloc[-2]  # Second last row
                    candle_3 = ohlc_data.iloc[-1]  # Last row (most recent candle)
                else:
                    # Handle the case when there are fewer than 3 candles
                    candle_1 = candle_2 = candle_3 = None
                    logging.warning("Not enough candles in ohlc_data_1h to retrieve the last 3.")
                    return
                # import pdb;pdb.set_trace()
            elif timeframe == '15m':
                if len(ohlc_data) >= 3:
                    candle_1 = ohlc_data.iloc[-3]  # Third last row
                    candle_2 = ohlc_data.iloc[-2]  # Second last row
                    candle_3 = ohlc_data.iloc[-1]  # Last row (most recent candle)
                else:
                    # Handle the case when there are fewer than 3 candles
                    candle_1 = candle_2 = candle_3 = None
                    logging.warning("Not enough candles in ohlc_data_15m to retrieve the last 3.")
                    return
            
            
            # print(candle_1, candle_2, candle_3)

            if ((candle_1['high'] < candle_3['low']) and (candle_2['close'] > candle_2['open'])):
                
                candle_2_time = pd.to_datetime(candle_2['time(ns)']).tz_localize('UTC').tz_convert('America/New_York')

                inactive_pois.append({
                    'gap_direction': 'bullish',
                    'gap_low': float(candle_1['high']),
                    'gap_high': float(candle_3['low']),
                    'time': candle_2_time,
                    'timeframe': timeframe,
                    'active': False,
                    'erl_invalidation_price': None,
                    'price_ran_too_far': False
                })
                # if timeframe == '1h':
                #     import pdb;pdb.set_trace()

            # Check if candles 1 and 3 do not overlap and candle 2 is bearish
            if (candle_1['low'] > candle_3['high'] and (candle_2['close'] < candle_2['open'])):
                
                candle_2_time = pd.to_datetime(candle_2['time(ns)']).tz_localize('UTC').tz_convert('America/New_York')


                inactive_pois.append({
                    'gap_direction': 'bearish',
                    'gap_low': float(candle_3['high']),
                    'gap_high': float(candle_1['low']),
                    'time': candle_2_time,
                    'timeframe': timeframe,
                    'active': False,
                    'erl_invalidation_price': None,
                    'price_ran_too_far': False
                })



def check_if_in_poi(ohlc_data_1m, inactive_pois, can_run = 3):
    # Omitted for confidentiality purposes
    pass

def POI_invalidation_1(active_pois, most_recent_candle, timeframe):
    # Omitted for confidentiality purposes
    pass


def POI_invalidation_2(inactive_pois, active_pois, current_ts_event, timeframe, candles=15):
    # Omitted for confidentiality purposes
    pass
    

def POI_invalidation_3(active_pois, most_recent_candle_1m):
    # Omitted for confidentiality purposes
    pass

def identify_inverse_momentum_gap(ohlc_data_1m, potential_ifvgs, recent_bullish_momentum_gap, recent_bearish_momentum_gap, timeframe = '1m', extra_sl_margin=0.25):
    # Get the last 3 rows (candles) from the ohlc_data_1h DataFrame
    if len(ohlc_data_1m) >= 3:
        candle_1 = ohlc_data_1m.iloc[-3]  # Third last row
        candle_2 = ohlc_data_1m.iloc[-2]  # Second last row
        candle_3 = ohlc_data_1m.iloc[-1]  # Last row (most recent candle)
    else:
        # Handle the case when there are fewer than 3 candles
        candle_1 = candle_2 = candle_3 = None
        logging.info("Not enough candles in ohlc_data_1m to retrieve the last 3.")
        return potential_ifvgs, recent_bullish_momentum_gap, recent_bearish_momentum_gap

    # For bullish gaps (which will become bearish ifvgs)
    if not recent_bullish_momentum_gap:
        if ((candle_1['high'] < candle_3['low']) and (candle_2['close'] > candle_2['open'])):
                    
            candle_2_time = pd.to_datetime(candle_2['time(ns)']).tz_localize('UTC').tz_convert('America/New_York')
            date = pd.to_datetime(candle_2['time(ns)']).tz_localize('UTC').tz_convert('America/New_York').date()

            potential_ifvgs.append({
                'gap_direction': 'bullish',
                'gap_low': float(candle_1['high']),
                'gap_high': float(candle_3['low']),
                'fvg_time': candle_2_time,
                'date': date,
                'timeframe': timeframe,
                'proposed_sl': float(candle_3['low']) + extra_sl_margin,
                'inverted': False,
                'inverted_at_time': None,
                'inverted_at_price': None,
                'inverted_now': False,
                'already_used': False,
                'close_x_far': None
            })
            logging.info("New 1m bullish fvg detected.")
            recent_bullish_momentum_gap = True

    else:
        if ((candle_1['high'] < candle_3['low']) and (candle_2['close'] > candle_2['open'])):
            potential_ifvgs[-1]['gap_high'] = float(candle_3['low'])
            potential_ifvgs[-1]['proposed_sl'] = float(candle_3['low']) + extra_sl_margin
        else:
            recent_bullish_momentum_gap = False



    # For bearish gaps (which will become bullish ifvgs)
    if not recent_bearish_momentum_gap:
        if (candle_1['low'] > candle_3['high'] and (candle_2['close'] < candle_2['open'])):
                    
            candle_2_time = pd.to_datetime(candle_2['time(ns)']).tz_localize('UTC').tz_convert('America/New_York')
            date = pd.to_datetime(candle_2['time(ns)']).tz_localize('UTC').tz_convert('America/New_York').date()

            potential_ifvgs.append({
                'gap_direction': 'bearish',
                'gap_low': float(candle_3['high']),
                'gap_high': float(candle_1['low']),
                'fvg_time': candle_2_time,
                'date': date,
                'timeframe': timeframe,
                'proposed_sl': float(candle_3['high']) - extra_sl_margin,
                'inverted': False,
                'inverted_at_time': None,
                'inverted_at_price': None,
                'inverted_now': False,
                'already_used': False,
                'close_x_far': None
            })
            logging.info("New 1m bearish fvg detected.")
            recent_bearish_momentum_gap = True

    else:
        if (candle_1['low'] > candle_3['high'] and (candle_2['close'] < candle_2['open'])):
            potential_ifvgs[-1]['gap_low'] = float(candle_3['high'])
            potential_ifvgs[-1]['proposed_sl'] = float(candle_3['high']) - extra_sl_margin
        else:
            recent_bearish_momentum_gap = False

    # Finally, we check whether the already existing fvgs have been inverted or not. 
    for gap in potential_ifvgs:
        if (gap['gap_direction'] == "bullish") and (candle_3['close'] < gap['gap_low']) and (not gap['inverted']):
            gap['inverted'] = True
            gap['inverted_now'] = True
            gap['inverted_at_price'] = int(candle_3['close'])
            gap['inverted_at_time'] = pd.to_datetime(candle_3['time(ns)']).tz_localize('UTC').tz_convert('America/New_York') + pd.Timedelta(nanoseconds=NANOSECONDS_1M)

        elif (gap['gap_direction'] == "bearish") and (candle_3['close'] > gap['gap_high']) and (not gap['inverted']):
            gap['inverted'] = True
            gap['inverted_now'] = True
            gap['inverted_at_price'] = int(candle_3['close'])
            gap['inverted_at_time'] = pd.to_datetime(candle_3['time(ns)']).tz_localize('UTC').tz_convert('America/New_York') + pd.Timedelta(nanoseconds=NANOSECONDS_1M)

    
    return potential_ifvgs, recent_bullish_momentum_gap, recent_bearish_momentum_gap


def validate_inverse_momentum_gap(inverse_momentum_gap, ohlc_data_1m, close_too_far = 20, minimum_size = 4, max_candles_btwn = 15):
    "Invalidates an inverse momentum gap based on whether it satisfies 5 conditions."
    # Omitted for confidentiality purposes
    pass

def update_mean_delta(running_sum, recent_deltas, new_delta, window):    
    # Add the new delta to the running sum
    running_sum += new_delta
    
    # If the window exceeds 400, remove the oldest delta and subtract it from the running sum
    if len(recent_deltas) == window:
        oldest_delta = recent_deltas.pop(0)  # Remove the oldest delta
        running_sum -= oldest_delta  # Subtract it from the running sum
    
    # Add the new delta to the list of recent deltas
    recent_deltas.append(new_delta)
    
    # Compute the updated mean delta
    mean_delta = running_sum / len(recent_deltas)  # This is efficient in O(1)
    
    return running_sum, recent_deltas, mean_delta