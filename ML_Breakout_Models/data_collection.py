import pandas as pd
from datetime import datetime, date, timedelta
import warnings
from utils import *



def identify_consolidation(olhcv_data, window, range_threshold, filter_start_time, filter_end_time, date):
    consolidations = []
    currently_consolidating = False
    range_low = None
    range_high = None
    start_time = None
    range_size = None
    close_prices = []
    volumes = []
    
    filter_start_time = pd.to_datetime(filter_start_time, format='%H:%M:%S').time()
    filter_end_time = pd.to_datetime(filter_end_time, format='%H:%M:%S').time()

    # Convert 'time' column to datetime if it's not already, and then extract the time part
    olhcv_data['time'] = pd.to_datetime(olhcv_data['time'], errors='coerce')  # Handle invalid formats with 'coerce'
    olhcv_data['time_only'] = olhcv_data['time'].dt.time  # Extract time part

    # Filter the data to only include rows within the specified time range
    olhcv_data = olhcv_data[(olhcv_data['time_only'] >= filter_start_time) & (olhcv_data['time_only'] <= filter_end_time)].copy()
    # print(olhcv_data.head())


    olhcv_data['in_consolidation'] = False
    # print(olhcv_data)
    for i in range(window, len(olhcv_data)):  # Start after the first min_window candles
        # If we are currently consolidating, check if we are still inside the range
        if currently_consolidating:
            if olhcv_data['high'].iloc[i] <= range_high and olhcv_data['low'].iloc[i] >= range_low:
                # Candle still in consolidation range, continue
                olhcv_data.iloc[i, olhcv_data.columns.get_loc('in_consolidation')] = True
                close_prices.append(olhcv_data['close'].iloc[i])
                volumes.append(olhcv_data['volume'].iloc[i])
                continue
            else:
                # Breakout from consolidation, finalize the consolidation period
                currently_consolidating = False
                std_dev = pd.Series(close_prices).std()
                average_volume = sum(volumes)/len(volumes)
                consolidations.append({
                    'date': date,
                    'start_time': start_time,
                    'end_time': olhcv_data['time'].iloc[i],
                    'range_high': range_high,
                    'range_low': range_low,
                    'consolidation_range': range_high - range_low,
                    'consolidation_duration': (olhcv_data['time'].iloc[i] - start_time),
                    'std_dev': std_dev,
                    'average_volume': average_volume
                })
                close_prices.clear()
                volumes.clear()
                continue
        

        # Calculate range of the previous 'window' candles
        rolling_high = olhcv_data['high'].iloc[(i - window):i].max()
        rolling_low = olhcv_data['low'].iloc[(i - window):i].min()
        range_size = rolling_high - rolling_low
        

        # If the range is smaller than the threshold, it's consolidation
        if range_size <= range_threshold:
            # print("range_size is ", range_size, "and range_threshold is ", range_threshold)
            if olhcv_data['in_consolidation'].iloc[(i - window):i].any():
                continue  # Skip if any of the prior candles are already in consolidation
            # Mark all candles in the current window as in consolidation
            olhcv_data.iloc[(i - window):i, olhcv_data.columns.get_loc('in_consolidation')] = True
            range_low = rolling_low
            range_high = rolling_high
            start_time = olhcv_data['time'].iloc[i - window]  # Time when consolidation started
            currently_consolidating = True
            close_prices.extend(olhcv_data['close'].iloc[(i - window):i])
            volumes.extend(olhcv_data['volume'].iloc[(i - window):i])

    # Handle the final consolidation if it ends at the last candle
    if currently_consolidating:
        std_dev = pd.Series(close_prices).std()
        average_volume = sum(volumes)/len(volumes)
        consolidations.append({
            'date': date,
            'start_time': start_time,
            'end_time': olhcv_data['time'].iloc[len(olhcv_data) - 1],
            'range_high': range_high,
            'range_low': range_low,
            'consolidation_range': range_high - range_low,
            'consolidation_duration': (olhcv_data['time'].iloc[len(olhcv_data) - 1] - start_time),
            'std_dev': std_dev,
            'average_volume': average_volume
        })
    
    return pd.DataFrame(consolidations)


def check_consolidation_breakout(ohlc_data, consolidation_info, timeframe_in_minutes = 1):
    """

    Parameters:
    ohlc_data (pd.DataFrame): DataFrame containing OHLC data with a 'time' column. Must have columns: 'open', 'high', 'low', 'close', 'time'.
    consolidation_info: is the output of the previous function

    Returns:
    dict: Breakout direction ('up' or 'down') and breakout candle details (time, price, etc.), or None if no breakout.
    """

    # print(consolidation_info)
    date = consolidation_info['date']
    end_time = consolidation_info['end_time']
    range_high = consolidation_info['range_high']
    range_low = consolidation_info['range_low']
    std_dev = consolidation_info['std_dev']
    average_volume = consolidation_info['average_volume']
    consolidation_range = consolidation_info['consolidation_range']
    consolidation_duration = consolidation_info['consolidation_duration']
    # breakout_distance = consolidation_info['breakout_distance']
    breakout = None
    hour = end_time.hour
    minute = end_time.minute
    
    # Convert 'time' column to datetime
    ohlc_data['time'] = pd.to_datetime(ohlc_data['time'], format='%H:%M:%S', errors='coerce')  

    filtered_ohlc_data = ohlc_data[(ohlc_data['time'].dt.hour == hour) &  (ohlc_data['time'].dt.minute >= minute) | 
                                       (ohlc_data['time'].dt.hour > hour)]
    
    volumes_after_breakout = []
    # Iterate over the filtered data
    for i in range(len(filtered_ohlc_data)):
        
        candle = filtered_ohlc_data.iloc[i]
        volumes_after_breakout.append(candle['volume'])

        # Check breakout upwards
        if candle['close'] > range_high:
            average_volume_after_breakout = sum(volumes_after_breakout)/len(volumes_after_breakout)
            breakout = {
                'date': date,
                'direction': 'up',
                'range_low': float(range_low),
                'range_high': float(range_high),
                'breakout_time': (candle['time'] + timedelta(minutes=timeframe_in_minutes)).strftime('%H:%M:%S'),
                'breakout_price': float(candle['close']),
                'breakout_distance': float(candle['close'] - range_high),
                'consolidation_range': consolidation_range,
                'consolidation_duration': consolidation_duration,
                'std_dev': float(std_dev),
                'average_volume': float(average_volume),
                'average_volume_after_breakout': float(average_volume_after_breakout)

            }
            # print(breakout)
            break

        # Check breakout 
        elif candle['close'] < range_low:
            average_volume_after_breakout = sum(volumes_after_breakout)/len(volumes_after_breakout)
            breakout = {
                'date': date,
                'direction': 'down',
                'range_low': float(range_low),
                'range_high': float(range_high),
                'breakout_time': (candle['time'] + timedelta(minutes=timeframe_in_minutes)).strftime('%H:%M:%S'),
                'breakout_price': float(candle['close']),
                'breakout_distance': float(range_low - candle['close']),
                'consolidation_range': consolidation_range,
                'consolidation_duration': consolidation_duration,
                'std_dev': float(std_dev),
                'average_volume': float(average_volume),
                'average_volume_after_breakout': float(average_volume_after_breakout)
            }
            # print(breakout)

            break
        elif candle['time'].hour == 16 and candle['time'].minute == 0:
            # print("No breakout occured and the day ended. Date: ", date)
            return breakout
        
    breakout_time = pd.to_datetime(breakout['breakout_time'], format='%H:%M:%S')
        
    hour = breakout_time.hour
    minute = breakout_time.minute
    filtered_ohlc_data = ohlc_data[(ohlc_data['time'].dt.hour == hour) &  (ohlc_data['time'].dt.minute >= minute) | 
                                       (ohlc_data['time'].dt.hour > hour)]
    
    if breakout['direction'] == 'up':
        pretend_sl = range_low
        pretend_tp = breakout['breakout_price'] + (breakout['breakout_price'] - range_low)
        # print(pretend_sl, "pretend_sl")
        # print(pretend_tp, "pretend_tp")
    
        for i in range(len(filtered_ohlc_data)):
            
            candle = filtered_ohlc_data.iloc[i]
            # print(candle)
            if candle['high'] >= pretend_tp and candle['low'] > pretend_sl:
                breakout['worked?'] = True
                return breakout
            elif candle['low'] <= pretend_sl and candle['high'] < pretend_tp:
                breakout['worked?'] = False
                return breakout
            elif candle['high'] >= pretend_tp and candle['low'] <= pretend_sl:
                breakout['worked?'] = None
                # print("Consolidation was found and broken out of, but the output was ambiguous. We don't know if it hit SL or TP first. Date: ", date)
                return breakout
            elif candle['time'].hour == 16 and candle['time'].minute == 1:
                breakout['worked?'] = None
                return breakout
            
    if breakout['direction'] == 'down':
        pretend_sl = range_high
        pretend_tp = breakout['breakout_price'] - abs(breakout['breakout_price'] - range_high)

    
        for i in range(len(filtered_ohlc_data)):
            candle = filtered_ohlc_data.iloc[i]
            # print(candle['time'].hour, candle['time'].minute)

            if candle['low'] <= pretend_tp and candle['high'] < pretend_sl:
                breakout['worked?'] = True
                return breakout
            elif candle['high'] >= pretend_sl and candle['low'] > pretend_tp:
                breakout['worked?'] = False
                return breakout
            elif candle['low'] <= pretend_tp and candle['high'] >= pretend_sl:
                breakout['worked?'] = None
                # print("Consolidation was found and broken out of, but the output was ambiguous. We don't know if it hit SL or TP first. Date: ", date)
                return breakout
            elif candle['time'].hour == 16 and candle['time'].minute == 1:
                breakout['worked?'] = None
                return breakout
            
    
    print(consolidation_info)
    breakout['worked?'] = None
    print("Conditions for checking the output were not met.")
    return breakout
    # raise ValueError("Conditions for checking the output were not met for some reason.")


def collect_consolidation_data(pd_trading_data, df, window, range_threshold, filter_start_time="9:00:00", filter_end_time="12:00:00"):
    warnings.filterwarnings("ignore")
    all_dates = pd_trading_data['date'].unique()  # Assuming pd_trading_data has a 'date' column

    # Initialize an empty list to collect results
    results = []

    # Loop over each day and run model_1_day
    for date in all_dates:
        olhc_data = olhcv_data(df, date, '1m') 
        olhc_data_pandas = olhc_data.to_pandas()
        if date.strftime('%Y-%m-%d %H:%M:%S') == "2024-07-04 00:00:00":
            continue
        if olhc_data_pandas.empty:
            # print(date, "had empty olhc data")
            continue
        # print(date)
        # print(window, range_threshold)
        consolidations = identify_consolidation(olhc_data_pandas, window, range_threshold, filter_start_time, filter_end_time, date)
        # print(consolidations)
        for consolid in consolidations.to_dict(orient="records"):
            # consolid is now a dictionary, access using column names
            consolidation_info = {
                'date': date,
                'start_time': consolid['start_time'],
                'end_time': consolid['end_time'],
                'range_low': consolid['range_low'],
                'range_high': consolid['range_high'],
                'consolidation_range': consolid['consolidation_range'],
                'consolidation_duration': consolid['consolidation_duration'],
                'std_dev': consolid['std_dev'],
                'average_volume': consolid['average_volume']
            }
            check_breakout = check_consolidation_breakout(olhc_data_pandas, consolidation_info=consolidation_info)
            # print(check_breakout)
            if check_breakout is not None:
                results.append(check_breakout)  
              

    results = pd.DataFrame(results)
    
    return results