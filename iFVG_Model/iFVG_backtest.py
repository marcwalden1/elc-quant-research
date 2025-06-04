import pandas as pd
from datetime import datetime
from utils_2 import (
    olhcv_data,
    contract_size,
    get_contract_for_date
)


def find_HTF_POI(date, timeframe, df, asset):
    """
    Function to find momentum gaps.

    Input:
    date: string format
    timeframe: string format (eg, '1h', '15m')
    
    Returns:
    momentum_gaps: A list of dictionaries, each containing details of a momentum gap in the corresponding direction.
    """

    ohlc_data = olhcv_data(df, date, timeframe, asset)

    ohlc_data = ohlc_data.to_pandas()

    # Convert the 'time' column to datetime using pandas
    # Ensure 'date' is a string, if it's not already
    date_str = str(date) if isinstance(date, pd.Timestamp) else date

    # Now concatenate 'date_str' with 'ohlc_data['time']'
    ohlc_data['datetime_time'] = pd.to_datetime(date_str + ' ' + ohlc_data['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    
    momentum_gaps = []
    
    # Loop through the candles looking for 3-candle formations
    for i in range(len(ohlc_data) - 2):
        candle_1 = ohlc_data.iloc[i]
        candle_2 = ohlc_data.iloc[i + 1]
        candle_3 = ohlc_data.iloc[i + 2]
            
        # Check if candles 1 and 3 do not overlap and candle 2 is bullish
        if ((candle_1['high'] < candle_3['low']) and (candle_2['close'] > candle_2['open'])):
            
            # If candle_2['time'] is a string, convert it to a datetime object first
            if isinstance(candle_2['time'], str):
                candle_2_time = datetime.strptime(str(date.date()) + ' ' + candle_2['time'], '%Y-%m-%d %H:%M:%S')
            else:
                candle_2_time = candle_2['time']

            momentum_gaps.append({
                'gap_direction': 'bullish',
                'gap_low': float(candle_1['high']),
                'gap_high': float(candle_3['low']),
                'time': candle_2_time,
                'timeframe': timeframe,
                'already_used': False
            })
        
        # Check if candles 1 and 3 do not overlap and candle 2 is bearish
        if (candle_1['low'] > candle_3['high'] and (candle_2['close'] < candle_2['open'])):
            # If candle_2['time'] is a string, convert it to a datetime object first
            if isinstance(candle_2['time'], str):
                candle_2_time = datetime.strptime(str(date.date()) + ' ' + candle_2['time'], '%Y-%m-%d %H:%M:%S')
            else:
                candle_2_time = candle_2['time']

            momentum_gaps.append({
                'gap_direction': 'bearish',
                'gap_low': float(candle_3['high']),
                'gap_high': float(candle_1['low']),
                'time': candle_2_time,
                'timeframe': timeframe,
                'already_used': False
            })
    
    return momentum_gaps


def momentum_gap_validity(momentum_gap, df, asset, maximum_age = 15, can_run = 3):
    # Omitted for confidentiality purposes
   pass


def identify_inverse_momentum_gap(date, timeframe, direction, df, asset, extra_sl_margin=0.25, stop_modification = False):
    # bullish scenario:
    # identify all bearish momentum gaps
    # once closed above, classify as a possible momentum gap
    # output list of dictionaries of possible inverse momentum gaps

    ohlc_data = olhcv_data(df, date, timeframe, asset).to_pandas()

    # Filter data to get candles between 8:30 AM and 12:00 PM
    filtered_data = ohlc_data[((ohlc_data['interval_time'].dt.hour == 13) & 
                                          (ohlc_data['interval_time'].dt.minute >= 30)) |
                                          ((ohlc_data['interval_time'].dt.hour > 13) & 
                                          (ohlc_data['interval_time'].dt.hour < 17))]
    # return filtered_data.iloc[0:32]
    
    inverse_momentum_gaps = []  # List to store found momentum gaps
    for i in range(len(filtered_data) - 5):
        candle_1 = filtered_data.iloc[i]
        candle_2 = filtered_data.iloc[i + 1]
        candle_3 = filtered_data.iloc[i + 2]
        candle_4 = filtered_data.iloc[i + 3]
        candle_5 = filtered_data.iloc[i + 4]
        candle_6 = filtered_data.iloc[i + 5]
        # print(f"Processing candles: {candle_1['time']}, {candle_2['time']}, {candle_3['time']}")
            
        
        if direction == 'bearish':
            # Check if candles 1 and 3 do not overlap and candle 2 is bullish
            if ((candle_1['high'] < candle_3['low']) and (candle_2['close'] > candle_2['open'])):
                
                
                # If candle_2['time'] is a string, convert it to a datetime object first
                if isinstance(candle_2['time'], str):
                    candle_2_time = datetime.strptime(candle_2['time'], '%H:%M:%S')
                else:
                    candle_2_time = candle_2['time']
                # print("It should be detecting it.", candle_2_time)

                gap_high = float(candle_3['low'])
                proposed_sl = float(candle_3['low']) + extra_sl_margin

                if ((candle_2['high'] < candle_4['low']) and (candle_3['close'] > candle_3['open'])):
                    gap_high = float(candle_4['low'])
                    proposed_sl = float(candle_4['low']) + extra_sl_margin
                    if ((candle_3['high'] < candle_5['low']) and (candle_4['close'] > candle_4['open'])):
                        gap_high = float(candle_5['low'])
                        proposed_sl = float(candle_5['low']) + extra_sl_margin
                        if ((candle_4['high'] < candle_6['low']) and (candle_5['close'] > candle_5['open'])):
                            gap_high = float(candle_6['low'])
                            proposed_sl = float(candle_6['low']) + extra_sl_margin
                    

                inverse_momentum_gaps.append({
                    'gap_direction': 'bullish',
                    'gap_low': float(candle_1['high']),
                    'gap_high': gap_high,
                    'time': candle_2_time.strftime('%H:%M:%S'),
                    'proposed_sl': proposed_sl,
                    'date': date,
                    'timeframe': timeframe,
                    # 'momentum_gap_creation_time': momentum_gap_creation_time
                    'inverted': False,
                    'inverted_at_price': None,
                    'inverted_at_time': None,
                    'already_used': False
                })

            for gap in inverse_momentum_gaps:
                if (candle_2['close'] < gap['gap_low']) and (not gap['inverted']):
                    gap['inverted'] = True
                    gap['inverted_at_price'] = int(candle_2['close'])
                    gap['inverted_at_time'] = candle_3['time']
                    if stop_modification:
                        date = pd.to_datetime(date)
                        gap['time'] = pd.to_datetime(gap['time'])
                        gap['inverted_at_time'] = pd.to_datetime(gap['inverted_at_time'])
                        gap_time = gap['time']
                        inverted_time = gap['inverted_at_time']

                        start_time = pd.Timestamp(
                            year=date.year, month=date.month, day=date.day,
                            hour=gap_time.hour, minute=gap_time.minute, tz='UTC'
                        )

                        end_time = pd.Timestamp(
                            year=date.year, month=date.month, day=date.day,
                            hour=inverted_time.hour, minute=inverted_time.minute, tz='UTC'
                        )
                        # print("start_time", start_time)
                        # print("end time ", end_time)

                        filtered_data_for_stop = ohlc_data[
                            (ohlc_data['interval_time'] >= start_time) & 
                            (ohlc_data['interval_time'] <= end_time)]
                        filtered_data_for_stop = filtered_data_for_stop[:-1]
                        # print(filtered_data_for_stop)
                        # print("\n ------- \n")
                        lowest_price = min(filtered_data_for_stop.iloc[0]['low'], filtered_data_for_stop.iloc[1]['low'])
                        highest_price = max(filtered_data_for_stop.iloc[0]['high'], filtered_data_for_stop.iloc[1]['high'])
                        # print("highest_price", highest_price)
                        bearish_fvg_highs = []
                        for i in range(len(filtered_data_for_stop) - 2):
                            candle_1 = filtered_data_for_stop.iloc[i]
                            candle_2 = filtered_data_for_stop.iloc[i + 1]
                            candle_3 = filtered_data_for_stop.iloc[i + 2]

                            if ((candle_1['low'] > candle_3['high']) and (candle_2['close'] < candle_2['open'])):
                                bearish_fvg_highs.append(candle_1['low'])
                            if candle_3['high'] > highest_price:
                                highest_price = candle_3['high']
                            if candle_3['low'] < lowest_price:
                                lowest_price = candle_3['low']

                        current_stop = gap['proposed_sl']
                        fvg_stop = min(bearish_fvg_highs) + 0.25 if len(bearish_fvg_highs) > 0 else 0

                        if fvg_stop == 0:
                            seventy_percent_range = highest_price - 0.3 * (highest_price - lowest_price)
                            seventy_percent_range = round(seventy_percent_range / 0.25) * 0.25
                            gap['proposed_sl'] = max(current_stop, seventy_percent_range)
                            print(gap['inverted_at_time'], "current_stop", current_stop, "seventy", seventy_percent_range)
                            # print("no fvg")
                        else:
                            print(gap['inverted_at_time'], "current_stop", current_stop, "fvg_stop", fvg_stop)
                            gap['proposed_sl'] = max(current_stop, fvg_stop)



        
        elif direction == 'bullish':
            # Check if candles 1 and 3 do not overlap and candle 2 is bearish
            if (candle_1['low'] > candle_3['high'] and (candle_2['close'] < candle_2['open'])):
                # print(f"Detected bearish gap between {candle_1['time']} and {candle_3['time']}")

                # If candle_2['time'] is a string, convert it to a datetime object first
                if isinstance(candle_2['time'], str):
                    candle_2_time = datetime.strptime(candle_2['time'], '%H:%M:%S')
                else:
                    candle_2_time = candle_2['time']

                gap_low = float(candle_3['high'])
                proposed_sl = float(candle_3['high']) - extra_sl_margin

                if (candle_2['low'] > candle_4['high'] and (candle_3['close'] < candle_3['open'])):
                    gap_low = float(candle_4['high'])
                    proposed_sl = float(candle_4['high']) - extra_sl_margin
                    if (candle_3['low'] > candle_5['high'] and (candle_4['close'] < candle_4['open'])):
                        gap_low = float(candle_5['high'])
                        proposed_sl = float(candle_5['high']) - extra_sl_margin
                        if (candle_4['low'] > candle_6['high'] and (candle_5['close'] < candle_5['open'])):
                            gap_low = float(candle_6['high'])
                            proposed_sl = float(candle_6['high']) - extra_sl_margin

                inverse_momentum_gaps.append({
                    'gap_direction': 'bearish',
                    'gap_low': gap_low,
                    'gap_high': float(candle_1['low']),
                    'time': candle_2_time.strftime('%H:%M:%S'),
                    'proposed_sl': proposed_sl,
                    'date': date,
                    'timeframe': timeframe,
                    # 'momentum_gap_creation_time': momentum_gap_creation_time
                    'inverted': False,
                    'inverted_at_price': None,
                    'inverted_at_time': None,
                    'already_used': False
                })
                
            for gap in inverse_momentum_gaps:
                if (candle_2['close'] > gap['gap_high']) and (not gap['inverted']):
                    gap['inverted'] = True
                    gap['inverted_at_price'] = int(candle_2['close'])
                    gap['inverted_at_time'] = candle_3['time']
                    if stop_modification:
                        date = pd.to_datetime(date)
                        gap['time'] = pd.to_datetime(gap['time'])
                        gap['inverted_at_time'] = pd.to_datetime(gap['inverted_at_time'])
                        gap_time = gap['time']
                        inverted_time = gap['inverted_at_time']

                        start_time = pd.Timestamp(
                            year=date.year, month=date.month, day=date.day,
                            hour=gap_time.hour, minute=gap_time.minute, tz='UTC'
                        )

                        end_time = pd.Timestamp(
                            year=date.year, month=date.month, day=date.day,
                            hour=inverted_time.hour, minute=inverted_time.minute, tz='UTC'
                        )
                        # print("start_time", start_time)
                        # print("end time ", end_time)

                        filtered_data_for_stop = ohlc_data[
                            (ohlc_data['interval_time'] >= start_time) & 
                            (ohlc_data['interval_time'] <= end_time)]
                        filtered_data_for_stop = filtered_data_for_stop[:-1]
                        # print(filtered_data_for_stop)
                        # print("\n ------- \n")
                        lowest_price = min(filtered_data_for_stop.iloc[0]['low'], filtered_data_for_stop.iloc[1]['low'])
                        highest_price = max(filtered_data_for_stop.iloc[0]['high'], filtered_data_for_stop.iloc[1]['high'])
                        # print("highest_price", highest_price)
                        bullish_fvg_lows = []
                        for i in range(len(filtered_data_for_stop) - 2):
                            candle_1 = filtered_data_for_stop.iloc[i]
                            candle_2 = filtered_data_for_stop.iloc[i + 1]
                            candle_3 = filtered_data_for_stop.iloc[i + 2]

                            if ((candle_1['high'] < candle_3['low']) and (candle_2['close'] > candle_2['open'])):
                                bullish_fvg_lows.append(candle_1['high'])
                            if candle_3['high'] > highest_price:
                                highest_price = candle_3['high']
                            if candle_3['low'] < lowest_price:
                                lowest_price = candle_3['low']

                        current_stop = gap['proposed_sl']
                        fvg_stop = min(bullish_fvg_lows) - 0.25 if len(bullish_fvg_lows) > 0 else 999999
                        # print(fvg_stop)
                        # print(gap['time'], "lowest_price", lowest_price, "and highest price", highest_price)
                        # print(lowest_price + 0.3 * (highest_price - lowest_price))

                        if fvg_stop == 999999:
                            seventy_percent_range = lowest_price + 0.3 * (highest_price - lowest_price)
                            seventy_percent_range = round(seventy_percent_range / 0.25) * 0.25
                            gap['proposed_sl'] = min(current_stop, seventy_percent_range)
                            # print("current_stop", current_stop, "seventy", seventy_percent_range)
                            # print("no fvg")
                        else:
                            # print(gap['time'], "current_stop", current_stop, "fvg_stop", fvg_stop)
                            gap['proposed_sl'] = min(current_stop, fvg_stop)

    return inverse_momentum_gaps
    


def validate_inverse_momentum_gap(inverse_momentum_gap, df, close_too_far, minimum_size, asset, max_candles_btwn = 15):
    "Invalidates an inverse momentum gap based on whether it satisfies 5 conditions."
    # Omitted for confidentiality purposes
    pass



def is_gap_in_and_above_50_poi(inverse_gap, poi):
    # Check if the gap direction matches the POI direction (optional based on your strategy)

    
    if inverse_gap['gap_direction'] == poi['gap_direction']:
        return False
    
    # Convert the POI time window to datetime objects for comparison
    poi_start_time = poi['tapped_time']
    poi_end_time = poi['valid_until']

    if poi_start_time is None:
        return False

    if isinstance(poi_start_time, str):
        poi_start_time = datetime.strptime(poi_start_time, '%Y-%m-%d %H:%M:%S')

    # Ensure poi_end_time is a datetime object
    if isinstance(poi_end_time, str):
        poi_end_time = datetime.strptime(poi_end_time, '%Y-%m-%d %H:%M:%S')

    # Now, ensure inverted_at_time is in datetime format
    if isinstance(inverse_gap['inverted_at_time'], str):
        gap_time = datetime.strptime(inverse_gap['inverted_at_time'], '%H:%M:%S').replace(
            year=poi_start_time.year, month=poi_start_time.month, day=poi_start_time.day)
    else:
        gap_time = inverse_gap['inverted_at_time'].replace(
            year=poi_start_time.year, month=poi_start_time.month, day=poi_start_time.day)
        
    # Check if the gap time is within the POI time window
    if not (poi_start_time <= gap_time <= poi_end_time):
        return False
    
    poi_high = poi['gap_high']
    poi_low = poi['gap_low']
    mid_point = poi_low + (poi_high - poi_low)/2

    if poi['gap_direction'] == "bullish":
        if inverse_gap['inverted_at_price'] <= mid_point:
            return False
    elif poi['gap_direction'] == "bearish":
        if inverse_gap['inverted_at_price'] >= mid_point:
            return False

    
    # If both conditions are met, the gap is inside the POI and at a good price to take the trade.
    return True


def execute_trade(ifvg, pd_trading_data, asset, fixed_R):
    """
    Function: execute_trade

    Inputs:
    - ifvg (dict): A dictionary representing an inverse momentum gap with information like the gap's direction, time, price, stop loss, etc.
    - pd_trading_data (pandas.DataFrame): A DataFrame containing trading data with columns like 'date', 'symbol', 'time', and 'price'.

    Outputs:
    - final_result (dict): A dictionary with trade details including 'Direction', 'PnL' (Profit and Loss), 'R' (Risk-to-Reward ratio), 
    'Win' (Boolean indicating if the trade was a win), 'Contracts' (number of contracts), 'Entry_Time', 'Entry_Price', 
    'Exit_Time', 'Exit_Price', 'Date', 'Trade_Duration', 'Profit_Target_Placement', and 'Stop_Loss_Placement'.
    """
    # print(ifvg)
    date = ifvg['date']
    # print("Executing a trade")
    final_result = {
        'Direction': None,
        'PnL': None,
        'R': None,
        'Win': "None",
        'Contracts': None,
        'Entry_Time': None,
        'Entry_Price': None,
        'Exit_Time': None,
        'Exit_Price': None,
        'Date': date,
        'Trade_Duration': None,
        'Profit_Target_Placement': None,
        'Stop_Loss_Placement': None,
        'Close_x_far': None
        
                    }

    # Define variables
    # timeframe = ifvg['timeframe']
    entry_price = ifvg['inverted_at_price']
    stop_loss = ifvg['proposed_sl']
    stop_loss_size = abs(entry_price - stop_loss)
    entry_time = ifvg['inverted_at_time']
    close_x_far = ifvg['close_x_far']
    # print(entry_time)
    # entry_time = (datetime.strptime(entry_time, "%H:%M:%S") + timedelta(minutes=int(timeframe[0]))).strftime("%H:%M:%S")



    gap_direction = ifvg['gap_direction']
    

    if gap_direction == "bearish":
        target_price = entry_price + fixed_R * stop_loss_size # HYPERPARAMETER
    elif gap_direction == "bullish":
        target_price = entry_price - fixed_R * stop_loss_size # HYPERPARAMETER
    contracts = contract_size(stop_loss_size, 1000, asset)

    # Fetch and filter data
    daily_data = pd_trading_data[pd_trading_data['date'] == date]
    contract = get_contract_for_date(date, asset)
    daily_data = daily_data[daily_data['symbol'] == contract]

    # print(int(timeframe[0]))

    valid_trades = daily_data[daily_data['time'] >= entry_time]
    
    valid_trades = valid_trades[valid_trades['time'] < "21:00:00"]

    # print(valid_trades.head())
    # print("entering trade execution at ", entry_time, "and price ", entry_price)

    for trade in valid_trades.itertuples():
        # print(trade.price, trade.time)

        # print(entry_price, stop_loss, target_price, entry_time)

        if gap_direction == "bearish":
            if trade.price >= target_price:  # Target reached
                entry_time_2 = pd.to_datetime(entry_time)
                exit_time_2 = pd.to_datetime(trade.time)
                trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60
                if asset == 'ES':
                    PnL = (target_price - entry_price) * 50 * contracts
                elif asset == 'NQ':
                    PnL = (target_price - entry_price) * 20 * contracts

                final_result = {
                    'Direction': "bullish",
                    'PnL': PnL,
                    'R': 2,
                    'Win': True,
                    'Contracts': contracts,
                    'Entry_Time': entry_time,
                    'Entry_Price': entry_price,
                    'Exit_Time': trade.time,
                    'Exit_Price': target_price,
                    'Date': date,
                    'Trade_Duration (mins)': trade_duration,
                    'Profit_Target_Placement': target_price,
                    'Stop_Loss_Placement': stop_loss,
                    'Close_x_far': close_x_far
                }
                return final_result
            
            
            
            elif trade.price <= stop_loss:  # Stop loss hit
                # print("stop is now getting hit at price", trade.price, trade.time)
                entry_time_2 = pd.to_datetime(entry_time)
                exit_time_2 = pd.to_datetime(trade.time)
                trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60
                if asset == 'ES':
                    PnL = (stop_loss - entry_price) * 50 * contracts
                elif asset == 'NQ':
                    PnL = (stop_loss - entry_price) * 20 * contracts

                final_result = {
                    'Direction': "bullish",
                    'PnL': PnL,
                    'R': -1,
                    'Win': False,
                    'Contracts': contracts,
                    'Entry_Time': entry_time,
                    'Entry_Price': entry_price,
                    'Exit_Time': trade.time,
                    'Exit_Price': stop_loss,
                    'Date': date,
                    'Trade_Duration (mins)': trade_duration,
                    'Profit_Target_Placement': target_price,
                    'Stop_Loss_Placement': stop_loss,
                    'Close_x_far': close_x_far
                }
                

                return final_result
            
            elif trade.time >= "20:59:00":
                entry_time_2 = pd.to_datetime(entry_time)
                exit_time_2 = pd.to_datetime(trade.time)
                trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60

                if asset == 'ES':
                    PnL = (trade.price - entry_price) * 50 * contracts
                elif asset == 'NQ':
                    PnL = (trade.price - entry_price) * 20 * contracts
                    
                final_result = {
                'Direction': "bullish",
                'PnL': PnL,
                'R': (trade.price - entry_price) / abs(entry_price - stop_loss),
                'Win': True if (trade.price - entry_price) / abs(entry_price - stop_loss) > 0 else False,
                'Contracts': contracts,
                'Entry_Time': entry_time,
                'Entry_Price': entry_price,
                'Exit_Time': "20:59:00",
                'Exit_Price': trade.price,
                'Date': date,
                'Trade_Duration (mins)': trade_duration,
                'Profit_Target_Placement': target_price,
                'Stop_Loss_Placement': stop_loss,
                'Close_x_far': close_x_far
                
                }
                return final_result
            
        elif gap_direction == "bullish":

            
            if trade.price <= target_price:  # Target reached
                entry_time_2 = pd.to_datetime(entry_time)
                exit_time_2 = pd.to_datetime(trade.time)
                trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60

                if asset == 'ES':
                    PnL = (entry_price - target_price) * contracts * 50
                elif asset == 'NQ':
                    PnL = (entry_price - target_price) * contracts * 20
                
                final_result = {
                    'Direction': "bearish",
                    'PnL': PnL,
                    'R': 2,
                    'Win': True,
                    'Contracts': contracts,
                    'Entry_Time': entry_time,
                    'Entry_Price': entry_price,
                    'Exit_Time': trade.time,
                    'Exit_Price': target_price,
                    'Date': date,
                    'Trade_Duration (mins)': trade_duration,
                    'Profit_Target_Placement': target_price,
                    'Stop_Loss_Placement': stop_loss,
                    'Close_x_far': close_x_far
                }
                # print("Detected a bearish trade!!! target hit")
                return final_result
            elif trade.price >= stop_loss:  # Stop loss hit
                # print("stop is now getting hit at price", trade.price, trade.time)
                entry_time_2 = pd.to_datetime(entry_time)
                exit_time_2 = pd.to_datetime(trade.time)
                trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60

                if asset == 'ES':
                    PnL = (entry_price - stop_loss) * contracts * 50
                elif asset == 'NQ':
                    PnL = (entry_price - stop_loss) * contracts * 20

                final_result = {
                    'Direction': "bearish",
                    'PnL': PnL,
                    'R': -1,
                    'Win': False,
                    'Contracts': contracts,
                    'Entry_Time': entry_time,
                    'Entry_Price': entry_price,
                    'Exit_Time': trade.time,
                    'Exit_Price': stop_loss,
                    'Date': date,
                    'Trade_Duration (mins)': trade_duration,
                    'Profit_Target_Placement': target_price,
                    'Stop_Loss_Placement': stop_loss,
                    'Close_x_far': close_x_far
                }
                # print("Detected a bearish trade!!! stoploss hit")
                return final_result
            
            elif trade.time >= "20:59:00":
                entry_time_2 = pd.to_datetime(entry_time)
                exit_time_2 = pd.to_datetime(trade.time)
                trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60

                if asset == 'ES':
                    PnL = (entry_price - trade.price) * contracts * 50
                elif asset == 'NQ':
                    PnL = (entry_price - trade.price) * contracts * 20

                final_result = {
                'Direction': "bearish",
                'PnL': PnL,
                'R': (entry_price - trade.price) / abs(entry_price - stop_loss),
                'Win': True if PnL > 0 else False,
                'Contracts': contracts,
                'Entry_Time': entry_time,
                'Entry_Price': entry_price,
                'Exit_Time': "20:59:00",
                'Exit_Price': trade.price,
                'Date': date,
                'Trade_Duration (mins)': trade_duration,
                'Profit_Target_Placement': target_price,
                'Stop_Loss_Placement': stop_loss,
                'Close_x_far': close_x_far
                }
                # print("Detected a bearish trade!!! f hit")
                return final_result
    raise ValueError("Personal value error: Trade execution did not work successfully. There was no output to the trade!")



def model_2_day(date, df, pandas_data, close_too_far = 4, minimum_size = 1, asset = 'ES', maximum_age = 15, can_run = 3, max_candles_btwn = 15, fixed_R = 2, extra_sl_margin=0.25):
    # take the HTF POIs that day
    # analyze which are valid during killzone
    # for a valid HTF POI, during valid times and prices for i_momentum_gap to occur, call identify_inverse_momentum_gap() to find it
    # take the trade
    # output trade result in a dictionary

    valid_HTF_POIs_1h = []
    valid_HTF_POIs_15m = []

    # print(date, "\n and the date type is : ", type(date))

    HTF_POIs_1h = find_HTF_POI(date, '1h', df, asset)
    HTF_POIs_15m = find_HTF_POI(date, '15m', df, asset)

    # print(HTF_POIs_1h, "\n")

    for poi in HTF_POIs_1h:
        valid_HTF_POIs_1h.append(momentum_gap_validity(poi, df, asset, maximum_age=maximum_age, can_run=can_run))

    for poi in HTF_POIs_15m:
        valid_HTF_POIs_15m.append(momentum_gap_validity(poi, df, asset, maximum_age=maximum_age, can_run=can_run))

    valid_bull1_ifvgs = []
    valid_bear1_ifvgs = []

    for gap in identify_inverse_momentum_gap(date, '1m', "bullish", df, asset, extra_sl_margin=extra_sl_margin):
        
        if validate_inverse_momentum_gap(gap, df, close_too_far, minimum_size, asset, max_candles_btwn)['valid']:
            valid_bull1_ifvgs.append(gap)

    for gap in identify_inverse_momentum_gap(date, '1m', "bearish", df, asset, extra_sl_margin=extra_sl_margin):
        
        if validate_inverse_momentum_gap(gap, df, close_too_far, minimum_size, asset, max_candles_btwn)['valid']:
            valid_bear1_ifvgs.append(gap)
    
    trades = []

    for ifvg in valid_bull1_ifvgs:
        for poi in valid_HTF_POIs_1h + valid_HTF_POIs_15m:
            if is_gap_in_and_above_50_poi(ifvg, poi) and not ifvg['already_used']:  # Check if the gap is inside a valid POI
                
                ifvg['already_used'] =  True
                trade_result = execute_trade(ifvg, pandas_data, asset, fixed_R)  # Execute trade if gap is inside POI
                trades.append(trade_result)

    for ifvg in valid_bear1_ifvgs:
        
        for poi in valid_HTF_POIs_1h + valid_HTF_POIs_15m:
            
            if is_gap_in_and_above_50_poi(ifvg, poi) and not ifvg['already_used']:  # Check if the gap is inside a valid POI
                ifvg['already_used'] =  True
                trade_result = execute_trade(ifvg, pandas_data, asset, fixed_R)  # Execute trade if gap is inside POI
                trades.append(trade_result)

    return trades


def model_2(pd_trading_data, df, model_day, close_too_far, minimum_size, asset, maximum_age = 15, can_run = 3, max_candles_btwn = 15, fixed_R = 2, extra_sl_margin=0.25):
    """
    This function runs a specified model for all dates in the trading data.
    
    Parameters:
    - pd_trading_data: The trading data (dataframe).
    - df: Dataframe needed for model (e.g., OHLC data).
    - model_day: The function that processes each day's data (e.g., model_1_day, model_1_day_2, model_1_day_3).
    
    Returns:
    - results_df: DataFrame containing the results for each day.
    """
    # Create a list of all the dates for the year
    all_dates = pd_trading_data['date'].unique()  # Assuming pd_trading_data has a 'date' column

    # Initialize an empty list to collect results
    results = []

    # Loop over each day and run model_1_day
    for date in all_dates:
        result = model_day(date, df, pd_trading_data, close_too_far, minimum_size, asset, maximum_age, can_run, max_candles_btwn, fixed_R, extra_sl_margin)
        
        # Append the result to the results list
        results.extend(result)
        

    # Convert the list of dictionaries into a DataFrame
    results_df = pd.DataFrame(results)
    
    # Reorder the columns to ensure 'date' is the first column
    cols = ['Date'] + [col for col in results_df.columns if col != 'Date']
    results_df = results_df[cols]
    
    return results_df