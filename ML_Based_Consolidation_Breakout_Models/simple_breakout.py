from datetime import datetime, timedelta
import pandas as pd
from utils import get_contract_for_date, olhcv_data, contract_size, overnight_range_large


def check_market_conditions(df, date, threshold=15):
    """
    Parameters:
    - df: DataFrame with trading data
    - date: The specific date for which to check the range
    - threshold: Threshold for determining whether the overnight range was large/expansionary

    Returns:
    possible_trading_day: A boolean that is True if market condition requirements are met and false otherwise.
    """

    possible_trading_day = True
    if not overnight_range_large(df, date, threshold):
        possible_trading_day = False
    
    return possible_trading_day




def calculate_IB(pd_trading_data, date, NYSEOpen = False):
    '''
    Parameters:
    pd_trading_data: Trading data that is in Pandas form. 
    date

    Output:
    ib_values: List of two elements where the first is the low of the IB and the second is the high of the IB. 
    '''

    # Filter data for the correct date and contract
    daily_data = pd_trading_data[pd_trading_data['date'] == date]
    contract = get_contract_for_date(date)
    daily_data = daily_data[daily_data['symbol'] == contract]

    if NYSEOpen:
        ib_data = daily_data[(daily_data['ts_event'].dt.hour == 9) & (daily_data['ts_event'].dt.minute >= 30) & (daily_data['ts_event'].dt.minute < 60)]
    else:
    # Filter for trades between 10:30 AM and 11:00 AM
        ib_data = daily_data[(daily_data['ts_event'].dt.hour == 10) & (daily_data['ts_event'].dt.minute >= 30) & (daily_data['ts_event'].dt.minute < 60)]

    # Calculate the lowest and highest price during the 9:30 AM to 10:00 AM period
    ib_values = [ib_data['price'].min(), ib_data['price'].max()]
    return ib_values



def check_IB_breakout(ohlc_data, IB_range, NYSEOpen = False):
    """
    Check if the 5-minute candles have broken out of the IB range between 10:05 and 11:30.

    Parameters:
    ohlc_data (pd.DataFrame): DataFrame containing OHLC data with a 'time' column. Must have columns: 'open', 'high', 'low', 'close', 'time'.
    IB_range (tuple): Tuple containing the IB low and IB high prices (IB_low, IB_high).

    Returns:
    dict: Breakout direction ('up' or 'down') and breakout candle details (time, price, etc.), or None if no breakout.
    """
    IB_low, IB_high = IB_range[0], IB_range[1]
    breakout = None
    
    # Convert 'time' column to datetime if it's not already
    ohlc_data['time'] = pd.to_datetime(ohlc_data['time'], format='%H:%M:%S', errors='coerce')  

    if NYSEOpen:
        filtered_ohlc_data = ohlc_data[((ohlc_data['time'].dt.hour == 10)) | 
                               ((ohlc_data['time'].dt.hour == 11) & (ohlc_data['time'].dt.minute <= 30))]

    else:
        # Filter data to include only between 10:05 and 11:30
        filtered_ohlc_data = ohlc_data[((ohlc_data['time'].dt.hour == 11)) | 
                               ((ohlc_data['time'].dt.hour == 12) & (ohlc_data['time'].dt.minute <= 30))]
    
    # Iterate over the filtered data
    for i in range(len(filtered_ohlc_data)):
        candle = filtered_ohlc_data.iloc[i]
        
        # Check breakout upwards
        if candle['close'] > IB_high:
            breakout = {
                'direction': 'up',
                'IB_Low': float(IB_low),
                'IB_High': float(IB_high),
                'breakout_time': (candle['time'] + timedelta(minutes=5)).strftime('%H:%M:%S'),
                'breakout_price': float(candle['close']),
            }
            break

        # Check breakout downwards
        elif candle['close'] < IB_low:
            breakout = {
                'direction': 'down',
                'IB_Low': float(IB_low),
                'IB_High': float(IB_high),
                'breakout_time': (candle['time'] + timedelta(minutes=5)).strftime('%H:%M:%S'),
                'breakout_price': float(candle['close']),
            }
            break

    return breakout

def find_momentum_gap(ohlc_data, breakout_direction, IB_High, IB_Low, NYSEOpen = False):
    """
    Function to find momentum gaps after the breakout from the Initial Balance (IB) in the specified direction.
    
    Parameters:
    ohlc_data (Pandas DataFrame): The OHLC data (with timestamps) for the 5-minute chart.
    breakout_time (str): The time at which the breakout occurs (e.g., '10:40:00').
    breakout_direction (str): The direction of the breakout ('up' or 'down').
    IB_High (float): The high of the Initial Balance.
    IB_Low (float): The low of the Initial Balance.
    
    Returns:
    list: A list of dictionaries, each containing details of a momentum gap in the corresponding direction.
    """


    # Convert the 'time' column to datetime using pandas
    ohlc_data['datetime_time'] = pd.to_datetime(ohlc_data['time'], format='%H:%M:%S')

    if NYSEOpen:
        # Filter data to get candles between 9:30 AM and 11:00 PM
        filtered_data = ohlc_data[((ohlc_data['datetime_time'].dt.hour == 9) & 
                                          (ohlc_data['datetime_time'].dt.minute >= 30)) |
                                          (ohlc_data['datetime_time'].dt.hour > 9) & 
                                          (ohlc_data['datetime_time'].dt.hour < 11)]

    else: 
        # Filter data to get candles between 10:30 AM and 12:00 PM
        filtered_data = ohlc_data[((ohlc_data['datetime_time'].dt.hour == 10) & 
                                          (ohlc_data['datetime_time'].dt.minute >= 30)) |
                                          (ohlc_data['datetime_time'].dt.hour > 10) & 
                                          (ohlc_data['datetime_time'].dt.hour < 12)]


    
    
    momentum_gaps = []  # List to store found momentum gaps
    
    # Loop through the 5-minute candles looking for 3-candle formations
    for i in range(len(filtered_data) - 2):
        candle_1 = filtered_data.iloc[i]
        candle_2 = filtered_data.iloc[i + 1]
        candle_3 = filtered_data.iloc[i + 2]
            
        # Only look for momentum gaps in the direction of the breakout
        # print(candle_2['time'])

        if breakout_direction == 'up':
            # Check if candles 1 and 3 do not overlap and candle 2 is bullish
            if ((candle_1['high'] < candle_3['low']) and (candle_2['close'] > candle_2['open'])):
                # print("It should be detecting it.")


                # Invalid gap if the low of candle 3 is more than 5 points below the breakout price

                if candle_3['low'] < IB_High - 5:
                    # print("It got invalidated since candle_3['low'] < IB_High - 5")
                    continue

                if candle_1['open'] >= candle_1['close']:
                    candle_1_body = candle_1['open']
                else:
                    candle_1_body = candle_1['close']

                to_body = candle_3['low'] - candle_1_body
                to_low = candle_3['low'] - candle_1['low']

                diff_to_low = abs(to_low - 6)
                diff_to_body = abs(to_body - 6)

                # Check which difference is smaller
                if diff_to_low < diff_to_body:
                    proposed_sl_size = to_low
                else:
                    proposed_sl_size = to_body

                if proposed_sl_size <= 0.50 or (proposed_sl_size >= 10):
                    continue

                proposed_sl = candle_3['low'] - proposed_sl_size
                proposed_tp = candle_3['low'] + 2 * proposed_sl_size
                proposed_be_level = candle_3['low'] + 1.5 * proposed_sl_size

                # If candle_2['time'] is a string, convert it to a datetime object first
                if isinstance(candle_2['time'], str):
                    candle_2_time = datetime.strptime(candle_2['time'], '%H:%M:%S')
                else:
                    candle_2_time = candle_2['time']

                # Now you can use strftime on the datetime object
                momentum_gap_creation_time = (candle_2_time + timedelta(minutes=10)).strftime('%H:%M:%S')

                momentum_gaps.append({
                    'gap_direction': 'bullish',
                    'gap_low': float(candle_1['high']),
                    'gap_high (entry)': float(candle_3['low']),
                    'gap_size': float(candle_3['low']) - float(candle_1['high']),
                    'time': candle_2_time,
                    'proposed_sl': float(proposed_sl),
                    'proposed_tp': float(proposed_tp),
                    'proposed_be_level': float(round(proposed_be_level * 4) / 4),
                    'momentum_gap_creation_time': momentum_gap_creation_time
                })
        
        elif breakout_direction == 'down':
            # Check if candles 1 and 3 do not overlap and candle 2 is bearish
            if (candle_1['low'] > candle_3['high'] and (candle_2['close'] < candle_2['open'])):
                # Invalid gap if the high of candle 3 is more than 5 points above the breakout price
                if candle_3['high'] > IB_Low + 5:
                    # print("It got invalidated since candle_3['low'] < IB_High - 5")
                    continue

                if candle_1['open'] >= candle_1['close']:
                    candle_1_body = candle_1['close']
                else:
                    candle_1_body = candle_1['open']

                to_body = candle_1_body - candle_3['high']
                to_low = candle_1['high'] - candle_3['high']

                diff_to_low = abs(to_low - 6)
                diff_to_body = abs(to_body - 6)

                # Check which difference is smaller
                if diff_to_low < diff_to_body:
                    proposed_sl_size = to_low
                else:
                    proposed_sl_size = to_body

                if proposed_sl_size <= 0.50 or (proposed_sl_size >= 10):
                    continue

                proposed_sl = candle_3['high'] + proposed_sl_size
                proposed_tp = candle_3['high'] - 2 * proposed_sl_size
                proposed_be_level = candle_3['high'] - 1.5 * proposed_sl_size

                # If candle_2['time'] is a string, convert it to a datetime object first
                if isinstance(candle_2['time'], str):
                    candle_2_time = datetime.strptime(candle_2['time'], '%H:%M:%S')
                else:
                    candle_2_time = candle_2['time']

                # Now you can use strftime on the datetime object
                momentum_gap_creation_time = (candle_2_time + timedelta(minutes=10)).strftime('%H:%M:%S')

                momentum_gaps.append({
                    'gap_direction': 'bearish',
                    'gap_low (entry)': float(candle_3['high']),
                    'gap_high': float(candle_1['low']),
                    'gap_size': float(candle_1['low']) - float(candle_3['high']),
                    'time': candle_2_time,
                    'proposed_sl': float(proposed_sl),
                    'proposed_tp': float(proposed_tp),
                    'proposed_be_level': float(round(proposed_be_level * 4) / 4),
                    'momentum_gap_creation_time': momentum_gap_creation_time
                })
    
    return momentum_gaps

def model_1_day(date, pd_trading_data, df, considering_mc, NYSEOpen = False):
    """
    Main function to test Model 1 for a given day.
    
    Parameters:
    date (str): The date of the trading day.
    pd_trading_data: Same as df but in pandas
    df: 
    
    Returns:
    dict: Contains Direction, PnL, R won/lost, contracts traded, time and price of entry, and time and price of exit, date.
    """

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
        'Gap_Size': None
                    }
    
    if considering_mc:
        if not check_market_conditions(df, date):
            return final_result # print("Market conditions not suitable for trading.")

    daily_data = pd_trading_data[pd_trading_data['date'] == date]

    if daily_data.empty:
        return final_result # print(f"No data found for {date}")

    contract = get_contract_for_date(date)
    daily_data = daily_data[daily_data['symbol'] == contract]
    olhc_data = olhcv_data(df, date, timeframe='5m')
    olhc_data_to_pandas = olhc_data.to_pandas()
    IB_Range = calculate_IB(pd_trading_data, date, NYSEOpen)
    IB_Low = IB_Range[0]
    IB_High = IB_Range[1]
    IB_breakout = check_IB_breakout(olhc_data_to_pandas, IB_Range, NYSEOpen)
    if not IB_breakout:
        return final_result # print("There was no break out of the IB range in time!")
    breakout_direction = IB_breakout['direction']
    breakout_time = IB_breakout['breakout_time']

    # Debug
    # print(f"Breakout Direction: {breakout_direction}, Breakout Time: {breakout_time}")

    momentum_gaps = find_momentum_gap(olhc_data_to_pandas, breakout_direction, IB_High, IB_Low, NYSEOpen)

    # Debug
    # print(f"Found Momentum Gaps: {momentum_gaps}")

    if momentum_gaps == []:
        return final_result # print("No valid momentum gaps found.")
    
    if breakout_direction == 'up':
        for i, gap in enumerate(momentum_gaps):
            limit_price = gap['gap_high (entry)']
            stop_loss = gap['proposed_sl']
            target_price = gap['proposed_tp']

            limit_order_triggered = False

            # Debug
            # print(f"Processing Momentum Gap {i} ({gap['gap_direction']}), Limit Price: {limit_price}, Stop Loss: {stop_loss}, Target Price: {target_price}")

            # Filter trades between the current gap and the next gap (if any)
            if i + 1 < len(momentum_gaps):
                next_gap_time = momentum_gaps[i + 1]['momentum_gap_creation_time']
                # Debug
                # print("Next gap time:", next_gap_time)
            else:
                next_gap_time = None
                # Debug
                # print("Next gap time:", next_gap_time)

            


            valid_trades = daily_data[(daily_data['time'] >= gap['momentum_gap_creation_time'])]
            
            if next_gap_time is None:
                if NYSEOpen:
                    valid_trades = valid_trades[valid_trades['time'] < "12:30:00"]
                else:
                    valid_trades = valid_trades[valid_trades['time'] < "13:30:00"]
            else:
                valid_trades = valid_trades[valid_trades['time'] < next_gap_time]

            
            # Debug
            # print("Breakout time and current momentum_gap_creation_time:", breakout_time, gap['momentum_gap_creation_time'], " and next gap time:", next_gap_time)

            for trade in valid_trades.itertuples():
                # 3 scenarios: price goes up without stopping, price hits our limit, time runs out without no other gap getting created

                if trade.price >= limit_price + 2 * abs(limit_price - stop_loss): # if price runs too far
                    # Remove limit order and stop loss
                    limit_price = None
                    stop_loss = None
                    target_price = None
                    # Debug
                    # print("Trade was invalidated because price ran too far without offering entry.")
                    break
                elif trade.price <= limit_price and trade.time < breakout_time:
                    # print("Entry was offered before the breakout")
                    # print("Trade time: ", trade.time, " and Breakout time:", breakout_time)
                    continue
                
                elif trade.price <= limit_price and trade.time >= breakout_time:  # Limit order triggered
                    entry_time = trade.time
                    limit_order_triggered = True
                    # Set target price based on the target
                    entry_price = limit_price

                    # Debug
                    # print(f"Limit order triggered at {entry_time} for price {entry_price}")
                    break

                elif trade.time >= ("11:59:00" if NYSEOpen else "12:59:00"):                    
                    return final_result # print("No trade: Our limit was not triggered and our entry was not invalidated after the entire morning passed")



            # If limit order is placed, check for target or stop loss
            if limit_order_triggered:
                contracts = contract_size(abs(entry_price - stop_loss), 1000)
                trades_after_entry = daily_data[daily_data['time'] >= entry_time]
                for trade in trades_after_entry.itertuples():



                    if trade.price >= target_price:  # Target reached
                        entry_time_2 = pd.to_datetime(entry_time)
                        exit_time_2 = pd.to_datetime(trade.time)
                        trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60
                        final_result = {
                            'Direction': gap['gap_direction'],
                            'PnL': (target_price - entry_price) * 50 * contracts,
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
                            'Gap_Size': gap['gap_size']
                        }
                        return final_result
                    elif trade.price <= stop_loss:  # Stop loss hit
                        entry_time_2 = pd.to_datetime(entry_time)
                        exit_time_2 = pd.to_datetime(trade.time)
                        trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60
                        final_result = {
                            'Direction': gap['gap_direction'],
                            'PnL': (stop_loss - entry_price) * 50 * contracts,
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
                            'Gap_Size': gap['gap_size']
                            
                        }
                        return final_result
                    elif trade.time >= "16:59:00":
                        entry_time_2 = pd.to_datetime(entry_time)
                        exit_time_2 = pd.to_datetime(trade.time)
                        trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60
                        final_result = {
                        'Direction': gap['gap_direction'],
                        'PnL': (trade.price - entry_price) * contracts * 50,
                        'R': (trade.price - entry_price) / abs(entry_price - stop_loss),
                        'Win': True if (trade.price - entry_price) / abs(entry_price - stop_loss) > 0 else False,
                        'Contracts': contracts,
                        'Entry_Time': entry_time,
                        'Entry_Price': entry_price,
                        'Exit_Time': "16:59:00",
                        'Exit_Price': trade.price,
                        'Date': date,
                        'Trade_Duration (mins)': trade_duration,
                        'Profit_Target_Placement': target_price,
                        'Stop_Loss_Placement': stop_loss,
                        'Gap_Size': gap['gap_size']
                        
                        }
                        return final_result
    elif breakout_direction == 'down':
        for i, gap in enumerate(momentum_gaps):
            limit_price = gap['gap_low (entry)']
            stop_loss = gap['proposed_sl']
            target_price = gap['proposed_tp']

            limit_order_triggered = False

            # Debug
            # print(f"Processing Momentum Gap {i} ({gap['gap_direction']}), Limit Price: {limit_price}, Stop Loss: {stop_loss}, Target Price: {target_price}")

            # Filter trades between the current gap and the next gap (if any)
            if i + 1 < len(momentum_gaps):
                next_gap_time = momentum_gaps[i + 1]['momentum_gap_creation_time']
                # Debug
                # print("Next gap time:", next_gap_time)
            else:
                next_gap_time = None
                # Debug
                # print("Next gap time:", next_gap_time)

            


            valid_trades = daily_data[(daily_data['time'] >= gap['momentum_gap_creation_time'])]
            
            if next_gap_time is None:
                if NYSEOpen:
                    valid_trades = valid_trades[valid_trades['time'] < "12:30:00"]
                else:
                    valid_trades = valid_trades[valid_trades['time'] < "13:30:00"]
            else:
                valid_trades = valid_trades[valid_trades['time'] < next_gap_time]

            
            # Debug
            # print("Breakout time and current momentum_gap_creation_time:", breakout_time, gap['momentum_gap_creation_time'], " and next gap time:", next_gap_time)
            # print("Limit price is ", limit_price)
            for trade in valid_trades.itertuples():
                # 3 scenarios: price goes up without stopping, price hits our limit, time runs out without no other gap getting created

                if trade.price <= limit_price - 2 * abs(limit_price - stop_loss): # if price runs too far (change for 'down')
                    # Remove limit order and stop loss
                    limit_price = None
                    stop_loss = None
                    target_price = None
                    # Debug
                    # print("Trade was invalidated because price ran too far without offering entry.")
                    break
                elif trade.price >= limit_price and trade.time < breakout_time:
                    continue
                
                elif trade.price >= limit_price and trade.time >= breakout_time:  # Limit order triggered
                    entry_time = trade.time
                    limit_order_triggered = True
                    # Set target price based on the target
                    entry_price = limit_price

                    # Debug
                    # print(f"Limit order triggered at {entry_time} for price {entry_price}")
                    break

                elif trade.time >= ("11:59:00" if NYSEOpen else "12:59:00"):
                    # print("No trade: Our limit was not triggered and our entry was not invalidated after the entire morning passed")
                    return final_result


            # If limit order is placed, check for target or stop loss
            if limit_order_triggered:
                contracts = contract_size(abs(entry_price - stop_loss), 1000)
                trades_after_entry = daily_data[daily_data['time'] >= entry_time]
                for trade in trades_after_entry.itertuples():
                    if trade.time >= "16:59:00":
                        entry_time_2 = pd.to_datetime(entry_time)
                        exit_time_2 = pd.to_datetime(trade.time)
                        trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60
                        final_result = {
                        'Direction': gap['gap_direction'],
                        'PnL': (entry_price - trade.price) * contracts * 50,
                        'R': (entry_price - trade.price) / abs(entry_price - stop_loss),
                        'Win': True if (entry_price - trade.price) * contracts * 50 > 0 else False,
                        'Contracts': contracts,
                        'Entry_Time': entry_time,
                        'Entry_Price': entry_price,
                        'Exit_Time': "16:59:00",
                        'Exit_Price': trade.price,
                        'Date': date,
                        'Trade_Duration (mins)': trade_duration,
                        'Profit_Target_Placement': target_price,
                        'Stop_Loss_Placement': stop_loss,
                        'Gap_Size': gap['gap_size']
                        }
                        return final_result

                    if trade.price <= target_price:  # Target reached
                        entry_time_2 = pd.to_datetime(entry_time)
                        exit_time_2 = pd.to_datetime(trade.time)
                        trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60
                        final_result = {
                            'Direction': gap['gap_direction'],
                            'PnL': (entry_price - target_price) * contracts * 50,
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
                            'Gap_Size': gap['gap_size']
                        }
                        return final_result
                    elif trade.price >= stop_loss:  # Stop loss hit
                        entry_time_2 = pd.to_datetime(entry_time)
                        exit_time_2 = pd.to_datetime(trade.time)
                        trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60
                        final_result = {
                            'Direction': gap['gap_direction'],
                            'PnL': (entry_price - stop_loss) * contracts * 50,
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
                            'Gap_Size': gap['gap_size']
                        }
                        return final_result
    return final_result


def model_1(pd_trading_data, df, model_day, considering_mc = False, NYSEOpen = False):
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
        result = model_day(date, pd_trading_data, df, considering_mc, NYSEOpen)
        
        # Append the result to the results list
        results.append(result)

    # Convert the list of dictionaries into a DataFrame
    results_df = pd.DataFrame(results)
    
    # Reorder the columns to ensure 'date' is the first column
    cols = ['Date'] + [col for col in results_df.columns if col != 'Date']
    results_df = results_df[cols]
    
    return results_df