import pandas as pd
from utils_2 import (
    get_contract_for_date,
    contract_size,
)
from model_2 import (
    find_HTF_POI,
    momentum_gap_validity,
    validate_inverse_momentum_gap,
    identify_inverse_momentum_gap,
    is_gap_in_and_above_50_poi
)


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

def execute_trade_2(ifvg, pd_trading_data, asset, threshold, window, dynamic_position_sizing=False, total_capital=None):
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
    # print(date)
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
    # entry_time = (datetime.strptime(entry_time, "%H:%M:%S") + timedelta(minutes=int(timeframe[0]))).strftime("%H:%M:%S")
    close_x_far = ifvg['close_x_far']


    gap_direction = ifvg['gap_direction']
    
    contracts = contract_size(stop_loss_size, 1000, asset, dynamic_position_sizing=dynamic_position_sizing, total_capital=total_capital)

    # Fetch and filter data
    daily_data = pd_trading_data[pd_trading_data['date'] == date]
    contract = get_contract_for_date(date, asset)
    daily_data = daily_data[daily_data['symbol'] == contract]

    # print(int(timeframe[0]))

    valid_trades = daily_data[daily_data['time'] >= entry_time]
    
    valid_trades = valid_trades[valid_trades['time'] < "21:00:00"]

    # print(valid_trades.head())

    # print(valid_trades['time'].iloc[-1], "and the type of this object is ", type(valid_trades['time'].iloc[-1]))

    running_sum = 0
    recent_deltas = [0] * window

    # print("Executing a trade with entry price", entry_price, entry_time, "and stop at: ", stop_loss)

    for trade in valid_trades.itertuples():
        # print(entry_price, stop_loss, target_price, entry_time)

        if gap_direction == "bearish":

            if trade.side == 'A':  # Sell aggressor
                delta = trade.size  
            elif trade.side == 'B':  # Buy aggressor
                delta = -trade.size  

            # Update the mean delta efficiently
            running_sum, recent_deltas, recent_mean_delta = update_mean_delta(running_sum, recent_deltas, delta, window)

            if recent_mean_delta < -threshold:
                entry_time_2 = pd.to_datetime(entry_time)
                exit_time_2 = pd.to_datetime(trade.time)
                trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60
                if asset == 'ES':
                    PnL = (trade.price - entry_price) * 50 * contracts
                elif asset == 'NQ':
                    PnL = (trade.price - entry_price) * 20 * contracts
                # print("running_sum:", running_sum, " recent_deltas: ", recent_deltas, " recent_mean_delta: ", recent_mean_delta)
                # print("Bullish trade where dynamic exit was used.")
                R = (trade.price - entry_price) / abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 0 else 0
                final_result = {
                    'Direction': "bullish",
                    'PnL': PnL,
                    'R': R,
                    'Win': False if trade.price <= entry_price else True,
                    'Contracts': contracts,
                    'Entry_Time': entry_time,
                    'Entry_Price': entry_price,
                    'Exit_Time': trade.time,
                    'Exit_Price': trade.price,
                    'Date': date,
                    'Trade_Duration (mins)': trade_duration,
                    'Stop_Loss_Placement': stop_loss,
                    'Close_x_far': close_x_far
                }
                if dynamic_position_sizing:
                    total_capital += final_result['PnL']
                
                return final_result
            
            if trade.price <= stop_loss:  # Stop loss hit
                entry_time_2 = pd.to_datetime(entry_time)
                exit_time_2 = pd.to_datetime(trade.time)
                trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60
                if asset == 'ES':
                    PnL = (stop_loss - entry_price) * 50 * contracts
                elif asset == 'NQ':
                    PnL = (stop_loss - entry_price) * 20 * contracts
                # print("stop hit")
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
                    # 'Profit_Target_Placement': target_price,
                    'Stop_Loss_Placement': stop_loss,
                    'Close_x_far': close_x_far  
                }
                if dynamic_position_sizing:
                    total_capital += final_result['PnL']

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
                # 'Profit_Target_Placement': target_price,
                'Stop_Loss_Placement': stop_loss,
                'Close_x_far': close_x_far
                
                }
                if dynamic_position_sizing:
                    total_capital += final_result['PnL']
                return final_result
            
        elif gap_direction == "bullish":

            if trade.side == 'A':  # Sell aggressor
                delta = trade.size  # Use size for order quantity
            elif trade.side == 'B':  # Buy aggressor
                delta = -trade.size  # Use size for order quantity

            # Update the mean delta efficiently
            running_sum, recent_deltas, recent_mean_delta = update_mean_delta(running_sum, recent_deltas, delta, window)
            # print(recent_mean_delta)

            # Adjust stop loss if the delta is more negative than the threshold (selling pressure)
            if recent_mean_delta > threshold:
                entry_time_2 = pd.to_datetime(entry_time)
                exit_time_2 = pd.to_datetime(trade.time)
                trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60

                if asset == 'ES':
                    PnL = (entry_price - trade.price) * contracts * 50
                elif asset == 'NQ':
                    PnL = (entry_price - trade.price) * contracts * 20
                # print("Bearish trade where dynamic exit was used.")
                R = (entry_price - trade.price) / abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 0 else 0
                final_result = {
                    'Direction': "bearish",
                    'PnL': PnL,
                    'R': R,
                    'Win': False if trade.price >= entry_price else True,
                    'Contracts': contracts,
                    'Entry_Time': entry_time,
                    'Entry_Price': entry_price,
                    'Exit_Time': trade.time,
                    'Exit_Price': trade.price,
                    'Date': date,
                    'Trade_Duration (mins)': trade_duration,
                    'Stop_Loss_Placement': stop_loss,
                    'Close_x_far': close_x_far
                }
                if dynamic_position_sizing:
                    total_capital += final_result['PnL']
                return final_result

            if trade.price >= stop_loss:  # Stop loss hit
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
                    # 'Profit_Target_Placement': target_price,
                    'Stop_Loss_Placement': stop_loss,
                    'Close_x_far': close_x_far
                }
                if dynamic_position_sizing:
                    total_capital += final_result['PnL']
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
                # 'Profit_Target_Placement': target_price,
                'Stop_Loss_Placement': stop_loss,
                'Close_x_far': close_x_far
                }
                if dynamic_position_sizing:
                    total_capital += final_result['PnL']
                # print("Detected a bearish trade!!! f hit")
                return final_result
    raise ValueError("Personal value error: Trade execution did not work successfully. There was no output to the trade!")

def model_2_day_2(
        all_trades_backtested, 
        date, 
        df, 
        pandas_data, 
        close_too_far = 4, 
        minimum_size = 1, 
        asset = 'ES', 
        maximum_age = 15, 
        can_run = 3, 
        max_candles_btwn = 15, 
        threshold = 2.5, 
        window = 400, 
        extra_sl_margin=0.25, 
        stop_modification = False,
        dynamic_position_sizing = False,
        total_capital = None
        ):
    # take the HTF POIs that day
    # analyze which are valid during killzone
    # for a valid HTF POI, during valid times and prices for i_momentum_gap to occur, call identify_inverse_momentum_gap() to find it
    # take the trade
    # output trade result in a dictionary

    valid_HTF_POIs_1h = []
    valid_HTF_POIs_15m = []

    HTF_POIs_1h = find_HTF_POI(date, '1h', df, asset)
    HTF_POIs_15m = find_HTF_POI(date, '15m', df, asset)

    for poi in HTF_POIs_1h:
        valid_HTF_POIs_1h.append(momentum_gap_validity(poi, df, asset, maximum_age=maximum_age, can_run=can_run))

    for poi in HTF_POIs_15m:
        valid_HTF_POIs_15m.append(momentum_gap_validity(poi, df, asset, maximum_age=maximum_age, can_run=can_run))
    # print(valid_HTF_POIs_15m)
    # print(valid_HTF_POIs_1h, "\n \n")

    valid_bull1_ifvgs = []
    valid_bear1_ifvgs = []

    for gap in identify_inverse_momentum_gap(date, '1m', "bullish", df, asset, extra_sl_margin=extra_sl_margin, stop_modification=stop_modification):
        # print(validate_inverse_momentum_gap(gap, df)['valid'])
        # print(gap)
        if validate_inverse_momentum_gap(gap, df, close_too_far, minimum_size, asset, max_candles_btwn)['valid']:
            valid_bull1_ifvgs.append(gap)

    for gap in identify_inverse_momentum_gap(date, '1m', "bearish", df, asset, extra_sl_margin=extra_sl_margin, stop_modification=stop_modification):
        # print(validate_inverse_momentum_gap(gap, df)['valid'])
        # print(gap)
        if validate_inverse_momentum_gap(gap, df, close_too_far, minimum_size, asset, max_candles_btwn)['valid']:
            valid_bear1_ifvgs.append(gap)
    # print(valid_bull1_ifvgs)
    # print(valid_bear1_ifvgs)
    
    trades = []

    for ifvg in valid_bull1_ifvgs:
        for poi in valid_HTF_POIs_1h + valid_HTF_POIs_15m:
            # print(ifvg)
            if is_gap_in_and_above_50_poi(ifvg, poi) and not ifvg['already_used']:  # Check if the gap is inside a valid POI
                # print(poi)
                ifvg['already_used'] =  True
                
                trade_info = {"Entry_price": ifvg['inverted_at_price'],
                                            "Stoploss_price": ifvg['proposed_sl'],
                                            "Trade_direction": 'bullish',
                                            "Trade_time": ifvg['inverted_at_time'],
                                            "Date": ifvg['date']}
                all_trades_backtested.append(trade_info)
                trade_result = execute_trade_2(ifvg, pandas_data, asset, threshold, window, dynamic_position_sizing, total_capital)  # Execute trade if gap is inside POI
                # print(ifvg['proposed_sl'])
                trades.append(trade_result)

    for ifvg in valid_bear1_ifvgs:
        
        for poi in valid_HTF_POIs_1h + valid_HTF_POIs_15m:
            
            if is_gap_in_and_above_50_poi(ifvg, poi) and not ifvg['already_used']:  # Check if the gap is inside a valid POI
                ifvg['already_used'] =  True
                
                trade_info = {"Entry_price": ifvg['inverted_at_price'],
                                            "Stoploss_price": ifvg['proposed_sl'],
                                            "Trade_direction": 'bearish',
                                            "Trade_time": ifvg['inverted_at_time'],
                                            "Date": ifvg['date']}
                all_trades_backtested.append(trade_info)
                trade_result = execute_trade_2(ifvg, pandas_data, asset, threshold, window, dynamic_position_sizing, total_capital)  # Execute trade if gap is inside POI
                
                trades.append(trade_result)
    
    # print(valid_bear1_ifvgs)
    # print(valid_HTF_POIs_15m[15])
    return trades

def model_2_2(pd_trading_data, df, model_day, close_too_far, minimum_size, asset, maximum_age = 15, can_run = 3, max_candles_btwn = 15, threshold = 2.5, window = 400, stop_modification = False, dynamic_position_sizing=False, total_capital=None):
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
    all_trades_backtested = []

    # Initialize an empty list to collect results
    results = []

    # Loop over each day and run model_1_day
    for date in all_dates:
        result = model_day(all_trades_backtested, date, df, pd_trading_data, close_too_far, minimum_size, asset, maximum_age, can_run, max_candles_btwn, threshold, window, stop_modification, dynamic_position_sizing, total_capital)
        # print(all_trades_backtested)
        results.extend(result)
        

    # Convert the list of dictionaries into a DataFrame
    results_df = pd.DataFrame(results)
    
    # Reorder the columns to ensure 'date' is the first column
    cols = ['Date'] + [col for col in results_df.columns if col != 'Date']
    results_df = results_df[cols]

    # Save all_trades_backtested to a CSV
    # all_trades_backtested = pd.DataFrame(all_trades_backtested)
    # all_trades_backtested.to_csv("all_trades_backtested.csv", index=False)
    # print("📁 all_trades_backtested saved to all_trades_backtested.csv")
    
    return results_df



def execute_trade_3(ifvg, pd_trading_data, asset, threshold, window):
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
    # print(date)
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
    # entry_time = (datetime.strptime(entry_time, "%H:%M:%S") + timedelta(minutes=int(timeframe[0]))).strftime("%H:%M:%S")
    close_x_far = ifvg['close_x_far']


    gap_direction = ifvg['gap_direction']
    
    contracts = contract_size(stop_loss_size, 1000, asset)

    # Fetch and filter data
    daily_data = pd_trading_data[pd_trading_data['date'] == date]
    contract = get_contract_for_date(date, asset)
    daily_data = daily_data[daily_data['symbol'] == contract]

    # print(int(timeframe[0]))

    valid_trades = daily_data[daily_data['time'] >= entry_time]
    
    valid_trades = valid_trades[valid_trades['time'] < "21:00:00"]

    # print(valid_trades.head())

    # print(valid_trades['time'].iloc[-1], "and the type of this object is ", type(valid_trades['time'].iloc[-1]))

    running_sum = 0
    recent_deltas = [0] * window

    # print("Executing a trade with entry price", entry_price, entry_time, "and stop at: ", stop_loss)

    for trade in valid_trades.itertuples():
        # print(entry_price, stop_loss, target_price, entry_time)

        if gap_direction == "bearish":

            if trade.side == 'A':  # Sell aggressor
                delta = trade.size  
            elif trade.side == 'B':  # Buy aggressor
                delta = -trade.size  

            # Update the mean delta efficiently
            running_sum, recent_deltas, recent_mean_delta = update_mean_delta(running_sum, recent_deltas, delta, window)

            if recent_mean_delta < -threshold:
                entry_time_2 = pd.to_datetime(entry_time)
                exit_time_2 = pd.to_datetime(trade.time)
                trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60
                if asset == 'ES':
                    PnL = (trade.price - entry_price) * 50 * contracts
                elif asset == 'NQ':
                    PnL = (trade.price - entry_price) * 20 * contracts
                # print("running_sum:", running_sum, " recent_deltas: ", recent_deltas, " recent_mean_delta: ", recent_mean_delta)
                # print("Bullish trade where dynamic exit was used.")
                R = (trade.price - entry_price) / abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 0 else 0
                final_result = {
                    'Direction': "bullish",
                    'PnL': PnL,
                    'R': R,
                    'Win': False if trade.price <= entry_price else True,
                    'Contracts': contracts,
                    'Entry_Time': entry_time,
                    'Entry_Price': entry_price,
                    'Exit_Time': trade.time,
                    'Exit_Price': trade.price,
                    'Date': date,
                    'Trade_Duration (mins)': trade_duration,
                    'Stop_Loss_Placement': stop_loss,
                    'Close_x_far': close_x_far
                }
                return final_result
            
            if trade.price <= stop_loss:  # Stop loss hit
                entry_time_2 = pd.to_datetime(entry_time)
                exit_time_2 = pd.to_datetime(trade.time)
                trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60
                if asset == 'ES':
                    PnL = (stop_loss - entry_price) * 50 * contracts
                elif asset == 'NQ':
                    PnL = (stop_loss - entry_price) * 20 * contracts
                # print("stop hit")
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
                    # 'Profit_Target_Placement': target_price,
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
                # 'Profit_Target_Placement': target_price,
                'Stop_Loss_Placement': stop_loss,
                'Close_x_far': close_x_far
                
                }
                return final_result
            
        elif gap_direction == "bullish":

            if trade.side == 'A':  # Sell aggressor
                delta = trade.size  # Use size for order quantity
            elif trade.side == 'B':  # Buy aggressor
                delta = -trade.size  # Use size for order quantity

            # Update the mean delta efficiently
            running_sum, recent_deltas, recent_mean_delta = update_mean_delta(running_sum, recent_deltas, delta, window)
            # print(recent_mean_delta)

            # Adjust stop loss if the delta is more negative than the threshold (selling pressure)
            if recent_mean_delta > threshold:
                entry_time_2 = pd.to_datetime(entry_time)
                exit_time_2 = pd.to_datetime(trade.time)
                trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60

                if asset == 'ES':
                    PnL = (entry_price - trade.price) * contracts * 50
                elif asset == 'NQ':
                    PnL = (entry_price - trade.price) * contracts * 20
                # print("Bearish trade where dynamic exit was used.")
                R = (entry_price - trade.price) / abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 0 else 0
                final_result = {
                    'Direction': "bearish",
                    'PnL': PnL,
                    'R': R,
                    'Win': False if trade.price >= entry_price else True,
                    'Contracts': contracts,
                    'Entry_Time': entry_time,
                    'Entry_Price': entry_price,
                    'Exit_Time': trade.time,
                    'Exit_Price': trade.price,
                    'Date': date,
                    'Trade_Duration (mins)': trade_duration,
                    'Stop_Loss_Placement': stop_loss,
                    'Close_x_far': close_x_far
                }
                return final_result

            if trade.price >= stop_loss:  # Stop loss hit
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
                    # 'Profit_Target_Placement': target_price,
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
                # 'Profit_Target_Placement': target_price,
                'Stop_Loss_Placement': stop_loss,
                'Close_x_far': close_x_far
                }
                # print("Detected a bearish trade!!! f hit")
                return final_result
    raise ValueError("Personal value error: Trade execution did not work successfully. There was no output to the trade!")