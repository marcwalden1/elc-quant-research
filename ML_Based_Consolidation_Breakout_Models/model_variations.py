import pandas as pd
from utils import get_contract_for_date, olhcv_data, contract_size
from model import calculate_IB, check_IB_breakout, find_momentum_gap, check_market_conditions




def model_1_day_2(date, pd_trading_data, df, NYSEOpen = False):
    """
    Main function to test Model 1 for a given day.
    
    Parameters:
    date (str): The date of the trading day.
    pd_trading_data (dict): Any trading data such as economic calendar, market conditions, etc.
    df
    
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
        'Gap_Size': None
                    }

    #if not check_market_conditions(date, trading_data):
    #    return final_result # print("Market conditions not suitable for trading.")

    daily_data = pd_trading_data[pd_trading_data['date'] == date]

    daily_data = pd_trading_data[pd_trading_data['date'] == date]
    if daily_data.empty:
        
        return final_result # print(f"No data found for {date}")

    contract = get_contract_for_date(date)
    daily_data = daily_data[daily_data['symbol'] == contract]
    olhc_data = olhcv_data(df, date, timeframe='5m')
    IB_Range = calculate_IB(pd_trading_data, date, NYSEOpen)
    IB_Low = IB_Range[0]
    IB_High = IB_Range[1]
    IB_breakout = check_IB_breakout(olhc_data.to_pandas(), IB_Range, NYSEOpen)
    if not IB_breakout:
        return final_result # print("There was no break out of the IB range in time!")
    breakout_direction = IB_breakout['direction']
    breakout_time = IB_breakout['breakout_time']

    # Debug
    # print(f"Breakout Direction: {breakout_direction}, Breakout Time: {breakout_time}")

    momentum_gaps = find_momentum_gap(olhc_data.to_pandas(), breakout_direction, IB_High, IB_Low, NYSEOpen)

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

                if trade.price >= limit_price + 2 * abs(limit_price - stop_loss): # if price runs too far (change for 'down')
                    # Remove limit order and stop loss
                    limit_price = None
                    stop_loss = None
                    target_price = None
                    # Debug
                    # print("Trade was invalidated because price ran too far without offering entry.")
                    break
                elif trade.price <= limit_price and trade.time < breakout_time:
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



            # 7. If limit order is placed, check for target or stop loss
            if limit_order_triggered:
                contracts = contract_size(abs(entry_price - stop_loss), 1000)
                trades_after_entry = daily_data[daily_data['time'] >= entry_time]
                stop_at_BE = False # Track
                
                for trade in trades_after_entry.itertuples():

                    if (trade.price >= entry_price + 1.3 * abs(entry_price - stop_loss)) and not stop_at_BE:  # Stop Moved to BE
                            stop_loss = entry_price
                            stop_at_BE  = True
                        

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
                            'Gap_Size': gap['gap_size']
                        }
                        return final_result
                    elif trade.price <= stop_loss:  # Stop loss hit
                        entry_time_2 = pd.to_datetime(entry_time)
                        exit_time_2 = pd.to_datetime(trade.time)
                        trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60
                        R = 0 if stop_at_BE else -1
                        final_result = {
                            'Direction': gap['gap_direction'],
                            'PnL': (stop_loss - entry_price) * 50 * contracts,
                            'R': R,
                            'Win': False,
                            'Contracts': contracts,
                            'Entry_Time': entry_time,
                            'Entry_Price': entry_price,
                            'Exit_Time': trade.time,
                            'Exit_Price': stop_loss,
                            'Date': date,
                            'Trade_Duration (mins)': trade_duration,
                            'Gap_Size': gap['gap_size']
                        }
                        return final_result
                    
                    elif trade.time >= "16:59:00":
                        entry_time_2 = pd.to_datetime(entry_time)
                        exit_time_2 = pd.to_datetime(trade.time)
                        trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60
                        R = (trade.price - entry_price) / abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 0 else 0
                        final_result = {
                        'Direction': gap['gap_direction'],
                        'PnL': (trade.price - entry_price) * contracts * 50,
                        'R': R,
                        'Win': True if (trade.price - entry_price) / abs(entry_price - stop_loss) > 0 else False,
                        'Contracts': contracts,
                        'Entry_Time': entry_time,
                        'Entry_Price': entry_price,
                        'Exit_Time': "16:59:00",
                        'Exit_Price': trade.price,
                        'Date': date,
                        'Trade_Duration (mins)': trade_duration,
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
                    valid_trades = valid_trades[valid_trades['time'] < "13:30:00"] # 12:30:00 is arbitrary (hyper-parameter)
            else:
                valid_trades = valid_trades[valid_trades['time'] < next_gap_time]

            
            # Debug
            # print("Breakout time and current momentum_gap_creation_time:", breakout_time, gap['momentum_gap_creation_time'], " and next gap time:", next_gap_time)

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
                    return final_result # print("No trade: Our limit was not triggered and our entry was not invalidated after the entire day passed")



            # 7. If limit order is placed, check for target or stop loss
            if limit_order_triggered:
                contracts = contract_size(abs(entry_price - stop_loss), 1000)
                trades_after_entry = daily_data[daily_data['time'] >= entry_time]
                stop_at_BE = False
                for trade in trades_after_entry.itertuples():

                    if (trade.price >= entry_price + 1.3 * abs(entry_price - stop_loss)) and not stop_at_BE:  # Stop Moved to BE
                            stop_loss = entry_price
                            stop_at_BE  = True

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
                            'Gap_Size': gap['gap_size']
                        }
                        return final_result
                    
                    elif trade.price >= stop_loss:  # Stop loss hit
                        

                        entry_time_2 = pd.to_datetime(entry_time)
                        exit_time_2 = pd.to_datetime(trade.time)
                        trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60
                        R = 0 if stop_at_BE else -1
                        final_result = {
                            'Direction': gap['gap_direction'],
                            'PnL': (entry_price - stop_loss) * contracts * 50,
                            'R': R,
                            'Win': False,
                            'Contracts': contracts,
                            'Entry_Time': entry_time,
                            'Entry_Price': entry_price,
                            'Exit_Time': trade.time,
                            'Exit_Price': stop_loss,
                            'Date': date,
                            'Trade_Duration (mins)': trade_duration,
                            'Gap_Size': gap['gap_size']
                        }
                        return final_result
                    elif trade.time >= "16:59:00":
                        entry_time_2 = pd.to_datetime(entry_time)
                        exit_time_2 = pd.to_datetime(trade.time)
                        trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60
                        R = (entry_price - trade.price) / abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 0 else 0 
                        final_result = {
                        'Direction': gap['gap_direction'],
                        'PnL': (entry_price - trade.price) * contracts * 50,
                        'R': R,
                        'Win': True if (entry_price - trade.price) * contracts * 50 > 0 else False,
                        'Contracts': contracts,
                        'Entry_Time': entry_time,
                        'Entry_Price': entry_price,
                        'Exit_Time': "16:59:00",
                        'Exit_Price': trade.price,
                        'Date': date,
                        'Trade_Duration (mins)': trade_duration,
                        'Gap_Size': gap['gap_size']
                        }
                        return final_result
    return final_result



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





def model_1_day_6(date, pd_trading_data, df, threshold, considering_mc, window, NYSEOpen = False):
    """
    Main function to test Model 1 for a given day.
    
    Parameters:
    date (str): The date of the trading day.
    pd_trading_data (dict): Any trading data such as economic calendar, market conditions, etc.
    df
    
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
        'Trade_Duration': None
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
                    valid_trades = valid_trades[valid_trades['time'] < "13:30:00"] # 12:30:00 is arbitrary (hyper-parameter)
            else:
                valid_trades = valid_trades[valid_trades['time'] < next_gap_time]

            
            # Debug
            # print("Breakout time and current momentum_gap_creation_time:", breakout_time, gap['momentum_gap_creation_time'], " and next gap time:", next_gap_time)

            for trade in valid_trades.itertuples():
                # 3 scenarios: price goes up without stopping, price hits our limit, time runs out without no other gap getting created

                if trade.price >= limit_price + 2 * abs(limit_price - stop_loss): # if price runs too far (change for 'down')
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

                running_sum = 0
                recent_deltas = []
                
                for trade in trades_after_entry.itertuples():

                    if trade.side == 'A':  # Sell aggressor
                        delta = trade.size  
                    elif trade.side == 'B':  # Buy aggressor
                        delta = -trade.size  

                    # Update the mean delta efficiently
                    running_sum, recent_deltas, recent_mean_delta = update_mean_delta(running_sum, recent_deltas, delta, window)
                    # print(recent_mean_delta)

                    # Adjust stop loss if the delta is more negative than the threshold (selling pressure)
                    if recent_mean_delta < -threshold:
                        entry_time_2 = pd.to_datetime(entry_time)
                        exit_time_2 = pd.to_datetime(trade.time)
                        trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60
                        R = (trade.price - entry_price) / abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 0 else 0
                        final_result = {
                            'Direction': gap['gap_direction'],
                            'PnL': (trade.price - entry_price) * 50 * contracts,
                            'R': R,
                            'Win': False if trade.price <= entry_price else True,
                            'Contracts': contracts,
                            'Entry_Time': entry_time,
                            'Entry_Price': entry_price,
                            'Exit_Time': trade.time,
                            'Exit_Price': trade.price,
                            'Date': date,
                            'Trade_Duration (mins)': trade_duration
                        }
                        return final_result
                    

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
                            'Trade_Duration (mins)': trade_duration
                        }
                        return final_result
                    elif trade.price <= stop_loss:  # Stop loss hit
                        entry_time_2 = pd.to_datetime(entry_time)
                        exit_time_2 = pd.to_datetime(trade.time)
                        trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60
                        R = (trade.price - entry_price) / abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 0 else 0
                        final_result = {
                            'Direction': gap['gap_direction'],
                            'PnL': (stop_loss - entry_price) * 50 * contracts,
                            'R': R,
                            'Win': False if trade.price <= entry_price else True,
                            'Contracts': contracts,
                            'Entry_Time': entry_time,
                            'Entry_Price': entry_price,
                            'Exit_Time': trade.time,
                            'Exit_Price': stop_loss,
                            'Date': date,
                            'Trade_Duration (mins)': trade_duration
                        }
                        return final_result
                    elif trade.time >= "16:59:00":
                        entry_time_2 = pd.to_datetime(entry_time)
                        exit_time_2 = pd.to_datetime(trade.time)
                        trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60
                        R = (trade.price - entry_price) / abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 0 else 0
                        final_result = {
                        'Direction': gap['gap_direction'],
                        'PnL': (trade.price - entry_price) * contracts * 50,
                        'R': R,
                        'Win': True if (trade.price - entry_price) / abs(entry_price - stop_loss) > 0 else False,
                        'Contracts': contracts,
                        'Entry_Time': entry_time,
                        'Entry_Price': entry_price,
                        'Exit_Time': "16:59:00",
                        'Exit_Price': trade.price,
                        'Date': date,
                        'Trade_Duration (mins)': trade_duration
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
                    valid_trades = valid_trades[valid_trades['time'] < "13:30:00"] # 12:30:00 is arbitrary (hyper-parameter)
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
                    # print("Entry was given before the breakout time!")
                    # print("Invalid trade time: ", trade.time, " and Breakout time:", breakout_time)
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

                running_sum = 0
                recent_deltas = [] 

                for trade in trades_after_entry.itertuples():

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
                        R = (entry_price - trade.price) / abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 0 else 0
                        final_result = {
                            'Direction': gap['gap_direction'],
                            'PnL': (entry_price - trade.price) * contracts * 50,
                            'R': R,
                            'Win': False if trade.price >= entry_price else True,
                            'Contracts': contracts,
                            'Entry_Time': entry_time,
                            'Entry_Price': entry_price,
                            'Exit_Time': trade.time,
                            'Exit_Price': trade.price,
                            'Date': date,
                            'Trade_Duration (mins)': trade_duration
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
                            'Trade_Duration (mins)': trade_duration
                        }
                        return final_result
                    elif trade.price >= stop_loss:  # Stop loss hit
                        entry_time_2 = pd.to_datetime(entry_time)
                        exit_time_2 = pd.to_datetime(trade.time)
                        trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60
                        R = (entry_price - trade.price) / abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 0 else 0
                        final_result = {
                            'Direction': gap['gap_direction'],
                            'PnL': (entry_price - stop_loss) * contracts * 50,
                            'R': R,
                            'Win': False if trade.price >= entry_price else True,
                            'Contracts': contracts,
                            'Entry_Time': entry_time,
                            'Entry_Price': entry_price,
                            'Exit_Time': trade.time,
                            'Exit_Price': stop_loss,
                            'Date': date,
                            'Trade_Duration (mins)': trade_duration
                        }
                        return final_result
                    
                    elif trade.time >= "16:59:00":
                        entry_time_2 = pd.to_datetime(entry_time)
                        exit_time_2 = pd.to_datetime(trade.time)
                        trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60
                        R = (entry_price - trade.price) / abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 0 else 0
                        final_result = {
                        'Direction': gap['gap_direction'],
                        'PnL': (entry_price - trade.price) * contracts * 50,
                        'R': R,
                        'Win': True if (entry_price - trade.price) * contracts * 50 > 0 else False,
                        'Contracts': contracts,
                        'Entry_Time': entry_time,
                        'Entry_Price': entry_price,
                        'Exit_Time': "16:59:00",
                        'Exit_Price': trade.price,
                        'Date': date,
                        'Trade_Duration (mins)': trade_duration
                        }
                        return final_result
    return final_result


def model_1_6(pd_trading_data, df, model_day, threshold, window = 400, minimum_fvg_size = 0, considering_mc = False, NYSEOpen = False):
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
        result = model_day(date, pd_trading_data, df, threshold, window, minimum_fvg_size, considering_mc, NYSEOpen)
        
        # Append the result to the results list
        results.append(result)

    # Convert the list of dictionaries into a DataFrame
    results_df = pd.DataFrame(results)
    
    # Reorder the columns to ensure 'date' is the first column
    cols = ['Date'] + [col for col in results_df.columns if col != 'Date']
    results_df = results_df[cols]
    
    return results_df


def model_1_day_8(date, pd_trading_data, df, considering_mc, NYSEOpen = False):
    """
    Main function to test Model 1 for a given day.
    
    Parameters:
    date (str): The date of the trading day.
    pd_trading_data (dict): Any trading data such as economic calendar, market conditions, etc.
    df
    
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
            if (gap['gap_high (entry)'] - gap['gap_low']) < 1:
                continue

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
            valid_trades = valid_trades[valid_trades['time'] >= "15:00:00"]
            valid_trades = valid_trades[valid_trades['time'] <= "16:00:00"]
            
            if next_gap_time is None:
                if NYSEOpen:
                    valid_trades = valid_trades[valid_trades['time'] < "12:30:00"]
                else:
                    valid_trades = valid_trades[valid_trades['time'] < "13:30:00"] # 12:30:00 is arbitrary (hyper-parameter)
            else:
                valid_trades = valid_trades[valid_trades['time'] < next_gap_time]

            
            # Debug
            # print("Breakout time and current momentum_gap_creation_time:", breakout_time, gap['momentum_gap_creation_time'], " and next gap time:", next_gap_time)

            for trade in valid_trades.itertuples():
                # 3 scenarios: price goes up without stopping, price hits our limit, time runs out without no other gap getting created

                if trade.price >= limit_price + 2 * abs(limit_price - stop_loss): # if price runs too far (change for 'down')
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
                        'Gap_Size': gap['gap_size']
                        }
                        return final_result
    elif breakout_direction == 'down':
        for i, gap in enumerate(momentum_gaps):
            if (gap['gap_high'] - gap['gap_low (entry)']) < 1:
                continue
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
            valid_trades = valid_trades[valid_trades['time'] >= "15:00:00"]
            valid_trades = valid_trades[valid_trades['time'] <= "16:00:00"]
            
            if next_gap_time is None:
                if NYSEOpen:
                    valid_trades = valid_trades[valid_trades['time'] < "12:30:00"]
                else:
                    valid_trades = valid_trades[valid_trades['time'] < "13:30:00"] # 12:30:00 is arbitrary (hyper-parameter)
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
                    # print("Entry was given before the breakout time!")
                    # print("Invalid trade time: ", trade.time, " and Breakout time:", breakout_time)
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
                            'Gap_Size': gap['gap_size']
                        }
                        return final_result
    return final_result


### MODEL 9

def model_1_day_9(date, pd_trading_data, df, threshold, window, minimum_fvg_size, considering_mc, NYSEOpen = False):
    """
    Main function to test Model 1 for a given day.
    
    Parameters:
    date (str): The date of the trading day.
    pd_trading_data (dict): Any trading data such as economic calendar, market conditions, etc.
    df
    
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
        'IB_Range_Size': None,
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
    IB_Range_Size = IB_High - IB_Low
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

            if (gap['gap_high (entry)'] - gap['gap_low']) < minimum_fvg_size:
                continue

            limit_price = gap['gap_high (entry)']
            stop_loss = gap['proposed_sl']
            # target_price = gap['proposed_tp']

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
                    valid_trades = valid_trades[valid_trades['time'] < "11:30:00"]
                else:
                    valid_trades = valid_trades[valid_trades['time'] < "12:30:00"] # 12:30:00 is arbitrary (hyper-parameter)
            else:
                valid_trades = valid_trades[valid_trades['time'] < next_gap_time]

            
            # Debug
            # print("Breakout time and current momentum_gap_creation_time:", breakout_time, gap['momentum_gap_creation_time'], " and next gap time:", next_gap_time)

            for trade in valid_trades.itertuples():
                # 3 scenarios: price goes up without stopping, price hits our limit, time runs out without no other gap getting created
                # print(trade.price, trade.time, trade.price <= limit_price)
                if trade.price >= limit_price + 2 * abs(limit_price - stop_loss): # if price runs too far (change for 'down')
                    # Remove limit order and stop loss
                    limit_price = None
                    stop_loss = None
                    # target_price = None
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
                # print("NOOOOOOOOOOWWWW")
                contracts = contract_size(abs(entry_price - stop_loss), 1000)
                trades_after_entry = daily_data[daily_data['time'] >= entry_time]

                running_sum = 0
                recent_deltas = [0] * window
                
                for trade in trades_after_entry.itertuples():

                    # print(trade.price, trade.time)

                    if trade.side == 'A':  # Sell aggressor
                        delta = trade.size  
                    elif trade.side == 'B':  # Buy aggressor
                        delta = -trade.size  

                    # Update the mean delta efficiently
                    running_sum, recent_deltas, recent_mean_delta = update_mean_delta(running_sum, recent_deltas, delta, window)

                    # print(recent_mean_delta, trade.time, trade.price)

                    # Adjust stop loss if the delta is more negative than the threshold (selling pressure)
                    if recent_mean_delta < -threshold:
                        entry_time_2 = pd.to_datetime(entry_time)
                        exit_time_2 = pd.to_datetime(trade.time)
                        trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60
                        R = (trade.price - entry_price) / abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 0 else 0
                        final_result = {
                            'Direction': gap['gap_direction'],
                            'PnL': (trade.price - entry_price) * 50 * contracts,
                            'R': R,
                            'Win': False if trade.price <= entry_price else True,
                            'Contracts': contracts,
                            'Entry_Time': entry_time,
                            'Entry_Price': entry_price,
                            'Exit_Time': trade.time,
                            'Exit_Price': trade.price,
                            'Date': date,
                            'Trade_Duration (mins)': trade_duration,
                            'IB_Range_Size': IB_Range_Size,
                            'Stop_Loss_Placement': stop_loss,
                            'Gap_Size': gap['gap_size']
                        }
                        return final_result
                    
                    elif trade.price <= stop_loss:  # Stop loss hit
                        entry_time_2 = pd.to_datetime(entry_time)
                        exit_time_2 = pd.to_datetime(trade.time)
                        trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60
                        R = (trade.price - entry_price) / abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 0 else 0
                        final_result = {
                            'Direction': gap['gap_direction'],
                            'PnL': (stop_loss - entry_price) * 50 * contracts,
                            'R': R,
                            'Win': False if trade.price <= entry_price else True,
                            'Contracts': contracts,
                            'Entry_Time': entry_time,
                            'Entry_Price': entry_price,
                            'Exit_Time': trade.time,
                            'Exit_Price': stop_loss,
                            'Date': date,
                            'Trade_Duration (mins)': trade_duration,
                            'IB_Range_Size': IB_Range_Size,
                            'Stop_Loss_Placement': stop_loss,
                            'Gap_Size': gap['gap_size']
                        }
                        return final_result
                    elif trade.time >= "16:59:00":
                        entry_time_2 = pd.to_datetime(entry_time)
                        exit_time_2 = pd.to_datetime(trade.time)
                        trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60
                        R = (trade.price - entry_price) / abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 0 else 0
                        final_result = {
                        'Direction': gap['gap_direction'],
                        'PnL': (trade.price - entry_price) * contracts * 50,
                        'R': R,
                        'Win': True if (trade.price - entry_price) / abs(entry_price - stop_loss) > 0 else False,
                        'Contracts': contracts,
                        'Entry_Time': entry_time,
                        'Entry_Price': entry_price,
                        'Exit_Time': "16:59:00",
                        'Exit_Price': trade.price,
                        'Date': date,
                        'Trade_Duration (mins)': trade_duration,
                        'IB_Range_Size': IB_Range_Size,
                        'Stop_Loss_Placement': stop_loss,
                        'Gap_Size': gap['gap_size']
                        }
                        return final_result
    elif breakout_direction == 'down':
        for i, gap in enumerate(momentum_gaps):

            if (gap['gap_high'] - gap['gap_low (entry)']) < minimum_fvg_size:
                continue

            limit_price = gap['gap_low (entry)']
            stop_loss = gap['proposed_sl']
            # target_price = gap['proposed_tp']

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
                    valid_trades = valid_trades[valid_trades['time'] < "13:30:00"] # 12:30:00 is arbitrary (hyper-parameter)
            else:
                valid_trades = valid_trades[valid_trades['time'] < next_gap_time]

            
            # Debug
            # print("Breakout time and current momentum_gap_creation_time:", breakout_time, gap['momentum_gap_creation_time'], " and next gap time:", next_gap_time)
            for trade in valid_trades.itertuples():
                # 3 scenarios: price goes up without stopping, price hits our limit, time runs out without no other gap getting created

                if trade.price <= limit_price - 2 * abs(limit_price - stop_loss): # if price runs too far (change for 'down')
                    # Remove limit order and stop loss
                    limit_price = None
                    stop_loss = None
                    # target_price = None
                    # Debug
                    # print("Trade was invalidated because price ran too far without offering entry.")
                    break
                elif trade.price >= limit_price and trade.time < breakout_time:
                    # print("Entry was given before the breakout time!")
                    # print("Invalid trade time: ", trade.time, " and Breakout time:", breakout_time)
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

                running_sum = 0
                recent_deltas = [0] * window

                for trade in trades_after_entry.itertuples():

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
                        R = (entry_price - trade.price) / abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 0 else 0
                        final_result = {
                            'Direction': gap['gap_direction'],
                            'PnL': (entry_price - trade.price) * contracts * 50,
                            'R': R,
                            'Win': False if trade.price >= entry_price else True,
                            'Contracts': contracts,
                            'Entry_Time': entry_time,
                            'Entry_Price': entry_price,
                            'Exit_Time': trade.time,
                            'Exit_Price': trade.price,
                            'Date': date,
                            'Trade_Duration (mins)': trade_duration,
                            'IB_Range_Size': IB_Range_Size,
                            'Stop_Loss_Placement': stop_loss,
                            'Gap_Size': gap['gap_size']
                        }
                        return final_result
                    
                    elif trade.price >= stop_loss:  # Stop loss hit
                        entry_time_2 = pd.to_datetime(entry_time)
                        exit_time_2 = pd.to_datetime(trade.time)
                        trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60
                        R = (entry_price - trade.price) / abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 0 else 0
                        final_result = {
                            'Direction': gap['gap_direction'],
                            'PnL': (entry_price - stop_loss) * contracts * 50,
                            'R': R,
                            'Win': False if trade.price >= entry_price else True,
                            'Contracts': contracts,
                            'Entry_Time': entry_time,
                            'Entry_Price': entry_price,
                            'Exit_Time': trade.time,
                            'Exit_Price': stop_loss,
                            'Date': date,
                            'Trade_Duration (mins)': trade_duration,
                            'IB_Range_Size': IB_Range_Size,
                            'Stop_Loss_Placement': stop_loss,
                            'Gap_Size': gap['gap_size']
                        }
                        return final_result
                    
                    elif trade.time >= "16:59:00":
                        entry_time_2 = pd.to_datetime(entry_time)
                        exit_time_2 = pd.to_datetime(trade.time)
                        trade_duration = (exit_time_2 - entry_time_2).total_seconds() / 60
                        R = (entry_price - trade.price) / abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 0 else 0
                        final_result = {
                        'Direction': gap['gap_direction'],
                        'PnL': (entry_price - trade.price) * contracts * 50,
                        'R': R,
                        'Win': True if (entry_price - trade.price) * contracts * 50 > 0 else False,
                        'Contracts': contracts,
                        'Entry_Time': entry_time,
                        'Entry_Price': entry_price,
                        'Exit_Time': "16:59:00",
                        'Exit_Price': trade.price,
                        'Date': date,
                        'Trade_Duration (mins)': trade_duration,
                        'IB_Range_Size': IB_Range_Size,
                        'Stop_Loss_Placement': stop_loss,
                        'Gap_Size': gap['gap_size']
                        }
                        return final_result
    return final_result





